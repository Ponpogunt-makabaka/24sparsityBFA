#!/usr/bin/env python3
"""
Ladder Test for ResNet-18 on Imagenette (mapped to ImageNet-1k indices).

Check 1: Dense FP32 baseline
Check 2: Sparse FP32 (2:4 magnitude mask, skip conv1/fc)
Check 3: Sparse INT8 (PTQ on masked model)
"""
import argparse
from typing import Dict, Optional

import torch
import torch.nn as nn

from torchvision.models import resnet18

from train.imagenet_utils import get_imagenet_loader
from train.ptq_convert import Int8QuantizedConv2d, Int8QuantizedLinear
from models.factory import _compute_2_4_mask_conv, _compute_2_4_mask_linear, _load_local_weights


def evaluate(model: nn.Module, loader, device: str, max_samples: Optional[int] = None) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if max_samples is not None and total >= max_samples:
                break
    return 100.0 * correct / max(1, total)


def apply_sparse_mask(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Apply 2:4 magnitude masks to conv/linear, skipping conv1 and fc."""
    mask_map: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if name == "conv1" or ".downsample." in name:
                continue
            if module.in_channels % 4 != 0:
                continue
            mask = _compute_2_4_mask_conv(module.weight)
            if mask is None:
                continue
            module.weight.data.mul_(mask)
            mask_map[name] = mask
        elif isinstance(module, nn.Linear):
            if name == "fc":
                continue
            mask = _compute_2_4_mask_linear(module.weight)
            module.weight.data.mul_(mask)
            mask_map[name] = mask
    return mask_map


def _replace_with_int8_from_mask(module: nn.Module, mask_map: Dict[str, torch.Tensor], prefix: str = ""):
    """Replace Conv2d/Linear with Int8 modules, reusing masks if present."""
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d):
            mask = mask_map.get(full_name)
            new_conv = Int8QuantizedConv2d(
                child.in_channels,
                child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                sparse_mask=mask
            )
            new_conv.weight.data.copy_(child.weight.data)
            if child.bias is not None and new_conv.bias is not None:
                new_conv.bias.data.copy_(child.bias.data)
            setattr(module, name, new_conv)
        elif isinstance(child, nn.Linear):
            mask = mask_map.get(full_name)
            new_fc = Int8QuantizedLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                sparse_mask=mask
            )
            new_fc.weight.data.copy_(child.weight.data)
            if child.bias is not None and new_fc.bias is not None:
                new_fc.bias.data.copy_(child.bias.data)
            setattr(module, name, new_fc)
        else:
            _replace_with_int8_from_mask(child, mask_map, prefix=full_name)


def calibrate_int8(model: nn.Module) -> None:
    for mod in model.modules():
        if isinstance(mod, (Int8QuantizedConv2d, Int8QuantizedLinear)):
            mod.calibrate_quantization()


def _set_bn_trainable(model: nn.Module) -> list:
    for param in model.parameters():
        param.requires_grad = False
    bn_params = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = True
                bn_params.append(param)
    return bn_params


def run_bn_recalibration(model: nn.Module, train_loader, device: str,
                         max_steps: int = 500, lr: float = 1e-3) -> None:
    bn_params = _set_bn_trainable(model)
    if not bn_params:
        print("[BN-Calib] No BatchNorm parameters found. Skipping.")
        return

    optimizer = torch.optim.SGD(bn_params, lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()

    steps = 0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        steps += 1
        if steps >= max_steps:
            break
    print(f"[BN-Calib] Completed {steps} steps of BN recalibration.")


def main():
    parser = argparse.ArgumentParser(description="ResNet-18 Ladder Test (Imagenette mapped)")
    parser.add_argument("--data-root", type=str, required=True, help="ImageNet root (optional)")
    parser.add_argument("--imagenette-root", type=str, required=True, help="Local Imagenette root")
    parser.add_argument("--weights-path", type=str, required=True, help="Local ResNet-18 weights")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--stop-after-check2", action="store_true",
                        help="Stop after Check 2 (skip INT8 check)")
    parser.add_argument("--bn-calib-steps", type=int, default=500)
    parser.add_argument("--bn-lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    batch_size = args.batch_size
    num_workers = 0 if args.num_workers > 0 else args.num_workers
    while True:
        try:
            val_loader = get_imagenet_loader(
                root=args.data_root,
                split="val",
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                imagenette_root=args.imagenette_root
            )
            break
        except PermissionError as exc:
            if num_workers > 0:
                print(f"[Ladder] DataLoader worker error ({exc}). Falling back to num_workers=0.")
                num_workers = 0
                continue
            raise

    # Train loader for BN recalibration (Imagenette train split)
    while True:
        try:
            train_loader = get_imagenet_loader(
                root=args.data_root,
                split="train",
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                imagenette_root=args.imagenette_root
            )
            break
        except PermissionError as exc:
            if num_workers > 0:
                print(f"[Ladder] DataLoader worker error during train loader ({exc}). Falling back to num_workers=0.")
                num_workers = 0
                continue
            raise

    # Check 1: Dense FP32
    model = resnet18(weights=None).to(device)
    _load_local_weights(model, args.weights_path, strict=True)
    while True:
        try:
            acc_dense = evaluate(model, val_loader, device, max_samples=args.max_eval_samples)
            break
        except PermissionError as exc:
            if num_workers > 0:
                print(f"[Ladder] DataLoader worker error during eval ({exc}). Falling back to num_workers=0.")
                num_workers = 0
                val_loader = get_imagenet_loader(
                    root=args.data_root,
                    split="val",
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False,
                    imagenette_root=args.imagenette_root
                )
                continue
            raise
    print(f"[Check1] Dense FP32 Accuracy: {acc_dense:.2f}%")
    if acc_dense < 80.0:
        print("[Check1] Accuracy < 80%. Label mapping or data loading likely broken. Stopping.")
        return

    # Check 2: Sparse FP32 (skip conv1/fc)
    mask_map = apply_sparse_mask(model)
    acc_sparse_fp32 = evaluate(model, val_loader, device, max_samples=args.max_eval_samples)
    print(f"[Check2] Sparse FP32 Accuracy (skip conv1/fc): {acc_sparse_fp32:.2f}%")
    try:
        run_bn_recalibration(model, train_loader, device, max_steps=args.bn_calib_steps, lr=args.bn_lr)
    except PermissionError as exc:
        if num_workers > 0:
            print(f"[Ladder] DataLoader worker error during BN calib ({exc}). Falling back to num_workers=0.")
            num_workers = 0
            train_loader = get_imagenet_loader(
                root=args.data_root,
                split="train",
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                imagenette_root=args.imagenette_root
            )
            run_bn_recalibration(model, train_loader, device, max_steps=args.bn_calib_steps, lr=args.bn_lr)
        else:
            raise
    model.eval()
    acc_sparse_fp32_recal = evaluate(model, val_loader, device, max_samples=args.max_eval_samples)
    print(f"[Check2-PostBN] Sparse FP32 Accuracy: {acc_sparse_fp32_recal:.2f}%")
    if args.stop_after_check2:
        return

    # Check 3: Sparse INT8 (PTQ)
    _replace_with_int8_from_mask(model, mask_map)
    calibrate_int8(model)
    acc_sparse_int8 = evaluate(model, val_loader, device, max_samples=args.max_eval_samples)
    print(f"[Check3] Sparse INT8 Accuracy: {acc_sparse_int8:.2f}%")


if __name__ == "__main__":
    main()
