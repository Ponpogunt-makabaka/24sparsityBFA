#!/usr/bin/env python3
"""
Step 3 (T09): strict 2:4 sparsify + fine-tune for ResNet-18 / MobileNet-V2 / DeiT-Tiny.

Strict grouping rules (T08-compatible):
- Conv2d (4D):  w.permute(0, 2, 3, 1).contiguous().view(-1, 4)
- Linear-like (2D): w.view(-1, 4)

After every optimizer.step(), re-apply mask to keep masked weights exactly zero.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, resnet18
from torchvision.models.vision_transformer import VisionTransformer


IMAGENETTE_WNID_TO_IMAGENET_INDEX = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}


@dataclass
class EvalResult:
    loss: float
    top1: float


class InputResizeWrapper(nn.Module):
    def __init__(self, model: nn.Module, target_size: int):
        super().__init__()
        self.model = model
        self.target_size = int(target_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.target_size or x.shape[-2] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        return self.model(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_imagenette_target_transform(classes) -> callable:
    if not classes:
        raise ValueError("Empty class list from dataset.")
    unknown = [c for c in classes if c not in IMAGENETTE_WNID_TO_IMAGENET_INDEX]
    if unknown:
        raise ValueError(f"Unknown class wnids (not official Imagenette): {unknown}")
    mapping = [IMAGENETTE_WNID_TO_IMAGENET_INDEX[c] for c in classes]

    def _map_target(y: int) -> int:
        return mapping[y]

    return _map_target


def create_imagenette_loaders(
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(f"Imagenette root missing train/val: {dataset_root}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_set = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_set = datasets.ImageFolder(str(val_dir), transform=val_transform)

    train_set.target_transform = build_imagenette_target_transform(train_set.classes)
    val_set.target_transform = build_imagenette_target_transform(val_set.classes)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def create_cifar100_loaders(
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    dataset_root.mkdir(parents=True, exist_ok=True)
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR100(
        root=str(dataset_root),
        train=True,
        download=download,
        transform=train_transform,
    )
    val_set = datasets.CIFAR100(
        root=str(dataset_root),
        train=False,
        download=download,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    if all(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _looks_like_timm_deit_state_dict(state_dict: dict) -> bool:
    if not isinstance(state_dict, dict) or not state_dict:
        return False
    keys = list(state_dict.keys())
    return (
        any(k == "cls_token" for k in keys)
        and any(k == "pos_embed" for k in keys)
        and any(k.startswith("patch_embed.proj.") for k in keys)
        and any(k.startswith("blocks.0.") for k in keys)
    )


def _remap_timm_deit_to_torchvision_vit(state_dict: dict) -> dict:
    remapped = {}
    for k, v in state_dict.items():
        if k == "cls_token":
            remapped["class_token"] = v
            continue
        if k == "pos_embed":
            remapped["encoder.pos_embedding"] = v
            continue
        if k.startswith("patch_embed.proj."):
            remapped["conv_proj." + k[len("patch_embed.proj."):]] = v
            continue
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) < 3:
                continue
            blk = parts[1]
            rest = ".".join(parts[2:])
            prefix = f"encoder.layers.encoder_layer_{blk}."
            if rest.startswith("norm1."):
                remapped[prefix + "ln_1." + rest[len("norm1."):]] = v
                continue
            if rest.startswith("norm2."):
                remapped[prefix + "ln_2." + rest[len("norm2."):]] = v
                continue
            if rest.startswith("attn.qkv.weight"):
                remapped[prefix + "self_attention.in_proj_weight"] = v
                continue
            if rest.startswith("attn.qkv.bias"):
                remapped[prefix + "self_attention.in_proj_bias"] = v
                continue
            if rest.startswith("attn.proj.weight"):
                remapped[prefix + "self_attention.out_proj.weight"] = v
                continue
            if rest.startswith("attn.proj.bias"):
                remapped[prefix + "self_attention.out_proj.bias"] = v
                continue
            if rest.startswith("mlp.fc1."):
                remapped[prefix + "mlp.0." + rest[len("mlp.fc1."):]] = v
                continue
            if rest.startswith("mlp.fc2."):
                remapped[prefix + "mlp.3." + rest[len("mlp.fc2."):]] = v
                continue
            continue
        if k.startswith("norm."):
            remapped["encoder.ln." + k[len("norm."):]] = v
            continue
        if k.startswith("head."):
            remapped["heads.head." + k[len("head."):]] = v
            continue
    return remapped


def _resize_deit_pos_embedding(pos_embed: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if pos_embed.ndim != 3 or pos_embed.shape[0] != 1:
        raise ValueError(f"Unexpected pos embedding shape: {tuple(pos_embed.shape)}")
    if pos_embed.shape[1] == target_tokens:
        return pos_embed

    cls_tok = pos_embed[:, :1, :]
    grid_tok = pos_embed[:, 1:, :]
    src_n = grid_tok.shape[1]
    dst_n = target_tokens - 1
    src_hw = int(math.sqrt(src_n))
    dst_hw = int(math.sqrt(dst_n))
    if src_hw * src_hw != src_n or dst_hw * dst_hw != dst_n:
        raise ValueError(f"Non-square pos embed resize: src_n={src_n} dst_n={dst_n}")

    grid_tok = grid_tok.reshape(1, src_hw, src_hw, -1).permute(0, 3, 1, 2).contiguous()
    grid_tok = F.interpolate(grid_tok, size=(dst_hw, dst_hw), mode="bicubic", align_corners=False)
    grid_tok = grid_tok.permute(0, 2, 3, 1).contiguous().reshape(1, dst_n, -1)
    return torch.cat([cls_tok, grid_tok], dim=1)


def _load_compatible_state_dict(model: nn.Module, state_dict: dict) -> Tuple[int, int, int]:
    model_sd = model.state_dict()
    filtered = {}
    skipped = 0
    for k, v in state_dict.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped += 1

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return skipped, len(missing), len(unexpected)


def create_model_and_load(
    arch: str,
    dense_weights: Path,
    device: str,
    dataset: str,
    num_classes: int,
    image_size: int,
) -> nn.Module:
    if not dense_weights.exists():
        raise FileNotFoundError(f"Dense weights not found: {dense_weights}")

    if arch == "resnet18":
        model = resnet18(weights=None, num_classes=num_classes)
        if dataset == "cifar100":
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        checkpoint = torch.load(dense_weights, map_location="cpu")
        state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
        skipped, missing, unexpected = _load_compatible_state_dict(model, state_dict)
        print(f"[Load] resnet18 skipped={skipped} missing={missing} unexpected={unexpected}")
        model.to(device)
        return model

    if arch == "mobilenet_v2":
        model = mobilenet_v2(weights=None, num_classes=num_classes)
        if dataset == "cifar100":
            # Match CIFAR-100 MobileNetV2 stride pattern used by public checkpoints.
            model.features[0][0].stride = (1, 1)
            model.features[2].conv[1][0].stride = (1, 1)
        checkpoint = torch.load(dense_weights, map_location="cpu")
        state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
        skipped, missing, unexpected = _load_compatible_state_dict(model, state_dict)
        print(f"[Load] mobilenet_v2 skipped={skipped} missing={missing} unexpected={unexpected}")
        model.to(device)
        return model

    if arch == "deit_tiny":
        patch_size = 4 if dataset == "cifar100" else 16
        model = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=12,
            num_heads=3,
            hidden_dim=192,
            mlp_dim=768,
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=num_classes,
        )
        checkpoint = torch.load(dense_weights, map_location="cpu")
        state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
        if _looks_like_timm_deit_state_dict(state_dict):
            state_dict = _remap_timm_deit_to_torchvision_vit(state_dict)
        if "encoder.pos_embedding" in state_dict:
            target_tokens = model.state_dict()["encoder.pos_embedding"].shape[1]
            state_dict["encoder.pos_embedding"] = _resize_deit_pos_embedding(
                state_dict["encoder.pos_embedding"],
                target_tokens=target_tokens,
            )
        skipped, missing, unexpected = _load_compatible_state_dict(model, state_dict)
        print(f"[Load] deit_tiny skipped={skipped} missing={missing} unexpected={unexpected}")
        model.to(device)
        return model

    raise ValueError(f"Unsupported arch: {arch}")


def compute_2_4_mask_conv(weight: torch.Tensor) -> torch.Tensor:
    """
    Conv grouping must match T08 flatten_groups semantics exactly.
    """
    w_perm = weight.detach().permute(0, 2, 3, 1).contiguous()
    if w_perm.numel() % 4 != 0:
        raise ValueError(f"Conv weight numel not divisible by 4: {tuple(weight.shape)}")
    flat_abs = w_perm.abs().view(-1, 4)
    prune_idx = torch.argsort(flat_abs, dim=1)[:, :2]
    keep_flat = torch.ones_like(flat_abs, dtype=torch.bool)
    keep_flat.scatter_(dim=1, index=prune_idx, value=False)
    keep_perm = keep_flat.view_as(w_perm)
    keep_orig = keep_perm.permute(0, 3, 1, 2).contiguous()
    return keep_orig


def compute_2_4_mask_linear(weight: torch.Tensor) -> torch.Tensor:
    if weight.numel() % 4 != 0:
        raise ValueError(f"Linear-like weight numel not divisible by 4: {tuple(weight.shape)}")
    flat_abs = weight.detach().abs().view(-1, 4)
    prune_idx = torch.argsort(flat_abs, dim=1)[:, :2]
    keep_flat = torch.ones_like(flat_abs, dtype=torch.bool)
    keep_flat.scatter_(dim=1, index=prune_idx, value=False)
    return keep_flat.view_as(weight)


def build_fixed_masks(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Build fixed masks for all 4D (Conv-like) and 2D (Linear/QKV-like) parameters.
    """
    mask_map: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.dim() == 4:
            mask_map[name] = compute_2_4_mask_conv(param.data)
        elif param.dim() == 2:
            mask_map[name] = compute_2_4_mask_linear(param.data)
    return mask_map


def apply_masks(model: nn.Module, mask_map: Dict[str, torch.Tensor]) -> None:
    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, mask in mask_map.items():
            if name not in named_params:
                continue
            p = named_params[name]
            p.data.mul_(mask.to(device=p.device, dtype=p.dtype))


def apply_mask_to_grads(model: nn.Module, mask_map: Dict[str, torch.Tensor]) -> None:
    named_params = dict(model.named_parameters())
    for name, mask in mask_map.items():
        if name not in named_params:
            continue
        p = named_params[name]
        if p.grad is None:
            continue
        p.grad.mul_(mask.to(device=p.grad.device, dtype=p.grad.dtype))


def verify_masks(mask_map: Dict[str, torch.Tensor]) -> None:
    for name, mask in mask_map.items():
        if mask.dim() == 4:
            flat = mask.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        elif mask.dim() == 2:
            flat = mask.contiguous().view(-1, 4)
        else:
            raise ValueError(f"Unexpected mask dim for {name}: {mask.dim()}")
        ones_per_group = flat.sum(dim=1)
        if not torch.all(ones_per_group == 2):
            bad = int((ones_per_group != 2).sum().item())
            raise RuntimeError(f"Mask pattern violation in {name}: bad_groups={bad}")


def verify_mask_enforcement(model: nn.Module, mask_map: Dict[str, torch.Tensor]) -> float:
    max_abs_masked = 0.0
    named_params = dict(model.named_parameters())
    for name, mask in mask_map.items():
        if name not in named_params:
            continue
        p = named_params[name]
        mask_dev = mask.to(device=p.device)
        masked_vals = p.data[~mask_dev]
        if masked_vals.numel() == 0:
            continue
        layer_max = float(masked_vals.abs().max().item())
        max_abs_masked = max(max_abs_masked, layer_max)
    return max_abs_masked


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> EvalResult:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += float(loss.item()) * int(images.size(0))
            pred = outputs.argmax(dim=1)
            correct += int((pred == targets).sum().item())
            total += int(images.size(0))
    return EvalResult(
        loss=total_loss / max(total, 1),
        top1=100.0 * correct / max(total, 1),
    )


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mask_map: Dict[str, torch.Tensor],
    criterion: nn.Module,
    device: str,
    grad_clip: float = 0.0,
) -> EvalResult:
    model.train()
    total = 0
    correct = 0
    total_loss = 0.0

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        # Prevent masked coordinates from receiving optimizer momentum/state updates.
        apply_mask_to_grads(model, mask_map)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Critical requirement: enforce fixed sparsity immediately after update.
        apply_masks(model, mask_map)

        total_loss += float(loss.item()) * int(images.size(0))
        pred = outputs.argmax(dim=1)
        correct += int((pred == targets).sum().item())
        total += int(images.size(0))

    return EvalResult(
        loss=total_loss / max(total, 1),
        top1=100.0 * correct / max(total, 1),
    )


def default_dense_weights_for_arch(arch: str, dataset: str) -> Path:
    base = Path("data/T09_ImageNet_Scale/weights")
    if dataset == "cifar100":
        if arch == "resnet18":
            return base / "resnet18_cifar100_hf.bin"
        if arch == "mobilenet_v2":
            return base / "mobilenet_v2_cifar100_chenyaofo.pth"
        if arch == "deit_tiny":
            return base / "deit_tiny_patch16_224-a1311bcf.pth"
    if arch == "resnet18":
        return base / "resnet18-f37072fd.pth"
    if arch == "mobilenet_v2":
        return base / "mobilenet_v2-b0353104.pth"
    if arch == "deit_tiny":
        return base / "deit_tiny_patch16_224-a1311bcf.pth"
    raise ValueError(f"Unsupported arch: {arch}")


def default_output_for_arch(arch: str, dataset: str) -> Path:
    base = Path("data/T09_ImageNet_Scale/weights")
    suffix = "imagenette" if dataset == "imagenette" else "cifar100"
    if arch == "resnet18":
        return base / f"resnet18_2_4_sparse_{suffix}.pth"
    if arch == "mobilenet_v2":
        return base / f"mobilenet_v2_2_4_sparse_{suffix}.pth"
    if arch == "deit_tiny":
        return base / f"deit_tiny_2_4_sparse_{suffix}.pth"
    raise ValueError(f"Unsupported arch: {arch}")


def default_batch_size_for_arch(arch: str, dataset: str) -> int:
    if dataset == "imagenette" and arch == "deit_tiny":
        return 16
    if dataset == "cifar100" and arch == "deit_tiny":
        return 128
    if dataset == "cifar100":
        return 256
    return 64


def dataset_num_classes(dataset: str) -> int:
    if dataset == "imagenette":
        return 1000
    if dataset == "cifar100":
        return 100
    raise ValueError(f"Unsupported dataset: {dataset}")


def dataset_image_size(dataset: str) -> int:
    if dataset == "imagenette":
        return 224
    if dataset == "cifar100":
        return 32
    raise ValueError(f"Unsupported dataset: {dataset}")


def default_epochs_for_run(dataset: str, arch: str) -> int:
    if dataset == "cifar100":
        if arch == "deit_tiny":
            # ViT fine-tuning on CIFAR-100 usually needs substantially longer schedules.
            return 100
        return 40
    return 3


def default_eval_every(dataset: str) -> int:
    if dataset == "cifar100":
        return 5
    return 1


def default_lr_for_run(dataset: str, arch: str) -> float:
    if dataset == "cifar100":
        if arch == "deit_tiny":
            return 3e-4
        return 5e-2
    return 1e-3


def default_weight_decay_for_run(dataset: str, arch: str) -> float:
    if dataset == "cifar100" and arch == "deit_tiny":
        return 5e-2
    return 1e-4


def default_warmup_epochs_for_run(dataset: str, arch: str, epochs: int) -> int:
    if epochs <= 1:
        return 0
    if arch == "deit_tiny":
        if dataset == "cifar100":
            return min(5, max(1, epochs // 10))
        return min(3, max(1, epochs // 10))
    return 0


def build_optimizer(
    model: nn.Module,
    dataset: str,
    arch: str,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if dataset == "cifar100" and arch == "deit_tiny":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    if warmup_epochs > 0 and epochs > 1:
        warmup_steps = min(warmup_epochs, max(epochs - 1, 1))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_steps = max(epochs - warmup_steps, 1)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Step3: strict 2:4 sparsify + fine-tune")
    parser.add_argument("--arch", type=str, choices=["resnet18", "mobilenet_v2", "deit_tiny"], required=True)
    parser.add_argument("--dataset", type=str, choices=["imagenette", "cifar100"], default="imagenette")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Dataset root. If omitted, uses dataset-specific default.",
    )
    parser.add_argument("--dense-weights", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=0, help="0 means dataset default")
    parser.add_argument("--batch-size", type=int, default=0, help="0 means auto by arch")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0, help="0 means dataset/arch default")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=-1.0, help="-1 means dataset/arch default")
    parser.add_argument("--eval-every", type=int, default=0, help="0 means dataset default")
    parser.add_argument("--warmup-epochs", type=int, default=-1, help="-1 means dataset/arch default")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = dataset_num_classes(args.dataset)
    image_size = dataset_image_size(args.dataset)
    epochs = args.epochs if args.epochs > 0 else default_epochs_for_run(args.dataset, args.arch)
    eval_every = args.eval_every if args.eval_every > 0 else default_eval_every(args.dataset)
    lr = args.lr if args.lr > 0 else default_lr_for_run(args.dataset, args.arch)
    weight_decay = args.weight_decay if args.weight_decay >= 0 else default_weight_decay_for_run(args.dataset, args.arch)
    warmup_epochs = (
        args.warmup_epochs
        if args.warmup_epochs >= 0
        else default_warmup_epochs_for_run(args.dataset, args.arch, epochs)
    )
    grad_clip = 1.0 if args.dataset == "cifar100" and args.arch == "deit_tiny" else 0.0
    if args.dataset == "cifar100" and args.arch == "deit_tiny" and epochs < 20:
        print(
            f"[Warn] deit_tiny on CIFAR-100 with epochs={epochs} is likely under-trained; "
            "recommend >=20 epochs with warmup."
        )

    dense_weights = Path(args.dense_weights) if args.dense_weights else default_dense_weights_for_arch(args.arch, args.dataset)
    output_path = Path(args.output) if args.output else default_output_for_arch(args.arch, args.dataset)
    batch_size = args.batch_size if args.batch_size > 0 else default_batch_size_for_arch(args.arch, args.dataset)
    if args.dataset_root is None:
        dataset_root = Path("data/imagenette_full" if args.dataset == "imagenette" else "data/cifar100")
    else:
        dataset_root = Path(args.dataset_root)

    print(f"[Env] device={device}")
    print(
        "[Args] "
        f"arch={args.arch} dataset={args.dataset} dataset_root={dataset_root} dense_weights={dense_weights} "
        f"output={output_path} epochs={epochs} batch_size={batch_size} "
        f"num_workers={args.num_workers} lr={lr} weight_decay={weight_decay} warmup_epochs={warmup_epochs} "
        f"eval_every={eval_every} "
        f"num_classes={num_classes} image_size={image_size}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.dataset == "imagenette":
        train_loader, val_loader = create_imagenette_loaders(
            dataset_root=dataset_root,
            batch_size=batch_size,
            num_workers=args.num_workers,
        )
    else:
        train_loader, val_loader = create_cifar100_loaders(
            dataset_root=dataset_root,
            batch_size=batch_size,
            num_workers=args.num_workers,
            download=True,
        )
    print(f"[Data] train_batches={len(train_loader)} val_batches={len(val_loader)}")

    model = create_model_and_load(
        args.arch,
        dense_weights,
        device=device,
        dataset=args.dataset,
        num_classes=num_classes,
        image_size=image_size,
    )
    dense_val = evaluate(model, val_loader, device)
    print(f"[Dense] val_top1={dense_val.top1:.2f}% val_loss={dense_val.loss:.4f}")

    t0 = time.time()
    mask_map = build_fixed_masks(model)
    verify_masks(mask_map)
    apply_masks(model, mask_map)
    sparse_init_val = evaluate(model, val_loader, device)
    print(f"[SparseInit] val_top1={sparse_init_val.top1:.2f}% val_loss={sparse_init_val.loss:.4f}")
    print(f"[Mask] masked_param_tensors={len(mask_map)}")

    optimizer = build_optimizer(
        model=model,
        dataset=args.dataset,
        arch=args.arch,
        lr=lr,
        momentum=args.momentum,
        weight_decay=weight_decay,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
    )
    train_criterion = nn.CrossEntropyLoss(label_smoothing=0.1 if args.dataset == "cifar100" else 0.0)

    history = []
    best_top1 = -1.0
    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, epochs + 1):
        train_res = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            mask_map=mask_map,
            criterion=train_criterion,
            device=device,
            grad_clip=grad_clip,
        )
        do_eval = (epoch % eval_every == 0) or (epoch == epochs)
        val_res = evaluate(model, val_loader, device) if do_eval else None
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        masked_max = verify_mask_enforcement(model, mask_map)
        if val_res is not None and val_res.top1 >= best_top1:
            best_top1 = val_res.top1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if val_res is not None:
            print(
                f"[Epoch {epoch}/{epochs}] "
                f"train_top1={train_res.top1:.2f}% train_loss={train_res.loss:.4f} | "
                f"val_top1={val_res.top1:.2f}% val_loss={val_res.loss:.4f} | "
                f"best_val_top1={best_top1:.2f}% lr={lr_now:.6f} | "
                f"max_abs_masked={masked_max:.3e}"
            )
        else:
            print(
                f"[Epoch {epoch}/{epochs}] "
                f"train_top1={train_res.top1:.2f}% train_loss={train_res.loss:.4f} | "
                f"val_top1=SKIP best_val_top1={best_top1:.2f}% lr={lr_now:.6f} | "
                f"max_abs_masked={masked_max:.3e}"
            )
        history.append(
            {
                "epoch": epoch,
                "train_top1": train_res.top1,
                "train_loss": train_res.loss,
                "val_top1": (None if val_res is None else val_res.top1),
                "val_loss": (None if val_res is None else val_res.loss),
                "best_val_top1": best_top1,
                "lr": lr_now,
                "max_abs_masked": masked_max,
            }
        )

    model.load_state_dict(best_state_dict, strict=True)
    apply_masks(model, mask_map)
    final_val = evaluate(model, val_loader, device)
    final_masked_max = verify_mask_enforcement(model, mask_map)
    elapsed = time.time() - t0

    ckpt = {
        "arch": args.arch,
        "model_state_dict": model.state_dict(),
        "sparsity_type": "2:4",
        "mask_grouping": {
            "conv2d": "w.permute(0,2,3,1).contiguous().view(-1,4)",
            "linear": "w.view(-1,4)",
            "mha_qkv": "in_proj_weight.view(-1,4) via 2D parameter handling",
        },
        "dataset": args.dataset,
        "num_classes": num_classes,
        "image_size": image_size,
        "dense_val_top1": dense_val.top1,
        "sparse_init_val_top1": sparse_init_val.top1,
        "final_val_top1": final_val.top1,
        "final_val_loss": final_val.loss,
        "final_max_abs_masked": final_masked_max,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "best_val_top1": best_top1,
        "history": history,
    }
    torch.save(ckpt, output_path)

    metrics_path = output_path.with_suffix(".json")
    metrics_path.write_text(
        json.dumps(
            {
                "output_checkpoint": str(output_path),
                "metrics": {
                    "arch": args.arch,
                    "dense_val_top1": dense_val.top1,
                    "sparse_init_val_top1": sparse_init_val.top1,
                    "final_val_top1": final_val.top1,
                    "final_val_loss": final_val.loss,
                    "final_max_abs_masked": final_masked_max,
                    "elapsed_sec": elapsed,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "warmup_epochs": warmup_epochs,
                    "best_val_top1": best_top1,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[Done] Saved sparse checkpoint: {output_path}")
    print(f"[Done] Saved metrics json: {metrics_path}")
    print(
        f"[Final] arch={args.arch} val_top1={final_val.top1:.2f}% val_loss={final_val.loss:.4f} "
        f"max_abs_masked={final_masked_max:.3e} elapsed_sec={elapsed:.1f}"
    )


if __name__ == "__main__":
    main()
