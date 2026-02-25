#!/usr/bin/env python3
"""
Utilities for Imagenette/ImageNet sparse + BN recalibration + PTQ pipelines.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn

from train.ptq_convert import Int8QuantizedConv2d, Int8QuantizedLinear


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


def apply_sparse_mask(
    model: nn.Module,
    compute_conv_mask,
    compute_linear_mask,
    conv_filter,
    linear_filter
) -> Dict[str, torch.Tensor]:
    """Apply 2:4 masks in-place and return a name->mask map."""
    mask_map: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if not conv_filter(module, name):
                continue
            mask = compute_conv_mask(module.weight)
            if mask is None:
                continue
            module.weight.data.mul_(mask)
            mask_map[name] = mask
        elif isinstance(module, nn.Linear):
            if not linear_filter(module, name):
                continue
            mask = compute_linear_mask(module.weight)
            if mask is None:
                continue
            module.weight.data.mul_(mask)
            mask_map[name] = mask
    return mask_map


def replace_with_int8_from_mask(
    module: nn.Module,
    mask_map: Dict[str, torch.Tensor],
    prefix: str = ""
):
    """Replace Conv2d/Linear with Int8 modules, reusing precomputed masks."""
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
            replace_with_int8_from_mask(child, mask_map, prefix=full_name)


def calibrate_int8(model: nn.Module) -> None:
    for mod in model.modules():
        if isinstance(mod, (Int8QuantizedConv2d, Int8QuantizedLinear)):
            mod.calibrate_quantization()


def run_bn_recalibration(
    model: nn.Module,
    train_loader,
    device: str,
    max_steps: int = 300,
    lr: float = 1e-3
) -> int:
    for param in model.parameters():
        param.requires_grad = False
    bn_params = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = True
                bn_params.append(param)
    if not bn_params:
        return 0

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
    return steps
