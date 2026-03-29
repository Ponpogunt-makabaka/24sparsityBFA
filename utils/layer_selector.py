"""
Layer selection strategies for 2:4 sparsity.

Determines which layers are eligible for 2:4 structured sparsity
based on model architecture and hardware constraints.

Rules:
  - Linear: eligible if in_features % 4 == 0
  - Conv2d: eligible if standard conv (groups=1), K = in_ch * kH * kW,
            and K % 4 == 0
  - Skip: depthwise conv, conv with groups > 1, 1D params, bias
  - Configurable: classification head, patch embedding
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LayerSelectionConfig:
    """Configuration for layer selection."""
    sparsify_linear: bool = True
    sparsify_conv_1x1: bool = True
    sparsify_conv_3x3: bool = True
    sparsify_conv_other: bool = False
    sparsify_head: bool = False
    sparsify_first_conv: bool = False
    sparsify_patch_embed: bool = False
    sparsify_downsample: bool = True
    min_K_for_conv: int = 4  # minimum K dimension to be eligible


def _is_depthwise(mod: nn.Conv2d) -> bool:
    return mod.groups == mod.in_channels and mod.in_channels == mod.out_channels


def _is_pointwise(mod: nn.Conv2d) -> bool:
    return mod.kernel_size == (1, 1) and mod.groups == 1


def _get_K_dim(mod: nn.Conv2d) -> int:
    """Get K dimension for GEMM view of this conv layer."""
    in_per_group = mod.in_channels // mod.groups
    kH, kW = mod.kernel_size
    return in_per_group * kH * kW


def select_layers_resnet18(
    model: nn.Module,
    config: LayerSelectionConfig | None = None,
) -> Dict[str, str]:
    """
    Select eligible layers for ResNet-18.

    Default strategy:
      - Skip: conv1 (in_ch=3), classification head (fc)
      - Sparsify: all layer{1-4} conv layers (3x3 and 1x1) where K%4==0
      - Sparsify: downsample conv (1x1)
    """
    if config is None:
        config = LayerSelectionConfig(
            sparsify_head=False,
            sparsify_first_conv=False,
            sparsify_conv_3x3=True,
            sparsify_downsample=True,
        )

    eligible = {}
    for name, mod in model.named_modules():
        param_name = name + ".weight"

        if isinstance(mod, nn.Linear):
            if name == "fc" and not config.sparsify_head:
                continue
            if mod.in_features % 4 != 0:
                continue
            eligible[param_name] = "linear"
            continue

        if isinstance(mod, nn.Conv2d):
            if name == "conv1" and not config.sparsify_first_conv:
                continue
            if _is_depthwise(mod):
                continue
            if mod.groups > 1:
                continue

            K = _get_K_dim(mod)
            if K % 4 != 0:
                continue
            if K < config.min_K_for_conv:
                continue

            kH, kW = mod.kernel_size
            if kH == 1 and kW == 1:
                if ".downsample." in name and not config.sparsify_downsample:
                    continue
                if config.sparsify_conv_1x1:
                    eligible[param_name] = "conv1x1"
            elif kH == 3 and kW == 3:
                if config.sparsify_conv_3x3:
                    eligible[param_name] = "conv3x3"
            else:
                if config.sparsify_conv_other:
                    eligible[param_name] = f"conv{kH}x{kW}"

    return eligible


def select_layers_mobilenet_v2(
    model: nn.Module,
    config: LayerSelectionConfig | None = None,
) -> Dict[str, str]:
    """
    Select eligible layers for MobileNet-V2.

    Default strategy:
      - Skip: ALL depthwise conv (groups == in_channels)
      - Skip: first conv (features.0.0, in_ch=3)
      - Sparsify: pointwise 1x1 conv (groups=1)
      - Sparsify: classifier Linear
    """
    if config is None:
        config = LayerSelectionConfig(
            sparsify_head=True,  # classifier is important for MobileNet
            sparsify_first_conv=False,
            sparsify_conv_1x1=True,
            sparsify_conv_3x3=False,  # all 3x3 in MobileNet are depthwise
        )

    eligible = {}
    for name, mod in model.named_modules():
        param_name = name + ".weight"

        if isinstance(mod, nn.Linear):
            if mod.in_features % 4 != 0:
                continue
            eligible[param_name] = "linear"
            continue

        if isinstance(mod, nn.Conv2d):
            # Skip depthwise
            if _is_depthwise(mod):
                logger.debug(f"Skip depthwise: {name}")
                continue
            # Skip any grouped conv
            if mod.groups > 1:
                logger.debug(f"Skip grouped conv: {name} groups={mod.groups}")
                continue
            # Skip first conv (in_channels=3)
            if mod.in_channels == 3 and not config.sparsify_first_conv:
                continue

            K = _get_K_dim(mod)
            if K % 4 != 0:
                continue

            if _is_pointwise(mod):
                if config.sparsify_conv_1x1:
                    eligible[param_name] = "pointwise1x1"
            elif mod.kernel_size == (3, 3):
                # Standard 3x3 convs (rare in MobileNet, most are depthwise)
                if config.sparsify_conv_3x3:
                    eligible[param_name] = "conv3x3"

    return eligible


def select_layers_deit_tiny(
    model: nn.Module,
    config: LayerSelectionConfig | None = None,
) -> Dict[str, str]:
    """
    Select eligible layers for DeiT-Tiny (torchvision VisionTransformer).

    Default strategy:
      - Skip: conv_proj (patch embedding)
      - Skip: heads.head (classification head)
      - Sparsify: all encoder MLP Linear (mlp.0, mlp.3)
      - Sparsify: attention in_proj_weight (QKV) and out_proj
    """
    if config is None:
        config = LayerSelectionConfig(
            sparsify_head=False,
            sparsify_patch_embed=False,
            sparsify_linear=True,
        )

    eligible = {}
    for name, param in model.named_parameters():
        if param.dim() != 2:
            # Skip conv_proj (4D) unless explicitly enabled
            if param.dim() == 4 and "conv_proj" in name:
                if config.sparsify_patch_embed:
                    K = param.shape[1] * param.shape[2] * param.shape[3]
                    if K % 4 == 0:
                        eligible[name] = "patch_embed_conv"
                continue
            continue

        # 2D parameters
        if "heads.head" in name:
            if not config.sparsify_head:
                continue

        # LayerNorm weight is 1D, won't reach here
        out_f, in_f = param.shape
        if in_f % 4 != 0:
            continue

        if ".mlp.0." in name or ".mlp.3." in name:
            eligible[name] = "mlp_linear"
        elif ".self_attention.in_proj_weight" in name:
            eligible[name] = "qkv_proj"
        elif ".self_attention.out_proj.weight" in name:
            eligible[name] = "out_proj"
        elif "heads.head" in name and config.sparsify_head:
            eligible[name] = "head"
        elif config.sparsify_linear:
            eligible[name] = "other_linear"

    return eligible


def select_layers(
    model: nn.Module,
    arch: str,
    config: LayerSelectionConfig | None = None,
) -> Dict[str, str]:
    """Dispatch to arch-specific layer selection."""
    if arch == "resnet18":
        return select_layers_resnet18(model, config)
    elif arch == "mobilenet_v2":
        return select_layers_mobilenet_v2(model, config)
    elif arch == "deit_tiny":
        return select_layers_deit_tiny(model, config)
    else:
        raise ValueError(f"Unknown arch: {arch}")


def print_selection_summary(eligible: Dict[str, str], model: nn.Module) -> None:
    """Print layer selection summary."""
    named_params = dict(model.named_parameters())
    total_params = sum(p.numel() for p in model.parameters() if p.dim() >= 2)
    sparse_params = sum(
        named_params[n].numel() for n in eligible if n in named_params
    )

    print(f"\n{'='*80}")
    print(f"Layer Selection Summary")
    print(f"{'-'*80}")
    print(f"{'Layer':<55} {'Type':<15} {'Shape'}")
    print(f"{'-'*80}")
    for name, layer_type in sorted(eligible.items()):
        if name in named_params:
            shape = list(named_params[name].shape)
            print(f"{name:<55} {layer_type:<15} {shape}")
    print(f"{'='*80}")
    print(f"Eligible layers: {len(eligible)} / {sum(1 for p in model.parameters() if p.dim() >= 2)}")
    print(f"Sparse params: {sparse_params:,} / {total_params:,} ({100*sparse_params/max(total_params,1):.1f}%)")
