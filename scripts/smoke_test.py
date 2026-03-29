#!/usr/bin/env python3
"""
Smoke test: verify 2:4 sparsity semantics, compliance, and baseline accuracy.

Runs minimal checks that all Phase 2-4 fixes are intact:
  1. Mask computation correctness (K-dim grouping)
  2. Layer selection per architecture
  3. 2:4 compliance on saved checkpoints
  4. Baseline accuracy sanity (from JSON metadata)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, resnet18
from torchvision.models.vision_transformer import VisionTransformer

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.T09_ImageNet_Scale.step3_sparsify_finetune import (
    _should_sparsify,
    build_fixed_masks,
    compute_2_4_mask_conv,
    compute_2_4_mask_linear,
)

WEIGHTS_DIR = PROJECT_ROOT / "data" / "T09_ImageNet_Scale" / "weights"

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if not condition:
        FAIL += 1
    else:
        PASS += 1
    suffix = f" ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")


def test_mask_k_dim_grouping() -> None:
    """Test that mask computation respects K-dimension."""
    print("\n=== Test 1: K-dimension grouping ===")

    # Conv2d with K%4==0 should produce valid mask
    w = torch.randn(64, 64, 3, 3)  # K=64*3*3=576, 576%4==0
    mask = compute_2_4_mask_conv(w)
    check("Conv K%4==0 produces mask", mask is not None)
    if mask is not None:
        perm = mask.float().permute(0, 2, 3, 1).contiguous().view(-1, 4)
        ones_per_group = perm.sum(dim=1)
        check("Conv mask has exactly 2 ones per group",
              bool(torch.all(ones_per_group == 2)),
              f"min={ones_per_group.min()}, max={ones_per_group.max()}")

    # Conv2d with K%4!=0 should return None
    w_skip = torch.randn(64, 3, 7, 7)  # K=3*7*7=147, 147%4==3
    mask_skip = compute_2_4_mask_conv(w_skip)
    check("Conv K%4!=0 returns None (skip)", mask_skip is None, "K=147")

    # Linear with in_features%4==0
    w_lin = torch.randn(100, 192)  # in=192, 192%4==0
    mask_lin = compute_2_4_mask_linear(w_lin)
    check("Linear in%4==0 produces mask", mask_lin is not None)
    if mask_lin is not None:
        flat = mask_lin.float().view(-1, 4)
        ones = flat.sum(dim=1)
        check("Linear mask has exactly 2 ones per group",
              bool(torch.all(ones == 2)))

    # Linear with in_features%4!=0
    w_lin_skip = torch.randn(10, 3)  # in=3, 3%4!=0
    mask_lin_skip = compute_2_4_mask_linear(w_lin_skip)
    check("Linear in%4!=0 returns None (skip)", mask_lin_skip is None)


def test_layer_selection() -> None:
    """Test architecture-aware layer selection."""
    print("\n=== Test 2: Layer selection ===")

    # ResNet-18
    model_r18 = resnet18(weights=None, num_classes=1000)
    masks_r18 = build_fixed_masks(model_r18, arch="resnet18")
    check("ResNet-18: conv1.weight NOT in masks",
          "conv1.weight" not in masks_r18, "K=3*7*7=147, K%4!=0")
    check("ResNet-18: fc.weight IN masks",
          "fc.weight" in masks_r18, "in=512, 512%4==0")
    check("ResNet-18: layer1.0.conv1.weight IN masks",
          "layer1.0.conv1.weight" in masks_r18)
    expected_r18 = 20  # 19 conv + fc
    check(f"ResNet-18: {expected_r18} masked layers",
          len(masks_r18) == expected_r18, f"got {len(masks_r18)}")

    # MobileNetV2
    model_mv2 = mobilenet_v2(weights=None, num_classes=1000)
    masks_mv2 = build_fixed_masks(model_mv2, arch="mobilenet_v2")
    check("MobileNetV2: first conv NOT in masks",
          "features.0.0.weight" not in masks_mv2, "in_ch=3")
    # Check depthwise not in masks
    check("MobileNetV2: depthwise NOT in masks",
          "features.2.conv.1.0.weight" not in masks_mv2, "depthwise conv")
    check("MobileNetV2: classifier.1.weight IN masks",
          "classifier.1.weight" in masks_mv2, "Linear 1280→1000")
    expected_mv2 = 35
    check(f"MobileNetV2: {expected_mv2} masked layers",
          len(masks_mv2) == expected_mv2, f"got {len(masks_mv2)}")

    # DeiT-Tiny
    model_deit = VisionTransformer(
        image_size=224, patch_size=16, num_layers=12, num_heads=3,
        hidden_dim=192, mlp_dim=768, num_classes=1000,
    )
    masks_deit = build_fixed_masks(model_deit, arch="deit_tiny")
    check("DeiT-Tiny: conv_proj.weight NOT in masks",
          "conv_proj.weight" not in masks_deit, "in_ch=3, K%4!=0")
    check("DeiT-Tiny: heads.head.weight IN masks",
          "heads.head.weight" in masks_deit, "Linear 192→1000")
    check("DeiT-Tiny: MLP linear IN masks",
          "encoder.layers.encoder_layer_0.mlp.0.weight" in masks_deit)
    expected_deit = 49
    check(f"DeiT-Tiny: {expected_deit} masked layers",
          len(masks_deit) == expected_deit, f"got {len(masks_deit)}")


def test_compliance_on_checkpoints() -> None:
    """Verify 2:4 compliance on saved sparse checkpoints."""
    print("\n=== Test 3: Checkpoint compliance ===")

    checkpoints = [
        ("ResNet-18 Imagenette", "resnet18_sparse_ft_imagenette_full.pth",
         lambda: resnet18(weights=None, num_classes=1000)),
        ("MobileNetV2 Imagenette", "mobilenet_v2_sparse_ft_imagenette_full.pth",
         lambda: mobilenet_v2(weights=None, num_classes=1000)),
        ("DeiT-Tiny Imagenette", "deit_tiny_sparse_ft_imagenette_full.pth",
         lambda: VisionTransformer(image_size=224, patch_size=16, num_layers=12,
                                   num_heads=3, hidden_dim=192, mlp_dim=768, num_classes=1000)),
        ("DeiT-Tiny CIFAR-100", "deit_tiny_sparse_ft_cifar100_20ep.pth",
         lambda: VisionTransformer(image_size=224, patch_size=16, num_layers=12,
                                   num_heads=3, hidden_dim=192, mlp_dim=768, num_classes=100)),
    ]

    for name, ckpt_name, model_fn in checkpoints:
        ckpt_path = WEIGHTS_DIR / ckpt_name
        if not ckpt_path.exists():
            check(f"{name}: checkpoint exists", False, str(ckpt_path))
            continue

        model = model_fn()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd)

        # Check all sparsified params have 2:4 pattern
        violations = 0
        checked = 0
        for pname, param in model.named_parameters():
            if param.dim() == 4:
                perm = param.data.detach().permute(0, 2, 3, 1).contiguous()
                K = param.shape[1] * param.shape[2] * param.shape[3]
                if K % 4 != 0:
                    continue
                flat = perm.view(-1, 4)
                nonzero_per_group = (flat != 0).sum(dim=1)
                is_sparse = bool(torch.all(nonzero_per_group <= 2))
                if is_sparse:
                    checked += 1
                    bad = int((nonzero_per_group != 2).sum().item())
                    if bad > 0:
                        violations += 1
            elif param.dim() == 2:
                in_f = param.shape[1]
                if in_f % 4 != 0:
                    continue
                flat = param.data.detach().view(-1, 4)
                nonzero_per_group = (flat != 0).sum(dim=1)
                is_sparse = bool(nonzero_per_group.float().mean() < 3.5)
                if is_sparse:
                    checked += 1
                    bad = int((nonzero_per_group != 2).sum().item())
                    if bad > 0:
                        violations += 1

        check(f"{name}: 0 violations in {checked} sparse layers",
              violations == 0, f"violations={violations}")


def test_baseline_accuracy() -> None:
    """Verify baseline accuracy from JSON metadata."""
    print("\n=== Test 4: Baseline accuracy sanity ===")

    expected = [
        ("resnet18_sparse_ft_imagenette_full.json", "final_val_top1", 95.0,
         "ResNet-18 Imagenette Sparse-FT >= 95%"),
        ("mobilenet_v2_sparse_ft_imagenette_full.json", "final_val_top1", 90.0,
         "MobileNetV2 Imagenette Sparse-FT >= 90%"),
        ("deit_tiny_sparse_ft_imagenette_full.json", "final_val_top1", 90.0,
         "DeiT-Tiny Imagenette Sparse-FT >= 90%"),
        ("deit_tiny_dense_ft_cifar100_20ep.json", "final_val_top1", 55.0,
         "DeiT-Tiny CIFAR-100 Dense-FT >= 55% (Phase 4 gate)"),
        ("deit_tiny_sparse_ft_cifar100_20ep.json", "final_val_top1", 55.0,
         "DeiT-Tiny CIFAR-100 Sparse-FT >= 55%"),
    ]

    for json_name, metric_key, threshold, desc in expected:
        json_path = WEIGHTS_DIR / json_name
        if not json_path.exists():
            check(desc, False, f"JSON not found: {json_path}")
            continue
        data = json.loads(json_path.read_text())
        val = data.get("metrics", {}).get(metric_key, 0.0)
        check(desc, val >= threshold, f"actual={val:.2f}%")


def main() -> None:
    global PASS, FAIL
    print("=" * 60)
    print("Smoke Test: 2:4 Sparsity Fix Verification")
    print("=" * 60)

    test_mask_k_dim_grouping()
    test_layer_selection()
    test_compliance_on_checkpoints()
    test_baseline_accuracy()

    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"Results: {PASS}/{total} passed, {FAIL}/{total} failed")
    print("=" * 60)

    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
