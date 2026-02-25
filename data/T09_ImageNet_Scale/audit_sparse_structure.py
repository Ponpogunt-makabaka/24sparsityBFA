#!/usr/bin/env python3
"""
Structural Audit Script for 2:4 Sparse Model

This script verifies that a checkpoint perfectly complies with the 2:4 sparsity rule:
- For every group of 4 elements, exactly 2 are zeros and 2 are non-zeros.

This audit uses the T08 spatial alignment reshaping for Conv2d layers:
- Conv2d (4D): w.permute(0, 2, 3, 1).contiguous().view(-1, 4)
- Linear (2D): w.view(-1, 4)

This is a READ-ONLY audit. No modifications are made to the model.
"""

import torch
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def audit_conv2d_weight(weight: torch.Tensor, layer_name: str) -> Tuple[int, int, int, int]:
    """
    Audit a Conv2d weight tensor using T08 spatial alignment.

    IMPORTANT: The first conv layer (conv1) typically has in_ch=3 which is not
    divisible by 4. This means the 2:4 sparsity pattern is NOT applied to
    conv1 in most implementations. We skip conv1 from the audit.

    Args:
        weight: 4D weight tensor (out_ch, in_ch, kh, kw)
        layer_name: Name of the layer for reporting

    Returns:
        (total_groups, valid_groups, invalid_groups, total_violations)
    """
    assert weight.dim() == 4, f"{layer_name}: Expected 4D tensor, got {weight.dim()}D"

    out_ch, in_ch, kh, kw = weight.shape

    # Skip conv1 (first layer) as it typically doesn't follow 2:4 pattern
    # when in_ch is not divisible by 4 (e.g., RGB images have in_ch=3)
    if layer_name == "conv1.weight" or in_ch < 4:
        print(f"{layer_name:<40} {'Conv2d':<10} {str(tuple(weight.shape)):<20} {'SKIPPED':<10} {'(in_ch not div by 4)':<50}")
        return 0, 0, 0, 0  # Skipped, not counted

    # T08 spatial alignment: permute(0, 2, 3, 1) -> view(-1, 4)
    w_perm = weight.permute(0, 2, 3, 1).contiguous()
    numel = w_perm.numel()

    if numel % 4 != 0:
        print(f"[ERROR] {layer_name}: numel={numel} not divisible by 4!")
        return 0, 0, 0, 0

    flat = w_perm.view(-1, 4)
    num_groups = flat.shape[0]

    valid_groups = 0
    invalid_groups = 0
    total_violations = 0

    zero_threshold = 1e-7

    for g_idx in range(num_groups):
        group = flat[g_idx]
        zeros = (group.abs() < zero_threshold).sum().item()

        if zeros == 2:
            valid_groups += 1
        else:
            invalid_groups += 1
            total_violations += abs(zeros - 2)

    return num_groups, valid_groups, invalid_groups, total_violations


def audit_linear_weight(weight: torch.Tensor, layer_name: str) -> Tuple[int, int, int, int]:
    """
    Audit a Linear weight tensor using standard grouping.

    Args:
        weight: 2D weight tensor (out_ch, in_ch)
        layer_name: Name of the layer for reporting

    Returns:
        (total_groups, valid_groups, invalid_groups, total_violations)
    """
    assert weight.dim() == 2, f"{layer_name}: Expected 2D tensor, got {weight.dim()}D"

    numel = weight.numel()

    if numel % 4 != 0:
        print(f"[ERROR] {layer_name}: numel={numel} not divisible by 4!")
        return 0, 0, 0, 0

    flat = weight.contiguous().view(-1, 4)
    num_groups = flat.shape[0]

    valid_groups = 0
    invalid_groups = 0
    total_violations = 0

    zero_threshold = 1e-7

    for g_idx in range(num_groups):
        group = flat[g_idx]
        zeros = (group.abs() < zero_threshold).sum().item()

        if zeros == 2:
            valid_groups += 1
        else:
            invalid_groups += 1
            total_violations += abs(zeros - 2)

    return num_groups, valid_groups, invalid_groups, total_violations


def main():
    checkpoint_path = "data/T09_ImageNet_Scale/weights/resnet18_2_4_sparse_imagenette.pth"

    print("=" * 70)
    print("2:4 SPARSE MODEL STRUCTURAL AUDIT")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print()

    # Load checkpoint
    if not Path(checkpoint_path).exists():
        print(f"[ERROR] Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)

    print("[1] Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"    Format: State dict with 'model_state_dict' key")
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print(f"    Format: State dict with 'state_dict' key")
    else:
        state_dict = checkpoint
        print(f"    Format: Raw state dict")

    print(f"    Total keys in state_dict: {len(state_dict)}")
    print()

    # Audit results
    total_groups = 0
    total_valid = 0
    total_invalid = 0
    total_violations = 0
    layer_results = []

    print("[2] Auditing weight tensors...")
    print()
    print(f"{'Layer Name':<40} {'Type':<10} {'Shape':<20} {'Groups':<10} {'Valid':<10} {'Invalid':<10} {'Status'}")
    print("-" * 120)

    # Sort keys for consistent output
    sorted_keys = sorted(state_dict.keys())

    for key in sorted_keys:
        # Skip non-weight tensors
        if "weight" not in key:
            continue

        weight = state_dict[key]

        # Skip 1D weights (like bias)
        if weight.dim() < 2:
            continue

        layer_type = "Unknown"
        if "conv" in key.lower():
            layer_type = "Conv2d"
            groups, valid, invalid, violations = audit_conv2d_weight(weight, key)
        elif "fc" in key.lower() or "linear" in key.lower() or weight.dim() == 2:
            layer_type = "Linear"
            groups, valid, invalid, violations = audit_linear_weight(weight, key)
        else:
            # Unknown type, skip
            continue

        if groups > 0:
            total_groups += groups
            total_valid += valid
            total_invalid += invalid
            total_violations += violations

            status = "PASS" if invalid == 0 else f"FAIL ({invalid} invalid)"

            layer_results.append({
                "name": key,
                "type": layer_type,
                "shape": tuple(weight.shape),
                "groups": groups,
                "valid": valid,
                "invalid": invalid,
                "violations": violations,
            })

            print(f"{key:<40} {layer_type:<10} {str(tuple(weight.shape)):<20} {groups:<10} {valid:<10} {invalid:<10} {status}")

    print("-" * 120)
    print()

    # Summary
    print("=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    print(f"Total layers audited: {len(layer_results)}")
    print(f"Total groups checked: {total_groups:,}")
    print(f"Valid groups (2:4):   {total_valid:,}")
    print(f"Invalid groups:       {total_invalid:,}")
    print(f"Total violations:     {total_violations:,}")
    print()

    compliance_rate = (total_valid / total_groups * 100) if total_groups > 0 else 0
    print(f"2:4 Compliance Rate:  {compliance_rate:.2f}%")
    print()

    if total_invalid == 0:
        print("=" * 70)
        print("RESULT: PASS - Model is 100% ready for T08 Index Encoding attack!")
        print("=" * 70)
        return 0
    else:
        print("=" * 70)
        print("RESULT: FAIL - Model has 2:4 sparsity violations!")
        print("=" * 70)
        print()
        print("Details of invalid layers:")
        for result in layer_results:
            if result["invalid"] > 0:
                print(f"  - {result['name']}: {result['invalid']} invalid groups, {result['violations']} total violations")
        return 1


if __name__ == "__main__":
    sys.exit(main())
