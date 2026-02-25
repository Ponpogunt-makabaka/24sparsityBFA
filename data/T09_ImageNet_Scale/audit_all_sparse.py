#!/usr/bin/env python3
"""
Comprehensive Structural Audit and Gap Analysis for Route A (T09) Preparations

This script performs:
1. Strict structural audit of all sparse models in data/T09_ImageNet_Scale/weights/
2. Gap analysis for CIFAR-100, MobileNet-V2, and ViT-Transformer readiness

Output: A detailed "Readiness Report" with compliance percentages and missing items checklist.
"""

import os
import sys
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# Part 1: Structural Audit Functions
# =============================================================================

def audit_conv2d_weight(weight: torch.Tensor, layer_name: str, skip_conv1: bool = True) -> Tuple[int, int, int, int, bool]:
    """
    Audit a Conv2d weight tensor using T08 spatial alignment.

    Args:
        weight: 4D weight tensor (out_ch, in_ch, kh, kw)
        layer_name: Name of the layer for reporting
        skip_conv1: Whether to skip conv1 (in_ch not div by 4)

    Returns:
        (total_groups, valid_groups, invalid_groups, total_violations, was_skipped)
    """
    assert weight.dim() == 4, f"{layer_name}: Expected 4D tensor, got {weight.dim()}D"

    out_ch, in_ch, kh, kw = weight.shape

    # Skip conv1 (first layer) as it typically doesn't follow 2:4 pattern
    # when in_ch is not divisible by 4 (e.g., RGB images have in_ch=3)
    if skip_conv1 and (layer_name == "conv1.weight" or in_ch < 4):
        return 0, 0, 0, 0, True  # Skipped

    # T08 spatial alignment: permute(0, 2, 3, 1) -> view(-1, 4)
    w_perm = weight.permute(0, 2, 3, 1).contiguous()
    numel = w_perm.numel()

    if numel % 4 != 0:
        return 0, 0, 0, 0, False  # Error

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

    return num_groups, valid_groups, invalid_groups, total_violations, False


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
        return 0, 0, 0, 0  # Error

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


def audit_checkpoint(checkpoint_path: str) -> Dict:
    """
    Audit a single checkpoint file.

    Returns:
        Dictionary with audit results
    """
    print(f"\n{'='*70}")
    print(f"AUDITING: {checkpoint_path}")
    print(f"{'='*70}")

    if not Path(checkpoint_path).exists():
        return {"error": f"File not found: {checkpoint_path}"}

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        format_type = "model_state_dict"
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        format_type = "state_dict"
    else:
        state_dict = checkpoint
        format_type = "raw"

    results = {
        "path": checkpoint_path,
        "format": format_type,
        "layers": [],
        "total_groups": 0,
        "total_valid": 0,
        "total_invalid": 0,
        "total_violations": 0,
        "skipped_layers": 0,
        "compliance_rate": 0.0,
        "status": "UNKNOWN",
        "val_top1": None,
    }

    # Try to recover validation accuracy from checkpoint metadata if available
    if isinstance(checkpoint, dict):
        if "final_val_top1" in checkpoint:
            results["val_top1"] = float(checkpoint["final_val_top1"])
        elif "metrics" in checkpoint and isinstance(checkpoint["metrics"], dict) and "final_val_top1" in checkpoint["metrics"]:
            results["val_top1"] = float(checkpoint["metrics"]["final_val_top1"])

    print(f"Format: {format_type}")
    print(f"Total keys: {len(state_dict)}")
    print()
    print(f"{'Layer Name':<40} {'Type':<10} {'Shape':<20} {'Groups':<10} {'Valid':<10} {'Invalid':<10}")
    print("-" * 110)

    sorted_keys = sorted(state_dict.keys())

    for key in sorted_keys:
        if "weight" not in key:
            continue

        weight = state_dict[key]

        if weight.dim() < 2:
            continue

        if weight.dim() == 4:
            groups, valid, invalid, violations, skipped = audit_conv2d_weight(weight, key)
            layer_type = "Conv2d"
        elif weight.dim() == 2:
            groups, valid, invalid, violations = audit_linear_weight(weight, key)
            layer_type = "Linear"
            skipped = False
        else:
            continue

        if skipped:
            results["skipped_layers"] += 1
            print(f"{key:<40} {layer_type:<10} {str(tuple(weight.shape)):<20} {'SKIPPED':<50}")
            continue

        if groups > 0:
            results["layers"].append({
                "name": key,
                "type": layer_type,
                "shape": tuple(weight.shape),
                "groups": groups,
                "valid": valid,
                "invalid": invalid,
                "violations": violations,
            })
            results["total_groups"] += groups
            results["total_valid"] += valid
            results["total_invalid"] += invalid
            results["total_violations"] += violations

            status = "PASS" if invalid == 0 else f"FAIL ({invalid} invalid)"
            print(f"{key:<40} {layer_type:<10} {str(tuple(weight.shape)):<20} {groups:<10} {valid:<10} {invalid:<10} {status}")

    # Calculate compliance rate
    if results["total_groups"] > 0:
        results["compliance_rate"] = 100.0 * results["total_valid"] / results["total_groups"]
        if results["total_invalid"] == 0:
            results["status"] = "PASS"
        else:
            results["status"] = "FAIL"

    return results


def audit_all_checkpoints(weights_dir: str) -> List[Dict]:
    """Audit all checkpoint files in a directory."""
    weights_path = Path(weights_dir)
    if not weights_path.exists():
        print(f"[ERROR] Weights directory not found: {weights_dir}")
        return []

    # Find all .pth files and de-duplicate
    pth_paths = set()
    for p in weights_path.glob("*.pth"):
        pth_paths.add(p.resolve())
    for p in weights_path.glob("**/*.pth"):
        pth_paths.add(p.resolve())
    pth_files = sorted([Path(p) for p in pth_paths])

    # Filter for sparse models (assume they have "sparse" in filename or we check all)
    sparse_files = [f for f in pth_files if "sparse" in f.name.lower()]

    if not sparse_files:
        print("[INFO] No files with 'sparse' in filename found. Auditing all .pth files.")
        sparse_files = pth_files

    results = []
    for pth_file in sparse_files:
        result = audit_checkpoint(str(pth_file))
        results.append(result)

    return results


# =============================================================================
# Part 2: Gap Analysis Functions
# =============================================================================

def check_cifar100_readiness(data_dir: str) -> Dict:
    """Check CIFAR-100 dataset and dataloader readiness."""
    print(f"\n{'='*70}")
    print("GAP ANALYSIS: CIFAR-100")
    print(f"{'='*70}")

    result = {
        "target": "CIFAR-100",
        "dataset_downloaded": False,
        "dataloader_script_exists": False,
        "dataloader_path": None,
        "missing_items": [],
        "status": "NOT_READY"
    }

    # Check for CIFAR-100 data
    cifar_paths = [
        Path(data_dir) / "cifar-100-python",
        Path(data_dir) / "cifar-100",
        Path(data_dir) / "cifar100",
    ]

    # Also check for tar files
    cifar_tars = [
        Path(data_dir) / "cifar-100-python.tar.gz",
    ]

    dataset_found = False
    for p in cifar_paths:
        if p.exists():
            dataset_found = True
            result["dataset_downloaded"] = True
            print(f"[FOUND] CIFAR-100 data: {p}")
            break

    if not dataset_found:
        for p in cifar_tars:
            if p.exists():
                dataset_found = True
                result["dataset_downloaded"] = True
                print(f"[FOUND] CIFAR-100 tarball: {p}")
                break

    if not dataset_found:
        print("[MISSING] CIFAR-100 dataset not found")
        result["missing_items"].append("CIFAR-100 dataset download")

    # Check for dataloader scripts
    t09_dir = Path(data_dir) / "T09_ImageNet_Scale"
    if t09_dir.exists():
        # Check for CIFAR-100 specific scripts
        cifar100_scripts = [
            t09_dir / "scripts" / "download_cifar100.py",
            t09_dir / "scripts" / "cifar100_utils.py",
            t09_dir / "train" / "cifar100_utils.py",
        ]

        for script in cifar100_scripts:
            if script.exists():
                result["dataloader_script_exists"] = True
                result["dataloader_path"] = str(script)
                print(f"[FOUND] CIFAR-100 script: {script}")
                break

        # Also check train_utils.py for CIFAR-100 support
        train_utils = t09_dir / "train" / "train_utils.py"
        if train_utils.exists():
            content = train_utils.read_text()
            if "cifar100" in content.lower() or "CIFAR100" in content:
                result["dataloader_script_exists"] = True
                print(f"[FOUND] CIFAR-100 support in: {train_utils}")

    if not result["dataloader_script_exists"]:
        print("[MISSING] CIFAR-100 dataloader script")
        result["missing_items"].append("CIFAR-100 dataloader implementation")

    # Determine status
    if result["dataset_downloaded"] and result["dataloader_script_exists"]:
        result["status"] = "READY"
    elif result["dataset_downloaded"]:
        result["status"] = "PARTIAL"
    else:
        result["status"] = "NOT_READY"

    return result


def check_mobilenet_v2_readiness(weights_dir: str) -> Dict:
    """Check MobileNet-V2 sparsity and fine-tuning readiness."""
    print(f"\n{'='*70}")
    print("GAP ANALYSIS: MobileNet-V2")
    print(f"{'='*70}")

    result = {
        "target": "MobileNet-V2",
        "dense_weights_downloaded": False,
        "sparse_checkpoint_exists": False,
        "sparse_path": None,
        "missing_items": [],
        "status": "NOT_READY"
    }

    weights_path = Path(weights_dir)

    # Check for dense weights
    dense_names = ["mobilenet_v2", "mobilenetv2", "mobile_net"]
    for f in weights_path.glob("*.pth"):
        if any(name in f.name.lower() for name in dense_names):
            result["dense_weights_downloaded"] = True
            print(f"[FOUND] Dense MobileNet-V2 weights: {f}")

            # Check if this is already a sparse checkpoint
            if "sparse" in f.name.lower() or "2_4" in f.name.lower() or "2-4" in f.name.lower():
                result["sparse_checkpoint_exists"] = True
                result["sparse_path"] = str(f)
                print(f"[FOUND] Sparse MobileNet-V2 checkpoint: {f}")
            break

    # Check for separate sparse checkpoint
    if not result["sparse_checkpoint_exists"]:
        for f in weights_path.glob("*sparse*mobile*.pth"):
            result["sparse_checkpoint_exists"] = True
            result["sparse_path"] = str(f)
            print(f"[FOUND] Sparse MobileNet-V2 checkpoint: {f}")
            break
        for f in weights_path.glob("*mobile*sparse*.pth"):
            result["sparse_checkpoint_exists"] = True
            result["sparse_path"] = str(f)
            print(f"[FOUND] Sparse MobileNet-V2 checkpoint: {f}")
            break

    # Determine missing items
    if not result["dense_weights_downloaded"]:
        result["missing_items"].append("MobileNet-V2 dense weights download")

    if not result["sparse_checkpoint_exists"]:
        result["missing_items"].append("MobileNet-V2 2:4 sparsification")
        result["missing_items"].append("MobileNet-V2 sparse fine-tuning")

    # Determine status
    if result["sparse_checkpoint_exists"]:
        result["status"] = "READY"
    elif result["dense_weights_downloaded"]:
        result["status"] = "PARTIAL"
    else:
        result["status"] = "NOT_READY"

    return result


def check_vit_transformer_readiness(weights_dir: str) -> Dict:
    """Check ViT/DeiT-Tiny model readiness."""
    print(f"\n{'='*70}")
    print("GAP ANALYSIS: ViT-Transformer (DeiT-Tiny)")
    print(f"{'='*70}")

    result = {
        "target": "ViT-Transformer (DeiT-Tiny)",
        "dense_weights_downloaded": False,
        "sparse_checkpoint_exists": False,
        "sparse_path": None,
        "model_factory_support": False,
        "missing_items": [],
        "status": "NOT_READY"
    }

    weights_path = Path(weights_dir)

    # Check for DeiT/ViT dense weights
    vit_names = ["deit", "vit", "vision_transformer", "transformer"]
    for f in weights_path.glob("*.pth"):
        if any(name in f.name.lower() for name in vit_names):
            result["dense_weights_downloaded"] = True
            print(f"[FOUND] DeiT/ViT dense weights: {f}")

            # Check if already sparse
            if "sparse" in f.name.lower() or "2_4" in f.name.lower():
                result["sparse_checkpoint_exists"] = True
                result["sparse_path"] = str(f)
                print(f"[FOUND] Sparse ViT checkpoint: {f}")
            break

    # Check for separate sparse checkpoint
    if not result["sparse_checkpoint_exists"]:
        for f in weights_path.glob("*sparse*deit*.pth"):
            result["sparse_checkpoint_exists"] = True
            result["sparse_path"] = str(f)
            print(f"[FOUND] Sparse DeiT checkpoint: {f}")
            break
        for f in weights_path.glob("*deit*sparse*.pth"):
            result["sparse_checkpoint_exists"] = True
            result["sparse_path"] = str(f)
            print(f"[FOUND] Sparse DeiT checkpoint: {f}")
            break

    # Check model factory for ViT/DeiT support
    factory_path = Path(weights_path).parent.parent / "T09_ImageNet_Scale" / "models" / "factory.py"
    if factory_path.exists():
        content = factory_path.read_text()
        if "create_imagenet_deit" in content or "deit_tiny" in content.lower():
            result["model_factory_support"] = True
            print(f"[FOUND] ViT/DeiT support in factory.py")
        else:
            print("[INFO] factory.py exists but no DeiT support detected")

    # Determine missing items
    if not result["dense_weights_downloaded"]:
        result["missing_items"].append("DeiT-Tiny dense weights download")

    if not result["sparse_checkpoint_exists"]:
        result["missing_items"].append("DeiT-Tiny 2:4 sparsification")
        result["missing_items"].append("DeiT-Tiny sparse fine-tuning")

    if not result["model_factory_support"]:
        result["missing_items"].append("ViT/DeiT model factory implementation")

    # Determine status
    if result["sparse_checkpoint_exists"] and result["model_factory_support"]:
        result["status"] = "READY"
    elif result["dense_weights_downloaded"] or result["model_factory_support"]:
        result["status"] = "PARTIAL"
    else:
        result["status"] = "NOT_READY"

    return result


# =============================================================================
# Main Report Generation
# =============================================================================

def generate_readiness_report(data_dir: str = "data"):
    """Generate comprehensive readiness report."""
    print("\n" + "="*70)
    print(" "*20 + "ROUTE A (T09) READINESS REPORT")
    print("="*70)

    weights_dir = Path(data_dir) / "T09_ImageNet_Scale" / "weights"

    # Part 1: Structural Audit
    print("\n" + "="*70)
    print("PART 1: STRUCTURAL AUDIT OF SPARSE MODELS")
    print("="*70)

    audit_results = audit_all_checkpoints(str(weights_dir))

    # Part 2: Gap Analysis
    print("\n" + "="*70)
    print("PART 2: GAP ANALYSIS FOR REMAINING TARGETS")
    print("="*70)

    gap_results = {
        "cifar100": check_cifar100_readiness(data_dir),
        "mobilenet_v2": check_mobilenet_v2_readiness(str(weights_dir)),
        "vit_transformer": check_vit_transformer_readiness(str(weights_dir)),
    }

    # Generate Summary
    print("\n" + "="*70)
    print("READINESS SUMMARY")
    print("="*70)

    # Audit Summary
    print("\n--- SPARSE MODEL AUDIT RESULTS ---")
    for result in audit_results:
        status_icon = "✓" if result["status"] == "PASS" else "✗"
        print(f"\n{status_icon} {Path(result['path']).name}")
        print(f"   Compliance Rate: {result['compliance_rate']:.2f}%")
        print(f"   Groups Audited: {result['total_groups']:,}")
        print(f"   Valid Groups:   {result['total_valid']:,}")
        if result.get("val_top1") is not None:
            print(f"   Val Top-1:      {result['val_top1']:.2f}%")
        if result['total_invalid'] > 0:
            print(f"   Invalid Groups: {result['total_invalid']:,} ← FAIL")
        if result['skipped_layers'] > 0:
            print(f"   Skipped Layers: {result['skipped_layers']} (typically conv1)")

    # Gap Analysis Summary
    print("\n--- TARGET READINESS CHECKLIST ---")
    for target_name, result in gap_results.items():
        status_icon = {
            "READY": "✓",
            "PARTIAL": "◐",
            "NOT_READY": "✗"
        }.get(result["status"], "?")

        print(f"\n{status_icon} {result['target']}: {result['status']}")

        if result['missing_items']:
            print("   Missing items:")
            for item in result['missing_items']:
                print(f"      • {item}")

    # Final Verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)

    ready_count = sum(1 for r in gap_results.values() if r["status"] == "READY")
    partial_count = sum(1 for r in gap_results.values() if r["status"] == "PARTIAL")
    not_ready_count = sum(1 for r in gap_results.values() if r["status"] == "NOT_READY")

    print(f"\nReady for T08 Attack:    {ready_count}")
    print(f"Partially Ready:          {partial_count}")
    print(f"Not Ready:                {not_ready_count}")

    # Save report to file
    report_path = Path(data_dir) / "T09_ImageNet_Scale" / "readiness_report.json"
    report_data = {
        "audit_results": audit_results,
        "gap_analysis": gap_results,
        "summary": {
            "ready": ready_count,
            "partial": partial_count,
            "not_ready": not_ready_count,
        }
    }

    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    print(f"\n[Report saved to: {report_path}]")

    return audit_results, gap_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive T09 Readiness Audit")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory path")
    args = parser.parse_args()

    generate_readiness_report(args.data_dir)
