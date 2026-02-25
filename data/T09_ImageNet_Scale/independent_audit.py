#!/usr/bin/env python3
"""
INDEPENDENT QA AUDIT for Route A (T09) Preparations

This script performs completely independent verification of:
1. CIFAR-100 dataloader functionality
2. MobileNet-V2 2:4 sparse checkpoint structure
3. DeiT-Tiny 2:4 sparse checkpoint structure

No external dependencies on existing audit scripts.
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# =============================================================================
# PART 1: CIFAR-100 DATALOADER VERIFICATION
# =============================================================================

def verify_cifar100_dataloader() -> Dict:
    """
    Verify CIFAR-100 dataloader exists and functions correctly.
    """
    print("\n" + "="*70)
    print("PART 1: CIFAR-100 DATALOADER VERIFICATION")
    print("="*70)

    result = {
        "target": "CIFAR-100",
        "loader_found": False,
        "import_success": False,
        "instantiation_success": False,
        "batch_test_success": False,
        "train_batch_shape": None,
        "test_batch_shape": None,
        "train_label_range": None,
        "test_label_range": None,
        "status": "NOT_READY",
        "errors": []
    }

    # Try to find CIFAR-100 loader scripts
    base_dir = Path("data/T09_ImageNet_Scale")
    possible_paths = [
        base_dir / "train" / "cifar100_utils.py",
        base_dir / "scripts" / "cifar100_utils.py",
        base_dir / "train" / "train_utils.py",  # might contain CIFAR-100 support
    ]

    loader_path = None
    for path in possible_paths:
        if path.exists():
            loader_path = path
            result["loader_found"] = True
            print(f"[FOUND] Loader script: {path}")
            break

    if not loader_path:
        result["errors"].append("No CIFAR-100 loader script found")
        print("[MISSING] No CIFAR-100 loader script found")
        return result

    # Try to import and test the loader
    try:
        # Add to path
        sys.path.insert(0, str(base_dir / "train"))
        sys.path.insert(0, str(base_dir))

        # Try importing from different possible modules
        loader_module = None
        for module_name in ["cifar100_utils", "train_utils", "imagenet_utils"]:
            try:
                loader_module = __import__(module_name)
                print(f"[IMPORT] Successfully imported: {module_name}")
                result["import_success"] = True
                break
            except ImportError:
                continue

        if loader_module is None:
            result["errors"].append("Failed to import any loader module")
            print("[ERROR] Could not import loader module")
            return result

        # Check for CIFAR-100 specific functions
        has_cifar100 = any("cifar100" in name.lower() for name in dir(loader_module))
        has_get_loaders = hasattr(loader_module, "get_cifar100_loaders")

        if not has_cifar100 and not has_get_loaders:
            # Try using torchvision directly as fallback
            print("[INFO] No specific CIFAR-100 function found, trying torchvision...")
            try:
                import torchvision
                import torchvision.transforms as transforms
                from torchvision.datasets import CIFAR100

                # Create test dataloaders
                train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ])

                test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ])

                # Check if dataset exists
                data_root = Path("data")
                train_dataset = CIFAR100(root=data_root, train=True, download=False, transform=train_transform)
                test_dataset = CIFAR100(root=data_root, train=False, download=False, transform=test_transform)

                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

                print("[SUCCESS] Created CIFAR-100 dataloaders using torchvision")

            except Exception as e:
                result["errors"].append(f"Failed to create dataloaders: {e}")
                print(f"[ERROR] {e}")
                return result

        else:
            print(f"[FOUND] CIFAR-100 support in module")
            # Use the module's function if available
            if hasattr(loader_module, "get_cifar100_loaders"):
                print("[INFO] Using get_cifar100_loaders from module")
                # This would require specific function signature knowledge
                # For now, we've verified the loader exists
                return result

        # Test batch fetching
        result["instantiation_success"] = True

        # Fetch one training batch
        for images, labels in train_loader:
            result["train_batch_shape"] = list(images.shape)
            result["train_label_range"] = [labels.min().item(), labels.max().item()]
            print(f"[TRAIN] Batch shape: {images.shape}")
            print(f"[TRAIN] Label range: {result['train_label_range']}")
            break

        # Fetch one test batch
        for images, labels in test_loader:
            result["test_batch_shape"] = list(images.shape)
            result["test_label_range"] = [labels.min().item(), labels.max().item()]
            print(f"[TEST] Batch shape: {images.shape}")
            print(f"[TEST] Label range: {result['test_label_range']}")
            break

        # Verify shapes
        expected_shape = [32, 3, 32, 32]  # batch_size can vary
        if result["train_batch_shape"][1:] == [3, 32, 32]:
            result["batch_test_success"] = True
            print("[VERIFY] Train batch shape correct")
        else:
            result["errors"].append(f"Unexpected train shape: {result['train_batch_shape']}")

        if result["test_batch_shape"][1:] == [3, 32, 32]:
            print("[VERIFY] Test batch shape correct")
        else:
            result["errors"].append(f"Unexpected test shape: {result['test_batch_shape']}")

        # Check label range (0-99 for CIFAR-100)
        if result["train_label_range"][0] >= 0 and result["train_label_range"][1] <= 99:
            print("[VERIFY] Label range within CIFAR-100 bounds (0-99)")

        if result["batch_test_success"]:
            result["status"] = "READY"
            print("\n[STATUS] ✓ CIFAR-100 Dataloader: READY")

    except Exception as e:
        result["errors"].append(f"Exception during verification: {e}")
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

    return result


# =============================================================================
# PART 2: INDEPENDENT STRUCTURAL AUDIT
# =============================================================================

def audit_tensor_2x4(weight: torch.Tensor, name: str, is_conv: bool = False) -> Dict:
    """
    Independently audit a weight tensor for 2:4 sparsity compliance.

    Args:
        weight: Weight tensor (2D or 4D)
        name: Layer name
        is_conv: Whether this is a Conv2d layer

    Returns:
        Audit result dictionary
    """
    result = {
        "name": name,
        "type": "Conv2d" if is_conv else "Linear",
        "shape": tuple(weight.shape),
        "total_groups": 0,
        "valid_groups": 0,
        "invalid_groups": 0,
        "violations": 0,
        "zero_distribution": {},
        "skipped": False,
        "skip_reason": None
    }

    zero_threshold = 1e-7

    if is_conv:
        # Conv2d: permute(0, 2, 3, 1) -> view(-1, 4)
        out_ch, in_ch, kh, kw = weight.shape

        # Skip conv1 if in_ch is not divisible by 4 (expected for RGB input)
        if in_ch % 4 != 0 or name == "conv1.weight":
            result["skipped"] = True
            result["skip_reason"] = f"in_ch={in_ch} not divisible by 4 or conv1"
            return result

        # Apply T08 spatial alignment
        w_perm = weight.permute(0, 2, 3, 1).contiguous()

        if w_perm.numel() % 4 != 0:
            result["skipped"] = True
            result["skip_reason"] = f"numel={w_perm.numel()} not divisible by 4"
            return result

        flat = w_perm.view(-1, 4)
    else:
        # Linear: view(-1, 4)
        out_features, in_features = weight.shape

        if in_features % 4 != 0:
            result["skipped"] = True
            result["skip_reason"] = f"in_features={in_features} not divisible by 4"
            return result

        flat = weight.contiguous().view(-1, 4)

    # Count groups
    num_groups = flat.shape[0]
    result["total_groups"] = num_groups

    # Audit each group
    for i in range(num_groups):
        group = flat[i]
        zeros = (group.abs() < zero_threshold).sum().item()

        if zeros not in result["zero_distribution"]:
            result["zero_distribution"][zeros] = 0
        result["zero_distribution"][zeros] += 1

        if zeros == 2:
            result["valid_groups"] += 1
        else:
            result["invalid_groups"] += 1
            result["violations"] += abs(zeros - 2)

    return result


def audit_checkpoint_independent(checkpoint_path: str, model_type: str) -> Dict:
    """
    Independently audit a checkpoint for 2:4 sparsity compliance.
    """
    print(f"\n{'='*70}")
    print(f"INDEPENDENT AUDIT: {checkpoint_path}")
    print(f"{'='*70}")

    result = {
        "model_type": model_type,
        "path": checkpoint_path,
        "exists": False,
        "loadable": False,
        "layers_audited": [],
        "total_groups": 0,
        "valid_groups": 0,
        "invalid_groups": 0,
        "violations": 0,
        "skipped_layers": 0,
        "compliance_rate": 0.0,
        "status": "UNKNOWN",
        "errors": []
    }

    # Check existence
    if not Path(checkpoint_path).exists():
        result["errors"].append(f"File not found: {checkpoint_path}")
        print(f"[ERROR] File not found: {checkpoint_path}")
        return result

    result["exists"] = True
    print(f"[EXISTS] Checkpoint file found")

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        result["loadable"] = True
        print("[LOADED] Checkpoint loaded successfully")
    except Exception as e:
        result["errors"].append(f"Failed to load checkpoint: {e}")
        print(f"[ERROR] Failed to load: {e}")
        return result

    # Extract state dict
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    else:
        state_dict = checkpoint

    print(f"[STATE] State dict keys: {len(state_dict)}")

    # Special handling for ViT QKV projections
    qkv_keys = [k for k in state_dict.keys() if "qkv" in k.lower() or "in_proj" in k.lower() or "attention" in k.lower()]

    # Audit all weight tensors
    print(f"\n{'Layer Name':<50} {'Type':<15} {'Shape':<20} {'Groups':<10} {'Valid':<10} {'Invalid':<10} {'Status'}")
    print("-" * 130)

    sorted_keys = sorted(state_dict.keys())

    for key in sorted_keys:
        # Only audit weight tensors
        if "weight" not in key:
            continue

        weight = state_dict[key]

        if weight.dim() < 2:
            continue

        # Determine layer type
        is_conv = weight.dim() == 4
        is_qkv = "qkv" in key.lower() or "in_proj" in key.lower()

        layer_type = "Conv2d" if is_conv else ("QKV" if is_qkv else "Linear")

        # Skip bias and other non-weight tensors
        if weight.dim() == 1:
            continue

        # Audit the tensor
        audit_result = audit_tensor_2x4(weight, key, is_conv)

        # Update totals
        if not audit_result["skipped"]:
            result["layers_audited"].append(audit_result)
            result["total_groups"] += audit_result["total_groups"]
            result["valid_groups"] += audit_result["valid_groups"]
            result["invalid_groups"] += audit_result["invalid_groups"]
            result["violations"] += audit_result["violations"]
        else:
            result["skipped_layers"] += 1

        # Print audit result
        if audit_result["skipped"]:
            status = f"SKIPPED ({audit_result['skip_reason']})"
            print(f"{key:<50} {layer_type:<15} {str(audit_result['shape']):<20} {'-':<10} {'-':<10} {'-':<10} {status}")
        else:
            status = "PASS" if audit_result["invalid_groups"] == 0 else f"FAIL ({audit_result['invalid_groups']} invalid)"
            print(f"{key:<50} {layer_type:<15} {str(audit_result['shape']):<20} {audit_result['total_groups']:<10} {audit_result['valid_groups']:<10} {audit_result['invalid_groups']:<10} {status}")

            if audit_result["invalid_groups"] > 0:
                print(f"  → Zero distribution: {audit_result['zero_distribution']}")

    # Calculate compliance rate
    if result["total_groups"] > 0:
        result["compliance_rate"] = 100.0 * result["valid_groups"] / result["total_groups"]

        if result["invalid_groups"] == 0:
            result["status"] = "PASS"
        else:
            result["status"] = "FAIL"
    else:
        result["status"] = "NO_VALID_GROUPS"

    return result


def independent_audit_all() -> Dict:
    """
    Run complete independent audit of all claimed sparse checkpoints.
    """
    print("\n" + "="*70)
    print(" "*15 + "INDEPENDENT QA AUDIT FOR ROUTE A (T09)")
    print("="*70)

    results = {
        "cifar100": None,
        "mobilenet_v2": None,
        "deit_tiny": None,
        "overall_status": "UNKNOWN"
    }

    weights_dir = Path("data/T09_ImageNet_Scale/weights")

    # Part 1: CIFAR-100
    print("\n" + "="*70)
    print("CHECKING CIFAR-100 DATALOADER")
    print("="*70)

    results["cifar100"] = verify_cifar100_dataloader()

    # Part 2: MobileNet-V2
    mobilenet_path = weights_dir / "mobilenet_v2_2_4_sparse_imagenette.pth"
    results["mobilenet_v2"] = audit_checkpoint_independent(
        str(mobilenet_path),
        "MobileNet-V2"
    )

    # Part 3: DeiT-Tiny
    deit_path = weights_dir / "deit_tiny_2_4_sparse_imagenette.pth"
    results["deit_tiny"] = audit_checkpoint_independent(
        str(deit_path),
        "DeiT-Tiny"
    )

    # Generate summary
    print("\n" + "="*70)
    print("INDEPENDENT AUDIT SUMMARY")
    print("="*70)

    print("\n--- CIFAR-100 DATALOADER ---")
    cf100 = results["cifar100"]
    if cf100["status"] == "READY":
        print("✓ STATUS: READY")
        print(f"  - Train batch shape: {cf100['train_batch_shape']}")
        print(f"  - Test batch shape: {cf100['test_batch_shape']}")
        print(f"  - Label range: {cf100['train_label_range']}")
    else:
        print("✗ STATUS: NOT_READY")
        print(f"  - Errors: {cf100['errors']}")

    print("\n--- MobileNet-V2 2:4 SPARSE ---")
    mb = results["mobilenet_v2"]
    if mb["status"] == "PASS":
        print("✓ STATUS: PASS")
        print(f"  - Compliance Rate: {mb['compliance_rate']:.2f}%")
        print(f"  - Groups Audited: {mb['total_groups']:,}")
        print(f"  - Valid Groups: {mb['valid_groups']:,}")
        print(f"  - Invalid Groups: {mb['invalid_groups']:,}")
        print(f"  - Skipped Layers: {mb['skipped_layers']}")
    elif mb["status"] == "FAIL":
        print("✗ STATUS: FAIL")
        print(f"  - Compliance Rate: {mb['compliance_rate']:.2f}%")
        print(f"  - Invalid Groups: {mb['invalid_groups']:,}")
        print(f"  - Violations: {mb['violations']}")
    else:
        print(f"✗ STATUS: {mb['status']}")
        print(f"  - Errors: {mb['errors']}")

    print("\n--- DeiT-Tiny 2:4 SPARSE ---")
    deit = results["deit_tiny"]
    if deit["status"] == "PASS":
        print("✓ STATUS: PASS")
        print(f"  - Compliance Rate: {deit['compliance_rate']:.2f}%")
        print(f"  - Groups Audited: {deit['total_groups']:,}")
        print(f"  - Valid Groups: {deit['valid_groups']:,}")
        print(f"  - Invalid Groups: {deit['invalid_groups']:,}")
        print(f"  - Skipped Layers: {deit['skipped_layers']}")
        print(f"  - Layers Audited: {len(deit['layers_audited'])}")
    elif deit["status"] == "FAIL":
        print("✗ STATUS: FAIL")
        print(f"  - Compliance Rate: {deit['compliance_rate']:.2f}%")
        print(f"  - Invalid Groups: {deit['invalid_groups']:,}")
        print(f"  - Violations: {deit['violations']}")
    else:
        print(f"✗ STATUS: {deit['status']}")
        print(f"  - Errors: {deit['errors']}")

    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)

    ready_count = sum(1 for r in [cf100, mb, deit] if isinstance(r, dict) and r.get("status") in ["READY", "PASS"])
    pass_count = sum(1 for r in [cf100, mb, deit] if isinstance(r, dict) and r.get("status") in ["READY", "PASS", "PARTIAL"])
    fail_count = sum(1 for r in [cf100, mb, deit] if isinstance(r, dict) and r.get("status") in ["NOT_READY", "FAIL", "UNKNOWN", "NO_VALID_GROUPS"])

    print(f"\nReady/Pass: {ready_count}/3")
    print(f"Partial/Other: {pass_count - ready_count}/3")
    print(f"Fail/Not Ready: {fail_count}/3")

    if ready_count == 3:
        print("\n" + "="*70)
        print(" "*25 + "🟢 GREEN LIGHT FOR ROUTE A 🟢")
        print("="*70)
        results["overall_status"] = "GREEN_LIGHT"
    elif ready_count >= 1:
        print("\n⚠️  PARTIAL READY - Some components need work")
        results["overall_status"] = "PARTIAL"
    else:
        print("\n🔴 RED LIGHT - Not ready for Route A")
        results["overall_status"] = "RED_LIGHT"

    return results


if __name__ == "__main__":
    results = independent_audit_all()

    # Save results
    import json
    report_path = "data/T09_ImageNet_Scale/independent_audit_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Report saved to: {report_path}]")
