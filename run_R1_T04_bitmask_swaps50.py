#!/usr/bin/env python3
"""
R1_T04: Group-based 2:4 Metadata Attack (Bitmask Encoding, 50 Logical Swaps)

This is the FOURTH attack in the revised R1 workflow (R1_T04) that implements a
group-local greedy search using BITMASK encoding with a fixed number of LOGICAL
swaps for fair comparison with index-encoding attacks.

Purpose:
- R1_T03 used physical_budget=50, yielding only 25 logical swaps (cost=2 per swap)
- R1_T04 fixes logical_swaps=50 to compare fairly with R1_T01/T02 (50 operations)
- This allows direct comparison: bitmask (50 swaps) vs index (50 operations)

Key difference from R1_T03:
- Primary control: max_logical_swaps (default 50)
- Physical flips auto-calculated as 2 * max_logical_swaps
- Plot x-axis: logical swaps (0..50) instead of physical flips
- Comparison with R1_T03 included in output

Bitmask encoding:
- Direct 4-bit mask where each bit corresponds to a position in the group
- Valid mask: popcount(mask) == 2 (exactly two active positions)
- Cost-2 swap: flip one 1->0 and one 0->1 (maintains popcount=2)

This is NOT a retraining - reuses existing baseline checkpoint.

Naming Convention:
- R1_ = Revised workflow 1
- T04 = Task 04 (bitmask encoding with 50 logical swaps)
- Outputs follow: results/R1/R1_T04_<name>_<type>.ext

Outputs:
- results/R1/R1_T04_bitmask_swaps50_curve.png
- results/R1/R1_T04_bitmask_swaps50_table.csv
- results/R1/R1_T04_bitmask_swaps50_log.txt
- results/R1/R1_T04_bitmask_swaps50_result.pkl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import pickle
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from models.factory import create_resnet20
from scripts.p012_17_utils import (
    evaluate_subset,
    load_cifar10_loaders_offline,
    set_all_seeds,
)
from train.ptq_convert import Int8QuantizedResNet


# =============================================================================
# Constants - Bitmask Encoding
# =============================================================================

def pattern_to_bitmask(pattern: Tuple[int, int]) -> int:
    """Convert a 2-of-4 pattern tuple to a 4-bit bitmask."""
    mask = 0
    mask |= (1 << pattern[0])
    mask |= (1 << pattern[1])
    return mask


def bitmask_to_pattern(mask: int) -> Optional[Tuple[int, int]]:
    """Convert a 4-bit bitmask to a 2-of-4 pattern tuple. Returns None if popcount != 2."""
    if mask < 0 or mask > 15:
        return None
    positions = [i for i in range(4) if (mask >> i) & 1]
    if len(positions) != 2:
        return None  # Invalid: popcount != 2
    return (positions[0], positions[1])


def popcount(mask: int) -> int:
    """Count the number of set bits in a 4-bit mask."""
    return bin(mask).count('1')


def enumerate_cost2_swaps(current_mask: int) -> List[Tuple[int, int, int, int]]:
    """
    Enumerate all valid cost-2 swaps from current mask.

    A cost-2 swap flips exactly two bits:
    - One 1 -> 0 (turn off an active position)
    - One 0 -> 1 (turn on an inactive position)

    This yields exactly 4 candidates for any valid 2-of-4 mask.

    Returns: List of (new_mask, bit_off, bit_on, swap_code)
    """
    pattern = bitmask_to_pattern(current_mask)
    if pattern is None:
        return []

    pos0, pos1 = pattern
    inactive = [i for i in range(4) if i not in (pos0, pos1)]

    swaps = []
    for active_pos in (pos0, pos1):
        for inactive_pos in inactive:
            new_mask = current_mask ^ ((1 << active_pos) | (1 << inactive_pos))
            swap_code = (active_pos << 8) | inactive_pos
            swaps.append((new_mask, active_pos, inactive_pos, swap_code))

    return swaps


def get_bitmask_hash(model: nn.Module) -> str:
    """Compute a hash of all sparse_mask metadata for tracking changes."""
    hasher = hashlib.sha256()
    for name, module in model.named_modules():
        if hasattr(module, "sparse_mask") and module.sparse_mask is not None:
            mask_bytes = module.sparse_mask.cpu().numpy().tobytes()
            hasher.update(mask_bytes)
    return hasher.hexdigest()[:16]


ALL_2OF4_PATTERNS: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
]


# =============================================================================
# Data Structures
# =============================================================================

@dataclass(frozen=True)
class SwapCandidate:
    """A candidate swap change within a single group."""
    score: float
    layer_name: str
    module: nn.Module
    group_idx: int
    old_mask: int
    new_mask: int
    bit_off: int
    bit_on: int
    swap_code: int
    old_pattern: Tuple[int, int]
    new_pattern: Tuple[int, int]
    delta_w_tilde: torch.Tensor


@dataclass
class AttackStepLog:
    """Log entry for a single attack step."""
    swap_idx: int
    physical_flips: int
    layer_name: str
    group_idx: int
    old_mask: int
    new_mask: int
    bit_off: int
    bit_on: int
    old_pattern: Tuple[int, int]
    new_pattern: Tuple[int, int]
    proxy_score: float
    accuracy: float = 0.0
    loss: float = 0.0
    metadata_hash: str = ""
    validity_check: str = ""  # For tracking validity assertions


# =============================================================================
# Core Utilities
# =============================================================================

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def flatten_groups(t: torch.Tensor):
    """Flatten tensor into groups of 4 for 2:4 sparsity."""
    if t.dim() == 4:
        t_perm = t.permute(0, 2, 3, 1).contiguous()
        flat = t_perm.view(-1, 4)
        meta = ("conv", t_perm.shape)
        return flat, meta
    if t.dim() == 2:
        t_perm = t.contiguous()
        flat = t_perm.view(-1, 4)
        meta = ("linear", t_perm.shape)
        return flat, meta
    return None, None


def restore_groups(flat: torch.Tensor, meta):
    """Restore original tensor shape from flattened groups."""
    kind, shape = meta
    if kind == "conv":
        t_perm = flat.view(shape)
        return t_perm.permute(0, 3, 1, 2).contiguous()
    return flat.view(shape)


def get_sparse_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Get all layers with sparse masks."""
    layers = []
    for name, module in model.named_modules():
        if hasattr(module, "int8_weights") and hasattr(module, "scale") and hasattr(module, "sparse_mask"):
            if module.sparse_mask is not None:
                module.sparse_mask = (module.sparse_mask > 0.5).to(module.sparse_mask.dtype)
                layers.append((name, module))
    return layers


def get_current_bitmask(mask_group: torch.Tensor) -> Optional[int]:
    """Extract the current 4-bit bitmask from a mask group. Returns None if popcount != 2."""
    active = (mask_group > 0.5).nonzero(as_tuple=False).flatten().tolist()
    if len(active) != 2:
        return None
    a, b = int(active[0]), int(active[1])
    return pattern_to_bitmask((a, b))


def bitmask_to_mask_tensor(mask: int, device: torch.device) -> torch.Tensor:
    """Convert a 4-bit bitmask to a 4-element mask tensor."""
    tensor_mask = torch.zeros(4, dtype=torch.float32, device=device)
    for i in range(4):
        if (mask >> i) & 1:
            tensor_mask[i] = 1.0
    return tensor_mask


def compute_dense_reconstruction(
    int8_w_flat: torch.Tensor,
    mask_flat: torch.Tensor,
    scale: float,
    group_idx: int
) -> torch.Tensor:
    """Compute the dense reconstruction w̃_g for a single group."""
    w_group = int8_w_flat[group_idx]
    m_group = mask_flat[group_idx]
    w_tilde = w_group.float() * scale * m_group
    return w_tilde


# =============================================================================
# Gradient Computation
# =============================================================================

def compute_gradients(
    model: nn.Module,
    calib_loader,
    device: str,
    calib_samples: int
) -> bool:
    """Compute gradients wrt reconstructed dense weights for all layers."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    calib_data = []
    calib_targets = []
    for inputs, targets in calib_loader:
        calib_data.append(inputs.to(device))
        calib_targets.append(targets.to(device))
        if len(calib_data) * inputs.size(0) >= calib_samples:
            break

    calib_inputs = torch.cat(calib_data, dim=0)[:calib_samples]
    calib_targets = torch.cat(calib_targets, dim=0)[:calib_samples]

    outputs = model(calib_inputs)
    loss = criterion(outputs, calib_targets)
    model.zero_grad()
    loss.backward()

    return True


# =============================================================================
# Candidate Enumeration - Cost-2 Swaps
# =============================================================================

def enumerate_cost2_swap_candidates(
    model: nn.Module,
    device: str,
    exclude_groups: Set[Tuple[str, int]] = None,
    forbidden_swaps: Set[Tuple[str, int, int]] = None,
) -> Tuple[List[SwapCandidate], Dict[str, int]]:
    """
    Enumerate all valid cost-2 swap candidates for all groups.

    For each group:
    1. Get current 4-bit mask m (must have popcount=2)
    2. Enumerate all 4 cost-2 swaps
    3. Skip forbidden swaps
    4. Compute proxy score
    5. Keep the best candidate for this group
    """
    if exclude_groups is None:
        exclude_groups = set()
    if forbidden_swaps is None:
        forbidden_swaps = set()

    counters = {
        "total_groups": 0,
        "valid_groups": 0,
        "candidates_considered": 0,
        "candidates_valid": 0,
        "candidates_skipped_excluded": 0,
        "candidates_skipped_forbidden": 0,
        "candidates_no_gradient": 0,
        "validity_checks_passed": 0,
        "validity_checks_failed": 0,
    }

    candidates: List[SwapCandidate] = []

    for layer_name, module in get_sparse_layers(model):
        if module.weight.grad is None:
            continue

        grad = module.weight.grad.data
        int8_w = module.int8_weights
        mask = module.sparse_mask
        scale = float(module.scale.item())

        g_flat, _ = flatten_groups(grad)
        w_flat, w_meta = flatten_groups(int8_w)
        m_flat, m_meta = flatten_groups(mask)

        if g_flat is None or w_flat is None or m_flat is None:
            continue

        num_groups = w_flat.shape[0]
        counters["total_groups"] += num_groups

        for g_idx in range(num_groups):
            if (layer_name, int(g_idx)) in exclude_groups:
                counters["candidates_skipped_excluded"] += 1
                continue

            m_group = m_flat[g_idx]
            current_mask = get_current_bitmask(m_group)

            if current_mask is None:
                continue

            # Validity check: popcount must be 2
            if popcount(current_mask) != 2:
                counters["validity_checks_failed"] += 1
                continue
            counters["validity_checks_passed"] += 1

            counters["valid_groups"] += 1

            current_pattern = bitmask_to_pattern(current_mask)
            if current_pattern is None:
                continue

            w_tilde_current = compute_dense_reconstruction(w_flat, m_flat, scale, g_idx)
            grad_group = g_flat[g_idx]

            swaps = enumerate_cost2_swaps(current_mask)

            best_score_for_group = float('-inf')
            best_candidate_for_group: Optional[SwapCandidate] = None

            for new_mask, bit_off, bit_on, swap_code in swaps:
                # Validity check for new mask
                if popcount(new_mask) != 2:
                    counters["validity_checks_failed"] += 1
                    continue

                swap_key = (layer_name, int(g_idx), swap_code)
                if swap_key in forbidden_swaps:
                    counters["candidates_skipped_forbidden"] += 1
                    continue

                counters["candidates_considered"] += 1

                new_pattern = bitmask_to_pattern(new_mask)
                if new_pattern is None:
                    continue

                new_mask_tensor = bitmask_to_mask_tensor(new_mask, device)
                w_tilde_new = w_flat[g_idx].float() * scale * new_mask_tensor

                delta_w_tilde = w_tilde_new - w_tilde_current
                proxy_score = float(torch.dot(grad_group, delta_w_tilde).item())

                counters["candidates_valid"] += 1
                counters["validity_checks_passed"] += 1

                if proxy_score > best_score_for_group:
                    best_score_for_group = proxy_score
                    best_candidate_for_group = SwapCandidate(
                        score=proxy_score,
                        layer_name=layer_name,
                        module=module,
                        group_idx=int(g_idx),
                        old_mask=current_mask,
                        new_mask=new_mask,
                        bit_off=bit_off,
                        bit_on=bit_on,
                        swap_code=swap_code,
                        old_pattern=current_pattern,
                        new_pattern=new_pattern,
                        delta_w_tilde=delta_w_tilde.cpu(),
                    )

            if best_candidate_for_group is not None:
                candidates.append(best_candidate_for_group)

        if module.weight.grad is not None:
            module.weight.grad = None

    return candidates, counters


def apply_swap_change(candidate: SwapCandidate) -> bool:
    """Apply a swap change to the model."""
    module = candidate.module
    g_idx = candidate.group_idx
    new_mask = candidate.new_mask

    mask = module.sparse_mask
    int8_w = module.int8_weights

    m_flat, m_meta = flatten_groups(mask)
    w_flat, w_meta = flatten_groups(int8_w)

    if m_flat is None or w_flat is None:
        return False

    device = mask.device

    old_pattern = candidate.old_pattern
    old_mask_tensor = m_flat[g_idx].clone()
    old_active_indices = (old_mask_tensor > 0.5).nonzero().flatten()
    old_values = w_flat[g_idx, old_active_indices].clone()

    w_flat[g_idx, :] = 0
    m_flat[g_idx, :] = 0

    for i, pos in enumerate(candidate.new_pattern):
        if i < len(old_values):
            w_flat[g_idx, pos] = old_values[i]
            m_flat[g_idx, pos] = 1

    m_new = restore_groups(m_flat, m_meta)
    w_new = restore_groups(w_flat, w_meta)

    module.sparse_mask.copy_(m_new.clone())
    module.int8_weights.copy_(w_new.clone())

    return True


# =============================================================================
# Main Attack Loop
# =============================================================================

def run_bitmask_swap_attack(
    model: nn.Module,
    test_loader,
    calib_loader,
    device: str,
    seed: int,
    max_logical_swaps: int = 50,
    calib_samples: int = 256,
    eval_samples: int = 2000,
) -> Dict:
    """
    Run the group-based 2:4 metadata attack with cost-2 bitmask swaps.

    Controlled by logical swaps (each costs 2 physical flips).
    """
    rng = random.Random(seed)

    # Initial evaluation
    acc0, loss0 = evaluate_subset(model, test_loader, device=device, max_samples=eval_samples)
    initial_hash = get_bitmask_hash(model)

    # Track by logical swaps
    acc_by_logical = [float(acc0)]
    loss_by_logical = [float(loss0)]
    step_logs: List[AttackStepLog] = []

    wall_t0 = time.time()
    total_counters = {
        "total_groups": 0,
        "valid_groups": 0,
        "candidates_considered": 0,
        "candidates_valid": 0,
        "candidates_skipped_excluded": 0,
        "candidates_skipped_forbidden": 0,
        "validity_checks_passed": 0,
        "validity_checks_failed": 0,
    }

    exclude_groups: Set[Tuple[str, int]] = set()
    forbidden_swaps: Set[Tuple[str, int, int]] = set()

    for swap_idx in range(max_logical_swaps):
        step_t0 = time.time()

        compute_gradients(model, calib_loader, device, calib_samples)

        candidates, counters = enumerate_cost2_swap_candidates(
            model, device,
            exclude_groups=exclude_groups,
            forbidden_swaps=forbidden_swaps
        )

        for key in total_counters:
            if key in counters:
                total_counters[key] += counters[key]

        if not candidates:
            print(f"[Attack] No candidates available at swap {swap_idx}. Stopping.")
            break

        chosen_candidate = max(candidates, key=lambda c: c.score)

        # Apply swap
        apply_swap_change(chosen_candidate)
        new_hash = get_bitmask_hash(model)

        # Track forbidden swaps
        swap_key = (
            chosen_candidate.layer_name,
            int(chosen_candidate.group_idx),
            chosen_candidate.swap_code
        )
        forbidden_swaps.add(swap_key)

        reverse_swap_code = (chosen_candidate.bit_on << 8) | chosen_candidate.bit_off
        reverse_swap_key = (
            chosen_candidate.layer_name,
            int(chosen_candidate.group_idx),
            reverse_swap_code
        )
        forbidden_swaps.add(reverse_swap_key)

        exclude_groups.add((chosen_candidate.layer_name, int(chosen_candidate.group_idx)))

        if len(exclude_groups) > 20:
            exclude_groups = set(list(exclude_groups)[-20:])

        if len(forbidden_swaps) > 1000:
            forbidden_swaps = set(list(forbidden_swaps)[-1000:])

        # Evaluate
        physical_flips_used = 2 * (swap_idx + 1)
        acc, loss = evaluate_subset(model, test_loader, device=device, max_samples=eval_samples)

        acc_by_logical.append(float(acc))
        loss_by_logical.append(float(loss))

        step_time = time.time() - step_t0

        # Validity assertion
        validity_check = "OK"
        if popcount(chosen_candidate.new_mask) != 2:
            validity_check = f"FAIL: popcount={popcount(chosen_candidate.new_mask)}"

        step_log = AttackStepLog(
            swap_idx=swap_idx,
            physical_flips=physical_flips_used,
            layer_name=chosen_candidate.layer_name,
            group_idx=int(chosen_candidate.group_idx),
            old_mask=chosen_candidate.old_mask,
            new_mask=chosen_candidate.new_mask,
            bit_off=chosen_candidate.bit_off,
            bit_on=chosen_candidate.bit_on,
            old_pattern=chosen_candidate.old_pattern,
            new_pattern=chosen_candidate.new_pattern,
            proxy_score=float(chosen_candidate.score),
            accuracy=float(acc),
            loss=float(loss),
            metadata_hash=new_hash,
            validity_check=validity_check,
        )
        step_logs.append(step_log)

        print(f"[Swap {swap_idx:2d} / phy={physical_flips_used:3d}] "
              f"{chosen_candidate.layer_name:30s} g={chosen_candidate.group_idx:6d} "
              f"{chosen_candidate.old_mask}(0b{chosen_candidate.old_mask:04b}) -> "
              f"{chosen_candidate.new_mask}(0b{chosen_candidate.new_mask:04b}) | "
              f"bit{chosen_candidate.bit_off}->0, bit{chosen_candidate.bit_on}->1 | "
              f"{chosen_candidate.old_pattern} -> {chosen_candidate.new_pattern} | "
              f"proxy={chosen_candidate.score:8.4f} | "
              f"acc={acc:6.2f}% loss={loss:8.4f} "
              f"valid={validity_check} t={step_time:.2f}s")

        if new_hash == initial_hash:
            print(f"[Warning] Metadata hash unchanged at swap {swap_idx}!")

    wall_time = time.time() - wall_t0

    return {
        "seed": seed,
        "max_logical_swaps": max_logical_swaps,
        "logical_swaps_executed": len(step_logs),
        "physical_flips_used": 2 * len(step_logs),
        "calib_samples": calib_samples,
        "eval_samples": eval_samples,
        "initial_accuracy": float(acc_by_logical[0]),
        "final_accuracy": float(acc_by_logical[-1]),
        "accuracy_history_logical": acc_by_logical,
        "loss_history_logical": loss_by_logical,
        "step_logs": step_logs,
        "counters": total_counters,
        "initial_metadata_hash": initial_hash,
        "final_metadata_hash": get_bitmask_hash(model),
        "timing": {
            "wall_sec": wall_time,
        },
    }


# =============================================================================
# Load R1_T03 for Comparison
# =============================================================================

def load_r1_t03_results(base_path: str = "results/R1/R1_T03_group_metadata_bitmask_swap_cost2") -> Optional[Dict]:
    """Load R1_T03 results for comparison."""
    csv_path = f"{base_path}_table.csv"
    if not os.path.exists(csv_path):
        return None

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return None
        return rows[0]


# =============================================================================
# Main Script
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="R1_T04: Group-based 2:4 Metadata Attack (Bitmask, 50 Logical Swaps)"
    )
    parser.add_argument("--model", type=str, default="resnet20", choices=["resnet20"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"])
    parser.add_argument("--max-logical-swaps", type=int, default=50,
                        help="Maximum number of logical swaps (default: 50)")
    parser.add_argument("--max-physical-flips", type=int, default=None,
                        help="Maximum physical flips (default: 2*max-logical-swaps)")
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--calib-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint (required)")
    parser.add_argument("--out-prefix", type=str, default=None,
                        help="Output prefix (default: results/R1/R1_T04_bitmask_swaps50)")

    args = parser.parse_args()

    # Determine physical budget
    if args.max_physical_flips is not None:
        physical_budget = args.max_physical_flips
        max_logical_swaps = min(args.max_logical_swaps, physical_budget // 2)
    else:
        max_logical_swaps = args.max_logical_swaps
        physical_budget = 2 * max_logical_swaps

    # Set up output directory
    if args.out_prefix is None:
        args.out_prefix = "results/R1/R1_T04_bitmask_swaps50"
    os.makedirs(os.path.dirname(args.out_prefix) or "results/R1", exist_ok=True)
    os.makedirs("results/R1", exist_ok=True)

    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[Warning] CUDA not available, using CPU")

    # Set seeds
    set_all_seeds(int(args.seed))

    # Load data
    print(f"[Data] Loading CIFAR-10 from {args.data_dir}")
    _, test_loader = load_cifar10_loaders_offline(
        batch_size=256, data_dir=args.data_dir, num_workers=0
    )

    # Load model
    print(f"[Model] Loading sparse ResNet20 from {args.ckpt}")

    checkpoint = torch.load(args.ckpt, map_location="cpu")

    ckpt_to_check = checkpoint
    if "model_state_dict" in checkpoint:
        ckpt_to_check = checkpoint["model_state_dict"]

    is_int8_ckpt = any("int8_weights" in k for k in ckpt_to_check.keys())

    if is_int8_ckpt:
        print("[Model] Detected Int8 checkpoint, loading directly as Int8QuantizedResNet")
        base_model = create_resnet20(sparsity_type="2:4", pretrained_path=None).to(device)
        base_model.eval()

        state_dict_to_load = checkpoint.get("model_state_dict", checkpoint)

        for name, module in base_model.named_modules():
            mask_key = f"{name}.sparse_mask"
            if mask_key in state_dict_to_load:
                mask_tensor = state_dict_to_load[mask_key]
                if hasattr(module, 'register_buffer'):
                    if hasattr(module, 'cached_mask'):
                        del module.cached_mask
                    module.register_buffer('cached_mask', mask_tensor)

        model = Int8QuantizedResNet(base_model, copy_sparse_masks=True).to(device)

        filtered_state_dict = {k: v for k, v in state_dict_to_load.items() if 'sparse_mask' not in k}
        model.load_state_dict(filtered_state_dict, strict=False)

        model.calibrate_all_layers()
        model.eval()
        print(f"[Model] Int8 model loaded and calibrated (top1: {checkpoint.get('top1', 'N/A')}%)")
    else:
        print("[Model] Loading FP32 checkpoint and converting to Int8")
        base_model = create_resnet20(sparsity_type="2:4", pretrained_path=args.ckpt).to(device)
        base_model.eval()
        base_model.freeze_sparse_masks()

        model = Int8QuantizedResNet(base_model, copy_sparse_masks=True).to(device)
        model.calibrate_all_layers()
        model.eval()

    # Load R1_T03 for comparison
    r1_t03_results = load_r1_t03_results()

    print(f"\n[Attack] Starting R1_T04: Bitmask encoding with {max_logical_swaps} logical swaps")
    print(f"[Attack] Physical budget: {physical_budget} flips")
    print(f"[Attack] Constraint: Cost-2 swaps (one 1->0, one 0->1) to maintain popcount=2")

    if r1_t03_results:
        print(f"\n[Comparison] R1_T03 reference:")
        print(f"  Logical swaps: {r1_t03_results.get('logical_swaps', 'N/A')}")
        print(f"  Physical flips: {r1_t03_results.get('physical_budget', 'N/A')}")
        print(f"  Final accuracy: {r1_t03_results.get('final_acc', 'N/A')}%")
        print(f"  Accuracy drop: {r1_t03_results.get('acc_drop', 'N/A')}%")

    result = run_bitmask_swap_attack(
        model=model,
        test_loader=test_loader,
        calib_loader=test_loader,
        device=device,
        seed=int(args.seed),
        max_logical_swaps=max_logical_swaps,
        calib_samples=int(args.calib_samples),
        eval_samples=int(args.eval_samples),
    )

    # Save results
    base_name = args.out_prefix

    # 1. Save pickle
    pkl_path = f"{base_name}_result.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(result, f)
    print(f"\n[Output] Saved {pkl_path}")

    # 2. Generate and save plot (vs logical swaps)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    acc_hist = result["accuracy_history_logical"]
    logical_swaps = list(range(len(acc_hist)))
    physical_flips = [2 * s for s in logical_swaps]

    # Plot vs logical swaps
    color = 'tab:blue'
    ax1.set_xlabel("Logical Swaps (Cost: 2 physical flips per swap)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Top-1 Accuracy (%)", color=color, fontsize=12, fontweight="bold")
    ax1.plot(logical_swaps, acc_hist, marker="o", linewidth=2, markersize=4,
             label="R1_T04: Bitmask 50 Swaps", color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_ylim(0, 100)
    ax1.set_xlim(0, max_logical_swaps)

    # Secondary x-axis for physical flips
    ax2 = ax1.twiny()
    ax2.set_xlabel("Physical Flips", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, physical_budget)
    ax2.tick_params(axis='x')

    plt.title(f"R1_T04: Group-based 2:4 Metadata Attack (Bitmask, {result['logical_swaps_executed']} Swaps, {result['physical_flips_used']} Flips)",
              fontsize=14, fontweight="bold")
    ax1.legend(loc="best")
    fig.tight_layout()
    png_path = f"{base_name}_curve.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Output] Saved {png_path}")

    # 3. Save summary table with comparison
    csv_path = f"{base_name}_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task", "baseline_acc", "final_acc", "acc_drop",
            "logical_swaps", "physical_flips", "runtime_sec",
        ])
        writer.writerow([
            "R1_T04",
            f"{result['initial_accuracy']:.2f}",
            f"{result['final_accuracy']:.2f}",
            f"{result['initial_accuracy'] - result['final_accuracy']:.2f}",
            result['logical_swaps_executed'],
            result['physical_flips_used'],
            f"{result['timing']['wall_sec']:.2f}",
        ])
        if r1_t03_results:
            writer.writerow([
                "R1_T03",
                r1_t03_results.get('baseline_acc', 'N/A'),
                r1_t03_results.get('final_acc', 'N/A'),
                r1_t03_results.get('acc_drop', 'N/A'),
                r1_t03_results.get('logical_swaps', 'N/A'),
                r1_t03_results.get('physical_budget', 'N/A'),
                r1_t03_results.get('runtime_sec', 'N/A'),
            ])
    print(f"[Output] Saved {csv_path}")

    # 4. Save detailed log
    log_path = f"{base_name}_log.txt"
    with open(log_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("R1_T04: Group-based 2:4 Metadata Attack (Bitmask Encoding, 50 Logical Swaps)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Script: run_R1_T04_bitmask_swaps50.py\n")
        f.write(f"Timestamp: {now_ts()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Checkpoint: {os.path.abspath(args.ckpt)}\n")
        f.write(f"\nConstraint:\n")
        f.write(f"  Bitmask encoding: 4-bit mask with popcount=2 for validity\n")
        f.write(f"  Cost-2 swap: flip one 1->0 and one 0->1 (maintains popcount=2)\n")
        f.write(f"  Target logical swaps: {max_logical_swaps}\n")
        f.write(f"  Physical flips used: {result['physical_flips_used']}\n")
        f.write(f"\nConfig:\n")
        f.write(f"  seed: {args.seed}\n")
        f.write(f"  max_logical_swaps: {max_logical_swaps}\n")
        f.write(f"  calib_samples: {args.calib_samples}\n")
        f.write(f"  eval_samples: {args.eval_samples}\n")
        f.write(f"\nResults:\n")
        f.write(f"  baseline_acc: {result['initial_accuracy']:.2f}%\n")
        f.write(f"  final_acc: {result['final_accuracy']:.2f}%\n")
        f.write(f"  acc_drop: {result['initial_accuracy'] - result['final_accuracy']:.2f}%\n")
        f.write(f"  logical_swaps_executed: {result['logical_swaps_executed']}\n")
        f.write(f"  physical_flips_used: {result['physical_flips_used']}\n")
        f.write(f"\nValidity Statistics:\n")
        f.write(f"  validity_checks_passed: {result['counters']['validity_checks_passed']}\n")
        f.write(f"  validity_checks_failed: {result['counters']['validity_checks_failed']}\n")
        f.write(f"\nCounters:\n")
        for key, val in result['counters'].items():
            f.write(f"  {key}: {val}\n")
        f.write(f"\nTiming:\n")
        f.write(f"  wall_sec: {result['timing']['wall_sec']:.2f}\n")
        f.write(f"\nMetadata hashes:\n")
        f.write(f"  initial: {result['initial_metadata_hash']}\n")
        f.write(f"  final: {result['final_metadata_hash']}\n")

        f.write(f"\nComparison with R1_T03:\n")
        f.write("-" * 40 + "\n")
        if r1_t03_results:
            f.write(f"R1_T03 (25 swaps, 50 flips):\n")
            f.write(f"  Baseline: {r1_t03_results.get('baseline_acc', 'N/A')}%\n")
            f.write(f"  Final: {r1_t03_results.get('final_acc', 'N/A')}%\n")
            f.write(f"  Drop: {r1_t03_results.get('acc_drop', 'N/A')}%\n")
            f.write(f"\nR1_T04 (50 swaps, 100 flips):\n")
            f.write(f"  Baseline: {result['initial_accuracy']:.2f}%\n")
            f.write(f"  Final: {result['final_accuracy']:.2f}%\n")
            f.write(f"  Drop: {result['initial_accuracy'] - result['final_accuracy']:.2f}%\n")
        else:
            f.write("R1_T03 results not found for comparison.\n")

        f.write(f"\nStep-by-step details:\n")
        f.write("-" * 80 + "\n")
        for log in result['step_logs']:
            f.write(f"Swap {log.swap_idx} (physical={log.physical_flips}): {log.layer_name} g={log.group_idx} "
                   f"{log.old_mask}(0b{log.old_mask:04b}) -> {log.new_mask}(0b{log.new_mask:04b}) | "
                   f"bit{log.bit_off}->0, bit{log.bit_on}->1 | "
                   f"{log.old_pattern} -> {log.new_pattern} | "
                   f"proxy={log.proxy_score:.4f} | "
                   f"acc={log.accuracy:.2f}% loss={log.loss:.4f} "
                   f"valid={log.validity_check}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("Reproduction command:\n")
        f.write(f"  python run_R1_T04_bitmask_swaps50.py \\\n")
        f.write(f"    --device {device} --seed {args.seed} --max-logical-swaps {max_logical_swaps} \\\n")
        f.write(f"    --calib-samples {args.calib_samples} --eval-samples {args.eval_samples} \\\n")
        f.write(f"    --ckpt {args.ckpt}\n")
    print(f"[Output] Saved {log_path}")

    print(f"\n[Summary] R1_T04 Attack completed:")
    print(f"  Baseline accuracy: {result['initial_accuracy']:.2f}%")
    print(f"  Final accuracy: {result['final_accuracy']:.2f}%")
    print(f"  Accuracy drop: {result['initial_accuracy'] - result['final_accuracy']:.2f}%")
    print(f"  Logical swaps: {result['logical_swaps_executed']}")
    print(f"  Physical flips: {result['physical_flips_used']}")
    print(f"  Wall time: {result['timing']['wall_sec']:.2f}s")

    if r1_t03_results:
        print(f"\n[Comparison with R1_T03]:")
        print(f"  R1_T03 (25 swaps, 50 flips): {r1_t03_results.get('final_acc', 'N/A')}% (drop: {r1_t03_results.get('acc_drop', 'N/A')}%)")
        print(f"  R1_T04 (50 swaps, 100 flips): {result['final_accuracy']:.2f}% (drop: {result['initial_accuracy'] - result['final_accuracy']:.2f}%)")


if __name__ == "__main__":
    main()
