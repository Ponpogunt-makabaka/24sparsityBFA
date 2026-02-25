#!/usr/bin/env python3
"""
R1_T01: Group-based 2:4 Metadata Attack (Index/Position Encoding, Any Pattern)

This is the FIRST attack in the revised R1 workflow (R1_T01) that implements a
group-local greedy search allowing ANY 2-of-4 pattern transition:
- Treats each 2:4 group (4 positions) as one unit
- Enumerates all VALID 2-of-4 patterns (6 combinations) within the group
- Uses group-local score: ΔL_g ≈ ∇_{w̃_g} L · (w̃_g(p') − w̃_g(p))
- Respects non-collision validity (NCA rules)
- Optional topK exact verification

This is NOT a retraining - reuses existing baseline checkpoint.

Naming Convention:
- R1_ = Revised workflow 1
- T01 = Task 01 (index/position encoding with any pattern transition)
- Outputs follow: results/R1/R1_T01_<name>_<type>.ext

Outputs:
- results/R1/R1_T01_group_metadata_index_anypattern_curve.png
- results/R1/R1_T01_group_metadata_index_anypattern_table.csv
- results/R1/R1_T01_group_metadata_index_anypattern_log.txt
- results/R1/R1_T01_group_metadata_index_anypattern_result.pkl
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
# Constants
# =============================================================================

# All 6 valid 2-of-4 patterns (each is a sorted tuple of active positions)
# Pattern representation: (pos0, pos1) where pos0 < pos1
ALL_2OF4_PATTERNS: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
]

# Map pattern to index for quick lookup
PATTERN_TO_IDX = {p: i for i, p in enumerate(ALL_2OF4_PATTERNS)}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass(frozen=True)
class GroupCandidate:
    """A candidate pattern change within a single group."""
    score: float
    layer_name: str
    module: nn.Module
    group_idx: int
    old_pattern: Tuple[int, int]
    new_pattern: Tuple[int, int]
    delta_w_tilde: torch.Tensor  # 4-element dense reconstruction delta


@dataclass
class AttackStepLog:
    """Log entry for a single attack step."""
    step: int
    layer_name: str
    group_idx: int
    old_pattern: Tuple[int, int]
    new_pattern: Tuple[int, int]
    proxy_score: float
    exact_score: Optional[float] = None
    accuracy: float = 0.0
    loss: float = 0.0
    delta_loss: float = 0.0  # Loss change from previous step
    delta_accuracy: float = 0.0  # Accuracy change from previous step
    metadata_hash: str = ""


# =============================================================================
# Core Utilities
# =============================================================================

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_metadata_hash(model: nn.Module) -> str:
    """Compute a hash of all sparse_mask metadata for tracking changes."""
    hasher = hashlib.sha256()
    for name, module in model.named_modules():
        if hasattr(module, "sparse_mask") and module.sparse_mask is not None:
            mask_bytes = module.sparse_mask.cpu().numpy().tobytes()
            hasher.update(mask_bytes)
    return hasher.hexdigest()[:16]


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
                # Ensure mask is binary
                module.sparse_mask = (module.sparse_mask > 0.5).to(module.sparse_mask.dtype)
                layers.append((name, module))
    return layers


def get_current_pattern(mask_group: torch.Tensor) -> Optional[Tuple[int, int]]:
    """Extract the current 2-of-4 pattern from a mask group."""
    active = (mask_group > 0.5).nonzero(as_tuple=False).flatten().tolist()
    if len(active) != 2:
        return None
    a, b = int(active[0]), int(active[1])
    if a > b:
        a, b = b, a
    return (a, b)


def pattern_to_mask(pattern: Tuple[int, int], device: torch.device) -> torch.Tensor:
    """Convert a pattern tuple to a 4-element mask tensor."""
    mask = torch.zeros(4, dtype=torch.float32, device=device)
    mask[list(pattern)] = 1.0
    return mask


def compute_dense_reconstruction(
    int8_w_flat: torch.Tensor,
    mask_flat: torch.Tensor,
    scale: float,
    group_idx: int
) -> torch.Tensor:
    """
    Compute the dense reconstruction w̃_g for a single group.

    w̃_g = int8_w * scale * mask (only non-zero at active positions)
    """
    w_group = int8_w_flat[group_idx]  # (4,) int8 values
    m_group = mask_flat[group_idx]     # (4,) mask values

    # Apply mask and scale to get dense reconstruction
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
    """
    Compute gradients wrt reconstructed dense weights for all layers.
    Returns True if successful.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Collect calibration batch
    calib_data = []
    calib_targets = []
    for inputs, targets in calib_loader:
        calib_data.append(inputs.to(device))
        calib_targets.append(targets.to(device))
        if len(calib_data) * inputs.size(0) >= calib_samples:
            break

    calib_inputs = torch.cat(calib_data, dim=0)[:calib_samples]
    calib_targets = torch.cat(calib_targets, dim=0)[:calib_samples]

    # Forward and backward
    outputs = model(calib_inputs)
    loss = criterion(outputs, calib_targets)
    model.zero_grad()
    loss.backward()

    return True


# =============================================================================
# Candidate Enumeration
# =============================================================================

def enumerate_group_candidates(
    model: nn.Module,
    device: str,
    exclude_groups: Set[Tuple[str, int]] = None,
    forbidden_transitions: Set[Tuple[str, int, Tuple[int, int], Tuple[int, int]]] = None,
    topk_verify: int = 0,
    verify_batch_size: int = 64
) -> Tuple[List[GroupCandidate], Dict[str, int]]:
    """
    Enumerate all valid pattern changes for all groups.

    For each group:
    1. Get current pattern p
    2. For each candidate pattern p' in ALL_2OF4_PATTERNS:
       - Skip if p' == p (no change)
       - Skip if (layer, group, p, p') is in forbidden_transitions
       - Check NCA validity (unique indices)
    3. Compute proxy score: ∇_{w̃_g} L · (w̃_g(p') − w̃_g(p))

    Args:
        model: The sparse model
        device: Device to use
        exclude_groups: Set of (layer_name, group_idx) tuples to exclude from consideration
        forbidden_transitions: Set of (layer, group, old_pattern, new_pattern) to forbid
        topk_verify: Number of top candidates for exact verification (unused here)
        verify_batch_size: Batch size for verification (unused here)

    Returns:
        - List of all valid candidates (one per group)
        - Counters for statistics
    """
    if exclude_groups is None:
        exclude_groups = set()
    if forbidden_transitions is None:
        forbidden_transitions = set()
    counters = {
        "total_groups": 0,
        "valid_groups": 0,
        "candidates_considered": 0,
        "candidates_valid": 0,
        "candidates_invalid_collision": 0,
        "candidates_no_gradient": 0,
    }

    candidates: List[GroupCandidate] = []

    for layer_name, module in get_sparse_layers(model):
        if module.weight.grad is None:
            continue

        grad = module.weight.grad.data
        int8_w = module.int8_weights
        mask = module.sparse_mask
        scale = float(module.scale.item())

        # Flatten to groups
        g_flat, _ = flatten_groups(grad)
        w_flat, w_meta = flatten_groups(int8_w)
        m_flat, m_meta = flatten_groups(mask)

        if g_flat is None or w_flat is None or m_flat is None:
            continue

        num_groups = w_flat.shape[0]
        counters["total_groups"] += num_groups

        # For each group, find the best candidate pattern
        for g_idx in range(num_groups):
            # Skip if this group was recently modified
            if (layer_name, int(g_idx)) in exclude_groups:
                counters["candidates_skipped_excluded"] = counters.get("candidates_skipped_excluded", 0) + 1
                continue

            m_group = m_flat[g_idx]
            current_pattern = get_current_pattern(m_group)

            if current_pattern is None:
                continue

            counters["valid_groups"] += 1

            # Get current dense reconstruction
            w_tilde_current = compute_dense_reconstruction(w_flat, m_flat, scale, g_idx)
            grad_group = g_flat[g_idx]

            best_score_for_group = float('-inf')
            best_candidate_for_group: Optional[GroupCandidate] = None

            # Try all 6 candidate patterns
            for candidate_pattern in ALL_2OF4_PATTERNS:
                if candidate_pattern == current_pattern:
                    continue

                # Skip forbidden transitions (e.g., reversing a recent change)
                transition_key = (layer_name, int(g_idx), current_pattern, candidate_pattern)
                if transition_key in forbidden_transitions:
                    counters["candidates_skipped_forbidden"] = counters.get("candidates_skipped_forbidden", 0) + 1
                    continue

                counters["candidates_considered"] += 1

                # NCA validity check: new pattern must have 2 unique positions
                # (by definition, all patterns in ALL_2OF4_PATTERNS are valid)
                # But we need to ensure the move doesn't create a collision with
                # the current active positions in a way that violates NCA.
                # Since we're doing a complete pattern swap (not a single index move),
                # all 6 patterns are valid by construction.

                # Compute new dense reconstruction
                new_mask = pattern_to_mask(candidate_pattern, device)
                w_tilde_new = w_flat[g_idx].float() * scale * new_mask

                # Compute delta_w_tilde
                delta_w_tilde = w_tilde_new - w_tilde_current

                # Proxy score: gradient dot delta
                proxy_score = float(torch.dot(grad_group, delta_w_tilde).item())

                counters["candidates_valid"] += 1

                # Keep the best candidate for this group
                if proxy_score > best_score_for_group:
                    best_score_for_group = proxy_score
                    best_candidate_for_group = GroupCandidate(
                        score=proxy_score,
                        layer_name=layer_name,
                        module=module,
                        group_idx=int(g_idx),
                        old_pattern=current_pattern,
                        new_pattern=candidate_pattern,
                        delta_w_tilde=delta_w_tilde.cpu(),
                    )

            if best_candidate_for_group is not None:
                candidates.append(best_candidate_for_group)

        # Clear gradient
        if module.weight.grad is not None:
            module.weight.grad = None

    return candidates, counters


def select_topk_candidates(
    candidates: List[GroupCandidate],
    k: int
) -> List[GroupCandidate]:
    """Select top K candidates by proxy score."""
    sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
    return sorted_candidates[:k]


def exact_verify_candidates(
    model: nn.Module,
    candidates: List[GroupCandidate],
    verify_inputs: torch.Tensor,
    verify_targets: torch.Tensor,
    device: str
) -> List[Tuple[GroupCandidate, float]]:
    """
    Perform exact verification on top-K candidates using a small validation batch.

    For each candidate:
    1. Temporarily apply the pattern change
    2. Compute actual loss on verification batch
    3. Revert the change
    4. Return candidates ranked by actual loss increase

    Returns:
        List of (candidate, exact_score) tuples sorted by exact_score
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Get baseline loss
    with torch.no_grad():
        baseline_outputs = model(verify_inputs)
        baseline_loss = criterion(baseline_outputs, verify_targets).item()

    results: List[Tuple[GroupCandidate, float]] = []

    for candidate in candidates:
        # Apply the change
        apply_pattern_change(candidate)
        model.eval()

        # Compute new loss
        with torch.no_grad():
            new_outputs = model(verify_inputs)
            new_loss = criterion(new_outputs, verify_targets).item()

        # Exact score is loss increase (we want to maximize loss)
        exact_score = new_loss - baseline_loss

        # Revert the change
        revert_pattern_change(candidate)

        results.append((candidate, exact_score))

    # Sort by exact score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# =============================================================================
# Pattern Application
# =============================================================================

def apply_pattern_change(candidate: GroupCandidate) -> bool:
    """Apply a pattern change to the model."""
    module = candidate.module
    g_idx = candidate.group_idx
    new_pattern = candidate.new_pattern

    # Get mask and weights
    mask = module.sparse_mask
    int8_w = module.int8_weights

    m_flat, m_meta = flatten_groups(mask)
    w_flat, w_meta = flatten_groups(int8_w)

    if m_flat is None or w_flat is None:
        return False

    # Create new mask
    device = mask.device
    new_mask = pattern_to_mask(new_pattern, device)

    # Store old active values
    old_mask = m_flat[g_idx].clone()
    old_active_indices = (old_mask > 0.5).nonzero().flatten()

    # Move values from old active positions to new active positions
    # We need to preserve the actual weight values when moving positions
    new_mask_local = torch.zeros(4, dtype=mask.dtype, device=device)
    new_mask_local[list(new_pattern)] = 1

    # Get the weight values at old active positions
    old_values = w_flat[g_idx, old_active_indices].clone()

    # Zero out all positions in this group
    w_flat[g_idx, :] = 0
    m_flat[g_idx, :] = 0

    # Set new active positions with preserved values
    # We need to map old values to new positions
    # For simplicity, we keep values in order and move them
    for i, pos in enumerate(new_pattern):
        if i < len(old_values):
            w_flat[g_idx, pos] = old_values[i]
            m_flat[g_idx, pos] = 1

    # Restore shapes
    m_new = restore_groups(m_flat, m_meta)
    w_new = restore_groups(w_flat, w_meta)

    module.sparse_mask.copy_(m_new.clone())
    module.int8_weights.copy_(w_new.clone())

    return True


def revert_pattern_change(candidate: GroupCandidate) -> bool:
    """Revert a pattern change by restoring the old pattern."""
    module = candidate.module
    g_idx = candidate.group_idx
    old_pattern = candidate.old_pattern

    # Get mask and weights
    mask = module.sparse_mask
    int8_w = module.int8_weights

    m_flat, m_meta = flatten_groups(mask)
    w_flat, w_meta = flatten_groups(int8_w)

    if m_flat is None or w_flat is None:
        return False

    device = mask.device

    # Get current (new) active values to preserve
    current_mask = m_flat[g_idx].clone()
    current_active_indices = (current_mask > 0.5).nonzero().flatten()
    current_values = w_flat[g_idx, current_active_indices].clone()

    # Zero out all positions
    w_flat[g_idx, :] = 0
    m_flat[g_idx, :] = 0

    # Restore old pattern with preserved values
    for i, pos in enumerate(old_pattern):
        if i < len(current_values):
            w_flat[g_idx, pos] = current_values[i]
            m_flat[g_idx, pos] = 1

    # Restore shapes
    m_new = restore_groups(m_flat, m_meta)
    w_new = restore_groups(w_flat, w_meta)

    module.sparse_mask.copy_(m_new.clone())
    module.int8_weights.copy_(w_new.clone())

    return True


# =============================================================================
# Main Attack Loop
# =============================================================================

def run_group_metadata_attack(
    model: nn.Module,
    test_loader,
    calib_loader,
    device: str,
    seed: int,
    max_flips: int = 50,
    calib_samples: int = 256,
    eval_samples: int = 2000,
    topk_verify: int = 64,
) -> Dict:
    """
    Run the group-based 2:4 metadata attack.

    The attack iteratively selects the best pattern change across all groups
    and applies it to maximize loss increase.
    """
    rng = random.Random(seed)

    # Initial evaluation
    acc0, loss0 = evaluate_subset(model, test_loader, device=device, max_samples=eval_samples)
    initial_hash = get_metadata_hash(model)

    acc_history = [float(acc0)]
    loss_history = [float(loss0)]
    step_logs: List[AttackStepLog] = []
    topk_verify_logs: List[Dict] = []

    # Track loss/accuracy changes
    acc_increase_steps: List[int] = []
    loss_decrease_steps: List[int] = []

    prev_acc = float(acc0)
    prev_loss = float(loss0)

    # Prepare verification batch for exact verification
    verify_inputs, verify_targets = None, None
    if topk_verify > 0:
        for inputs, targets in test_loader:
            if verify_inputs is None:
                verify_inputs = inputs[:min(topk_verify * 2, inputs.size(0))].to(device)
                verify_targets = targets[:min(topk_verify * 2, targets.size(0))].to(device)
            else:
                verify_inputs = torch.cat([verify_inputs, inputs[:32].to(device)], dim=0)[:topk_verify * 2]
                verify_targets = torch.cat([verify_targets, targets[:32].to(device)], dim=0)[:topk_verify * 2]
            if verify_inputs.size(0) >= topk_verify:
                break

    wall_t0 = time.time()
    total_counters = {
        "total_groups": 0,
        "valid_groups": 0,
        "candidates_considered": 0,
        "candidates_valid": 0,
        "candidates_invalid_collision": 0,
        "candidates_skipped_excluded": 0,
        "candidates_skipped_forbidden": 0,
    }

    # Track recently modified groups to prevent immediate reversal
    exclude_groups: Set[Tuple[str, int]] = set()
    # Track forbidden transitions to prevent flipping back to a previous state
    forbidden_transitions: Set[Tuple[str, int, Tuple[int, int], Tuple[int, int]]] = set()

    for step in range(1, max_flips + 1):
        step_t0 = time.time()

        # Compute gradients
        compute_gradients(model, calib_loader, device, calib_samples)

        # Enumerate all group candidates (excluding recently modified groups and forbidden transitions)
        candidates, counters = enumerate_group_candidates(
            model, device,
            exclude_groups=exclude_groups,
            forbidden_transitions=forbidden_transitions
        )

        # Accumulate counters
        for key in total_counters:
            if key in counters:
                total_counters[key] += counters[key]

        if not candidates:
            print(f"[Attack] No candidates available at step {step}. Stopping.")
            break

        # Select global best by proxy score
        best_by_proxy = max(candidates, key=lambda c: c.score)

        # Optional: TopK exact verification
        chosen_candidate = best_by_proxy
        exact_score = None

        if topk_verify > 0:
            topk = select_topk_candidates(candidates, min(topk_verify, len(candidates)))
            verified = exact_verify_candidates(model, topk, verify_inputs, verify_targets, device)

            if verified:
                chosen_candidate, exact_score = verified[0]
                topk_verify_logs.append({
                    "step": step,
                    "topk_size": len(topk),
                    "proxy_score": float(best_by_proxy.score),
                    "exact_score": float(exact_score),
                    "proxy_exact_match": chosen_candidate == best_by_proxy,
                })

        # Apply the chosen change
        apply_pattern_change(chosen_candidate)
        new_hash = get_metadata_hash(model)

        # Record the transition to forbid reversing it later
        transition_key = (
            chosen_candidate.layer_name,
            int(chosen_candidate.group_idx),
            chosen_candidate.old_pattern,
            chosen_candidate.new_pattern
        )
        forbidden_transitions.add(transition_key)

        # Also add the reverse transition to prevent flipping back
        reverse_transition = (
            chosen_candidate.layer_name,
            int(chosen_candidate.group_idx),
            chosen_candidate.new_pattern,
            chosen_candidate.old_pattern
        )
        forbidden_transitions.add(reverse_transition)

        # Update exclude_groups to prevent immediate re-modification
        # Add the current group to exclude set for next few iterations
        exclude_groups.add((chosen_candidate.layer_name, int(chosen_candidate.group_idx)))

        # Keep only recent groups (e.g., last 20) to allow eventual re-modification
        if len(exclude_groups) > 20:
            # Remove oldest entries (FIFO)
            exclude_groups = set(list(exclude_groups)[-(20):])

        # Also clean up old forbidden transitions to prevent unbounded growth
        # Keep only recent 1000 transitions
        if len(forbidden_transitions) > 1000:
            forbidden_transitions = set(list(forbidden_transitions)[-1000:])

        # Evaluate
        acc, loss = evaluate_subset(model, test_loader, device=device, max_samples=eval_samples)
        acc_history.append(float(acc))
        loss_history.append(float(loss))

        # Compute deltas
        delta_loss = float(loss) - prev_loss
        delta_acc = float(acc) - prev_acc

        # Track anomalies
        if delta_acc > 0:
            acc_increase_steps.append(step)
        if delta_loss < 0:
            loss_decrease_steps.append(step)

        step_time = time.time() - step_t0

        # Log this step
        step_log = AttackStepLog(
            step=step,
            layer_name=chosen_candidate.layer_name,
            group_idx=int(chosen_candidate.group_idx),
            old_pattern=chosen_candidate.old_pattern,
            new_pattern=chosen_candidate.new_pattern,
            proxy_score=float(chosen_candidate.score),
            exact_score=exact_score,
            accuracy=float(acc),
            loss=float(loss),
            delta_loss=delta_loss,
            delta_accuracy=delta_acc,
            metadata_hash=new_hash,
        )
        step_logs.append(step_log)

        # Update previous values
        prev_acc = float(acc)
        prev_loss = float(loss)

        print(f"[Step {step:3d}] {chosen_candidate.layer_name:30s} g={chosen_candidate.group_idx:6d} "
              f"{chosen_candidate.old_pattern} -> {chosen_candidate.new_pattern} | "
              f"proxy={chosen_candidate.score:8.4f}"
              + (f" exact={exact_score:8.4f}" if exact_score is not None else "")
              + f" | acc={acc:6.2f}% ({delta_acc:+.2f}%) loss={loss:8.4f} ({delta_loss:+.6f}) hash={new_hash} t={step_time:.2f}s")

        # Sanity check: metadata should change
        if new_hash == initial_hash:
            print(f"[Warning] Metadata hash unchanged at step {step}!")

    wall_time = time.time() - wall_t0

    return {
        "seed": seed,
        "max_flips": max_flips,
        "calib_samples": calib_samples,
        "eval_samples": eval_samples,
        "topk_verify": topk_verify,
        "initial_accuracy": float(acc_history[0]),
        "final_accuracy": float(acc_history[-1]),
        "initial_loss": float(loss_history[0]),
        "final_loss": float(loss_history[-1]),
        "total_flips": len(step_logs),
        "accuracy_history": acc_history,
        "loss_history": loss_history,
        "step_logs": step_logs,
        "topk_verify_logs": topk_verify_logs,
        "acc_increase_steps": acc_increase_steps,
        "loss_decrease_steps": loss_decrease_steps,
        "counters": total_counters,
        "initial_metadata_hash": initial_hash,
        "final_metadata_hash": get_metadata_hash(model),
        "timing": {
            "wall_sec": wall_time,
        },
    }


# =============================================================================
# Main Script
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="R1_T01: Group-based 2:4 Metadata Attack (Index/Position Encoding, Any Pattern)"
    )
    parser.add_argument("--model", type=str, default="resnet20", choices=["resnet20"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"])
    parser.add_argument("--encoding", type=str, default="position", choices=["position"])
    parser.add_argument("--max-flips", type=int, default=50)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--calib-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk-verify", type=int, default=64,
                        help="Top-K candidates for exact verification (0 to disable)")
    parser.add_argument("--verify-batch-size", type=int, default=64,
                        help="Batch size for exact verification")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint (required, no default)")
    parser.add_argument("--out-prefix", type=str, default=None,
                        help="Output prefix (default: results/R1/R1_T01_group_metadata_index_anypattern)")

    args = parser.parse_args()

    # Set up output directory
    if args.out_prefix is None:
        args.out_prefix = "results/R1/R1_T01_group_metadata_index_anypattern"
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

    # Check if checkpoint is already an Int8QuantizedResNet model
    checkpoint = torch.load(args.ckpt, map_location="cpu")

    # Detect checkpoint type by checking for Int8QuantizedResNet-specific keys
    # Handle nested checkpoint structure (e.g., {'model_state_dict': {...}})
    ckpt_to_check = checkpoint
    if "model_state_dict" in checkpoint:
        ckpt_to_check = checkpoint["model_state_dict"]

    is_int8_ckpt = any("int8_weights" in k for k in ckpt_to_check.keys())

    if is_int8_ckpt:
        print("[Model] Detected Int8 checkpoint, loading directly as Int8QuantizedResNet")
        # Create base model structure
        base_model = create_resnet20(sparsity_type="2:4", pretrained_path=None).to(device)
        base_model.eval()

        # First, manually load sparse_mask into base_model layers
        # This ensures Int8QuantizedResNet can pick them up via copy_sparse_masks
        state_dict_to_load = checkpoint.get("model_state_dict", checkpoint)

        # Extract sparse_mask tensors and register them in base_model
        for name, module in base_model.named_modules():
            # Find corresponding sparse_mask key in checkpoint
            mask_key = f"{name}.sparse_mask"
            if mask_key in state_dict_to_load:
                mask_tensor = state_dict_to_load[mask_key]
                # Register as cached_mask for Int8QuantizedResNet to pick up
                if hasattr(module, 'register_buffer'):
                    # Remove old cached_mask if exists
                    if hasattr(module, 'cached_mask'):
                        del module.cached_mask
                    module.register_buffer('cached_mask', mask_tensor)
                    print(f"[Model] Registered sparse_mask for {name}")

        # Now create Int8 model with copy_sparse_masks=True
        model = Int8QuantizedResNet(base_model, copy_sparse_masks=True).to(device)

        # Load the rest of the Int8 checkpoint (excluding sparse_mask which is already loaded)
        # Filter out sparse_mask keys to avoid "unexpected key" warnings
        filtered_state_dict = {k: v for k, v in state_dict_to_load.items() if 'sparse_mask' not in k}
        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        if missing:
            print(f"[Model] Warning: Missing keys during load: {missing[:5]}...")
        if unexpected:
            print(f"[Model] Warning: Unexpected keys during load: {unexpected[:5]}...")

        # CRITICAL FIX: Must call calibrate_all_layers() to set quantized=True
        # Without this, Int8QuantizedConv2d.forward() will skip get_dequantized_weights()
        # and sparse_mask will have NO effect on the forward pass
        model.calibrate_all_layers()
        model.eval()
        print(f"[Model] Int8 model loaded and calibrated (top1: {checkpoint.get('top1', 'N/A')}%)")
    else:
        print("[Model] Loading FP32 checkpoint and converting to Int8")
        base_model = create_resnet20(
            sparsity_type="2:4",
            pretrained_path=args.ckpt
        ).to(device)
        base_model.eval()
        base_model.freeze_sparse_masks()

        model = Int8QuantizedResNet(base_model, copy_sparse_masks=True).to(device)
        model.calibrate_all_layers()
        model.eval()

    print(f"[Model] Loaded. Baseline accuracy check...")

    # Run attack
    print(f"[Attack] Starting group-based 2:4 metadata attack")
    print(f"[Attack] Config: max_flips={args.max_flips}, calib_samples={args.calib_samples}, "
          f"eval_samples={args.eval_samples}, topk_verify={args.topk_verify}")

    result = run_group_metadata_attack(
        model=model,
        test_loader=test_loader,
        calib_loader=test_loader,
        device=device,
        seed=int(args.seed),
        max_flips=int(args.max_flips),
        calib_samples=int(args.calib_samples),
        eval_samples=int(args.eval_samples),
        topk_verify=int(args.topk_verify),
    )

    # Save results
    base_name = args.out_prefix

    # 1. Save pickle
    pkl_path = f"{base_name}_result.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[Output] Saved {pkl_path}")

    # 2. Generate and save plot
    plt.figure(figsize=(10, 6))
    acc_hist = result["accuracy_history"]
    plt.plot(range(len(acc_hist)), acc_hist, marker="o", linewidth=2, markersize=4,
             label="Group-based Metadata Attack")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlabel("Flip Step", fontsize=12, fontweight="bold")
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title(f"Task 1.xx: Group-based 2:4 Metadata Attack ({args.encoding} encoding)", fontsize=14, fontweight="bold")
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.legend(loc="best")
    plt.tight_layout()
    png_path = f"{base_name}_curve.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Output] Saved {png_path}")

    # 3. Save summary table
    csv_path = f"{base_name}_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "baseline_acc", "final_acc", "acc_drop",
            "baseline_loss", "final_loss", "loss_increase",
            "max_flips", "encoding",
            "eval_samples", "calib_samples", "seed", "topk_verify_K",
            "num_groups_total", "num_groups_valid",
            "candidates_total", "candidates_valid",
            "n_acc_increase_steps", "n_loss_decrease_steps",
            "acc_increase_steps", "loss_decrease_steps",
        ])
        writer.writerow([
            f"{result['initial_accuracy']:.2f}%",
            f"{result['final_accuracy']:.2f}%",
            f"{result['initial_accuracy'] - result['final_accuracy']:.2f}%",
            f"{result['initial_loss']:.6f}",
            f"{result['final_loss']:.6f}",
            f"{result['final_loss'] - result['initial_loss']:.6f}",
            result['max_flips'],
            args.encoding,
            result['eval_samples'],
            result['calib_samples'],
            result['seed'],
            result['topk_verify'],
            result['counters']['total_groups'],
            result['counters']['valid_groups'],
            result['counters']['candidates_considered'],
            result['counters']['candidates_valid'],
            len(result.get('acc_increase_steps', [])),
            len(result.get('loss_decrease_steps', [])),
            str(result.get('acc_increase_steps', [])),
            str(result.get('loss_decrease_steps', [])),
        ])
    print(f"[Output] Saved {csv_path}")

    # 4. Save topK verification log (if applicable)
    if result['topk_verify_logs']:
        topk_csv_path = f"{base_name}_topk_verify.csv"
        with open(topk_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "topk_size", "proxy_score", "exact_score", "proxy_exact_match"])
            writer.writeheader()
            for row in result['topk_verify_logs']:
                writer.writerow(row)
        print(f"[Output] Saved {topk_csv_path}")

    # 5. Save detailed log
    log_path = f"{base_name}_log.txt"
    with open(log_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("R1_T01: Group-based 2:4 Metadata Attack (Position Encoding, Any Pattern)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Script: run_R1_T01_group_metadata_index_anypattern.py\n")
        f.write(f"Timestamp: {now_ts()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Encoding: {args.encoding}\n")
        f.write(f"Checkpoint: {os.path.abspath(args.ckpt)}\n")
        f.write(f"\nConfig:\n")
        f.write(f"  seed: {args.seed}\n")
        f.write(f"  max_flips: {args.max_flips}\n")
        f.write(f"  calib_samples: {args.calib_samples}\n")
        f.write(f"  eval_samples: {args.eval_samples}\n")
        f.write(f"  topk_verify: {args.topk_verify}\n")
        f.write(f"\nResults:\n")
        f.write(f"  baseline_acc: {result['initial_accuracy']:.2f}%\n")
        f.write(f"  final_acc: {result['final_accuracy']:.2f}%\n")
        f.write(f"  accuracy_drop: {result['initial_accuracy'] - result['final_accuracy']:.2f}%\n")
        f.write(f"  baseline_loss: {result['initial_loss']:.6f}\n")
        f.write(f"  final_loss: {result['final_loss']:.6f}\n")
        f.write(f"  loss_increase: {result['final_loss'] - result['initial_loss']:.6f}\n")
        f.write(f"  total_flips: {result['total_flips']}\n")
        f.write(f"  n_acc_increase_steps: {len(result.get('acc_increase_steps', []))}\n")
        f.write(f"  n_loss_decrease_steps: {len(result.get('loss_decrease_steps', []))}\n")
        f.write(f"  acc_increase_steps: {result.get('acc_increase_steps', [])}\n")
        f.write(f"  loss_decrease_steps: {result.get('loss_decrease_steps', [])}\n")
        f.write(f"\nCounters:\n")
        for key, val in result['counters'].items():
            f.write(f"  {key}: {val}\n")
        f.write(f"\nTiming:\n")
        f.write(f"  wall_sec: {result['timing']['wall_sec']:.2f}\n")
        f.write(f"\nMetadata hashes:\n")
        f.write(f"  initial: {result['initial_metadata_hash']}\n")
        f.write(f"  final: {result['final_metadata_hash']}\n")
        f.write(f"\nStep-by-step details:\n")
        f.write("-" * 80 + "\n")
        for log in result['step_logs']:
            delta_acc_str = f" ({log.delta_accuracy:+.2f}%)" if hasattr(log, 'delta_accuracy') else ""
            delta_loss_str = f" ({log.delta_loss:+.6f})" if hasattr(log, 'delta_loss') else ""
            f.write(f"Step {log.step}: {log.layer_name} g={log.group_idx} "
                   f"{log.old_pattern} -> {log.new_pattern} | "
                   f"proxy={log.proxy_score:.4f}")
            if log.exact_score is not None:
                f.write(f" exact={log.exact_score:.4f}")
            f.write(f" | acc={log.accuracy:.2f}%{delta_acc_str} loss={log.loss:.4f}{delta_loss_str} hash={log.metadata_hash}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("Reproduction command:\n")
        f.write(f"  python run_R1_T01_group_metadata_index_anypattern.py \\\n")
        f.write(f"    --device {device} --seed {args.seed} --max-flips {args.max_flips} \\\n")
        f.write(f"    --calib-samples {args.calib_samples} --eval-samples {args.eval_samples} \\\n")
        f.write(f"    --topk-verify {args.topk_verify} --ckpt {args.ckpt}\n")
    print(f"[Output] Saved {log_path}")

    print(f"\n[Summary] Attack completed:")
    print(f"  Baseline accuracy: {result['initial_accuracy']:.2f}%")
    print(f"  Final accuracy: {result['final_accuracy']:.2f}%")
    print(f"  Accuracy drop: {result['initial_accuracy'] - result['final_accuracy']:.2f}%")
    print(f"  Total flips: {result['total_flips']}")
    print(f"  Wall time: {result['timing']['wall_sec']:.2f}s")


if __name__ == "__main__":
    main()
