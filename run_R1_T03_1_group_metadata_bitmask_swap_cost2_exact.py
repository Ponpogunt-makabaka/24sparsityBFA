#!/usr/bin/env python3
"""
R1_T03.1: Group-based 2:4 Metadata Attack (Bitmask Encoding, Cost-2 Swap, Exact Verification)

This is the EXACT VERIFICATION upgrade of R1_T03.

Key differences from R1_T03:
- Uses Top-K Exact Verification instead of pure proxy score selection
- Stage 1: Proxy Screening - enumerates all candidates and computes proxy scores
- Stage 2: Exact Verification - for top-K candidates, performs real forward pass
- Reverts model state after verification before applying final flip
- More reliable loss increase at the cost of additional computation

Bitmask encoding:
- Direct 4-bit mask where each bit corresponds to a position in the group
- Valid mask: popcount(mask) == 2 (exactly two active positions)
- Cost-2 swap: flip one 1->0 and one 0->1

This is NOT a retraining - reuses existing baseline checkpoint.

Naming Convention:
- R1_ = Revised workflow 1
- T03.1 = Task 03.1 (bitmask encoding with cost-2 swap + exact verification)
- Outputs follow: results/R1/R1_T03_1_<name>_<type>.ext

Outputs:
- results/R1/R1_T03_1_group_metadata_bitmask_swap_cost2_exact_curve.png
- results/R1/R1_T03_1_group_metadata_bitmask_swap_cost2_exact_table.csv
- results/R1/R1_T03_1_group_metadata_bitmask_swap_cost2_exact_log.txt
- results/R1/R1_T03_1_group_metadata_bitmask_swap_cost2_exact_result.pkl
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
# Constants - Bitmask Encoding
# =============================================================================

def pattern_to_bitmask(pattern: Tuple[int, int]) -> int:
    """
    Convert a 2-of-4 pattern tuple to a 4-bit bitmask.

    Args:
        pattern: Tuple of two distinct positions (i, j) where i < j

    Returns:
        4-bit bitmask (0-15) with bits i and j set
    """
    mask = 0
    mask |= (1 << pattern[0])
    mask |= (1 << pattern[1])
    return mask


def bitmask_to_pattern(mask: int) -> Optional[Tuple[int, int]]:
    """
    Convert a 4-bit bitmask to a 2-of-4 pattern tuple.

    Returns None if popcount != 2 (invalid).

    Args:
        mask: 4-bit bitmask (0-15)

    Returns:
        Tuple of two positions (i, j) sorted, or None if invalid
    """
    if mask < 0 or mask > 15:
        return None

    # Extract positions of set bits
    positions = [i for i in range(4) if (mask >> i) & 1]

    if len(positions) != 2:
        return None  # Invalid: popcount != 2

    return (positions[0], positions[1])


def enumerate_cost2_swaps(current_mask: int) -> List[Tuple[int, int, int, int]]:
    """
    Enumerate all valid cost-2 swaps from current mask.

    A cost-2 swap flips exactly two bits:
    - One 1 -> 0 (turn off an active position)
    - One 0 -> 1 (turn on an inactive position)

    This yields exactly 4 candidates for any valid 2-of-4 mask.

    Args:
        current_mask: Current 4-bit mask (must have popcount=2)

    Returns:
        List of tuples: (new_mask, bit_off, bit_on, swap_code)
        - bit_off: which bit position was turned off (0-3)
        - bit_on: which bit position was turned on (0-3)
        - swap_code: encoded as (bit_off << 8) | bit_on for tracking
    """
    pattern = bitmask_to_pattern(current_mask)
    if pattern is None:
        return []

    pos0, pos1 = pattern  # Two active positions

    # Inactive positions
    inactive = [i for i in range(4) if i not in (pos0, pos1)]

    swaps = []
    for active_pos in (pos0, pos1):
        for inactive_pos in inactive:
            # Create new mask by flipping active_pos off and inactive_pos on
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


# All 6 valid 2-of-4 patterns for reference
ALL_2OF4_PATTERNS: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
]

# =============================================================================
# Data Structures
# =============================================================================

@dataclass(frozen=True)
class SwapCandidate:
    """A candidate swap change within a single group."""
    proxy_score: float
    layer_name: str
    module: nn.Module
    group_idx: int
    old_mask: int
    new_mask: int
    bit_off: int  # Which bit was turned off (0-3)
    bit_on: int   # Which bit was turned on (0-3)
    swap_code: int  # Encoded swap for tracking
    old_pattern: Tuple[int, int]
    new_pattern: Tuple[int, int]
    delta_w_tilde: torch.Tensor  # 4-element dense reconstruction delta


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
    exact_loss: float  # Exact verified loss
    exact_delta_loss: float  # Exact delta from baseline
    topk_considered: int  # How many candidates were verified
    verification_time: float  # Time spent on exact verification
    accuracy: float = 0.0
    loss: float = 0.0
    metadata_hash: str = ""
    candidate_total: int = 0
    candidate_evaluated: int = 0
    candidate_rejected: int = 0


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
                # Ensure mask is binary
                module.sparse_mask = (module.sparse_mask > 0.5).to(module.sparse_mask.dtype)
                layers.append((name, module))
    return layers


def get_current_bitmask(mask_group: torch.Tensor) -> Optional[int]:
    """
    Extract the current 4-bit bitmask from a mask group.

    Returns None if popcount != 2.
    """
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


def compute_exact_loss(
    model: nn.Module,
    calib_loader,
    device: str,
    calib_samples: int
) -> float:
    """
    Compute exact loss on calibration subset.
    Used for exact verification of candidates.
    """
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

    with torch.no_grad():
        outputs = model(calib_inputs)
        loss = criterion(outputs, calib_targets)

    return float(loss.item())


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
    2. Enumerate all 4 cost-2 swaps (one 1->0, one 0->1)
    3. Skip forbidden swaps
    4. Compute proxy score: ∇_{w̃_g} L · (w̃_g(m') − w̃_g(m))
    5. Keep the best candidate for this group

    Args:
        model: The sparse model
        device: Device to use
        exclude_groups: Set of (layer_name, group_idx) to exclude
        forbidden_swaps: Set of (layer, group, swap_code) to forbid

    Returns:
        - List of best valid candidates (one per group)
        - Counters for statistics
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
    }

    candidates: List[SwapCandidate] = []

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

        # For each group, find the best cost-2 swap
        for g_idx in range(num_groups):
            # Skip if this group was recently modified
            if (layer_name, int(g_idx)) in exclude_groups:
                counters["candidates_skipped_excluded"] += 1
                continue

            m_group = m_flat[g_idx]
            current_mask = get_current_bitmask(m_group)

            if current_mask is None:
                continue

            counters["valid_groups"] += 1

            # Get current pattern
            current_pattern = bitmask_to_pattern(current_mask)
            if current_pattern is None:
                continue

            # Get current dense reconstruction
            w_tilde_current = compute_dense_reconstruction(w_flat, m_flat, scale, g_idx)
            grad_group = g_flat[g_idx]

            # Enumerate all 4 cost-2 swaps
            swaps = enumerate_cost2_swaps(current_mask)

            best_score_for_group = float('-inf')
            best_candidate_for_group: Optional[SwapCandidate] = None

            for new_mask, bit_off, bit_on, swap_code in swaps:
                # Check if this swap is forbidden
                swap_key = (layer_name, int(g_idx), swap_code)
                if swap_key in forbidden_swaps:
                    counters["candidates_skipped_forbidden"] += 1
                    continue

                counters["candidates_considered"] += 1

                # Get new pattern
                new_pattern = bitmask_to_pattern(new_mask)
                if new_pattern is None:
                    continue

                # Compute new dense reconstruction
                new_mask_tensor = bitmask_to_mask_tensor(new_mask, device)
                w_tilde_new = w_flat[g_idx].float() * scale * new_mask_tensor

                # Compute delta_w_tilde
                delta_w_tilde = w_tilde_new - w_tilde_current

                # Proxy score: gradient dot delta
                proxy_score = float(torch.dot(grad_group, delta_w_tilde).item())

                counters["candidates_valid"] += 1

                # Keep the best candidate for this group
                if proxy_score > best_score_for_group:
                    best_score_for_group = proxy_score
                    best_candidate_for_group = SwapCandidate(
                        proxy_score=proxy_score,
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

        # Clear gradient
        if module.weight.grad is not None:
            module.weight.grad = None

    return candidates, counters


def apply_swap_change(candidate: SwapCandidate) -> bool:
    """Apply a swap change to the model."""
    # Defensive validity checks for Stage B application.
    if bitmask_to_pattern(candidate.old_mask) is None:
        return False
    if bitmask_to_pattern(candidate.new_mask) is None:
        return False
    if candidate.new_mask.bit_count() != 2:
        return False
    if candidate.bit_off == candidate.bit_on:
        return False
    if ((candidate.old_mask >> candidate.bit_off) & 1) != 1:
        return False
    if ((candidate.old_mask >> candidate.bit_on) & 1) != 0:
        return False

    module = candidate.module
    g_idx = candidate.group_idx
    new_mask = candidate.new_mask

    # Get mask and weights
    mask = module.sparse_mask
    int8_w = module.int8_weights

    m_flat, m_meta = flatten_groups(mask)
    w_flat, w_meta = flatten_groups(int8_w)

    if m_flat is None or w_flat is None:
        return False

    device = mask.device

    # Get the weight values at old active positions to preserve
    old_pattern = candidate.old_pattern
    old_mask_tensor = m_flat[g_idx].clone()
    old_active_indices = (old_mask_tensor > 0.5).nonzero().flatten()
    old_values = w_flat[g_idx, old_active_indices].clone()

    # Zero out all positions in this group
    w_flat[g_idx, :] = 0
    m_flat[g_idx, :] = 0

    # Set new active positions with preserved values
    for i, pos in enumerate(candidate.new_pattern):
        if i < len(old_values):
            w_flat[g_idx, pos] = old_values[i]
            m_flat[g_idx, pos] = 1

    # Restore shapes
    m_new = restore_groups(m_flat, m_meta)
    w_new = restore_groups(w_flat, w_meta)

    module.sparse_mask.copy_(m_new.clone())
    module.int8_weights.copy_(w_new.clone())

    return True


def save_model_state(model: nn.Module, candidate: SwapCandidate) -> Tuple[torch.Tensor, torch.Tensor]:
    """Save the current state of a group's mask and weights."""
    module = candidate.module
    g_idx = candidate.group_idx

    m_flat, m_meta = flatten_groups(module.sparse_mask)
    w_flat, w_meta = flatten_groups(module.int8_weights)

    # Save the specific group
    saved_mask = m_flat[g_idx].clone()
    saved_weights = w_flat[g_idx].clone()

    return saved_mask, saved_weights


def restore_model_state(
    model: nn.Module,
    candidate: SwapCandidate,
    saved_mask: torch.Tensor,
    saved_weights: torch.Tensor
) -> None:
    """Restore the state of a group's mask and weights."""
    module = candidate.module
    g_idx = candidate.group_idx

    m_flat, m_meta = flatten_groups(module.sparse_mask)
    w_flat, w_meta = flatten_groups(module.int8_weights)

    m_flat[g_idx].copy_(saved_mask)
    w_flat[g_idx].copy_(saved_weights)

    # Restore shapes
    m_new = restore_groups(m_flat, m_meta)
    w_new = restore_groups(w_flat, w_meta)

    module.sparse_mask.copy_(m_new.clone())
    module.int8_weights.copy_(w_new.clone())


# =============================================================================
# Top-K Exact Verification
# =============================================================================

def exact_verify_topk_candidates(
    candidates: List[SwapCandidate],
    model: nn.Module,
    calib_loader,
    device: str,
    calib_samples: int,
    baseline_loss: float,
    topk: int = 64,
) -> Tuple[Optional[SwapCandidate], float, int]:
    """
    Perform exact verification on top-K candidates.

    Stage 1: Proxy Screening - already done, candidates are pre-sorted
    Stage 2: Exact Verification - compute real loss for each candidate

    For each candidate:
    1. Save current model state
    2. Apply the candidate swap
    3. Compute exact loss on calib subset
    4. Revert to original state
    5. Track the candidate with highest exact loss increase

    Args:
        candidates: List of candidates sorted by proxy score
        model: The model
        calib_loader: Calibration data loader
        device: Device
        calib_samples: Number of samples for exact loss computation
        baseline_loss: Current loss before any swap
        topk: Number of top candidates to verify

    Returns:
        - Best candidate by exact loss
        - Best exact loss value
        - Number of candidates actually verified
    """
    if not candidates:
        return None, baseline_loss, 0

    # Take top-K candidates
    topk_candidates = candidates[:min(topk, len(candidates))]

    best_exact_loss = baseline_loss
    best_candidate: Optional[SwapCandidate] = None

    for candidate in topk_candidates:
        # Save current state
        saved_mask, saved_weights = save_model_state(model, candidate)

        # Apply candidate; skip if application failed.
        if not apply_swap_change(candidate):
            restore_model_state(model, candidate, saved_mask, saved_weights)
            continue

        # Always revert, even if exact evaluation raises.
        try:
            exact_loss = compute_exact_loss(model, calib_loader, device, calib_samples)
        finally:
            restore_model_state(model, candidate, saved_mask, saved_weights)

        # Check if this is the best
        if exact_loss > best_exact_loss:
            best_exact_loss = exact_loss
            best_candidate = candidate

    return best_candidate, best_exact_loss, len(topk_candidates)


# =============================================================================
# Main Attack Loop
# =============================================================================

def run_bitmask_swap_attack_exact(
    model: nn.Module,
    test_loader,
    calib_loader,
    device: str,
    seed: int,
    physical_budget: int = 50,
    calib_samples: int = 256,
    eval_samples: int = 2000,
    topk: int = 64,
) -> Dict:
    """
    Run the group-based 2:4 metadata attack with cost-2 bitmask swaps
    and Top-K Exact Verification.

    Physical budget = total bit flips allowed.
    Each swap consumes cost=2 (flip one 1->0 and one 0->1).
    Number of logical swaps = floor(physical_budget / 2).
    """
    rng = random.Random(seed)

    # Initial evaluation
    acc0, loss0 = evaluate_subset(model, test_loader, device=device, max_samples=eval_samples)
    initial_hash = get_bitmask_hash(model)

    # Compute baseline loss on calibration subset for exact verification
    baseline_calib_loss = compute_exact_loss(model, calib_loader, device, calib_samples)

    # Track accuracy by physical flips (interleaved for odd positions)
    max_swaps = physical_budget // 2
    acc_by_physical = [float(acc0)]  # Index = physical flips used
    loss_by_physical = [float(loss0)]
    step_logs: List[AttackStepLog] = []

    wall_t0 = time.time()
    total_counters = {
        "total_groups": 0,
        "valid_groups": 0,
        "candidates_considered": 0,
        "candidates_valid": 0,
        "candidates_skipped_excluded": 0,
        "candidates_skipped_forbidden": 0,
    }

    # Track recently modified groups to prevent immediate reversal
    exclude_groups: Set[Tuple[str, int]] = set()
    # Track forbidden swaps to prevent flipping back to a previous state
    forbidden_swaps: Set[Tuple[str, int, int]] = set()

    prev_calib_loss = baseline_calib_loss

    for swap_idx in range(max_swaps):
        step_t0 = time.time()

        # Compute gradients
        compute_gradients(model, calib_loader, device, calib_samples)

        # Enumerate all group candidates (cost-2 swaps only)
        candidates, counters = enumerate_cost2_swap_candidates(
            model, device,
            exclude_groups=exclude_groups,
            forbidden_swaps=forbidden_swaps
        )

        # Accumulate counters
        for key in total_counters:
            if key in counters:
                total_counters[key] += counters[key]

        if not candidates:
            print(f"[Attack] No candidates available at swap {swap_idx}. Stopping.")
            break

        # Sort by proxy score for Top-K screening
        candidates.sort(key=lambda c: c.proxy_score, reverse=True)

        # Stage 2: Exact Verification on top-K candidates
        verify_t0 = time.time()
        chosen_candidate, exact_loss, topk_verified = exact_verify_topk_candidates(
            candidates=candidates,
            model=model,
            calib_loader=calib_loader,
            device=device,
            calib_samples=calib_samples,
            baseline_loss=prev_calib_loss,
            topk=topk,
        )
        verification_time = time.time() - verify_t0

        if chosen_candidate is None:
            print(f"[Attack] No valid candidate after exact verification at swap {swap_idx}. Stopping.")
            break

        exact_delta_loss = exact_loss - prev_calib_loss

        # Apply the chosen swap (for real this time)
        if not apply_swap_change(chosen_candidate):
            print(f"[Attack] Failed to apply chosen swap at swap {swap_idx}. Stopping.")
            break
        new_hash = get_bitmask_hash(model)

        # Record the swap to forbid reversing it later
        swap_key = (
            chosen_candidate.layer_name,
            int(chosen_candidate.group_idx),
            chosen_candidate.swap_code
        )
        forbidden_swaps.add(swap_key)

        # Also add the reverse swap to prevent flipping back
        reverse_swap_code = (chosen_candidate.bit_on << 8) | chosen_candidate.bit_off
        reverse_swap_key = (
            chosen_candidate.layer_name,
            int(chosen_candidate.group_idx),
            reverse_swap_code
        )
        forbidden_swaps.add(reverse_swap_key)

        # Update exclude_groups to prevent immediate re-modification
        exclude_groups.add((chosen_candidate.layer_name, int(chosen_candidate.group_idx)))

        # Keep only recent groups (e.g., last 20)
        if len(exclude_groups) > 20:
            exclude_groups = set(list(exclude_groups)[-20:])

        # Clean up old forbidden swaps
        if len(forbidden_swaps) > 1000:
            forbidden_swaps = set(list(forbidden_swaps)[-1000:])

        # Evaluate
        physical_flips_used = 2 * (swap_idx + 1)
        acc, loss = evaluate_subset(model, test_loader, device=device, max_samples=eval_samples)

        # Compute new calib loss for next iteration
        new_calib_loss = compute_exact_loss(model, calib_loader, device, calib_samples)

        # Add to physical-flips-indexed arrays
        # For odd physical flips, interpolate from previous even
        if len(acc_by_physical) < physical_flips_used:
            while len(acc_by_physical) < physical_flips_used:
                acc_by_physical.append(acc_by_physical[-1])
                loss_by_physical.append(loss_by_physical[-1])

        acc_by_physical.append(float(acc))
        loss_by_physical.append(float(loss))

        step_time = time.time() - step_t0

        # Log this step
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
            proxy_score=float(chosen_candidate.proxy_score),
            exact_loss=exact_loss,
            exact_delta_loss=exact_delta_loss,
            topk_considered=topk_verified,
            verification_time=verification_time,
            accuracy=float(acc),
            loss=float(loss),
            metadata_hash=new_hash,
            candidate_total=4,  # Always 4 for cost-2 swaps
            candidate_evaluated=counters["candidates_valid"],
            candidate_rejected=counters["candidates_skipped_forbidden"],
        )
        step_logs.append(step_log)

        prev_calib_loss = new_calib_loss

        print(f"[Swap {swap_idx:2d} / phy={physical_flips_used:2d}] "
              f"{chosen_candidate.layer_name:30s} g={chosen_candidate.group_idx:6d} "
              f"{chosen_candidate.old_mask}(0b{chosen_candidate.old_mask:04b}) -> "
              f"{chosen_candidate.new_mask}(0b{chosen_candidate.new_mask:04b}) | "
              f"bit{chosen_candidate.bit_off}->0, bit{chosen_candidate.bit_on}->1 | "
              f"{chosen_candidate.old_pattern} -> {chosen_candidate.new_pattern} | "
              f"proxy={chosen_candidate.proxy_score:8.4f} "
              f"exact_L={exact_loss:.4f} (Δ{exact_delta_loss:+.4f}) | "
              f"acc={acc:6.2f}% loss={loss:8.4f} | "
              f"topk={topk_verified}/{len(candidates)} "
              f"verify_t={verification_time:.2f}s "
              f"t={step_time:.2f}s")

        # Sanity check: metadata should change
        if new_hash == initial_hash:
            print(f"[Warning] Metadata hash unchanged at swap {swap_idx}!")

    # Fill remaining physical budget with last value
    while len(acc_by_physical) <= physical_budget:
        acc_by_physical.append(acc_by_physical[-1])
        loss_by_physical.append(loss_by_physical[-1])

    wall_time = time.time() - wall_t0

    return {
        "seed": seed,
        "physical_budget": physical_budget,
        "logical_swaps": len(step_logs),
        "calib_samples": calib_samples,
        "eval_samples": eval_samples,
        "topk": topk,
        "initial_accuracy": float(acc_by_physical[0]),
        "final_accuracy": float(acc_by_physical[physical_budget]),
        "accuracy_history_physical": acc_by_physical,
        "loss_history_physical": loss_by_physical,
        "step_logs": step_logs,
        "counters": total_counters,
        "initial_metadata_hash": initial_hash,
        "final_metadata_hash": get_bitmask_hash(model),
        "timing": {
            "wall_sec": wall_time,
        },
    }


# =============================================================================
# Main Script
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="R1_T03.1: Group-based 2:4 Metadata Attack (Bitmask Encoding, Cost-2 Swap, Exact Verification)"
    )
    parser.add_argument("--model", type=str, default="resnet20", choices=["resnet20"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"])
    parser.add_argument("--physical-budget", type=int, default=50,
                        help="Physical bit flip budget (default: 50, each swap costs 2)")
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--calib-samples", type=int, default=256)
    parser.add_argument("--topk", type=int, default=64,
                        help="Number of top candidates to verify exactly (default: 64)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint (required)")
    parser.add_argument("--out-prefix", type=str, default=None,
                        help="Output prefix (default: results/R1/R1_T03_1_group_metadata_bitmask_swap_cost2_exact)")

    args = parser.parse_args()

    # Set up output directory
    if args.out_prefix is None:
        args.out_prefix = "results/R1/R1_T03_1_group_metadata_bitmask_swap_cost2_exact"
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
    ckpt_to_check = checkpoint
    if "model_state_dict" in checkpoint:
        ckpt_to_check = checkpoint["model_state_dict"]

    is_int8_ckpt = any("int8_weights" in k for k in ckpt_to_check.keys())

    if is_int8_ckpt:
        print("[Model] Detected Int8 checkpoint, loading directly as Int8QuantizedResNet")
        base_model = create_resnet20(sparsity_type="2:4", pretrained_path=None).to(device)
        base_model.eval()

        state_dict_to_load = checkpoint.get("model_state_dict", checkpoint)

        # Extract sparse_mask tensors and register them in base_model
        for name, module in base_model.named_modules():
            mask_key = f"{name}.sparse_mask"
            if mask_key in state_dict_to_load:
                mask_tensor = state_dict_to_load[mask_key]
                if hasattr(module, 'register_buffer'):
                    if hasattr(module, 'cached_mask'):
                        del module.cached_mask
                    module.register_buffer('cached_mask', mask_tensor)
                    print(f"[Model] Registered sparse_mask for {name}")

        model = Int8QuantizedResNet(base_model, copy_sparse_masks=True).to(device)

        filtered_state_dict = {k: v for k, v in state_dict_to_load.items() if 'sparse_mask' not in k}
        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        if missing:
            print(f"[Model] Warning: Missing keys during load: {missing[:5]}...")
        if unexpected:
            print(f"[Model] Warning: Unexpected keys during load: {unexpected[:5]}...")

        # CRITICAL FIX: Must call calibrate_all_layers() to set quantized=True
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
    print(f"[Attack] Starting R1_T03.1: Bitmask encoding with cost-2 swap attack + Exact Verification")
    print(f"[Attack] Config: physical_budget={args.physical_budget}, calib_samples={args.calib_samples}, "
          f"eval_samples={args.eval_samples}, topk={args.topk}")
    print(f"[Attack] Constraint: Cost-2 swaps (one 1->0, one 0->1) to maintain popcount=2")
    print(f"[Attack] Method: Top-K Exact Verification")

    result = run_bitmask_swap_attack_exact(
        model=model,
        test_loader=test_loader,
        calib_loader=test_loader,
        device=device,
        seed=int(args.seed),
        physical_budget=int(args.physical_budget),
        calib_samples=int(args.calib_samples),
        eval_samples=int(args.eval_samples),
        topk=int(args.topk),
    )

    # Save results
    base_name = args.out_prefix

    # 1. Save pickle
    pkl_path = f"{base_name}_result.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[Output] Saved {pkl_path}")

    # 2. Generate and save plot (vs physical flips)
    plt.figure(figsize=(10, 6))
    acc_hist = result["accuracy_history_physical"]
    physical_flips = list(range(len(acc_hist)))
    plt.plot(physical_flips, acc_hist, marker="o", linewidth=2, markersize=4,
             label="R1_T03.1: Bitmask Cost-2 Swap + Exact Verification")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlabel("Physical Flips (Cost: 2 per swap)", fontsize=12, fontweight="bold")
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title(f"R1_T03.1: Group-based 2:4 Metadata Attack (Bitmask, Cost-2 Swap, Exact Verification, {result['logical_swaps']} swaps)",
              fontsize=14, fontweight="bold")
    plt.ylim(0, 100)
    plt.xlim(0, args.physical_budget)
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
            "physical_budget", "logical_swaps", "topk", "runtime_sec",
            "num_groups_total", "num_groups_valid",
            "candidates_considered", "candidates_valid",
        ])
        writer.writerow([
            f"{result['initial_accuracy']:.2f}",
            f"{result['final_accuracy']:.2f}",
            f"{result['initial_accuracy'] - result['final_accuracy']:.2f}",
            result['physical_budget'],
            result['logical_swaps'],
            result['topk'],
            f"{result['timing']['wall_sec']:.2f}",
            result['counters']['total_groups'],
            result['counters']['valid_groups'],
            result['counters']['candidates_considered'],
            result['counters']['candidates_valid'],
        ])
    print(f"[Output] Saved {csv_path}")

    # 4. Save detailed log
    log_path = f"{base_name}_log.txt"
    with open(log_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("R1_T03.1: Group-based 2:4 Metadata Attack (Bitmask Encoding, Cost-2 Swap, Exact Verification)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Script: run_R1_T03_1_group_metadata_bitmask_swap_cost2_exact.py\n")
        f.write(f"Timestamp: {now_ts()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Checkpoint: {os.path.abspath(args.ckpt)}\n")
        f.write(f"\nConstraint:\n")
        f.write(f"  Bitmask encoding: 4-bit mask with popcount=2 for validity\n")
        f.write(f"  Cost-2 swap: flip one 1->0 and one 0->1 (maintains popcount=2)\n")
        f.write(f"  Physical budget: {args.physical_budget} flips\n")
        f.write(f"  Logical swaps: {result['logical_swaps']} (each costs 2 physical flips)\n")
        f.write(f"\nMethod:\n")
        f.write(f"  Top-K Exact Verification: K={args.topk}\n")
        f.write(f"  Stage 1: Proxy Screening (first-order Taylor expansion)\n")
        f.write(f"  Stage 2: Exact Verification (real forward pass on calib subset)\n")
        f.write(f"  Revert: Model state restored after each verification\n")
        f.write(f"\nConfig:\n")
        f.write(f"  seed: {args.seed}\n")
        f.write(f"  physical_budget: {args.physical_budget}\n")
        f.write(f"  calib_samples: {args.calib_samples}\n")
        f.write(f"  eval_samples: {args.eval_samples}\n")
        f.write(f"  topk: {args.topk}\n")
        f.write(f"\nResults:\n")
        f.write(f"  baseline_acc: {result['initial_accuracy']:.2f}%\n")
        f.write(f"  final_acc: {result['final_accuracy']:.2f}%\n")
        f.write(f"  acc_drop: {result['initial_accuracy'] - result['final_accuracy']:.2f}%\n")
        f.write(f"  logical_swaps: {result['logical_swaps']}\n")
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
            f.write(f"Swap {log.swap_idx} (physical={log.physical_flips}): {log.layer_name} g={log.group_idx} "
                   f"{log.old_mask}(0b{log.old_mask:04b}) -> {log.new_mask}(0b{log.new_mask:04b}) | "
                   f"bit{log.bit_off}->0, bit{log.bit_on}->1 | "
                   f"{log.old_pattern} -> {log.new_pattern} | "
                   f"proxy={log.proxy_score:.4f} "
                   f"exact_L={log.exact_loss:.4f} (Δ{log.exact_delta_loss:+.4f}) | "
                   f"acc={log.accuracy:.2f}% loss={log.loss:.4f} | "
                   f"topk={log.topk_considered} verify_t={log.verification_time:.2f}s\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("Reproduction command:\n")
        f.write(f"  python run_R1_T03_1_group_metadata_bitmask_swap_cost2_exact.py \\\n")
        f.write(f"    --device {device} --seed {args.seed} --physical-budget {args.physical_budget} \\\n")
        f.write(f"    --calib-samples {args.calib_samples} --eval-samples {args.eval_samples} \\\n")
        f.write(f"    --topk {args.topk} \\\n")
        f.write(f"    --ckpt {args.ckpt}\n")
    print(f"[Output] Saved {log_path}")

    print(f"\n[Summary] R1_T03.1 Attack completed:")
    print(f"  Baseline accuracy: {result['initial_accuracy']:.2f}%")
    print(f"  Final accuracy: {result['final_accuracy']:.2f}%")
    print(f"  Accuracy drop: {result['initial_accuracy'] - result['final_accuracy']:.2f}%")
    print(f"  Physical budget: {args.physical_budget}")
    print(f"  Logical swaps: {result['logical_swaps']}")
    print(f"  Wall time: {result['timing']['wall_sec']:.2f}s")


if __name__ == "__main__":
    main()
