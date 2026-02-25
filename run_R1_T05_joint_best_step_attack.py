#!/usr/bin/env python3
"""
R1_T05: Joint Best-Step Attack (Weight-Bit + Index Metadata + Bitmask Metadata)

This is the FIFTH attack in the revised R1 workflow (R1_T05) that implements a
UNIFIED attack combining three action types:

1. Weight-bit flip (cost=1): Flip bits in INT8 weights
   - Reuses logic from bfa/int8_attack.py
   - Targets non-zero weights in sparse 2:4 model

2. Index metadata 1-bit move (cost=1): Hamming-1 in 4-bit code space
   - Reuses logic from R1_T02
   - Encodes active positions as two 2-bit indices

3. Bitmask metadata swap (cost=2): Cost-2 swap between valid masks
   - Reuses logic from R1_T03/R1_T04
   - Flips one 1->0 and one 0->1

Key difference from R1_T01-T04:
- TWO-STAGE SELECTION:
  - Stage A: Fast proxy scoring (first-order) to get top-K candidates
  - Stage B: Exact forward loss verification on top-K, then choose argmax(ΔL_exact / cost)
- Unified action selection across all three types
- Cost-aware ranking: ΔL_exact / cost

This is NOT a retraining - reuses existing baseline checkpoint.

Naming Convention:
- R1_ = Revised workflow 1
- T05 = Task 05 (joint best-step attack with exact verification)
- Outputs follow: results/R1/R1_T05_<name>_<type>.ext

Outputs:
- results/R1/R1_T05_joint_best_step_attack_curve_physical.png
- results/R1/R1_T05_joint_best_step_attack_curve_logical.png
- results/R1/R1_T05_joint_best_step_attack_table.csv
- results/R1/R1_T05_joint_best_step_attack_log.txt
- results/R1/R1_T05_joint_best_step_attack_result.pkl
- results/R1/R1_T05_joint_action_breakdown.png (optional)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import pickle
import random
import time
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Literal

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
# Constants - Index/Position Encoding (from R1_T02)
# =============================================================================

def encode_index_to_4bit(i: int, j: int) -> int:
    """Encode two 2-bit indices into a 4-bit code."""
    return (j << 2) | i


def decode_4bit_to_index(code: int) -> Tuple[int, int]:
    """Decode a 4-bit code into two 2-bit indices."""
    i = code & 0x3
    j = (code >> 2) & 0x3
    return (i, j)


def pattern_to_code(pattern: Tuple[int, int]) -> int:
    """Convert a 2-of-4 pattern tuple to a 4-bit code."""
    return encode_index_to_4bit(pattern[0], pattern[1])


def code_to_pattern(code: int) -> Optional[Tuple[int, int]]:
    """Convert a 4-bit code to a 2-of-4 pattern tuple."""
    i, j = decode_4bit_to_index(code)
    if i == j:
        return None  # Collision
    return (min(i, j), max(i, j))


# =============================================================================
# Constants - Bitmask Encoding (from R1_T03)
# =============================================================================

def pattern_to_bitmask(pattern: Tuple[int, int]) -> int:
    """Convert a 2-of-4 pattern tuple to a 4-bit bitmask."""
    mask = 0
    mask |= (1 << pattern[0])
    mask |= (1 << pattern[1])
    return mask


def bitmask_to_pattern(mask: int) -> Optional[Tuple[int, int]]:
    """Convert a 4-bit bitmask to a 2-of-4 pattern tuple."""
    if mask < 0 or mask > 15:
        return None
    positions = [i for i in range(4) if (mask >> i) & 1]
    if len(positions) != 2:
        return None
    return (positions[0], positions[1])


def enumerate_cost2_swaps(current_mask: int) -> List[Tuple[int, int, int, int]]:
    """Enumerate all valid cost-2 swaps from current mask."""
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


# =============================================================================
# Data Structures
# =============================================================================

@dataclass(frozen=True)
class UnifiedCandidate:
    """Immutable candidate action for unified attack."""
    # Common fields
    action_type: Literal["weight_bit", "index_1bit", "bitmask_swap"]
    score_proxy: float  # First-order proxy score
    score_proxy_norm: float  # score_proxy / cost
    layer_name: str
    cost: int  # 1 for weight_bit/index_1bit, 2 for bitmask_swap

    # Weight-bit flip specific
    weight_idx: Optional[int] = None
    bit_pos: Optional[int] = None  # 0-7 for INT8
    int8_value: Optional[int] = None  # Current INT8 value

    # Index 1-bit specific
    group_idx: Optional[int] = None
    old_code: Optional[int] = None  # 4-bit code
    new_code: Optional[int] = None
    flipped_bit: Optional[int] = None  # 0-3

    # Bitmask swap specific
    old_mask: Optional[int] = None
    new_mask: Optional[int] = None
    bit_off: Optional[int] = None
    bit_on: Optional[int] = None

    # For exact verification (computed in stage B)
    exact_delta_loss: Optional[float] = None
    exact_delta_loss_norm: Optional[float] = None  # exact_delta_loss / cost


@dataclass
class AttackStepLog:
    """Log entry for a single attack step."""
    step: int
    physical_flips_used: int
    action_type: str
    cost: int
    layer_name: str
    proxy_score_norm: float
    exact_delta: float
    exact_delta_norm: float
    accuracy: float = 0.0
    loss: float = 0.0
    description: str = ""
    candidates_total_by_type: Dict[str, int] = None
    candidates_topk_by_type: Dict[str, int] = None

    def __post_init__(self):
        if self.candidates_total_by_type is None:
            self.candidates_total_by_type = {}
        if self.candidates_topk_by_type is None:
            self.candidates_topk_by_type = {}


# =============================================================================
# Core Utilities (reused from R1_T02/T03)
# =============================================================================

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_metadata_hash(model: nn.Module) -> str:
    """Compute a hash of all sparse_mask metadata."""
    hasher = hashlib.sha256()
    for name, module in model.named_modules():
        if hasattr(module, "sparse_mask") and module.sparse_mask is not None:
            mask_bytes = module.sparse_mask.cpu().numpy().tobytes()
            hasher.update(mask_bytes)
    return hasher.hexdigest()[:16]


def get_int8_weights_hash(model: nn.Module) -> str:
    """Compute a hash of all int8 weights."""
    hasher = hashlib.sha256()
    for name, module in model.named_modules():
        if hasattr(module, "int8_weights"):
            weight_bytes = module.int8_weights.cpu().numpy().tobytes()
            hasher.update(weight_bytes)
    return hasher.hexdigest()[:16]


def flatten_groups(tensor: torch.Tensor, group_size: int = 4) -> Tuple[Optional[torch.Tensor], Optional[Tuple]]:
    """
    Flatten tensor into groups of size `group_size` for 2:4 sparsity.

    For Conv2d (4D): groups follow T02.1 semantics.
        (out_ch, in_ch, kh, kw) -> permute(0, 2, 3, 1) -> view(-1, 4)

    For Linear (2D):
        (out_ch, in_ch) -> view(-1, 4)

    Returns:
        (flat, meta) where `meta` is used by `restore_groups`.
        Returns (None, None) if unsupported or not divisible.
    """
    if tensor.dim() == 4:
        t_perm = tensor.permute(0, 2, 3, 1).contiguous()
        numel = t_perm.numel()
        if numel % group_size != 0:
            return None, None
        flat = t_perm.view(-1, group_size)
        meta = ("conv", tuple(tensor.shape), tuple(t_perm.shape))
        return flat, meta
    if tensor.dim() == 2:
        numel = tensor.numel()
        if numel % group_size != 0:
            return None, None
        flat = tensor.contiguous().view(-1, group_size)
        meta = ("linear", tuple(tensor.shape))
        return flat, meta
    return None, None


def restore_groups(flat: torch.Tensor, meta: Tuple) -> torch.Tensor:
    """
    Restore tensor from grouped representation produced by `flatten_groups`.
    """
    kind = meta[0]
    if kind == "conv":
        _, _, perm_shape = meta
        t_perm = flat.view(perm_shape)
        return t_perm.permute(0, 3, 1, 2).contiguous()
    _, original_shape = meta
    return flat.view(original_shape)


def get_sparse_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Get all layers with sparse masks."""
    layers = []
    for name, module in model.named_modules():
        if hasattr(module, "int8_weights") and hasattr(module, "scale") and hasattr(module, "sparse_mask"):
            if module.sparse_mask is not None:
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


def pattern_to_mask_tensor(pattern: Tuple[int, int], device: torch.device) -> torch.Tensor:
    """Convert a pattern tuple to a 4-element mask tensor."""
    mask = torch.zeros(4, dtype=torch.float32, device=device)
    mask[list(pattern)] = 1.0
    return mask


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
# INT8 Bit Flip Helper (from bfa/int8_attack.py)
# =============================================================================

def flip_int8_value(int8_val: int, bit_pos: int) -> int:
    """Flip a bit in a signed int8 value using proper 8-bit two's complement."""
    u8 = int8_val & 0xFF
    u8 ^= (1 << bit_pos)
    return u8 - 256 if u8 >= 128 else u8


def compute_int8_delta(int8_val: int, bit_pos: int, scale: float) -> float:
    """Compute the change in dequantized value when flipping a bit."""
    flipped_val = flip_int8_value(int8_val, bit_pos)
    original_fp32 = int8_val * scale
    flipped_fp32 = flipped_val * scale
    return flipped_fp32 - original_fp32


# =============================================================================
# Gradient Computation (reused from R1_T02/T03)
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
# Candidate Generation - Three Types
# =============================================================================

def enumerate_weight_bit_candidates(
    model: nn.Module,
    device: str,
    flipped_bits: Set[Tuple[str, int, int]],
    max_candidates_per_layer: int = 1000,
) -> Tuple[List[UnifiedCandidate], Dict[str, int]]:
    """
    Enumerate weight-bit flip candidates (cost=1).

    Sampling policy for tractability on CPU:
    - Only consider non-zero weights (sparse 2:4 model)
    - Only consider bits with non-zero gradient
    - For each layer, keep top-K by |grad * delta_value|

    Reuses sensitivity computation logic from bfa/int8_attack.py.
    """
    counters = {
        "total_layers": 0,
        "total_weights": 0,
        "nonzero_weights": 0,
        "candidates_considered": 0,
        "candidates_valid": 0,
        "candidates_skipped_flipped": 0,
    }

    candidates: List[UnifiedCandidate] = []

    for layer_name, module in get_sparse_layers(model):
        if module.weight.grad is None:
            continue

        counters["total_layers"] += 1
        grad = module.weight.grad.data
        int8_w = module.int8_weights
        mask = module.sparse_mask
        scale = float(module.scale.item())

        # Flatten for iteration
        int8_flat = int8_w.flatten()
        grad_flat = grad.flatten()
        mask_flat = mask.flatten() if mask is not None else None

        layer_candidates: List[Tuple[float, int, int]] = []  # (score, idx, bit)

        for idx in range(int8_flat.numel()):
            counters["total_weights"] += 1

            # Skip zero weights (sparse model)
            if mask_flat is not None and mask_flat[idx].item() < 0.5:
                continue

            int8_val = int8_flat[idx].item()
            if int8_val == 0:
                continue

            counters["nonzero_weights"] += 1
            grad_val = grad_flat[idx].item()

            # Skip if gradient is zero
            if abs(grad_val) < 1e-10:
                continue

            # For each bit position
            for bit_pos in range(8):
                bit_key = (layer_name, int(idx), bit_pos)
                if bit_key in flipped_bits:
                    counters["candidates_skipped_flipped"] += 1
                    continue

                counters["candidates_considered"] += 1

                # Compute proxy score
                delta_value = compute_int8_delta(int8_val, bit_pos, scale)
                score = grad_val * delta_value

                # Keep only flips that increase loss
                if score > 0:
                    counters["candidates_valid"] += 1
                    layer_candidates.append((score, idx, bit_pos))

        # Keep top candidates per layer
        layer_candidates.sort(key=lambda x: x[0], reverse=True)
        for score, idx, bit_pos in layer_candidates[:max_candidates_per_layer]:
            candidates.append(UnifiedCandidate(
                action_type="weight_bit",
                score_proxy=score,
                score_proxy_norm=score / 1.0,
                layer_name=layer_name,
                cost=1,
                weight_idx=int(idx),
                bit_pos=bit_pos,
                int8_value=int(int8_flat[idx].item()),
            ))

    return candidates, counters


def enumerate_index_1bit_candidates(
    model: nn.Module,
    device: str,
    exclude_groups: Set[Tuple[str, int]],
    forbidden_transitions: Set[Tuple[str, int, int, int]],
    top_m_per_group: int = 3,
) -> Tuple[List[UnifiedCandidate], Dict[str, int]]:
    """
    Enumerate index metadata 1-bit move candidates (cost=1).

    Upgraded with R1_T08 improvements:
    1) Proxy/Apply semantic alignment: w_new construction mirrors actual apply semantics
    2) Top-M retention per group (configurable via top_m_per_group)
    3) 1-bit reachable constraint (Index encoding)

    For each current pattern, flip each bit (0-3) once to generate candidates.
    A candidate is valid only if the decoded indices don't collide (i != j).
    """
    counters = {
        "total_groups": 0,
        "valid_groups": 0,
        "candidates_considered": 0,
        "candidates_valid": 0,
        "candidates_rejected_collision": 0,
        "candidates_rejected_no_change": 0,
        "candidates_skipped_excluded": 0,
        "candidates_skipped_forbidden": 0,
        "candidates_no_gradient": 0,
        "candidates_kept_topm": 0,
    }

    candidates: List[UnifiedCandidate] = []

    for layer_name, module in get_sparse_layers(model):
        if module.weight.grad is None:
            counters["candidates_no_gradient"] += 1
            continue

        grad = module.weight.grad.data
        int8_w = module.int8_weights
        mask = module.sparse_mask
        scale = float(module.scale.item())

        # Flatten to groups
        g_flat, _ = flatten_groups(grad)
        w_flat, _ = flatten_groups(int8_w)
        m_flat, _ = flatten_groups(mask)

        if g_flat is None or w_flat is None or m_flat is None:
            continue

        num_groups = w_flat.shape[0]
        counters["total_groups"] += num_groups

        for g_idx in range(num_groups):
            if (layer_name, int(g_idx)) in exclude_groups:
                counters["candidates_skipped_excluded"] += 1
                continue

            m_group = m_flat[g_idx]
            current_pattern = get_current_pattern(m_group)

            if current_pattern is None:
                continue

            counters["valid_groups"] += 1
            current_code = pattern_to_code(current_pattern)

            grad_group = g_flat[g_idx].float()
            w_group = w_flat[g_idx]

            # Current dense reconstruction (sparse-gated).
            old_mask = (m_group > 0.5).to(torch.float32)
            with torch.no_grad():
                w_tilde_current = w_group.float() * scale * old_mask

            # Must mirror apply op: values are read in current active-index order.
            old_active_indices = (old_mask > 0.5).nonzero(as_tuple=False).flatten()
            if old_active_indices.numel() != 2:
                continue
            old_values = w_group[old_active_indices].clone()

            group_candidates: List[UnifiedCandidate] = []

            # Enumerate 1-bit reachable transitions: flip each bit (0-3)
            for bit_pos in range(4):
                candidate_code = current_code ^ (1 << bit_pos)
                transition_key = (layer_name, int(g_idx), current_code, candidate_code)
                if transition_key in forbidden_transitions:
                    counters["candidates_skipped_forbidden"] += 1
                    continue

                counters["candidates_considered"] += 1

                candidate_pattern = code_to_pattern(candidate_code)
                if candidate_pattern is None:
                    # Collision: decoded indices are the same (i == j)
                    counters["candidates_rejected_collision"] += 1
                    continue
                if candidate_pattern == current_pattern:
                    counters["candidates_rejected_no_change"] += 1
                    continue

                # Proxy/Apply alignment:
                # DO NOT use `w_group * new_mask` directly (that keeps values at old positions).
                # Instead, emulate real apply semantics: zero group then move old active values.
                with torch.no_grad():
                    w_new_group = torch.zeros_like(w_group)
                    m_new_group = torch.zeros_like(old_mask)
                    for rank, dst_pos in enumerate(candidate_pattern):
                        w_new_group[dst_pos] = old_values[rank]
                        m_new_group[dst_pos] = 1.0

                    w_tilde_new = w_new_group.float() * scale * m_new_group
                    delta_w_tilde = w_tilde_new - w_tilde_current
                    proxy_score = float(torch.dot(grad_group, delta_w_tilde).item())

                if proxy_score > 0:
                    counters["candidates_valid"] += 1
                    group_candidates.append(
                        UnifiedCandidate(
                            action_type="index_1bit",
                            score_proxy=proxy_score,
                            score_proxy_norm=proxy_score / 1.0,
                            layer_name=layer_name,
                            cost=1,
                            group_idx=int(g_idx),
                            old_code=current_code,
                            new_code=candidate_code,
                            flipped_bit=bit_pos,
                        )
                    )

            # Deterministic Top-M within each group:
            # 1) Higher proxy first
            # 2) Tie-break by new_code
            # 3) Then by flipped_bit
            group_candidates.sort(key=lambda c: (-c.score_proxy, c.new_code, c.flipped_bit))
            keep_n = min(top_m_per_group, len(group_candidates))
            candidates.extend(group_candidates[:keep_n])
            counters["candidates_kept_topm"] += keep_n

    # Deterministic pooled ordering before Stage B global Top-K exact verification.
    candidates.sort(
        key=lambda c: (-c.score_proxy, c.layer_name, c.group_idx, c.new_code, c.flipped_bit)
    )

    return candidates, counters


def enumerate_bitmask_swap_candidates(
    model: nn.Module,
    device: str,
    exclude_groups: Set[Tuple[str, int]],
    forbidden_swaps: Set[Tuple[str, int, int]],
) -> Tuple[List[UnifiedCandidate], Dict[str, int]]:
    """
    Enumerate bitmask swap candidates (cost=2).

    Reused from R1_T03: enumerate_cost2_swap_candidates()
    """
    counters = {
        "total_groups": 0,
        "valid_groups": 0,
        "candidates_considered": 0,
        "candidates_valid": 0,
        "candidates_skipped_excluded": 0,
        "candidates_skipped_forbidden": 0,
    }

    candidates: List[UnifiedCandidate] = []

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

        for g_idx in range(num_groups):
            if (layer_name, int(g_idx)) in exclude_groups:
                counters["candidates_skipped_excluded"] += 1
                continue

            m_group = m_flat[g_idx]
            current_mask = get_current_bitmask(m_group)

            if current_mask is None:
                continue

            counters["valid_groups"] += 1
            current_pattern = bitmask_to_pattern(current_mask)
            if current_pattern is None:
                continue

            w_tilde_current = compute_dense_reconstruction(w_flat, m_flat, scale, g_idx)
            grad_group = g_flat[g_idx]

            # Enumerate all 4 cost-2 swaps
            swaps = enumerate_cost2_swaps(current_mask)

            best_score_for_group = float('-inf')
            best_candidate_for_group: Optional[UnifiedCandidate] = None

            for new_mask, bit_off, bit_on, swap_code in swaps:
                swap_key = (layer_name, int(g_idx), swap_code)
                if swap_key in forbidden_swaps:
                    counters["candidates_skipped_forbidden"] += 1
                    continue

                counters["candidates_considered"] += 1

                new_pattern = bitmask_to_pattern(new_mask)
                if new_pattern is None:
                    continue

                # Compute new dense reconstruction
                new_mask_tensor = bitmask_to_mask_tensor(new_mask, device)
                w_tilde_new = w_flat[g_idx].float() * scale * new_mask_tensor
                delta_w_tilde = w_tilde_new - w_tilde_current

                # Proxy score: gradient dot delta
                proxy_score = float(torch.dot(grad_group, delta_w_tilde).item())

                if proxy_score > 0:
                    counters["candidates_valid"] += 1
                    # Keep the best candidate for this group
                    if proxy_score > best_score_for_group:
                        best_score_for_group = proxy_score
                        best_candidate_for_group = UnifiedCandidate(
                            action_type="bitmask_swap",
                            score_proxy=proxy_score,
                            score_proxy_norm=proxy_score / 2.0,  # cost=2
                            layer_name=layer_name,
                            cost=2,
                            group_idx=int(g_idx),
                            old_mask=current_mask,
                            new_mask=new_mask,
                            bit_off=bit_off,
                            bit_on=bit_on,
                        )

            if best_candidate_for_group is not None:
                candidates.append(best_candidate_for_group)

    return candidates, counters


def get_current_bitmask(mask_group: torch.Tensor) -> Optional[int]:
    """Extract the current 4-bit bitmask from a mask group."""
    active = (mask_group > 0.5).nonzero(as_tuple=False).flatten().tolist()
    if len(active) != 2:
        return None
    a, b = int(active[0]), int(active[1])
    return pattern_to_bitmask((a, b))


# =============================================================================
# Candidate Application and Reversion
# =============================================================================

def apply_candidate(candidate: UnifiedCandidate, model: nn.Module, device: str) -> bool:
    """Apply a candidate action to the model."""
    if candidate.action_type == "weight_bit":
        return _apply_weight_bit(candidate, model)
    elif candidate.action_type == "index_1bit":
        return _apply_index_1bit(candidate, model, device)
    elif candidate.action_type == "bitmask_swap":
        return _apply_bitmask_swap(candidate, model, device)
    return False


def revert_candidate(candidate: UnifiedCandidate, model: nn.Module, device: str) -> bool:
    """Revert a candidate action from the model."""
    if candidate.action_type == "weight_bit":
        # Flipping same bit twice reverts
        return _apply_weight_bit(candidate, model)
    elif candidate.action_type == "index_1bit":
        # Swap old and new codes to revert
        reverted = replace(candidate, old_code=candidate.new_code, new_code=candidate.old_code)
        return _apply_index_1bit(reverted, model, device)
    elif candidate.action_type == "bitmask_swap":
        # Swap old and new masks to revert
        reverted = replace(candidate, old_mask=candidate.new_mask, new_mask=candidate.old_mask)
        return _apply_bitmask_swap(reverted, model, device)
    return False


def _apply_weight_bit(candidate: UnifiedCandidate, model: nn.Module) -> bool:
    """Apply a weight-bit flip."""
    # Find the module by layer name
    for name, module in model.named_modules():
        if name == candidate.layer_name and hasattr(module, 'int8_weights'):
            with torch.no_grad():
                int8_weights_flat = module.int8_weights.flatten()
                current_val = int8_weights_flat[candidate.weight_idx].item()
                new_val = flip_int8_value(int(current_val), candidate.bit_pos)
                int8_weights_flat[candidate.weight_idx] = torch.tensor(new_val, dtype=torch.int8)
            return True
    return False


def _apply_index_1bit(candidate: UnifiedCandidate, model: nn.Module, device: str) -> bool:
    """Apply an index 1-bit metadata change."""
    for name, module in model.named_modules():
        if name == candidate.layer_name and hasattr(module, 'sparse_mask'):
            mask = module.sparse_mask
            int8_w = module.int8_weights

            m_flat, m_meta = flatten_groups(mask)
            w_flat, w_meta = flatten_groups(int8_w)

            if m_flat is None or w_flat is None:
                return False

            g_idx = candidate.group_idx
            old_pattern = code_to_pattern(candidate.old_code)
            new_pattern = code_to_pattern(candidate.new_code)

            if old_pattern is None or new_pattern is None:
                return False

            # Get old active values
            old_mask_tensor = m_flat[g_idx].clone()
            old_active_indices = (old_mask_tensor > 0.5).nonzero().flatten()
            old_values = w_flat[g_idx, old_active_indices].clone()

            # Zero out and set new pattern
            w_flat[g_idx, :] = 0
            m_flat[g_idx, :] = 0

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
    return False


def _apply_bitmask_swap(candidate: UnifiedCandidate, model: nn.Module, device: str) -> bool:
    """Apply a bitmask swap change."""
    for name, module in model.named_modules():
        if name == candidate.layer_name and hasattr(module, 'sparse_mask'):
            mask = module.sparse_mask
            int8_w = module.int8_weights

            m_flat, m_meta = flatten_groups(mask)
            w_flat, w_meta = flatten_groups(int8_w)

            if m_flat is None or w_flat is None:
                return False

            g_idx = candidate.group_idx
            old_pattern = bitmask_to_pattern(candidate.old_mask)
            new_pattern = bitmask_to_pattern(candidate.new_mask)

            if old_pattern is None or new_pattern is None:
                return False

            # Get old active values
            old_mask_tensor = m_flat[g_idx].clone()
            old_active_indices = (old_mask_tensor > 0.5).nonzero().flatten()
            old_values = w_flat[g_idx, old_active_indices].clone()

            # Zero out and set new pattern
            w_flat[g_idx, :] = 0
            m_flat[g_idx, :] = 0

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
    return False


# =============================================================================
# Two-Stage Selection
# =============================================================================

def stage_a_proxy_scoring(
    model: nn.Module,
    calib_loader,
    device: str,
    calib_samples: int,
    top_k: int,
    flipped_bits: Set[Tuple[str, int, int]],
    exclude_groups: Set[Tuple[str, int]],
    forbidden_transitions: Set[Tuple[str, int, int, int]],
    forbidden_swaps: Set[Tuple[str, int, int]],
    enable_weight_bit: bool = True,
    enable_index_1bit: bool = True,
    enable_bitmask_swap: bool = True,
    top_m_per_group: int = 3,
) -> Tuple[List[UnifiedCandidate], Dict[str, int], Dict[str, int]]:
    """
    Stage A: Fast proxy scoring with Two-Pool Quota System.

    THEORETICAL INSIGHT:
    Metadata attacks change the selection of Activations (routing attack),
    whereas Weight-bit attacks only change the amplitude. The first-order
    Taylor expansion (gradient * delta_W) used in Proxy Score is inherently
    blind to massive activation shifts, mathematically disadvantaging the
    Metadata candidates during Stage A pooling.

    To create a mathematically fair Joint Attack, we implement a
    "Two-Pool Quota System" so that both attack types reach Stage B
    (Exact Verification), where the true forward pass will naturally
    and fairly evaluate the impact of the activation shifts.

    Returns:
        - Top-K candidates with quota-based pooling (K/2 from each pool)
        - Counters for each action type (total candidates)
        - Counters for top-K selection by type
    """
    # Compute gradients once
    compute_gradients(model, calib_loader, device, calib_samples)

    # Separate pools for fair competition
    weight_pool: List[UnifiedCandidate] = []
    metadata_pool: List[UnifiedCandidate] = []

    counters_total = {}
    counters_topk = {}

    # Generate candidates from each action type
    if enable_weight_bit:
        weight_candidates, weight_counters = enumerate_weight_bit_candidates(
            model, device, flipped_bits, max_candidates_per_layer=500
        )
        weight_pool.extend(weight_candidates)
        counters_total["weight_bit"] = len(weight_candidates)

    if enable_index_1bit:
        index_candidates, index_counters = enumerate_index_1bit_candidates(
            model, device, exclude_groups, forbidden_transitions, top_m_per_group
        )
        metadata_pool.extend(index_candidates)
        counters_total["index_1bit"] = len(index_candidates)

    if enable_bitmask_swap:
        bitmask_candidates, bitmask_counters = enumerate_bitmask_swap_candidates(
            model, device, exclude_groups, forbidden_swaps
        )
        metadata_pool.extend(bitmask_candidates)
        counters_total["bitmask_swap"] = len(bitmask_candidates)

    # Independent sorting within each pool
    # Weight pool: sort by proxy_score_norm (score / cost)
    weight_pool.sort(key=lambda c: c.score_proxy_norm, reverse=True)
    # Metadata pool: sort by proxy_score_norm (score / cost)
    metadata_pool.sort(key=lambda c: c.score_proxy_norm, reverse=True)

    # Quota Merge: Take top K/2 from each pool
    quota_per_pool = top_k // 2

    top_k_from_weight = weight_pool[:quota_per_pool]
    top_k_from_metadata = metadata_pool[:quota_per_pool]

    # Merge for Stage B exact verification
    top_k_candidates = top_k_from_weight + top_k_from_metadata

    # Count top-K by type for diagnostics
    for c in top_k_candidates:
        counters_topk[c.action_type] = counters_topk.get(c.action_type, 0) + 1

    # Fill in missing types
    for action_type in ["weight_bit", "index_1bit", "bitmask_swap"]:
        if action_type not in counters_total:
            counters_total[action_type] = 0
        if action_type not in counters_topk:
            counters_topk[action_type] = 0

    return top_k_candidates, counters_total, counters_topk


def stage_b_exact_verification(
    model: nn.Module,
    candidates: List[UnifiedCandidate],
    calib_loader,
    device: str,
    eval_samples: int,
) -> List[UnifiedCandidate]:
    """
    Stage B: Exact forward loss verification on top-K candidates.

    For each candidate:
    1. Apply candidate to model
    2. Compute exact loss on calibration batch
    3. Revert candidate
    4. Store exact_delta_loss in candidate
    """
    if not candidates:
        return []

    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Collect calibration batch for exact verification
    calib_data = []
    calib_targets = []
    for inputs, targets in calib_loader:
        calib_data.append(inputs.to(device))
        calib_targets.append(targets.to(device))
        if len(calib_data) * inputs.size(0) >= eval_samples:
            break

    calib_inputs = torch.cat(calib_data, dim=0)[:eval_samples]
    calib_targets = torch.cat(calib_targets, dim=0)[:eval_samples]

    # Compute baseline loss once
    with torch.no_grad():
        baseline_outputs = model(calib_inputs)
        baseline_loss = criterion(baseline_outputs, calib_targets).item()

    verified_candidates = []

    for candidate in candidates:
        # Apply candidate
        if not apply_candidate(candidate, model, device):
            continue

        # Evaluate exact loss
        with torch.no_grad():
            new_outputs = model(calib_inputs)
            new_loss = criterion(new_outputs, calib_targets).item()

        exact_delta = new_loss - baseline_loss

        # Revert candidate
        revert_candidate(candidate, model, device)

        # Create verified candidate with exact delta
        verified = replace(candidate,
                          exact_delta_loss=exact_delta,
                          exact_delta_loss_norm=exact_delta / candidate.cost)
        verified_candidates.append(verified)

    return verified_candidates


# =============================================================================
# Main Attack Loop
# =============================================================================

def run_joint_best_step_attack(
    model: nn.Module,
    test_loader,
    calib_loader,
    device: str,
    seed: int,
    physical_budget: int = 50,
    calib_samples: int = 256,
    eval_samples: int = 2000,
    top_k: int = 64,
    enable_weight_bit: bool = True,
    enable_index_1bit: bool = True,
    enable_bitmask_swap: bool = True,
    top_m_per_group: int = 3,
) -> Dict:
    """
    Run the unified best-step attack with two-stage selection.

    At each logical step:
    1. Stage A: Generate candidates from all action types, get top-K by proxy score/cost
    2. Stage B: Exact verification on top-K, compute exact delta loss
    3. Select action with max exact_delta_loss / cost
    4. Apply action, update physical flips used
    """
    rng = random.Random(seed)

    # Initial evaluation
    acc0, loss0 = evaluate_subset(model, test_loader, device=device, max_samples=eval_samples)
    initial_metadata_hash = get_metadata_hash(model)
    initial_int8_hash = get_int8_weights_hash(model)

    acc_by_physical = [float(acc0)]
    loss_by_physical = [float(loss0)]
    acc_by_logical = [float(acc0)]
    loss_by_logical = [float(loss0)]
    step_logs: List[AttackStepLog] = []

    wall_t0 = time.time()

    # State tracking
    flipped_bits: Set[Tuple[str, int, int]] = set()
    exclude_groups: Set[Tuple[str, int]] = set()
    forbidden_transitions: Set[Tuple[str, int, int, int]] = set()
    forbidden_swaps: Set[Tuple[str, int, int]] = set()

    # Action breakdown
    action_breakdown = {
        "weight_bit": 0,
        "index_1bit": 0,
        "bitmask_swap": 0,
    }

    step = 0
    physical_flips_used = 0

    quota_per_pool = top_k // 2
    print(f"[Attack] Starting R1_T05: Joint best-step attack (FAIR QUOTA SYSTEM)")
    print(f"[Attack] Physical budget: {physical_budget}, top-K: {top_k}, quota per pool: {quota_per_pool}")
    print(f"[Attack] top-M-per-group: {top_m_per_group}")
    print(f"[Attack] Enabled: weight_bit={enable_weight_bit}, index_1bit={enable_index_1bit}, "
          f"bitmask_swap={enable_bitmask_swap}")
    print(f"[Attack] {'Step':>4} | {'Phy':>3} | {'Type':>14} | {'Cost':>4} | "
          f"{'Proxy':>8} | {'Exact':>8} | {'Exact/C':>8} | {'Acc':>6} | {'Loss':>8}")

    while physical_flips_used < physical_budget:
        step += 1
        step_t0 = time.time()

        # Stage A: Fast proxy scoring
        top_k_candidates, counters_total, counters_topk = stage_a_proxy_scoring(
            model, calib_loader, device, calib_samples, top_k,
            flipped_bits, exclude_groups, forbidden_transitions, forbidden_swaps,
            enable_weight_bit, enable_index_1bit, enable_bitmask_swap,
            top_m_per_group,
        )

        if not top_k_candidates:
            print(f"[Attack] No candidates available at step {step}. Stopping.")
            break

        # Stage B: Exact verification
        verified_candidates = stage_b_exact_verification(
            model, top_k_candidates, calib_loader, device, eval_samples
        )

        if not verified_candidates:
            print(f"[Attack] No verified candidates at step {step}. Stopping.")
            break

        # Select best action by exact_delta_loss / cost
        best_candidate = max(verified_candidates, key=lambda c: c.exact_delta_loss_norm)

        # Check if we can afford this action
        if physical_flips_used + best_candidate.cost > physical_budget:
            print(f"[Attack] Cannot afford action (cost={best_candidate.cost}) at step {step}. Stopping.")
            break

        # Apply the chosen action
        apply_candidate(best_candidate, model, device)

        # Update state tracking
        if best_candidate.action_type == "weight_bit":
            flipped_bits.add((best_candidate.layer_name, best_candidate.weight_idx, best_candidate.bit_pos))
        elif best_candidate.action_type == "index_1bit":
            transition_key = (
                best_candidate.layer_name,
                best_candidate.group_idx,
                best_candidate.old_code,
                best_candidate.new_code
            )
            forbidden_transitions.add(transition_key)
            # Add reverse
            forbidden_transitions.add((
                best_candidate.layer_name,
                best_candidate.group_idx,
                best_candidate.new_code,
                best_candidate.old_code
            ))
            exclude_groups.add((best_candidate.layer_name, int(best_candidate.group_idx)))
        elif best_candidate.action_type == "bitmask_swap":
            swap_code = (best_candidate.bit_off << 8) | best_candidate.bit_on
            forbidden_swaps.add((best_candidate.layer_name, int(best_candidate.group_idx), swap_code))
            # Add reverse
            reverse_swap_code = (best_candidate.bit_on << 8) | best_candidate.bit_off
            forbidden_swaps.add((best_candidate.layer_name, int(best_candidate.group_idx), reverse_swap_code))
            exclude_groups.add((best_candidate.layer_name, int(best_candidate.group_idx)))

        # Clean up old state
        if len(exclude_groups) > 20:
            exclude_groups = set(list(exclude_groups)[-20:])
        if len(forbidden_transitions) > 1000:
            forbidden_transitions = set(list(forbidden_transitions)[-1000:])
        if len(forbidden_swaps) > 1000:
            forbidden_swaps = set(list(forbidden_swaps)[-1000:])

        # Update counters
        physical_flips_used += best_candidate.cost
        action_breakdown[best_candidate.action_type] += 1

        # Evaluate
        acc, loss = evaluate_subset(model, test_loader, device=device, max_samples=eval_samples)

        # Update history (fill in intermediate physical flips if needed)
        while len(acc_by_physical) < physical_flips_used:
            acc_by_physical.append(acc_by_physical[-1])
            loss_by_physical.append(loss_by_physical[-1])
        acc_by_physical.append(float(acc))
        loss_by_physical.append(float(loss))

        acc_by_logical.append(float(acc))
        loss_by_logical.append(float(loss))

        step_time = time.time() - step_t0

        # Generate description
        if best_candidate.action_type == "weight_bit":
            desc = f"bit{best_candidate.bit_pos}@w[{best_candidate.weight_idx}]"
        elif best_candidate.action_type == "index_1bit":
            desc = f"c{best_candidate.old_code}(0b{best_candidate.old_code:04b})->c{best_candidate.new_code}(0b{best_candidate.new_code:04b})"
        else:  # bitmask_swap
            desc = f"m{best_candidate.old_mask:02x}(0b{best_candidate.old_mask:04b})->m{best_candidate.new_mask:02x}(0b{best_candidate.new_mask:04b})"

        # Log this step
        step_log = AttackStepLog(
            step=step,
            physical_flips_used=physical_flips_used,
            action_type=best_candidate.action_type,
            cost=best_candidate.cost,
            layer_name=best_candidate.layer_name,
            proxy_score_norm=best_candidate.score_proxy_norm,
            exact_delta=best_candidate.exact_delta_loss,
            exact_delta_norm=best_candidate.exact_delta_loss_norm,
            accuracy=float(acc),
            loss=float(loss),
            description=desc,
            candidates_total_by_type=counters_total.copy(),
            candidates_topk_by_type=counters_topk.copy(),
        )
        step_logs.append(step_log)

        print(f"[Attack] {step:4d} | {physical_flips_used:3d} | "
              f"{best_candidate.action_type:>14} | {best_candidate.cost:4d} | "
              f"{best_candidate.score_proxy_norm:>8.4f} | "
              f"{best_candidate.exact_delta_loss:>8.4f} | "
              f"{best_candidate.exact_delta_loss_norm:>8.4f} | "
              f"{acc:6.2f}% | {loss:8.4f} | {desc} t={step_time:.2f}s")

    # Fill remaining physical budget
    while len(acc_by_physical) <= physical_budget:
        acc_by_physical.append(acc_by_physical[-1])
        loss_by_physical.append(loss_by_physical[-1])

    wall_time = time.time() - wall_t0

    return {
        "seed": seed,
        "physical_budget": physical_budget,
        "top_k": top_k,
        "calib_samples": calib_samples,
        "eval_samples": eval_samples,
        "initial_accuracy": float(acc_by_physical[0]),
        "final_accuracy": float(acc_by_physical[physical_budget]),
        "total_flips": physical_flips_used,
        "logical_steps": len(step_logs),
        "accuracy_history_physical": acc_by_physical,
        "loss_history_physical": loss_by_physical,
        "accuracy_history_logical": acc_by_logical,
        "loss_history_logical": loss_by_logical,
        "step_logs": step_logs,
        "action_breakdown": action_breakdown,
        "initial_metadata_hash": initial_metadata_hash,
        "final_metadata_hash": get_metadata_hash(model),
        "initial_int8_hash": initial_int8_hash,
        "final_int8_hash": get_int8_weights_hash(model),
        "timing": {
            "wall_sec": wall_time,
        },
        "config": {
            "enable_weight_bit": enable_weight_bit,
            "enable_index_1bit": enable_index_1bit,
            "enable_bitmask_swap": enable_bitmask_swap,
        },
    }


# =============================================================================
# Main Script
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="R1_T05: Joint Best-Step Attack (Weight-Bit + Index + Bitmask)"
    )
    parser.add_argument("--model", type=str, default="resnet20", choices=["resnet20"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"])
    parser.add_argument("--physical-budget", type=int, default=50,
                        help="Physical bit flip budget (default: 50)")
    parser.add_argument("--topk", type=int, default=64,
                        help="Top-K candidates for exact verification (default: 64)")
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--calib-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint (required)")
    parser.add_argument("--out-prefix", type=str, default=None,
                        help="Output prefix (default: results/R1/R1_T05_joint_best_step_attack)")
    parser.add_argument("--disable-weight-bit", action="store_true",
                        help="Disable weight-bit flip action")
    parser.add_argument("--disable-index-1bit", action="store_true",
                        help="Disable index 1-bit metadata action")
    parser.add_argument("--disable-bitmask-swap", action="store_true",
                        help="Disable bitmask swap metadata action")
    parser.add_argument("--top-m-per-group", type=int, default=3,
                        help="Top-M candidates per group in Stage A (default: 3)")

    args = parser.parse_args()

    # Set up output directory
    if args.out_prefix is None:
        args.out_prefix = "results/R1/R1_T05_joint_best_step_attack"
    os.makedirs(os.path.dirname(args.out_prefix) or "results/R1", exist_ok=True)
    os.makedirs("results/R1", exist_ok=True)

    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[Warning] CUDA not available, using CPU")

    # Set seeds
    set_all_seeds(int(args.seed))

    # Determine enabled actions
    enable_weight_bit = not args.disable_weight_bit
    enable_index_1bit = not args.disable_index_1bit
    enable_bitmask_swap = not args.disable_bitmask_swap

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

        # Extract sparse_mask tensors
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
            print(f"[Model] Warning: Missing keys: {missing[:5]}...")
        if unexpected:
            print(f"[Model] Warning: Unexpected keys: {unexpected[:5]}...")

        # CRITICAL: Must call calibrate_all_layers()
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
    print(f"[Attack] Starting R1_T05: Joint best-step attack")
    print(f"[Attack] Config: physical_budget={args.physical_budget}, top_k={args.topk}, "
          f"calib_samples={args.calib_samples}, eval_samples={args.eval_samples}")
    print(f"[Attack] Enabled actions: weight_bit={enable_weight_bit}, "
          f"index_1bit={enable_index_1bit}, bitmask_swap={enable_bitmask_swap}")

    result = run_joint_best_step_attack(
        model=model,
        test_loader=test_loader,
        calib_loader=test_loader,
        device=device,
        seed=int(args.seed),
        physical_budget=int(args.physical_budget),
        calib_samples=int(args.calib_samples),
        eval_samples=int(args.eval_samples),
        top_k=int(args.topk),
        enable_weight_bit=enable_weight_bit,
        enable_index_1bit=enable_index_1bit,
        enable_bitmask_swap=enable_bitmask_swap,
        top_m_per_group=int(args.top_m_per_group),
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
             label="R1_T05: Joint Best-Step Attack")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlabel("Physical Flips (Cost: 1 for weight/index, 2 for bitmask)", fontsize=12, fontweight="bold")
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title(f"R1_T05: Joint Best-Step Attack ({result['logical_steps']} steps, "
              f"{result['total_flips']} physical flips)",
              fontsize=14, fontweight="bold")
    plt.ylim(0, 100)
    plt.xlim(0, args.physical_budget)
    plt.legend(loc="best")
    plt.tight_layout()
    png_path_physical = f"{base_name}_curve_physical.png"
    plt.savefig(png_path_physical, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Output] Saved {png_path_physical}")

    # 3. Generate plot (vs logical steps)
    plt.figure(figsize=(10, 6))
    acc_hist_logical = result["accuracy_history_logical"]
    logical_steps = list(range(len(acc_hist_logical)))
    plt.plot(logical_steps, acc_hist_logical, marker="s", linewidth=2, markersize=4,
             label="R1_T05: Joint Best-Step Attack")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlabel("Logical Steps", fontsize=12, fontweight="bold")
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title(f"R1_T05: Joint Best-Step Attack ({result['logical_steps']} logical steps)",
              fontsize=14, fontweight="bold")
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.legend(loc="best")
    plt.tight_layout()
    png_path_logical = f"{base_name}_curve_logical.png"
    plt.savefig(png_path_logical, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Output] Saved {png_path_logical}")

    # 4. Generate action breakdown plot
    steps = list(range(1, result["logical_steps"] + 1))
    cumulative = {
        "weight_bit": [],
        "index_1bit": [],
        "bitmask_swap": [],
    }
    running = {"weight_bit": 0, "index_1bit": 0, "bitmask_swap": 0}

    for log in result["step_logs"]:
        running[log.action_type] += 1
        cumulative["weight_bit"].append(running["weight_bit"])
        cumulative["index_1bit"].append(running["index_1bit"])
        cumulative["bitmask_swap"].append(running["bitmask_swap"])

    # Build stacked boundaries element-wise (avoid list concatenation).
    cumulative_weight = cumulative["weight_bit"]
    cumulative_weight_plus_index = [
        w + i for w, i in zip(cumulative["weight_bit"], cumulative["index_1bit"])
    ]
    cumulative_total = [
        w + i + b
        for w, i, b in zip(
            cumulative["weight_bit"],
            cumulative["index_1bit"],
            cumulative["bitmask_swap"],
        )
    ]

    plt.figure(figsize=(10, 6))
    plt.fill_between(steps, 0, cumulative_weight, alpha=0.7, label="Weight-Bit Flip", color="#1f77b4")
    plt.fill_between(steps, cumulative_weight,
                     cumulative_weight_plus_index,
                     alpha=0.7, label="Index 1-bit", color="#ff7f0e")
    plt.fill_between(steps,
                     cumulative_weight_plus_index,
                     cumulative_total,
                     alpha=0.7, label="Bitmask Swap", color="#2ca02c")
    plt.xlabel("Logical Steps", fontsize=12, fontweight="bold")
    plt.ylabel("Cumulative Action Count", fontsize=12, fontweight="bold")
    plt.title(f"R1_T05: Action Breakdown ({result['action_breakdown']})", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    breakdown_path = f"{base_name}_action_breakdown.png"
    plt.savefig(breakdown_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Output] Saved {breakdown_path}")

    # 5. Save summary table
    csv_path = f"{base_name}_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "baseline_acc", "final_acc", "drop",
            "physical_budget", "physical_flips_used", "logical_steps",
            "action_weight_bit", "action_index_1bit", "action_bitmask_swap",
            "top_k", "runtime_sec",
        ])
        writer.writerow([
            f"{result['initial_accuracy']:.2f}",
            f"{result['final_accuracy']:.2f}",
            f"{result['initial_accuracy'] - result['final_accuracy']:.2f}",
            result['physical_budget'],
            result['total_flips'],
            result['logical_steps'],
            result['action_breakdown']['weight_bit'],
            result['action_breakdown']['index_1bit'],
            result['action_breakdown']['bitmask_swap'],
            result['top_k'],
            f"{result['timing']['wall_sec']:.2f}",
        ])
    print(f"[Output] Saved {csv_path}")

    # 6. Save detailed log
    log_path = f"{base_name}_log.txt"
    with open(log_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("R1_T05: Joint Best-Step Attack (Weight-Bit + Index + Bitmask)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Script: run_R1_T05_joint_best_step_attack.py\n")
        f.write(f"Timestamp: {now_ts()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Checkpoint: {os.path.abspath(args.ckpt)}\n")
        f.write(f"\nConfig:\n")
        f.write(f"  seed: {args.seed}\n")
        f.write(f"  physical_budget: {args.physical_budget}\n")
        f.write(f"  top_k: {args.topk}\n")
        f.write(f"  calib_samples: {args.calib_samples}\n")
        f.write(f"  eval_samples: {args.eval_samples}\n")
        f.write(f"  enabled_actions: weight_bit={enable_weight_bit}, "
                f"index_1bit={enable_index_1bit}, bitmask_swap={enable_bitmask_swap}\n")
        f.write(f"\nResults:\n")
        f.write(f"  baseline_acc: {result['initial_accuracy']:.2f}%\n")
        f.write(f"  final_acc: {result['final_accuracy']:.2f}%\n")
        f.write(f"  acc_drop: {result['initial_accuracy'] - result['final_accuracy']:.2f}%\n")
        f.write(f"  physical_flips_used: {result['total_flips']}\n")
        f.write(f"  logical_steps: {result['logical_steps']}\n")
        f.write(f"\nAction Breakdown:\n")
        for action_type, count in result['action_breakdown'].items():
            f.write(f"  {action_type}: {count}\n")
        f.write(f"\nTiming:\n")
        f.write(f"  wall_sec: {result['timing']['wall_sec']:.2f}\n")
        f.write(f"\nHashes:\n")
        f.write(f"  initial_metadata: {result['initial_metadata_hash']}\n")
        f.write(f"  final_metadata: {result['final_metadata_hash']}\n")
        f.write(f"  initial_int8: {result['initial_int8_hash']}\n")
        f.write(f"  final_int8: {result['final_int8_hash']}\n")
        f.write(f"\nStep-by-step details:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Step':>4} | {'Phy':>3} | {'Type':>14} | {'Cost':>4} | "
                f"{'Proxy/C':>8} | {'Exact':>8} | {'Exact/C':>8} | {'Acc':>6} | {'Loss':>8} | Desc\n")
        f.write("-" * 80 + "\n")
        for log in result['step_logs']:
            f.write(f"{log.step:4d} | {log.physical_flips_used:3d} | "
                   f"{log.action_type:>14} | {log.cost:4d} | "
                   f"{log.proxy_score_norm:>8.4f} | "
                   f"{log.exact_delta:>8.4f} | "
                   f"{log.exact_delta_norm:>8.4f} | "
                   f"{log.accuracy:6.2f}% | {log.loss:8.4f} | {log.description}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("Reproduction command:\n")
        f.write(f"  python run_R1_T05_joint_best_step_attack.py \\\n")
        cmd_parts = [
            f"--device {device}",
            f"--seed {args.seed}",
            f"--physical-budget {args.physical_budget}",
            f"--topk {args.topk}",
            f"--calib-samples {args.calib_samples}",
            f"--eval-samples {args.eval_samples}",
        ]
        if not enable_weight_bit:
            cmd_parts.append("--disable-weight-bit")
        if not enable_index_1bit:
            cmd_parts.append("--disable-index-1bit")
        if not enable_bitmask_swap:
            cmd_parts.append("--disable-bitmask-swap")
        cmd_parts.append(f"--ckpt {args.ckpt}")
        for part in cmd_parts:
            f.write(f"    {part} \\\n")
        f.write("\n")
    print(f"[Output] Saved {log_path}")

    print(f"\n[Summary] R1_T05 Attack completed:")
    print(f"  Baseline accuracy: {result['initial_accuracy']:.2f}%")
    print(f"  Final accuracy: {result['final_accuracy']:.2f}%")
    print(f"  Accuracy drop: {result['initial_accuracy'] - result['final_accuracy']:.2f}%")
    print(f"  Physical budget: {args.physical_budget}")
    print(f"  Physical flips used: {result['total_flips']}")
    print(f"  Logical steps: {result['logical_steps']}")
    print(f"  Action breakdown: {result['action_breakdown']}")
    print(f"  Wall time: {result['timing']['wall_sec']:.2f}s")


if __name__ == "__main__":
    main()
