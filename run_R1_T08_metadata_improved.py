#!/usr/bin/env python3
"""
R1_T08: Improved Metadata BFA with Proxy/Apply Alignment and Top-M Retention

Key improvements over R1_T01-T05:
1. Proxy/Apply semantic alignment: w_new construction mirrors actual apply semantics
2. Top-M candidate retention per group (configurable via --top-m-per-group)
3. Deterministic candidate ordering with proper tie-breaking
4. Stage B with strict Save-Apply-Restore and tensor hash verification
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.factory import create_resnet20
from train.ptq_convert import Int8QuantizedConv2d, Int8QuantizedLinear, Int8QuantizedResNet


# =============================================================================
# Constants and Pattern Definitions (Index Encoding)
# =============================================================================

ALL_2OF4_PATTERNS = [
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
]


def encode_index_to_4bit(i: int, j: int) -> int:
    """
    Encode two 2-bit indices (positions in a group of 4) into a 4-bit code.

    Each index is 2 bits (0-3), so we pack them as: (j << 2) | i
    where i is the first index and j is the second index.

    Args:
        i: First position index (0-3)
        j: Second position index (0-3)

    Returns:
        4-bit code (0-15)
    """
    return (j << 2) | i


def decode_4bit_to_index(code: int) -> Tuple[int, int]:
    """
    Decode a 4-bit code into two 2-bit indices.

    Args:
        code: 4-bit code (0-15)

    Returns:
        Tuple of (i, j) where each is a position index (0-3)
    """
    i = code & 0x3  # Lower 2 bits
    j = (code >> 2) & 0x3  # Upper 2 bits
    return (i, j)


def pattern_to_code(pattern: Tuple[int, int]) -> int:
    """
    Convert a 2-of-4 pattern tuple to a 4-bit code using index encoding.

    Args:
        pattern: Tuple of two distinct positions (i, j) where i < j

    Returns:
        4-bit code (0-15)
    """
    return encode_index_to_4bit(pattern[0], pattern[1])


def code_to_pattern(code: int) -> Optional[Tuple[int, int]]:
    """
    Convert a 4-bit code to a 2-of-4 pattern tuple.

    Returns None if the code decodes to an invalid pattern (collision, i.e., i == j).

    Args:
        code: 4-bit code (0-15)

    Returns:
        Tuple of two positions (i, j) sorted, or None if invalid (collision)
    """
    i, j = decode_4bit_to_index(code)
    if i == j:
        # Collision: same position twice - invalid
        return None
    # Return sorted tuple
    return (min(i, j), max(i, j))


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class GroupCandidate:
    """Represents a single group-level metadata flip candidate."""
    proxy_score: float
    layer_name: str
    module: nn.Module
    group_idx: int
    old_code: int
    new_code: int
    flipped_bit: int
    old_pattern: Tuple[int, int]
    new_pattern: Tuple[int, int]
    delta_w_tilde: torch.Tensor


# =============================================================================
# Utilities
# =============================================================================

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Pattern Utilities
# =============================================================================
# Note: pattern_to_code and code_to_pattern are now using Index Encoding
# and are defined above in the Constants section.


def get_current_pattern(mask_group: torch.Tensor) -> Optional[Tuple[int, int]]:
    """Extract the current 2-of-4 pattern from a mask group."""
    active_indices = (mask_group > 0.5).nonzero(as_tuple=False).flatten()
    if active_indices.numel() != 2:
        return None
    return (int(active_indices[0].item()), int(active_indices[1].item()))


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


# =============================================================================
# Layer Utilities
# =============================================================================

def get_sparse_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Get all sparse convolutional layers (Int8QuantizedConv2d with sparse_mask)."""
    sparse_layers = []
    for name, module in model.named_modules():
        if isinstance(module, Int8QuantizedConv2d) and module.sparse_mask is not None:
            sparse_layers.append((name, module))
    return sparse_layers


# =============================================================================
# Stage A: Candidate Enumeration (Agent A's Refactored Version)
# =============================================================================

def enumerate_group_candidates(
    model: nn.Module,
    device: str,
    exclude_groups: Optional[Set[Tuple[str, int]]] = None,
    forbidden_transitions: Optional[Set[Tuple[str, int, int, int]]] = None,
    top_m_per_group: int = 3,
    debug: bool = False,
) -> Tuple[List[GroupCandidate], Dict[str, int]]:
    """
    Stage A candidate enumeration with three key features:

    1) Proxy/Apply semantic alignment:
       Build w_new exactly like apply_pattern_change() does:
       read current active INT8 values in active-index order, zero group,
       write values into new_pattern order, then reconstruct dense w_tilde_new.

    2) Top-M retention per group:
       Keep top-M candidates per group (deterministic tie-break), then pool
       all retained candidates for global Top-K exact verification.

    3) 1-bit reachable constraint (Index encoding):
       For each current pattern, flip each bit (0-3) once to generate candidates.
       A candidate is valid only if the decoded indices don't collide (i != j).
       This enforces physical BFA constraint: only 1-bit flips are allowed.
    """
    device_obj = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    if debug:
        sparse_layers = get_sparse_layers(model)
        print(f"[DEBUG] Found {len(sparse_layers)} sparse layers")
        for name, module in sparse_layers[:3]:
            has_grad = module.weight.grad is not None
            print(f"[DEBUG]   {name}: weight.grad={has_grad}")
            if has_grad:
                g_flat, _ = flatten_groups(module.weight.grad.data)
                w_flat, _ = flatten_groups(module.int8_weights)
                m_flat, _ = flatten_groups(module.sparse_mask)
                print(f"[DEBUG]     grad groups: {g_flat.shape if g_flat is not None else 'None'}")
                print(f"[DEBUG]     int8 groups: {w_flat.shape if w_flat is not None else 'None'}")
                print(f"[DEBUG]     mask groups: {m_flat.shape if m_flat is not None else 'None'}")

    if top_m_per_group < 1:
        raise ValueError(f"top_m_per_group must be >= 1, got {top_m_per_group}")

    if exclude_groups is None:
        exclude_groups = set()
    if forbidden_transitions is None:
        forbidden_transitions = set()

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

    pooled_candidates: List[GroupCandidate] = []

    for layer_name, module in get_sparse_layers(model):
        if module.weight.grad is None:
            counters["candidates_no_gradient"] += 1
            continue

        grad = module.weight.grad.data
        int8_w = module.int8_weights
        mask = module.sparse_mask
        scale = float(module.scale.item())

        g_flat, _ = flatten_groups(grad)
        w_flat, _ = flatten_groups(int8_w)
        m_flat, _ = flatten_groups(mask)

        if g_flat is None or w_flat is None or m_flat is None:
            continue

        num_groups = int(w_flat.shape[0])
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

            group_candidates: List[GroupCandidate] = []

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

                counters["candidates_valid"] += 1
                group_candidates.append(
                    GroupCandidate(
                        proxy_score=proxy_score,
                        layer_name=layer_name,
                        module=module,
                        group_idx=int(g_idx),
                        old_code=current_code,
                        new_code=candidate_code,
                        flipped_bit=bit_pos,
                        old_pattern=current_pattern,
                        new_pattern=candidate_pattern,
                        delta_w_tilde=delta_w_tilde.detach().cpu(),
                    )
                )

            # Deterministic Top-M within each group:
            # 1) Higher proxy first
            # 2) Tie-break by new_code
            # 3) Then by flipped_bit
            group_candidates.sort(key=lambda c: (-c.proxy_score, c.new_code, c.flipped_bit))
            keep_n = min(top_m_per_group, len(group_candidates))
            pooled_candidates.extend(group_candidates[:keep_n])
            counters["candidates_kept_topm"] += keep_n

        # Match existing behavior: free gradient storage after this layer.
        if module.weight.grad is not None:
            module.weight.grad = None

    # Deterministic pooled ordering before Stage B global Top-K exact verification.
    pooled_candidates.sort(
        key=lambda c: (-c.proxy_score, c.layer_name, c.group_idx, c.new_code, c.flipped_bit)
    )

    if debug:
        print(f"[DEBUG] Stage A counters: {counters}")

    return pooled_candidates, counters


# =============================================================================
# Tensor Hash Utilities for State Verification
# =============================================================================

def compute_model_hash(model: nn.Module) -> int:
    """Compute a hash of all model parameters for state verification."""
    hash_value = 0
    for name, p in sorted(model.named_parameters()):
        if p.data is not None:
            hash_value ^= hash((name, p.data.shape, p.data.sum().item(), p.data.abs().sum().item()))
    for name, b in sorted(model.named_buffers()):
        if b.data is not None:
            hash_value ^= hash((name, b.data.shape, b.data.sum().item(), b.data.abs().sum().item()))
    return hash_value


# =============================================================================
# Stage B: Exact Verification with Save-Apply-Restore
# =============================================================================

def exact_verification_topk(
    model: nn.Module,
    candidates: List[GroupCandidate],
    test_imgs: torch.Tensor,
    test_tgts: torch.Tensor,
    criterion: nn.Module,
    baseline_loss: float,
    top_k: int = 100,
) -> Tuple[Optional[GroupCandidate], Dict[str, int]]:
    """
    Stage B: Exact verification with strict Save-Apply-Restore pattern.

    For each candidate:
        1. Hash model state before modification
        2. Save original values
        3. Apply candidate change
        4. Compute new loss
        5. Restore original values
        6. Hash model state after restoration
        7. Verify hashes match

    Returns the candidate with maximum positive loss increase.
    """
    counters = {
        "candidates_tested": 0,
        "candidates_positive_delta": 0,
        "candidates_negative_delta": 0,
        "hash_mismatches": 0,
    }

    best_candidate = None
    best_delta = 0.0
    best_new_loss = baseline_loss

    model_hash_before = compute_model_hash(model)

    # Test top-K candidates from Stage A
    for candidate in candidates[:top_k]:
        counters["candidates_tested"] += 1

        module = candidate.module
        group_idx = candidate.group_idx
        old_pattern = candidate.old_pattern
        new_pattern = candidate.new_pattern

        # Flatten to get group position
        int8_w = module.int8_weights
        w_flat, w_meta = flatten_groups(int8_w)
        m_flat, m_meta = flatten_groups(module.sparse_mask)

        if w_flat is None or w_meta is None or m_flat is None or m_meta is None:
            continue

        w_group = w_flat[group_idx]
        m_group = m_flat[group_idx]

        # Save original state
        orig_mask = m_group.clone()
        orig_weights = w_group.clone()

        # Read old active values in order
        old_active_indices = (orig_mask > 0.5).nonzero(as_tuple=False).flatten()
        if old_active_indices.numel() != 2:
            continue
        old_values = orig_weights[old_active_indices].clone()

        # Apply change with torch.no_grad().
        # For Conv tensors, grouped views are built from permuted copies, so we
        # must restore back and copy into module buffers after each mutation.
        with torch.no_grad():
            # Zero out the group
            w_group.zero_()
            m_group.zero_()

            # Write values to new pattern positions
            for rank, dst_pos in enumerate(new_pattern):
                w_group[dst_pos] = old_values[rank]
                m_group[dst_pos] = 1.0

            m_new = restore_groups(m_flat, m_meta)
            w_new = restore_groups(w_flat, w_meta)
            module.sparse_mask.copy_(m_new.clone())
            module.int8_weights.copy_(w_new.clone())

        # Compute new loss
        with torch.no_grad():
            model.eval()
            new_output = model(test_imgs)
            new_loss = criterion(new_output, test_tgts).item()

        delta = new_loss - baseline_loss

        # Restore original state with torch.no_grad()
        with torch.no_grad():
            m_group.copy_(orig_mask)
            w_group.copy_(orig_weights)

            m_restore = restore_groups(m_flat, m_meta)
            w_restore = restore_groups(w_flat, w_meta)
            module.sparse_mask.copy_(m_restore.clone())
            module.int8_weights.copy_(w_restore.clone())

        # Verify model state is unchanged
        model_hash_after = compute_model_hash(model)
        if model_hash_before != model_hash_after:
            counters["hash_mismatches"] += 1
            continue

        if delta > 0:
            counters["candidates_positive_delta"] += 1
            if delta > best_delta:
                best_delta = delta
                best_candidate = candidate
                best_new_loss = new_loss
        else:
            counters["candidates_negative_delta"] += 1

    return best_candidate, counters


# =============================================================================
# Pattern Application (Permanent)
# =============================================================================

def apply_pattern_change(
    model: nn.Module,
    candidate: GroupCandidate,
) -> bool:
    """
    Permanently apply a pattern change to the model.

    Returns True if successful, False otherwise.
    """
    module = candidate.module
    group_idx = candidate.group_idx
    old_pattern = candidate.old_pattern
    new_pattern = candidate.new_pattern

    int8_w = module.int8_weights
    w_flat, w_meta = flatten_groups(int8_w)
    m_flat, m_meta = flatten_groups(module.sparse_mask)

    if w_flat is None or w_meta is None or m_flat is None or m_meta is None:
        return False

    w_group = w_flat[group_idx]
    m_group = m_flat[group_idx]

    # Save original state
    orig_mask = m_group.clone()
    orig_weights = w_group.clone()

    # Read old active values in order
    old_active_indices = (orig_mask > 0.5).nonzero(as_tuple=False).flatten()
    if old_active_indices.numel() != 2:
        return False
    old_values = orig_weights[old_active_indices].clone()

    # Apply change
    with torch.no_grad():
        # Zero out the group
        w_group.zero_()
        m_group.zero_()

        # Write values to new pattern positions
        for rank, dst_pos in enumerate(new_pattern):
            w_group[dst_pos] = old_values[rank]
            m_group[dst_pos] = 1.0

        m_new = restore_groups(m_flat, m_meta)
        w_new = restore_groups(w_flat, w_meta)
        module.sparse_mask.copy_(m_new.clone())
        module.int8_weights.copy_(w_new.clone())

    return True


# =============================================================================
# Model Loading (Following R1_T06.1 Pattern)
# =============================================================================

def load_int8_sparse_model(ckpt_path: str, device: str) -> Tuple[nn.Module, bool, Dict[str, torch.Tensor]]:
    """
    Load INT8 sparse model from checkpoint.

    Follows the same pattern as R1_T06.1 for consistency.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    ckpt_to_check = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    is_int8_ckpt = any("int8_weights" in k for k in ckpt_to_check.keys())

    state_dict_to_load = checkpoint.get("model_state_dict", checkpoint)

    if is_int8_ckpt:
        base_model = create_resnet20(sparsity_type="2:4", pretrained_path=None).to(device)
        base_model.eval()

        for name, module in base_model.named_modules():
            mask_key = f"{name}.sparse_mask"
            if mask_key in state_dict_to_load and hasattr(module, "register_buffer"):
                mask_tensor = state_dict_to_load[mask_key]
                if hasattr(module, "cached_mask"):
                    del module.cached_mask
                module.register_buffer("cached_mask", mask_tensor)

        model = Int8QuantizedResNet(base_model, copy_sparse_masks=True).to(device)
        filtered = {k: v for k, v in state_dict_to_load.items() if "sparse_mask" not in k}
        model.load_state_dict(filtered, strict=False)
        model.calibrate_all_layers()
        model.eval()
    else:
        base_model = create_resnet20(sparsity_type="2:4", pretrained_path=ckpt_path).to(device)
        base_model.eval()
        base_model.freeze_sparse_masks()
        model = Int8QuantizedResNet(base_model, copy_sparse_masks=True).to(device)
        model.calibrate_all_layers()
        model.eval()

    return model, is_int8_ckpt, state_dict_to_load


def reload_model_for_mode(
    is_int8_ckpt: bool,
    state_dict_to_load: Dict[str, torch.Tensor],
    ckpt_path: str,
    device: str,
) -> nn.Module:
    """Reload model for a specific mode."""
    if is_int8_ckpt:
        base_model = create_resnet20(sparsity_type="2:4", pretrained_path=None).to(device)
        base_model.eval()
        for name, module in base_model.named_modules():
            mask_key = f"{name}.sparse_mask"
            if mask_key in state_dict_to_load and hasattr(module, "register_buffer"):
                mask_tensor = state_dict_to_load[mask_key]
                if hasattr(module, "cached_mask"):
                    del module.cached_mask
                module.register_buffer("cached_mask", mask_tensor)

        model = Int8QuantizedResNet(base_model, copy_sparse_masks=True).to(device)
        filtered = {k: v for k, v in state_dict_to_load.items() if "sparse_mask" not in k}
        model.load_state_dict(filtered, strict=False)
        model.calibrate_all_layers()
        model.eval()
        return model

    base_model = create_resnet20(sparsity_type="2:4", pretrained_path=ckpt_path).to(device)
    base_model.eval()
    base_model.freeze_sparse_masks()
    model = Int8QuantizedResNet(base_model, copy_sparse_masks=True).to(device)
    model.calibrate_all_layers()
    model.eval()
    return model


def evaluate_top1(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Evaluate Top-1 accuracy."""
    model = model.to(device).eval()
    correct = total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            preds = model(images).argmax(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return 100.0 * correct / total if total else 0.0


def evaluate_top1_with_loss(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    """Evaluate Top-1 accuracy and average cross-entropy loss."""
    model = model.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            total_loss += criterion(outputs, targets).item()
            n_batches += 1
    acc = 100.0 * correct / total if total else 0.0
    avg_loss = total_loss / n_batches if n_batches else 0.0
    return acc, avg_loss


# =============================================================================
# Main Attack Loop
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="R1_T08: Improved Metadata BFA with Proxy/Apply Alignment"
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to sparse INT8 checkpoint")
    parser.add_argument("--seed", type=int, default=0, nargs="+", help="Seed(s), e.g. --seed 0 42")
    parser.add_argument("--physical-budget", type=int, default=50, help="Max number of flips")
    parser.add_argument("--calib-samples", type=int, default=256, help="Calibration samples for PTQ")
    parser.add_argument("--attack-batch-size", type=int, default=128, help="Attack batch size")
    parser.add_argument("--top-m-per-group", type=int, default=3, help="Top-M candidates per group in Stage A")
    parser.add_argument("--top-k-verify", type=int, default=100, help="Top-K candidates for Stage B verification")
    parser.add_argument("--output-dir", type=str, default="results/R1_T08", help="Output directory")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Ensure device is valid
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU")
        device = "cpu"

    seeds = args.seed if isinstance(args.seed, list) else [args.seed]

    # Load model
    print(f"\n[{now_ts()}] Loading checkpoint: {args.ckpt}")
    model, is_int8_ckpt, state_dict_to_load = load_int8_sparse_model(args.ckpt, device)
    print(f"[{now_ts()}] Checkpoint type: {'INT8' if is_int8_ckpt else 'FP32'}")

    # Count sparse parameters
    total_groups = 0
    for name, module in model.named_modules():
        if isinstance(module, Int8QuantizedConv2d) and module.sparse_mask is not None:
            n_groups = module.sparse_mask.numel() // 4
            total_groups += n_groups
    print(f"[{now_ts()}] Total sparse groups: {total_groups}")

    # Data loaders
    from train.train_utils import get_cifar10_loaders
    _, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.attack_batch_size,
        num_workers=0,
    )

    # Prepare attack batch (subset of training set)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    full_trainset = datasets.CIFAR10(
        root=os.path.join(args.data_dir, "cifar10"),
        train=True,
        download=True,
        transform=train_transform
    )
    attack_subset = Subset(full_trainset, list(range(min(args.calib_samples, len(full_trainset)))))
    attack_loader = DataLoader(
        attack_subset,
        batch_size=args.attack_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Prepare fixed verification batch
    fixed_indices = list(range(min(args.attack_batch_size, len(test_loader.dataset))))
    verify_subset = Subset(test_loader.dataset, fixed_indices)
    verify_loader = DataLoader(
        verify_subset,
        batch_size=args.attack_batch_size,
        shuffle=False,
    )

    # Get verification batch
    verify_imgs, verify_tgts = None, None
    for images, targets in verify_loader:
        verify_imgs, verify_tgts = images.to(device), targets.to(device)
        break

    criterion = nn.CrossEntropyLoss()

    # Results storage
    all_results = []

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"[{now_ts()}] Running seed {seed}/{seeds}")
        print(f"{'='*60}")

        set_all_seeds(seed)

        # Reload model for this seed
        model = reload_model_for_mode(is_int8_ckpt, state_dict_to_load, args.ckpt, device)

        # Initial evaluation
        init_acc, init_loss = evaluate_top1_with_loss(model, test_loader, device)
        print(f"[{now_ts()}] Initial Top-1: {init_acc:.2f}%, Loss: {init_loss:.6f}")

        # Attack state
        # Keep a short cooldown window for recently touched groups.
        # This matches T02.1 behavior (prevent immediate ping-pong, allow later revisits).
        exclude_groups_set: Set[Tuple[str, int]] = set()
        exclude_groups_queue: deque[Tuple[str, int]] = deque(maxlen=20)

        # Track forbidden transitions with bounded memory and deterministic eviction.
        forbidden_transitions_set: Set[Tuple[str, int, int, int]] = set()
        forbidden_transitions_queue: deque[Tuple[str, int, int, int]] = deque(maxlen=1000)

        attack_history = []
        total_attack_start = time.time()

        for flip_idx in range(args.physical_budget):
            step_start = time.time()
            print(f"\n--- Flip {flip_idx + 1}/{args.physical_budget} ---")

            # Compute gradients
            model.eval()
            model.zero_grad()
            for images, targets in attack_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                break

            # Stage A: Enumerate candidates
            candidates, stage_a_counters = enumerate_group_candidates(
                model,
                device,
                exclude_groups=exclude_groups_set,
                forbidden_transitions=forbidden_transitions_set,
                top_m_per_group=args.top_m_per_group,
                debug=(flip_idx == 0),  # Debug on first flip
            )

            if len(candidates) == 0:
                print("[Stage A] No candidates generated; stopping.")
                break

            # Compute baseline loss
            model.eval()
            with torch.no_grad():
                baseline_output = model(verify_imgs)
                baseline_loss = criterion(baseline_output, verify_tgts).item()

            print(f"[Stage A] Retained: {stage_a_counters['candidates_kept_topm']} candidates "
                  f"from {stage_a_counters['total_groups']} groups")

            # Stage B: Exact verification
            best_candidate, stage_b_counters = exact_verification_topk(
                model,
                candidates,
                verify_imgs,
                verify_tgts,
                criterion,
                baseline_loss,
                top_k=args.top_k_verify,
            )

            if best_candidate is None:
                print("[Stage B] No positive-delta candidate; stopping.")
                break

            # Evaluate accuracy before flip
            acc_before = evaluate_top1(model, test_loader, device)

            # Apply flip
            success = apply_pattern_change(model, best_candidate)
            if not success:
                print("[Error] Failed to apply pattern change!")
                break

            # Update tracking
            group_key = (best_candidate.layer_name, best_candidate.group_idx)

            if group_key not in exclude_groups_set:
                if len(exclude_groups_queue) == exclude_groups_queue.maxlen:
                    popped_group = exclude_groups_queue.popleft()
                    exclude_groups_set.discard(popped_group)
                exclude_groups_queue.append(group_key)
                exclude_groups_set.add(group_key)

            transition_key = (
                best_candidate.layer_name,
                best_candidate.group_idx,
                best_candidate.old_code,
                best_candidate.new_code
            )

            reverse_transition = (
                best_candidate.layer_name,
                best_candidate.group_idx,
                best_candidate.new_code,
                best_candidate.old_code,
            )

            for t_key in (transition_key, reverse_transition):
                if t_key in forbidden_transitions_set:
                    continue
                if len(forbidden_transitions_queue) == forbidden_transitions_queue.maxlen:
                    popped_t = forbidden_transitions_queue.popleft()
                    forbidden_transitions_set.discard(popped_t)
                forbidden_transitions_queue.append(t_key)
                forbidden_transitions_set.add(t_key)

            # Evaluate after flip
            acc_after, loss_after = evaluate_top1_with_loss(model, test_loader, device)
            step_time = time.time() - step_start

            print(f"[Flip] layer={best_candidate.layer_name} "
                  f"group={best_candidate.group_idx} "
                  f"pattern={best_candidate.old_pattern}->{best_candidate.new_pattern}")
            print(f"[Result] Acc: {acc_before:.2f}% -> {acc_after:.2f}% (drop: {acc_before - acc_after:.2f}%) "
                  f"Loss: {loss_after:.6f} t={step_time:.2f}s")

            attack_history.append({
                "flip": flip_idx + 1,
                "layer": best_candidate.layer_name,
                "group": best_candidate.group_idx,
                "old_pattern": best_candidate.old_pattern,
                "new_pattern": best_candidate.new_pattern,
                "acc_before": acc_before,
                "acc_after": acc_after,
                "acc_drop": acc_before - acc_after,
                "loss_after": loss_after,
                "search_time": step_time,
            })

        # Final evaluation
        final_acc, final_loss = evaluate_top1_with_loss(model, test_loader, device)
        total_attack_time = time.time() - total_attack_start
        num_flips = len(attack_history)
        avg_step_time = total_attack_time / num_flips if num_flips > 0 else 0.0
        print(f"\n[{now_ts()}] Seed {seed} finished. Final Top-1: {final_acc:.2f}%, Loss: {final_loss:.6f}")
        print(f"[Timing] Total: {total_attack_time:.2f}s, Avg per step: {avg_step_time:.2f}s")

        all_results.append({
            "seed": seed,
            "initial_acc": init_acc,
            "initial_loss": init_loss,
            "final_acc": final_acc,
            "final_loss": final_loss,
            "acc_drop": init_acc - final_acc,
            "loss_increase": final_loss - init_loss,
            "num_flips": num_flips,
            "total_search_time": total_attack_time,
            "avg_search_time_per_step": avg_step_time,
            "attack_history": attack_history,
        })

    # Save results
    results_path = os.path.join(args.output_dir, "results.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "initial_acc", "final_acc", "num_flips"])
        for r in all_results:
            writer.writerow([r["seed"], f"{r['initial_acc']:.2f}",
                           f"{r['final_acc']:.2f}", r["num_flips"]])

    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[{now_ts()}] Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
