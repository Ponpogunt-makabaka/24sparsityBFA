#!/usr/bin/env python3
"""
R1_T10: Group-First Search Heuristic for Metadata BFA

Core algorithm:
  Step 1 (Group Proxy Scoring): For every 2:4 group (4 elements), compute
      group_score = sum(abs(grad[0..3]))
  Step 2 (Global Top-K Groups): Sort all groups by group_score, select Top-K.
  Step 3 (Candidate Generation): ONLY within selected Top-K groups, generate
      valid index_1bit metadata flip candidates following strict NCA constraints.
  Step 4 (Exact Verification): Forward-pass evaluation on candidates, select
      the one with maximum exact loss increase.

Compared to T08 (which enumerates ALL groups then does Top-M per group + global
Top-K verification), T10 pre-filters groups by gradient magnitude BEFORE
generating any candidates, drastically reducing candidate enumeration cost.
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

# Ensure project root is in sys.path
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.factory import create_resnet20
from train.ptq_convert import Int8QuantizedConv2d, Int8QuantizedLinear, Int8QuantizedResNet


# =============================================================================
# Constants and Pattern Definitions (Index Encoding)
# =============================================================================

ALL_2OF4_PATTERNS = [
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
]


def encode_index_to_4bit(i: int, j: int) -> int:
    return (j << 2) | i


def decode_4bit_to_index(code: int) -> Tuple[int, int]:
    i = code & 0x3
    j = (code >> 2) & 0x3
    return (i, j)


def pattern_to_code(pattern: Tuple[int, int]) -> int:
    return encode_index_to_4bit(pattern[0], pattern[1])


def code_to_pattern(code: int) -> Optional[Tuple[int, int]]:
    i, j = decode_4bit_to_index(code)
    if i == j:
        return None
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

def get_current_pattern(mask_group: torch.Tensor) -> Optional[Tuple[int, int]]:
    active_indices = (mask_group > 0.5).nonzero(as_tuple=False).flatten()
    if active_indices.numel() != 2:
        return None
    return (int(active_indices[0].item()), int(active_indices[1].item()))


def flatten_groups(tensor: torch.Tensor, group_size: int = 4) -> Tuple[Optional[torch.Tensor], Optional[Tuple]]:
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
    sparse_layers = []
    for name, module in model.named_modules():
        if isinstance(module, Int8QuantizedConv2d) and module.sparse_mask is not None:
            sparse_layers.append((name, module))
    return sparse_layers


# =============================================================================
# T10 Step 1+2: Group Proxy Scoring and Global Top-K Group Selection
# =============================================================================

def select_topk_groups(
    model: nn.Module,
    topk_groups: int,
    exclude_groups: Optional[Set[Tuple[str, int]]] = None,
) -> List[Tuple[str, nn.Module, int, float]]:
    """
    Step 1: For every 2:4 group, compute group_score = sum(abs(grad[0..3])).
    Step 2: Sort globally and return Top-K highest-scoring groups.

    Returns list of (layer_name, module, group_idx, group_score).
    """
    if exclude_groups is None:
        exclude_groups = set()

    scored_groups: List[Tuple[float, str, nn.Module, int]] = []

    for layer_name, module in get_sparse_layers(model):
        if module.weight.grad is None:
            continue

        grad = module.weight.grad.data
        g_flat, _ = flatten_groups(grad)
        m_flat, _ = flatten_groups(module.sparse_mask)

        if g_flat is None or m_flat is None:
            continue

        num_groups = int(g_flat.shape[0])

        # Vectorized group scoring: sum(abs(grad)) per group
        group_scores = g_flat.abs().sum(dim=1)  # shape: (num_groups,)

        for g_idx in range(num_groups):
            if (layer_name, g_idx) in exclude_groups:
                continue

            # Verify this group has a valid 2:4 pattern
            m_group = m_flat[g_idx]
            active = (m_group > 0.5).sum().item()
            if active != 2:
                continue

            score = float(group_scores[g_idx].item())
            scored_groups.append((score, layer_name, module, g_idx))

    # Sort by score descending, then layer_name/group_idx for determinism
    scored_groups.sort(key=lambda x: (-x[0], x[1], x[3]))

    # Select Top-K
    topk = scored_groups[:topk_groups]

    return [(name, mod, gidx, sc) for sc, name, mod, gidx in topk]


# =============================================================================
# T10 Step 3: Candidate Generation (ONLY within Top-K groups)
# =============================================================================

def generate_candidates_for_groups(
    selected_groups: List[Tuple[str, nn.Module, int, float]],
    forbidden_transitions: Optional[Set[Tuple[str, int, int, int]]] = None,
) -> Tuple[List[GroupCandidate], Dict[str, int]]:
    """
    Step 3: For each selected Top-K group, generate valid index_1bit
    metadata flip candidates following strict NCA constraints.

    Uses the same proxy/apply alignment as T08.
    """
    if forbidden_transitions is None:
        forbidden_transitions = set()

    counters = {
        "groups_selected": len(selected_groups),
        "candidates_considered": 0,
        "candidates_valid": 0,
        "candidates_rejected_collision": 0,
        "candidates_rejected_no_change": 0,
        "candidates_skipped_forbidden": 0,
    }

    all_candidates: List[GroupCandidate] = []

    for layer_name, module, g_idx, _ in selected_groups:
        int8_w = module.int8_weights
        mask = module.sparse_mask
        scale = float(module.scale.item())

        grad = module.weight.grad.data if module.weight.grad is not None else None
        if grad is None:
            continue

        g_flat, _ = flatten_groups(grad)
        w_flat, _ = flatten_groups(int8_w)
        m_flat, _ = flatten_groups(mask)

        if g_flat is None or w_flat is None or m_flat is None:
            continue

        m_group = m_flat[g_idx]
        current_pattern = get_current_pattern(m_group)
        if current_pattern is None:
            continue

        current_code = pattern_to_code(current_pattern)
        grad_group = g_flat[g_idx].float()
        w_group = w_flat[g_idx]

        old_mask = (m_group > 0.5).to(torch.float32)
        with torch.no_grad():
            w_tilde_current = w_group.float() * scale * old_mask

        old_active_indices = (old_mask > 0.5).nonzero(as_tuple=False).flatten()
        if old_active_indices.numel() != 2:
            continue
        old_values = w_group[old_active_indices].clone()

        # Enumerate 1-bit reachable transitions
        for bit_pos in range(4):
            candidate_code = current_code ^ (1 << bit_pos)
            transition_key = (layer_name, int(g_idx), current_code, candidate_code)
            if transition_key in forbidden_transitions:
                counters["candidates_skipped_forbidden"] += 1
                continue

            counters["candidates_considered"] += 1

            candidate_pattern = code_to_pattern(candidate_code)
            if candidate_pattern is None:
                counters["candidates_rejected_collision"] += 1
                continue
            if candidate_pattern == current_pattern:
                counters["candidates_rejected_no_change"] += 1
                continue

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
            all_candidates.append(
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

    # Sort by proxy_score descending with deterministic tie-break
    all_candidates.sort(
        key=lambda c: (-c.proxy_score, c.layer_name, c.group_idx, c.new_code, c.flipped_bit)
    )

    return all_candidates, counters


# =============================================================================
# Tensor Hash Utilities for State Verification
# =============================================================================

def compute_model_hash(model: nn.Module) -> int:
    hash_value = 0
    for name, p in sorted(model.named_parameters()):
        if p.data is not None:
            hash_value ^= hash((name, p.data.shape, p.data.sum().item(), p.data.abs().sum().item()))
    for name, b in sorted(model.named_buffers()):
        if b.data is not None:
            hash_value ^= hash((name, b.data.shape, b.data.sum().item(), b.data.abs().sum().item()))
    return hash_value


# =============================================================================
# T10 Step 4: Exact Verification (same as T08 Stage B)
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
    Step 4: Exact verification with Save-Apply-Restore pattern.
    Same logic as T08 Stage B.
    """
    counters = {
        "candidates_tested": 0,
        "candidates_positive_delta": 0,
        "candidates_negative_delta": 0,
        "hash_mismatches": 0,
    }

    best_candidate = None
    best_delta = 0.0

    model_hash_before = compute_model_hash(model)

    for candidate in candidates[:top_k]:
        counters["candidates_tested"] += 1

        module = candidate.module
        group_idx = candidate.group_idx
        new_pattern = candidate.new_pattern

        int8_w = module.int8_weights
        w_flat, w_meta = flatten_groups(int8_w)
        m_flat, m_meta = flatten_groups(module.sparse_mask)

        if w_flat is None or w_meta is None or m_flat is None or m_meta is None:
            continue

        w_group = w_flat[group_idx]
        m_group = m_flat[group_idx]

        orig_mask = m_group.clone()
        orig_weights = w_group.clone()

        old_active_indices = (orig_mask > 0.5).nonzero(as_tuple=False).flatten()
        if old_active_indices.numel() != 2:
            continue
        old_values = orig_weights[old_active_indices].clone()

        with torch.no_grad():
            w_group.zero_()
            m_group.zero_()
            for rank, dst_pos in enumerate(new_pattern):
                w_group[dst_pos] = old_values[rank]
                m_group[dst_pos] = 1.0
            m_new = restore_groups(m_flat, m_meta)
            w_new = restore_groups(w_flat, w_meta)
            module.sparse_mask.copy_(m_new.clone())
            module.int8_weights.copy_(w_new.clone())

        with torch.no_grad():
            model.eval()
            new_output = model(test_imgs)
            new_loss = criterion(new_output, test_tgts).item()

        delta = new_loss - baseline_loss

        with torch.no_grad():
            m_group.copy_(orig_mask)
            w_group.copy_(orig_weights)
            m_restore = restore_groups(m_flat, m_meta)
            w_restore = restore_groups(w_flat, w_meta)
            module.sparse_mask.copy_(m_restore.clone())
            module.int8_weights.copy_(w_restore.clone())

        model_hash_after = compute_model_hash(model)
        if model_hash_before != model_hash_after:
            counters["hash_mismatches"] += 1
            continue

        if delta > 0:
            counters["candidates_positive_delta"] += 1
            if delta > best_delta:
                best_delta = delta
                best_candidate = candidate
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
    module = candidate.module
    group_idx = candidate.group_idx
    new_pattern = candidate.new_pattern

    int8_w = module.int8_weights
    w_flat, w_meta = flatten_groups(int8_w)
    m_flat, m_meta = flatten_groups(module.sparse_mask)

    if w_flat is None or w_meta is None or m_flat is None or m_meta is None:
        return False

    w_group = w_flat[group_idx]
    m_group = m_flat[group_idx]

    orig_mask = m_group.clone()
    orig_weights = w_group.clone()

    old_active_indices = (orig_mask > 0.5).nonzero(as_tuple=False).flatten()
    if old_active_indices.numel() != 2:
        return False
    old_values = orig_weights[old_active_indices].clone()

    with torch.no_grad():
        w_group.zero_()
        m_group.zero_()
        for rank, dst_pos in enumerate(new_pattern):
            w_group[dst_pos] = old_values[rank]
            m_group[dst_pos] = 1.0
        m_new = restore_groups(m_flat, m_meta)
        w_new = restore_groups(w_flat, w_meta)
        module.sparse_mask.copy_(m_new.clone())
        module.int8_weights.copy_(w_new.clone())

    return True


# =============================================================================
# Model Loading (Same as T08)
# =============================================================================

def load_int8_sparse_model(ckpt_path: str, device: str) -> Tuple[nn.Module, bool, Dict[str, torch.Tensor]]:
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
    model = model.to(device).eval()
    correct = total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            preds = model(images).argmax(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return 100.0 * correct / total if total else 0.0


# =============================================================================
# Main Attack Loop
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="R1_T10: Group-First Search Heuristic for Metadata BFA"
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to sparse INT8 checkpoint")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--n-iter", type=int, default=50, help="Max number of flips")
    parser.add_argument("--attack-sample-size", type=int, default=256, help="Attack batch sample size")
    parser.add_argument("--attack-batch-size", type=int, default=128, help="Attack batch size for forward pass")
    parser.add_argument("--topk", type=int, default=64, help="Top-K groups to select in Step 2")
    parser.add_argument("--verify-topk", type=int, default=100, help="Top-K candidates for exact verification in Step 4")
    parser.add_argument("--output-dir", type=str, default="results/R1_T10", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU")
        device = "cpu"

    set_all_seeds(args.seed)

    # Load model
    print(f"\n[{now_ts()}] R1_T10: Group-First Search Heuristic")
    print(f"[{now_ts()}] Loading checkpoint: {args.ckpt}")
    model, is_int8_ckpt, state_dict_to_load = load_int8_sparse_model(args.ckpt, device)
    print(f"[{now_ts()}] Checkpoint type: {'INT8' if is_int8_ckpt else 'FP32'}")

    # Count sparse parameters
    total_groups = 0
    for name, module in model.named_modules():
        if isinstance(module, Int8QuantizedConv2d) and module.sparse_mask is not None:
            n_groups = module.sparse_mask.numel() // 4
            total_groups += n_groups
    print(f"[{now_ts()}] Total sparse groups: {total_groups}")
    print(f"[{now_ts()}] Top-K groups per iteration: {args.topk}")

    # Data loaders - test set with NO augmentation (validation transforms)
    from train.train_utils import get_cifar10_loaders
    _, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.attack_batch_size,
        num_workers=0,
    )

    # Attack batch: subset of TEST set with validation transforms (no augmentation)
    # This ensures deterministic, augmentation-free attack data
    test_dataset = test_loader.dataset
    attack_indices = list(range(min(args.attack_sample_size, len(test_dataset))))
    attack_subset = Subset(test_dataset, attack_indices)
    attack_loader = DataLoader(
        attack_subset,
        batch_size=args.attack_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Verification batch (first batch_size samples from test set)
    verify_indices = list(range(min(args.attack_batch_size, len(test_dataset))))
    verify_subset = Subset(test_dataset, verify_indices)
    verify_loader = DataLoader(
        verify_subset,
        batch_size=args.attack_batch_size,
        shuffle=False,
    )

    verify_imgs, verify_tgts = None, None
    for images, targets in verify_loader:
        verify_imgs, verify_tgts = images.to(device), targets.to(device)
        break

    criterion = nn.CrossEntropyLoss()

    # Reload model for this seed
    model = reload_model_for_mode(is_int8_ckpt, state_dict_to_load, args.ckpt, device)

    # Initial evaluation
    init_acc = evaluate_top1(model, test_loader, device)
    print(f"[{now_ts()}] Initial Top-1: {init_acc:.2f}%")

    # Attack state
    exclude_groups_set: Set[Tuple[str, int]] = set()
    exclude_groups_queue: deque[Tuple[str, int]] = deque(maxlen=20)

    forbidden_transitions_set: Set[Tuple[str, int, int, int]] = set()
    forbidden_transitions_queue: deque[Tuple[str, int, int, int]] = deque(maxlen=1000)

    attack_history = []
    search_times = []

    for flip_idx in range(args.n_iter):
        print(f"\n--- Flip {flip_idx + 1}/{args.n_iter} ---")

        iter_start = time.time()

        # ---- Compute gradients ----
        model.eval()
        model.zero_grad()
        for images, targets in attack_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            break

        # ---- Step 1+2: Group Proxy Scoring + Global Top-K Selection ----
        step12_start = time.time()
        selected_groups = select_topk_groups(
            model,
            topk_groups=args.topk,
            exclude_groups=exclude_groups_set,
        )
        step12_time = time.time() - step12_start

        if len(selected_groups) == 0:
            print("[Step 1+2] No groups available; stopping.")
            break

        print(f"[Step 1+2] Selected {len(selected_groups)} groups "
              f"(top score={selected_groups[0][3]:.6f}, "
              f"min score={selected_groups[-1][3]:.6f}) "
              f"[{step12_time:.3f}s]")

        # ---- Step 3: Candidate Generation within Top-K groups ----
        step3_start = time.time()
        candidates, gen_counters = generate_candidates_for_groups(
            selected_groups,
            forbidden_transitions=forbidden_transitions_set,
        )
        step3_time = time.time() - step3_start

        # Free gradients after candidate generation
        for _, module in get_sparse_layers(model):
            if module.weight.grad is not None:
                module.weight.grad = None

        if len(candidates) == 0:
            print("[Step 3] No valid candidates generated; stopping.")
            break

        print(f"[Step 3] Generated {gen_counters['candidates_valid']} valid candidates "
              f"from {gen_counters['groups_selected']} groups "
              f"(collisions={gen_counters['candidates_rejected_collision']}, "
              f"forbidden={gen_counters['candidates_skipped_forbidden']}) "
              f"[{step3_time:.3f}s]")

        # ---- Compute baseline loss for verification ----
        model.eval()
        with torch.no_grad():
            baseline_output = model(verify_imgs)
            baseline_loss = criterion(baseline_output, verify_tgts).item()

        # ---- Step 4: Exact Verification ----
        step4_start = time.time()
        best_candidate, verify_counters = exact_verification_topk(
            model,
            candidates,
            verify_imgs,
            verify_tgts,
            criterion,
            baseline_loss,
            top_k=args.verify_topk,
        )
        step4_time = time.time() - step4_start

        iter_time = time.time() - iter_start
        search_times.append(iter_time)

        print(f"[Step 4] Tested {verify_counters['candidates_tested']} candidates, "
              f"positive={verify_counters['candidates_positive_delta']} "
              f"[{step4_time:.3f}s]")
        print(f"[Timing] Total search: {iter_time:.3f}s "
              f"(Step1+2={step12_time:.3f}s, Step3={step3_time:.3f}s, Step4={step4_time:.3f}s)")

        if best_candidate is None:
            print("[Step 4] No positive-delta candidate; stopping.")
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
        acc_after = evaluate_top1(model, test_loader, device)

        print(f"[Flip] layer={best_candidate.layer_name} "
              f"group={best_candidate.group_idx} "
              f"pattern={best_candidate.old_pattern}->{best_candidate.new_pattern}")
        print(f"[Result] Acc: {acc_before:.2f}% -> {acc_after:.2f}% "
              f"(drop: {acc_before - acc_after:.2f}%)")

        attack_history.append({
            "flip": flip_idx + 1,
            "layer": best_candidate.layer_name,
            "group": best_candidate.group_idx,
            "old_pattern": list(best_candidate.old_pattern),
            "new_pattern": list(best_candidate.new_pattern),
            "acc_before": acc_before,
            "acc_after": acc_after,
            "acc_drop": acc_before - acc_after,
            "search_time": iter_time,
            "step12_time": step12_time,
            "step3_time": step3_time,
            "step4_time": step4_time,
        })

    # Final evaluation
    final_acc = evaluate_top1(model, test_loader, device)
    avg_search_time = sum(search_times) / len(search_times) if search_times else 0.0

    print(f"\n{'='*60}")
    print(f"[{now_ts()}] T10 Group-First Search Complete")
    print(f"{'='*60}")
    print(f"  Seed:                {args.seed}")
    print(f"  Top-K Groups:        {args.topk}")
    print(f"  Initial Acc:         {init_acc:.2f}%")
    print(f"  Final Acc:           {final_acc:.2f}%")
    print(f"  Acc Drop:            {init_acc - final_acc:.2f}%")
    print(f"  Flips Used:          {len(attack_history)}/{args.n_iter}")
    print(f"  Avg Search Time:     {avg_search_time:.3f}s/step")
    print(f"  Total Search Time:   {sum(search_times):.3f}s")
    print(f"{'='*60}")

    # Save results
    result = {
        "method": "T10_group_first",
        "seed": args.seed,
        "topk_groups": args.topk,
        "n_iter": args.n_iter,
        "attack_sample_size": args.attack_sample_size,
        "verify_topk": args.verify_topk,
        "initial_acc": init_acc,
        "final_acc": final_acc,
        "acc_drop": init_acc - final_acc,
        "num_flips": len(attack_history),
        "avg_search_time_per_step": avg_search_time,
        "total_search_time": sum(search_times),
        "attack_history": attack_history,
    }

    results_csv_path = os.path.join(args.output_dir, "results.csv")
    with open(results_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "topk", "initial_acc", "final_acc", "acc_drop",
                         "num_flips", "avg_search_time"])
        writer.writerow([
            args.seed, args.topk,
            f"{init_acc:.2f}", f"{final_acc:.2f}", f"{init_acc - final_acc:.2f}",
            len(attack_history), f"{avg_search_time:.4f}"
        ])

    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[{now_ts()}] Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
