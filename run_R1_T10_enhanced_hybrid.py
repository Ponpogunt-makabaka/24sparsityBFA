#!/usr/bin/env python3
"""
R1_T10_Enhanced: Hybrid Two-Stage Search Heuristic for Metadata BFA

Fixes the directionality and reachability flaws of original T10 by combining:
  - T10's efficient group-level pre-filtering (coarse stage)
  - T08's precise candidate-level directional proxy scoring (fine stage)

5-Step Pipeline:
  Step 1 (Coarse Group Pre-filter): group_score = sum(abs(grad[0..3])).
      Select Top-N groups globally (N=1000).
  Step 2 (Candidate Generation): Within Top-N groups only, generate valid
      index_1bit NCA candidates.
  Step 3 (Fine Candidate Proxy): For each candidate, compute directional
      proxy_score = grad · (w_tilde_new - w_tilde_old).
  Step 4 (Global Candidate Top-K): Sort candidates by proxy_score, select
      Top-K globally (K=64).
  Step 5 (Exact Forward Verification): Forward-pass evaluation on K candidates,
      select the one with maximum exact loss increase.

Key difference from T10:
  T10:          64 groups → ~170 candidates → verify 100
  T10_Enhanced: 1000 groups → ~2700 candidates → proxy-rank → Top-64 → verify 64
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.factory import create_resnet20
from train.ptq_convert import Int8QuantizedConv2d, Int8QuantizedResNet


# =============================================================================
# Index Encoding
# =============================================================================

def encode_index_to_4bit(i: int, j: int) -> int:
    return (j << 2) | i


def decode_4bit_to_index(code: int) -> Tuple[int, int]:
    return (code & 0x3, (code >> 2) & 0x3)


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


def get_current_pattern(mask_group: torch.Tensor) -> Optional[Tuple[int, int]]:
    active_indices = (mask_group > 0.5).nonzero(as_tuple=False).flatten()
    if active_indices.numel() != 2:
        return None
    return (int(active_indices[0].item()), int(active_indices[1].item()))


def flatten_groups(tensor: torch.Tensor, group_size: int = 4) -> Tuple[Optional[torch.Tensor], Optional[Tuple]]:
    if tensor.dim() == 4:
        t_perm = tensor.permute(0, 2, 3, 1).contiguous()
        if t_perm.numel() % group_size != 0:
            return None, None
        flat = t_perm.view(-1, group_size)
        meta = ("conv", tuple(tensor.shape), tuple(t_perm.shape))
        return flat, meta
    if tensor.dim() == 2:
        if tensor.numel() % group_size != 0:
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


def get_sparse_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    sparse_layers = []
    for name, module in model.named_modules():
        if isinstance(module, Int8QuantizedConv2d) and module.sparse_mask is not None:
            sparse_layers.append((name, module))
    return sparse_layers


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
# Step 1: Coarse Group Pre-filter
# =============================================================================

def step1_coarse_group_prefilter(
    model: nn.Module,
    coarse_n: int,
    exclude_groups: Optional[Set[Tuple[str, int]]] = None,
) -> List[Tuple[str, nn.Module, int, float]]:
    """
    For every 2:4 group, compute group_score = sum(abs(grad[0..3])).
    Sort globally and return Top-N highest-scoring groups.
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
        group_scores = g_flat.abs().sum(dim=1)

        for g_idx in range(num_groups):
            if (layer_name, g_idx) in exclude_groups:
                continue
            m_group = m_flat[g_idx]
            if (m_group > 0.5).sum().item() != 2:
                continue
            score = float(group_scores[g_idx].item())
            scored_groups.append((score, layer_name, module, g_idx))

    scored_groups.sort(key=lambda x: (-x[0], x[1], x[3]))
    topn = scored_groups[:coarse_n]
    return [(name, mod, gidx, sc) for sc, name, mod, gidx in topn]


# =============================================================================
# Step 2+3: Candidate Generation + Fine Directional Proxy Scoring
# =============================================================================

def step23_generate_and_score_candidates(
    selected_groups: List[Tuple[str, nn.Module, int, float]],
    forbidden_transitions: Optional[Set[Tuple[str, int, int, int]]] = None,
) -> Tuple[List[GroupCandidate], Dict[str, int]]:
    """
    Step 2: Within Top-N groups, generate valid index_1bit NCA candidates.
    Step 3: Compute directional proxy: proxy_score = grad · (w_tilde_new - w_tilde_old).

    Uses proxy/apply alignment (same as T08): read INT8 values at old active
    positions, place into new pattern positions, reconstruct dense.
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
        grad = module.weight.grad.data if module.weight.grad is not None else None
        if grad is None:
            continue

        int8_w = module.int8_weights
        mask = module.sparse_mask
        scale = float(module.scale.item())

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

    return all_candidates, counters


# =============================================================================
# Step 4: Global Candidate Top-K Selection
# =============================================================================

def step4_select_topk_candidates(
    candidates: List[GroupCandidate],
    topk: int,
) -> List[GroupCandidate]:
    """Sort candidates by directional proxy_score descending, return Top-K."""
    candidates.sort(
        key=lambda c: (-c.proxy_score, c.layer_name, c.group_idx, c.new_code, c.flipped_bit)
    )
    return candidates[:topk]


# =============================================================================
# Step 5: Exact Forward Verification
# =============================================================================

def step5_exact_verification(
    model: nn.Module,
    candidates: List[GroupCandidate],
    test_imgs: torch.Tensor,
    test_tgts: torch.Tensor,
    criterion: nn.Module,
    baseline_loss: float,
) -> Tuple[Optional[GroupCandidate], Dict[str, int]]:
    """
    Exact verification with Save-Apply-Restore pattern.
    Tests ALL provided candidates (already pre-filtered to Top-K).
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

    for candidate in candidates:
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

        # Apply
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

        # Evaluate
        with torch.no_grad():
            model.eval()
            new_output = model(test_imgs)
            new_loss = criterion(new_output, test_tgts).item()

        delta = new_loss - baseline_loss

        # Restore
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

def apply_pattern_change(model: nn.Module, candidate: GroupCandidate) -> bool:
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
# Model Loading
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
        description="R1_T10_Enhanced: Hybrid Two-Stage Search for Metadata BFA"
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-iter", type=int, default=50, help="Max number of flips")
    parser.add_argument("--attack-sample-size", type=int, default=256)
    parser.add_argument("--attack-batch-size", type=int, default=128)
    parser.add_argument("--coarse-groups", type=int, default=1000,
                        help="Step 1: Top-N coarse group pre-filter size")
    parser.add_argument("--topk", type=int, default=64,
                        help="Step 4: Top-K candidates for exact verification")
    parser.add_argument("--output-dir", type=str, default="results/R1_T10_enhanced")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU")
        device = "cpu"

    set_all_seeds(args.seed)

    print(f"\n[{now_ts()}] R1_T10_Enhanced: Hybrid Two-Stage Search")
    print(f"[{now_ts()}] Loading checkpoint: {args.ckpt}")
    model, is_int8_ckpt, state_dict_to_load = load_int8_sparse_model(args.ckpt, device)
    print(f"[{now_ts()}] Checkpoint type: {'INT8' if is_int8_ckpt else 'FP32'}")

    total_groups = 0
    for name, module in model.named_modules():
        if isinstance(module, Int8QuantizedConv2d) and module.sparse_mask is not None:
            total_groups += module.sparse_mask.numel() // 4
    print(f"[{now_ts()}] Total sparse groups: {total_groups}")
    print(f"[{now_ts()}] Coarse filter N={args.coarse_groups}, Fine Top-K={args.topk}")

    # Data loaders - validation transforms only (no augmentation)
    from train.train_utils import get_cifar10_loaders
    _, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.attack_batch_size,
        num_workers=0,
    )

    test_dataset = test_loader.dataset
    attack_indices = list(range(min(args.attack_sample_size, len(test_dataset))))
    attack_subset = Subset(test_dataset, attack_indices)
    attack_loader = DataLoader(
        attack_subset,
        batch_size=args.attack_batch_size,
        shuffle=False,
        num_workers=0,
    )

    verify_indices = list(range(min(args.attack_batch_size, len(test_dataset))))
    verify_subset = Subset(test_dataset, verify_indices)
    verify_loader = DataLoader(verify_subset, batch_size=args.attack_batch_size, shuffle=False)
    verify_imgs, verify_tgts = None, None
    for images, targets in verify_loader:
        verify_imgs, verify_tgts = images.to(device), targets.to(device)
        break

    criterion = nn.CrossEntropyLoss()

    model = reload_model_for_mode(is_int8_ckpt, state_dict_to_load, args.ckpt, device)
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

        # ---- Gradient computation ----
        model.eval()
        model.zero_grad()
        for images, targets in attack_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            break

        # ---- Step 1: Coarse Group Pre-filter ----
        t1 = time.time()
        selected_groups = step1_coarse_group_prefilter(
            model,
            coarse_n=args.coarse_groups,
            exclude_groups=exclude_groups_set,
        )
        step1_time = time.time() - t1

        if len(selected_groups) == 0:
            print("[Step 1] No groups available; stopping.")
            break

        print(f"[Step 1] Coarse filter: {len(selected_groups)} groups "
              f"(top={selected_groups[0][3]:.6f}, "
              f"min={selected_groups[-1][3]:.6f}) [{step1_time:.3f}s]")

        # ---- Step 2+3: Candidate Generation + Fine Proxy Scoring ----
        t23 = time.time()
        all_candidates, gen_counters = step23_generate_and_score_candidates(
            selected_groups,
            forbidden_transitions=forbidden_transitions_set,
        )
        step23_time = time.time() - t23

        # Free gradients
        for _, module in get_sparse_layers(model):
            if module.weight.grad is not None:
                module.weight.grad = None

        if len(all_candidates) == 0:
            print("[Step 2+3] No valid candidates; stopping.")
            break

        print(f"[Step 2+3] {gen_counters['candidates_valid']} candidates from "
              f"{gen_counters['groups_selected']} groups "
              f"(collisions={gen_counters['candidates_rejected_collision']}, "
              f"forbidden={gen_counters['candidates_skipped_forbidden']}) "
              f"[{step23_time:.3f}s]")

        # ---- Step 4: Global Candidate Top-K by proxy_score ----
        t4 = time.time()
        topk_candidates = step4_select_topk_candidates(all_candidates, args.topk)
        step4_time = time.time() - t4

        print(f"[Step 4] Top-{args.topk} candidates selected "
              f"(best proxy={topk_candidates[0].proxy_score:.6f}, "
              f"worst proxy={topk_candidates[-1].proxy_score:.6f}) "
              f"[{step4_time:.4f}s]")

        # ---- Baseline loss ----
        model.eval()
        with torch.no_grad():
            baseline_output = model(verify_imgs)
            baseline_loss = criterion(baseline_output, verify_tgts).item()

        # ---- Step 5: Exact Forward Verification ----
        t5 = time.time()
        best_candidate, verify_counters = step5_exact_verification(
            model, topk_candidates, verify_imgs, verify_tgts,
            criterion, baseline_loss,
        )
        step5_time = time.time() - t5

        iter_time = time.time() - iter_start
        search_times.append(iter_time)

        print(f"[Step 5] Verified {verify_counters['candidates_tested']} candidates, "
              f"positive={verify_counters['candidates_positive_delta']} "
              f"[{step5_time:.3f}s]")
        print(f"[Timing] {iter_time:.3f}s total "
              f"(S1={step1_time:.3f}s, S2+3={step23_time:.3f}s, "
              f"S4={step4_time:.4f}s, S5={step5_time:.3f}s)")

        if best_candidate is None:
            print("[Step 5] No positive-delta candidate; stopping.")
            break

        acc_before = evaluate_top1(model, test_loader, device)

        success = apply_pattern_change(model, best_candidate)
        if not success:
            print("[Error] Failed to apply pattern change!")
            break

        # Update tracking
        group_key = (best_candidate.layer_name, best_candidate.group_idx)
        if group_key not in exclude_groups_set:
            if len(exclude_groups_queue) == exclude_groups_queue.maxlen:
                popped = exclude_groups_queue.popleft()
                exclude_groups_set.discard(popped)
            exclude_groups_queue.append(group_key)
            exclude_groups_set.add(group_key)

        transition_key = (
            best_candidate.layer_name, best_candidate.group_idx,
            best_candidate.old_code, best_candidate.new_code,
        )
        reverse_transition = (
            best_candidate.layer_name, best_candidate.group_idx,
            best_candidate.new_code, best_candidate.old_code,
        )
        for t_key in (transition_key, reverse_transition):
            if t_key in forbidden_transitions_set:
                continue
            if len(forbidden_transitions_queue) == forbidden_transitions_queue.maxlen:
                popped = forbidden_transitions_queue.popleft()
                forbidden_transitions_set.discard(popped)
            forbidden_transitions_queue.append(t_key)
            forbidden_transitions_set.add(t_key)

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
            "proxy_score": best_candidate.proxy_score,
            "acc_before": acc_before,
            "acc_after": acc_after,
            "acc_drop": acc_before - acc_after,
            "search_time": iter_time,
            "step1_time": step1_time,
            "step23_time": step23_time,
            "step4_time": step4_time,
            "step5_time": step5_time,
        })

    # Final results
    final_acc = evaluate_top1(model, test_loader, device)
    avg_search_time = sum(search_times) / len(search_times) if search_times else 0.0

    print(f"\n{'='*60}")
    print(f"[{now_ts()}] T10_Enhanced Hybrid Two-Stage Complete")
    print(f"{'='*60}")
    print(f"  Seed:                {args.seed}")
    print(f"  Coarse Groups (N):   {args.coarse_groups}")
    print(f"  Fine Top-K:          {args.topk}")
    print(f"  Initial Acc:         {init_acc:.2f}%")
    print(f"  Final Acc:           {final_acc:.2f}%")
    print(f"  Acc Drop:            {init_acc - final_acc:.2f}%")
    print(f"  Flips Used:          {len(attack_history)}/{args.n_iter}")
    print(f"  Avg Search Time:     {avg_search_time:.3f}s/step")
    print(f"  Total Search Time:   {sum(search_times):.3f}s")
    print(f"{'='*60}")

    result = {
        "method": "T10_enhanced_hybrid",
        "seed": args.seed,
        "coarse_groups": args.coarse_groups,
        "topk": args.topk,
        "n_iter": args.n_iter,
        "attack_sample_size": args.attack_sample_size,
        "initial_acc": init_acc,
        "final_acc": final_acc,
        "acc_drop": init_acc - final_acc,
        "num_flips": len(attack_history),
        "avg_search_time_per_step": avg_search_time,
        "total_search_time": sum(search_times),
        "attack_history": attack_history,
    }

    csv_path = os.path.join(args.output_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "coarse_groups", "topk", "initial_acc",
                         "final_acc", "acc_drop", "num_flips", "avg_search_time"])
        writer.writerow([
            args.seed, args.coarse_groups, args.topk,
            f"{init_acc:.2f}", f"{final_acc:.2f}", f"{init_acc - final_acc:.2f}",
            len(attack_history), f"{avg_search_time:.4f}",
        ])

    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[{now_ts()}] Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
