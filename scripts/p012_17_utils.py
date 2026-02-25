#!/usr/bin/env python3
"""
Shared utilities for Tasks 12-17 experiments.

Design goals:
- Keep all runs CPU-friendly and deterministic.
- Reuse existing model/attack pipeline primitives.
- Provide scoring-ablation variants over the same non-collision move space.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from models.factory import create_resnet20
from train.ptq_convert import Int8QuantizedResNet
from train.train_utils import get_cifar10_loaders

from bfa.csr_non_collision_attack import _apply_non_collision_move  # noqa: SLF001


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> str:
    # Project requirement: default to CPU unless CUDA is explicitly available.
    # We still select CUDA when available for compatibility, but scripts can override to CPU.
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_cifar10_loaders_offline(
    batch_size: int = 256,
    data_dir: str = "./data",
    num_workers: int = 0,
):
    # train_utils uses download=True, but if local files exist this is offline-safe.
    # We force num_workers=0 due known multiprocessing PermissionError in this repo.
    return get_cifar10_loaders(batch_size=batch_size, data_dir=data_dir, num_workers=0)


def load_sparse_int8_resnet20(
    device: str,
    ckpt_path: str = "models/sparse_model.pth",
) -> nn.Module:
    base_model = create_resnet20(
        sparsity_type="2:4",
        pretrained_path=ckpt_path
    ).to(device)
    base_model.eval()
    base_model.freeze_sparse_masks()
    model = Int8QuantizedResNet(base_model, copy_sparse_masks=True).to(device)
    model.calibrate_all_layers()
    model.eval()
    return model


def evaluate_subset(
    model: nn.Module,
    loader,
    device: str,
    max_samples: Optional[int] = None,
) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += float(loss.item()) * int(inputs.size(0))
            _, predicted = outputs.max(1)
            total += int(targets.size(0))
            correct += int(predicted.eq(targets).sum().item())
            if max_samples is not None and total >= max_samples:
                break
    acc = 100.0 * float(correct) / float(total)
    avg_loss = float(total_loss) / float(total)
    return acc, avg_loss


def flatten_groups(t: torch.Tensor):
    # Keep consistent with bfa/csr_non_collision_attack.py
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


def get_sparse_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    layers = []
    for name, module in model.named_modules():
        if hasattr(module, "int8_weights") and hasattr(module, "scale") and hasattr(module, "sparse_mask"):
            if module.sparse_mask is not None:
                module.sparse_mask = (module.sparse_mask > 0.5).to(module.sparse_mask.dtype)
                layers.append((name, module))
    return layers


@dataclass
class Candidate:
    score: float
    layer_name: str
    module: nn.Module
    group_idx: int
    old_idx: int
    new_idx: int
    bit_pos: int
    w_fp: float
    grad_delta: float
    grad_old: float
    grad_new: float


def build_candidates(
    model: nn.Module,
    calib_loader,
    device: str,
    calib_samples: int,
    flipped: set,
    counters: Dict[str, int],
    score_mode: str = "ncsa",  # ncsa|grad_only|weight_only|random_valid
    max_groups_per_layer: Optional[int] = None,
    allow_nonpositive: bool = False,
    rng: Optional[random.Random] = None,
) -> List[Candidate]:
    """
    Build candidate non-collision moves under different scoring rules.
    """
    if rng is None:
        rng = random.Random(0)

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

    out: List[Candidate] = []
    layers = get_sparse_layers(model)
    for layer_name, module in layers:
        if module.weight.grad is None:
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
        if max_groups_per_layer is not None and num_groups > max_groups_per_layer:
            # Deterministic sampling from RNG.
            group_indices = rng.sample(range(num_groups), k=max_groups_per_layer)
        else:
            group_indices = range(num_groups)

        for g_idx in group_indices:
            m_group = m_flat[g_idx]
            active = (m_group > 0.5).nonzero(as_tuple=False).flatten().tolist()
            if len(active) != 2:
                continue
            a_idx, b_idx = int(active[0]), int(active[1])

            for old_idx, neighbor_idx in ((a_idx, b_idx), (b_idx, a_idx)):
                w_val = int(w_flat[g_idx, old_idx].item())
                if w_val == 0:
                    continue
                w_fp = float(w_val) * scale
                g_old = float(g_flat[g_idx, old_idx].item())

                for bit_pos in (0, 1):
                    counters["attempted"] += 1
                    new_idx = int(old_idx ^ (1 << bit_pos))
                    if new_idx == neighbor_idx:
                        counters["collisions"] += 1
                        continue
                    if int(m_group[new_idx].item()) == 1:
                        counters["collisions"] += 1
                        continue

                    key = (layer_name, int(g_idx), int(old_idx), int(bit_pos))
                    if key in flipped:
                        continue

                    g_new = float(g_flat[g_idx, new_idx].item())
                    grad_delta = g_new - g_old

                    if score_mode == "ncsa":
                        score = w_fp * grad_delta
                        if (score <= 0.0) and (not allow_nonpositive):
                            continue
                    elif score_mode == "grad_only":
                        score = grad_delta
                        if (score <= 0.0) and (not allow_nonpositive):
                            continue
                    elif score_mode == "weight_only":
                        score = abs(w_fp)
                    elif score_mode == "random_valid":
                        score = 0.0
                    else:
                        raise ValueError(f"Unsupported score_mode: {score_mode}")

                    out.append(Candidate(
                        score=float(score),
                        layer_name=layer_name,
                        module=module,
                        group_idx=int(g_idx),
                        old_idx=int(old_idx),
                        new_idx=int(new_idx),
                        bit_pos=int(bit_pos),
                        w_fp=float(w_fp),
                        grad_delta=float(grad_delta),
                        grad_old=float(g_old),
                        grad_new=float(g_new),
                    ))

        if module.weight.grad is not None:
            module.weight.grad = None

    return out


def select_candidate(
    candidates: List[Candidate],
    mode: str,
    rng: random.Random,
) -> Optional[Candidate]:
    if not candidates:
        return None
    if mode == "random_valid":
        return rng.choice(candidates)
    # Deterministic max selection with tie-break by tuple fields.
    best = max(
        candidates,
        key=lambda c: (c.score, c.layer_name, -c.group_idx, -c.old_idx, -c.new_idx),
    )
    return best


def flips_to_threshold(acc_history: Sequence[float], threshold: float) -> Optional[int]:
    for i, acc in enumerate(acc_history):
        if float(acc) <= float(threshold):
            return int(i)
    return None


def layer_category(name: str) -> str:
    if name == "conv1":
        return "stem"
    if "layer1" in name:
        return "stage1"
    if "layer2" in name:
        return "stage2"
    if "layer3" in name:
        return "stage3"
    if ".downsample." in name:
        return "downsample"
    if name == "fc" or name.endswith(".fc"):
        return "head"
    return "other"


def run_scored_noncollision_attack(
    model: nn.Module,
    test_loader,
    calib_loader,
    device: str,
    score_mode: str,
    seed: int,
    max_success: int = 50,
    calib_samples: int = 256,
    eval_samples: Optional[int] = 2000,
    max_groups_per_layer: Optional[int] = 2000,
    allow_nonpositive: bool = False,
    track_trace: bool = False,
    enable_timing: bool = False,
) -> Dict:
    """
    Variant runner over the Task5 non-collision move space.
    """
    rng = random.Random(seed)

    acc0, loss0 = evaluate_subset(model, test_loader, device=device, max_samples=eval_samples)
    counters = {"attempted": 0, "collisions": 0}
    flipped = set()

    acc_hist = [acc0]
    loss_hist = [loss0]
    trace: List[Dict] = []
    search_ms: List[float] = []
    apply_ms: List[float] = []
    eval_ms: List[float] = []

    success = 0
    wall_t0 = time.perf_counter()
    while success < max_success:
        t0 = time.perf_counter()
        candidates = build_candidates(
            model=model,
            calib_loader=calib_loader,
            device=device,
            calib_samples=calib_samples,
            flipped=flipped,
            counters=counters,
            score_mode=score_mode,
            max_groups_per_layer=max_groups_per_layer,
            allow_nonpositive=allow_nonpositive,
            rng=rng,
        )
        if enable_timing:
            search_ms.append((time.perf_counter() - t0) * 1000.0)
        chosen = select_candidate(candidates, mode=score_mode, rng=rng)
        if chosen is None:
            break

        flip_key = (chosen.layer_name, chosen.group_idx, chosen.old_idx, chosen.bit_pos)
        flipped.add(flip_key)

        t1 = time.perf_counter()
        ok = _apply_non_collision_move(
            chosen.module,
            chosen.group_idx,
            chosen.old_idx,
            chosen.new_idx,
        )
        if enable_timing:
            apply_ms.append((time.perf_counter() - t1) * 1000.0)
        if not ok:
            continue

        success += 1

        t2 = time.perf_counter()
        acc, loss = evaluate_subset(model, test_loader, device=device, max_samples=eval_samples)
        if enable_timing:
            eval_ms.append((time.perf_counter() - t2) * 1000.0)
        acc_hist.append(acc)
        loss_hist.append(loss)

        if track_trace:
            trace.append({
                "flip": int(success),
                "layer": chosen.layer_name,
                "layer_category": layer_category(chosen.layer_name),
                "group": int(chosen.group_idx),
                "old_idx": int(chosen.old_idx),
                "new_idx": int(chosen.new_idx),
                "bit_pos": int(chosen.bit_pos),
                "score": float(chosen.score),
                "w_fp": float(chosen.w_fp),
                "grad_old": float(chosen.grad_old),
                "grad_new": float(chosen.grad_new),
                "grad_delta": float(chosen.grad_delta),
                "effective_type": "rewire",
                "accuracy": float(acc),
                "loss": float(loss),
            })

    wall_sec = time.perf_counter() - wall_t0
    return {
        "mode": score_mode,
        "seed": int(seed),
        "initial_accuracy": float(acc_hist[0]),
        "final_accuracy": float(acc_hist[-1]),
        "total_flips": int(success),
        "attempted_flips": int(counters["attempted"]),
        "collisions_skipped": int(counters["collisions"]),
        "accuracy_history": acc_hist,
        "loss_history": loss_hist,
        "trace": trace,
        "timing": {
            "search_ms": search_ms,
            "apply_ms": apply_ms,
            "eval_ms": eval_ms,
            "wall_sec": float(wall_sec),
            "avg_search_ms": float(sum(search_ms) / len(search_ms)) if search_ms else 0.0,
            "avg_apply_ms": float(sum(apply_ms) / len(apply_ms)) if apply_ms else 0.0,
            "avg_eval_ms": float(sum(eval_ms) / len(eval_ms)) if eval_ms else 0.0,
        },
    }

