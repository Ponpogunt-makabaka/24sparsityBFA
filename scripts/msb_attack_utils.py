#!/usr/bin/env python3
"""
Shared helper for Tasks 19/21/22: Int8 weight MSB (sign-bit) attack.

Design constraints:
  - CPU-friendly: compute forward/backward ONCE per iteration, scan weights once.
  - No extra deps beyond torch/torchvision/matplotlib/pickle.
  - Keep semantics consistent with existing Int8 BFA: first-order loss increase proxy.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MSBFlip:
    score: float
    layer: str
    module: nn.Module
    weight_idx: int
    bit_pos: int
    int8_before: int
    int8_after: int
    scale: float
    grad: float
    delta_fp: float
    has_sparse_mask: bool
    is_masked_active: bool


def flip_int8_value(int8_val: int, bit_pos: int) -> int:
    """Flip a bit in a signed int8 using proper 8-bit two's complement."""
    u8 = int(int8_val) & 0xFF
    u8 ^= (1 << int(bit_pos))
    return u8 - 256 if u8 >= 128 else u8


def _get_calib_batch(calib_loader, device: str, calib_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    seen = 0
    for x, y in calib_loader:
        xs.append(x.to(device))
        ys.append(y.to(device))
        seen += int(x.size(0))
        if seen >= int(calib_samples):
            break
    x_cat = torch.cat(xs, dim=0)[:calib_samples]
    y_cat = torch.cat(ys, dim=0)[:calib_samples]
    return x_cat, y_cat


def _iter_int8_modules(model: nn.Module):
    for name, module in model.named_modules():
        if hasattr(module, "int8_weights") and hasattr(module, "scale") and hasattr(module, "weight"):
            yield name, module


def select_best_msb_flip(
    model: nn.Module,
    calib_loader,
    device: str,
    calib_samples: int,
    flipped: Set[Tuple[str, int]],
    restrict_to_nonzero_int8: bool,
    restrict_to_sparse_active: bool,
    allow_nonpositive: bool,
) -> Optional[MSBFlip]:
    """
    Greedily select the best MSB flip (bit_pos=7) using first-order proxy:
      score = grad * (w_fp_after - w_fp_before)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    x, y = _get_calib_batch(calib_loader, device=device, calib_samples=calib_samples)
    out = model(x)
    loss = criterion(out, y)
    model.zero_grad()
    loss.backward()

    best: Optional[MSBFlip] = None

    for layer, module in _iter_int8_modules(model):
        if getattr(module, "weight").grad is None:
            continue
        grad = getattr(module, "weight").grad.detach()
        grad_flat = grad.flatten()
        int8_flat = module.int8_weights.flatten()
        scale = float(module.scale.item())

        has_mask = bool(hasattr(module, "sparse_mask") and getattr(module, "sparse_mask") is not None)
        if has_mask:
            mask_flat = module.sparse_mask.flatten()
        else:
            mask_flat = None

        for idx in range(int8_flat.numel()):
            key = (layer, int(idx))
            if key in flipped:
                continue

            int8_before = int(int8_flat[idx].item())
            if restrict_to_nonzero_int8 and int8_before == 0:
                continue

            is_active = True
            if has_mask and mask_flat is not None:
                is_active = bool(mask_flat[idx].item() > 0.5)
                if restrict_to_sparse_active and not is_active:
                    continue

            g = float(grad_flat[idx].item())
            if abs(g) < 1e-12:
                continue

            int8_after = flip_int8_value(int8_before, 7)
            delta_fp = float((int8_after - int8_before) * scale)
            score = float(g * delta_fp)
            if (score <= 0.0) and (not allow_nonpositive):
                continue

            cand = MSBFlip(
                score=score,
                layer=layer,
                module=module,
                weight_idx=int(idx),
                bit_pos=7,
                int8_before=int8_before,
                int8_after=int8_after,
                scale=scale,
                grad=g,
                delta_fp=delta_fp,
                has_sparse_mask=has_mask,
                is_masked_active=is_active,
            )
            if best is None:
                best = cand
            else:
                # Deterministic tie-break.
                b_key = (best.score, best.layer, -best.weight_idx)
                c_key = (cand.score, cand.layer, -cand.weight_idx)
                if c_key > b_key:
                    best = cand

        # Clear per-module gradients to reduce memory pressure.
        if getattr(module, "weight").grad is not None:
            getattr(module, "weight").grad = None

    return best


def apply_msb_flip(flip: MSBFlip) -> None:
    with torch.no_grad():
        flat = flip.module.int8_weights.flatten()
        flat[flip.weight_idx] = torch.tensor(int(flip.int8_after), dtype=torch.int8, device=flat.device)


def run_msb_attack(
    model: nn.Module,
    test_loader,
    calib_loader,
    device: str,
    seed: int,
    max_flips: int,
    calib_samples: int,
    eval_fn,
    eval_samples: Optional[int],
    restrict_to_nonzero_int8: bool,
    restrict_to_sparse_active: bool,
    allow_nonpositive: bool = False,
    log_interval: int = 1,
) -> Dict:
    """
    Run greedy MSB attack for a fixed number of successful flips.
    Returns a dict with accuracy_history/loss_history + trace.
    """
    rng = random.Random(int(seed))
    _ = rng  # kept for API symmetry; selection is deterministic.

    acc0, loss0 = eval_fn(model, test_loader, device=device, max_samples=eval_samples)
    acc_hist = [float(acc0)]
    loss_hist = [float(loss0)]

    flipped: Set[Tuple[str, int]] = set()
    trace: List[Dict] = []

    t0 = time.perf_counter()
    flips_done = 0
    attempts_no_candidate = 0
    while flips_done < int(max_flips):
        chosen = select_best_msb_flip(
            model=model,
            calib_loader=calib_loader,
            device=device,
            calib_samples=int(calib_samples),
            flipped=flipped,
            restrict_to_nonzero_int8=bool(restrict_to_nonzero_int8),
            restrict_to_sparse_active=bool(restrict_to_sparse_active),
            allow_nonpositive=bool(allow_nonpositive),
        )
        if chosen is None:
            attempts_no_candidate += 1
            if attempts_no_candidate >= 3:
                break
            continue
        attempts_no_candidate = 0

        flipped.add((chosen.layer, int(chosen.weight_idx)))
        apply_msb_flip(chosen)
        flips_done += 1

        acc, loss = eval_fn(model, test_loader, device=device, max_samples=eval_samples)
        acc_hist.append(float(acc))
        loss_hist.append(float(loss))

        trace.append({
            "flip": int(flips_done),
            "layer": str(chosen.layer),
            "weight_idx": int(chosen.weight_idx),
            "bit_pos": int(chosen.bit_pos),
            "score": float(chosen.score),
            "int8_before": int(chosen.int8_before),
            "int8_after": int(chosen.int8_after),
            "scale": float(chosen.scale),
            "grad": float(chosen.grad),
            "delta_fp": float(chosen.delta_fp),
            "has_sparse_mask": bool(chosen.has_sparse_mask),
            "is_masked_active": bool(chosen.is_masked_active),
            "accuracy": float(acc),
            "loss": float(loss),
        })

    wall_sec = float(time.perf_counter() - t0)
    return {
        "seed": int(seed),
        "max_flips": int(max_flips),
        "calib_samples": int(calib_samples),
        "eval_samples": int(eval_samples) if eval_samples is not None else None,
        "restrict_to_nonzero_int8": bool(restrict_to_nonzero_int8),
        "restrict_to_sparse_active": bool(restrict_to_sparse_active),
        "allow_nonpositive": bool(allow_nonpositive),
        "initial_accuracy": float(acc_hist[0]),
        "final_accuracy": float(acc_hist[-1]),
        "total_flips": int(flips_done),
        "accuracy_history": acc_hist,
        "loss_history": loss_hist,
        "trace": trace,
        "timing": {
            "wall_sec": float(wall_sec),
        },
    }

