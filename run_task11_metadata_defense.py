#!/usr/bin/env python3
"""
Task 11: Metadata Integrity Defense + Overhead/Effectiveness

Goal:
  Provide a software-simulated integrity protection mechanism for *metadata only*
  (2:4 indices) and evaluate its effect against NCSA (non-collision index attack).

Defense mechanisms implemented:
  A) Per-group parity (1-bit) over the 4 metadata bits (two 2-bit indices).
  B) Per-line CRC-8 over 64B worth of metadata (default).

Mitigation policies:
  - detect-and-revert: restore last-known-good metadata/values for affected group/line.
  - detect-and-drop: set affected group/line to zeros (safe degradation), and treat that as new good state.

Outputs (fixed filenames):
  - results/task11_defense_attack_curves.png
  - results/task11_defense_overhead_table.csv
  - results/task11_defense_result.pkl
  - results/task11_defense_log.txt

Constraints:
  - CPU-only
  - No extra deps beyond torch/torchvision/matplotlib/pickle
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import pickle
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models.factory import create_resnet20
from train.ptq_convert import Int8QuantizedResNet
from train.train_utils import get_cifar10_loaders

# Reuse the Task5+ NCSA candidate scoring implementation for consistency.
from bfa.csr_non_collision_attack import _compute_candidates, _apply_non_collision_move  # noqa: SLF001


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _flatten_groups(t: torch.Tensor):
    """
    Match bfa/csr_non_collision_attack.py grouping:
      - conv: permute to [out, kh, kw, in] then flatten last dim in groups of 4
      - linear: view as [-1,4]
    """
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


def _restore_groups(flat: torch.Tensor, meta):
    kind, shape = meta
    if kind == "conv":
        t_perm = flat.view(shape)
        return t_perm.permute(0, 3, 1, 2).contiguous()
    return flat.view(shape)


def _get_sparse_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    layers: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if hasattr(module, "int8_weights") and hasattr(module, "scale") and hasattr(module, "sparse_mask"):
            if module.sparse_mask is not None:
                module.sparse_mask = (module.sparse_mask > 0.5).to(module.sparse_mask.dtype)
                layers.append((name, module))
    return layers


def _encode_nibble_from_group_mask(m_group: torch.Tensor) -> Tuple[int, bool]:
    """
    Encode 2:4 positions as a 4-bit nibble: idx0 | (idx1<<2), with idx0<idx1.
    Returns (nibble, valid_2of4).
    """
    active = (m_group > 0.5).nonzero(as_tuple=False).flatten().tolist()
    if len(active) != 2:
        return 0, False
    a, b = int(active[0]), int(active[1])
    if a > b:
        a, b = b, a
    nibble = (a & 0x3) | ((b & 0x3) << 2)
    return int(nibble), True


def _parity4(nibble: int) -> int:
    p = 0
    for i in range(4):
        p ^= (nibble >> i) & 1
    return int(p)


def _crc8(data: bytes, poly: int = 0x07, init: int = 0x00) -> int:
    """
    Simple CRC-8 implementation (MSB-first), polynomial 0x07.
    """
    crc = init & 0xFF
    for b in data:
        crc ^= (b & 0xFF)
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return int(crc)


def _pack_nibbles_to_bytes(nibbles: List[int]) -> bytes:
    out = bytearray()
    i = 0
    while i < len(nibbles):
        lo = nibbles[i] & 0xF
        hi = (nibbles[i + 1] & 0xF) if (i + 1) < len(nibbles) else 0
        out.append(lo | (hi << 4))
        i += 2
    return bytes(out)


def _line_payload_bytes_from_flat(
    m_flat: torch.Tensor,
    line_idx: int,
    groups_per_line: int,
) -> bytes:
    lo = int(line_idx * groups_per_line)
    hi = min(int((line_idx + 1) * groups_per_line), int(m_flat.shape[0]))
    nibbles: List[int] = []
    for g in range(lo, hi):
        nib, _ = _encode_nibble_from_group_mask(m_flat[g])
        nibbles.append(int(nib))
    return _pack_nibbles_to_bytes(nibbles)


def _metadata_hash_sha1(model: nn.Module) -> str:
    """
    Tiny sanity hash over metadata nibbles across all sparse layers.
    """
    payload = bytearray()
    for _, module in _get_sparse_layers(model):
        m_flat, _ = _flatten_groups(module.sparse_mask)
        if m_flat is None:
            continue
        for g in range(int(m_flat.shape[0])):
            nib, _ = _encode_nibble_from_group_mask(m_flat[g])
            payload.append(int(nib) & 0xF)
    return hashlib.sha1(bytes(payload)).hexdigest()


def _evaluate(model: nn.Module, loader, device: str, max_samples: Optional[int]) -> Tuple[float, float]:
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


def _load_sparse_int8_resnet20(device: str, ckpt_path: str) -> nn.Module:
    base = create_resnet20(sparsity_type="2:4", pretrained_path=ckpt_path).to(device)
    base.eval()
    base.freeze_sparse_masks()
    model = Int8QuantizedResNet(base, copy_sparse_masks=True).to(device)
    model.calibrate_all_layers()
    model.eval()
    return model


@dataclass
class IntegrityState:
    # Flattening metadata for restore.
    w_meta: Tuple[str, Tuple[int, ...]]
    m_meta: Tuple[str, Tuple[int, ...]]

    # Last-known-good state (flat groups).
    good_w_flat: torch.Tensor  # int8, [G,4]
    good_m_flat: torch.Tensor  # float/binary, [G,4]

    # Integrity codes for last-known-good.
    good_parity: torch.Tensor  # uint8, [G]
    good_crc: torch.Tensor     # uint8, [L]

    groups_per_line: int
    line_bytes: int


def _build_integrity_state_for_layer(
    module: nn.Module,
    line_bytes: int,
) -> IntegrityState:
    w_flat, w_meta = _flatten_groups(module.int8_weights)
    m_flat, m_meta = _flatten_groups(module.sparse_mask)
    assert w_flat is not None and m_flat is not None

    good_w_flat = w_flat.detach().clone()
    good_m_flat = m_flat.detach().clone()

    nibbles = []
    parity = torch.zeros(good_m_flat.shape[0], dtype=torch.uint8)
    for g in range(good_m_flat.shape[0]):
        nib, _ = _encode_nibble_from_group_mask(good_m_flat[g])
        nibbles.append(int(nib))
        parity[g] = _parity4(int(nib))

    # 64B of metadata = 512 bits = 128 groups (4 bits/group). Generalize:
    groups_per_line = int(line_bytes * 2)
    if groups_per_line <= 0:
        groups_per_line = 128

    line_crc_list: List[int] = []
    for line_start in range(0, len(nibbles), groups_per_line):
        line_nibbles = nibbles[line_start: line_start + groups_per_line]
        payload = _pack_nibbles_to_bytes(line_nibbles)
        line_crc_list.append(_crc8(payload))
    good_crc = torch.tensor(line_crc_list, dtype=torch.uint8)

    return IntegrityState(
        w_meta=w_meta,
        m_meta=m_meta,
        good_w_flat=good_w_flat,
        good_m_flat=good_m_flat,
        good_parity=parity,
        good_crc=good_crc,
        groups_per_line=groups_per_line,
        line_bytes=int(line_bytes),
    )


def _compute_current_parity_for_group(m_flat: torch.Tensor, g_idx: int) -> int:
    nib, _ = _encode_nibble_from_group_mask(m_flat[g_idx])
    return _parity4(int(nib))


def _compute_current_crc_for_line(m_flat: torch.Tensor, line_idx: int, groups_per_line: int) -> int:
    lo = int(line_idx * groups_per_line)
    hi = min(int((line_idx + 1) * groups_per_line), int(m_flat.shape[0]))
    nibbles = []
    for g in range(lo, hi):
        nib, _ = _encode_nibble_from_group_mask(m_flat[g])
        nibbles.append(int(nib))
    payload = _pack_nibbles_to_bytes(nibbles)
    return _crc8(payload)


def _restore_group_from_good(
    w_flat: torch.Tensor,
    m_flat: torch.Tensor,
    state: IntegrityState,
    g_idx: int,
) -> None:
    w_flat[g_idx].copy_(state.good_w_flat[g_idx])
    m_flat[g_idx].copy_(state.good_m_flat[g_idx])


def _restore_line_from_good(
    w_flat: torch.Tensor,
    m_flat: torch.Tensor,
    state: IntegrityState,
    line_idx: int,
) -> None:
    lo = int(line_idx * state.groups_per_line)
    hi = min(int((line_idx + 1) * state.groups_per_line), int(w_flat.shape[0]))
    w_flat[lo:hi].copy_(state.good_w_flat[lo:hi])
    m_flat[lo:hi].copy_(state.good_m_flat[lo:hi])


def _drop_group_to_zero_and_commit(
    w_flat: torch.Tensor,
    m_flat: torch.Tensor,
    state: IntegrityState,
    g_idx: int,
) -> None:
    w_flat[g_idx].zero_()
    m_flat[g_idx].zero_()
    state.good_w_flat[g_idx].zero_()
    state.good_m_flat[g_idx].zero_()
    state.good_parity[g_idx] = _parity4(0)
    # Update the line CRC to reflect the new good state.
    line_idx = int(g_idx // state.groups_per_line)
    new_crc = _compute_current_crc_for_line(state.good_m_flat, line_idx, state.groups_per_line)
    if line_idx < int(state.good_crc.numel()):
        state.good_crc[line_idx] = int(new_crc)


def _drop_line_to_zero_and_commit(
    w_flat: torch.Tensor,
    m_flat: torch.Tensor,
    state: IntegrityState,
    line_idx: int,
) -> None:
    lo = int(line_idx * state.groups_per_line)
    hi = min(int((line_idx + 1) * state.groups_per_line), int(w_flat.shape[0]))
    w_flat[lo:hi].zero_()
    m_flat[lo:hi].zero_()
    state.good_w_flat[lo:hi].zero_()
    state.good_m_flat[lo:hi].zero_()
    state.good_parity[lo:hi] = _parity4(0)
    new_crc = _compute_current_crc_for_line(state.good_m_flat, line_idx, state.groups_per_line)
    if line_idx < int(state.good_crc.numel()):
        state.good_crc[line_idx] = int(new_crc)


def _commit_current_group_as_good(
    w_flat: torch.Tensor,
    m_flat: torch.Tensor,
    state: IntegrityState,
    g_idx: int,
) -> None:
    state.good_w_flat[g_idx].copy_(w_flat[g_idx])
    state.good_m_flat[g_idx].copy_(m_flat[g_idx])
    state.good_parity[g_idx] = int(_compute_current_parity_for_group(m_flat, g_idx))
    line_idx = int(g_idx // state.groups_per_line)
    new_crc = _compute_current_crc_for_line(state.good_m_flat, line_idx, state.groups_per_line)
    if line_idx < int(state.good_crc.numel()):
        state.good_crc[line_idx] = int(new_crc)


def _commit_current_line_as_good(
    w_flat: torch.Tensor,
    m_flat: torch.Tensor,
    state: IntegrityState,
    line_idx: int,
) -> None:
    lo = int(line_idx * state.groups_per_line)
    hi = min(int((line_idx + 1) * state.groups_per_line), int(w_flat.shape[0]))
    state.good_w_flat[lo:hi].copy_(w_flat[lo:hi])
    state.good_m_flat[lo:hi].copy_(m_flat[lo:hi])
    for g in range(lo, hi):
        state.good_parity[g] = int(_compute_current_parity_for_group(m_flat, g))
    new_crc = _compute_current_crc_for_line(state.good_m_flat, line_idx, state.groups_per_line)
    if line_idx < int(state.good_crc.numel()):
        state.good_crc[line_idx] = int(new_crc)


def run_defended_ncsa(
    model: nn.Module,
    test_loader,
    calib_loader,
    device: str,
    defense: str,
    mitigation: str,
    max_flips: int,
    calib_samples: int,
    eval_samples: Optional[int],
    max_groups_per_layer: Optional[int],
    allow_nonpositive: bool,
    line_bytes: int,
    bypass_parity: bool = False,
    bypass_crc: bool = False,
    debug_trace: bool = False,
    trace_records: Optional[List[Dict]] = None,
    run_name: str = "",
) -> Dict:
    """
    defense: "none" | "parity" | "crc"
    mitigation: "revert" | "drop"
    """
    model.eval()
    layers = _get_sparse_layers(model)
    integrity: Dict[str, IntegrityState] = {}
    if defense in ("parity", "crc"):
        for name, module in layers:
            integrity[name] = _build_integrity_state_for_layer(module=module, line_bytes=line_bytes)

    # Attack bookkeeping
    flipped = set()
    counters = {"attempted": 0, "collisions": 0}
    detected = 0
    mitigated = 0

    acc0, loss0 = _evaluate(model, test_loader, device=device, max_samples=eval_samples)
    acc_hist = [acc0]
    loss_hist = [loss0]
    step_log = [{"flip_attempt": 0, "accuracy": acc0, "loss": loss0, "move": None, "detected": False, "mitigated": False}]
    trace = trace_records if trace_records is not None else []
    timing_checks = {
        "hash_b_diff_from_a": 0,
        "hash_c_eq_d": 0,
        "revert_hash_a_eq_c": 0,
        "assertion_failures": 0,
    }

    for step in range(1, max_flips + 1):
        candidates = _compute_candidates(
            model=model,
            calib_loader=calib_loader,
            device=device,
            calib_samples=calib_samples,
            flipped=flipped,
            counters=counters,
            max_groups_per_layer=max_groups_per_layer,
            allow_nonpositive=allow_nonpositive,
        )
        if not candidates:
            break

        score, layer_name, module, g_idx, old_idx, new_idx = candidates[0]
        bit_pos = (old_idx ^ new_idx).bit_length() - 1
        flip_key = (layer_name, int(g_idx), int(old_idx), int(bit_pos))
        flipped.add(flip_key)

        hash_a = _metadata_hash_sha1(model) if debug_trace else None
        pre_nibble = None
        pre_parity_cur = None
        pre_parity_stored = None
        pre_line_idx = None
        pre_crc_cur = None
        pre_crc_stored = None
        pre_line_bytes_hex = None
        state = integrity.get(layer_name) if defense in ("parity", "crc") else None
        if debug_trace and state is not None:
            pre_w_flat, _ = _flatten_groups(module.int8_weights)
            pre_m_flat, _ = _flatten_groups(module.sparse_mask)
            if pre_w_flat is not None and pre_m_flat is not None:
                pre_nibble, _ = _encode_nibble_from_group_mask(pre_m_flat[int(g_idx)])
                pre_parity_cur = _compute_current_parity_for_group(pre_m_flat, int(g_idx))
                pre_parity_stored = int(state.good_parity[int(g_idx)].item())
                pre_line_idx = int(int(g_idx) // state.groups_per_line)
                pre_crc_cur = _compute_current_crc_for_line(pre_m_flat, pre_line_idx, state.groups_per_line)
                pre_crc_stored = (
                    int(state.good_crc[pre_line_idx].item())
                    if pre_line_idx < int(state.good_crc.numel())
                    else None
                )
                pre_line_bytes_hex = _line_payload_bytes_from_flat(
                    pre_m_flat, pre_line_idx, state.groups_per_line
                ).hex()

        ok = _apply_non_collision_move(module, int(g_idx), int(old_idx), int(new_idx))
        if not ok:
            # Still record as an attempted flip attempt.
            acc, loss = _evaluate(model, test_loader, device=device, max_samples=eval_samples)
            acc_hist.append(acc)
            loss_hist.append(loss)
            step_log.append({
                "flip_attempt": step,
                "accuracy": acc,
                "loss": loss,
                "move": {"layer": layer_name, "group": int(g_idx), "old_idx": int(old_idx), "new_idx": int(new_idx), "score": float(score)},
                "detected": False,
                "mitigated": False,
                "note": "apply_move_failed",
            })
            continue

        hash_b = _metadata_hash_sha1(model) if debug_trace else None

        detected_this = False
        mitigated_this = False
        bypass_applied = False
        post_nibble = None
        post_parity_cur = None
        post_parity_stored = None
        post_crc_cur = None
        post_crc_stored = None
        post_line_bytes_hex = None

        if defense in ("parity", "crc"):
            if state is not None:
                # Flatten current module state for checks and potential restore.
                w_flat, _ = _flatten_groups(module.int8_weights)
                m_flat, _ = _flatten_groups(module.sparse_mask)
                assert w_flat is not None and m_flat is not None

                line_idx = int(int(g_idx) // state.groups_per_line)
                post_nibble, _ = _encode_nibble_from_group_mask(m_flat[int(g_idx)])
                post_parity_cur = _compute_current_parity_for_group(m_flat, int(g_idx))
                post_parity_stored = int(state.good_parity[int(g_idx)].item())
                post_crc_cur = _compute_current_crc_for_line(m_flat, line_idx, state.groups_per_line)
                post_crc_stored = (
                    int(state.good_crc[line_idx].item())
                    if line_idx < int(state.good_crc.numel())
                    else None
                )
                post_line_bytes_hex = _line_payload_bytes_from_flat(
                    m_flat, line_idx, state.groups_per_line
                ).hex()

                # Adaptive white-box bypass hooks for checksum-in-same-fault-domain scenario.
                if defense == "parity" and bypass_parity:
                    state.good_parity[int(g_idx)] = int(post_parity_cur)
                    post_parity_stored = int(state.good_parity[int(g_idx)].item())
                    bypass_applied = True
                if defense == "crc" and bypass_crc:
                    if line_idx < int(state.good_crc.numel()):
                        state.good_crc[line_idx] = int(post_crc_cur)
                    post_crc_stored = (
                        int(state.good_crc[line_idx].item())
                        if line_idx < int(state.good_crc.numel())
                        else None
                    )
                    bypass_applied = True

                if defense == "parity":
                    cur_p = _compute_current_parity_for_group(m_flat, int(g_idx))
                    exp_p = int(state.good_parity[int(g_idx)].item())
                    if cur_p != exp_p:
                        detected_this = True
                        detected += 1
                        if mitigation == "revert":
                            _restore_group_from_good(w_flat, m_flat, state, int(g_idx))
                            mitigated_this = True
                            mitigated += 1
                        else:
                            _drop_group_to_zero_and_commit(w_flat, m_flat, state, int(g_idx))
                            mitigated_this = True
                            mitigated += 1
                    else:
                        # Treat as legitimate update (or an undetected corruption) and commit.
                        _commit_current_group_as_good(w_flat, m_flat, state, int(g_idx))

                elif defense == "crc":
                    line_idx = int(int(g_idx) // state.groups_per_line)
                    cur_crc = _compute_current_crc_for_line(m_flat, line_idx, state.groups_per_line)
                    exp_crc = int(state.good_crc[line_idx].item()) if line_idx < int(state.good_crc.numel()) else int(cur_crc)
                    if int(cur_crc) != int(exp_crc):
                        detected_this = True
                        detected += 1
                        if mitigation == "revert":
                            _restore_line_from_good(w_flat, m_flat, state, line_idx)
                            mitigated_this = True
                            mitigated += 1
                        else:
                            _drop_line_to_zero_and_commit(w_flat, m_flat, state, line_idx)
                            mitigated_this = True
                            mitigated += 1
                    else:
                        _commit_current_line_as_good(w_flat, m_flat, state, line_idx)

                # Copy back to module original layout.
                with torch.no_grad():
                    # Clone to avoid copy_ overlap when flattening returns a view (linear case).
                    module.int8_weights.copy_(_restore_groups(w_flat, state.w_meta).clone())
                    module.sparse_mask.copy_(_restore_groups(m_flat, state.m_meta).clone())

        hash_c = _metadata_hash_sha1(model) if debug_trace else None
        final_nibble = None
        final_parity_cur = None
        final_parity_stored = None
        final_crc_cur = None
        final_crc_stored = None
        final_line_bytes_hex = None
        if debug_trace and defense in ("parity", "crc") and state is not None:
            final_w_flat, _ = _flatten_groups(module.int8_weights)
            final_m_flat, _ = _flatten_groups(module.sparse_mask)
            if final_w_flat is not None and final_m_flat is not None:
                line_idx_final = int(int(g_idx) // state.groups_per_line)
                final_nibble, _ = _encode_nibble_from_group_mask(final_m_flat[int(g_idx)])
                final_parity_cur = _compute_current_parity_for_group(final_m_flat, int(g_idx))
                final_parity_stored = int(state.good_parity[int(g_idx)].item())
                final_crc_cur = _compute_current_crc_for_line(final_m_flat, line_idx_final, state.groups_per_line)
                final_crc_stored = (
                    int(state.good_crc[line_idx_final].item())
                    if line_idx_final < int(state.good_crc.numel())
                    else None
                )
                final_line_bytes_hex = _line_payload_bytes_from_flat(
                    final_m_flat, line_idx_final, state.groups_per_line
                ).hex()

        # Timing/order assertions: flip -> check/mitigate -> eval
        hash_d = _metadata_hash_sha1(model) if debug_trace else None
        if debug_trace:
            if hash_b is not None and hash_a is not None and hash_b != hash_a:
                timing_checks["hash_b_diff_from_a"] += 1
            if hash_c is not None and hash_d is not None and hash_c == hash_d:
                timing_checks["hash_c_eq_d"] += 1
            if detected_this and mitigation == "revert" and hash_a is not None and hash_c is not None and hash_a == hash_c:
                timing_checks["revert_hash_a_eq_c"] += 1

            try:
                if hash_c != hash_d:
                    raise AssertionError("hash_c != hash_d: metadata changed between mitigation and eval.")
                if detected_this and mitigation == "revert" and hash_a != hash_c:
                    raise AssertionError("revert expected hash_a == hash_c but mismatch observed.")
            except AssertionError:
                timing_checks["assertion_failures"] += 1

        acc, loss = _evaluate(model, test_loader, device=device, max_samples=eval_samples)
        acc_hist.append(acc)
        loss_hist.append(loss)
        step_log.append({
            "flip_attempt": step,
            "accuracy": acc,
            "loss": loss,
            "move": {"layer": layer_name, "group": int(g_idx), "old_idx": int(old_idx), "new_idx": int(new_idx), "score": float(score)},
            "bit_pos": int(bit_pos),
            "detected": bool(detected_this),
            "mitigated": bool(mitigated_this),
        })

        if debug_trace:
            trace.append({
                "run_name": run_name,
                "attempt_id": int(step),
                "defense": defense,
                "mitigation": mitigation,
                "layer": layer_name,
                "group_idx": int(g_idx),
                "old_idx": int(old_idx),
                "new_idx": int(new_idx),
                "bit_pos": int(bit_pos),
                "hash_a_pre_flip": hash_a,
                "hash_b_post_flip_pre_check": hash_b,
                "hash_c_post_mitigation": hash_c,
                "hash_d_pre_eval": hash_d,
                "pre_nibble": pre_nibble,
                "post_nibble": post_nibble,
                "pre_parity_cur": pre_parity_cur,
                "pre_parity_stored": pre_parity_stored,
                "post_parity_cur": post_parity_cur,
                "post_parity_stored": post_parity_stored,
                "pre_crc_cur": pre_crc_cur,
                "pre_crc_stored": pre_crc_stored,
                "post_crc_cur": post_crc_cur,
                "post_crc_stored": post_crc_stored,
                "final_nibble": final_nibble,
                "final_parity_cur": final_parity_cur,
                "final_parity_stored": final_parity_stored,
                "final_crc_cur": final_crc_cur,
                "final_crc_stored": final_crc_stored,
                "line_idx": pre_line_idx,
                "pre_line_bytes_hex": pre_line_bytes_hex,
                "post_line_bytes_hex": post_line_bytes_hex,
                "final_line_bytes_hex": final_line_bytes_hex,
                "bypass_applied": bool(bypass_applied),
                "detected": bool(detected_this),
                "mitigated": bool(mitigated_this),
                "accuracy_after": float(acc),
                "loss_after": float(loss),
            })

    return {
        "defense": defense,
        "mitigation": mitigation,
        "initial_accuracy": float(acc_hist[0]),
        "final_accuracy": float(acc_hist[-1]),
        "flip_attempts_executed": int(len(acc_hist) - 1),
        "accuracy_history": acc_hist,
        "loss_history": loss_hist,
        "detected": int(detected),
        "mitigated": int(mitigated),
        "attempted_candidates": int(counters["attempted"]),
        "collisions_skipped": int(counters["collisions"]),
        "step_log": step_log,
        "timing_checks": timing_checks,
    }


def _compute_overhead_rows(line_bytes: int, crc_bits: int = 8) -> List[Dict[str, str]]:
    # Baseline per 2:4 group: 2 int8 values + 4-bit metadata (two 2-bit indices).
    base_value_bits = 16.0
    base_meta_bits = 4.0
    base_total = base_value_bits + base_meta_bits

    groups_per_line = float(line_bytes * 2)
    crc_bits_per_group = float(crc_bits) / groups_per_line

    rows = []
    for name, added in [
        ("none", 0.0),
        ("parity", 1.0),
        ("crc8", crc_bits_per_group),
    ]:
        total = base_total + added
        storage_over = 100.0 * (added / base_total) if base_total > 0 else 0.0
        meta_over = 100.0 * (added / base_meta_bits) if base_meta_bits > 0 else 0.0
        rows.append({
            "defense": name,
            "baseline_bits_per_group": f"{base_total:.6f}",
            "added_bits_per_group": f"{added:.6f}",
            "total_bits_per_group": f"{total:.6f}",
            "storage_overhead_pct": f"{storage_over:.6f}",
            "metadata_only_overhead_pct": f"{meta_over:.6f}",
            "line_size_bytes": str(int(line_bytes)),
            "crc_bits": str(int(crc_bits)) if name.startswith("crc") else "",
            "groups_per_line": f"{groups_per_line:.0f}" if name.startswith("crc") else "",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 11: Metadata Integrity Defense (parity/CRC) vs NCSA")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-flips", type=int, default=50)
    parser.add_argument("--calib-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=2000,
                        help="Max eval samples per accuracy measurement (<=0 means full test set).")
    parser.add_argument("--max-groups-per-layer", type=int, default=2000,
                        help="Sampled groups per layer during search (<=0 means all groups).")
    parser.add_argument("--allow-nonpositive", action="store_true",
                        help="If set, allow non-positive score moves in candidate list.")
    parser.add_argument("--defense", choices=["all", "none", "parity", "crc8"], default="all",
                        help="Defense to run. 'all' runs baseline+parity+crc8.")
    parser.add_argument("--mitigation", choices=["revert", "drop"], default="revert")
    parser.add_argument("--bypass-parity", action="store_true",
                        help="Adaptive bypass: after metadata flip, also update stored parity bit.")
    parser.add_argument("--bypass-crc", action="store_true",
                        help="Adaptive bypass: after metadata flip, also update stored CRC byte for affected line.")
    parser.add_argument("--debug-trace", action="store_true",
                        help="Record per-attempt trace and timing/order hashes.")
    parser.add_argument("--trace-path", type=str, default="results/task11_debug_trace.pkl",
                        help="Where to dump trace records when --debug-trace is enabled.")
    parser.add_argument("--out-prefix", type=str, default="results/task11_defense",
                        help="Output prefix. Files will be <prefix>_*.{png,csv,pkl,txt}.")
    parser.add_argument("--line-bytes", type=int, default=64, help="CRC line size in bytes of metadata.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--ckpt", type=str, default="models/sparse_model.pth")
    args = parser.parse_args()

    device = "cpu"
    _set_seed(args.seed)

    out_dir = os.path.dirname(args.out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)
    log_path = f"{args.out_prefix}_log.txt"
    out_png = f"{args.out_prefix}_attack_curves.png"
    out_csv = f"{args.out_prefix}_overhead_table.csv"
    out_pkl = f"{args.out_prefix}_result.pkl"

    # Load CIFAR-10 (fallback to a tiny subset if needed).
    dataset_path = os.path.abspath(args.data_dir)
    try:
        _, test_loader = get_cifar10_loaders(batch_size=args.batch_size, data_dir=args.data_dir, num_workers=0)
    except Exception as e:
        # Fallback: deterministic FakeData for script robustness.
        import torchvision
        import torchvision.transforms as transforms

        dataset_path = "torchvision.datasets.FakeData (fallback)"
        tfm = transforms.Compose([transforms.ToTensor()])
        test_ds = torchvision.datasets.FakeData(size=512, image_size=(3, 32, 32), num_classes=10, transform=tfm)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        with open(log_path, "w", encoding="utf-8") as log:
            log.write("WARNING: CIFAR-10 loader failed; using FakeData fallback.\n")
            log.write(f"Exception: {repr(e)}\n")

    # For calibration we can reuse test_loader (as other tasks do).
    calib_loader = test_loader

    eval_samples = None if args.eval_samples <= 0 else int(args.eval_samples)
    max_groups_per_layer = None if args.max_groups_per_layer <= 0 else int(args.max_groups_per_layer)

    # Run baseline + defended variants on fresh models.
    defense_list = []
    if args.defense == "all":
        defense_list = ["none", "parity", "crc"]
    elif args.defense == "none":
        defense_list = ["none"]
    elif args.defense == "parity":
        defense_list = ["parity"]
    elif args.defense == "crc8":
        defense_list = ["crc"]

    trace_records: List[Dict] = []
    runs = []
    for defense in defense_list:
        model = _load_sparse_int8_resnet20(device=device, ckpt_path=args.ckpt)
        runs.append(run_defended_ncsa(
            model=model,
            test_loader=test_loader,
            calib_loader=calib_loader,
            device=device,
            defense=defense,
            mitigation=args.mitigation,
            max_flips=int(args.max_flips),
            calib_samples=int(args.calib_samples),
            eval_samples=eval_samples,
            max_groups_per_layer=max_groups_per_layer,
            allow_nonpositive=bool(args.allow_nonpositive),
            line_bytes=int(args.line_bytes),
            bypass_parity=bool(args.bypass_parity and defense == "parity"),
            bypass_crc=bool(args.bypass_crc and defense == "crc"),
            debug_trace=bool(args.debug_trace),
            trace_records=trace_records,
            run_name=f"{defense}:{args.mitigation}",
        ))

    if args.debug_trace:
        trace_dir = os.path.dirname(args.trace_path) or "."
        os.makedirs(trace_dir, exist_ok=True)
        with open(args.trace_path, "wb") as f:
            pickle.dump(trace_records, f)

    # Overhead table.
    overhead_rows = _compute_overhead_rows(line_bytes=int(args.line_bytes), crc_bits=8)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(overhead_rows[0].keys()))
        w.writeheader()
        for r in overhead_rows:
            w.writerow(r)

    # Plot curves.
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    label_map = {
        "none": "Baseline (no defense)",
        "parity": f"Parity + {args.mitigation}",
        "crc": f"CRC-8 + {args.mitigation}",
    }
    color_map = {"none": "#1f77b4", "parity": "#ff7f0e", "crc": "#2ca02c"}
    for r in runs:
        xs = list(range(len(r["accuracy_history"])))
        ax.plot(
            xs,
            r["accuracy_history"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=label_map.get(r["defense"], r["defense"]),
            color=color_map.get(r["defense"], None),
        )
    ax.set_xlabel("Flip Attempts")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Task 11: Metadata Integrity Defense vs NCSA (ResNet-20, CIFAR-10)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(0, 100)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save result.pkl.
    result = {
        "timestamp": _now_ts(),
        "seed": int(args.seed),
        "device": device,
        "dataset_path": dataset_path,
        "model_checkpoint_path": os.path.abspath(args.ckpt),
        "config": {
            "max_flips": int(args.max_flips),
            "calib_samples": int(args.calib_samples),
            "eval_samples": eval_samples,
            "max_groups_per_layer": max_groups_per_layer,
            "allow_nonpositive": bool(args.allow_nonpositive),
            "mitigation": args.mitigation,
            "line_bytes": int(args.line_bytes),
            "crc": "crc8(poly=0x07, init=0x00, msb-first)",
            "defense": args.defense,
            "bypass_parity": bool(args.bypass_parity),
            "bypass_crc": bool(args.bypass_crc),
            "debug_trace": bool(args.debug_trace),
        },
        "overhead": overhead_rows,
        "runs": runs,
        "trace_path": args.trace_path if args.debug_trace else None,
        "coverage_notes": {
            "parity": "Detects all single-bit metadata errors; misses even-numbered bit errors.",
            "crc8": "Detects all single-bit errors; for random multi-bit errors, undetected probability ~ 1/2^8.",
        },
    }
    with open(out_pkl, "wb") as f:
        pickle.dump(result, f)

    # Write log (append if fallback wrote warning).
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("=" * 100 + "\n")
        log.write("Task 11: Metadata Integrity Defense + Overhead/Effectiveness\n")
        log.write("=" * 100 + "\n")
        log.write(f"Script: run_task11_metadata_defense.py\n")
        log.write(f"Timestamp: {_now_ts()}\n")
        log.write(f"Seed: {args.seed}\n")
        log.write(f"Device: {device}\n")
        log.write(f"Dataset path: {dataset_path}\n")
        log.write(f"Model checkpoint path: {os.path.abspath(args.ckpt)}\n")
        log.write("\nConfig:\n")
        log.write(f"- max_flips: {args.max_flips}\n")
        log.write(f"- calib_samples: {args.calib_samples}\n")
        log.write(f"- eval_samples: {eval_samples}\n")
        log.write(f"- max_groups_per_layer: {max_groups_per_layer}\n")
        log.write(f"- allow_nonpositive: {bool(args.allow_nonpositive)}\n")
        log.write(f"- defense: {args.defense}\n")
        log.write(f"- mitigation: {args.mitigation}\n")
        log.write(f"- line_bytes: {args.line_bytes}\n")
        log.write(f"- bypass_parity: {bool(args.bypass_parity)}\n")
        log.write(f"- bypass_crc: {bool(args.bypass_crc)}\n")
        log.write(f"- debug_trace: {bool(args.debug_trace)}\n")
        if args.debug_trace:
            log.write(f"- trace_path: {args.trace_path}\n")
        log.write("\nCoverage assumptions:\n")
        log.write("- Parity is computed over the 4-bit metadata nibble (two 2-bit indices), not the 4-bit mask.\n")
        log.write("- CRC-8 is computed per 64B of metadata (default), packed as 4-bit nibbles.\n")
        log.write("- Stored parity/CRC live in software state (`IntegrityState.good_*`) and are trusted by default.\n")
        log.write("\nOverhead:\n")
        for r in overhead_rows:
            log.write(
                f"- {r['defense']}: +{r['added_bits_per_group']} bits/group "
                f"(storage +{r['storage_overhead_pct']}%, metadata-only +{r['metadata_only_overhead_pct']}%)\n"
            )
        log.write("\nResults:\n")
        for r in runs:
            log.write(
                f"- {r['defense']}: init={r['initial_accuracy']:.2f}% final={r['final_accuracy']:.2f}% "
                f"steps={r['flip_attempts_executed']} detected={r['detected']} mitigated={r['mitigated']}\n"
            )
            if "timing_checks" in r:
                tc = r["timing_checks"]
                log.write(
                    f"  timing: hash_b!=a={tc.get('hash_b_diff_from_a', 0)} "
                    f"hash_c==d={tc.get('hash_c_eq_d', 0)} "
                    f"revert(a==c)={tc.get('revert_hash_a_eq_c', 0)} "
                    f"assert_fail={tc.get('assertion_failures', 0)}\n"
                )
        log.write("\nOutputs:\n")
        log.write(f"- {out_png}\n")
        log.write(f"- {out_csv}\n")
        log.write(f"- {out_pkl}\n")
        log.write(f"- {log_path}\n")
        if args.debug_trace:
            log.write(f"- {args.trace_path}\n")
        log.write("\nHow to run:\n")
        log.write("  python run_task11_metadata_defense.py --mitigation revert\n")
        log.write("  python run_task11_metadata_defense.py --mitigation drop\n")

    print(f"[Task11] Wrote: {out_png}")
    print(f"[Task11] Wrote: {out_csv}")
    print(f"[Task11] Wrote: {out_pkl}")
    print(f"[Task11] Wrote: {log_path}")


if __name__ == "__main__":
    main()
