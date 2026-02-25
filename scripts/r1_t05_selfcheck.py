#!/usr/bin/env python3
"""
R1_T05 Self-Check runner.

This script performs targeted correctness checks for:
1) candidate enumeration and top-k composition
2) exact verification consistency (including apply/revert invariants)
3) Stage-A selection-bias controls (stratified / equal-sampling)
4) action-family functional toggles

It writes:
- summary JSON
- per-step CSV logs
- stage-B candidate-level CSV logs (selected monitored steps)
- best-of-type exact table (step0)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

# Ensure repository root is importable when launched via `python scripts/...`
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import run_R1_T05_joint_best_step_attack as t05
from models.factory import create_resnet20
from scripts.p012_17_utils import load_cifar10_loaders_offline, set_all_seeds
from train.ptq_convert import Int8QuantizedResNet


ACTION_TYPES = ["weight_bit", "index_1bit", "bitmask_swap"]


def _empty_counts() -> Dict[str, int]:
    return {k: 0 for k in ACTION_TYPES}


def collect_fixed_batch(loader, device: str, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    seen = 0
    for inputs, targets in loader:
        xs.append(inputs.to(device))
        ys.append(targets.to(device))
        seen += int(inputs.size(0))
        if seen >= num_samples:
            break
    x = torch.cat(xs, dim=0)[:num_samples]
    y = torch.cat(ys, dim=0)[:num_samples]
    return x, y


def evaluate_on_batch(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        logits = model(x)
        loss = float(criterion(logits, y).item())
        pred = logits.argmax(dim=1)
        acc = float((pred == y).float().mean().item() * 100.0)
    return acc, loss


def compute_gradients_on_batch(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> None:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    out = model(x)
    loss = criterion(out, y)
    model.zero_grad(set_to_none=True)
    loss.backward()


def score_stats(candidates: List[t05.UnifiedCandidate]) -> Dict[str, Optional[float]]:
    if not candidates:
        return {"count": 0, "max": None, "p99": None, "median": None, "min": None}

    vals = sorted(float(c.score_proxy_norm) for c in candidates)
    n = len(vals)

    def _percentile(p: float) -> float:
        idx = max(0, min(n - 1, int(math.ceil(p * n) - 1)))
        return float(vals[idx])

    if n % 2 == 1:
        med = vals[n // 2]
    else:
        med = (vals[n // 2 - 1] + vals[n // 2]) / 2.0

    return {
        "count": n,
        "max": float(vals[-1]),
        "p99": _percentile(0.99),
        "median": float(med),
        "min": float(vals[0]),
    }


def count_by_type(candidates: List[t05.UnifiedCandidate]) -> Dict[str, int]:
    out = _empty_counts()
    for c in candidates:
        out[c.action_type] += 1
    return out


def select_topk(
    per_type: Dict[str, List[t05.UnifiedCandidate]],
    topk: int,
    mode: str,
    rng: random.Random,
    stratified_quota: Dict[str, int],
    equal_sample_n: int,
) -> Tuple[List[t05.UnifiedCandidate], Dict[str, int], Dict[str, int], Optional[float]]:
    sampled: Dict[str, List[t05.UnifiedCandidate]] = {k: list(per_type.get(k, [])) for k in ACTION_TYPES}

    if mode == "equal":
        for k in ACTION_TYPES:
            cur = sampled[k]
            if equal_sample_n > 0 and len(cur) > equal_sample_n:
                sampled[k] = rng.sample(cur, equal_sample_n)

    if mode == "stratified":
        sorted_by_type = {
            k: sorted(sampled[k], key=lambda c: c.score_proxy_norm, reverse=True)
            for k in ACTION_TYPES
        }
        selected: List[t05.UnifiedCandidate] = []
        selected_set: Set[t05.UnifiedCandidate] = set()

        for k in ACTION_TYPES:
            quota = int(stratified_quota.get(k, 0))
            for c in sorted_by_type[k][:quota]:
                if c not in selected_set:
                    selected.append(c)
                    selected_set.add(c)

        if len(selected) < topk:
            remain: List[t05.UnifiedCandidate] = []
            for k in ACTION_TYPES:
                for c in sorted_by_type[k]:
                    if c not in selected_set:
                        remain.append(c)
            remain.sort(key=lambda c: c.score_proxy_norm, reverse=True)
            selected.extend(remain[: max(0, topk - len(selected))])

        selected = sorted(selected[:topk], key=lambda c: c.score_proxy_norm, reverse=True)
    else:
        all_candidates: List[t05.UnifiedCandidate] = []
        for k in ACTION_TYPES:
            all_candidates.extend(sampled[k])
        all_candidates.sort(key=lambda c: c.score_proxy_norm, reverse=True)
        selected = all_candidates[:topk]

    threshold = float(selected[-1].score_proxy_norm) if selected else None
    return selected, count_by_type(selected), {k: len(sampled[k]) for k in ACTION_TYPES}, threshold


def stage_b_verify(
    model: nn.Module,
    candidates: List[t05.UnifiedCandidate],
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    device: str,
    check_invariants: bool,
    repeat_eval: bool,
    repeat_tol: float,
) -> Tuple[List[t05.UnifiedCandidate], List[Dict], float]:
    if not candidates:
        return [], [], 0.0

    criterion = nn.CrossEntropyLoss()
    model.eval()

    with torch.no_grad():
        base_loss = float(criterion(model(eval_x), eval_y).item())

    baseline_int8 = t05.get_int8_weights_hash(model) if check_invariants else ""
    baseline_meta = t05.get_metadata_hash(model) if check_invariants else ""

    verified: List[t05.UnifiedCandidate] = []
    logs: List[Dict] = []

    for rank, c in enumerate(candidates):
        pre_int8 = t05.get_int8_weights_hash(model) if check_invariants else ""
        pre_meta = t05.get_metadata_hash(model) if check_invariants else ""

        apply_ok = t05.apply_candidate(c, model, device)
        loss1 = None
        loss2 = None
        exact_delta = None

        if apply_ok:
            with torch.no_grad():
                loss1 = float(criterion(model(eval_x), eval_y).item())
            if repeat_eval:
                with torch.no_grad():
                    loss2 = float(criterion(model(eval_x), eval_y).item())
            exact_delta = float(loss1 - base_loss)

        revert_ok = t05.revert_candidate(c, model, device) if apply_ok else False

        post_int8 = t05.get_int8_weights_hash(model) if check_invariants else ""
        post_meta = t05.get_metadata_hash(model) if check_invariants else ""

        repeat_abs = abs(loss1 - loss2) if (loss1 is not None and loss2 is not None) else None
        repeat_ok = (repeat_abs is None) or (repeat_abs <= repeat_tol)

        pre_eq_base = None
        post_eq_pre = None
        post_eq_base = None
        if check_invariants:
            pre_eq_base = (pre_int8 == baseline_int8 and pre_meta == baseline_meta)
            post_eq_pre = (post_int8 == pre_int8 and post_meta == pre_meta)
            post_eq_base = (post_int8 == baseline_int8 and post_meta == baseline_meta)

        logs.append(
            {
                "rank": rank,
                "action_type": c.action_type,
                "layer_name": c.layer_name,
                "cost": int(c.cost),
                "proxy_norm": float(c.score_proxy_norm),
                "exact_delta": exact_delta,
                "exact_norm": None if exact_delta is None else float(exact_delta / c.cost),
                "apply_ok": bool(apply_ok),
                "revert_ok": bool(revert_ok),
                "pre_eq_baseline": pre_eq_base,
                "post_eq_pre": post_eq_pre,
                "post_eq_baseline": post_eq_base,
                "repeat_abs": repeat_abs,
                "repeat_ok": repeat_ok,
            }
        )

        if apply_ok and revert_ok:
            verified.append(
                replace(
                    c,
                    exact_delta_loss=exact_delta,
                    exact_delta_loss_norm=exact_delta / c.cost,
                )
            )

    return verified, logs, base_loss


def load_int8_model_from_ckpt(ckpt_path: str, device: str) -> nn.Module:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    ckpt_to_check = checkpoint.get("model_state_dict", checkpoint)
    is_int8_ckpt = any("int8_weights" in k for k in ckpt_to_check.keys())

    if is_int8_ckpt:
        base_model = create_resnet20(sparsity_type="2:4", pretrained_path=None).to(device)
        base_model.eval()
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        for name, module in base_model.named_modules():
            mask_key = f"{name}.sparse_mask"
            if mask_key in state_dict and hasattr(module, "register_buffer"):
                if hasattr(module, "cached_mask"):
                    del module.cached_mask
                module.register_buffer("cached_mask", state_dict[mask_key])

        model = Int8QuantizedResNet(base_model, copy_sparse_masks=True).to(device)
        filtered = {k: v for k, v in state_dict.items() if "sparse_mask" not in k}
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


def run_attack(
    model: nn.Module,
    calib_x: torch.Tensor,
    calib_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    device: str,
    seed: int,
    physical_budget: int,
    topk: int,
    stage_a_mode: str,
    stratified_quota: Dict[str, int],
    equal_sample_n: int,
    enable_weight: bool,
    enable_index: bool,
    enable_bitmask: bool,
    monitored_steps: Set[int],
    invariant_steps: Set[int],
    repeat_steps: Set[int],
    repeat_tol: float = 1e-8,
) -> Dict:
    rng = random.Random(seed)

    acc0, loss0 = evaluate_on_batch(model, eval_x, eval_y)

    flipped_bits: Set[Tuple[str, int, int]] = set()
    exclude_groups: Set[Tuple[str, int]] = set()
    forbidden_transitions: Set[Tuple[str, int, int, int]] = set()
    forbidden_swaps: Set[Tuple[str, int, int]] = set()

    action_breakdown = _empty_counts()
    step_logs: List[Dict] = []
    stageb_logs_by_step: Dict[str, List[Dict]] = {}
    best_of_type_step0: List[Dict] = []

    physical = 0
    step1 = 0

    while physical < physical_budget:
        step1 += 1
        step0 = step1 - 1

        compute_gradients_on_batch(model, calib_x, calib_y)

        per_type: Dict[str, List[t05.UnifiedCandidate]] = {k: [] for k in ACTION_TYPES}
        enum_counters: Dict[str, Dict] = {k: {} for k in ACTION_TYPES}

        if enable_weight:
            wc, w_counter = t05.enumerate_weight_bit_candidates(
                model, device, flipped_bits=flipped_bits, max_candidates_per_layer=500
            )
            per_type["weight_bit"] = wc
            enum_counters["weight_bit"] = w_counter

        if enable_index:
            ic, i_counter = t05.enumerate_index_1bit_candidates(
                model,
                device,
                exclude_groups=exclude_groups,
                forbidden_transitions=forbidden_transitions,
            )
            per_type["index_1bit"] = ic
            enum_counters["index_1bit"] = i_counter

        if enable_bitmask:
            bc, b_counter = t05.enumerate_bitmask_swap_candidates(
                model,
                device,
                exclude_groups=exclude_groups,
                forbidden_swaps=forbidden_swaps,
            )
            per_type["bitmask_swap"] = bc
            enum_counters["bitmask_swap"] = b_counter

        total_counts = {k: len(v) for k, v in per_type.items()}
        proxy_stats = {k: score_stats(per_type[k]) for k in ACTION_TYPES}
        best_proxy_by_type = {
            k: (max(per_type[k], key=lambda c: c.score_proxy_norm) if per_type[k] else None)
            for k in ACTION_TYPES
        }

        topk_candidates, topk_counts, sampled_counts, topk_threshold = select_topk(
            per_type=per_type,
            topk=topk,
            mode=stage_a_mode,
            rng=rng,
            stratified_quota=stratified_quota,
            equal_sample_n=equal_sample_n,
        )

        if not topk_candidates:
            break

        do_invariant = step0 in invariant_steps
        do_repeat = step0 in repeat_steps
        verified, stageb_logs, baseline_loss = stage_b_verify(
            model=model,
            candidates=topk_candidates,
            eval_x=eval_x,
            eval_y=eval_y,
            device=device,
            check_invariants=do_invariant,
            repeat_eval=do_repeat,
            repeat_tol=repeat_tol,
        )
        if not verified:
            break

        if step0 == 0:
            best_of_type_candidates = [c for c in best_proxy_by_type.values() if c is not None]
            best_verified, best_logs, _ = stage_b_verify(
                model=model,
                candidates=best_of_type_candidates,
                eval_x=eval_x,
                eval_y=eval_y,
                device=device,
                check_invariants=True,
                repeat_eval=True,
                repeat_tol=repeat_tol,
            )
            best_map = {c.action_type: c for c in best_verified}
            for action_type in ACTION_TYPES:
                c = best_proxy_by_type[action_type]
                vc = best_map.get(action_type)
                best_of_type_step0.append(
                    {
                        "action_type": action_type,
                        "proxy_norm": None if c is None else float(c.score_proxy_norm),
                        "proxy_raw": None if c is None else float(c.score_proxy),
                        "cost": None if c is None else int(c.cost),
                        "exact_delta": None if vc is None else float(vc.exact_delta_loss),
                        "exact_norm": None if vc is None else float(vc.exact_delta_loss_norm),
                        "layer_name": None if c is None else c.layer_name,
                    }
                )

        best = max(verified, key=lambda c: c.exact_delta_loss_norm)

        if physical + best.cost > physical_budget:
            break

        t05.apply_candidate(best, model, device)

        if best.action_type == "weight_bit":
            flipped_bits.add((best.layer_name, int(best.weight_idx), int(best.bit_pos)))
        elif best.action_type == "index_1bit":
            key = (best.layer_name, int(best.group_idx), int(best.old_code), int(best.new_code))
            forbidden_transitions.add(key)
            forbidden_transitions.add((best.layer_name, int(best.group_idx), int(best.new_code), int(best.old_code)))
            exclude_groups.add((best.layer_name, int(best.group_idx)))
        elif best.action_type == "bitmask_swap":
            swap_code = (int(best.bit_off) << 8) | int(best.bit_on)
            rev_code = (int(best.bit_on) << 8) | int(best.bit_off)
            forbidden_swaps.add((best.layer_name, int(best.group_idx), swap_code))
            forbidden_swaps.add((best.layer_name, int(best.group_idx), rev_code))
            exclude_groups.add((best.layer_name, int(best.group_idx)))

        if len(exclude_groups) > 20:
            exclude_groups = set(list(exclude_groups)[-20:])
        if len(forbidden_transitions) > 1000:
            forbidden_transitions = set(list(forbidden_transitions)[-1000:])
        if len(forbidden_swaps) > 1000:
            forbidden_swaps = set(list(forbidden_swaps)[-1000:])

        physical += int(best.cost)
        action_breakdown[best.action_type] += 1

        acc, loss = evaluate_on_batch(model, eval_x, eval_y)

        record = {
            "step0": step0,
            "step1": step1,
            "physical_flips_used": physical,
            "selected_action_type": best.action_type,
            "selected_cost": int(best.cost),
            "selected_proxy_norm": float(best.score_proxy_norm),
            "selected_exact_delta": float(best.exact_delta_loss),
            "selected_exact_norm": float(best.exact_delta_loss_norm),
            "accuracy": float(acc),
            "loss": float(loss),
            "baseline_loss_stageb": float(baseline_loss),
            "candidates_total_by_type": total_counts,
            "candidates_topk_by_type": topk_counts,
            "sampled_counts_by_type": sampled_counts,
            "proxy_norm_stats_by_type": proxy_stats,
            "topk_threshold": topk_threshold,
            "enum_counters_by_type": enum_counters,
        }
        step_logs.append(record)

        if step0 in monitored_steps:
            stageb_logs_by_step[str(step0)] = stageb_logs

    final_acc, final_loss = evaluate_on_batch(model, eval_x, eval_y)

    return {
        "seed": int(seed),
        "physical_budget": int(physical_budget),
        "topk": int(topk),
        "stage_a_mode": stage_a_mode,
        "equal_sample_n": int(equal_sample_n),
        "stratified_quota": {k: int(v) for k, v in stratified_quota.items()},
        "enable_weight": bool(enable_weight),
        "enable_index": bool(enable_index),
        "enable_bitmask": bool(enable_bitmask),
        "initial_accuracy": float(acc0),
        "initial_loss": float(loss0),
        "final_accuracy": float(final_acc),
        "final_loss": float(final_loss),
        "accuracy_drop": float(acc0 - final_acc),
        "physical_flips_used": int(physical),
        "logical_steps": len(step_logs),
        "action_breakdown": action_breakdown,
        "step_logs": step_logs,
        "stageb_logs_by_step": stageb_logs_by_step,
        "best_of_type_step0": best_of_type_step0,
        "final_int8_hash": t05.get_int8_weights_hash(model),
        "final_metadata_hash": t05.get_metadata_hash(model),
    }


def write_step_stats_csv(path: str, run_result: Dict) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step0",
                "step1",
                "action_type",
                "N_total",
                "N_topk",
                "N_sampled",
                "proxy_norm_max",
                "proxy_norm_p99",
                "proxy_norm_median",
                "proxy_norm_min",
                "selected_action_type",
                "selected_exact_norm",
                "topk_threshold",
            ]
        )
        for s in run_result["step_logs"]:
            for action_type in ACTION_TYPES:
                stats = s["proxy_norm_stats_by_type"][action_type]
                writer.writerow(
                    [
                        s["step0"],
                        s["step1"],
                        action_type,
                        s["candidates_total_by_type"][action_type],
                        s["candidates_topk_by_type"][action_type],
                        s["sampled_counts_by_type"][action_type],
                        stats["max"],
                        stats["p99"],
                        stats["median"],
                        stats["min"],
                        s["selected_action_type"],
                        s["selected_exact_norm"],
                        s["topk_threshold"],
                    ]
                )


def write_stageb_csv(path: str, stageb_logs: List[Dict]) -> None:
    if not stageb_logs:
        return
    keys = [
        "rank",
        "action_type",
        "layer_name",
        "cost",
        "proxy_norm",
        "exact_delta",
        "exact_norm",
        "apply_ok",
        "revert_ok",
        "pre_eq_baseline",
        "post_eq_pre",
        "post_eq_baseline",
        "repeat_abs",
        "repeat_ok",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in stageb_logs:
            writer.writerow({k: row.get(k) for k in keys})


def write_best_of_type_csv(path: str, rows: List[Dict]) -> None:
    keys = ["action_type", "layer_name", "cost", "proxy_raw", "proxy_norm", "exact_delta", "exact_norm"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def parse_stratified_quota(s: str) -> Dict[str, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("stratified quota must be: weight,index,bitmask")
    return {
        "weight_bit": int(parts[0]),
        "index_1bit": int(parts[1]),
        "bitmask_swap": int(parts[2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run R1_T05 self-check experiments.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default="results/R1/selfcheck")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--baseline-budget", type=int, default=50)
    parser.add_argument("--short-budget", type=int, default=10)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--equal-sample-n", type=int, default=2000)
    parser.add_argument("--stratified-quota", type=str, default="32,16,16")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"[SelfCheck] device={device}")
    print(f"[SelfCheck] loading data from {args.data_dir}")
    _, test_loader = load_cifar10_loaders_offline(batch_size=256, data_dir=args.data_dir, num_workers=0)

    print(f"[SelfCheck] collecting fixed calib/eval batches")
    # build once using a temporary model/device context
    calib_x, calib_y = collect_fixed_batch(test_loader, device, args.calib_samples)
    eval_x, eval_y = collect_fixed_batch(test_loader, device, args.eval_samples)
    quota = parse_stratified_quota(args.stratified_quota)

    def _fresh_model(seed: int) -> nn.Module:
        set_all_seeds(seed)
        return load_int8_model_from_ckpt(args.ckpt, device)

    monitored_steps = {0, 1, 10, 49}
    invariant_steps = {0, 10}
    repeat_steps = {0, 10}

    # Baseline (global)
    print("[SelfCheck] run baseline(global) ...")
    model = _fresh_model(args.seed)
    baseline = run_attack(
        model=model,
        calib_x=calib_x,
        calib_y=calib_y,
        eval_x=eval_x,
        eval_y=eval_y,
        device=device,
        seed=args.seed,
        physical_budget=args.baseline_budget,
        topk=args.topk,
        stage_a_mode="global",
        stratified_quota=quota,
        equal_sample_n=args.equal_sample_n,
        enable_weight=True,
        enable_index=True,
        enable_bitmask=True,
        monitored_steps=monitored_steps,
        invariant_steps=invariant_steps,
        repeat_steps=repeat_steps,
    )

    # C1 stratified top-k
    print("[SelfCheck] run C1(stratified) ...")
    model = _fresh_model(args.seed)
    c1 = run_attack(
        model=model,
        calib_x=calib_x,
        calib_y=calib_y,
        eval_x=eval_x,
        eval_y=eval_y,
        device=device,
        seed=args.seed,
        physical_budget=args.short_budget,
        topk=args.topk,
        stage_a_mode="stratified",
        stratified_quota=quota,
        equal_sample_n=args.equal_sample_n,
        enable_weight=True,
        enable_index=True,
        enable_bitmask=True,
        monitored_steps={0, min(10, args.short_budget - 1)},
        invariant_steps={0},
        repeat_steps={0},
    )

    # C2 equal sampling
    print("[SelfCheck] run C2(equal sampling) ...")
    model = _fresh_model(args.seed)
    c2 = run_attack(
        model=model,
        calib_x=calib_x,
        calib_y=calib_y,
        eval_x=eval_x,
        eval_y=eval_y,
        device=device,
        seed=args.seed,
        physical_budget=args.short_budget,
        topk=args.topk,
        stage_a_mode="equal",
        stratified_quota=quota,
        equal_sample_n=args.equal_sample_n,
        enable_weight=True,
        enable_index=True,
        enable_bitmask=True,
        monitored_steps={0, min(10, args.short_budget - 1)},
        invariant_steps={0},
        repeat_steps={0},
    )

    # F1/F2/F3 toggles
    print("[SelfCheck] run F1(weight-only) ...")
    model = _fresh_model(args.seed)
    f1 = run_attack(
        model=model,
        calib_x=calib_x,
        calib_y=calib_y,
        eval_x=eval_x,
        eval_y=eval_y,
        device=device,
        seed=args.seed,
        physical_budget=args.short_budget,
        topk=args.topk,
        stage_a_mode="global",
        stratified_quota=quota,
        equal_sample_n=args.equal_sample_n,
        enable_weight=True,
        enable_index=False,
        enable_bitmask=False,
        monitored_steps={0},
        invariant_steps={0},
        repeat_steps={0},
    )

    print("[SelfCheck] run F2(index-only) ...")
    model = _fresh_model(args.seed)
    f2 = run_attack(
        model=model,
        calib_x=calib_x,
        calib_y=calib_y,
        eval_x=eval_x,
        eval_y=eval_y,
        device=device,
        seed=args.seed,
        physical_budget=args.short_budget,
        topk=args.topk,
        stage_a_mode="global",
        stratified_quota=quota,
        equal_sample_n=args.equal_sample_n,
        enable_weight=False,
        enable_index=True,
        enable_bitmask=False,
        monitored_steps={0},
        invariant_steps={0},
        repeat_steps={0},
    )

    print("[SelfCheck] run F3(bitmask-only) ...")
    model = _fresh_model(args.seed)
    f3 = run_attack(
        model=model,
        calib_x=calib_x,
        calib_y=calib_y,
        eval_x=eval_x,
        eval_y=eval_y,
        device=device,
        seed=args.seed,
        physical_budget=args.short_budget,
        topk=args.topk,
        stage_a_mode="global",
        stratified_quota=quota,
        equal_sample_n=args.equal_sample_n,
        enable_weight=False,
        enable_index=False,
        enable_bitmask=True,
        monitored_steps={0},
        invariant_steps={0},
        repeat_steps={0},
    )

    summary = {
        "baseline_global": baseline,
        "c1_stratified": c1,
        "c2_equal_sampling": c2,
        "f1_weight_only": f1,
        "f2_index_only": f2,
        "f3_bitmask_only": f3,
        "config": {
            "seed": args.seed,
            "baseline_budget": args.baseline_budget,
            "short_budget": args.short_budget,
            "topk": args.topk,
            "calib_samples": args.calib_samples,
            "eval_samples": args.eval_samples,
            "equal_sample_n": args.equal_sample_n,
            "stratified_quota": quota,
            "device": device,
            "ckpt": os.path.abspath(args.ckpt),
        },
    }

    summary_json = os.path.join(args.out_dir, "R1_T05_selfcheck_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SelfCheck] wrote {summary_json}")

    # Step-stat logs
    write_step_stats_csv(
        os.path.join(args.out_dir, "R1_T05_baseline_step_stats.csv"),
        baseline,
    )
    write_step_stats_csv(
        os.path.join(args.out_dir, "R1_T05_c1_stratified_step_stats.csv"),
        c1,
    )
    write_step_stats_csv(
        os.path.join(args.out_dir, "R1_T05_c2_equal_step_stats.csv"),
        c2,
    )

    # Stage-B candidate logs (monitored steps)
    for step_key, rows in baseline["stageb_logs_by_step"].items():
        write_stageb_csv(
            os.path.join(args.out_dir, f"R1_T05_baseline_stageb_step{step_key}.csv"),
            rows,
        )
    write_best_of_type_csv(
        os.path.join(args.out_dir, "R1_T05_baseline_best_of_type_step0.csv"),
        baseline["best_of_type_step0"],
    )

    print("[SelfCheck] done")


if __name__ == "__main__":
    main()
