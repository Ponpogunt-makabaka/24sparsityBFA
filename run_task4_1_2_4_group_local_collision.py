#!/usr/bin/env python3
"""
Task 4.1: 2:4 Group-Local Position Index Attack (Collision Allowed)

Objective:
  - Attack the 2-bit (0/1) group-local indices within each 2:4 group (indices in {0,1,2,3}).
  - Use the same ResNet-20 CIFAR-10 2:4 sparse INT8 PTQ pipeline as Task5-style NCSA.
  - Remove the non-collision constraint: collisions are allowed and must be applied.

Collision semantics (IMPORTANT, repo-established):
  - This repo stores 2:4 sparsity via (int8_weights, sparse_mask) in a dense layout.
  - We apply the exact same update rule as Task5's non-collision move:
      * move old value to new index (overwrite),
      * set old slot to 0,
      * toggle mask bits at old and new with XOR 1.
  - When the target index is already active (collision), XOR toggles the target mask OFF,
    and old mask also toggles OFF, effectively dropping both non-zeros in that group.
  - This behavior was verified in Task9 ("nm_linear" collision => drop).

Outputs (fixed filenames under results/):
  - results/task4_1_2_4_group_local_collision_result.pkl
  - results/task4_1_2_4_group_local_collision_log.txt
  - results/task4_1_2_4_group_local_collision.png
  - results/task4_1_vs_task5_attack_curves.png
  - results/task4_1_vs_task5_summary.csv

CPU-only by default; no extra deps beyond torch/torchvision/matplotlib/pickle.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import random
import types
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from scripts.p012_17_utils import (
    flatten_groups,
    get_sparse_layers,
    load_cifar10_loaders_offline,
    load_sparse_int8_resnet20,
    now_ts,
    run_scored_noncollision_attack,
    set_all_seeds,
    evaluate_subset,
)

from bfa.csr_non_collision_attack import _apply_non_collision_move  # noqa: SLF001


@dataclass(frozen=True)
class _ChosenFlip:
    score: float
    layer_name: str
    module: nn.Module
    group_idx: int
    old_idx: int
    new_idx: int
    bit_pos: int
    w_fp: float
    grad_old: float
    grad_new: float
    grad_delta: float
    is_collision: bool


class _SafeUnpickler(pickle.Unpickler):
    """
    Some older pkls were created from scripts defining result classes in __main__.
    Map unknown classes to SimpleNamespace so we can still access attributes.
    """

    def find_class(self, module, name):  # noqa: D401
        try:
            return super().find_class(module, name)
        except Exception:
            return types.SimpleNamespace


def _safe_pickle_load(path: str):
    with open(path, "rb") as f:
        return _SafeUnpickler(f).load()


def _select_best(best: Optional[_ChosenFlip], cand: _ChosenFlip) -> _ChosenFlip:
    if best is None:
        return cand
    # Deterministic tie-break (mirrors scripts/p012_17_utils.select_candidate ordering).
    b_key = (best.score, best.layer_name, -best.group_idx, -best.old_idx, -best.new_idx)
    c_key = (cand.score, cand.layer_name, -cand.group_idx, -cand.old_idx, -cand.new_idx)
    return cand if c_key > b_key else best


def _build_best_candidate_collision_allowed(
    model: nn.Module,
    calib_loader,
    device: str,
    calib_samples: int,
    flipped: set,
    counters: Dict[str, int],
    max_groups_per_layer: int,
    allow_nonpositive: bool,
    rng: random.Random,
) -> Optional[_ChosenFlip]:
    """
    Enumerate candidates over group-local 2-bit indices, allowing collisions.
    Greedily returns the best candidate under score = w_fp * (g_new - g_old).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    calib_data = []
    calib_targets = []
    for inputs, targets in calib_loader:
        calib_data.append(inputs.to(device))
        calib_targets.append(targets.to(device))
        if len(calib_data) * int(inputs.size(0)) >= int(calib_samples):
            break
    calib_inputs = torch.cat(calib_data, dim=0)[:calib_samples]
    calib_targets = torch.cat(calib_targets, dim=0)[:calib_samples]

    outputs = model(calib_inputs)
    loss = criterion(outputs, calib_targets)
    model.zero_grad()
    loss.backward()

    best: Optional[_ChosenFlip] = None
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
        if num_groups <= max_groups_per_layer:
            group_indices = range(num_groups)
        else:
            # Deterministic sampling.
            group_indices = rng.sample(range(num_groups), k=max_groups_per_layer)

        for g_idx in group_indices:
            m_group = m_flat[g_idx]
            active = (m_group > 0.5).nonzero(as_tuple=False).flatten().tolist()
            if len(active) != 2:
                continue

            a_idx, b_idx = int(active[0]), int(active[1])
            for old_idx, _neighbor_idx in ((a_idx, b_idx), (b_idx, a_idx)):
                w_val = int(w_flat[g_idx, old_idx].item())
                if w_val == 0:
                    continue
                w_fp = float(w_val) * scale
                g_old = float(g_flat[g_idx, old_idx].item())

                for bit_pos in (0, 1):
                    counters["attempted"] += 1
                    new_idx = int(old_idx ^ (1 << bit_pos))

                    key = (layer_name, int(g_idx), int(old_idx), int(bit_pos))
                    if key in flipped:
                        continue

                    g_new = float(g_flat[g_idx, new_idx].item())
                    grad_delta = g_new - g_old
                    score = float(w_fp * grad_delta)
                    if (score <= 0.0) and (not allow_nonpositive):
                        continue

                    is_collision = bool(int(m_group[new_idx].item()) == 1)
                    if is_collision:
                        counters["collision_candidates"] += 1

                    cand = _ChosenFlip(
                        score=score,
                        layer_name=layer_name,
                        module=module,
                        group_idx=int(g_idx),
                        old_idx=int(old_idx),
                        new_idx=int(new_idx),
                        bit_pos=int(bit_pos),
                        w_fp=float(w_fp),
                        grad_old=float(g_old),
                        grad_new=float(g_new),
                        grad_delta=float(grad_delta),
                        is_collision=is_collision,
                    )
                    best = _select_best(best, cand)

        if module.weight.grad is not None:
            module.weight.grad = None

    return best


def _group_active_count(module: nn.Module, g_idx: int) -> Optional[int]:
    mask = module.sparse_mask
    m_flat, _ = flatten_groups(mask)
    if m_flat is None:
        return None
    return int((m_flat[g_idx] > 0.5).sum().item())


def _load_task5_curve_if_compatible(
    task5_pkl: str,
    expected_baseline: float,
    tol: float = 2.0,
) -> Optional[Tuple[str, List[float], float, float, int, str]]:
    """
    Returns (label, acc_curve, baseline_acc, final_acc, flips, notes) if compatible.
    """
    if not os.path.exists(task5_pkl):
        return None
    try:
        obj = _safe_pickle_load(task5_pkl)
    except Exception:
        return None

    # Accept dict-like or attribute-like.
    if isinstance(obj, dict):
        acc_hist = obj.get("accuracy_history") or obj.get("acc_hist") or obj.get("acc_curve")
        acc0 = obj.get("initial_accuracy") or obj.get("baseline_acc") or (acc_hist[0] if acc_hist else None)
    else:
        acc_hist = getattr(obj, "accuracy_history", None)
        acc0 = getattr(obj, "initial_accuracy", None)
        if acc0 is None and acc_hist is not None:
            acc0 = acc_hist[0]

    if acc_hist is None or acc0 is None:
        return None
    acc_hist = [float(x) for x in list(acc_hist)]
    acc0 = float(acc0)

    if abs(acc0 - float(expected_baseline)) > float(tol):
        notes = f"stale_or_drift: task5_baseline={acc0:.2f} vs expected={expected_baseline:.2f}"
        return ("Task5 (stale)", acc_hist, acc0, float(acc_hist[-1]), int(len(acc_hist) - 1), notes)

    return ("Task5", acc_hist, acc0, float(acc_hist[-1]), int(len(acc_hist) - 1), "loaded_from_existing_pkl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4.1: 2:4 group-local collision-allowed index attack")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-flips", type=int, default=50)
    parser.add_argument("--calib-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out-prefix", type=str, default="task4_1_2_4_group_local_collision")
    parser.add_argument("--max-groups-per-layer", type=int, default=2000,
                        help="Deterministic per-layer group sampling for CPU runtime.")
    parser.add_argument("--allow-nonpositive", action="store_true",
                        help="If set, allow score<=0 candidates (not recommended; default off).")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    set_all_seeds(int(args.seed))

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Fixed output paths required by prompt.
    log_path = f"results/{args.out_prefix}_log.txt"
    out_pkl = f"results/{args.out_prefix}_result.pkl"
    out_png = f"results/{args.out_prefix}.png"
    overlay_png = "results/task4_1_vs_task5_attack_curves.png"
    summary_csv = "results/task4_1_vs_task5_summary.csv"

    ckpt_path = "models/sparse_model.pth"
    data_dir = "./data"

    # Load data/model (Task5-style pipeline).
    _, test_loader = load_cifar10_loaders_offline(batch_size=256, data_dir=data_dir, num_workers=0)
    model = load_sparse_int8_resnet20(device=device, ckpt_path=ckpt_path)

    # Baseline eval.
    acc0, loss0 = evaluate_subset(model, test_loader, device=device, max_samples=int(args.eval_samples))

    # Attack loop (collision allowed).
    rng = random.Random(int(args.seed))
    criterion = nn.CrossEntropyLoss()
    counters: Dict[str, int] = {"attempted": 0, "collision_candidates": 0}
    flipped = set()

    acc_hist: List[float] = [float(acc0)]
    loss_hist: List[float] = [float(loss0)]
    flip_trace: List[Dict] = []

    collision_count = 0
    noncollision_rewire_count = 0
    delete_or_drop_count = 0
    noop_count = 0

    t_wall0 = time.perf_counter()  # type: ignore[name-defined]
    success = 0
    while success < int(args.max_flips):
        chosen = _build_best_candidate_collision_allowed(
            model=model,
            calib_loader=test_loader,
            device=device,
            calib_samples=int(args.calib_samples),
            flipped=flipped,
            counters=counters,
            max_groups_per_layer=int(args.max_groups_per_layer),
            allow_nonpositive=bool(args.allow_nonpositive),
            rng=rng,
        )
        if chosen is None:
            break

        flip_key = (chosen.layer_name, chosen.group_idx, chosen.old_idx, chosen.bit_pos)
        flipped.add(flip_key)

        active_before = _group_active_count(chosen.module, chosen.group_idx)
        ok = _apply_non_collision_move(chosen.module, chosen.group_idx, chosen.old_idx, chosen.new_idx)
        active_after = _group_active_count(chosen.module, chosen.group_idx)

        if not ok:
            noop_count += 1
            continue

        success += 1

        # Outcome bookkeeping.
        if chosen.is_collision:
            collision_count += 1
            delete_or_drop_count += 1
            eff_type = "collision_drop"
        else:
            noncollision_rewire_count += 1
            eff_type = "rewire"

        acc, loss = evaluate_subset(model, test_loader, device=device, max_samples=int(args.eval_samples))
        acc_hist.append(float(acc))
        loss_hist.append(float(loss))

        flip_trace.append({
            "flip": int(success),
            "layer": chosen.layer_name,
            "group": int(chosen.group_idx),
            "old_idx": int(chosen.old_idx),
            "new_idx": int(chosen.new_idx),
            "bit_pos": int(chosen.bit_pos),
            "score": float(chosen.score),
            "w_fp": float(chosen.w_fp),
            "grad_old": float(chosen.grad_old),
            "grad_new": float(chosen.grad_new),
            "grad_delta": float(chosen.grad_delta),
            "is_collision": bool(chosen.is_collision),
            "effective_type": eff_type,
            "active_before": active_before,
            "active_after": active_after,
            "accuracy": float(acc),
            "loss": float(loss),
        })

    wall_sec = float((time.perf_counter() - t_wall0))  # type: ignore[name-defined]
    final_acc = float(acc_hist[-1])

    collision_semantics_note = (
        "Collision allowed; applied via Task5 update rule (move+mask XOR). "
        "If target slot already active, XOR toggles both mask bits OFF => effective drop of both nonzeros "
        "(Task9 nm_linear classified as drop)."
    )

    result = {
        "task_name": "Task4.1_2:4_group_local_collision_allowed",
        "script": "run_task4_1_2_4_group_local_collision.py",
        "timestamp": now_ts(),
        "seed": int(args.seed),
        "device": str(device),
        "dataset": "CIFAR-10",
        "dataset_path": os.path.abspath(data_dir),
        "model": "ResNet-20 (2:4 sparse) + INT8 PTQ",
        "checkpoint_path": os.path.abspath(ckpt_path),
        "max_flips": int(args.max_flips),
        "calib_samples": int(args.calib_samples),
        "eval_samples": int(args.eval_samples),
        "max_groups_per_layer": int(args.max_groups_per_layer),
        "allow_nonpositive": bool(args.allow_nonpositive),
        "baseline_acc": float(acc0),
        "final_acc": float(final_acc),
        "acc_curve": acc_hist,
        "loss_curve": loss_hist,
        "successful_flips": int(success),
        "attempted_candidates": int(counters["attempted"]),
        "collision_candidate_count": int(counters["collision_candidates"]),
        "collision_count": int(collision_count),
        "noncollision_rewire_count": int(noncollision_rewire_count),
        "delete_or_drop_count": int(delete_or_drop_count),
        "noop_count": int(noop_count),
        "notes": {
            "collision_semantics": collision_semantics_note,
        },
        "flip_trace": flip_trace,
        "timing": {
            "wall_sec": float(wall_sec),
        },
    }

    with open(out_pkl, "wb") as f:
        pickle.dump(result, f)

    # Write log.
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("=" * 100 + "\n")
        log.write("Task 4.1: 2:4 Group-Local Position Index Attack (Collision Allowed)\n")
        log.write("=" * 100 + "\n")
        log.write(f"Script: {result['script']}\n")
        log.write(f"Timestamp: {result['timestamp']}\n")
        log.write(f"Seed: {result['seed']}\n")
        log.write(f"Device: {result['device']}\n")
        log.write(f"Dataset path: {result['dataset_path']}\n")
        log.write(f"Model checkpoint path: {result['checkpoint_path']}\n")
        log.write("\nConfig:\n")
        log.write(f"- max_flips: {result['max_flips']}\n")
        log.write(f"- calib_samples: {result['calib_samples']}\n")
        log.write(f"- eval_samples: {result['eval_samples']}\n")
        log.write(f"- max_groups_per_layer: {result['max_groups_per_layer']}\n")
        log.write(f"- allow_nonpositive: {result['allow_nonpositive']}\n")
        log.write("\nAssumptions / Semantics:\n")
        log.write(f"- {collision_semantics_note}\n")
        log.write("\nResults:\n")
        log.write(f"- baseline_acc: {result['baseline_acc']:.2f}%\n")
        log.write(f"- final_acc:    {result['final_acc']:.2f}%\n")
        log.write(f"- drop:         {result['baseline_acc'] - result['final_acc']:.2f} pts\n")
        log.write(f"- successful_flips: {result['successful_flips']}\n")
        log.write("\nOutcome counts (applied flips):\n")
        log.write(f"- collision_count:           {result['collision_count']}\n")
        log.write(f"- noncollision_rewire_count: {result['noncollision_rewire_count']}\n")
        log.write(f"- delete_or_drop_count:      {result['delete_or_drop_count']}\n")
        log.write(f"- noop_count:                {result['noop_count']}\n")
        log.write("\nSearch counters:\n")
        log.write(f"- attempted_candidates:      {result['attempted_candidates']}\n")
        log.write(f"- collision_candidate_count: {result['collision_candidate_count']}\n")
        log.write("\nHow to run:\n")
        log.write("  python run_task4_1_2_4_group_local_collision.py --device cpu --max-flips 50\n")

    # Plot Task4.1 curve.
    plt.figure(figsize=(10, 6))
    xs = list(range(len(acc_hist)))
    plt.plot(xs, acc_hist, marker="o", linewidth=2, markersize=4, label="Task4.1 (collision allowed)")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlabel("Successful Flips", fontsize=12, fontweight="bold")
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title("Task 4.1: 2:4 Group-Local Index Attack (Collision Allowed)", fontsize=14, fontweight="bold")
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    # Load Task5 curve; if stale/drifted, rerun a reference Task5 non-collision NCSA without overwriting.
    task5_pkl = "results/task5_csr_non_collision_result.pkl"
    task5_loaded = _load_task5_curve_if_compatible(task5_pkl, expected_baseline=float(acc0), tol=2.0)
    task5_label = None
    task5_curve: Optional[List[float]] = None
    task5_acc0 = None
    task5_accN = None
    task5_flips = None
    task5_notes = None

    if task5_loaded is not None and task5_loaded[0] != "Task5 (stale)":
        task5_label, task5_curve, task5_acc0, task5_accN, task5_flips, task5_notes = task5_loaded
    else:
        # Rerun a Task5-style non-collision reference (same seed/settings) and cache under a new name.
        ref_model = load_sparse_int8_resnet20(device=device, ckpt_path=ckpt_path)
        ref = run_scored_noncollision_attack(
            model=ref_model,
            test_loader=test_loader,
            calib_loader=test_loader,
            device=device,
            score_mode="ncsa",
            seed=int(args.seed),
            max_success=int(args.max_flips),
            calib_samples=int(args.calib_samples),
            eval_samples=int(args.eval_samples),
            max_groups_per_layer=int(args.max_groups_per_layer),
            allow_nonpositive=False,
            track_trace=False,
            enable_timing=False,
        )
        task5_label = "Task5 (rerun non-collision NCSA)"
        task5_curve = [float(x) for x in ref["accuracy_history"]]
        task5_acc0 = float(ref["initial_accuracy"])
        task5_accN = float(ref["final_accuracy"])
        task5_flips = int(ref["total_flips"])
        task5_notes = "rerun_in_task4_1_runner_due_to_stale_or_incompatible_task5_pkl"

        ref_pkl = f"results/task5_ref_ncsa_non_collision_seed{int(args.seed)}_eval{int(args.eval_samples)}.pkl"
        with open(ref_pkl, "wb") as f:
            pickle.dump(ref, f)

    # Overlay plot (Task4.1 vs Task5).
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(len(acc_hist))), acc_hist, marker="o", linewidth=2, markersize=4, label="Task4.1 (collision allowed)")
    if task5_curve is not None:
        plt.plot(list(range(len(task5_curve))), task5_curve, marker="o", linewidth=2, markersize=4, label=task5_label)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlabel("Successful Flips", fontsize=12, fontweight="bold")
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title("Task 4.1 vs Task 5 (Non-Collision): Attack Curves", fontsize=14, fontweight="bold")
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(overlay_png, dpi=150, bbox_inches="tight")
    plt.close()

    # Summary CSV (Task4.1 + Task5 reference).
    rows = [
        {
            "task": "task4_1_collision_allowed",
            "seed": int(args.seed),
            "baseline_acc": float(acc0),
            "final_acc": float(final_acc),
            "drop": float(acc0 - final_acc),
            "successful_flips": int(success),
            "collision_count": int(collision_count),
            "noncollision_rewire_count": int(noncollision_rewire_count),
            "delete_or_drop_count": int(delete_or_drop_count),
            "noop_count": int(noop_count),
            "notes": collision_semantics_note,
        },
        {
            "task": "task5_non_collision_ncsa",
            "seed": int(args.seed),
            "baseline_acc": float(task5_acc0) if task5_acc0 is not None else float("nan"),
            "final_acc": float(task5_accN) if task5_accN is not None else float("nan"),
            "drop": float(task5_acc0 - task5_accN) if (task5_acc0 is not None and task5_accN is not None) else float("nan"),
            "successful_flips": int(task5_flips) if task5_flips is not None else 0,
            "collision_count": 0,
            "noncollision_rewire_count": int(task5_flips) if task5_flips is not None else 0,
            "delete_or_drop_count": 0,
            "noop_count": 0,
            "notes": str(task5_notes),
        },
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "seed",
                "baseline_acc",
                "final_acc",
                "drop",
                "successful_flips",
                "collision_count",
                "noncollision_rewire_count",
                "delete_or_drop_count",
                "noop_count",
                "notes",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Append comparison takeaways to Task4.1 log (answer prompt questions).
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n" + "=" * 100 + "\n")
        log.write("Comparison vs Task5 (Non-Collision)\n")
        log.write("=" * 100 + "\n")
        if task5_acc0 is not None and task5_accN is not None:
            log.write(f"- Task5 reference label: {task5_label}\n")
            log.write(f"- Task5 baseline/final: {task5_acc0:.2f}% -> {task5_accN:.2f}% (@{int(task5_flips)} flips)\n")
            log.write(f"- Task4.1 baseline/final: {acc0:.2f}% -> {final_acc:.2f}% (@{int(success)} flips)\n")
            log.write("\n(i) Strength vs Task5:\n")
            stronger = "stronger (lower final acc)" if final_acc < task5_accN else "weaker (higher final acc)"
            log.write(f"- Under same flip budget ({int(args.max_flips)}), Task4.1 is {stronger} vs Task5 reference.\n")
            log.write("\n(ii) Collision vs Rewire contribution:\n")
            log.write(f"- collisions_applied={collision_count}/{success} (each collision uses 'drop' semantics).\n")
            log.write(f"- rewires_applied={noncollision_rewire_count}/{success}.\n")
            log.write("\n(iii) Noop/clamp waste vs CSR-absolute index flips (Task4):\n")
            log.write("- Task4.1 flips are always in-range (2-bit index within {0,1,2,3}) so clamp-noop is structurally avoided.\n")
            log.write(f"- Observed noop_count={noop_count} (applied move failures).\n")
        else:
            log.write("- Task5 reference unavailable; overlay plot still generated with Task4.1 only.\n")


if __name__ == "__main__":
    # Defer heavy imports to runtime; keep main guarded.
    import time
    main()

