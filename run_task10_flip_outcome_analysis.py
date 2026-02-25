#!/usr/bin/env python3
"""
Task 10: Flip Outcome Taxonomy + Effective-Rewire Analysis

Goal:
  Decompose attempted index flips into outcome categories and explain plateau effects.

This script:
  1) Parses existing Task4/5 outputs (and Tasks 6-8 if present).
  2) Uses Task9 collision semantics to decide whether CSR collisions behave like merge vs delete.
  3) Produces:
     - Stacked bar chart of outcome proportions per task.
     - Accuracy vs effective rewires curve.

Outputs (fixed filenames):
  - results/task10_flip_outcome_breakdown.png
  - results/task10_acc_vs_effective_rewire.png
  - results/task10_flip_outcome_summary.pkl
  - results/task10_flip_outcome_log.txt

Constraints:
  - CPU-only
  - No extra deps beyond torch/torchvision/matplotlib/pickle
"""

from __future__ import annotations

import argparse
import io
import os
import pickle
import random
import types
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import matplotlib.pyplot as plt

from bfa.encoded_sparse_attack import simulate_csr_index_attack
from models.sparse_csr import create_csr_model_from_sparse


OUTCOME_ORDER = ["rewire", "delete", "merge", "clamp-noop", "invalid-skipped", "other"]


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class _SafeUnpickler(pickle.Unpickler):
    """
    Some older result.pkls were created from scripts that defined result classes in __main__.
    This unpickler maps unknown classes to SimpleNamespace so we can still access attributes.
    """

    def find_class(self, module, name):  # noqa: D401
        try:
            return super().find_class(module, name)
        except Exception:
            return types.SimpleNamespace


def _safe_pickle_load(path: str):
    with open(path, "rb") as f:
        return _SafeUnpickler(f).load()


def _as_dict(obj) -> Dict:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return {"value": obj}


def _dominant_csr_collision_effect_from_task9(task9_pkl: str) -> Tuple[str, Dict]:
    """
    Returns:
      collision_effect: "merge" or "delete" (what a duplicate index effectively does)
      task9_info: dict with raw freq
    """
    if not os.path.exists(task9_pkl):
        # Default to current repo behavior: overwrite-last => delete.
        return "delete", {"warning": "Task9 result missing; defaulting CSR collision effect to delete."}

    obj = _safe_pickle_load(task9_pkl)
    d = _as_dict(obj)
    freq = d.get("freq", {})

    # Prefer csr_conv if present, else csr_linear.
    kind = "csr_conv" if "csr_conv" in freq else ("csr_linear" if "csr_linear" in freq else None)
    if kind is None:
        return "delete", {"warning": "Task9 freq missing; defaulting CSR collision effect to delete."}

    # Only compare the 4 oracle behaviors; ignore "other".
    oracle_keys = ["merge_add", "mask_first", "mask_last", "drop"]
    counts = {k: int(freq[kind].get(k, 0)) for k in oracle_keys}
    dominant = max(counts.items(), key=lambda kv: kv[1])[0]
    if dominant == "merge_add":
        return "merge", {"kind": kind, "dominant": dominant, "counts": counts}
    return "delete", {"kind": kind, "dominant": dominant, "counts": counts}


def _get_csr_layers(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    layers: List[Tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if hasattr(module, "csr_values") and hasattr(module, "csr_column_indices") and hasattr(module, "csr_row_ptr"):
            if int(getattr(module, "csr_values").numel()) > 0:
                layers.append((name, module))
    return layers


def _effective_row_map(
    cols: torch.Tensor,
    vals: torch.Tensor,
    start: int,
    end: int,
) -> Dict[int, int]:
    """
    CSR decode uses 'last write wins' semantics:
      dense[row, col] = val * scale
    For a given row segment [start,end), return map col -> last int8 value.
    """
    out: Dict[int, int] = {}
    # Keep this in Python loop; row lengths are modest for ResNet20.
    for i in range(start, end):
        out[int(cols[i].item())] = int(vals[i].item())
    return out


def _row_of_csr_idx(row_ptr: torch.Tensor, csr_idx: int) -> int:
    # row_ptr is int32 tensor of length out_channels+1.
    # torch.searchsorted returns insertion index.
    idx_t = torch.tensor([csr_idx], dtype=row_ptr.dtype, device=row_ptr.device)
    row = int(torch.searchsorted(row_ptr, idx_t, right=True).item()) - 1
    return max(0, row)


def _analyze_task4_attempt_space(
    csr_model: torch.nn.Module,
    collision_effect: str,
) -> Dict[str, int]:
    """
    Enumerate all possible index bit flips on the *initial* CSR model state.
    This approximates the distribution of outcomes for random/blind flips.
    """
    counts = {k: 0 for k in OUTCOME_ORDER}
    attempted = 0

    for layer_name, module in _get_csr_layers(csr_model):
        weight = module.weight
        in_features = int(weight.shape[1] * weight.shape[2] * weight.shape[3])
        bit_count = min(16, int(in_features).bit_length())

        cols = module.csr_column_indices
        row_ptr = module.csr_row_ptr

        out_channels = int(row_ptr.numel() - 1)
        for row in range(out_channels):
            start = int(row_ptr[row].item())
            end = int(row_ptr[row + 1].item())
            if end <= start:
                continue

            # Row set for fast collision checks.
            row_cols = set(int(cols[i].item()) for i in range(start, end))

            for i in range(start, end):
                old_col = int(cols[i].item())
                for bit_pos in range(bit_count):
                    attempted += 1
                    new_col = simulate_csr_index_attack(old_col, bit_pos, in_features)
                    if new_col == old_col:
                        counts["clamp-noop"] += 1
                        continue
                    if new_col in row_cols:
                        if collision_effect == "merge":
                            counts["merge"] += 1
                        else:
                            counts["delete"] += 1
                        continue
                    counts["rewire"] += 1

    counts["_attempted_total"] = attempted
    return counts


def _classify_task4_executed_flips(
    csr_model: torch.nn.Module,
    task4_bit_positions: List[Tuple[str, int, int]],
    collision_effect: str,
) -> List[Dict]:
    """
    Replay Task4's executed flips on a fresh CSR model and classify each flip.
    """
    layers = dict(_get_csr_layers(csr_model))
    per_flip: List[Dict] = []

    for step, (layer_name, csr_idx, bit_pos) in enumerate(task4_bit_positions, start=1):
        module = layers.get(layer_name)
        if module is None:
            per_flip.append({
                "step": step,
                "layer": layer_name,
                "csr_idx": int(csr_idx),
                "bit_pos": int(bit_pos),
                "category": "other",
                "reason": "layer_not_found",
            })
            continue

        weight = module.weight
        in_features = int(weight.shape[1] * weight.shape[2] * weight.shape[3])

        cols = module.csr_column_indices
        vals = module.csr_values
        row_ptr = module.csr_row_ptr

        if int(csr_idx) < 0 or int(csr_idx) >= int(cols.numel()):
            per_flip.append({
                "step": step,
                "layer": layer_name,
                "csr_idx": int(csr_idx),
                "bit_pos": int(bit_pos),
                "category": "other",
                "reason": "csr_idx_oob",
            })
            continue

        row = _row_of_csr_idx(row_ptr, int(csr_idx))
        start = int(row_ptr[row].item())
        end = int(row_ptr[row + 1].item())

        old_col = int(cols[int(csr_idx)].item())
        new_col = int(simulate_csr_index_attack(old_col, int(bit_pos), in_features))

        # Row mapping before flip.
        map_before = _effective_row_map(cols, vals, start, end)

        # Collision check before flip (excluding self).
        collision = False
        if new_col != old_col:
            for i in range(start, end):
                if i == int(csr_idx):
                    continue
                if int(cols[i].item()) == new_col:
                    collision = True
                    break

        # Apply flip (as attack engine does).
        with torch.no_grad():
            cols[int(csr_idx)] = torch.tensor(new_col, dtype=cols.dtype, device=cols.device)

        map_after = _effective_row_map(cols, vals, start, end)
        mapping_unchanged = (map_after == map_before)

        if mapping_unchanged:
            category = "clamp-noop"
            reason = "decoded_mapping_unchanged"
        elif new_col == old_col:
            category = "clamp-noop"
            reason = "clamp_no_change"
        elif collision:
            category = "merge" if collision_effect == "merge" else "delete"
            reason = "row_collision"
        else:
            category = "rewire"
            reason = "moved_to_empty_col"

        per_flip.append({
            "step": step,
            "layer": layer_name,
            "csr_idx": int(csr_idx),
            "bit_pos": int(bit_pos),
            "row": int(row),
            "old_col": int(old_col),
            "new_col": int(new_col),
            "collision": bool(collision),
            "mapping_unchanged": bool(mapping_unchanged),
            "category": category,
            "reason": reason,
        })

    return per_flip


def _task5plus_attempt_counts(result_dict: Dict) -> Optional[Dict[str, int]]:
    if "attempted_flips" not in result_dict or "collisions_skipped" not in result_dict:
        return None
    attempted = int(result_dict["attempted_flips"])
    collisions = int(result_dict["collisions_skipped"])
    counts = {k: 0 for k in OUTCOME_ORDER}
    counts["invalid-skipped"] = collisions
    counts["rewire"] = max(0, attempted - collisions)
    counts["_attempted_total"] = attempted
    return counts


def _extract_accuracy_history(result_dict: Dict) -> Optional[List[float]]:
    hist = result_dict.get("accuracy_history")
    if isinstance(hist, list) and hist:
        return [float(x) for x in hist]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 10: Flip Outcome Taxonomy + Effective-Rewire Analysis")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--include-imagenet-tasks", action="store_true",
                        help="Also include Tasks 6-8 in plots if their result.pkls exist.")
    args = parser.parse_args()

    device = "cpu"
    _set_seed(args.seed)
    os.makedirs("results", exist_ok=True)

    log_path = "results/task10_flip_outcome_log.txt"
    out_breakdown_png = "results/task10_flip_outcome_breakdown.png"
    out_acc_png = "results/task10_acc_vs_effective_rewire.png"
    out_pkl = "results/task10_flip_outcome_summary.pkl"

    # Decide collision semantics from Task9.
    collision_effect, task9_info = _dominant_csr_collision_effect_from_task9(
        "results/task9_collision_characterization_result.pkl"
    )

    # Load Task4 result (encoded CSR index attack).
    task4_pkl = "results/task4_sparse_csr_index_result.pkl"
    task4_log = "results/task4_sparse_csr_index_log.txt"
    task4_obj = _safe_pickle_load(task4_pkl) if os.path.exists(task4_pkl) else None
    task4_d = _as_dict(task4_obj) if task4_obj is not None else {}

    # Load Task5+ results.
    task_paths = [
        ("Task5", "results/task5_csr_non_collision_result.pkl", "results/task5_csr_non_collision_log.txt"),
    ]
    if args.include_imagenet_tasks:
        task_paths += [
            ("Task6", "results/task6_resnet18_csr_non_collision_result.pkl", "results/task6_resnet18_csr_non_collision_log.txt"),
            ("Task7", "results/task7_mobilenetv2_csr_non_collision_result.pkl", "results/task7_mobilenetv2_csr_non_collision_log.txt"),
            ("Task8", "results/task8_deit_tiny_csr_non_collision_result.pkl", "results/task8_deit_tiny_csr_non_collision_log.txt"),
        ]

    task5plus_results: Dict[str, Dict] = {}
    for name, pkl_path, _ in task_paths:
        if os.path.exists(pkl_path):
            obj = _safe_pickle_load(pkl_path)
            task5plus_results[name] = _as_dict(obj)

    # Task4: build CSR model (used for attempt space enumeration + executed flip replay).
    task4_attempt_counts = None
    task4_executed_records = None
    if task4_obj is not None:
        csr_model = create_csr_model_from_sparse(device=device)
        csr_model.eval()
        task4_attempt_counts = _analyze_task4_attempt_space(csr_model=csr_model, collision_effect=collision_effect)
        task4_executed_records = _classify_task4_executed_flips(
            csr_model=csr_model,
            task4_bit_positions=list(task4_d.get("bit_positions", [])),
            collision_effect=collision_effect,
        )

    # Aggregate per-task attempted outcome distributions.
    per_task_counts: Dict[str, Dict[str, int]] = {}
    per_task_attempted_total: Dict[str, int] = {}

    if task4_attempt_counts is not None:
        per_task_counts["Task4"] = {k: int(task4_attempt_counts.get(k, 0)) for k in OUTCOME_ORDER}
        per_task_attempted_total["Task4"] = int(task4_attempt_counts.get("_attempted_total", 0))

    for tname, tdict in task5plus_results.items():
        counts = _task5plus_attempt_counts(tdict)
        if counts is None:
            continue
        per_task_counts[tname] = {k: int(counts.get(k, 0)) for k in OUTCOME_ORDER}
        per_task_attempted_total[tname] = int(counts.get("_attempted_total", 0))

    # Build stacked bar plot of outcome proportions.
    tasks_for_plot = [t for t in ["Task4", "Task5", "Task6", "Task7", "Task8"] if t in per_task_counts]
    colors = {
        "rewire": "#2ca02c",
        "delete": "#d62728",
        "merge": "#9467bd",
        "clamp-noop": "#7f7f7f",
        "invalid-skipped": "#ff7f0e",
        "other": "#1f77b4",
    }

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    bottoms = [0.0 for _ in tasks_for_plot]
    for outcome in OUTCOME_ORDER:
        vals = []
        for t in tasks_for_plot:
            total = max(1, per_task_attempted_total.get(t, 0))
            vals.append(float(per_task_counts[t].get(outcome, 0)) / float(total))
        ax.bar(tasks_for_plot, vals, bottom=bottoms, label=outcome, color=colors.get(outcome, None))
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Proportion of Attempted Flips")
    ax.set_title("Task 10: Attempted Flip Outcome Breakdown")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_breakdown_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Build accuracy vs effective rewires curve.
    curves: Dict[str, Dict[str, List[float]]] = {}

    # Task4 curve uses effective rewires from executed flip classification.
    if task4_obj is not None and task4_executed_records is not None:
        acc_hist = _extract_accuracy_history(task4_d)
        if acc_hist is not None and len(acc_hist) == (1 + len(task4_executed_records)):
            eff = [0]
            c = 0
            for rec in task4_executed_records:
                if rec.get("category") == "rewire":
                    c += 1
                eff.append(float(c))
            curves["Task4"] = {"x_eff_rewires": eff, "y_acc": acc_hist}

    # Task5+ curves: each successful flip is a rewire, so x = flips.
    for tname, tdict in task5plus_results.items():
        acc_hist = _extract_accuracy_history(tdict)
        if acc_hist is None:
            continue
        x = list(range(len(acc_hist)))
        curves[tname] = {"x_eff_rewires": [float(v) for v in x], "y_acc": [float(v) for v in acc_hist]}

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    for t in ["Task4", "Task5", "Task6", "Task7", "Task8"]:
        if t not in curves:
            continue
        ax.plot(
            curves[t]["x_eff_rewires"],
            curves[t]["y_acc"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=t,
        )
    ax.set_xlabel("Effective Rewires (cumulative)")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Task 10: Accuracy vs Effective Rewires")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(0, 100)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_acc_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save summary pkl.
    summary = {
        "timestamp": _now_ts(),
        "seed": int(args.seed),
        "device": device,
        "task9_collision_effect": collision_effect,
        "task9_info": task9_info,
        "attempted_outcome_counts": per_task_counts,
        "attempted_totals": per_task_attempted_total,
        "task4_executed_flip_records": task4_executed_records,
        "curves": curves,
        "assumptions": [
            "Task4 attempt taxonomy is computed by enumerating all possible index-bit flips on the initial CSR model state "
            "(pre-attack), because Task4 result.pkl does not store per-attempt counters.",
            "Task5-8 attempted counts use the attack engine counters: attempted_flips and collisions_skipped.",
            "CSR collision effect (merge vs delete) is inferred from Task9 microbench dominant behavior.",
        ],
        "inputs": {
            "task4_pkl": task4_pkl if os.path.exists(task4_pkl) else None,
            "task4_log": task4_log if os.path.exists(task4_log) else None,
            "task5plus_pkls": {n: p for (n, p, _) in task_paths if os.path.exists(p)},
        },
    }
    with open(out_pkl, "wb") as f:
        pickle.dump(summary, f)

    # Write log.
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("=" * 100 + "\n")
        log.write("Task 10: Flip Outcome Taxonomy + Effective-Rewire Analysis\n")
        log.write("=" * 100 + "\n")
        log.write(f"Script: run_task10_flip_outcome_analysis.py\n")
        log.write(f"Timestamp: {_now_ts()}\n")
        log.write(f"Seed: {args.seed}\n")
        log.write(f"Device: {device}\n")
        log.write("Dataset path: N/A (offline analysis)\n")
        log.write("Model checkpoint path: models/sparse_model.pth (for Task4 structural replay)\n")
        log.write("\n")

        log.write("== Task9-Inferred CSR Collision Effect ==\n")
        log.write(f"- collision_effect: {collision_effect}\n")
        log.write(f"- task9_info: {task9_info}\n")
        log.write("\n")

        log.write("== Attempted Flip Outcome Counts ==\n")
        for t in tasks_for_plot:
            total = per_task_attempted_total.get(t, 0)
            log.write(f"\n[{t}] attempted_total={total}\n")
            for k in OUTCOME_ORDER:
                log.write(f"- {k:14s}: {per_task_counts[t].get(k, 0)}\n")

        if task4_executed_records is not None:
            exec_counts = {k: 0 for k in OUTCOME_ORDER}
            for rec in task4_executed_records:
                cat = rec.get("category", "other")
                exec_counts[cat] = exec_counts.get(cat, 0) + 1
            log.write("\n== Task4 Executed Flip Categories (50 flips) ==\n")
            for k in OUTCOME_ORDER:
                log.write(f"- {k:14s}: {exec_counts.get(k, 0)}\n")

        log.write("\nOutputs:\n")
        log.write(f"- {out_breakdown_png}\n")
        log.write(f"- {out_acc_png}\n")
        log.write(f"- {out_pkl}\n")
        log.write(f"- {log_path}\n")
        log.write("\nHow to run:\n")
        log.write("  python run_task10_flip_outcome_analysis.py --include-imagenet-tasks\n")

    print(f"[Task10] Wrote: {out_breakdown_png}")
    print(f"[Task10] Wrote: {out_acc_png}")
    print(f"[Task10] Wrote: {out_pkl}")
    print(f"[Task10] Wrote: {log_path}")


if __name__ == "__main__":
    main()
