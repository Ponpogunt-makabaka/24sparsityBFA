#!/usr/bin/env python3
"""
Task 23: Minimal Closed-Loop Summary (Case A/B/C) on ResNet-20 / CIFAR-10

Combines Tasks 18-22 into a single paper-ready figure + table:
  - Case A: dense weight MSB (Task22)
  - Case B: bitmask nonzero weight MSB (Task19)
  - Case B (optional): bitmask swap cost=2 (Task20)
  - Case C: position metadata NCSA (Task21)
  - Case C: position weight MSB (Task21)

Outputs:
  - results/task23_miniclose_curves.png
  - results/task23_miniclose_table.csv
  - results/task23_miniclose_log.txt
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _extract_curve(obj, kind: str) -> Optional[List[float]]:
    """
    kind:
      - "payload_result": obj["result"]["accuracy_history"]
      - "plain_result": obj["accuracy_history"]
    """
    if kind == "payload_result":
        if isinstance(obj, dict) and "result" in obj and isinstance(obj["result"], dict):
            return [float(x) for x in obj["result"].get("accuracy_history", [])]
        return None
    if kind == "plain_result":
        if isinstance(obj, dict):
            return [float(x) for x in obj.get("accuracy_history", [])]
        return None
    raise ValueError(kind)


def _row(
    case: str,
    attack: str,
    cost_per_step: int,
    physical_budget: int,
    logical_steps: int,
    curve: List[float],
    notes: str,
) -> Dict:
    init_acc = float(curve[0]) if curve else float("nan")
    final_acc = float(curve[-1]) if curve else float("nan")
    return {
        "case": case,
        "attack": attack,
        "cost_per_step": int(cost_per_step),
        "physical_budget": int(physical_budget),
        "logical_steps": int(logical_steps),
        "baseline_acc": init_acc,
        "final_acc": final_acc,
        "drop": init_acc - final_acc,
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task23: minimal closed-loop summary")
    parser.add_argument("--out-curves", type=str, default="results/task23_miniclose_curves.png")
    parser.add_argument("--out-table", type=str, default="results/task23_miniclose_table.csv")
    parser.add_argument("--out-log", type=str, default="results/task23_miniclose_log.txt")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # Input pkls (expected).
    p_task22 = "results/task22_dense_weight_msb_result.pkl"
    p_task19 = "results/task19_bitmask_weight_msb_result.pkl"
    p_task20 = "results/task20_bitmask_swap_result.pkl"
    p_task21_ncsa = "results/task21_position_ncsa_result.pkl"
    p_task21_msb = "results/task21_position_weight_msb_result.pkl"

    entries: List[Tuple[str, str, int, int, int, List[float], str]] = []
    loaded: List[str] = []

    if os.path.exists(p_task22):
        obj = _load_pkl(p_task22)
        curve = _extract_curve(obj, "payload_result")
        if curve:
            entries.append(("A_dense", "dense_weight_msb", 1, 50, len(curve) - 1, curve, p_task22))
            loaded.append(p_task22)

    if os.path.exists(p_task19):
        obj = _load_pkl(p_task19)
        curve = _extract_curve(obj, "payload_result")
        if curve:
            entries.append(("B_bitmask", "bitmask_weight_msb_nonzero", 1, 50, len(curve) - 1, curve, p_task19))
            loaded.append(p_task19)

    if os.path.exists(p_task20):
        obj = _load_pkl(p_task20)
        curve = _extract_curve(obj, "payload_result")
        if curve:
            # cost=2, x-axis will be physical flips (2*logical).
            entries.append(("B_bitmask", "bitmask_swap_cost2", 2, 50, len(curve) - 1, curve, p_task20))
            loaded.append(p_task20)

    if os.path.exists(p_task21_ncsa):
        obj = _load_pkl(p_task21_ncsa)
        curve = _extract_curve(obj, "plain_result")
        if curve:
            entries.append(("C_position", "position_metadata_ncsa", 1, 50, len(curve) - 1, curve, p_task21_ncsa))
            loaded.append(p_task21_ncsa)

    if os.path.exists(p_task21_msb):
        obj = _load_pkl(p_task21_msb)
        curve = _extract_curve(obj, "plain_result")
        if curve:
            entries.append(("C_position", "position_weight_msb_nonzero", 1, 50, len(curve) - 1, curve, p_task21_msb))
            loaded.append(p_task21_msb)

    # Write table.
    table_rows: List[Dict] = []
    for case, attack, cost, phys, logical, curve, src in entries:
        table_rows.append(_row(case, attack, cost, phys, logical, curve, notes=src))

    with open(args.out_table, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "attack",
                "cost_per_step",
                "physical_budget",
                "logical_steps",
                "baseline_acc",
                "final_acc",
                "drop",
                "notes",
            ],
        )
        w.writeheader()
        for r in table_rows:
            w.writerow(r)

    # Plot curves with x-axis in PHYSICAL flips.
    plt.figure(figsize=(11, 6))
    for case, attack, cost, phys, logical, curve, _src in entries:
        if not curve:
            continue
        if cost == 1:
            xs = list(range(len(curve)))
        else:
            xs = [i * cost for i in range(len(curve))]
        plt.plot(xs, curve, marker="o", linewidth=2, markersize=3, label=f"{case}:{attack} (cost={cost})")

    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlabel("Physical Flips", fontsize=12, fontweight="bold")
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title("Task23: Minimal Closed-Loop (ResNet-20 / CIFAR-10)", fontsize=14, fontweight="bold")
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out_curves, dpi=150, bbox_inches="tight")
    plt.close()

    # Log.
    with open(args.out_log, "w", encoding="utf-8") as log:
        log.write("=" * 100 + "\n")
        log.write("Task 23: Minimal Closed-Loop Summary\n")
        log.write("=" * 100 + "\n")
        log.write(f"Script: run_task23_miniclose_summary.py\n")
        log.write(f"Timestamp: {_now_ts()}\n")
        log.write("\nLoaded sources:\n")
        for p in loaded:
            log.write(f"- {p}\n")
        log.write("\nTable:\n")
        for r in table_rows:
            log.write(f"- {r['case']} {r['attack']} (cost={r['cost_per_step']}): {r['baseline_acc']:.2f}% -> {r['final_acc']:.2f}%\n")
        log.write("\nArtifacts:\n")
        log.write(f"- {args.out_curves}\n")
        log.write(f"- {args.out_table}\n")
        log.write("\nHow to run:\n")
        log.write("  python run_task23_miniclose_summary.py\n")

    print(f"[Task23] Wrote: {args.out_curves}")
    print(f"[Task23] Wrote: {args.out_table}")
    print(f"[Task23] Wrote: {args.out_log}")


if __name__ == "__main__":
    main()

