#!/usr/bin/env python3
"""
Plot paper-ready T08 accuracy-drop curves from topm1/topm3 JSON trajectories.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def resolve_input_path(preferred_path: str, fallback_paths: List[str]) -> str:
    """
    Resolve the input JSON path with deterministic fallback order.
    """
    if os.path.isfile(preferred_path):
        return preferred_path
    for path in fallback_paths:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Cannot find input file '{preferred_path}' or any fallback: {fallback_paths}"
    )


def load_accuracy_curve(json_path: str) -> Tuple[List[int], List[float], Dict]:
    """
    Load flip-index and accuracy trajectory from a T08 result JSON.

    Curve definition:
    - x=0: initial_acc
    - x=1..N: acc_after for each flip
    """
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        if len(payload) == 0:
            raise ValueError(f"JSON list is empty: {json_path}")
        run = payload[0]
    elif isinstance(payload, dict):
        run = payload
    else:
        raise ValueError(f"Unsupported JSON structure in {json_path}: {type(payload)}")

    initial_acc = float(run["initial_acc"])
    attack_history = run.get("attack_history", [])
    attack_history = sorted(attack_history, key=lambda x: int(x["flip"]))

    flips = [0]
    accs = [initial_acc]
    for item in attack_history:
        flips.append(int(item["flip"]))
        accs.append(float(item["acc_after"]))

    return flips, accs, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paper figure for R1_T08 accuracy trajectories.")
    parser.add_argument("--topm1-json", type=str, default="topm1_results.json")
    parser.add_argument("--topm3-json", type=str, default="topm3_results.json")
    parser.add_argument("--out-pdf", type=str, default="results/R1/fig_T08_accuracy_drop.pdf")
    parser.add_argument("--out-png", type=str, default="results/R1/fig_T08_accuracy_drop.png")
    args = parser.parse_args()

    topm1_path = resolve_input_path(
        args.topm1_json,
        [
            "results/R1_T08_topm1_full50_afterfix/topm1_results.json",
            "results/R1_T08_topm1/results.json",
            "results/R1_T08/results.json",
        ],
    )
    topm3_path = resolve_input_path(
        args.topm3_json,
        [
            "results/R1_T08_topm3_full50_afterfix/topm3_results.json",
            "results/R1_T08/results.json",
        ],
    )

    x1, y1, run1 = load_accuracy_curve(topm1_path)
    x3, y3, run3 = load_accuracy_curve(topm3_path)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("default")

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        x1,
        y1,
        label="NCSA (top-M=1)",
        color="#1f77b4",
        marker="o",
        markersize=4,
        linewidth=2.2,
        markevery=5,
    )
    ax.plot(
        x3,
        y3,
        label="NCSA (top-M=3)",
        color="#d62728",
        marker="s",
        markersize=4,
        linewidth=2.2,
        markevery=5,
    )

    ax.set_xlabel("Number of Bit Flips", fontsize=14)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=14)
    ax.set_title("R1_T08 NCSA Attack Trajectory (Full 50 Flips)", fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=12, frameon=True, loc="upper right")

    max_flip = max(max(x1), max(x3))
    ax.set_xlim(0, max_flip)
    ax.set_ylim(0, 100)

    os.makedirs(os.path.dirname(args.out_pdf), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)

    plt.tight_layout()
    fig.savefig(args.out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(args.out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] topm1 source: {topm1_path}")
    print(f"[OK] topm3 source: {topm3_path}")
    print(f"[OK] final acc topm1: {float(run1['final_acc']):.2f}%")
    print(f"[OK] final acc topm3: {float(run3['final_acc']):.2f}%")
    print(f"[OK] saved PDF: {args.out_pdf}")
    print(f"[OK] saved PNG: {args.out_png}")


if __name__ == "__main__":
    main()

