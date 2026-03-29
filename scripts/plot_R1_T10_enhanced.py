#!/usr/bin/env python3
"""
Plot R1_T10_Enhanced accuracy trajectory from results JSON.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt


def load_curve(json_path: str) -> Tuple[List[int], List[float], dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON in {json_path}, got {type(payload)}")

    initial_acc = float(payload["initial_acc"])
    attack_history = payload.get("attack_history", [])
    attack_history = sorted(attack_history, key=lambda x: int(x["flip"]))

    flips = [0]
    acc = [initial_acc]
    for item in attack_history:
        flips.append(int(item["flip"]))
        acc.append(float(item["acc_after"]))
    return flips, acc, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot R1_T10_Enhanced trajectory.")
    parser.add_argument("--json", default="results/R1_T10_enhanced/results.json")
    parser.add_argument("--out-png", default="results/R1_T10_enhanced/fig_T10_enhanced_accuracy.png")
    parser.add_argument("--out-pdf", default="results/R1_T10_enhanced/fig_T10_enhanced_accuracy.pdf")
    args = parser.parse_args()

    x, y, run = load_curve(args.json)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        x,
        y,
        color="#2ca02c",
        marker="o",
        markersize=4,
        linewidth=2.2,
        label="T10_Enhanced",
    )

    ax.set_xlabel("Number of Flips", fontsize=13)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=13)
    ax.set_title("R1_T10_Enhanced Attack Trajectory", fontsize=14)
    ax.set_xlim(0, max(x))
    ax.set_ylim(0, 100)
    ax.axhline(y=10, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.legend(loc="upper right", fontsize=11, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.45)

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_pdf), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=300, bbox_inches="tight")
    fig.savefig(args.out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] source: {args.json}")
    print(f"[OK] method: {run.get('method')}")
    print(f"[OK] initial_acc: {float(run['initial_acc']):.2f}%")
    print(f"[OK] final_acc: {float(run['final_acc']):.2f}%")
    print(f"[OK] flips: {len(run.get('attack_history', []))}")
    print(f"[OK] saved PNG: {args.out_png}")
    print(f"[OK] saved PDF: {args.out_pdf}")


if __name__ == "__main__":
    main()
