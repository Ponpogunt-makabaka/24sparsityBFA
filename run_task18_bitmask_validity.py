#!/usr/bin/env python3
"""
Task 18: Bitmask Metadata Validity Under 1 Physical Bit Flip (ResNet-20 / CIFAR-10, 2:4)

Reviewer-facing objective:
  - Alternative 2:4 metadata encoding: 4-bit bitmask per group with popcount=2.
  - Show that under a strict "1 physical bit flip" model, flipping ONE metadata bit
    almost always violates popcount=2, so index-move attacks on metadata are not applicable.

This script does NOT modify model/weights. It only analyzes the existing sparse_mask tensors.

Outputs:
  - results/task18_bitmask_validity_log.txt
  - results/task18_bitmask_validity_summary.csv
  - results/task18_bitmask_validity_breakdown.png

Constraints:
  - CPU-only
  - No extra deps beyond torch/torchvision/matplotlib/pickle
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import matplotlib.pyplot as plt

from scripts.p012_17_utils import (
    flatten_groups,
    get_sparse_layers,
    load_cifar10_loaders_offline,
    load_sparse_int8_resnet20,
    set_all_seeds,
    evaluate_subset,
)


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _analyze_layer_mask(m_flat_bin: torch.Tensor) -> Dict[str, int]:
    """
    Args:
      m_flat_bin: [G,4] binary mask (0/1) for a single layer.

    Returns counts over baseline-pop2 groups only.
    """
    baseline_pop = m_flat_bin.sum(dim=1)  # [G]
    pop2 = baseline_pop == 2
    groups_pop2 = int(pop2.sum().item())

    # Enumerate the 4 one-bit flips per group by toggling each position.
    valid = 0
    invalid = 0
    pop_after_hist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for b in range(4):
        toggled = m_flat_bin.clone()
        toggled[:, b] = 1 - toggled[:, b]
        pop_after = toggled.sum(dim=1)
        # Only count baseline-valid groups (popcount=2 before flip).
        pop_after_pop2 = pop_after[pop2]
        for k in (0, 1, 2, 3, 4):
            pop_after_hist[k] += int((pop_after_pop2 == k).sum().item())
        valid += int((pop_after_pop2 == 2).sum().item())
        invalid += int((pop_after_pop2 != 2).sum().item())

    return {
        "groups_total": int(m_flat_bin.shape[0]),
        "groups_pop2": groups_pop2,
        "single_bit_flips_total": groups_pop2 * 4,
        "valid_flips": int(valid),
        "invalid_flips": int(invalid),
        "pop_after_0": int(pop_after_hist[0]),
        "pop_after_1": int(pop_after_hist[1]),
        "pop_after_2": int(pop_after_hist[2]),
        "pop_after_3": int(pop_after_hist[3]),
        "pop_after_4": int(pop_after_hist[4]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task18: bitmask validity under single-bit metadata flip")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--ckpt", type=str, default="models/sparse_model.pth")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    set_all_seeds(int(args.seed))
    random.seed(int(args.seed))

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    log_path = "results/task18_bitmask_validity_log.txt"
    csv_path = "results/task18_bitmask_validity_summary.csv"
    png_path = "results/task18_bitmask_validity_breakdown.png"

    # Load model (Task5-style sparse INT8 PTQ pipeline).
    _, test_loader = load_cifar10_loaders_offline(batch_size=256, data_dir=args.data_dir, num_workers=0)
    model = load_sparse_int8_resnet20(device=device, ckpt_path=args.ckpt)
    acc0, loss0 = evaluate_subset(model, test_loader, device=device, max_samples=int(args.eval_samples))

    per_layer: List[Tuple[str, Dict[str, int]]] = []
    totals = {
        "groups_total": 0,
        "groups_pop2": 0,
        "single_bit_flips_total": 0,
        "valid_flips": 0,
        "invalid_flips": 0,
        "pop_after_0": 0,
        "pop_after_1": 0,
        "pop_after_2": 0,
        "pop_after_3": 0,
        "pop_after_4": 0,
    }

    for layer_name, module in get_sparse_layers(model):
        mask = module.sparse_mask
        m_flat, _ = flatten_groups(mask)
        if m_flat is None:
            continue
        m_bin = (m_flat > 0.5).to(torch.int64)
        stats = _analyze_layer_mask(m_bin)
        per_layer.append((layer_name, stats))
        for k in totals:
            totals[k] += int(stats[k])

    # Write CSV summary.
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "groups_total",
                "groups_pop2",
                "single_bit_flips_total",
                "valid_flips",
                "invalid_flips",
                "valid_ratio",
                "pop_after_1",
                "pop_after_3",
            ],
        )
        w.writeheader()

        def _row(layer: str, s: Dict[str, int]) -> Dict:
            total = float(s["single_bit_flips_total"]) if s["single_bit_flips_total"] else 0.0
            return {
                "layer": layer,
                "groups_total": s["groups_total"],
                "groups_pop2": s["groups_pop2"],
                "single_bit_flips_total": s["single_bit_flips_total"],
                "valid_flips": s["valid_flips"],
                "invalid_flips": s["invalid_flips"],
                "valid_ratio": (float(s["valid_flips"]) / total) if total else 0.0,
                "pop_after_1": s["pop_after_1"],
                "pop_after_3": s["pop_after_3"],
            }

        for name, s in per_layer:
            w.writerow(_row(name, s))
        w.writerow(_row("__TOTAL__", totals))

    # Plot: (1) overall valid vs invalid, (2) post-flip popcount distribution (baseline pop2 groups only).
    total_flips = totals["single_bit_flips_total"]
    valid = totals["valid_flips"]
    invalid = totals["invalid_flips"]
    pop1 = totals["pop_after_1"]
    pop3 = totals["pop_after_3"]

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.bar([0], [valid], label="valid (popcount=2)", color="#2ca02c")
    ax1.bar([0], [invalid], bottom=[valid], label="invalid (!=2)", color="#d62728")
    ax1.set_xticks([0])
    ax1.set_xticklabels(["All Layers"])
    ax1.set_ylabel("Count (single-bit metadata flips)")
    ax1.set_title("Task18: Bitmask Validity (1-bit flip)")
    ax1.legend(loc="best")
    ax1.grid(True, axis="y", alpha=0.3, linestyle="--")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar([0], [pop1], label="popcount=1", color="#ff7f0e")
    ax2.bar([0], [pop3], bottom=[pop1], label="popcount=3", color="#1f77b4")
    ax2.set_xticks([0])
    ax2.set_xticklabels(["Baseline popcount=2 groups"])
    ax2.set_ylabel("Count")
    ax2.set_title("Post-Flip Popcount Distribution")
    ax2.legend(loc="best")
    ax2.grid(True, axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Write log.
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("=" * 100 + "\n")
        log.write("Task 18: Bitmask Metadata Validity Under 1 Physical Bit Flip\n")
        log.write("=" * 100 + "\n")
        log.write(f"Script: run_task18_bitmask_validity.py\n")
        log.write(f"Timestamp: {_now_ts()}\n")
        log.write(f"Seed: {int(args.seed)}\n")
        log.write(f"Device: {device}\n")
        log.write(f"Dataset path: {os.path.abspath(args.data_dir)}\n")
        log.write(f"Model checkpoint path: {os.path.abspath(args.ckpt)}\n")
        log.write("\nBaseline (subset eval for sanity):\n")
        log.write(f"- eval_samples: {int(args.eval_samples)}\n")
        log.write(f"- acc:  {acc0:.2f}%\n")
        log.write(f"- loss: {loss0:.4f}\n")
        log.write("\nDefinition:\n")
        log.write("- Bitmask metadata per 2:4 group: 4 bits with popcount==2.\n")
        log.write("- 1 physical bit flip = toggle ONE of the 4 bits.\n")
        log.write("- A flip is VALID iff popcount(after)==2.\n")
        log.write("\nResults (counting only baseline-valid groups with popcount==2):\n")
        log.write(f"- total_groups(pop2): {totals['groups_pop2']}\n")
        log.write(f"- total_single_bit_flips: {total_flips}\n")
        log.write(f"- valid_flips:   {valid} ({(100.0*valid/total_flips) if total_flips else 0.0:.4f}%)\n")
        log.write(f"- invalid_flips: {invalid} ({(100.0*invalid/total_flips) if total_flips else 0.0:.4f}%)\n")
        log.write("\nPost-flip popcount breakdown:\n")
        log.write(f"- popcount=1: {pop1}\n")
        log.write(f"- popcount=3: {pop3}\n")
        log.write("\nArtifacts:\n")
        log.write(f"- {csv_path}\n")
        log.write(f"- {png_path}\n")
        log.write("\nHow to run:\n")
        log.write("  python run_task18_bitmask_validity.py --device cpu --seed 123 --eval-samples 2000\n")

    print(f"[Task18] Wrote: {log_path}")
    print(f"[Task18] Wrote: {csv_path}")
    print(f"[Task18] Wrote: {png_path}")


if __name__ == "__main__":
    main()

