#!/usr/bin/env python3
"""
Task 14: Layer-wise vulnerability / flip localization for NCSA.
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import Counter, defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

from scripts.p012_17_utils import (
    load_cifar10_loaders_offline,
    load_sparse_int8_resnet20,
    now_ts,
    run_scored_noncollision_attack,
    set_all_seeds,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 14: Layer-wise flip localization")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--ckpt", type=str, default="models/sparse_model.pth")
    parser.add_argument("--max-flips", type=int, default=50)
    parser.add_argument("--calib-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--max-groups-per-layer", type=int, default=2000)
    parser.add_argument("--allow-nonpositive", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--topk-layers", type=int, default=15)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    out_hist = "results/task14_layer_histogram.png"
    out_impact = "results/task14_layer_impact.png"
    out_trace = "results/task14_layer_trace.pkl"
    out_log = "results/task14_layer_trace_log.txt"

    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    set_all_seeds(args.seed)

    _, test_loader = load_cifar10_loaders_offline(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=0,
    )
    calib_loader = test_loader

    model = load_sparse_int8_resnet20(device=device, ckpt_path=args.ckpt)
    run = run_scored_noncollision_attack(
        model=model,
        test_loader=test_loader,
        calib_loader=calib_loader,
        device=device,
        score_mode="ncsa",
        seed=args.seed,
        max_success=args.max_flips,
        calib_samples=args.calib_samples,
        eval_samples=args.eval_samples,
        max_groups_per_layer=args.max_groups_per_layer,
        allow_nonpositive=bool(args.allow_nonpositive),
        track_trace=True,
        enable_timing=True,
    )
    trace: List[Dict] = run["trace"]
    acc_hist = run["accuracy_history"]

    # Histogram (top-K layers by selected flips)
    layer_counts = Counter([r["layer"] for r in trace])
    top_items = layer_counts.most_common(args.topk_layers)
    labels = [x[0] for x in top_items][::-1]
    counts = [x[1] for x in top_items][::-1]

    plt.figure(figsize=(12, max(6, 0.35 * len(labels))))
    plt.barh(labels, counts, color="#1f77b4")
    plt.xlabel("Selected Flips Count")
    plt.ylabel("Layer Name")
    plt.title(f"Task 14: Top-{args.topk_layers} Vulnerable Layers by Selection Frequency")
    plt.grid(True, axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_hist, dpi=150, bbox_inches="tight")
    plt.close()

    # Impact plot: accuracy trajectory + per-flip delta colored by layer category.
    cat_colors = {
        "stem": "#1f77b4",
        "stage1": "#2ca02c",
        "stage2": "#ff7f0e",
        "stage3": "#d62728",
        "downsample": "#9467bd",
        "head": "#8c564b",
        "other": "#7f7f7f",
    }
    xs = list(range(len(acc_hist)))
    plt.figure(figsize=(12, 7))
    plt.plot(xs, acc_hist, color="black", linewidth=1.8, label="Accuracy")
    seen = set()
    for r in trace:
        x = int(r["flip"])
        y = float(r["accuracy"])
        c = cat_colors.get(r["layer_category"], "#7f7f7f")
        lbl = r["layer_category"] if r["layer_category"] not in seen else None
        plt.scatter([x], [y], color=c, s=28, label=lbl)
        seen.add(r["layer_category"])
    plt.xlabel("Successful Flips")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("Task 14: Accuracy Impact with Flip Localization (colored by layer category)")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(out_impact, dpi=150, bbox_inches="tight")
    plt.close()

    # Save trace pkl
    with open(out_trace, "wb") as f:
        pickle.dump({
            "timestamp": now_ts(),
            "config": vars(args),
            "device": device,
            "run": run,
            "layer_counts": dict(layer_counts),
        }, f)

    # Log
    cat_counter = Counter([r["layer_category"] for r in trace])
    cat_drop = defaultdict(float)
    for i, r in enumerate(trace, start=1):
        prev = float(acc_hist[i - 1])
        cur = float(acc_hist[i])
        cat_drop[r["layer_category"]] += (prev - cur)

    with open(out_log, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("Task 14: Layer-wise Vulnerability / Flip Localization\n")
        f.write("=" * 100 + "\n")
        f.write(f"Script: run_task14_layer_localization.py\n")
        f.write(f"Timestamp: {now_ts()}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Dataset path: {os.path.abspath(args.data_dir)}\n")
        f.write(f"Model checkpoint path: {os.path.abspath(args.ckpt)}\n")
        f.write("\nConfig:\n")
        for k, v in vars(args).items():
            f.write(f"- {k}: {v}\n")
        f.write("\nHeadline:\n")
        f.write(
            f"- init_acc={run['initial_accuracy']:.2f}% final_acc={run['final_accuracy']:.2f}% "
            f"drop={(run['initial_accuracy'] - run['final_accuracy']):.2f}% total_flips={run['total_flips']}\n"
        )
        f.write("\nLayer frequency (Top-15):\n")
        for name, c in top_items:
            f.write(f"- {name}: {c}\n")
        f.write("\nCategory frequency:\n")
        for k, c in cat_counter.items():
            f.write(f"- {k}: {c}\n")
        f.write("\nCategory cumulative accuracy drop contribution:\n")
        for k, v in sorted(cat_drop.items(), key=lambda kv: kv[1], reverse=True):
            f.write(f"- {k}: {v:.4f}\n")
        f.write("\nOutputs:\n")
        f.write(f"- {out_hist}\n")
        f.write(f"- {out_impact}\n")
        f.write(f"- {out_trace}\n")
        f.write(f"- {out_log}\n")

    print(f"[Task14] Wrote: {out_hist}")
    print(f"[Task14] Wrote: {out_impact}")
    print(f"[Task14] Wrote: {out_trace}")
    print(f"[Task14] Wrote: {out_log}")


if __name__ == "__main__":
    main()

