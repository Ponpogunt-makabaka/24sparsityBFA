#!/usr/bin/env python3
"""
Task 12: NCSA vs Baselines (Random-valid / score ablations) on ResNet-20 CIFAR-10.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt

from scripts.p012_17_utils import (
    flips_to_threshold,
    load_cifar10_loaders_offline,
    load_sparse_int8_resnet20,
    now_ts,
    run_scored_noncollision_attack,
    set_all_seeds,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 12: NCSA scoring ablation")
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
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    out_png = "results/task12_ablation_attack_curves.png"
    out_csv = "results/task12_ablation_summary.csv"
    out_pkl = "results/task12_ablation_result.pkl"
    out_log = "results/task12_ablation_log.txt"

    device = "cuda" if (args.use_cuda and __import__("torch").cuda.is_available()) else "cpu"
    set_all_seeds(args.seed)

    _, test_loader = load_cifar10_loaders_offline(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=0,
    )
    calib_loader = test_loader

    modes = [
        ("ncsa", "NCSA (w*Δg)"),
        ("random_valid", "Random-valid"),
        ("grad_only", "Grad-only (Δg)"),
        ("weight_only", "Weight-only (|w|)"),
    ]

    results: Dict[str, Dict] = {}
    for mode, label in modes:
        model = load_sparse_int8_resnet20(device=device, ckpt_path=args.ckpt)
        run = run_scored_noncollision_attack(
            model=model,
            test_loader=test_loader,
            calib_loader=calib_loader,
            device=device,
            score_mode=mode,
            seed=args.seed,
            max_success=args.max_flips,
            calib_samples=args.calib_samples,
            eval_samples=args.eval_samples,
            max_groups_per_layer=args.max_groups_per_layer,
            allow_nonpositive=bool(args.allow_nonpositive),
            track_trace=False,
            enable_timing=True,
        )
        run["label"] = label
        run["flips_to_50"] = flips_to_threshold(run["accuracy_history"], 50.0)
        run["flips_to_20"] = flips_to_threshold(run["accuracy_history"], 20.0)
        run["effective_rewires"] = int(run["total_flips"])
        results[mode] = run

    # Plot
    plt.figure(figsize=(12, 7))
    for mode, label in modes:
        hist = results[mode]["accuracy_history"]
        xs = list(range(len(hist)))
        plt.plot(xs, hist, marker="o", linewidth=2, markersize=4, label=label)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlabel("Successful Flips")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("Task 12: NCSA vs Ablations (ResNet-20, CIFAR-10)")
    plt.ylim(0, 100)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    # CSV summary
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "label",
                "seed",
                "initial_accuracy",
                "final_accuracy",
                "accuracy_drop",
                "flips_to_50",
                "flips_to_20",
                "effective_rewires",
                "attempted_flips",
                "collisions_skipped",
                "avg_search_ms",
                "avg_apply_ms",
                "avg_eval_ms",
                "wall_sec",
            ],
        )
        writer.writeheader()
        for mode, label in modes:
            r = results[mode]
            writer.writerow({
                "mode": mode,
                "label": label,
                "seed": args.seed,
                "initial_accuracy": f"{r['initial_accuracy']:.6f}",
                "final_accuracy": f"{r['final_accuracy']:.6f}",
                "accuracy_drop": f"{(r['initial_accuracy'] - r['final_accuracy']):.6f}",
                "flips_to_50": "" if r["flips_to_50"] is None else r["flips_to_50"],
                "flips_to_20": "" if r["flips_to_20"] is None else r["flips_to_20"],
                "effective_rewires": r["effective_rewires"],
                "attempted_flips": r["attempted_flips"],
                "collisions_skipped": r["collisions_skipped"],
                "avg_search_ms": f"{r['timing']['avg_search_ms']:.6f}",
                "avg_apply_ms": f"{r['timing']['avg_apply_ms']:.6f}",
                "avg_eval_ms": f"{r['timing']['avg_eval_ms']:.6f}",
                "wall_sec": f"{r['timing']['wall_sec']:.6f}",
            })

    # PKL full
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "timestamp": now_ts(),
            "config": vars(args),
            "device": device,
            "results": results,
        }, f)

    # Log
    with open(out_log, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("Task 12: NCSA vs Baselines / Scoring Ablations\n")
        f.write("=" * 100 + "\n")
        f.write(f"Script: run_task12_ablation_greedy_vs_random.py\n")
        f.write(f"Timestamp: {now_ts()}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Dataset path: {os.path.abspath(args.data_dir)}\n")
        f.write(f"Model checkpoint path: {os.path.abspath(args.ckpt)}\n")
        f.write("\nConfig:\n")
        for k, v in vars(args).items():
            f.write(f"- {k}: {v}\n")
        f.write("\nResults Summary:\n")
        for mode, label in modes:
            r = results[mode]
            f.write(
                f"- {label}: init={r['initial_accuracy']:.2f}% final={r['final_accuracy']:.2f}% "
                f"drop={(r['initial_accuracy'] - r['final_accuracy']):.2f}% "
                f"flips_to_50={r['flips_to_50']} flips_to_20={r['flips_to_20']} "
                f"rewires={r['effective_rewires']}\n"
            )
        f.write("\nOutputs:\n")
        f.write(f"- {out_png}\n")
        f.write(f"- {out_csv}\n")
        f.write(f"- {out_pkl}\n")
        f.write(f"- {out_log}\n")

    print(f"[Task12] Wrote: {out_png}")
    print(f"[Task12] Wrote: {out_csv}")
    print(f"[Task12] Wrote: {out_pkl}")
    print(f"[Task12] Wrote: {out_log}")


if __name__ == "__main__":
    main()

