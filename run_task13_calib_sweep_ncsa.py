#!/usr/bin/env python3
"""
Task 13: calib_samples sweep sensitivity for NCSA.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from statistics import mean, pstdev
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

from scripts.p012_17_utils import (
    flips_to_threshold,
    load_cifar10_loaders_offline,
    load_sparse_int8_resnet20,
    now_ts,
    run_scored_noncollision_attack,
    set_all_seeds,
)


def _parse_seeds(s: str) -> List[int]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 13: calib_samples sweep for NCSA")
    parser.add_argument("--seeds", type=str, default="123")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--ckpt", type=str, default="models/sparse_model.pth")
    parser.add_argument("--max-flips", type=int, default=50)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--max-groups-per-layer", type=int, default=2000)
    parser.add_argument("--allow-nonpositive", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    out_png = "results/task13_calib_sweep_curves.png"
    out_csv = "results/task13_calib_sweep_table.csv"
    out_log = "results/task13_calib_sweep_log.txt"

    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    seeds = _parse_seeds(args.seeds)
    if not seeds:
        seeds = [123]

    calib_values = [32, 64, 128, 256, 512, 1024]
    _, test_loader = load_cifar10_loaders_offline(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=0,
    )
    calib_loader = test_loader

    raw: Dict[int, List[Dict]] = {}
    for c in calib_values:
        raw[c] = []
        for seed in seeds:
            set_all_seeds(seed)
            model = load_sparse_int8_resnet20(device=device, ckpt_path=args.ckpt)
            run = run_scored_noncollision_attack(
                model=model,
                test_loader=test_loader,
                calib_loader=calib_loader,
                device=device,
                score_mode="ncsa",
                seed=seed,
                max_success=args.max_flips,
                calib_samples=c,
                eval_samples=args.eval_samples,
                max_groups_per_layer=args.max_groups_per_layer,
                allow_nonpositive=bool(args.allow_nonpositive),
                track_trace=False,
                enable_timing=True,
            )
            run["flips_to_50"] = flips_to_threshold(run["accuracy_history"], 50.0)
            run["flips_to_20"] = flips_to_threshold(run["accuracy_history"], 20.0)
            raw[c].append(run)

    # Plot (mean curve per calib value)
    plt.figure(figsize=(12, 7))
    for c in calib_values:
        runs = raw[c]
        max_len = max(len(r["accuracy_history"]) for r in runs)
        # Pad with last value for mean aggregation.
        padded = []
        for r in runs:
            h = list(r["accuracy_history"])
            if len(h) < max_len:
                h = h + [h[-1]] * (max_len - len(h))
            padded.append(h)
        mean_curve = [mean([p[i] for p in padded]) for i in range(max_len)]
        xs = list(range(max_len))
        plt.plot(xs, mean_curve, marker="o", linewidth=2, markersize=3, label=f"calib={c}")

    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlabel("Successful Flips")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("Task 13: NCSA Calibration Samples Sweep")
    plt.ylim(0, 100)
    plt.legend(loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "calib_samples",
                "num_seeds",
                "mean_initial_accuracy",
                "mean_final_accuracy",
                "std_final_accuracy",
                "mean_accuracy_drop",
                "mean_flips_to_50",
                "mean_flips_to_20",
                "mean_attempted_flips",
                "mean_collisions_skipped",
                "mean_avg_search_ms",
                "mean_wall_sec",
            ],
        )
        writer.writeheader()
        for c in calib_values:
            runs = raw[c]
            finals = [r["final_accuracy"] for r in runs]
            init = [r["initial_accuracy"] for r in runs]
            drop = [r["initial_accuracy"] - r["final_accuracy"] for r in runs]
            f50 = [r["flips_to_50"] for r in runs if r["flips_to_50"] is not None]
            f20 = [r["flips_to_20"] for r in runs if r["flips_to_20"] is not None]
            writer.writerow({
                "calib_samples": c,
                "num_seeds": len(runs),
                "mean_initial_accuracy": f"{mean(init):.6f}",
                "mean_final_accuracy": f"{mean(finals):.6f}",
                "std_final_accuracy": f"{(pstdev(finals) if len(finals) > 1 else 0.0):.6f}",
                "mean_accuracy_drop": f"{mean(drop):.6f}",
                "mean_flips_to_50": f"{(mean(f50) if f50 else float('nan')):.6f}",
                "mean_flips_to_20": f"{(mean(f20) if f20 else float('nan')):.6f}",
                "mean_attempted_flips": f"{mean([r['attempted_flips'] for r in runs]):.6f}",
                "mean_collisions_skipped": f"{mean([r['collisions_skipped'] for r in runs]):.6f}",
                "mean_avg_search_ms": f"{mean([r['timing']['avg_search_ms'] for r in runs]):.6f}",
                "mean_wall_sec": f"{mean([r['timing']['wall_sec'] for r in runs]):.6f}",
            })

    with open(out_log, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("Task 13: Calibration Samples Sweep (NCSA)\n")
        f.write("=" * 100 + "\n")
        f.write(f"Script: run_task13_calib_sweep_ncsa.py\n")
        f.write(f"Timestamp: {now_ts()}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Dataset path: {os.path.abspath(args.data_dir)}\n")
        f.write(f"Model checkpoint path: {os.path.abspath(args.ckpt)}\n")
        f.write("\nConfig:\n")
        for k, v in vars(args).items():
            f.write(f"- {k}: {v}\n")
        f.write("\nSummary (mean over seeds):\n")
        for c in calib_values:
            runs = raw[c]
            f.write(
                f"- calib={c}: init={mean([r['initial_accuracy'] for r in runs]):.2f}% "
                f"final={mean([r['final_accuracy'] for r in runs]):.2f}% "
                f"drop={mean([r['initial_accuracy'] - r['final_accuracy'] for r in runs]):.2f}%\n"
            )
        f.write("\nOutputs:\n")
        f.write(f"- {out_png}\n")
        f.write(f"- {out_csv}\n")
        f.write(f"- {out_log}\n")

    # Optional debug dump for later analysis.
    with open("results/task13_calib_sweep_raw.pkl", "wb") as f:
        pickle.dump({"config": vars(args), "seeds": seeds, "raw": raw}, f)

    print(f"[Task13] Wrote: {out_png}")
    print(f"[Task13] Wrote: {out_csv}")
    print(f"[Task13] Wrote: {out_log}")


if __name__ == "__main__":
    main()

