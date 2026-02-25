#!/usr/bin/env python3
"""
Task 21 (Case C): Position/Index Encoding (2:4) - Compare Metadata NCSA vs Weight MSB

Goal:
  - On the SAME sparse INT8 ResNet-20 (CIFAR-10) pipeline, compare:
      (1) Metadata attack: non-collision NCSA-style index move (Task5 move space)
      (2) Weight attack: Int8 weight MSB (sign-bit) flips
  - Both use the same physical flip budget: 50.

Outputs:
  - results/task21_position_ncsa_curve.png
  - results/task21_position_weight_msb_curve.png
  - results/task21_position_compare_table.csv
  - results/task21_position_compare_log.txt
  - (optional) results/task21_position_ncsa_result.pkl
  - (optional) results/task21_position_weight_msb_result.pkl
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional

import torch
import matplotlib.pyplot as plt

from scripts.p012_17_utils import (
    load_cifar10_loaders_offline,
    load_sparse_int8_resnet20,
    run_scored_noncollision_attack,
    set_all_seeds,
    evaluate_subset,
    now_ts,
)
from scripts.msb_attack_utils import run_msb_attack


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _flips_to_threshold(acc_hist: List[float], threshold: float) -> Optional[int]:
    for i, a in enumerate(acc_hist):
        if float(a) <= float(threshold):
            return int(i)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Task21: compare metadata NCSA vs weight MSB (position encoding case)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-flips", type=int, default=50)
    parser.add_argument("--calib-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--max-groups-per-layer", type=int, default=2000)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--ckpt", type=str, default="models/sparse_model.pth")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    set_all_seeds(int(args.seed))

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    log_path = "results/task21_position_compare_log.txt"
    table_csv = "results/task21_position_compare_table.csv"
    ncsa_png = "results/task21_position_ncsa_curve.png"
    msb_png = "results/task21_position_weight_msb_curve.png"
    ncsa_pkl = "results/task21_position_ncsa_result.pkl"
    msb_pkl = "results/task21_position_weight_msb_result.pkl"

    _, test_loader = load_cifar10_loaders_offline(batch_size=256, data_dir=args.data_dir, num_workers=0)

    # --- (1) Metadata NCSA (non-collision move space) ---
    model_ncsa = load_sparse_int8_resnet20(device=device, ckpt_path=args.ckpt)
    res_ncsa = run_scored_noncollision_attack(
        model=model_ncsa,
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
        track_trace=True,
        enable_timing=False,
    )

    with open(ncsa_pkl, "wb") as f:
        pickle.dump(res_ncsa, f)

    plt.figure(figsize=(10, 6))
    acc = res_ncsa["accuracy_history"]
    plt.plot(list(range(len(acc))), acc, marker="o", linewidth=2, markersize=4, label="Metadata NCSA (non-collision)")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlabel("Successful Flips", fontsize=12, fontweight="bold")
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title("Task21: Position Encoding - Metadata NCSA Attack", fontsize=14, fontweight="bold")
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(ncsa_png, dpi=150, bbox_inches="tight")
    plt.close()

    # --- (2) Weight MSB (sign-bit) attack ---
    model_msb = load_sparse_int8_resnet20(device=device, ckpt_path=args.ckpt)
    res_msb = run_msb_attack(
        model=model_msb,
        test_loader=test_loader,
        calib_loader=test_loader,
        device=device,
        seed=int(args.seed),
        max_flips=int(args.max_flips),
        calib_samples=int(args.calib_samples),
        eval_fn=evaluate_subset,
        eval_samples=int(args.eval_samples),
        restrict_to_nonzero_int8=True,
        restrict_to_sparse_active=True,
        allow_nonpositive=False,
        log_interval=1,
    )

    with open(msb_pkl, "wb") as f:
        pickle.dump(res_msb, f)

    plt.figure(figsize=(10, 6))
    acc = res_msb["accuracy_history"]
    plt.plot(list(range(len(acc))), acc, marker="o", linewidth=2, markersize=4, label="Weight MSB (non-zero only)")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlabel("Successful Flips", fontsize=12, fontweight="bold")
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title("Task21: Position Encoding - Weight MSB Attack", fontsize=14, fontweight="bold")
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(msb_png, dpi=150, bbox_inches="tight")
    plt.close()

    # --- Summary table ---
    rows: List[Dict] = []
    for method, res in [
        ("position_metadata_ncsa", res_ncsa),
        ("position_weight_msb_nonzero", res_msb),
    ]:
        acc_hist = [float(x) for x in res["accuracy_history"]]
        init_acc = float(acc_hist[0])
        final_acc = float(acc_hist[-1])
        rows.append({
            "method": method,
            "seed": int(args.seed),
            "baseline_acc": init_acc,
            "final_acc": final_acc,
            "drop": init_acc - final_acc,
            "successful_flips": int(res["total_flips"]),
            "flips_to_50": _flips_to_threshold(acc_hist, 50.0),
            "flips_to_20": _flips_to_threshold(acc_hist, 20.0),
        })

    with open(table_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "seed",
                "baseline_acc",
                "final_acc",
                "drop",
                "successful_flips",
                "flips_to_50",
                "flips_to_20",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Log
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("=" * 100 + "\n")
        log.write("Task 21: Position Encoding Compare (Metadata NCSA vs Weight MSB)\n")
        log.write("=" * 100 + "\n")
        log.write(f"Script: run_task21_position_compare_ncsa_vs_weight.py\n")
        log.write(f"Timestamp: {now_ts()}\n")
        log.write(f"Seed: {int(args.seed)}\n")
        log.write(f"Device: {device}\n")
        log.write(f"Dataset path: {os.path.abspath(args.data_dir)}\n")
        log.write(f"Model checkpoint path: {os.path.abspath(args.ckpt)}\n")
        log.write("\nConfig (shared):\n")
        log.write(f"- max_flips: {int(args.max_flips)}\n")
        log.write(f"- calib_samples: {int(args.calib_samples)}\n")
        log.write(f"- eval_samples: {int(args.eval_samples)}\n")
        log.write(f"- max_groups_per_layer: {int(args.max_groups_per_layer)}\n")
        log.write("\nResults:\n")
        log.write(f"- metadata_ncsa: {res_ncsa['initial_accuracy']:.2f}% -> {res_ncsa['final_accuracy']:.2f}%\n")
        log.write(f"- weight_msb:    {res_msb['initial_accuracy']:.2f}% -> {res_msb['final_accuracy']:.2f}%\n")
        log.write("\nNotes:\n")
        log.write("- Metadata NCSA uses non-collision move space (Task5 semantics) and score=w_fp*(g_new-g_old).\n")
        log.write("- Weight MSB flips bit7 on int8 storage; restricted to int8_val!=0 and mask==1 when sparse_mask exists.\n")
        log.write("\nArtifacts:\n")
        log.write(f"- {ncsa_png}\n")
        log.write(f"- {msb_png}\n")
        log.write(f"- {table_csv}\n")
        log.write(f"- {ncsa_pkl} (optional)\n")
        log.write(f"- {msb_pkl} (optional)\n")
        log.write("\nHow to run:\n")
        log.write("  python run_task21_position_compare_ncsa_vs_weight.py --device cpu --seed 123 --max-flips 50 --calib-samples 256 --eval-samples 2000\n")

    print(f"[Task21] Wrote: {log_path}")
    print(f"[Task21] Wrote: {table_csv}")
    print(f"[Task21] Wrote: {ncsa_png}")
    print(f"[Task21] Wrote: {msb_png}")


if __name__ == "__main__":
    main()

