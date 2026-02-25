#!/usr/bin/env python3
"""
Task 19: Bitmask-Encoding Case - Weight MSB Attack (Non-Zero Weights Only)

Context (Case B for reviewer closed-loop):
  - Metadata uses a 4-bit bitmask per 2:4 group (popcount=2).
  - Under a strict 1 physical bit flip model, metadata flips are almost always invalid (Task18).
  - Therefore the best feasible 1-bit fault injection attack targets WEIGHT bits.

This script runs an Int8 weight MSB (sign-bit) attack on the sparse INT8 model,
restricted to NON-ZERO weights (and for sparse layers, only masked-active weights).

Outputs:
  - results/task19_bitmask_weight_msb_log.txt
  - results/task19_bitmask_weight_msb_curve.png
  - results/task19_bitmask_weight_msb_result.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
from datetime import datetime

import torch
import matplotlib.pyplot as plt

from scripts.p012_17_utils import (
    load_cifar10_loaders_offline,
    load_sparse_int8_resnet20,
    set_all_seeds,
    evaluate_subset,
)
from scripts.msb_attack_utils import run_msb_attack


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task19: bitmask case weight MSB attack (non-zero only)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-flips", type=int, default=50)
    parser.add_argument("--calib-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--ckpt", type=str, default="models/sparse_model.pth")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    set_all_seeds(int(args.seed))

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    log_path = "results/task19_bitmask_weight_msb_log.txt"
    out_png = "results/task19_bitmask_weight_msb_curve.png"
    out_pkl = "results/task19_bitmask_weight_msb_result.pkl"

    _, test_loader = load_cifar10_loaders_offline(batch_size=256, data_dir=args.data_dir, num_workers=0)
    model = load_sparse_int8_resnet20(device=device, ckpt_path=args.ckpt)

    result = run_msb_attack(
        model=model,
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

    payload = {
        "task_name": "Task19_bitmask_case_weight_msb_nonzero_only",
        "script": "run_task19_bitmask_weight_msb.py",
        "timestamp": _now_ts(),
        "seed": int(args.seed),
        "device": str(device),
        "dataset": "CIFAR-10",
        "dataset_path": os.path.abspath(args.data_dir),
        "model": "ResNet-20 (2:4 sparse) + INT8 PTQ",
        "checkpoint_path": os.path.abspath(args.ckpt),
        "physical_flip_budget": int(args.max_flips),
        "attack": {
            "type": "weight_msb",
            "bit_pos": 7,
            "restrict_to_nonzero_int8": True,
            "restrict_to_sparse_active": True,
        },
        "calib_samples": int(args.calib_samples),
        "eval_samples": int(args.eval_samples),
        "result": result,
    }

    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f)

    # Plot curve.
    acc = result["accuracy_history"]
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(len(acc))), acc, marker="o", linewidth=2, markersize=4, label="Weight MSB (non-zero only)")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlabel("Successful Flips", fontsize=12, fontweight="bold")
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title("Task19: Bitmask Case - Weight MSB Attack (Non-Zero Only)", fontsize=14, fontweight="bold")
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    # Log.
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("=" * 100 + "\n")
        log.write("Task 19: Bitmask Case - Weight MSB Attack (Non-Zero Weights Only)\n")
        log.write("=" * 100 + "\n")
        log.write(f"Script: run_task19_bitmask_weight_msb.py\n")
        log.write(f"Timestamp: {payload['timestamp']}\n")
        log.write(f"Seed: {payload['seed']}\n")
        log.write(f"Device: {payload['device']}\n")
        log.write(f"Dataset path: {payload['dataset_path']}\n")
        log.write(f"Model checkpoint path: {payload['checkpoint_path']}\n")
        log.write("\nConfig:\n")
        log.write(f"- physical_flip_budget: {payload['physical_flip_budget']}\n")
        log.write(f"- calib_samples: {payload['calib_samples']}\n")
        log.write(f"- eval_samples: {payload['eval_samples']}\n")
        log.write("\nAttack:\n")
        log.write("- weight MSB (sign-bit) flips on int8 storage (bit_pos=7)\n")
        log.write("- restricted to: int8_val != 0, and (if sparse_mask exists) mask==1\n")
        log.write("\nResults:\n")
        log.write(f"- init_acc:  {result['initial_accuracy']:.2f}%\n")
        log.write(f"- final_acc: {result['final_accuracy']:.2f}%\n")
        log.write(f"- flips:     {result['total_flips']}\n")
        log.write(f"- drop:      {result['initial_accuracy'] - result['final_accuracy']:.2f} pts\n")
        log.write(f"- wall_sec:  {result['timing']['wall_sec']:.2f}\n")
        log.write("\nArtifacts:\n")
        log.write(f"- {out_pkl}\n")
        log.write(f"- {out_png}\n")
        log.write("\nHow to run:\n")
        log.write("  python run_task19_bitmask_weight_msb.py --device cpu --seed 123 --max-flips 50 --calib-samples 256 --eval-samples 2000\n")

    print(f"[Task19] Wrote: {log_path}")
    print(f"[Task19] Wrote: {out_png}")
    print(f"[Task19] Wrote: {out_pkl}")


if __name__ == "__main__":
    main()

