#!/usr/bin/env python3
"""
Task 20 (Cost=2): Bitmask Metadata Swap Attack (Popcount Preserving)

Purpose (Case B extension):
  - Bitmask metadata (4 bits, popcount=2) is NOT attackable under a 1-bit metadata flip model (Task18).
  - If an attacker can perform TWO metadata bit flips inside a group ("swap"):
        1->0 and 0->1
    then popcount stays 2 and metadata becomes attackable again.

Threat model / budget:
  - physical_budget = 50 bit flips
  - cost_per_swap = 2 bit flips
  - logical_steps = 25 swaps

Implementation:
  - This swap is equivalent to a non-collision "rewire" move in Task5/NCSA:
      move a nonzero from old_idx to a currently-inactive new_idx within the same 2:4 group.
  - Score uses NCSA surrogate: score = w_fp * (g_new - g_old).

Outputs:
  - results/task20_bitmask_swap_log.txt
  - results/task20_bitmask_swap_curve.png
  - results/task20_bitmask_swap_result.pkl
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
    run_scored_noncollision_attack,
    set_all_seeds,
    now_ts,
)


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task20: bitmask swap (cost=2) metadata attack")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--physical-budget", type=int, default=50)
    parser.add_argument("--cost-per-swap", type=int, default=2)
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

    log_path = "results/task20_bitmask_swap_log.txt"
    out_png = "results/task20_bitmask_swap_curve.png"
    out_pkl = "results/task20_bitmask_swap_result.pkl"

    physical_budget = int(args.physical_budget)
    cost_per_swap = int(args.cost_per_swap)
    if cost_per_swap <= 0:
        raise ValueError("cost-per-swap must be >=1")
    logical_steps = physical_budget // cost_per_swap

    _, test_loader = load_cifar10_loaders_offline(batch_size=256, data_dir=args.data_dir, num_workers=0)
    model = load_sparse_int8_resnet20(device=device, ckpt_path=args.ckpt)

    res = run_scored_noncollision_attack(
        model=model,
        test_loader=test_loader,
        calib_loader=test_loader,
        device=device,
        score_mode="ncsa",
        seed=int(args.seed),
        max_success=int(logical_steps),
        calib_samples=int(args.calib_samples),
        eval_samples=int(args.eval_samples),
        max_groups_per_layer=int(args.max_groups_per_layer),
        allow_nonpositive=False,
        track_trace=True,
        enable_timing=False,
    )

    payload = {
        "task_name": "Task20_bitmask_swap_cost2",
        "script": "run_task20_bitmask_swap_cost2.py",
        "timestamp": now_ts(),
        "seed": int(args.seed),
        "device": str(device),
        "dataset": "CIFAR-10",
        "dataset_path": os.path.abspath(args.data_dir),
        "model": "ResNet-20 (2:4 sparse) + INT8 PTQ",
        "checkpoint_path": os.path.abspath(args.ckpt),
        "threat_model": {
            "physical_budget": int(physical_budget),
            "cost_per_swap": int(cost_per_swap),
            "logical_steps": int(logical_steps),
        },
        "attack": {
            "type": "bitmask_swap",
            "score": "ncsa (w_fp * (g_new - g_old))",
            "collision_policy": "non-collision only (swap chooses inactive target)",
        },
        "calib_samples": int(args.calib_samples),
        "eval_samples": int(args.eval_samples),
        "max_groups_per_layer": int(args.max_groups_per_layer),
        "result": res,
    }

    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f)

    # Plot curve.
    acc = res["accuracy_history"]
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(len(acc))), acc, marker="o", linewidth=2, markersize=4,
             label=f"Bitmask swap (cost=2, {logical_steps} swaps)")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlabel("Logical Swaps (Successful)", fontsize=12, fontweight="bold")
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title("Task20: Bitmask Metadata Swap Attack (Cost=2)", fontsize=14, fontweight="bold")
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    # Log.
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("=" * 100 + "\n")
        log.write("Task 20: Bitmask Metadata Swap Attack (Cost=2)\n")
        log.write("=" * 100 + "\n")
        log.write(f"Script: run_task20_bitmask_swap_cost2.py\n")
        log.write(f"Timestamp: {payload['timestamp']}\n")
        log.write(f"Seed: {payload['seed']}\n")
        log.write(f"Device: {payload['device']}\n")
        log.write(f"Dataset path: {payload['dataset_path']}\n")
        log.write(f"Model checkpoint path: {payload['checkpoint_path']}\n")
        log.write("\nThreat model:\n")
        log.write(f"- physical_budget: {physical_budget}\n")
        log.write(f"- cost_per_swap:   {cost_per_swap}\n")
        log.write(f"- logical_steps:   {logical_steps}\n")
        log.write("\nConfig:\n")
        log.write(f"- calib_samples: {payload['calib_samples']}\n")
        log.write(f"- eval_samples:  {payload['eval_samples']}\n")
        log.write(f"- max_groups_per_layer: {payload['max_groups_per_layer']}\n")
        log.write("\nAttack:\n")
        log.write("- swap = (turn 1->0) + (turn 0->1) within same 2:4 group (popcount preserved)\n")
        log.write("- score surrogate: w_fp * (g_new - g_old)\n")
        log.write("\nResults:\n")
        log.write(f"- init_acc:  {res['initial_accuracy']:.2f}%\n")
        log.write(f"- final_acc: {res['final_accuracy']:.2f}%\n")
        log.write(f"- swaps_done: {res['total_flips']} (should be <= logical_steps)\n")
        log.write(f"- physical_cost_used: {res['total_flips'] * cost_per_swap}\n")
        log.write(f"- drop: {res['initial_accuracy'] - res['final_accuracy']:.2f} pts\n")
        log.write("\nArtifacts:\n")
        log.write(f"- {out_pkl}\n")
        log.write(f"- {out_png}\n")
        log.write("\nHow to run:\n")
        log.write("  python run_task20_bitmask_swap_cost2.py --device cpu --seed 123 --physical-budget 50 --cost-per-swap 2\n")

    print(f"[Task20] Wrote: {log_path}")
    print(f"[Task20] Wrote: {out_png}")
    print(f"[Task20] Wrote: {out_pkl}")


if __name__ == "__main__":
    main()

