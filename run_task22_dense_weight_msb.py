#!/usr/bin/env python3
"""
Task 22 (Case A): Dense Baseline - Weight MSB Attack (ResNet-20 / CIFAR-10)

Purpose:
  - Provide the dense baseline "Case A" for the reviewer closed-loop:
    weight MSB (sign-bit) attack under physical budget 50.

Model pipeline:
  - Load dense FP32 ResNet-20 checkpoint (models/dense_model.pth)
  - PTQ to INT8 (Int8QuantizedResNet)

Outputs:
  - results/task22_dense_weight_msb_log.txt
  - results/task22_dense_weight_msb_curve.png
  - results/task22_dense_weight_msb_result.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
from datetime import datetime

import torch
import matplotlib.pyplot as plt

from models.factory import create_resnet20
from train.ptq_convert import Int8QuantizedResNet
from scripts.p012_17_utils import (
    load_cifar10_loaders_offline,
    set_all_seeds,
    evaluate_subset,
    now_ts,
)
from scripts.msb_attack_utils import run_msb_attack


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_dense_int8_resnet20(device: str, ckpt_path: str) -> Int8QuantizedResNet:
    base = create_resnet20(sparsity_type=None, pretrained_path=ckpt_path).to(device)
    base.eval()
    model = Int8QuantizedResNet(base, copy_sparse_masks=False).to(device)
    model.calibrate_all_layers()
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Task22: dense baseline weight MSB attack")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-flips", type=int, default=50)
    parser.add_argument("--calib-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--ckpt", type=str, default="models/dense_model.pth")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    set_all_seeds(int(args.seed))

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    log_path = "results/task22_dense_weight_msb_log.txt"
    out_png = "results/task22_dense_weight_msb_curve.png"
    out_pkl = "results/task22_dense_weight_msb_result.pkl"

    _, test_loader = load_cifar10_loaders_offline(batch_size=256, data_dir=args.data_dir, num_workers=0)
    model = _load_dense_int8_resnet20(device=device, ckpt_path=args.ckpt)

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
        restrict_to_nonzero_int8=False,     # dense case: attack all weights (including int8 zeros)
        restrict_to_sparse_active=False,    # dense case: no sparse mask semantics
        allow_nonpositive=False,
        log_interval=1,
    )

    payload = {
        "task_name": "Task22_dense_weight_msb",
        "script": "run_task22_dense_weight_msb.py",
        "timestamp": now_ts(),
        "seed": int(args.seed),
        "device": str(device),
        "dataset": "CIFAR-10",
        "dataset_path": os.path.abspath(args.data_dir),
        "model": "Dense ResNet-20 + INT8 PTQ",
        "checkpoint_path": os.path.abspath(args.ckpt),
        "physical_flip_budget": int(args.max_flips),
        "attack": {
            "type": "weight_msb",
            "bit_pos": 7,
            "restrict_to_nonzero_int8": False,
            "restrict_to_sparse_active": False,
        },
        "calib_samples": int(args.calib_samples),
        "eval_samples": int(args.eval_samples),
        "result": result,
    }

    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f)

    # Plot.
    acc = result["accuracy_history"]
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(len(acc))), acc, marker="o", linewidth=2, markersize=4, label="Dense weight MSB")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlabel("Successful Flips", fontsize=12, fontweight="bold")
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title("Task22: Dense Baseline - Weight MSB Attack", fontsize=14, fontweight="bold")
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    # Log.
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("=" * 100 + "\n")
        log.write("Task 22: Dense Baseline - Weight MSB Attack\n")
        log.write("=" * 100 + "\n")
        log.write(f"Script: run_task22_dense_weight_msb.py\n")
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
        log.write("- dense case: includes all weights (no sparse_mask filtering)\n")
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
        log.write("  python run_task22_dense_weight_msb.py --device cpu --seed 123 --max-flips 50 --calib-samples 256 --eval-samples 2000\n")

    print(f"[Task22] Wrote: {log_path}")
    print(f"[Task22] Wrote: {out_png}")
    print(f"[Task22] Wrote: {out_pkl}")


if __name__ == "__main__":
    main()

