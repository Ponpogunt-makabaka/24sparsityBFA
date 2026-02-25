#!/usr/bin/env python3
"""
Task 28 (Sanity): Minimal Closed-Loop Check on ResNet-20/CIFAR-10 (2:4).

Purpose:
  - Ensure the improved sparse baseline checkpoint produced by Task28 audit/finetune is
    interface-compatible with existing attacks (metadata NCSA + weight MSB).
  - Keep runtime manageable: default uses eval_samples=2000 and max_flips=10.

Outputs:
  - results/task28_closed_loop_sanity_log.txt
  - results/task28_closed_loop_sanity_table.csv
  - results/task28_closed_loop_sanity_curves.png
  - results/task28_closed_loop_sanity_result.pkl
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import random
import time
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from config import DATASET
from models.resnet20 import resnet20
from scripts.msb_attack_utils import run_msb_attack
from scripts.p012_17_utils import (
    evaluate_subset,
    load_sparse_int8_resnet20,
    run_scored_noncollision_attack,
)
from train.ptq_convert import Int8QuantizedResNet


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cifar10_offline(batch_size: int, data_dir: str) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    mean = tuple(float(x) for x in DATASET["mean"])
    std = tuple(float(x) for x in DATASET["std"])
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    cifar_dir = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.exists(cifar_dir):
        raise FileNotFoundError(f"CIFAR-10 not found under {data_dir} (expected {cifar_dir}).")

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=test_tf)

    pin_memory = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory
    )
    return train_loader, test_loader


def load_dense_int8(device: str, dense_ckpt_path: str) -> torch.nn.Module:
    ckpt = torch.load(dense_ckpt_path, map_location="cpu")
    base = resnet20(sparsity_type=None).to(device)
    base.load_state_dict(ckpt["model_state_dict"])
    base.eval()
    model = Int8QuantizedResNet(base, copy_sparse_masks=False).to(device)
    model.calibrate_all_layers()
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Task28 sanity: closed-loop compatibility on improved baseline")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--calib-samples", type=int, default=256)
    parser.add_argument("--max-flips", type=int, default=10)

    parser.add_argument("--dense-ckpt", type=str, default="models/dense_model.pth")
    parser.add_argument("--sparse-ckpt", type=str, default="results/task28_sparse_mask_fixed_finetune_ckpt.pth")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    set_seed(int(args.seed))

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    _, test_loader = load_cifar10_offline(batch_size=int(args.batch_size), data_dir=args.data_dir)

    out_log = "results/task28_closed_loop_sanity_log.txt"
    out_csv = "results/task28_closed_loop_sanity_table.csv"
    out_png = "results/task28_closed_loop_sanity_curves.png"
    out_pkl = "results/task28_closed_loop_sanity_result.pkl"

    # --- Dense weight MSB sanity (Int8) ---
    dense_int8 = load_dense_int8(device=device, dense_ckpt_path=args.dense_ckpt)
    dense_res = run_msb_attack(
        model=dense_int8,
        test_loader=test_loader,
        calib_loader=test_loader,
        device=device,
        seed=int(args.seed),
        max_flips=int(args.max_flips),
        calib_samples=int(args.calib_samples),
        eval_fn=evaluate_subset,
        eval_samples=int(args.eval_samples),
        restrict_to_nonzero_int8=True,
        restrict_to_sparse_active=False,
        allow_nonpositive=False,
        log_interval=1,
    )

    # --- Sparse metadata NCSA sanity (non-collision) ---
    sparse_int8 = load_sparse_int8_resnet20(device=device, ckpt_path=args.sparse_ckpt)
    sparse_res = run_scored_noncollision_attack(
        model=sparse_int8,
        test_loader=test_loader,
        calib_loader=test_loader,
        device=device,
        score_mode="ncsa",
        seed=int(args.seed),
        max_success=int(args.max_flips),
        calib_samples=int(args.calib_samples),
        eval_samples=int(args.eval_samples),
        max_groups_per_layer=2000,
        allow_nonpositive=False,
        track_trace=True,
        enable_timing=False,
    )

    with open(out_pkl, "wb") as f:
        pickle.dump({"dense_weight_msb": dense_res, "sparse_metadata_ncsa": sparse_res}, f)

    # Summary table
    rows = [
        {
            "method": "dense_weight_msb_int8",
            "seed": int(args.seed),
            "baseline_acc": float(dense_res["initial_accuracy"]),
            "final_acc": float(dense_res["final_accuracy"]),
            "successful_flips": int(dense_res["total_flips"]),
            "ckpt": os.path.abspath(args.dense_ckpt),
        },
        {
            "method": "sparse_metadata_ncsa_noncollision_int8",
            "seed": int(args.seed),
            "baseline_acc": float(sparse_res["initial_accuracy"]),
            "final_acc": float(sparse_res["final_accuracy"]),
            "successful_flips": int(sparse_res["total_flips"]),
            "ckpt": os.path.abspath(args.sparse_ckpt),
        },
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Curves plot (overlay)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(
        list(range(len(dense_res["accuracy_history"]))),
        dense_res["accuracy_history"],
        marker="o",
        linewidth=2,
        markersize=4,
        label="Dense weight MSB (int8)",
    )
    plt.plot(
        list(range(len(sparse_res["accuracy_history"]))),
        sparse_res["accuracy_history"],
        marker="s",
        linewidth=2,
        markersize=4,
        label="Sparse metadata NCSA (non-collision, int8)",
    )
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlabel("Successful Flips", fontsize=12, fontweight="bold")
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title("Task28 Sanity: Closed-Loop Compatibility (10 flips)", fontsize=14, fontweight="bold")
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    # Log
    with open(out_log, "w", encoding="utf-8") as log:
        log.write("=" * 100 + "\n")
        log.write("Task 28 Sanity: Minimal Closed-Loop Compatibility Check\n")
        log.write("=" * 100 + "\n")
        log.write(f"Script: run_task28_closed_loop_sanity.py\n")
        log.write(f"Timestamp: {now_ts()}\n")
        log.write(f"Seed: {int(args.seed)}\n")
        log.write(f"Device: {device}\n")
        log.write(f"Dataset path: {os.path.abspath(args.data_dir)}\n")
        log.write(f"Dense ckpt: {os.path.abspath(args.dense_ckpt)}\n")
        log.write(f"Sparse ckpt (improved): {os.path.abspath(args.sparse_ckpt)}\n")
        log.write("\nConfig:\n")
        log.write(f"- max_flips: {int(args.max_flips)}\n")
        log.write(f"- calib_samples: {int(args.calib_samples)}\n")
        log.write(f"- eval_samples: {int(args.eval_samples)}\n")
        log.write("\nResults:\n")
        log.write(
            f"- dense_weight_msb_int8: {dense_res['initial_accuracy']:.2f}% -> {dense_res['final_accuracy']:.2f}% "
            f"({dense_res['total_flips']} flips)\n"
        )
        log.write(
            f"- sparse_metadata_ncsa_noncollision_int8: {sparse_res['initial_accuracy']:.2f}% -> {sparse_res['final_accuracy']:.2f}% "
            f"({sparse_res['total_flips']} flips)\n"
        )
        log.write("\nArtifacts:\n")
        log.write(f"- {out_log}\n")
        log.write(f"- {out_csv}\n")
        log.write(f"- {out_png}\n")
        log.write(f"- {out_pkl}\n")
        log.write("\nHow to run:\n")
        log.write("  python run_task28_closed_loop_sanity.py --device cpu --seed 123 --max-flips 10 --eval-samples 2000\n")


if __name__ == "__main__":
    main()

