#!/usr/bin/env python3
"""
Task 17: Seed robustness for key headline experiments.

Default:
- Task5 (ResNet-20 CIFAR-10) over 3 seeds.
- Task6 (ResNet-18 Imagenette/ImageNet fallback) over 3 seeds.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from torchvision.models import resnet18

from bfa.csr_non_collision_attack import run_csr_non_collision_attack
from models.factory import _compute_2_4_mask_conv, _compute_2_4_mask_linear, _load_local_weights
from scripts.p012_17_utils import (
    load_cifar10_loaders_offline,
    load_sparse_int8_resnet20,
    now_ts,
    run_scored_noncollision_attack,
    set_all_seeds,
)
from train.imagenet_pipeline_utils import (
    apply_sparse_mask,
    calibrate_int8,
    evaluate,
    replace_with_int8_from_mask,
    run_bn_recalibration,
)
from train.imagenet_utils import get_imagenet_loader


def _parse_seeds(s: str) -> List[int]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def _run_task5_one(seed: int, args, device: str, test_loader, calib_loader) -> Dict:
    set_all_seeds(seed)
    t0 = time.perf_counter()
    model = load_sparse_int8_resnet20(device=device, ckpt_path=args.cifar_ckpt)
    run = run_scored_noncollision_attack(
        model=model,
        test_loader=test_loader,
        calib_loader=calib_loader,
        device=device,
        score_mode="ncsa",
        seed=seed,
        max_success=args.max_flips,
        calib_samples=args.calib_samples,
        eval_samples=args.eval_samples,
        max_groups_per_layer=args.max_groups_per_layer,
        allow_nonpositive=bool(args.allow_nonpositive),
        track_trace=False,
        enable_timing=True,
    )
    run["wall_sec_external"] = float(time.perf_counter() - t0)
    return run


def _run_task6_one(seed: int, args, device: str) -> Dict:
    set_all_seeds(seed)
    t0 = time.perf_counter()

    # Enforce num_workers=0 due known repo constraint.
    val_loader = get_imagenet_loader(
        root=args.imagenet_root,
        split="val",
        batch_size=args.imagenet_batch_size,
        num_workers=0,
        subset_root=None,
        shuffle=False,
        imagenette_root=args.imagenette_root,
    )
    calib_loader = get_imagenet_loader(
        root=args.imagenet_root,
        split="val",
        batch_size=args.imagenet_batch_size,
        num_workers=0,
        subset_root=None,
        shuffle=True,
        imagenette_root=args.imagenette_root,
    )
    train_loader = get_imagenet_loader(
        root=args.imagenet_root,
        split="train",
        batch_size=args.imagenet_batch_size,
        num_workers=0,
        subset_root=None,
        shuffle=True,
        imagenette_root=args.imagenette_root,
    )

    model = resnet18(weights=None).to(device)
    _load_local_weights(model, args.resnet18_weights, strict=True)
    model.eval()

    def conv_filter(mod, name: str) -> bool:
        if name == "conv1" or ".downsample." in name:
            return False
        return mod.in_channels % 4 == 0

    def linear_filter(mod, name: str) -> bool:
        return name != "fc"

    mask_map = apply_sparse_mask(
        model,
        compute_conv_mask=_compute_2_4_mask_conv,
        compute_linear_mask=_compute_2_4_mask_linear,
        conv_filter=conv_filter,
        linear_filter=linear_filter,
    )
    bn_steps = run_bn_recalibration(
        model=model,
        train_loader=train_loader,
        device=device,
        max_steps=args.bn_calib_steps,
        lr=args.bn_lr,
    )
    post_bn_acc = evaluate(model, val_loader, device=device, max_samples=args.imagenet_eval_samples)

    replace_with_int8_from_mask(model, mask_map)
    calibrate_int8(model)
    model.eval()

    result = run_csr_non_collision_attack(
        model=model,
        test_loader=val_loader,
        calib_loader=calib_loader,
        device=device,
        max_success=args.max_flips,
        calib_samples=args.imagenet_calib_samples,
        log_interval=1,
        max_groups_per_layer=args.imagenet_max_groups_per_layer,
        eval_samples=args.imagenet_eval_samples,
        allow_nonpositive=bool(args.allow_nonpositive),
    )
    return {
        "initial_accuracy": float(result.initial_accuracy),
        "final_accuracy": float(result.final_accuracy),
        "total_flips": int(result.total_flips),
        "attempted_flips": int(result.attempted_flips),
        "collisions_skipped": int(result.collisions_skipped),
        "wall_sec_external": float(time.perf_counter() - t0),
        "post_bn_acc": float(post_bn_acc),
        "bn_steps": int(bn_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 17: Seed robustness")
    parser.add_argument("--seeds", type=str, default="123,456,789")
    parser.add_argument("--max-flips", type=int, default=50)
    parser.add_argument("--calib-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--max-groups-per-layer", type=int, default=2000)
    parser.add_argument("--allow-nonpositive", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--cifar-ckpt", type=str, default="models/sparse_model.pth")
    parser.add_argument("--run-task6", action="store_true",
                        help="Enable Task6 (ResNet-18 Imagenette fallback) seed robustness.")
    parser.add_argument("--imagenet-root", type=str, default="/home/lab-2010/24sparsityBFA/data/imagenet")
    parser.add_argument("--imagenette-root", type=str,
                        default="/home/lab-2010/24sparsityBFA/data/imagenette/imagenette2")
    parser.add_argument("--resnet18-weights", type=str,
                        default="/home/lab-2010/24sparsityBFA/weights/resnet18-f37072fd.pth")
    parser.add_argument("--imagenet-batch-size", type=int, default=32)
    parser.add_argument("--imagenet-calib-samples", type=int, default=128)
    parser.add_argument("--imagenet-eval-samples", type=int, default=256)
    parser.add_argument("--imagenet-max-groups-per-layer", type=int, default=2000)
    parser.add_argument("--bn-calib-steps", type=int, default=120)
    parser.add_argument("--bn-lr", type=float, default=1e-3)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    out_csv = "results/task17_seed_robustness_table.csv"
    out_log = "results/task17_seed_robustness_log.txt"
    out_png = "results/task17_seed_robustness_boxplot.png"

    seeds = _parse_seeds(args.seeds)
    if len(seeds) < 3:
        seeds = [123, 456, 789]

    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    set_all_seeds(seeds[0])

    # Task5 loaders
    _, test_loader = load_cifar10_loaders_offline(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=0,
    )
    calib_loader = test_loader

    rows: List[Dict] = []

    # Task5 robustness (always run)
    for seed in seeds:
        r = _run_task5_one(seed, args, device, test_loader, calib_loader)
        rows.append({
            "task": "Task5_ResNet20_CIFAR10",
            "seed": seed,
            "initial_accuracy": r["initial_accuracy"],
            "final_accuracy": r["final_accuracy"],
            "accuracy_drop": r["initial_accuracy"] - r["final_accuracy"],
            "total_flips": r["total_flips"],
            "attempted_flips": r["attempted_flips"],
            "collisions_skipped": r["collisions_skipped"],
            "wall_sec": r["wall_sec_external"],
            "note": "",
        })

    # Optional Task6 robustness (compute-heavy)
    if args.run_task6:
        for seed in seeds:
            try:
                r = _run_task6_one(seed, args, device)
                rows.append({
                    "task": "Task6_ResNet18_Imagenette",
                    "seed": seed,
                    "initial_accuracy": r["initial_accuracy"],
                    "final_accuracy": r["final_accuracy"],
                    "accuracy_drop": r["initial_accuracy"] - r["final_accuracy"],
                    "total_flips": r["total_flips"],
                    "attempted_flips": r["attempted_flips"],
                    "collisions_skipped": r["collisions_skipped"],
                    "wall_sec": r["wall_sec_external"],
                    "note": f"post_bn_acc={r['post_bn_acc']:.2f},bn_steps={r['bn_steps']}",
                })
            except Exception as exc:  # keep table complete even if Task6 unavailable
                rows.append({
                    "task": "Task6_ResNet18_Imagenette",
                    "seed": seed,
                    "initial_accuracy": float("nan"),
                    "final_accuracy": float("nan"),
                    "accuracy_drop": float("nan"),
                    "total_flips": 0,
                    "attempted_flips": 0,
                    "collisions_skipped": 0,
                    "wall_sec": 0.0,
                    "note": f"FAILED: {repr(exc)}",
                })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "seed",
                "initial_accuracy",
                "final_accuracy",
                "accuracy_drop",
                "total_flips",
                "attempted_flips",
                "collisions_skipped",
                "wall_sec",
                "note",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "task": r["task"],
                "seed": r["seed"],
                "initial_accuracy": f"{r['initial_accuracy']:.6f}" if isinstance(r["initial_accuracy"], float) else r["initial_accuracy"],
                "final_accuracy": f"{r['final_accuracy']:.6f}" if isinstance(r["final_accuracy"], float) else r["final_accuracy"],
                "accuracy_drop": f"{r['accuracy_drop']:.6f}" if isinstance(r["accuracy_drop"], float) else r["accuracy_drop"],
                "total_flips": r["total_flips"],
                "attempted_flips": r["attempted_flips"],
                "collisions_skipped": r["collisions_skipped"],
                "wall_sec": f"{r['wall_sec']:.6f}" if isinstance(r["wall_sec"], float) else r["wall_sec"],
                "note": r["note"],
            })

    # Boxplot on final accuracy by task
    task_to_vals: Dict[str, List[float]] = {}
    for r in rows:
        if isinstance(r["final_accuracy"], float) and r["final_accuracy"] == r["final_accuracy"]:
            task_to_vals.setdefault(r["task"], []).append(r["final_accuracy"])

    plt.figure(figsize=(10, 6))
    labels = list(task_to_vals.keys())
    vals = [task_to_vals[k] for k in labels]
    if vals:
        plt.boxplot(vals, labels=labels, showmeans=True)
    plt.ylabel("Final Accuracy @50 flips (%)")
    plt.title("Task 17: Seed Robustness (3 seeds)")
    plt.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    with open(out_log, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("Task 17: Seed Robustness\n")
        f.write("=" * 100 + "\n")
        f.write("Script: run_task17_seed_robustness.py\n")
        f.write(f"Timestamp: {now_ts()}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Device: {device}\n")
        f.write(f"CIFAR dataset path: {os.path.abspath(args.data_dir)}\n")
        f.write(f"CIFAR model checkpoint path: {os.path.abspath(args.cifar_ckpt)}\n")
        f.write(f"Imagenette root (offline): {args.imagenette_root}\n")
        f.write(f"ResNet18 checkpoint path: {args.resnet18_weights}\n")
        f.write("\nConfig:\n")
        for k, v in vars(args).items():
            f.write(f"- {k}: {v}\n")
        f.write("\nPer-task summary:\n")
        for task, vals in task_to_vals.items():
            if vals:
                f.write(
                    f"- {task}: mean_final={statistics.mean(vals):.2f}% "
                    f"std_final={(statistics.pstdev(vals) if len(vals) > 1 else 0.0):.2f}% n={len(vals)}\n"
                )
        f.write("\nOutputs:\n")
        f.write(f"- {out_csv}\n")
        f.write(f"- {out_log}\n")
        f.write(f"- {out_png}\n")

    print(f"[Task17] Wrote: {out_csv}")
    print(f"[Task17] Wrote: {out_log}")
    print(f"[Task17] Wrote: {out_png}")


if __name__ == "__main__":
    main()

