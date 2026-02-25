#!/usr/bin/env python3
"""
Task 15: Defense realism (trusted checksum vs same-fault-domain adaptive attacker).
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

import run_task11_metadata_defense as t11
from scripts.p012_17_utils import load_cifar10_loaders_offline, now_ts, set_all_seeds


def _run_case(
    name: str,
    defense: str,
    bypass_parity: bool,
    bypass_crc: bool,
    max_flips: int,
    physical_cost_per_logical: int,
    args,
    test_loader,
    calib_loader,
    device: str,
) -> Dict:
    set_all_seeds(args.seed)
    model = t11._load_sparse_int8_resnet20(device=device, ckpt_path=args.ckpt)  # noqa: SLF001
    t0 = time.perf_counter()
    run = t11.run_defended_ncsa(
        model=model,
        test_loader=test_loader,
        calib_loader=calib_loader,
        device=device,
        defense=defense,
        mitigation="revert",
        max_flips=max_flips,
        calib_samples=args.calib_samples,
        eval_samples=args.eval_samples,
        max_groups_per_layer=args.max_groups_per_layer,
        allow_nonpositive=bool(args.allow_nonpositive),
        line_bytes=args.line_bytes,
        bypass_parity=bypass_parity,
        bypass_crc=bypass_crc,
        debug_trace=False,
        trace_records=None,
        run_name=name,
    )
    wall_sec = time.perf_counter() - t0
    logical = int(run["flip_attempts_executed"])
    physical = int(logical * physical_cost_per_logical)
    detected_rate = float(run["detected"]) / float(logical) if logical > 0 else 0.0
    mitigated_rate = float(run["mitigated"]) / float(logical) if logical > 0 else 0.0
    return {
        "name": name,
        "defense": defense,
        "bypass_parity": bool(bypass_parity),
        "bypass_crc": bool(bypass_crc),
        "logical_flips": logical,
        "physical_flips": physical,
        "physical_cost_per_logical": int(physical_cost_per_logical),
        "extra_flips_total": int(physical - logical),
        "initial_accuracy": float(run["initial_accuracy"]),
        "final_accuracy": float(run["final_accuracy"]),
        "detected": int(run["detected"]),
        "mitigated": int(run["mitigated"]),
        "detected_rate": detected_rate,
        "mitigated_rate": mitigated_rate,
        "accuracy_history": run["accuracy_history"],
        "wall_sec": float(wall_sec),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 15: Defense realism study")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--ckpt", type=str, default="models/sparse_model.pth")
    parser.add_argument("--max-flips", type=int, default=50)
    parser.add_argument("--calib-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--max-groups-per-layer", type=int, default=2000)
    parser.add_argument("--allow-nonpositive", action="store_true")
    parser.add_argument("--line-bytes", type=int, default=64)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    out_png = "results/task15_defense_realism_curves.png"
    out_csv = "results/task15_defense_realism_table.csv"
    out_log = "results/task15_defense_realism_log.txt"

    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    set_all_seeds(args.seed)
    _, test_loader = load_cifar10_loaders_offline(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=0,
    )
    calib_loader = test_loader

    budgeted_logical = max(1, args.max_flips // 2)  # 2 physical flips per bypassed logical attack.

    cases = [
        ("baseline_none", "none", False, False, args.max_flips, 1),
        ("parity_trusted", "parity", False, False, args.max_flips, 1),
        ("parity_adaptive_bypass", "parity", True, False, args.max_flips, 1),
        ("parity_budgeted_bypass", "parity", True, False, budgeted_logical, 2),
        ("crc_trusted", "crc", False, False, args.max_flips, 1),
        ("crc_adaptive_bypass", "crc", False, True, args.max_flips, 1),
        ("crc_budgeted_bypass", "crc", False, True, budgeted_logical, 2),
    ]

    results: List[Dict] = []
    for name, defense, bp, bc, logical_max, cost in cases:
        results.append(_run_case(
            name=name,
            defense=defense,
            bypass_parity=bp,
            bypass_crc=bc,
            max_flips=logical_max,
            physical_cost_per_logical=cost,
            args=args,
            test_loader=test_loader,
            calib_loader=calib_loader,
            device=device,
        ))

    # Curves with physical-flip x-axis.
    plt.figure(figsize=(12, 7))
    for r in results:
        x = [i * r["physical_cost_per_logical"] for i in range(len(r["accuracy_history"]))]
        plt.plot(x, r["accuracy_history"], marker="o", linewidth=2, markersize=3, label=r["name"])
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlabel("Physical Flip Budget (metadata + checksum edits)")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("Task 15: Defense Realism (trusted vs adaptive/budgeted bypass)")
    plt.ylim(0, 100)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "defense",
                "bypass_parity",
                "bypass_crc",
                "logical_flips",
                "physical_flips",
                "physical_cost_per_logical",
                "extra_flips_total",
                "initial_accuracy",
                "final_accuracy",
                "accuracy_drop",
                "detected",
                "mitigated",
                "detected_rate",
                "mitigated_rate",
                "wall_sec",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "name": r["name"],
                "defense": r["defense"],
                "bypass_parity": r["bypass_parity"],
                "bypass_crc": r["bypass_crc"],
                "logical_flips": r["logical_flips"],
                "physical_flips": r["physical_flips"],
                "physical_cost_per_logical": r["physical_cost_per_logical"],
                "extra_flips_total": r["extra_flips_total"],
                "initial_accuracy": f"{r['initial_accuracy']:.6f}",
                "final_accuracy": f"{r['final_accuracy']:.6f}",
                "accuracy_drop": f"{(r['initial_accuracy'] - r['final_accuracy']):.6f}",
                "detected": r["detected"],
                "mitigated": r["mitigated"],
                "detected_rate": f"{r['detected_rate']:.6f}",
                "mitigated_rate": f"{r['mitigated_rate']:.6f}",
                "wall_sec": f"{r['wall_sec']:.6f}",
            })

    with open(out_log, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("Task 15: Defense Realism (trusted checksum vs same-fault-domain attacker)\n")
        f.write("=" * 100 + "\n")
        f.write("Script: run_task15_defense_realism.py\n")
        f.write(f"Timestamp: {now_ts()}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Dataset path: {os.path.abspath(args.data_dir)}\n")
        f.write(f"Model checkpoint path: {os.path.abspath(args.ckpt)}\n")
        f.write("\nConfig:\n")
        for k, v in vars(args).items():
            f.write(f"- {k}: {v}\n")
        f.write(f"- budgeted_logical_flips (cost=2): {budgeted_logical}\n")
        f.write("\nResults:\n")
        for r in results:
            f.write(
                f"- {r['name']}: init={r['initial_accuracy']:.2f}% final={r['final_accuracy']:.2f}% "
                f"logical={r['logical_flips']} physical={r['physical_flips']} extra={r['extra_flips_total']} "
                f"det_rate={r['detected_rate']:.2%} mit_rate={r['mitigated_rate']:.2%}\n"
            )
        f.write("\nNotes:\n")
        f.write("- trusted cases correspond to current Task11 threat model (checksum storage trusted).\n")
        f.write("- adaptive bypass simulates co-modifying checksum state in same fault domain.\n")
        f.write("- budgeted bypass charges +1 extra physical flip per logical metadata attack.\n")
        f.write("\nOutputs:\n")
        f.write(f"- {out_png}\n")
        f.write(f"- {out_csv}\n")
        f.write(f"- {out_log}\n")

    print(f"[Task15] Wrote: {out_png}")
    print(f"[Task15] Wrote: {out_csv}")
    print(f"[Task15] Wrote: {out_log}")


if __name__ == "__main__":
    main()

