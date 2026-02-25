#!/usr/bin/env python3
"""
Task 1.1: Dense-Format vs Metadata Attacks Comparison Plot

Draws a comprehensive comparison of:
1. Legacy Task1-3 Dense-Format attacks (global, zero-only, nonzero-only)
2. NEW Task1.1 Group-based Metadata Attack (fixed with calibration)

Output:
  results/task1_1/task1.1_dense_format_vs_metadata_attacks.png
  results/task1_1/task1.1_dense_format_vs_metadata_attacks_table.csv
"""

from __future__ import annotations

import os
import argparse
import csv
from datetime import datetime
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_csv(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load CSV data into numpy arrays."""
    flips = []
    values = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            flips.append(int(row['flip']))
            values.append(float(row['accuracy']))
    return np.array(flips), np.array(values)


def load_task1_3_csv(path: str) -> Tuple[Dict, Dict, Dict]:
    """Load Task1-3 combined CSV and return three result dicts."""
    flips = []
    global_acc = []
    zero_acc = []
    nonzero_acc = []

    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            flips.append(int(row['flip']))
            global_acc.append(float(row['global']))
            zero_acc.append(float(row['zero_only']))
            nonzero_acc.append(float(row['nonzero_only']))

    return (
        {
            'name': 'Dense-format: Global (Task 1)',
            'baseline_acc': global_acc[0],
            'final_acc': global_acc[-1],
            'total_flips': len(global_acc) - 1,
            'accuracy_history': global_acc,
        },
        {
            'name': 'Dense-format: Zero-only (Task 2)',
            'baseline_acc': zero_acc[0],
            'final_acc': zero_acc[-1],
            'total_flips': len(zero_acc) - 1,
            'accuracy_history': zero_acc,
        },
        {
            'name': 'Dense-format: Nonzero-only (Task 3)',
            'baseline_acc': nonzero_acc[0],
            'final_acc': nonzero_acc[-1],
            'total_flips': len(nonzero_acc) - 1,
            'accuracy_history': nonzero_acc,
        },
    )


def extend_to_max_flips(data: Dict, max_flips: int = 50) -> Dict:
    """Extend or truncate accuracy history to exactly max_flips for comparison."""
    acc_hist = data['accuracy_history'][:]
    current_len = len(acc_hist)

    if current_len >= max_flips:
        # Truncate
        acc_hist = acc_hist[:max_flips]
    else:
        # Extend with final value
        acc_hist = acc_hist + [acc_hist[-1]] * (max_flips - current_len)

    return {
        **data,
        'accuracy_history': acc_hist,
    }


def main():
    parser = argparse.ArgumentParser(description="Plot Task 1.1: Dense-Format vs Metadata Attacks Comparison")
    parser.add_argument("--max-flips", type=int, default=50, help="Max flips for comparison")
    parser.add_argument("--out-dir", type=str, default="results/task1_1", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Define result paths
    task1_3_csv = os.path.join(args.out_dir, "task1_3_dense_format_attacks_curve.csv")
    task1_1_csv = os.path.join(args.out_dir, "task1.1_group_metadata_attack_curve.csv")

    # Load all results
    results = []

    # Load Task1-3 from combined CSV
    if os.path.exists(task1_3_csv):
        r1, r2, r3 = load_task1_3_csv(task1_3_csv)
        results.extend([r1, r2, r3])

    # Load Task1.1 from CSV
    if os.path.exists(task1_1_csv):
        flips, metadata_acc = load_csv(task1_1_csv)
        results.append({
            'name': 'Group Metadata Attack (Task 1.1, fixed)',
            'baseline_acc': metadata_acc[0],
            'final_acc': metadata_acc[-1],
            'total_flips': len(metadata_acc) - 1,
            'accuracy_history': metadata_acc.tolist(),
        })

    if not results:
        print("Error: No results found!")
        print(f"Looking for: {task1_3_csv}, {task1_1_csv}")
        return

    # Extract info for summary table
    table_rows = []
    for r in results:
        acc_hist = r['accuracy_history']
        table_rows.append({
            'attack_name': r['name'],
            'baseline_acc': f"{r['baseline_acc']:.2f}",
            'final_acc': f"{r['final_acc']:.2f}",
            'accuracy_drop': f"{r['baseline_acc'] - r['final_acc']:.2f}",
            'final_acc_at_50': f"{acc_hist[-1]:.2f}" if len(acc_hist) >= 50 else "N/A",
            'total_flips': r['total_flips'],
        })

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors and markers for different attacks
    # Each curve gets a distinct color for clarity
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '-', '-', '-']

    for i, r in enumerate(results):
        acc_hist = r['accuracy_history']
        flips = list(range(len(acc_hist)))
        ax.plot(flips, acc_hist,
                marker=markers[i], markersize=5,
                color=colors[i], linestyle=linestyles[i],
                linewidth=2,
                alpha=0.85,
                label=r['name'])

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("Physical Flips / Iterations", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("Dense-Format vs Metadata Attacks (ResNet-20/CIFAR-10, INT8)",
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_xlim(0, args.max_flips)

    # Add horizontal line at random chance (10% for CIFAR-10)
    ax.axhline(y=10, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(1, 11, 'Random (10%)', fontsize=8, color='gray')

    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()

    # Save plot
    png_path = os.path.join(args.out_dir, "task1.1_dense_format_vs_metadata_attacks.png")
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Output] Saved plot to {png_path}")

    # Save table
    csv_path = os.path.join(args.out_dir, "task1.1_dense_format_vs_metadata_attacks_table.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['curve_name', 'baseline_acc', 'final_acc@50', 'drop', 'notes'])
        for r in results:
            acc_hist = r['accuracy_history']
            notes = 'All dense weights attackable' if 'Global' in r['name'] else (
                'Only zero-weight positions' if 'Zero-only' in r['name'] else (
                    'Only nonzero-weight positions' if 'Nonzero-only' in r['name'] else
                    '2:4 group position encoding, calibrated'
                )
            )
            writer.writerow([
                r['name'],
                f"{r['baseline_acc']:.2f}",
                f"{acc_hist[-1]:.2f}",
                f"{r['baseline_acc'] - acc_hist[-1]:.2f}",
                notes
            ])
    print(f"[Output] Saved table to {csv_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("Task 1.1: Dense-Format vs Metadata Attacks - Summary")
    print("=" * 80)
    print(f"Generated: {now_ts()}")
    print(f"Max flips: {args.max_flips}")
    print()
    print(f"{'Attack Variant':<45} {'Baseline':>10} {'Final@50':>10} {'Drop':>10}")
    print("-" * 80)
    for r in results:
        acc_hist = r['accuracy_history']
        print(f"{r['name']:<45} {r['baseline_acc']:>10.2f} {acc_hist[-1]:>10.2f} {r['baseline_acc'] - acc_hist[-1]:>10.2f}")
    print("=" * 80)
    print()
    print(f"Files:")
    print(f"  Plot: {png_path}")
    print(f"  Table: {csv_path}")


if __name__ == "__main__":
    main()
