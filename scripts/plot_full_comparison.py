#!/usr/bin/env python3
"""
Complete Comparison Plot: R1_T01-T05 + Legacy Task1-3

Compares:
- Legacy Task1: Sparse Dense Global (weight-bit attack on sparse)
- Legacy Task2: Sparse Dense Zero (zero-targeting)
- Legacy Task3: Sparse Dense NonZero (nonzero-targeting)
- R1_T01: Index/Position encoding, any pattern (50 operations)
- R1_T02: Index/Position encoding, 1-bit reachable (50 operations)
- R1_T03: Bitmask encoding, 25 swaps (50 physical flips)
- R1_T04: Bitmask encoding, 50 swaps (100 physical flips)
- R1_T05: Joint best-step attack (50 weight-bit flips)

Output: results/R1/full_comparison.png
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def load_csv(filepath: str):
    """Load CSV data into lists."""
    flips = []
    values = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            flips.append(int(row['flip']))
            values.append(float(row['accuracy']))
    return flips, values


def load_txt(filepath: str):
    """Load accuracy values from text file (one per line)."""
    values = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    values.append(float(line))
                except ValueError:
                    pass
    return values


def main():
    # =============================================================================
    # Load Data
    # =============================================================================

    # Legacy Task data
    legacy_task1 = None
    legacy_task2 = None
    legacy_task3 = None

    with open('results/legacy_L0/by_date/task1_sparse_dense_global_result.pkl', 'rb') as f:
        legacy_task1 = pickle.load(f)
    with open('results/legacy_L0/by_date/task2_sparse_dense_zero_result.pkl', 'rb') as f:
        legacy_task2 = pickle.load(f)
    with open('results/legacy_L0/by_date/task3_sparse_dense_nonzero_result.pkl', 'rb') as f:
        legacy_task3 = pickle.load(f)

    # R1_T01 - from CSV
    t01_flips, t01_acc = load_csv("results/R1/R1_T01_group_metadata_index_anypattern_curve.csv")

    # R1_T02 - from log
    t02_acc = [92.33]  # baseline
    with open("results/R1/R1_T02_group_metadata_index_1bit_log.txt", 'r') as f:
        for line in f:
            if 'acc=' in line:
                parts = line.split('acc=')
                if len(parts) > 1:
                    acc_str = parts[1].split('%')[0].strip()
                    try:
                        acc = float(acc_str)
                        t02_acc.append(acc)
                    except:
                        pass

    # R1_T03 - from log (interpolated to physical flips)
    t03_acc = []
    with open("results/R1/R1_T03_group_metadata_bitmask_swap_cost2_log.txt", 'r') as f:
        for line in f:
            if 'acc=' in line:
                parts = line.split('acc=')
                if len(parts) > 1:
                    acc_str = parts[1].split('%')[0].strip()
                    try:
                        acc = float(acc_str)
                        t03_acc.append(acc)
                    except:
                        pass
    # R1_T03 has 25 swaps, expand to physical flips
    t03_acc_physical = []
    for i, acc in enumerate(t03_acc):
        t03_acc_physical.append(acc)
        t03_acc_physical.append(acc)  # duplicate for second flip

    # R1_T04 - from table/log (51 points for 50 flips)
    t04_acc = []
    with open("results/R1/R1_T04_bitmask_swaps50_log.txt", 'r') as f:
        for line in f:
            if 'acc=' in line:
                parts = line.split('acc=')
                if len(parts) > 1:
                    acc_str = parts[1].split('%')[0].strip()
                    try:
                        acc = float(acc_str)
                        t04_acc.append(acc)
                    except:
                        pass

    # R1_T05 - manually parsed
    t05_raw = "92.21 84.52 71.44 52.88 31.45 18.51 12.50 11.57 10.35 10.06 10.01 10.01 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96 9.96"
    t05_acc = [float(x) for x in t05_raw.split()]

    # =============================================================================
    # Create Plot
    # =============================================================================

    fig, ax = plt.subplots(figsize=(14, 9))

    # Define colors and styles for 8 curves
    colors = [
        '#444444',  # Dark Gray - Legacy Task1
        '#666666',  # Gray - Legacy Task2
        '#888888',  # Light Gray - Legacy Task3
        '#1f77b4',  # Blue - R1_T01
        '#ff7f0e',  # Orange - R1_T02
        '#2ca02c',  # Green - R1_T03
        '#d62728',  # Red - R1_T04
        '#9467bd',  # Purple - R1_T05
    ]
    markers = ['s', 's', 's', 'o', '^', 'D', 'v', 'p']
    linestyles = ['--', '--', '--', '-', '-', '-', '-', '-']
    linewidths = [2, 2, 2, 2.5, 2.5, 2.5, 2.5, 3]

    # Create x-axis for each
    t01_x = list(range(len(t01_acc)))
    t02_x = list(range(len(t02_acc)))
    t03_x = list(range(len(t03_acc_physical)))
    t04_x = list(range(len(t04_acc)))
    t05_x = list(range(len(t05_acc)))

    # Plot each curve
    # Legacy tasks
    ax.plot(legacy_task1.accuracy_history, marker=markers[0], markersize=4,
            color=colors[0], linestyle=linestyles[0], linewidth=linewidths[0], alpha=0.7,
            label='Legacy Task1: Global (Weight-Bit)')
    ax.plot(legacy_task2.accuracy_history, marker=markers[1], markersize=4,
            color=colors[1], linestyle=linestyles[1], linewidth=linewidths[1], alpha=0.7,
            label='Legacy Task2: Zero-Target (Weight-Bit)')
    ax.plot(legacy_task3.accuracy_history, marker=markers[2], markersize=4,
            color=colors[2], linestyle=linestyles[2], linewidth=linewidths[2], alpha=0.7,
            label='Legacy Task3: NonZero-Target (Weight-Bit)')

    # R1 tasks
    ax.plot(t01_x, t01_acc, marker=markers[3], markersize=5,
            color=colors[3], linestyle=linestyles[3], linewidth=linewidths[3], alpha=0.9,
            label='R1_T01: Index (Any Pattern)')
    ax.plot(t02_x, t02_acc, marker=markers[4], markersize=5,
            color=colors[4], linestyle=linestyles[4], linewidth=linewidths[4], alpha=0.9,
            label='R1_T02: Index (1-bit Reachable)')
    ax.plot(t03_x, t03_acc_physical, marker=markers[5], markersize=5,
            color=colors[5], linestyle=linestyles[5], linewidth=linewidths[5], alpha=0.9,
            label='R1_T03: Bitmask (25 Swaps)')
    ax.plot(t04_x, t04_acc, marker=markers[6], markersize=5,
            color=colors[6], linestyle=linestyles[6], linewidth=linewidths[6], alpha=0.9,
            label='R1_T04: Bitmask (50 Swaps)')
    ax.plot(t05_x, t05_acc, marker=markers[7], markersize=6,
            color=colors[7], linestyle=linestyles[7], linewidth=linewidths[7], alpha=1.0,
            label='R1_T05: Joint Best-Step (Weight-Bit)')

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("Physical Flips / Operations", fontsize=14, fontweight="bold")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_title("Complete BFA Comparison: Legacy Task1-3 + R1_T01-T05 (ResNet-20/CIFAR-10, INT8)",
                 fontsize=16, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)

    # Add horizontal line at random chance (10% for CIFAR-10)
    ax.axhline(y=10, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.text(1, 11, 'Random (10%)', fontsize=8, color='gray')

    # Add baseline reference line
    ax.axhline(y=92.21, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax.text(52, 93.5, f'Baseline: 92.21%', fontsize=9, color='black')

    # Legend
    legend = ax.legend(loc='lower left', fontsize=9, framealpha=0.95, ncol=2)

    fig.tight_layout()

    # Save plot
    output_path = "results/R1/full_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Output] Saved {output_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print("Complete Comparison Summary")
    print("=" * 100)
    print(f"{'Task':<20} {'Type':<25} {'Final':<10} {'Drop':<10}")
    print("-" * 100)
    print(f"{'Legacy Task1':<20} {'Weight-Bit (Global)':<25} {legacy_task1.final_accuracy:>10.2f}% {legacy_task1.initial_accuracy - legacy_task1.final_accuracy:>10.2f}%")
    print(f"{'Legacy Task2':<20} {'Weight-Bit (Zero)':<25} {legacy_task2.final_accuracy:>10.2f}% {legacy_task2.initial_accuracy - legacy_task2.final_accuracy:>10.2f}%")
    print(f"{'Legacy Task3':<20} {'Weight-Bit (NonZero)':<25} {legacy_task3.final_accuracy:>10.2f}% {legacy_task3.initial_accuracy - legacy_task3.final_accuracy:>10.2f}%")
    print(f"{'R1_T01':<20} {'Index (Any Pattern)':<25} {t01_acc[-1]:>10.2f}% {t01_acc[0] - t01_acc[-1]:>10.2f}%")
    print(f"{'R1_T02':<20} {'Index (1-bit Reachable)':<25} {t02_acc[-1]:>10.2f}% {t02_acc[0] - t02_acc[-1]:>10.2f}%")
    print(f"{'R1_T03':<20} {'Bitmask (25 Swaps)':<25} {t03_acc[-1]:>10.2f}% {t03_acc[0] - t03_acc[-1]:>10.2f}%")
    print(f"{'R1_T04':<20} {'Bitmask (50 Swaps)':<25} {t04_acc[-1]:>10.2f}% {t04_acc[0] - t04_acc[-1]:>10.2f}%")
    print(f"{'R1_T05':<20} {'Joint (Weight-Bit)':<25} {t05_acc[-1]:>10.2f}% {t05_acc[0] - t05_acc[-1]:>10.2f}%")
    print("=" * 100)


if __name__ == "__main__":
    import pickle
    import sys
    sys.path.insert(0, '.')
    from bfa.int8_attack import Int8AttackResult
    main()
