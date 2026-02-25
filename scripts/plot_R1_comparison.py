#!/usr/bin/env python3
"""
R1 Workflow Comparison Plot

Compares all four R1 tasks:
- R1_T01: Index/Position encoding, any pattern (50 operations)
- R1_T02: Index/Position encoding, 1-bit reachable (50 operations)
- R1_T03: Bitmask encoding, 25 swaps (50 physical flips)
- R1_T04: Bitmask encoding, 50 swaps (100 physical flips)

Output: results/R1/R1_comparison_curve.png
"""

import os
import csv
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_csv(filepath: str) -> Tuple[List[int], List[float]]:
    """Load CSV data into numpy arrays."""
    flips = []
    values = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            flips.append(int(row['flip']))
            values.append(float(row['accuracy']))
    return flips, values


def load_from_txt(filepath: str) -> List[float]:
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
    # Data paths
    r1_t01_csv = "results/R1/R1_T01_group_metadata_index_anypattern_curve.csv"
    r1_t02_txt = "/tmp/r1_t02_full.txt"
    r1_t03_txt = "/tmp/r1_t03_full.txt"
    r1_t04_txt = "/tmp/r1_t04_full.txt"

    # Load data
    t01_flips, t01_acc = load_csv(r1_t01_csv)
    t02_acc = load_from_txt(r1_t02_txt)
    t03_acc = load_from_txt(r1_t03_txt)
    t04_acc = load_from_txt(r1_t04_txt)

    # Create x-axis for each
    t01_x = list(range(len(t01_acc)))
    t02_x = list(range(len(t02_acc)))
    t03_x = list(range(len(t03_acc)))
    t04_x = list(range(len(t04_acc)))

    # Create figure with distinct colors
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define distinct colors and markers for each curve
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '-', '-', '-']

    # Plot each curve with its own style
    ax.plot(t01_x, t01_acc,
            marker=markers[0], markersize=5,
            color=colors[0], linestyle=linestyles[0],
            linewidth=2.5, alpha=0.9,
            label='R1_T01: Index (Any Pattern)')
    ax.plot(t02_x, t02_acc,
            marker=markers[1], markersize=5,
            color=colors[1], linestyle=linestyles[1],
            linewidth=2.5, alpha=0.9,
            label='R1_T02: Index (1-bit Reachable)')
    ax.plot(t03_x, t03_acc,
            marker=markers[2], markersize=5,
            color=colors[2], linestyle=linestyles[2],
            linewidth=2.5, alpha=0.9,
            label='R1_T03: Bitmask (25 Swaps, 50 Flips)')
    ax.plot(t04_x, t04_acc,
            marker=markers[3], markersize=5,
            color=colors[3], linestyle=linestyles[3],
            linewidth=2.5, alpha=0.9,
            label='R1_T04: Bitmask (50 Swaps, 100 Flips)')

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("Operations (Physical Flips or Logical Swaps)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=13, fontweight="bold")
    ax.set_title("R1 Workflow: Metadata Attack Comparison (ResNet-20/CIFAR-10, INT8)",
                 fontsize=15, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)

    # Add horizontal line at random chance (10% for CIFAR-10)
    ax.axhline(y=10, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    ax.text(1, 11, 'Random (10%)', fontsize=9, color='gray')

    # Add baseline reference line
    ax.axhline(y=92.33, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax.text(51, 93.5, f'Baseline: 92.33%', fontsize=9, color='black')

    # Legend with summary stats
    legend = ax.legend(loc='lower left', fontsize=10, framealpha=0.95)

    fig.tight_layout()

    # Save plot
    output_path = "results/R1/R1_comparison_curve.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Output] Saved {output_path}")

    # Also create a comparison table
    table_path = "results/R1/R1_comparison_table.csv"
    with open(table_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'task', 'encoding', 'constraint', 'ops_count', 'baseline_acc',
            'final_acc', 'acc_drop', 'color'
        ])
        writer.writerow([
            'R1_T01', 'Index/Position', 'Any Pattern', '50',
            '92.33', f'{t01_acc[-1]:.2f}', f'{92.33 - t01_acc[-1]:.2f}', 'Blue'
        ])
        writer.writerow([
            'R1_T02', 'Index/Position', '1-bit Reachable', '50',
            '92.33', f'{t02_acc[-1]:.2f}', f'{92.33 - t02_acc[-1]:.2f}', 'Orange'
        ])
        writer.writerow([
            'R1_T03', 'Bitmask', 'Cost-2 Swap', '25 swaps / 50 flips',
            '92.33', f'{t03_acc[-1]:.2f}', f'{92.33 - t03_acc[-1]:.2f}', 'Green'
        ])
        writer.writerow([
            'R1_T04', 'Bitmask', 'Cost-2 Swap', '50 swaps / 100 flips',
            '92.33', f'{t04_acc[-1]:.2f}', f'{92.33 - t04_acc[-1]:.2f}', 'Red'
        ])
    print(f"[Output] Saved {table_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("R1 Workflow Comparison Summary")
    print("=" * 80)
    print(f"{'Task':<10} {'Encoding':<20} {'Constraint':<25} {'Ops':<12} {'Final':<10} {'Drop':<10}")
    print("-" * 80)
    print(f"{'R1_T01':<10} {'Index/Position':<20} {'Any Pattern':<25} {'50':<12} {t01_acc[-1]:>10.2f}% {92.33 - t01_acc[-1]:>10.2f}%")
    print(f"{'R1_T02':<10} {'Index/Position':<20} {'1-bit Reachable':<25} {'50':<12} {t02_acc[-1]:>10.2f}% {92.33 - t02_acc[-1]:>10.2f}%")
    print(f"{'R1_T03':<10} {'Bitmask':<20} {'25 swaps / 50 flips':<25} {'25/50':<12} {t03_acc[-1]:>10.2f}% {92.33 - t03_acc[-1]:>10.2f}%")
    print(f"{'R1_T04':<10} {'Bitmask':<20} {'50 swaps / 100 flips':<25} {'50/100':<12} {t04_acc[-1]:>10.2f}% {92.33 - t04_acc[-1]:>10.2f}%")
    print("=" * 80)
    print(f"\nFiles:")
    print(f"  Plot: {output_path}")
    print(f"  Table: {table_path}")


if __name__ == "__main__":
    main()
