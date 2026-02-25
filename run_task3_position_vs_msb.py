#!/usr/bin/env python3
"""
Task 3: Encoded Sparse Model Attack (Position vs MSB)

Compares two attack vectors on CSR-encoded sparse models:
- Vector A: Attack Value MSB (weight value corruption)
- Vector B: Attack Index Position (column index corruption)

This tests whether shifting a weight to wrong position (index flip)
or negating the weight value (MSB flip) causes faster collapse.

Output:
- Comparison plot
- Index corruption analysis
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib.pyplot as plt
from typing import Dict

from models.sparse_csr import create_csr_model_from_sparse
from bfa.encoded_sparse_attack import run_csr_encoded_attack
from train.train_utils import get_cifar10_loaders


def run_task3_experiments():
    """Run all CSR-encoded attack experiments for Task 3."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Get data loaders
    _, test_loader = get_cifar10_loaders(batch_size=256)

    os.makedirs('./results', exist_ok=True)

    results = {}

    # ============================================================
    # Vector A: Attack Value MSB (Weight Value Corruption)
    # ============================================================
    print("\n" + "="*60)
    print("VECTOR A: Value MSB Attack (Weight Value Corruption)")
    print("="*60)
    print("Targeting: Bit 7 (sign bit) of CSR values array")
    print("Effect: Negating/Changing the weight value\n")

    # Load CSR-encoded model
    print("[Task 3] Creating CSR-encoded sparse model...")
    model_value_attack = create_csr_model_from_sparse(device)

    result_value = run_csr_encoded_attack(
        model=model_value_attack,
        test_loader=test_loader,
        attack_type='value_msb',
        max_flips=50,
        target_accuracy=0.1,
        calib_samples=100,
        log_interval=1,
        save_path='./results/task3_value_msb_result.pkl',
        save_log_path='./results/task3_value_msb_log.txt'
    )

    results['value_msb'] = result_value
    print(f"\n[Vector A] Value MSB Attack: {result_value.initial_accuracy:.2f}% → {result_value.final_accuracy:.2f}% ({result_value.total_flips} flips)")

    # ============================================================
    # Vector B: Attack Index Position (Column Index Corruption)
    # ============================================================
    print("\n" + "="*60)
    print("VECTOR B: Index Position Attack (Column Index Corruption)")
    print("="*60)
    print("Targeting: Bits of CSR column_indices array")
    print("Effect: Weight points to wrong input activation column\n")

    # Load fresh CSR-encoded model
    print("[Task 3] Creating fresh CSR-encoded sparse model...")
    model_index_attack = create_csr_model_from_sparse(device)

    result_index = run_csr_encoded_attack(
        model=model_index_attack,
        test_loader=test_loader,
        attack_type='index_position',
        max_flips=50,
        target_accuracy=0.1,
        calib_samples=100,
        log_interval=1,
        save_path='./results/task3_index_position_result.pkl',
        save_log_path='./results/task3_index_position_log.txt'
    )

    results['index_position'] = result_index
    print(f"\n[Vector B] Index Position Attack: {result_index.initial_accuracy:.2f}% → {result_index.final_accuracy:.2f}% ({result_index.total_flips} flips)")

    # ============================================================
    # Generate Comparison Plot
    # ============================================================
    print("\n" + "="*60)
    print("Generating comparison plot...")
    print("="*60)

    generate_comparison_plot(results)

    # ============================================================
    # Analysis and Conclusions
    # ============================================================
    print("\n" + "="*60)
    print("TASK 3 ANALYSIS")
    print("="*60)

    analyze_index_vs_value(results)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("TASK 3 SUMMARY")
    print("="*60)
    print(f"{'Attack Vector':<30} {'Initial Acc':>12} {'Final Acc':>12} {'Flips':>8} {'ΔAcc':>10}")
    print("-" * 70)

    for attack_type, result in results.items():
        delta_acc = result.initial_accuracy - result.final_accuracy
        attack_name = "Value MSB Attack" if attack_type == 'value_msb' else "Index Position Attack"
        print(f"{attack_name:<30} {result.initial_accuracy:>11.2f}% {result.final_accuracy:>11.2f}% {result.total_flips:>8d} {delta_acc:>9.2f}%")

    print("-" * 70)

    return results


def generate_comparison_plot(results):
    """Generate Bit-Flips vs Accuracy comparison plot for Task 3."""
    plt.figure(figsize=(12, 7))

    # Vector A: Value MSB Attack
    result_a = results['value_msb']
    flips_a = list(range(len(result_a.accuracy_history)))
    acc_a = result_a.accuracy_history

    # Vector B: Index Position Attack
    result_b = results['index_position']
    flips_b = list(range(len(result_b.accuracy_history)))
    acc_b = result_b.accuracy_history

    # Plot both vectors
    plt.plot(flips_a, acc_a, marker='o', linestyle='-', linewidth=2, markersize=6,
             label='Vector A: Value MSB Attack', color='#e377c2')
    plt.plot(flips_b, acc_b, marker='s', linestyle='-', linewidth=2, markersize=6,
             label='Vector B: Index Position Attack', color='#17becf')

    # Add grid and labels
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('Bit Flips (Iterations)', fontsize=14, fontweight='bold')
    plt.ylabel('Top-1 Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('Task 3: CSR Encoded Sparse Attack Comparison\nValue MSB vs Index Position Corruption',
              fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.ylim(0, 100)
    plt.xlim(left=0)

    # Add threshold lines
    plt.axhline(y=10, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    plt.axhline(y=50, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig('./results/task3_position_vs_msb.png', dpi=150, bbox_inches='tight')
    print("[Task 3] Plot saved to: ./results/task3_position_vs_msb.png")
    plt.close()


def analyze_index_vs_value(results):
    """Analyze which attack vector is more effective."""
    result_value = results['value_msb']
    result_index = results['index_position']

    # Find flips needed to reach 10% accuracy drop
    target_acc_drop = 10.0

    flips_to_10_value = None
    flips_to_10_index = None

    for i, acc in enumerate(result_value.accuracy_history):
        if result_value.initial_accuracy - acc >= target_acc_drop:
            flips_to_10_value = i
            break

    for i, acc in enumerate(result_index.accuracy_history):
        if result_index.initial_accuracy - acc >= target_acc_drop:
            flips_to_10_index = i
            break

    # Calculate final accuracy drops
    final_drop_value = result_value.initial_accuracy - result_value.final_accuracy
    final_drop_index = result_index.initial_accuracy - result_index.final_accuracy

    print("\nKey Findings:")
    print("-" * 60)

    if flips_to_10_value is not None:
        print(f"• Value MSB Attack: {flips_to_10_value} flips to reach 10% accuracy drop")
    else:
        print(f"• Value MSB Attack: Did NOT reach 10% drop in {result_value.total_flips} flips")

    if flips_to_10_index is not None:
        print(f"• Index Position Attack: {flips_to_10_index} flips to reach 10% accuracy drop")
    else:
        print(f"• Index Position Attack: Did NOT reach 10% drop in {result_index.total_flips} flips")

    print(f"\n• Value MSB Attack final accuracy drop: {final_drop_value:.2f}%")
    print(f"• Index Position Attack final accuracy drop: {final_drop_index:.2f}%")

    # Determine which is more effective
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if final_drop_value > final_drop_index * 1.5:
        print("Value MSB corruption is MORE DAMAGING than index position corruption.")
        print("This indicates that changing the weight values (magnitude/sign)")
        print("has greater impact than misaligning weights to wrong positions.")
    elif final_drop_index > final_drop_value * 1.5:
        print("Index Position corruption is MORE DAMAGING than value MSB corruption.")
        print("This indicates that structural misalignment (weights applied to")
        print("wrong activations) is more damaging than value changes.")
    else:
        print("Both attack vectors have SIMILAR effectiveness.")

    print("=" * 60)

    # Save analysis to file
    with open('./results/task3_index_corruption_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Task 3: CSR Encoded Sparse Attack Analysis\n")
        f.write("=" * 60 + "\n\n")

        f.write("Attack Vectors:\n")
        f.write("  Vector A: Value MSB Attack\n")
        f.write("    - Target: Bit 7 (sign bit) of CSR values array\n")
        f.write("    - Effect: Weight value sign/magnitude change\n\n")
        f.write("  Vector B: Index Position Attack\n")
        f.write("    - Target: Bits of CSR column_indices array\n")
        f.write("    - Effect: Weight points to wrong input activation column\n\n")

        f.write("Results:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Value MSB Attack:\n")
        f.write(f"  Initial Accuracy: {result_value.initial_accuracy:.2f}%\n")
        f.write(f"  Final Accuracy: {result_value.final_accuracy:.2f}%\n")
        f.write(f"  Total Flips: {result_value.total_flips}\n")
        f.write(f"  Accuracy Drop: {final_drop_value:.2f}%\n\n")

        f.write(f"Index Position Attack:\n")
        f.write(f"  Initial Accuracy: {result_index.initial_accuracy:.2f}%\n")
        f.write(f"  Final Accuracy: {result_index.final_accuracy:.2f}%\n")
        f.write(f"  Total Flips: {result_index.total_flips}\n")
        f.write(f"  Accuracy Drop: {final_drop_index:.2f}%\n\n")

        f.write("=" * 60 + "\n")
        f.write("CONCLUSION\n")
        f.write("=" * 60 + "\n")

        if final_drop_value > final_drop_index * 1.5:
            f.write("Value MSB corruption is MORE DAMAGING.\n")
            f.write("Weight value changes have greater impact than structural misalignment.\n")
        elif final_drop_index > final_drop_value * 1.5:
            f.write("Index Position corruption is MORE DAMAGING.\n")
            f.write("Structural misalignment is more damaging than value changes.\n")
        else:
            f.write("Both attack vectors have SIMILAR effectiveness.\n")

    print("[Task 3] Analysis saved to: ./results/task3_index_corruption_analysis.txt")


if __name__ == '__main__':
    # Run experiments
    results = run_task3_experiments()

    print("\n[Task 3] All experiments complete!")
