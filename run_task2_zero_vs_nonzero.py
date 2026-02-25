#!/usr/bin/env python3
"""
Task 2: Intra-Model Sparsity Attack (Zero vs Non-Zero)

Compares two attack scenarios:
- Scenario A: Attack only non-zero weights (value corruption)
- Scenario B: Attack only zero weights (sparsity structure destruction)

Output:
- Comparison plot showing accuracy degradation
- Analysis of which attack vector is more effective
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

from models.resnet20 import resnet20
from bfa.sparse_zero_attack import run_sparse_zero_attack
from train.ptq_convert import Int8QuantizedResNet
from train.train_utils import get_cifar10_loaders


def load_sparse_int8_model(device):
    """Load Sparse Int8 model for Task 2."""
    # Load base FP32 model first
    base_model = resnet20(sparsity_type="2:4").to(device)
    checkpoint = torch.load('models/sparse_model.pth', map_location=device)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()

    # Freeze masks to ensure sparse structure is captured
    base_model.freeze_sparse_masks()

    # Create Int8 model from base model
    model = Int8QuantizedResNet(base_model, copy_sparse_masks=True).to(device)

    # Calibrate quantization
    model.calibrate_all_layers()
    model.eval()
    return model


def run_task2_experiments():
    """Run all zero vs non-zero attack experiments for Task 2."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Get data loaders
    _, test_loader = get_cifar10_loaders(batch_size=256)

    os.makedirs('./results', exist_ok=True)

    results = {}

    # ============================================================
    # Scenario A: Attack Non-Zero Weights (Value Corruption)
    # ============================================================
    print("\n" + "="*60)
    print("SCENARIO A: Non-Zero Weight Attack (Value Corruption)")
    print("="*60)
    print("Targeting: Only non-zero weights (w ≠ 0)")
    print("This simulates: Corruption of existing weight values\n")

    # Load fresh model for Scenario A
    model_scenario_a = load_sparse_int8_model(device)

    result_nonzero = run_sparse_zero_attack(
        model=model_scenario_a,
        test_loader=test_loader,
        scenario='nonzero',
        max_flips=50,
        target_accuracy=0.1,
        calib_samples=100,
        log_interval=1,
        save_path='./results/task2_nonzero_result.pkl',
        save_log_path='./results/task2_nonzero_log.txt'
    )

    results['nonzero'] = result_nonzero
    print(f"\n[Scenario A] Non-Zero Attack: {result_nonzero.initial_accuracy:.2f}% → {result_nonzero.final_accuracy:.2f}% ({result_nonzero.total_flips} flips)")

    # ============================================================
    # Scenario B: Attack Zero Weights (Sparsity Structure Destruction)
    # ============================================================
    print("\n" + "="*60)
    print("SCENARIO B: Zero Weight Attack (Sparsity Structure Destruction)")
    print("="*60)
    print("Targeting: Only zero weights (w = 0)")
    print("This simulates: 'The Rise of the Dead' (breaking sparsity)\n")

    # Load fresh model for Scenario B
    model_scenario_b = load_sparse_int8_model(device)

    result_zero = run_sparse_zero_attack(
        model=model_scenario_b,
        test_loader=test_loader,
        scenario='zero',
        max_flips=50,
        target_accuracy=0.1,
        calib_samples=100,
        log_interval=1,
        save_path='./results/task2_zero_result.pkl',
        save_log_path='./results/task2_zero_log.txt'
    )

    results['zero'] = result_zero
    print(f"\n[Scenario B] Zero Attack: {result_zero.initial_accuracy:.2f}% → {result_zero.final_accuracy:.2f}% ({result_zero.total_flips} flips)")

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
    print("TASK 2 ANALYSIS")
    print("="*60)

    analyze_attack_effectiveness(results)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("TASK 2 SUMMARY")
    print("="*60)
    print(f"{'Scenario':<25} {'Initial Acc':>12} {'Final Acc':>12} {'Flips':>8} {'ΔAcc':>10}")
    print("-" * 60)

    for scenario, result in results.items():
        delta_acc = result.initial_accuracy - result.final_accuracy
        scenario_name = "Non-Zero Attack" if scenario == 'nonzero' else "Zero Attack"
        print(f"{scenario_name:<25} {result.initial_accuracy:>11.2f}% {result.final_accuracy:>11.2f}% {result.total_flips:>8d} {delta_acc:>9.2f}%")

    print("-" * 60)

    return results


def generate_comparison_plot(results):
    """Generate Bit-Flips vs Accuracy comparison plot for Task 2."""
    plt.figure(figsize=(12, 7))

    # Scenario A: Non-Zero Attack
    result_a = results['nonzero']
    flips_a = list(range(len(result_a.accuracy_history)))
    acc_a = result_a.accuracy_history

    # Scenario B: Zero Attack
    result_b = results['zero']
    flips_b = list(range(len(result_b.accuracy_history)))
    acc_b = result_b.accuracy_history

    # Plot both scenarios
    plt.plot(flips_a, acc_a, marker='o', linestyle='-', linewidth=2, markersize=6,
             label='Scenario A: Non-Zero Attack (Value Corruption)', color='#d62728')
    plt.plot(flips_b, acc_b, marker='s', linestyle='-', linewidth=2, markersize=6,
             label='Scenario B: Zero Attack (Structure Destruction)', color='#9467bd')

    # Add grid and labels
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('Bit Flips (Iterations)', fontsize=14, fontweight='bold')
    plt.ylabel('Top-1 Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('Task 2: Zero vs Non-Zero Weight Attack Comparison\nValue Corruption vs Sparsity Structure Destruction',
              fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.ylim(0, 100)
    plt.xlim(left=0)

    # Add threshold lines
    plt.axhline(y=10, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    plt.axhline(y=50, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='50% Accuracy')

    plt.tight_layout()
    plt.savefig('./results/task2_zero_vs_nonzero.png', dpi=150, bbox_inches='tight')
    print("[Task 2] Plot saved to: ./results/task2_zero_vs_nonzero.png")
    plt.close()


def analyze_attack_effectiveness(results):
    """Analyze which attack vector is more effective."""
    result_nonzero = results['nonzero']
    result_zero = results['zero']

    # Find flips needed to reach 10% accuracy drop
    target_acc_drop = 10.0

    flips_to_10_nonzero = None
    flips_to_10_zero = None

    for i, acc in enumerate(result_nonzero.accuracy_history):
        if result_nonzero.initial_accuracy - acc >= target_acc_drop:
            flips_to_10_nonzero = i
            break

    for i, acc in enumerate(result_zero.accuracy_history):
        if result_zero.initial_accuracy - acc >= target_acc_drop:
            flips_to_10_zero = i
            break

    # Calculate final accuracy drops
    final_drop_nonzero = result_nonzero.initial_accuracy - result_nonzero.final_accuracy
    final_drop_zero = result_zero.initial_accuracy - result_zero.final_accuracy

    print("\nKey Findings:")
    print("-" * 60)

    if flips_to_10_nonzero is not None:
        print(f"• Non-Zero Attack: {flips_to_10_nonzero} flips to reach 10% accuracy drop")
    else:
        print(f"• Non-Zero Attack: Did NOT reach 10% drop in {result_nonzero.total_flips} flips")

    if flips_to_10_zero is not None:
        print(f"• Zero Attack: {flips_to_10_zero} flips to reach 10% accuracy drop")
    else:
        print(f"• Zero Attack: Did NOT reach 10% drop in {result_zero.total_flips} flips")

    print(f"\n• Non-Zero Attack final accuracy drop: {final_drop_nonzero:.2f}%")
    print(f"• Zero Attack final accuracy drop: {final_drop_zero:.2f}%")

    # Determine which is more effective
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if final_drop_nonzero > final_drop_zero * 1.5:
        print("The model is MORE SENSITIVE to Non-Zero Weight Attack (Value Corruption).")
        print("This indicates that corrupting the actual weight values is more damaging")
        print("than introducing spurious non-zero weights.")
    elif final_drop_zero > final_drop_nonzero * 1.5:
        print("The model is MORE SENSITIVE to Zero Weight Attack (Structure Destruction).")
        print("This indicates that breaking the sparsity structure (zombie weights)")
        print("is more damaging than corrupting existing weight values.")
    else:
        print("Both attack vectors have SIMILAR effectiveness.")
        print("The model is equally vulnerable to value corruption and structure destruction.")

    print("=" * 60)

    # Save analysis to file
    with open('./results/task2_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Task 2: Zero vs Non-Zero Weight Attack Analysis\n")
        f.write("=" * 60 + "\n\n")

        f.write("Attack Scenarios:\n")
        f.write("  Scenario A: Non-Zero Attack (Value Corruption)\n")
        f.write("    - Only flips bits in non-zero weights (w ≠ 0)\n")
        f.write("    - Simulates corruption of existing weight values\n\n")
        f.write("  Scenario B: Zero Attack (Structure Destruction)\n")
        f.write("    - Only flips bits in zero weights (w = 0)\n")
        f.write("    - Simulates 'The Rise of the Dead' (breaking sparsity)\n\n")

        f.write("Results:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Non-Zero Attack:\n")
        f.write(f"  Initial Accuracy: {result_nonzero.initial_accuracy:.2f}%\n")
        f.write(f"  Final Accuracy: {result_nonzero.final_accuracy:.2f}%\n")
        f.write(f"  Total Flips: {result_nonzero.total_flips}\n")
        f.write(f"  Accuracy Drop: {final_drop_nonzero:.2f}%\n\n")

        f.write(f"Zero Attack:\n")
        f.write(f"  Initial Accuracy: {result_zero.initial_accuracy:.2f}%\n")
        f.write(f"  Final Accuracy: {result_zero.final_accuracy:.2f}%\n")
        f.write(f"  Total Flips: {result_zero.total_flips}\n")
        f.write(f"  Accuracy Drop: {final_drop_zero:.2f}%\n\n")

        f.write("=" * 60 + "\n")
        f.write("CONCLUSION\n")
        f.write("=" * 60 + "\n")

        if final_drop_nonzero > final_drop_zero * 1.5:
            f.write("The model is MORE SENSITIVE to Non-Zero Weight Attack.\n")
            f.write("Value corruption is more damaging than sparsity structure destruction.\n")
        elif final_drop_zero > final_drop_nonzero * 1.5:
            f.write("The model is MORE SENSITIVE to Zero Weight Attack.\n")
            f.write("Sparsity structure destruction is more damaging than value corruption.\n")
        else:
            f.write("Both attack vectors have SIMILAR effectiveness.\n")

    print("[Task 2] Analysis saved to: ./results/task2_analysis.txt")


if __name__ == '__main__':
    # Run experiments
    results = run_task2_experiments()

    print("\n[Task 2] All experiments complete!")
