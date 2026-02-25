#!/usr/bin/env python3
"""
Task 1: Quantization & Baseline BFA Comparison

Compares BFA vulnerability of:
1. Dense FP32 (baseline)
2. Dense Int8 (PTQ converted)
3. Sparse Int8 (PTQ converted, 2:4 sparsity)

Output:
- Bit-Flips vs Accuracy plot
- Gradient magnitude analysis
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

from models.resnet20 import resnet20
from bfa.fp32_attack import run_bfa_attack as run_fp32_bfa
from bfa.int8_attack import run_int8_bfa_attack
from train.train_utils import get_cifar10_loaders


def load_dense_fp32_model(device):
    """Load Dense FP32 model."""
    model = resnet20(sparsity_type=None).to(device)
    checkpoint = torch.load('models/dense_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def load_dense_int8_model(device):
    """Load Dense Int8 model."""
    from train.ptq_convert import Int8QuantizedResNet
    from models.resnet20 import resnet20

    # Load base FP32 model first
    base_model = resnet20(sparsity_type=None).to(device)
    checkpoint = torch.load('models/dense_model.pth', map_location=device)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()

    # Create Int8 model from base model
    model = Int8QuantizedResNet(base_model, copy_sparse_masks=False).to(device)

    # Calibrate quantization
    model.calibrate_all_layers()
    model.eval()
    return model


def load_sparse_int8_model(device):
    """Load Sparse Int8 model."""
    from train.ptq_convert import Int8QuantizedResNet
    from models.resnet20 import resnet20

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


def run_task1_experiments():
    """Run all BFA experiments for Task 1."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Get data loaders
    _, test_loader = get_cifar10_loaders(batch_size=256)

    os.makedirs('./results', exist_ok=True)

    results = {}

    # ============================================================
    # Experiment 1: Dense FP32 (Baseline)
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT 1: Dense FP32 BFA Attack (Baseline)")
    print("="*60)

    model_fp32 = load_dense_fp32_model(device)

    result_fp32 = run_fp32_bfa(
        model=model_fp32,
        test_loader=test_loader,
        mode='dense',
        max_flips=50,
        target_accuracy=0.1,
        bits_per_round=1,
        calib_samples=100,
        log_interval=1,
        save_path='./results/task1_fp32_result.pkl',
        save_log_path='./results/task1_fp32_log.txt',
        model_type='dense_fp32'
    )

    results['fp32'] = result_fp32
    print(f"\nDense FP32: {result_fp32.initial_accuracy:.2f}% → {result_fp32.final_accuracy:.2f}% ({result_fp32.total_flips} flips)")

    # ============================================================
    # Experiment 2: Dense Int8
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT 2: Dense Int8 BFA Attack")
    print("="*60)

    model_dense_int8 = load_dense_int8_model(device)

    result_dense_int8 = run_int8_bfa_attack(
        model=model_dense_int8,
        test_loader=test_loader,
        max_flips=50,
        target_accuracy=0.1,
        calib_samples=100,
        log_interval=1,
        save_path='./results/task1_dense_int8_result.pkl',
        save_log_path='./results/task1_dense_int8_log.txt',
        model_type='dense_int8'
    )

    results['dense_int8'] = result_dense_int8
    print(f"\nDense Int8: {result_dense_int8.initial_accuracy:.2f}% → {result_dense_int8.final_accuracy:.2f}% ({result_dense_int8.total_flips} flips)")

    # ============================================================
    # Experiment 3: Sparse Int8 (2:4)
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT 3: Sparse Int8 BFA Attack (2:4)")
    print("="*60)

    model_sparse_int8 = load_sparse_int8_model(device)

    result_sparse_int8 = run_int8_bfa_attack(
        model=model_sparse_int8,
        test_loader=test_loader,
        max_flips=50,
        target_accuracy=0.1,
        calib_samples=100,
        log_interval=1,
        save_path='./results/task1_sparse_int8_result.pkl',
        save_log_path='./results/task1_sparse_int8_log.txt',
        model_type='sparse_int8'
    )

    results['sparse_int8'] = result_sparse_int8
    print(f"\nSparse Int8: {result_sparse_int8.initial_accuracy:.2f}% → {result_sparse_int8.final_accuracy:.2f}% ({result_sparse_int8.total_flips} flips)")

    # ============================================================
    # Generate Comparison Plot
    # ============================================================
    print("\n" + "="*60)
    print("Generating comparison plot...")
    print("="*60)

    generate_comparison_plot(results)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("TASK 1 SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Initial Acc':>12} {'Final Acc':>12} {'Flips':>8} {'To 10%':>8}")
    print("-" * 60)

    for model_type, result in results.items():
        to_10 = result.total_flips if result.final_accuracy < 15 else ">50"
        print(f"{model_type:<20} {result.initial_accuracy:>11.2f}% {result.final_accuracy:>11.2f}% {result.total_flips:>8d} {to_10:>8}")

    print("-" * 60)

    return results


def generate_comparison_plot(results):
    """Generate Bit-Flips vs Accuracy comparison plot."""
    plt.figure(figsize=(12, 7))

    # Extract data
    for model_type, result in results.items():
        flips = list(range(len(result.accuracy_history)))
        acc = result.accuracy_history

        if model_type == 'fp32':
            label = 'Dense FP32'
            color = '#1f77b4'
            marker = 'o'
        elif model_type == 'dense_int8':
            label = 'Dense Int8'
            color = '#ff7f0e'
            marker = 's'
        else:  # sparse_int8
            label = 'Sparse Int8 (2:4)'
            color = '#2ca02c'
            marker = '^'

        plt.plot(flips, acc, marker=marker, linestyle='-', linewidth=2, markersize=6,
                label=label, color=color)

    # Add grid and labels
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('Bit Flips (Iterations)', fontsize=14, fontweight='bold')
    plt.ylabel('Top-1 Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('Task 1: BFA Vulnerability Comparison\nBit-Flips vs Accuracy', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=12)
    plt.ylim(0, 100)
    plt.xlim(left=0)

    # Add threshold lines
    plt.axhline(y=10, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Random (10%)')
    plt.axhline(y=50, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='50% Accuracy')

    plt.tight_layout()
    plt.savefig('./results/task1_baseline_comparison.png', dpi=150, bbox_inches='tight')
    print("[Task 1] Plot saved to: ./results/task1_baseline_comparison.png")
    plt.close()


def analyze_gradient_magnitudes():
    """Analyze gradient magnitude differences between FP32 and Int8."""
    print("\n" + "="*60)
    print("Gradient Magnitude Analysis")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, test_loader = get_cifar10_loaders(batch_size=256)

    # Get a small batch for analysis
    inputs, targets = next(iter(test_loader))
    inputs, targets = inputs.to(device), targets.to(device)

    # Analyze FP32
    model_fp32 = load_dense_fp32_model(device)
    model_fp32.eval()

    criterion = torch.nn.CrossEntropyLoss()

    # FP32 gradients
    outputs = model_fp32(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    fp32_grads = []
    for name, param in model_fp32.named_parameters():
        if 'weight' in name and param.grad is not None:
            fp32_grads.extend(param.grad.data.abs().flatten().cpu().tolist())

    # Int8 gradients
    model_int8 = load_dense_int8_model(device)
    model_int8.eval()

    model_fp32.zero_grad()
    outputs = model_int8(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    int8_grads = []
    for name, module in model_int8.named_modules():
        if hasattr(module, 'weight') and module.weight.grad is not None:
            int8_grads.extend(module.weight.grad.data.abs().flatten().cpu().tolist())

    # Statistics
    fp32_grads = np.array(fp32_grads)
    int8_grads = np.array(int8_grads)

    print(f"\nFP32 Gradient Statistics:")
    print(f"  Mean: {fp32_grads.mean():.6f}")
    print(f"  Std:  {fp32_grads.std():.6f}")
    print(f"  Max:  {fp32_grads.max():.6f}")
    print(f"  Median: {np.median(fp32_grads):.6f}")

    print(f"\nInt8 Gradient Statistics:")
    print(f"  Mean: {int8_grads.mean():.6f}")
    print(f"  Std:  {int8_grads.std():.6f}")
    print(f"  Max:  {int8_grads.max():.6f}")
    print(f"  Median: {np.median(int8_grads):.6f}")

    # Save analysis
    with open('./results/task1_gradient_analysis.txt', 'w') as f:
        f.write("Gradient Magnitude Analysis: FP32 vs Int8\n")
        f.write("="*60 + "\n\n")
        f.write(f"FP32 Gradient Statistics:\n")
        f.write(f"  Mean: {fp32_grads.mean():.6f}\n")
        f.write(f"  Std:  {fp32_grads.std():.6f}\n")
        f.write(f"  Max:  {fp32_grads.max():.6f}\n")
        f.write(f"  Median: {np.median(fp32_grads):.6f}\n\n")
        f.write(f"Int8 Gradient Statistics:\n")
        f.write(f"  Mean: {int8_grads.mean():.6f}\n")
        f.write(f"  Std:  {int8_grads.std():.6f}\n")
        f.write(f"  Max:  {int8_grads.max():.6f}\n")
        f.write(f"  Median: {np.median(int8_grads):.6f}\n\n")
        f.write(f"Ratio (Int8/FP32):\n")
        f.write(f"  Mean: {int8_grads.mean()/fp32_grads.mean():.6f}\n")
        f.write(f"  Std:  {int8_grads.std()/fp32_grads.std():.6f}\n")

    print("\n[Task 1] Gradient analysis saved to: ./results/task1_gradient_analysis.txt")


if __name__ == '__main__':
    # Run experiments
    results = run_task1_experiments()

    # Analyze gradients
    analyze_gradient_magnitudes()

    print("\n[Task 1] All experiments complete!")
