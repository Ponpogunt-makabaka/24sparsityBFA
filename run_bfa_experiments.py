#!/usr/bin/env python3
"""
Run BFA experiments on Dense and Sparse models.

CLAUDEv4 Paper Baseline Reproduction:
- Parameters: max_flips=50, bits_per_round=1, log_interval=1
- Each iteration flips exactly 1 bit
- Gradients recomputed each iteration
- Max 50 iterations or until accuracy < 10%
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.resnet20 import resnet20
from bfa.fp32_attack import run_bfa_attack
from train.train_utils import get_cifar10_loaders

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Get data loaders
    _, test_loader = get_cifar10_loaders(batch_size=128)

    # CSV output path
    csv_path = './results/comparison_table.csv'
    os.makedirs('./results', exist_ok=True)

    # Remove old CSV if exists
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"Removed old CSV: {csv_path}")

    # ============================================================
    # Experiment 1: Dense Model
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT 1: Dense Model BFA Attack")
    print("="*60)

    model_dense = resnet20(sparsity_type=None).to(device)
    checkpoint = torch.load('./models/dense_model.pth', map_location=device)
    model_dense.load_state_dict(checkpoint['model_state_dict'])
    model_dense.eval()

    print(f"Dense model loaded. Best accuracy: {checkpoint.get('best_acc', 'N/A')}%")

    result_dense = run_bfa_attack(
        model=model_dense,
        test_loader=test_loader,
        mode='dense',
        max_flips=50,
        target_accuracy=0.1,
        bits_per_round=1,
        calib_samples=100,
        log_interval=1,
        save_path='./results/baseline_dense_history.pkl',
        save_log_path='./results/attack_dense_log.txt',
        save_csv_path=csv_path,
        model_type='dense',
        attack_mode='initial'
    )

    print(f"\nDense Model BFA Result:")
    print(f"  Initial Acc: {result_dense.initial_accuracy:.2f}%")
    print(f"  Final Acc: {result_dense.final_accuracy:.2f}%")
    print(f"  Total Flips: {result_dense.total_flips}")

    # ============================================================
    # Experiment 2: Sparse Model - Mode A (Dynamic)
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT 2: Sparse Model BFA Attack (Mode A - Dynamic)")
    print("="*60)

    model_sparse_a = resnet20(sparsity_type="2:4").to(device)
    checkpoint = torch.load('./models/sparse_model.pth', map_location=device)
    model_sparse_a.load_state_dict(checkpoint['model_state_dict'])
    model_sparse_a.eval()

    print(f"Sparse model loaded. Best accuracy: {checkpoint.get('best_acc', 'N/A')}%")

    result_sparse_a = run_bfa_attack(
        model=model_sparse_a,
        test_loader=test_loader,
        mode='dynamic',
        max_flips=50,
        target_accuracy=0.1,
        bits_per_round=1,
        calib_samples=100,
        log_interval=1,
        save_path='./results/sparse_mode_A_history.pkl',
        save_log_path='./results/attack_sparse_A_log.txt',
        save_csv_path=csv_path,
        model_type='sparse',
        attack_mode='dynamic'
    )

    print(f"\nSparse Model (Mode A) BFA Result:")
    print(f"  Initial Acc: {result_sparse_a.initial_accuracy:.2f}%")
    print(f"  Final Acc: {result_sparse_a.final_accuracy:.2f}%")
    print(f"  Total Flips: {result_sparse_a.total_flips}")

    # ============================================================
    # Experiment 3: Sparse Model - Mode B (Static)
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT 3: Sparse Model BFA Attack (Mode B - Static)")
    print("="*60)

    model_sparse_b = resnet20(sparsity_type="2:4").to(device)
    checkpoint = torch.load('./models/sparse_model.pth', map_location=device)
    model_sparse_b.load_state_dict(checkpoint['model_state_dict'])
    model_sparse_b.eval()

    print(f"Sparse model loaded. Best accuracy: {checkpoint.get('best_acc', 'N/A')}%")

    result_sparse_b = run_bfa_attack(
        model=model_sparse_b,
        test_loader=test_loader,
        mode='static',
        max_flips=50,
        target_accuracy=0.1,
        bits_per_round=1,
        calib_samples=100,
        log_interval=1,
        save_path='./results/sparse_mode_B_history.pkl',
        save_log_path='./results/attack_sparse_B_log.txt',
        save_csv_path=csv_path,
        model_type='sparse',
        attack_mode='static'
    )

    print(f"\nSparse Model (Mode B) BFA Result:")
    print(f"  Initial Acc: {result_sparse_b.initial_accuracy:.2f}%")
    print(f"  Final Acc: {result_sparse_b.final_accuracy:.2f}%")
    print(f"  Total Flips: {result_sparse_b.total_flips}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"{'Model':<15} {'Mode':<12} {'Initial Acc':>12} {'Final Acc':>12} {'Flips':>8}")
    print("-" * 60)
    print(f"{'Dense':<15} {'-':<12} {result_dense.initial_accuracy:>11.2f}% {result_dense.final_accuracy:>11.2f}% {result_dense.total_flips:>8d}")
    print(f"{'Sparse (2:4)':<15} {'Dynamic':<12} {result_sparse_a.initial_accuracy:>11.2f}% {result_sparse_a.final_accuracy:>11.2f}% {result_sparse_a.total_flips:>8d}")
    print(f"{'Sparse (2:4)':<15} {'Static':<12} {result_sparse_b.initial_accuracy:>11.2f}% {result_sparse_b.final_accuracy:>11.2f}% {result_sparse_b.total_flips:>8d}")
    print("-" * 60)
    print(f"\nCSV report saved to: {csv_path}")

if __name__ == '__main__':
    main()
