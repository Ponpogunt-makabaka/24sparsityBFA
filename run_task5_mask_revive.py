#!/usr/bin/env python3
"""
Task 5: Bitmask Format Attack (0 -> 1 Revival)

Flip sparse mask bits from 0 to 1 on the 2:4 Sparse Int8 model.
Produces a PNG curve in results/.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib.pyplot as plt

from models.factory import create_resnet20
from train.ptq_convert import Int8QuantizedResNet
from train.train_utils import get_cifar10_loaders
from bfa.zero_mask_attack import run_zero_mask_attack


def load_sparse_int8_model(device):
    """Load 2:4 sparse FP32 model via factory and convert to Int8."""
    base_model = create_resnet20(
        sparsity_type="2:4",
        pretrained_path="models/sparse_model.pth"
    ).to(device)
    base_model.eval()
    base_model.freeze_sparse_masks()

    model = Int8QuantizedResNet(base_model, copy_sparse_masks=True).to(device)
    model.calibrate_all_layers()
    model.eval()
    return model


def run_task5():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    _, test_loader = get_cifar10_loaders(batch_size=256)
    os.makedirs('./results', exist_ok=True)

    print("\n" + "=" * 60)
    print("TASK 5: Mask Revival Attack (0 -> 1)")
    print("=" * 60)
    print("Targeting: Only zero masks (revive pruned weights)\n")

    model = load_sparse_int8_model(device)

    result = run_zero_mask_attack(
        model=model,
        test_loader=test_loader,
        max_flips=50,
        target_accuracy=0.1,
        calib_samples=100,
        log_interval=1,
        save_path='./results/task5_mask_revive_result.pkl',
        save_log_path='./results/task5_mask_revive_log.txt'
    )

    # Plot
    plt.figure(figsize=(10, 6))
    flips = list(range(len(result.accuracy_history)))
    plt.plot(flips, result.accuracy_history, marker='o', linewidth=2, markersize=5,
             label='Mask Revival (0->1)')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('Bit Flips (Iterations)', fontsize=12, fontweight='bold')
    plt.ylabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Task 5: Mask Revival Attack (0->1)', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('./results/task5_mask_revive.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[Task 5] Plot saved to: ./results/task5_mask_revive.png")

    # Summary file
    with open('./results/task5_mask_revive_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Task 5: Mask Revival Attack Summary (0->1)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Initial Accuracy: {result.initial_accuracy:.2f}%\n")
        f.write(f"Final Accuracy: {result.final_accuracy:.2f}%\n")
        f.write(f"Total Flips: {result.total_flips}\n")
        f.write(f"Total Revived Masks: {len(result.mask_positions)}\n")

    print("[Task 5] Summary saved to: ./results/task5_mask_revive_summary.txt")

    return result


if __name__ == '__main__':
    run_task5()

