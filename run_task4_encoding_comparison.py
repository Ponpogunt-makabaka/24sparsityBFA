#!/usr/bin/env python3
"""
Task 4: Encoding Format Comparison (Bitmask vs. Position)

Compares two sparse encoding schemes:
- Position-based (CSR indices): flip bits in column indices
- Bitmask-based (2:4 mask): flip bits in sparse mask
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib.pyplot as plt

from models.resnet20 import resnet20
from models.sparse_csr import create_csr_model_from_sparse
from train.train_utils import get_cifar10_loaders
from train.ptq_convert import Int8QuantizedResNet
from bfa.encoded_sparse_attack import run_csr_encoded_attack
from bfa.bitmask_attack import run_bitmask_attack


def load_sparse_int8_model(device):
    """Load Sparse Int8 model with frozen sparse masks."""
    base_model = resnet20(sparsity_type="2:4").to(device)
    checkpoint = torch.load('models/sparse_model.pth', map_location=device)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()
    base_model.freeze_sparse_masks()

    model = Int8QuantizedResNet(base_model, copy_sparse_masks=True).to(device)
    model.calibrate_all_layers()
    model.eval()
    return model


def run_task4_experiments():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    _, test_loader = get_cifar10_loaders(batch_size=256)
    os.makedirs('./results', exist_ok=True)

    results = {}

    # ============================================================
    # Attack 1: Position-based (CSR indices)
    # ============================================================
    print("\n" + "=" * 60)
    print("ATTACK 1: Position-based Encoding (CSR Index Bits)")
    print("=" * 60)

    csr_model = create_csr_model_from_sparse(device)
    result_position = run_csr_encoded_attack(
        model=csr_model,
        test_loader=test_loader,
        attack_type='index_position',
        max_flips=50,
        target_accuracy=0.1,
        calib_samples=100,
        log_interval=1,
        save_path='./results/task4_position_result.pkl',
        save_log_path='./results/task4_position_log.txt'
    )
    results['position'] = result_position
    print(f"\n[Position] {result_position.initial_accuracy:.2f}% → {result_position.final_accuracy:.2f}% ({result_position.total_flips} flips)")

    # ============================================================
    # Attack 2: Bitmask-based (2:4 mask bits)
    # ============================================================
    print("\n" + "=" * 60)
    print("ATTACK 2: Bitmask-based Encoding (Sparse Mask Bits)")
    print("=" * 60)

    bitmask_model = load_sparse_int8_model(device)
    result_bitmask = run_bitmask_attack(
        model=bitmask_model,
        test_loader=test_loader,
        max_flips=50,
        target_accuracy=0.1,
        calib_samples=100,
        log_interval=1,
        save_path='./results/task4_bitmask_result.pkl',
        save_log_path='./results/task4_bitmask_log.txt'
    )
    results['bitmask'] = result_bitmask
    print(f"\n[Bitmask] {result_bitmask.initial_accuracy:.2f}% → {result_bitmask.final_accuracy:.2f}% ({result_bitmask.total_flips} flips)")

    # ============================================================
    # Plot: Accuracy vs Flips
    # ============================================================
    generate_comparison_plot(results)

    # ============================================================
    # Analysis & Summary
    # ============================================================
    analyze_and_write_summary(results)

    return results


def generate_comparison_plot(results):
    """Plot accuracy vs flips for both encodings."""
    plt.figure(figsize=(12, 7))

    pos = results['position']
    bm = results['bitmask']

    flips_pos = list(range(len(pos.accuracy_history)))
    flips_bm = list(range(len(bm.accuracy_history)))

    plt.plot(flips_pos, pos.accuracy_history, marker='o', linestyle='-', linewidth=2,
             markersize=6, label='Position-based (CSR indices)', color='#1f77b4')
    plt.plot(flips_bm, bm.accuracy_history, marker='s', linestyle='-', linewidth=2,
             markersize=6, label='Bitmask-based (2:4 mask)', color='#ff7f0e')

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('Bit Flips (Iterations)', fontsize=14, fontweight='bold')
    plt.ylabel('Top-1 Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('Task 4: Encoding Robustness\nBitmask vs Position-based Encoding', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.ylim(0, 100)
    plt.xlim(left=0)

    plt.axhline(y=10, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    plt.axhline(y=50, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig('./results/task4_bitmask_vs_position.png', dpi=150, bbox_inches='tight')
    print("[Task 4] Plot saved to: ./results/task4_bitmask_vs_position.png")
    plt.close()


def analyze_and_write_summary(results):
    """Analyze robustness and write summary to file."""
    pos = results['position']
    bm = results['bitmask']

    # Flips to 10% absolute accuracy
    def flips_to_target(acc_history, target=10.0):
        for i, acc in enumerate(acc_history):
            if acc <= target:
                return i
        return None

    pos_to_10 = flips_to_target(pos.accuracy_history)
    bm_to_10 = flips_to_target(bm.accuracy_history)

    pos_drop = pos.initial_accuracy - pos.final_accuracy
    bm_drop = bm.initial_accuracy - bm.final_accuracy

    # Structural divergence proxy: number of flips (unique bits)
    pos_divergence = pos.total_flips
    bm_divergence = bm.total_flips

    conclusion = "Bitmasks are MORE robust than explicit indices." if bm_drop < pos_drop else "Bitmasks are LESS robust than explicit indices."

    with open('./results/task4_encoding_comparison.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Task 4: Encoding Format Comparison (Bitmask vs Position)\n")
        f.write("=" * 60 + "\n\n")

        f.write("Position-based (CSR indices):\n")
        f.write(f"  Initial Accuracy: {pos.initial_accuracy:.2f}%\n")
        f.write(f"  Final Accuracy: {pos.final_accuracy:.2f}%\n")
        f.write(f"  Total Flips: {pos.total_flips}\n")
        f.write(f"  Accuracy Drop: {pos_drop:.2f}%\n")
        f.write(f"  Flips to 10% Acc: {pos_to_10 if pos_to_10 is not None else 'Not reached'}\n")
        f.write(f"  Structural Divergence (proxy): {pos_divergence}\n\n")

        f.write("Bitmask-based (2:4 masks):\n")
        f.write(f"  Initial Accuracy: {bm.initial_accuracy:.2f}%\n")
        f.write(f"  Final Accuracy: {bm.final_accuracy:.2f}%\n")
        f.write(f"  Total Flips: {bm.total_flips}\n")
        f.write(f"  Accuracy Drop: {bm_drop:.2f}%\n")
        f.write(f"  Flips to 10% Acc: {bm_to_10 if bm_to_10 is not None else 'Not reached'}\n")
        f.write(f"  Structural Divergence (proxy): {bm_divergence}\n\n")

        f.write("Conclusion:\n")
        f.write(f"  {conclusion}\n")

    print("[Task 4] Summary saved to: ./results/task4_encoding_comparison.txt")


if __name__ == '__main__':
    run_task4_experiments()
