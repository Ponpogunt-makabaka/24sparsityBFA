#!/usr/bin/env python3
"""
Sparse INT8 Phase Tasks (Dense format + CSR index)

Tasks:
1) Dense-format global attack (all weights)
2) Dense-format attack only zero weights
3) Dense-format attack only non-zero weights
4) CSR format index-position attack
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import torch
import matplotlib.pyplot as plt

from models.factory import create_resnet20
from train.ptq_convert import Int8QuantizedResNet
from train.train_utils import get_cifar10_loaders
from bfa.int8_attack import run_int8_bfa_attack
from bfa.encoded_sparse_attack import run_csr_encoded_attack, simulate_csr_index_attack
from models.sparse_csr import create_csr_model_from_sparse


def detect_zero_point(model: torch.nn.Module) -> int:
    """
    Detect zero_point if present. Default to 0 for symmetric int8.
    """
    for _, module in model.named_modules():
        if hasattr(module, "zero_point"):
            zp = int(module.zero_point.item()) if hasattr(module.zero_point, "item") else int(module.zero_point)
            return zp
    return 0


def load_sparse_int8_model(device: str):
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


def convert_to_dense_format(model: torch.nn.Module, zero_point: int):
    """
    Convert sparse Int8 model to dense-format storage:
    - set pruned weights to zero_point
    - set mask to all ones (disable pruning in forward)
    """
    for _, module in model.named_modules():
        if hasattr(module, "sparse_mask") and module.sparse_mask is not None:
            mask = module.sparse_mask
            int8_w = module.int8_weights
            # zero out pruned positions
            int8_w[mask < 0.5] = torch.tensor(zero_point, dtype=torch.int8, device=int8_w.device)
            # disable pruning (dense-format behavior)
            module.sparse_mask = torch.ones_like(mask)


def run_tasks():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    _, test_loader = get_cifar10_loaders(batch_size=256)
    os.makedirs("./results", exist_ok=True)

    def load_result_if_exists(path: str):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    zero_point = None

    # ---- Task 1: Dense-format global attack ----
    print("\n" + "=" * 70)
    print("Task 1: Sparse INT8 (Dense Format) - Global Attack")
    print("=" * 70)
    result_t1 = load_result_if_exists("./results/task1_sparse_dense_global_result.pkl")
    if result_t1 is None:
        model_t1 = load_sparse_int8_model(device)
        zero_point = detect_zero_point(model_t1)
        print(f"[Task1] Detected zero_point = {zero_point} (symmetric int8 expected)")
        convert_to_dense_format(model_t1, zero_point)

        result_t1 = run_int8_bfa_attack(
            model=model_t1,
            test_loader=test_loader,
            max_flips=50,
            target_accuracy=0.1,
            calib_samples=100,
            log_interval=1,
            save_path="./results/task1_sparse_dense_global_result.pkl",
            save_log_path="./results/task1_sparse_dense_global_log.txt",
            model_type="sparse_int8_dense_global"
        )
    else:
        print("[Task1] Result exists, skipping recompute.")

    # ---- Task 2: Dense-format attack only zero ----
    print("\n" + "=" * 70)
    print("Task 2: Sparse INT8 (Dense Format) - Attack Only Zero")
    print("=" * 70)
    result_t2 = load_result_if_exists("./results/task2_sparse_dense_zero_result.pkl")
    if result_t2 is None:
        model_t2 = load_sparse_int8_model(device)
        zero_point = detect_zero_point(model_t2)
        print(f"[Task2] Detected zero_point = {zero_point} (symmetric int8 expected)")
        convert_to_dense_format(model_t2, zero_point)

        result_t2 = run_int8_bfa_attack(
            model=model_t2,
            test_loader=test_loader,
            max_flips=50,
            target_accuracy=0.1,
            calib_samples=100,
            log_interval=1,
            save_path="./results/task2_sparse_dense_zero_result.pkl",
            save_log_path="./results/task2_sparse_dense_zero_log.txt",
            model_type="sparse_int8_dense_zero",
            weight_filter=lambda v, zp=zero_point: v == zp
        )
    else:
        print("[Task2] Result exists, skipping recompute.")

    # ---- Task 3: Dense-format attack only non-zero ----
    print("\n" + "=" * 70)
    print("Task 3: Sparse INT8 (Dense Format) - Attack Only Non-Zero")
    print("=" * 70)
    result_t3 = load_result_if_exists("./results/task3_sparse_dense_nonzero_result.pkl")
    if result_t3 is None:
        model_t3 = load_sparse_int8_model(device)
        zero_point = detect_zero_point(model_t3)
        print(f"[Task3] Detected zero_point = {zero_point} (symmetric int8 expected)")
        convert_to_dense_format(model_t3, zero_point)

        result_t3 = run_int8_bfa_attack(
            model=model_t3,
            test_loader=test_loader,
            max_flips=50,
            target_accuracy=0.1,
            calib_samples=100,
            log_interval=1,
            save_path="./results/task3_sparse_dense_nonzero_result.pkl",
            save_log_path="./results/task3_sparse_dense_nonzero_log.txt",
            model_type="sparse_int8_dense_nonzero",
            weight_filter=lambda v, zp=zero_point: v != zp
        )
    else:
        print("[Task3] Result exists, skipping recompute.")

    # ---- Task 4: CSR format index-position attack ----
    print("\n" + "=" * 70)
    print("Task 4: Sparse INT8 (CSR Format) - Index Position Attack")
    print("=" * 70)
    print("[Task4] Example helper: simulate_csr_index_attack(3, bit=3) ->",
          simulate_csr_index_attack(3, 3, 64))

    result_t4 = load_result_if_exists("./results/task4_sparse_csr_index_result.pkl")
    if result_t4 is None:
        csr_model = create_csr_model_from_sparse(device)
        result_t4 = run_csr_encoded_attack(
            model=csr_model,
            test_loader=test_loader,
            attack_type="index_position",
            max_flips=50,
            target_accuracy=0.1,
            calib_samples=100,
            log_interval=1,
            save_path="./results/task4_sparse_csr_index_result.pkl",
            save_log_path="./results/task4_sparse_csr_index_log.txt"
        )
    else:
        print("[Task4] Result exists, skipping recompute.")

    if zero_point is None:
        zero_point = 0

    # ---- Summary plot (Tasks 1-3) ----
    plt.figure(figsize=(12, 7))
    for label, result in [
        ("Dense-Format Global", result_t1),
        ("Dense-Format Zero-Only", result_t2),
        ("Dense-Format NonZero-Only", result_t3),
    ]:
        flips = list(range(len(result.accuracy_history)))
        plt.plot(flips, result.accuracy_history, marker='o', linewidth=2, markersize=4, label=label)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel("Bit Flips (Iterations)", fontsize=12, fontweight='bold')
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight='bold')
    plt.title("Sparse INT8 Dense-Format Attacks (Tasks 1-3)", fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("./results/task1_3_sparse_dense_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ---- Write summary ----
    with open("./results/sparse_tasks_summary.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Sparse INT8 Phase Summary (Tasks 1-4)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Zero point detected: {zero_point}\n\n")
        f.write("Task 1 (Dense Global):\n")
        f.write(f"  Initial Acc: {result_t1.initial_accuracy:.2f}%\n")
        f.write(f"  Final Acc:   {result_t1.final_accuracy:.2f}%\n")
        f.write(f"  Flips:       {result_t1.total_flips}\n\n")
        f.write("Task 2 (Dense Zero-Only):\n")
        f.write(f"  Initial Acc: {result_t2.initial_accuracy:.2f}%\n")
        f.write(f"  Final Acc:   {result_t2.final_accuracy:.2f}%\n")
        f.write(f"  Flips:       {result_t2.total_flips}\n\n")
        f.write("Task 3 (Dense NonZero-Only):\n")
        f.write(f"  Initial Acc: {result_t3.initial_accuracy:.2f}%\n")
        f.write(f"  Final Acc:   {result_t3.final_accuracy:.2f}%\n")
        f.write(f"  Flips:       {result_t3.total_flips}\n\n")
        f.write("Task 4 (CSR Index Attack):\n")
        f.write(f"  Initial Acc: {result_t4.initial_accuracy:.2f}%\n")
        f.write(f"  Final Acc:   {result_t4.final_accuracy:.2f}%\n")
        f.write(f"  Flips:       {result_t4.total_flips}\n")

    print("[Summary] Saved to: ./results/sparse_tasks_summary.txt")


if __name__ == "__main__":
    run_tasks()
