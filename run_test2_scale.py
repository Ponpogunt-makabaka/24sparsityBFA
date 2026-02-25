#!/usr/bin/env python3
"""
Test 2: Quantization Scale Sensitivity Exam

Compare mask revival attacks on layers with largest vs smallest scales.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models.factory import create_resnet20
from train.ptq_convert import Int8QuantizedResNet
from train.train_utils import get_cifar10_loaders


@dataclass
class ScaleAttackResult:
    initial_accuracy: float
    final_accuracy: float
    total_flips: int
    accuracy_history: List[float]
    loss_history: List[float]
    mask_positions: List[Tuple[str, int]]
    target_layers: List[str]


def _evaluate(model: nn.Module, loader, device: str) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.0 * correct / total
    avg_loss = total_loss / total
    return acc, avg_loss


def _load_sparse_int8_model(device: str) -> nn.Module:
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


def _get_scales(model: nn.Module) -> List[Tuple[str, float, bool]]:
    """
    Return list of (layer_name, scale, has_mask).
    Only Int8 quantized layers have scale buffers.
    """
    scales = []
    for name, module in model.named_modules():
        if hasattr(module, "scale"):
            has_mask = hasattr(module, "sparse_mask") and module.sparse_mask is not None
            scales.append((name, float(module.scale.item()), has_mask))
    return scales


def _targeted_zero_mask_attack(
    model: nn.Module,
    test_loader,
    target_layers: List[str],
    max_flips: int = 50,
    calib_samples: int = 100
) -> ScaleAttackResult:
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()

    # Build layer map
    module_map = dict(model.named_modules())
    allowed = {ln for ln in target_layers if ln in module_map}

    # Precompute mask indices for allowed layers
    pruned = {}
    for name in allowed:
        module = module_map[name]
        if hasattr(module, "sparse_mask") and module.sparse_mask is not None:
            mask_flat = module.sparse_mask.flatten()
            zero_idx = (mask_flat < 0.5).nonzero(as_tuple=False).flatten().tolist()
            if zero_idx:
                pruned[name] = zero_idx

    acc0, loss0 = _evaluate(model, test_loader, device)
    acc_hist = [acc0]
    loss_hist = [loss0]
    mask_positions = []

    total_flips = 0
    flipped_bits: Set[Tuple[str, int]] = set()

    while total_flips < max_flips and pruned:
        # Compute sensitivities only within allowed layers
        sensitivities = []
        # Calibration batch
        calib_data = []
        calib_targets = []
        for inputs, targets in test_loader:
            calib_data.append(inputs.to(device))
            calib_targets.append(targets.to(device))
            if len(calib_data) * inputs.size(0) >= calib_samples:
                break
        calib_inputs = torch.cat(calib_data, dim=0)[:calib_samples]
        calib_targets = torch.cat(calib_targets, dim=0)[:calib_samples]

        for name in list(pruned.keys()):
            module = module_map[name]
            outputs = model(calib_inputs)
            loss = criterion(outputs, calib_targets)
            model.zero_grad()
            loss.backward()

            if module.weight.grad is None:
                continue

            grad = module.weight.grad.data.flatten()
            int8_flat = module.int8_weights.flatten()
            scale = module.scale.item()
            mask_flat = module.sparse_mask.flatten()

            for idx in pruned[name]:
                key = (name, idx)
                if key in flipped_bits:
                    continue
                if mask_flat[idx].item() >= 0.5:
                    continue
                grad_val = grad[idx].item()
                if abs(grad_val) < 1e-10:
                    continue
                w_val = int8_flat[idx].item() * scale
                if abs(w_val) < 1e-12:
                    continue
                score = grad_val * w_val
                if score > 0:
                    sensitivities.append((score, name, module, idx))

            if module.weight.grad is not None:
                module.weight.grad = None

        if not sensitivities:
            break

        sensitivities.sort(key=lambda x: x[0], reverse=True)
        score, layer_name, module, idx = sensitivities[0]

        # Flip mask 0 -> 1
        with torch.no_grad():
            mask_flat = module.sparse_mask.flatten()
            if mask_flat[idx].item() < 0.5:
                mask_flat[idx] = torch.tensor(1.0, dtype=module.sparse_mask.dtype)
                flipped_bits.add((layer_name, idx))
                total_flips += 1
                mask_positions.append((layer_name, idx))
            else:
                # remove if no longer valid
                pruned[layer_name].remove(idx)
                if not pruned[layer_name]:
                    pruned.pop(layer_name, None)
                continue

        acc, loss = _evaluate(model, test_loader, device)
        acc_hist.append(acc)
        loss_hist.append(loss)

    return ScaleAttackResult(
        initial_accuracy=acc0,
        final_accuracy=acc_hist[-1],
        total_flips=total_flips,
        accuracy_history=acc_hist,
        loss_history=loss_hist,
        mask_positions=mask_positions,
        target_layers=list(target_layers)
    )


def run_test2_scale():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    _, test_loader = get_cifar10_loaders(batch_size=256)
    os.makedirs("./results", exist_ok=True)

    # Load model and gather scales
    model = _load_sparse_int8_model(device)
    scales = _get_scales(model)

    # Only consider layers with masks for targeted attack
    mask_scales = [(n, s) for (n, s, has_mask) in scales if has_mask]
    mask_scales.sort(key=lambda x: x[1], reverse=True)

    top3 = [n for (n, _) in mask_scales[:3]]
    bottom3 = [n for (n, _) in mask_scales[-3:]]

    # Save scale ranking
    with open("./results/test2_scale_layers.txt", "w", encoding="utf-8") as f:
        f.write("Layer scale ranking (masked layers only):\n")
        for n, s in mask_scales:
            f.write(f"{n:40s} scale={s:.6f}\n")
        f.write("\nTop-3 (largest scales):\n")
        for n in top3:
            f.write(f"- {n}\n")
        f.write("\nBottom-3 (smallest scales):\n")
        for n in bottom3:
            f.write(f"- {n}\n")

    print("Top-3 layers (largest scales):", top3)
    print("Bottom-3 layers (smallest scales):", bottom3)

    # Group A: largest scales
    model_a = _load_sparse_int8_model(device)
    result_a = _targeted_zero_mask_attack(
        model_a, test_loader, top3, max_flips=50
    )

    # Group B: smallest scales
    model_b = _load_sparse_int8_model(device)
    result_b = _targeted_zero_mask_attack(
        model_b, test_loader, bottom3, max_flips=50
    )

    # Save results
    with open("./results/test2_scale_groupA_result.pkl", "wb") as f:
        pickle.dump(result_a, f)
    with open("./results/test2_scale_groupB_result.pkl", "wb") as f:
        pickle.dump(result_b, f)

    with open("./results/test2_scale_log.txt", "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("Test 2: Quantization Scale Sensitivity\n")
        f.write("=" * 100 + "\n\n")
        f.write("Group A (Top-3 scales):\n")
        f.write(", ".join(top3) + "\n")
        f.write(f"Initial Acc: {result_a.initial_accuracy:.2f}%\n")
        f.write(f"Final Acc: {result_a.final_accuracy:.2f}%\n")
        f.write(f"Total Flips: {result_a.total_flips}\n\n")
        f.write("Group B (Bottom-3 scales):\n")
        f.write(", ".join(bottom3) + "\n")
        f.write(f"Initial Acc: {result_b.initial_accuracy:.2f}%\n")
        f.write(f"Final Acc: {result_b.final_accuracy:.2f}%\n")
        f.write(f"Total Flips: {result_b.total_flips}\n")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(result_a.accuracy_history)), result_a.accuracy_history,
             marker='o', linewidth=2, markersize=4, label='Group A (Top-3 scales)')
    plt.plot(range(len(result_b.accuracy_history)), result_b.accuracy_history,
             marker='s', linewidth=2, markersize=4, label='Group B (Bottom-3 scales)')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('Mask Revivals (Iterations)', fontsize=12, fontweight='bold')
    plt.ylabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Test 2: Scale Sensitivity (Mask Revival)', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('./results/test2_scale_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[Test2] Plot saved to: ./results/test2_scale_comparison.png")


if __name__ == "__main__":
    run_test2_scale()
