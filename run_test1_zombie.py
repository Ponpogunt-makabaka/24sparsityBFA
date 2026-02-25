#!/usr/bin/env python3
"""
Test 1: Zombie Revival Attack (Value + Mask flip)

Simulate a worst-case fault where a pruned weight is revived and its Int8
value MSB is flipped in the same step.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models.factory import create_resnet20
from train.ptq_convert import Int8QuantizedResNet
from train.train_utils import get_cifar10_loaders


@dataclass
class ZombieAttackResult:
    initial_accuracy: float
    final_accuracy: float
    total_flips: int
    accuracy_history: List[float]
    loss_history: List[float]
    zombie_positions: List[Tuple[str, int]]  # (layer_name, weight_idx)


def _flip_int8_msb(int8_val: int) -> int:
    """Flip MSB (bit7) using proper int8 two's complement."""
    u8 = int8_val & 0xFF
    u8 ^= 0x80
    return u8 - 256 if u8 >= 128 else u8


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


def _collect_pruned_indices(model: nn.Module) -> Dict[str, List[int]]:
    """Collect mask==0 indices for each layer."""
    pruned = {}
    for name, module in model.named_modules():
        if hasattr(module, "sparse_mask") and module.sparse_mask is not None:
            if hasattr(module, "int8_weights") and hasattr(module, "scale"):
                mask_flat = module.sparse_mask.flatten()
                zero_idx = (mask_flat < 0.5).nonzero(as_tuple=False).flatten().tolist()
                if zero_idx:
                    pruned[name] = zero_idx
    return pruned


def _layer_importance(model: nn.Module) -> Dict[str, float]:
    """Compute simple magnitude-based importance per layer (L1 of dequantized weights)."""
    importance = {}
    for name, module in model.named_modules():
        if hasattr(module, "int8_weights") and hasattr(module, "scale"):
            w = module.int8_weights.float() * module.scale.item()
            importance[name] = w.abs().sum().item()
    return importance


def run_zombie_attack(max_flips: int = 50, seed: int = 42):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    random.seed(seed)

    _, test_loader = get_cifar10_loaders(batch_size=256)
    os.makedirs("./results", exist_ok=True)

    model = _load_sparse_int8_model(device)
    pruned = _collect_pruned_indices(model)
    importance = _layer_importance(model)

    # Keep only layers that actually have pruned weights
    layers = [name for name in pruned.keys() if name in importance]
    layers.sort(key=lambda n: importance[n], reverse=True)

    initial_acc, initial_loss = _evaluate(model, test_loader, device)
    print(f"[Zombie] Initial accuracy: {initial_acc:.2f}%, loss: {initial_loss:.4f}")
    print(f"[Zombie] {'Flips':>8} | {'Accuracy':>10} | {'Loss':>10} | {'Zombie (layer:idx)'}")
    print(f"[Zombie] {'-'*8}-+-{'-'*11}-+-{'-'*11}-+-{'-'*60}")

    acc_hist = [initial_acc]
    loss_hist = [initial_loss]
    zombie_positions = []

    total_flips = 0

    while total_flips < max_flips and layers:
        # Pick the most "important" layer that still has pruned indices
        layer_name = None
        for ln in layers:
            if pruned.get(ln):
                layer_name = ln
                break
        if layer_name is None:
            break

        module = dict(model.named_modules())[layer_name]
        idx_list = pruned[layer_name]
        idx = random.choice(idx_list)
        idx_list.remove(idx)

        # Zombie step: flip MSB of value + revive mask
        with torch.no_grad():
            int8_flat = module.int8_weights.flatten()
            mask_flat = module.sparse_mask.flatten()
            current_val = int8_flat[idx].item()
            new_val = _flip_int8_msb(current_val)
            int8_flat[idx] = torch.tensor(new_val, dtype=torch.int8)
            mask_flat[idx] = torch.tensor(1.0, dtype=module.sparse_mask.dtype)

        total_flips += 1
        zombie_positions.append((layer_name, idx))

        current_acc, current_loss = _evaluate(model, test_loader, device)
        acc_hist.append(current_acc)
        loss_hist.append(current_loss)

        layer_short = layer_name.split('.')[-1]
        print(f"[Zombie] {total_flips:8d} | {current_acc:9.2f}% | {current_loss:10.4f} | {layer_short}:{idx}")

    final_acc, final_loss = _evaluate(model, test_loader, device)
    print(f"[Zombie] {'-'*8}-+-{'-'*11}-+-{'-'*11}-+-{'-'*60}")
    print(f"[Zombie] Final accuracy: {final_acc:.2f}%, loss: {final_loss:.4f} after {total_flips} flips")

    result = ZombieAttackResult(
        initial_accuracy=initial_acc,
        final_accuracy=final_acc,
        total_flips=total_flips,
        accuracy_history=acc_hist,
        loss_history=loss_hist,
        zombie_positions=zombie_positions
    )

    # Save result
    with open("./results/test1_zombie_result.pkl", "wb") as f:
        pickle.dump(result, f)

    # Save log
    with open("./results/test1_zombie_log.txt", "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("Test 1: Zombie Revival Attack Log\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Initial Accuracy: {initial_acc:.2f}%\n")
        f.write(f"Final Accuracy: {final_acc:.2f}%\n")
        f.write(f"Total Flips: {total_flips}\n\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Flips':>8} | {'Accuracy':>10} | {'Loss':>10} | {'Zombie (layer:idx)'}\n")
        f.write("-" * 100 + "\n")
        for i, (ln, idx) in enumerate(zombie_positions, 1):
            acc = acc_hist[i]
            loss = loss_hist[i]
            f.write(f"{i:8d} | {acc:9.2f}% | {loss:10.4f} | {ln.split('.')[-1]}:{idx}\n")

    # Plot
    plt.figure(figsize=(10, 6))
    flips = list(range(len(acc_hist)))
    plt.plot(flips, acc_hist, marker='o', linewidth=2, markersize=5,
             label='Zombie Revival (Value+Mask)')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('Revived Zombies (Iterations)', fontsize=12, fontweight='bold')
    plt.ylabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Test 1: Zombie Revival Attack', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('./results/test1_zombie.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[Zombie] Plot saved to: ./results/test1_zombie.png")

    return result


if __name__ == "__main__":
    run_zombie_attack()
