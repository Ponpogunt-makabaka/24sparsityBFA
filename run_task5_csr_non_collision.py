#!/usr/bin/env python3
"""
Task 5: Sparse INT8 CSR Index Attack (Non-Collision)

Operate on 2:4 groups and flip index bits (0/1) of the non-zero positions.
Skip flips that collide with the neighbor non-zero index.
Move the weight value to the new index (dense representation), neighbor unchanged.
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
class CSRNonCollisionResult:
    initial_accuracy: float
    final_accuracy: float
    total_flips: int
    attempted_flips: int
    collisions_skipped: int
    accuracy_history: List[float]
    loss_history: List[float]
    flip_log: List[Dict]


def load_sparse_int8_model(device: str) -> nn.Module:
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


def evaluate(model: nn.Module, loader, device: str) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0
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


def _flatten_groups(t: torch.Tensor):
    if t.dim() == 4:
        t_perm = t.permute(0, 2, 3, 1).contiguous()
        flat = t_perm.view(-1, 4)
        meta = ("conv", t_perm.shape)
        return flat, meta
    if t.dim() == 2:
        t_perm = t.contiguous()
        flat = t_perm.view(-1, 4)
        meta = ("linear", t_perm.shape)
        return flat, meta
    return None, None


def _restore_groups(flat: torch.Tensor, meta):
    kind, shape = meta
    if kind == "conv":
        t_perm = flat.view(shape)
        return t_perm.permute(0, 3, 1, 2).contiguous()
    return flat.view(shape)


def _get_sparse_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    layers = []
    for name, module in model.named_modules():
        if hasattr(module, "int8_weights") and hasattr(module, "scale"):
            if hasattr(module, "sparse_mask") and module.sparse_mask is not None:
                # Enforce binary mask
                module.sparse_mask = (module.sparse_mask > 0.5).to(module.sparse_mask.dtype)
                layers.append((name, module))
    return layers


def compute_candidates(
    model: nn.Module,
    calib_loader,
    device: str,
    calib_samples: int,
    flipped: Set[Tuple[str, int, int, int]],
    counters: Dict[str, int]
) -> List[Tuple[float, str, nn.Module, int, int, int]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Calibration batch
    calib_data = []
    calib_targets = []
    for inputs, targets in calib_loader:
        calib_data.append(inputs.to(device))
        calib_targets.append(targets.to(device))
        if len(calib_data) * inputs.size(0) >= calib_samples:
            break
    calib_inputs = torch.cat(calib_data, dim=0)[:calib_samples]
    calib_targets = torch.cat(calib_targets, dim=0)[:calib_samples]

    # Forward/backward once
    outputs = model(calib_inputs)
    loss = criterion(outputs, calib_targets)
    model.zero_grad()
    loss.backward()

    candidates = []
    layers = _get_sparse_layers(model)

    for layer_name, module in layers:
        if module.weight.grad is None:
            continue

        grad = module.weight.grad.data
        int8_w = module.int8_weights
        mask = module.sparse_mask
        scale = module.scale.item()

        g_flat, _ = _flatten_groups(grad)
        w_flat, _ = _flatten_groups(int8_w)
        m_flat, _ = _flatten_groups(mask)
        if g_flat is None or w_flat is None or m_flat is None:
            continue

        num_groups = w_flat.shape[0]
        for g_idx in range(num_groups):
            m_group = m_flat[g_idx]
            active = (m_group > 0.5).nonzero(as_tuple=False).flatten().tolist()
            if len(active) != 2:
                continue

            a_idx, b_idx = active[0], active[1]
            for old_idx, neighbor_idx in [(a_idx, b_idx), (b_idx, a_idx)]:
                w_val = int(w_flat[g_idx, old_idx].item())
                if w_val == 0:
                    continue
                w_fp = w_val * scale
                g_old = g_flat[g_idx, old_idx].item()

                for bit_pos in (0, 1):
                    counters["attempted"] += 1
                    new_idx = old_idx ^ (1 << bit_pos)
                    if new_idx == neighbor_idx:
                        counters["collisions"] += 1
                        continue
                    if int(m_group[new_idx].item()) == 1:
                        counters["collisions"] += 1
                        continue

                    g_new = g_flat[g_idx, new_idx].item()
                    score = w_fp * (g_new - g_old)
                    if score <= 0:
                        continue

                    key = (layer_name, g_idx, old_idx, bit_pos)
                    if key in flipped:
                        continue

                    candidates.append((score, layer_name, module, g_idx, old_idx, new_idx))

        if module.weight.grad is not None:
            module.weight.grad = None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates


def apply_non_collision_move(
    module: nn.Module,
    g_idx: int,
    old_idx: int,
    new_idx: int
):
    int8_w = module.int8_weights
    mask = module.sparse_mask

    w_flat, w_meta = _flatten_groups(int8_w)
    m_flat, m_meta = _flatten_groups(mask)
    if w_flat is None or m_flat is None:
        return False

    # Move value
    old_val = int(w_flat[g_idx, old_idx].item())
    w_flat[g_idx, new_idx] = torch.tensor(old_val, dtype=int8_w.dtype, device=int8_w.device)
    w_flat[g_idx, old_idx] = torch.tensor(0, dtype=int8_w.dtype, device=int8_w.device)

    # Move mask (bitmask 0/1)
    old_mask = int(m_flat[g_idx, old_idx].item())
    new_mask = int(m_flat[g_idx, new_idx].item())
    m_flat[g_idx, old_idx] = torch.tensor(old_mask ^ 1, dtype=mask.dtype, device=mask.device)
    m_flat[g_idx, new_idx] = torch.tensor(new_mask ^ 1, dtype=mask.dtype, device=mask.device)

    # Restore to original layout
    int8_new = _restore_groups(w_flat, w_meta)
    mask_new = _restore_groups(m_flat, m_meta)

    module.int8_weights.copy_(int8_new.clone())
    module.sparse_mask.copy_(mask_new.clone())
    return True


def run_task5():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    _, test_loader = get_cifar10_loaders(batch_size=256)
    os.makedirs("./results", exist_ok=True)

    print("\n" + "=" * 70)
    print("Task 5: Sparse INT8 CSR Index Attack (Non-Collision)")
    print("=" * 70)

    model = load_sparse_int8_model(device)

    initial_acc, initial_loss = evaluate(model, test_loader, device)
    print(f"[Task5] Initial accuracy: {initial_acc:.2f}%, loss: {initial_loss:.4f}")

    max_success = 50
    calib_samples = 100
    log_interval = 1

    success = 0
    counters = {"attempted": 0, "collisions": 0}
    flipped = set()

    accuracy_history = [initial_acc]
    loss_history = [initial_loss]
    flip_log = [{
        "flips": 0,
        "accuracy": initial_acc,
        "loss": initial_loss,
        "move": None
    }]

    while success < max_success:
        candidates = compute_candidates(
            model=model,
            calib_loader=test_loader,
            device=device,
            calib_samples=calib_samples,
            flipped=flipped,
            counters=counters
        )

        if not candidates:
            print("[Task5] No valid non-collision candidates found.")
            break

        score, layer_name, module, g_idx, old_idx, new_idx = candidates[0]
        bit_pos = (old_idx ^ new_idx).bit_length() - 1
        flip_key = (layer_name, g_idx, old_idx, bit_pos)
        flipped.add(flip_key)

        ok = apply_non_collision_move(module, g_idx, old_idx, new_idx)
        if not ok:
            continue

        success += 1
        current_acc, current_loss = evaluate(model, test_loader, device)
        accuracy_history.append(current_acc)
        loss_history.append(current_loss)

        if success % log_interval == 0:
            layer_short = layer_name.split('.')[-1]
            move_str = f"{layer_short}:g{g_idx}:{old_idx}->{new_idx}"
            print(f"[Task5] {success:8d} | {current_acc:9.2f}% | {current_loss:10.4f} | {move_str}")
            flip_log.append({
                "flips": success,
                "accuracy": current_acc,
                "loss": current_loss,
                "move": {
                    "layer": layer_name,
                    "group": g_idx,
                    "old_idx": old_idx,
                    "new_idx": new_idx,
                    "score": score
                }
            })

    final_acc = accuracy_history[-1]
    final_loss = loss_history[-1]
    print(f"[Task5] Final accuracy: {final_acc:.2f}%, loss: {final_loss:.4f}")

    result = CSRNonCollisionResult(
        initial_accuracy=initial_acc,
        final_accuracy=final_acc,
        total_flips=success,
        attempted_flips=counters["attempted"],
        collisions_skipped=counters["collisions"],
        accuracy_history=accuracy_history,
        loss_history=loss_history,
        flip_log=flip_log
    )

    # Save results
    result_path = "./results/task5_csr_non_collision_result.pkl"
    log_path = "./results/task5_csr_non_collision_log.txt"
    plot_path = "./results/task5_csr_non_collision.png"

    with open(result_path, "wb") as f:
        pickle.dump(result, f)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("Task 5: Sparse INT8 CSR Index Attack (Non-Collision)\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Initial Accuracy: {initial_acc:.2f}%\n")
        f.write(f"Final Accuracy: {final_acc:.2f}%\n")
        f.write(f"Successful Non-Collision Flips Executed: {success}\n")
        f.write(f"Total Flips Attempted: {counters['attempted']}\n")
        f.write(f"Collisions Avoided (Skipped): {counters['collisions']}\n")
        f.write(f"Accuracy Drop: {initial_acc - final_acc:.2f}%\n\n")

        f.write("-" * 100 + "\n")
        f.write(f"{'Flips':>8} | {'Accuracy':>10} | {'Loss':>10} | {'Move (layer:g:old->new)'}\n")
        f.write("-" * 100 + "\n")
        for entry in flip_log:
            flips = entry["flips"]
            acc = entry["accuracy"]
            loss = entry["loss"]
            move = entry["move"]
            if move is None:
                move_str = "(initial state)"
            else:
                layer_short = move["layer"].split(".")[-1]
                move_str = f"{layer_short}:g{move['group']}:{move['old_idx']}->{move['new_idx']}"
            f.write(f"{flips:8d} | {acc:9.2f}% | {loss:10.4f} | {move_str}\n")

    # Plot
    plt.figure(figsize=(10, 6))
    flips = list(range(len(accuracy_history)))
    plt.plot(flips, accuracy_history, marker='o', linewidth=2, markersize=5,
             label='CSR Non-Collision Index Attack')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('Successful Flips', fontsize=12, fontweight='bold')
    plt.ylabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Task 5: CSR Index Attack (Non-Collision)', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[Task5] Results saved to {result_path}")
    print(f"[Task5] Detailed log saved to {log_path}")
    print(f"[Task5] Plot saved to {plot_path}")


if __name__ == "__main__":
    run_task5()
