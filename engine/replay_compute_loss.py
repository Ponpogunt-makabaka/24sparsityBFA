#!/usr/bin/env python3
"""
Replay attack histories from T10/T10_Enhanced results and compute initial/final loss.
Also usable as a generic utility for any method's results.json.
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.factory import create_resnet20
from train.train_utils import get_cifar10_loaders
from train.ptq_convert import Int8QuantizedConv2d, Int8QuantizedResNet


def load_int8_model(ckpt_path: str, device: str) -> nn.Module:
    """Load INT8 quantized sparse model from checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    base_model = create_resnet20(sparsity_type="2:4", pretrained_path=None).to(device)
    base_model.eval()

    for name, module in base_model.named_modules():
        mask_key = f"{name}.sparse_mask"
        if mask_key in state_dict:
            mask_tensor = state_dict[mask_key]
            if hasattr(module, 'register_buffer'):
                if hasattr(module, 'cached_mask'):
                    del module.cached_mask
                module.register_buffer('cached_mask', mask_tensor)

    model = Int8QuantizedResNet(base_model, copy_sparse_masks=True).to(device)
    filtered = {k: v for k, v in state_dict.items() if 'sparse_mask' not in k}
    model.load_state_dict(filtered, strict=False)
    model.calibrate_all_layers()
    model.eval()
    return model


def flatten_groups(tensor: torch.Tensor):
    """Flatten a weight/mask tensor to (num_groups, 4) view."""
    if tensor.dim() == 4:  # Conv: (C_out, C_in, H, W)
        t = tensor.permute(0, 2, 3, 1).contiguous().view(-1, 4)
    elif tensor.dim() == 2:  # Linear: (out, in)
        t = tensor.contiguous().view(-1, 4)
    elif tensor.dim() == 1:  # Already flat
        t = tensor.contiguous().view(-1, 4)
    else:
        return None
    return t


def restore_groups(flat: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """Restore flat groups back to original tensor shape."""
    if original.dim() == 4:
        C_out, C_in, H, W = original.shape
        return flat.view(C_out, H, W, C_in).permute(0, 3, 1, 2).contiguous()
    else:
        return flat.view(original.shape)


def apply_flip(model: nn.Module, layer_name: str, group_idx: int,
               old_pattern: List[int], new_pattern: List[int]) -> bool:
    """Apply a single pattern change to the model."""
    # Find the module
    module = None
    for name, m in model.named_modules():
        if name == layer_name and isinstance(m, Int8QuantizedConv2d):
            module = m
            break
    if module is None:
        print(f"[Warning] Module {layer_name} not found")
        return False

    int8_w = module.int8_weights
    mask = module.sparse_mask

    w_flat = flatten_groups(int8_w)
    m_flat = flatten_groups(mask)
    if w_flat is None or m_flat is None:
        return False

    w_group = w_flat[group_idx]
    m_group = m_flat[group_idx]

    # Read old active values
    old_active = (m_group > 0.5).nonzero(as_tuple=False).flatten()
    if old_active.numel() != 2:
        return False
    old_values = w_group[old_active].clone()

    with torch.no_grad():
        # Zero the group
        w_group.zero_()
        m_group.zero_()
        # Write values to new pattern positions
        for rank, dst_pos in enumerate(new_pattern):
            w_group[dst_pos] = old_values[rank]
            m_group[dst_pos] = 1.0
        # Restore to original tensor shape
        module.sparse_mask.copy_(restore_groups(m_flat, mask))
        module.int8_weights.copy_(restore_groups(w_flat, int8_w))

    return True


def evaluate_acc_loss(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    """Evaluate accuracy and average cross-entropy loss."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
            total_loss += criterion(outputs, targets).item()
            n_batches += 1
    acc = 100.0 * correct / total if total else 0.0
    avg_loss = total_loss / n_batches if n_batches else 0.0
    return acc, avg_loss


def replay_and_evaluate(results_path: str, ckpt_path: str, device: str = "cuda",
                        data_dir: str = "./data") -> Dict:
    """Replay attack history from a results JSON and compute loss at each step."""
    with open(results_path) as f:
        results = json.load(f)

    # Handle list format (T08) vs dict format (T10/T10_Enhanced)
    if isinstance(results, list):
        results = results[0]

    attack_history = results["attack_history"]
    method = results.get("method", os.path.basename(os.path.dirname(results_path)))

    # Load model
    model = load_int8_model(ckpt_path, device)

    # Load test data
    _, test_loader = get_cifar10_loaders(data_dir=data_dir, batch_size=128, num_workers=0)

    # Evaluate initial state
    init_acc, init_loss = evaluate_acc_loss(model, test_loader, device)
    print(f"[{method}] Initial: acc={init_acc:.2f}%, loss={init_loss:.6f}")

    # Replay flips
    for i, flip in enumerate(attack_history):
        layer = flip["layer"]
        group = flip["group"]
        old_pat = list(flip["old_pattern"])
        new_pat = list(flip["new_pattern"])

        success = apply_flip(model, layer, group, old_pat, new_pat)
        if not success:
            print(f"[{method}] Failed to apply flip {i+1}: {layer} g={group}")
            break

    # Evaluate final state
    final_acc, final_loss = evaluate_acc_loss(model, test_loader, device)
    print(f"[{method}] Final:   acc={final_acc:.2f}%, loss={final_loss:.6f}")
    print(f"[{method}] Drop:    acc_drop={init_acc - final_acc:.2f}%, loss_increase={final_loss - init_loss:.6f}")

    return {
        "method": method,
        "initial_acc": init_acc,
        "initial_loss": init_loss,
        "final_acc": final_acc,
        "final_loss": final_loss,
        "acc_drop": init_acc - final_acc,
        "loss_increase": final_loss - init_loss,
        "num_flips": len(attack_history),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="Path to results.json")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-dir", type=str, default="./data")
    args = parser.parse_args()

    result = replay_and_evaluate(args.results, args.ckpt, args.device, args.data_dir)
    print(f"\n{json.dumps(result, indent=2)}")
