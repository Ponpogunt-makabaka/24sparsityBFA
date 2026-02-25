#!/usr/bin/env python3
"""
Step 2: One-Step Metadata Sanity Test

验证修改 metadata 是否真的影响 logits/loss。

测试流程:
1. 加载模型和固定 mini-batch
2. 记录 logits0, loss0
3. 选择一个特定 layer + group，强制修改 pattern
4. 记录 logits1, loss1
5. 输出差分指标

判据:
- 若 logits diff ~ 0 且 loss_delta ~ 0: 说明 metadata 不进 forward
- 若 diff 显著: 说明 metadata 会影响 forward
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import torch
import torch.nn as nn

from models.factory import create_resnet20
from scripts.p012_17_utils import load_cifar10_loaders_offline, set_all_seeds
from train.ptq_convert import Int8QuantizedResNet


def flatten_groups(t: torch.Tensor):
    """Flatten tensor into groups of 4 for 2:4 sparsity."""
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


def restore_groups(flat: torch.Tensor, meta):
    """Restore original tensor shape from flattened groups."""
    kind, shape = meta
    if kind == "conv":
        t_perm = flat.view(shape)
        return t_perm.permute(0, 3, 1, 2).contiguous()
    return flat.view(shape)


def get_current_pattern(mask_group: torch.Tensor):
    """Extract the current 2-of-4 pattern from a mask group."""
    active = (mask_group > 0.5).nonzero(as_tuple=False).flatten().tolist()
    if len(active) != 2:
        return None
    a, b = int(active[0]), int(active[1])
    if a > b:
        a, b = b, a
    return (a, b)


def pattern_to_mask(pattern: tuple, device: torch.device) -> torch.Tensor:
    """Convert a pattern tuple to a 4-element mask tensor."""
    mask = torch.zeros(4, dtype=torch.float32, device=device)
    mask[list(pattern)] = 1.0
    return mask


def compute_dense_reconstruction(int8_w_flat, m_flat, scale, g_idx):
    """Compute the dense reconstruction w̃_g for a single group."""
    w_group = int8_w_flat[g_idx]
    m_group = m_flat[g_idx]
    w_tilde = w_group.float() * scale * m_group
    return w_tilde


def hash_tensor(t: torch.Tensor) -> str:
    """Compute a simple hash of a tensor for verification."""
    return hex(hash((t.flatten()[:100].cpu().numpy().tobytes())))


def main():
    device = "cpu"
    seed = 0
    ckpt_path = "results/task28_sparse_mask_fixed_finetune_int8_ckpt.pth"

    # Set seeds
    set_all_seeds(seed)

    # Create output directory
    os.makedirs("results/debug_task1xx", exist_ok=True)

    # Log file
    log_path = "results/debug_task1xx/sanity_one_step_log.txt"
    result_path = "results/debug_task1xx/sanity_one_step_result.json"

    with open(log_path, "w") as log:
        log.write("=" * 80 + "\n")
        log.write("Step 2: One-Step Metadata Sanity Test\n")
        log.write("=" * 80 + "\n")
        log.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Device: {device}\n")
        log.write(f"Checkpoint: {ckpt_path}\n")
        log.write(f"Seed: {seed}\n\n")

        # Load model
        log.write("[1] Loading model...\n")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Check if Int8 checkpoint
        is_int8 = any("int8_weights" in k for k in state_dict.keys())
        log.write(f"    Is Int8 checkpoint: {is_int8}\n")

        base_model = create_resnet20(sparsity_type="2:4", pretrained_path=None).to(device)

        # Load sparse_mask into base_model
        for name, module in base_model.named_modules():
            mask_key = f"{name}.sparse_mask"
            if mask_key in state_dict:
                mask_tensor = state_dict[mask_key]
                if hasattr(module, 'register_buffer'):
                    if hasattr(module, 'cached_mask'):
                        del module.cached_mask
                    module.register_buffer('cached_mask', mask_tensor)

        model = Int8QuantizedResNet(base_model, copy_sparse_masks=True).to(device)
        filtered_state_dict = {k: v for k, v in state_dict.items() if 'sparse_mask' not in k}
        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()

        # Get fixed mini-batch
        log.write("[2] Getting fixed mini-batch...\n")
        _, test_loader = load_cifar10_loaders_offline(batch_size=256, data_dir="./data", num_workers=0)

        # Get a fixed batch
        for inputs, targets in test_loader:
            probe_inputs = inputs[:32].to(device)
            probe_targets = targets[:32].to(device)
            break

        log.write(f"    Probe batch shape: {probe_inputs.shape}\n")

        # Compute baseline
        log.write("[3] Computing baseline logits/loss...\n")
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            logits0 = model(probe_inputs)
            loss0 = criterion(logits0, probe_targets).item()

        log.write(f"    logits0 shape: {logits0.shape}\n")
        log.write(f"    loss0: {loss0:.6f}\n")

        # Find a good target group (with large weights)
        log.write("[4] Finding a target group with large weights...\n")
        best_group = None
        best_magnitude = -1

        for name, module in model.named_modules():
            if hasattr(module, "int8_weights") and hasattr(module, "sparse_mask"):
                if module.sparse_mask is not None:
                    int8_w = module.int8_weights
                    mask = module.sparse_mask
                    scale = module.scale.item()

                    w_flat, _ = flatten_groups(int8_w)
                    m_flat, _ = flatten_groups(mask)

                    if w_flat is None or m_flat is None:
                        continue

                    for g_idx in range(w_flat.shape[0]):
                        m_group = m_flat[g_idx]
                        pattern = get_current_pattern(m_group)

                        if pattern is None:
                            continue

                        # Compute magnitude of dense reconstruction
                        w_tilde = compute_dense_reconstruction(w_flat, m_flat, scale, g_idx)
                        magnitude = w_tilde.abs().sum().item()

                        if magnitude > best_magnitude:
                            best_magnitude = magnitude
                            best_group = (name, module, g_idx, pattern, w_tilde.clone())

        if best_group is None:
            log.write("    ERROR: No valid group found!\n")
            return

        layer_name, target_module, target_g_idx, old_pattern, w_tilde_before = best_group
        log.write(f"    Target: {layer_name}, group={target_g_idx}\n")
        log.write(f"    Old pattern: {old_pattern}\n")
        log.write(f"    w_tilde_before: {w_tilde_before.tolist()}\n")
        log.write(f"    Magnitude: {best_magnitude:.6f}\n")

        # Hash metadata before
        mask_hash_before = hash_tensor(target_module.sparse_mask)
        weights_hash_before = hash_tensor(target_module.int8_weights)
        log.write(f"    sparse_mask hash before: {mask_hash_before}\n")
        log.write(f"    int8_weights hash before: {weights_hash_before}\n")

        # Choose a new pattern (different from old)
        all_patterns = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        candidate_patterns = [p for p in all_patterns if p != old_pattern]
        new_pattern = candidate_patterns[0]  # Just pick the first different one
        log.write(f"    New pattern: {new_pattern}\n")

        # Apply pattern change manually
        log.write("[5] Applying pattern change...\n")

        mask = target_module.sparse_mask
        int8_w = target_module.int8_weights

        m_flat, m_meta = flatten_groups(mask)
        w_flat, w_meta = flatten_groups(int8_w)

        # Store old values
        old_mask_group = m_flat[target_g_idx].clone()
        old_active_indices = (old_mask_group > 0.5).nonzero().flatten()
        old_values = w_flat[target_g_idx, old_active_indices].clone()

        # Apply new pattern
        w_flat[target_g_idx, :] = 0
        m_flat[target_g_idx, :] = 0

        for i, pos in enumerate(new_pattern):
            if i < len(old_values):
                w_flat[target_g_idx, pos] = old_values[i]
                m_flat[target_g_idx, pos] = 1

        # Restore
        m_new = restore_groups(m_flat, m_meta)
        w_new = restore_groups(w_flat, w_meta)

        target_module.sparse_mask.copy_(m_new.clone())
        target_module.int8_weights.copy_(w_new.clone())

        # Verify change
        w_tilde_after = compute_dense_reconstruction(
            target_module.int8_weights,
            target_module.sparse_mask,
            target_module.scale.item(),
            target_g_idx
        )

        mask_hash_after = hash_tensor(target_module.sparse_mask)
        weights_hash_after = hash_tensor(target_module.int8_weights)

        log.write(f"    sparse_mask hash after: {mask_hash_after}\n")
        log.write(f"    int8_weights hash after: {weights_hash_after}\n")
        log.write(f"    w_tilde_after: {w_tilde_after.tolist()}\n")
        log.write(f"    Hashes changed: {mask_hash_before != mask_hash_after}\n")

        # Compute new logits/loss
        log.write("[6] Computing new logits/loss...\n")

        model.eval()
        with torch.no_grad():
            logits1 = model(probe_inputs)
            loss1 = criterion(logits1, probe_targets).item()

        log.write(f"    loss1: {loss1:.6f}\n")

        # Compute differences
        logits_diff = logits1 - logits0
        loss_delta = loss1 - loss0

        max_abs_diff = logits_diff.abs().max().item()
        l2_norm_diff = torch.norm(logits_diff).item()
        mean_abs_diff = logits_diff.abs().mean().item()

        log.write("\n[7] Results:\n")
        log.write(f"    loss_delta: {loss_delta:.6f}\n")
        log.write(f"    max_abs_diff(logits): {max_abs_diff:.8f}\n")
        log.write(f"    l2_norm_diff(logits): {l2_norm_diff:.8f}\n")
        log.write(f"    mean_abs_diff(logits): {mean_abs_diff:.8f}\n")

        # Also compute accuracy change
        _, pred0 = logits0.max(1)
        _, pred1 = logits1.max(1)
        acc0 = (pred0 == probe_targets).float().mean().item() * 100
        acc1 = (pred1 == probe_targets).float().mean().item() * 100

        log.write(f"    acc0: {acc0:.2f}%\n")
        log.write(f"    acc1: {acc1:.2f}%\n")
        log.write(f"    acc_delta: {acc1 - acc0:.2f}%\n")

        # Prediction changes
        pred_changed = (pred0 != pred1).sum().item()
        log.write(f"    predictions_changed: {pred_changed}/{len(probe_targets)}\n")

        # Criterion
        log.write("\n[8] Criterion:\n")
        if max_abs_diff < 1e-6 and abs(loss_delta) < 1e-6:
            log.write("    RESULT: logits diff ~ 0 and loss_delta ~ 0\n")
            log.write("    CONCLUSION: metadata does NOT affect forward pass\n")
            log.write("    → CHECK: Are we modifying the right object?\n")
        elif max_abs_diff > 0.01:
            log.write(f"    RESULT: max_abs_diff = {max_abs_diff:.6f} (> 0.01)\n")
            log.write("    CONCLUSION: metadata DOES affect forward pass\n")
            log.write("    → Task1xx search/application may have bugs\n")
        else:
            log.write(f"    RESULT: small but non-zero diff ({max_abs_diff:.8f})\n")
            log.write("    CONCLUSION: metadata affects forward, but weakly\n")

        # Save JSON result
        result = {
            "timestamp": datetime.now().isoformat(),
            "ckpt_path": ckpt_path,
            "seed": seed,
            "target": {
                "layer": layer_name,
                "group": int(target_g_idx),
                "old_pattern": list(old_pattern),
                "new_pattern": list(new_pattern),
                "w_tilde_before": w_tilde_before.tolist(),
                "w_tilde_after": w_tilde_after.tolist(),
            },
            "hashes": {
                "mask_before": mask_hash_before,
                "mask_after": mask_hash_after,
                "weights_before": weights_hash_before,
                "weights_after": weights_hash_after,
            },
            "metrics": {
                "loss0": float(loss0),
                "loss1": float(loss1),
                "loss_delta": float(loss_delta),
                "max_abs_diff_logits": float(max_abs_diff),
                "l2_norm_diff_logits": float(l2_norm_diff),
                "mean_abs_diff_logits": float(mean_abs_diff),
                "acc0": float(acc0),
                "acc1": float(acc1),
                "acc_delta": float(acc1 - acc0),
                "predictions_changed": int(pred_changed),
            },
            "criterion": {
                "metadata_affects_forward": max_abs_diff > 0.01 or abs(loss_delta) > 0.001,
                "reason": "large logits diff or loss change detected" if max_abs_diff > 0.01 else "small diff detected"
            }
        }

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        log.write(f"\n[9] Saved result to {result_path}\n")
        log.write("=" * 80 + "\n")

    print(f"[Done] See {log_path} for details")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
