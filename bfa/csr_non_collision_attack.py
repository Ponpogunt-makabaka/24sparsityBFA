#!/usr/bin/env python3
"""
CSR Index Non-Collision Attack (2:4 groups).

Targets only the index bits (0/1) within 2:4 groups.
Skips flips that collide with the neighbor non-zero index.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set

import torch
import torch.nn as nn


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
                module.sparse_mask = (module.sparse_mask > 0.5).to(module.sparse_mask.dtype)
                layers.append((name, module))
    return layers


def _evaluate(
    model: nn.Module,
    loader,
    device: str,
    max_samples: Optional[int] = None
) -> Tuple[float, float]:
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
            if max_samples is not None and total >= max_samples:
                break
    acc = 100.0 * correct / total
    avg_loss = total_loss / total
    return acc, avg_loss


def _compute_candidates(
    model: nn.Module,
    calib_loader,
    device: str,
    calib_samples: int,
    flipped: Set[Tuple[str, int, int, int]],
    counters: Dict[str, int],
    max_groups_per_layer: Optional[int] = None,
    allow_nonpositive: bool = False
) -> List[Tuple[float, str, nn.Module, int, int, int]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    calib_data = []
    calib_targets = []
    for inputs, targets in calib_loader:
        calib_data.append(inputs.to(device))
        calib_targets.append(targets.to(device))
        if len(calib_data) * inputs.size(0) >= calib_samples:
            break
    calib_inputs = torch.cat(calib_data, dim=0)[:calib_samples]
    calib_targets = torch.cat(calib_targets, dim=0)[:calib_samples]

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
        if max_groups_per_layer is not None and num_groups > max_groups_per_layer:
            sampled = torch.randint(0, num_groups, (max_groups_per_layer,), device=w_flat.device)
            group_indices = sampled.tolist()
        else:
            group_indices = range(num_groups)

        for g_idx in group_indices:
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
                    if score <= 0 and not allow_nonpositive:
                        continue

                    key = (layer_name, g_idx, old_idx, bit_pos)
                    if key in flipped:
                        continue

                    candidates.append((score, layer_name, module, g_idx, old_idx, new_idx))

        if module.weight.grad is not None:
            module.weight.grad = None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates


def _apply_non_collision_move(
    module: nn.Module,
    g_idx: int,
    old_idx: int,
    new_idx: int
) -> bool:
    int8_w = module.int8_weights
    mask = module.sparse_mask

    w_flat, w_meta = _flatten_groups(int8_w)
    m_flat, m_meta = _flatten_groups(mask)
    if w_flat is None or m_flat is None:
        return False

    old_val = int(w_flat[g_idx, old_idx].item())
    w_flat[g_idx, new_idx] = torch.tensor(old_val, dtype=int8_w.dtype, device=int8_w.device)
    w_flat[g_idx, old_idx] = torch.tensor(0, dtype=int8_w.dtype, device=int8_w.device)

    old_mask = int(m_flat[g_idx, old_idx].item())
    new_mask = int(m_flat[g_idx, new_idx].item())
    m_flat[g_idx, old_idx] = torch.tensor(old_mask ^ 1, dtype=mask.dtype, device=mask.device)
    m_flat[g_idx, new_idx] = torch.tensor(new_mask ^ 1, dtype=mask.dtype, device=mask.device)

    int8_new = _restore_groups(w_flat, w_meta)
    mask_new = _restore_groups(m_flat, m_meta)

    module.int8_weights.copy_(int8_new.clone())
    module.sparse_mask.copy_(mask_new.clone())
    return True


def run_csr_non_collision_attack(
    model: nn.Module,
    test_loader,
    calib_loader,
    device: str,
    max_success: int = 50,
    calib_samples: int = 1000,
    log_interval: int = 1,
    max_groups_per_layer: Optional[int] = None,
    eval_samples: Optional[int] = None,
    allow_nonpositive: bool = False
) -> CSRNonCollisionResult:
    initial_acc, initial_loss = _evaluate(model, test_loader, device, max_samples=eval_samples)

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
        candidates = _compute_candidates(
            model=model,
            calib_loader=calib_loader,
            device=device,
            calib_samples=calib_samples,
            flipped=flipped,
            counters=counters,
            max_groups_per_layer=max_groups_per_layer,
            allow_nonpositive=allow_nonpositive
        )

        if not candidates:
            break

        score, layer_name, module, g_idx, old_idx, new_idx = candidates[0]
        bit_pos = (old_idx ^ new_idx).bit_length() - 1
        flip_key = (layer_name, g_idx, old_idx, bit_pos)
        flipped.add(flip_key)

        ok = _apply_non_collision_move(module, g_idx, old_idx, new_idx)
        if not ok:
            continue

        success += 1
        current_acc, current_loss = _evaluate(model, test_loader, device, max_samples=eval_samples)
        accuracy_history.append(current_acc)
        loss_history.append(current_loss)

        if success % log_interval == 0:
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

    return CSRNonCollisionResult(
        initial_accuracy=initial_acc,
        final_accuracy=final_acc,
        total_flips=success,
        attempted_flips=counters["attempted"],
        collisions_skipped=counters["collisions"],
        accuracy_history=accuracy_history,
        loss_history=loss_history,
        flip_log=flip_log
    )
