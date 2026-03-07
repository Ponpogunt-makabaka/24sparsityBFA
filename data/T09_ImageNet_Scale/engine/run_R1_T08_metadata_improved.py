#!/usr/bin/env python3
"""
R1_T08 (T09-adapted): Improved Metadata BFA for sparse models.

Supports:
- ResNet-18
- MobileNet-V2
- DeiT-Tiny (torchvision VisionTransformer config)
- Datasets: Imagenette / CIFAR-100

Core constraints:
- 1-bit index-encoding transitions only
- Non-collision pattern transitions only
- Conv grouping uses T08 semantics:
    w.permute(0, 2, 3, 1).contiguous().view(-1, 4)
- Linear/QKV grouping uses:
    w.view(-1, 4)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, resnet18
from torchvision.models.vision_transformer import VisionTransformer


# Ensure repo root is importable when script is launched from arbitrary cwd.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


IMAGENETTE_WNID_TO_IMAGENET_INDEX = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}


class InputResizeWrapper(nn.Module):
    def __init__(self, model: nn.Module, target_size: int):
        super().__init__()
        self.model = model
        self.target_size = int(target_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.target_size or x.shape[-2] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        return self.model(x)


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_index_to_4bit(i: int, j: int) -> int:
    return (j << 2) | i


def decode_4bit_to_index(code: int) -> Tuple[int, int]:
    return (code & 0x3), ((code >> 2) & 0x3)


def pattern_to_code(pattern: Tuple[int, int]) -> int:
    return encode_index_to_4bit(pattern[0], pattern[1])


def code_to_pattern(code: int) -> Optional[Tuple[int, int]]:
    i, j = decode_4bit_to_index(code)
    if i == j:
        return None
    return (min(i, j), max(i, j))


def flatten_groups(tensor: torch.Tensor, group_size: int = 4) -> Tuple[Optional[torch.Tensor], Optional[Tuple]]:
    if tensor.dim() == 4:
        t_perm = tensor.permute(0, 2, 3, 1).contiguous()
        if t_perm.numel() % group_size != 0:
            return None, None
        flat = t_perm.view(-1, group_size)
        meta = ("conv", tuple(tensor.shape), tuple(t_perm.shape))
        return flat, meta
    if tensor.dim() == 2:
        if tensor.numel() % group_size != 0:
            return None, None
        flat = tensor.contiguous().view(-1, group_size)
        meta = ("linear", tuple(tensor.shape))
        return flat, meta
    return None, None


def restore_groups(flat: torch.Tensor, meta: Tuple) -> torch.Tensor:
    if meta[0] == "conv":
        _, _, perm_shape = meta
        t_perm = flat.view(perm_shape)
        return t_perm.permute(0, 3, 1, 2).contiguous()
    _, orig_shape = meta
    return flat.view(orig_shape)


def build_imagenette_target_transform(classes) -> callable:
    unknown = [c for c in classes if c not in IMAGENETTE_WNID_TO_IMAGENET_INDEX]
    if unknown:
        raise ValueError(f"Unknown Imagenette classes: {unknown}")
    mapping = [IMAGENETTE_WNID_TO_IMAGENET_INDEX[c] for c in classes]

    def _map_target(y: int) -> int:
        return mapping[y]

    return _map_target


def build_balanced_subset_indices(targets: List[int], num_samples: int, seed: int = 0) -> List[int]:
    rng = random.Random(seed)
    class_to_indices: Dict[int, List[int]] = {}
    for idx, target in enumerate(targets):
        class_to_indices.setdefault(int(target), []).append(idx)

    classes = sorted(class_to_indices.keys())
    for cls in classes:
        rng.shuffle(class_to_indices[cls])

    selected: List[int] = []
    while len(selected) < num_samples:
        made_progress = False
        for cls in classes:
            if class_to_indices[cls]:
                selected.append(class_to_indices[cls].pop())
                made_progress = True
                if len(selected) >= num_samples:
                    break
        if not made_progress:
            break

    rng.shuffle(selected)
    return selected


def materialize_calibration_batches(
    loader: DataLoader,
    device: str,
    max_samples: int,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int]:
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    total = 0
    for images, targets in loader:
        if total >= max_samples:
            break
        remaining = max_samples - total
        if images.size(0) > remaining:
            images = images[:remaining]
            targets = targets[:remaining]
        batches.append((images.to(device), targets.to(device)))
        total += int(targets.size(0))
    if total == 0:
        raise RuntimeError("Calibration loader produced zero samples.")
    return batches, total


def evaluate_loss_on_batches(
    model: nn.Module,
    calib_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
) -> float:
    model.eval()
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in calib_batches:
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += float(loss.item()) * int(targets.size(0))
            total += int(targets.size(0))
    return total_loss / max(total, 1)


def compute_loss_and_gradients(
    model: nn.Module,
    calib_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
) -> float:
    model.eval()
    model.zero_grad(set_to_none=True)
    total = sum(int(targets.size(0)) for _, targets in calib_batches)
    total_loss = 0.0

    for images, targets in calib_batches:
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += float(loss.item()) * int(targets.size(0))
        scaled_loss = loss * (float(targets.size(0)) / float(max(total, 1)))
        scaled_loss.backward()

    return total_loss / max(total, 1)


def build_imagenette_loaders(
    dataset_root: Path,
    batch_size: int,
    calib_samples: int,
) -> Tuple[DataLoader, DataLoader]:
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(f"Imagenette dataset root missing train/val: {dataset_root}")

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_set = datasets.ImageFolder(str(val_dir), transform=val_transform)
    val_set.target_transform = build_imagenette_target_transform(val_set.classes)

    test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    raw_targets = [target for _, target in val_set.samples]
    calib_indices = build_balanced_subset_indices(raw_targets, min(calib_samples, len(val_set)), seed=0)
    calib_subset = Subset(val_set, calib_indices)
    calib_loader = DataLoader(calib_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader, calib_loader


def build_cifar100_loaders(
    dataset_root: Path,
    batch_size: int,
    calib_samples: int,
    arch: str = "",
) -> Tuple[DataLoader, DataLoader]:
    dataset_root.mkdir(parents=True, exist_ok=True)
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    resize_224 = arch == "deit_tiny"

    test_transforms_list = []
    if resize_224:
        test_transforms_list.append(transforms.Resize(224))
    test_transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose(test_transforms_list)

    test_set = datasets.CIFAR100(
        root=str(dataset_root),
        train=False,
        download=True,
        transform=test_transform,
    )

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    calib_indices = build_balanced_subset_indices(list(test_set.targets), min(calib_samples, len(test_set)), seed=0)
    calib_subset = Subset(test_set, calib_indices)
    calib_loader = DataLoader(calib_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader, calib_loader


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _strip_module_prefix(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    if all(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def create_model(arch: str, dataset: str, device: str) -> nn.Module:
    if dataset == "imagenette":
        num_classes = 1000
        image_size = 224
        deit_patch_size = 16
    elif dataset == "cifar100":
        num_classes = 100
        # Route A: DeiT uses patch16+224 even for CIFAR-100 (images resized in loader)
        image_size = 224 if arch == "deit_tiny" else 32
        deit_patch_size = 16 if arch == "deit_tiny" else 4
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if arch == "resnet18":
        model = resnet18(weights=None, num_classes=num_classes)
        if dataset == "cifar100":
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        return model.to(device)
    if arch == "mobilenet_v2":
        model = mobilenet_v2(weights=None, num_classes=num_classes)
        if dataset == "cifar100":
            model.features[0][0].stride = (1, 1)
            model.features[2].conv[1][0].stride = (1, 1)
        return model.to(device)
    if arch == "deit_tiny":
        return VisionTransformer(
            image_size=image_size,
            patch_size=deit_patch_size,
            num_layers=12,
            num_heads=3,
            hidden_dim=192,
            mlp_dim=768,
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=num_classes,
        ).to(device)
    raise ValueError(f"Unsupported arch: {arch}")


def load_sparse_model(arch: str, ckpt_path: str, dataset: str, device: str) -> Tuple[nn.Module, dict]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint does not contain a valid state_dict")

    model = create_model(arch, dataset, device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # strict=False for robustness on metadata fields, but fail on gross mismatch.
    if len(missing) > 30 or len(unexpected) > 30:
        raise RuntimeError(
            f"Model load mismatch too large. missing={len(missing)} unexpected={len(unexpected)}"
        )
    model.eval()
    return model, state_dict


def evaluate_top1(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images).argmax(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return 100.0 * correct / max(total, 1)


def get_attack_params(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    params = []
    for name, p in model.named_parameters():
        if p.dim() not in (2, 4):
            continue
        if p.numel() % 4 != 0:
            continue
        params.append((name, p))
    return params


def get_current_pattern_from_mask(mask_group: torch.Tensor) -> Optional[Tuple[int, int]]:
    idx = (mask_group > 0.5).nonzero(as_tuple=False).flatten()
    if idx.numel() != 2:
        return None
    return (int(idx[0].item()), int(idx[1].item()))


def compute_model_hash(model: nn.Module) -> int:
    h = 0
    for n, p in sorted(model.named_parameters()):
        h ^= hash((n, tuple(p.shape), float(p.data.sum().item()), float(p.data.abs().sum().item())))
    return h


@dataclass
class GroupCandidate:
    proxy_score: float
    param_name: str
    group_idx: int
    old_code: int
    new_code: int
    flipped_bit: int
    old_pattern: Tuple[int, int]
    new_pattern: Tuple[int, int]
    delta_w_tilde: torch.Tensor


def default_coarse_groups_for_arch(arch: str) -> int:
    if arch == "deit_tiny":
        return 3000
    return 1000


def select_top_groups_by_grad(
    model: nn.Module,
    params: List[Tuple[str, nn.Parameter]],
    exclude_groups: Set[Tuple[str, int]],
    coarse_groups: int,
) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
    counters = {
        "total_groups": 0,
        "eligible_groups": 0,
        "selected_groups": 0,
        "excluded_groups": 0,
        "candidates_no_gradient": 0,
    }
    selected_with_scores: List[Tuple[float, str, int]] = []

    for param_name, p in params:
        if p.grad is None:
            counters["candidates_no_gradient"] += 1
            continue

        g_flat, _ = flatten_groups(p.grad.data)
        w_flat, _ = flatten_groups(p.data)
        m_flat, _ = flatten_groups((p.data != 0).to(torch.float32))
        if g_flat is None or w_flat is None or m_flat is None:
            continue

        num_groups = int(g_flat.shape[0])
        counters["total_groups"] += num_groups
        grad_scores = g_flat.float().abs().sum(dim=1)
        valid_mask = m_flat.sum(dim=1).eq(2)
        exclude_mask = torch.zeros(num_groups, dtype=torch.bool, device=grad_scores.device)
        for group_name, group_idx in exclude_groups:
            if group_name == param_name and 0 <= group_idx < num_groups:
                exclude_mask[group_idx] = True

        counters["excluded_groups"] += int(exclude_mask.sum().item())
        eligible_mask = valid_mask & (~exclude_mask)
        valid_count = int(eligible_mask.sum().item())
        counters["eligible_groups"] += valid_count
        if valid_count == 0:
            continue

        local_keep = min(max(int(coarse_groups), 1), valid_count)
        local_scores = grad_scores.masked_fill(~eligible_mask, float("-inf"))
        top_idx = torch.topk(local_scores, k=local_keep, largest=True).indices.tolist()
        for g_idx in top_idx:
            selected_with_scores.append((float(grad_scores[g_idx].item()), param_name, int(g_idx)))

    selected_with_scores.sort(key=lambda x: (-x[0], x[1], x[2]))
    top_selected = selected_with_scores[:max(int(coarse_groups), 1)]
    counters["selected_groups"] = len(top_selected)
    return [(param_name, group_idx) for _, param_name, group_idx in top_selected], counters


def generate_candidates_for_selected_groups(
    model: nn.Module,
    params: List[Tuple[str, nn.Parameter]],
    selected_groups: List[Tuple[str, int]],
    forbidden_transitions: Set[Tuple[str, int, int, int]],
) -> Tuple[List[GroupCandidate], Dict[str, int]]:
    counters = {
        "valid_groups": 0,
        "candidates_considered": 0,
        "candidates_valid": 0,
        "candidates_rejected_collision": 0,
        "candidates_rejected_no_change": 0,
        "candidates_skipped_forbidden": 0,
    }
    pooled: List[GroupCandidate] = []
    selected_map: Dict[str, Set[int]] = {}
    for param_name, group_idx in selected_groups:
        selected_map.setdefault(param_name, set()).add(group_idx)

    for param_name, p in params:
        group_indices = selected_map.get(param_name)
        if not group_indices or p.grad is None:
            continue

        grad = p.grad.data
        weight = p.data
        mask = (weight != 0).to(torch.float32)

        g_flat, _ = flatten_groups(grad)
        w_flat, _ = flatten_groups(weight)
        m_flat, _ = flatten_groups(mask)
        if g_flat is None or w_flat is None or m_flat is None:
            continue

        for g_idx in sorted(group_indices):
            m_group = m_flat[g_idx]
            current_pattern = get_current_pattern_from_mask(m_group)
            if current_pattern is None:
                continue
            counters["valid_groups"] += 1
            current_code = pattern_to_code(current_pattern)

            grad_group = g_flat[g_idx].float()
            w_group = w_flat[g_idx]
            old_mask = (m_group > 0.5).to(torch.float32)
            w_tilde_current = w_group.float() * old_mask
            old_active = (old_mask > 0.5).nonzero(as_tuple=False).flatten()
            if old_active.numel() != 2:
                continue
            old_values = w_group[old_active].clone()

            for bit_pos in range(4):
                candidate_code = current_code ^ (1 << bit_pos)
                trans_key = (param_name, g_idx, current_code, candidate_code)
                if trans_key in forbidden_transitions:
                    counters["candidates_skipped_forbidden"] += 1
                    continue

                counters["candidates_considered"] += 1
                candidate_pattern = code_to_pattern(candidate_code)
                if candidate_pattern is None:
                    counters["candidates_rejected_collision"] += 1
                    continue
                if candidate_pattern == current_pattern:
                    counters["candidates_rejected_no_change"] += 1
                    continue

                w_new_group = torch.zeros_like(w_group)
                for rank, dst in enumerate(candidate_pattern):
                    w_new_group[dst] = old_values[rank]
                delta = w_new_group.float() - w_tilde_current
                proxy = float(torch.dot(grad_group, delta).item())
                counters["candidates_valid"] += 1
                pooled.append(
                    GroupCandidate(
                        proxy_score=proxy,
                        param_name=param_name,
                        group_idx=g_idx,
                        old_code=current_code,
                        new_code=candidate_code,
                        flipped_bit=bit_pos,
                        old_pattern=current_pattern,
                        new_pattern=candidate_pattern,
                        delta_w_tilde=delta.detach().cpu(),
                    )
                )

    pooled.sort(key=lambda c: (-c.proxy_score, c.param_name, c.group_idx, c.new_code, c.flipped_bit))
    return pooled, counters


def _apply_group_pattern_to_param(param: torch.Tensor, group_idx: int, new_pattern: Tuple[int, int]) -> bool:
    w_flat, w_meta = flatten_groups(param.data)
    if w_flat is None or w_meta is None:
        return False
    w_group = w_flat[group_idx]
    old_mask = (w_group != 0)
    old_active = old_mask.nonzero(as_tuple=False).flatten()
    if old_active.numel() != 2:
        return False
    old_values = w_group[old_active].clone()
    w_group.zero_()
    for rank, dst in enumerate(new_pattern):
        w_group[dst] = old_values[rank]
    restored = restore_groups(w_flat, w_meta)
    param.data.copy_(restored.clone())
    return True


def exact_verification_topk(
    model: nn.Module,
    param_map: Dict[str, nn.Parameter],
    candidates: List[GroupCandidate],
    verify_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    baseline_loss: float,
    top_k: int,
) -> Tuple[Optional[GroupCandidate], Dict[str, int]]:
    counters = {
        "candidates_tested": 0,
        "candidates_positive_delta": 0,
        "candidates_negative_delta": 0,
        "hash_mismatches": 0,
    }

    best = None
    best_delta = 0.0

    for cand in candidates[:top_k]:
        counters["candidates_tested"] += 1
        if cand.param_name not in param_map:
            continue
        p = param_map[cand.param_name]
        p_backup = p.data.clone()

        with torch.no_grad():
            ok = _apply_group_pattern_to_param(p, cand.group_idx, cand.new_pattern)
        if not ok:
            p.data.copy_(p_backup)
            continue

        new_loss = evaluate_loss_on_batches(model, verify_batches, criterion)
        delta = new_loss - baseline_loss

        with torch.no_grad():
            p.data.copy_(p_backup)

        if delta > 0:
            counters["candidates_positive_delta"] += 1
            if delta > best_delta:
                best_delta = delta
                best = cand
        else:
            counters["candidates_negative_delta"] += 1

    return best, counters


def apply_pattern_change(model: nn.Module, param_map: Dict[str, nn.Parameter], candidate: GroupCandidate) -> bool:
    if candidate.param_name not in param_map:
        return False
    p = param_map[candidate.param_name]
    with torch.no_grad():
        return _apply_group_pattern_to_param(p, candidate.group_idx, candidate.new_pattern)


def reload_model_for_seed(arch: str, ckpt_path: str, dataset: str, device: str) -> nn.Module:
    model, _ = load_sparse_model(arch, ckpt_path, dataset, device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="R1_T08 NCSA attack for T09 sparse large models")
    parser.add_argument("--arch", type=str, required=True, choices=["resnet18", "mobilenet_v2", "deit_tiny"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["imagenette", "cifar100"], default="imagenette")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--physical-budget", type=int, default=50)
    parser.add_argument("--calib-samples", type=int, default=256)
    parser.add_argument("--attack-batch-size", type=int, default=64)
    parser.add_argument(
        "--coarse-groups",
        type=int,
        default=0,
        help="Global top-N groups kept by Stage-A coarse filtering. 0 means arch-specific default.",
    )
    parser.add_argument("--top-m-per-group", type=int, default=3)
    parser.add_argument("--top-k-verify", type=int, default=64)
    parser.add_argument("--topk", type=int, default=None, help="Alias of --top-k-verify")
    parser.add_argument(
        "--max-groups-per-layer",
        type=int,
        default=0,
        help="<=0 means full per-layer search; positive values cap layer search space.",
    )
    parser.add_argument(
        "--group-sampling",
        type=str,
        choices=["top_grad", "random"],
        default="top_grad",
        help="How to pick groups when max-groups-per-layer truncates a layer.",
    )
    parser.add_argument(
        "--min-sampled-group-ratio",
        type=float,
        default=0.05,
        help="Fail if sampled_groups / total_groups falls below this threshold.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    if args.topk is not None:
        args.top_k_verify = args.topk

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA 不可用，回退到 CPU")
        device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_root = Path(args.dataset_root) if args.dataset_root else (
        Path("data/imagenette_full") if args.dataset == "imagenette" else Path("data/cifar100")
    )

    print(f"\n[{now_ts()}] 加载模型: arch={args.arch} ckpt={args.ckpt}")
    model, _ = load_sparse_model(args.arch, args.ckpt, args.dataset, device)
    params = get_attack_params(model)
    print(f"[{now_ts()}] 可攻击参数张量数: {len(params)}")
    print(f"[{now_ts()}] dataset={args.dataset} dataset_root={dataset_root}")

    if args.dataset == "imagenette":
        test_loader, calib_loader = build_imagenette_loaders(
            dataset_root=dataset_root,
            batch_size=args.attack_batch_size,
            calib_samples=args.calib_samples,
        )
    else:
        test_loader, calib_loader = build_cifar100_loaders(
            dataset_root=dataset_root,
            batch_size=args.attack_batch_size,
            calib_samples=args.calib_samples,
            arch=args.arch,
        )
    calib_batches, calib_count = materialize_calibration_batches(
        calib_loader,
        device=device,
        max_samples=args.calib_samples,
    )
    criterion = nn.CrossEntropyLoss()
    coarse_groups = args.coarse_groups if args.coarse_groups > 0 else default_coarse_groups_for_arch(args.arch)
    print(
        f"[{now_ts()}] Calibration batches ready: samples={calib_count} "
        f"batch_size={args.attack_batch_size} coarse_groups={coarse_groups} "
        f"top_k_verify={args.top_k_verify}"
    )

    seeds = args.seed if isinstance(args.seed, list) else [args.seed]
    all_results = []

    for seed in seeds:
        print(f"\n{'='*72}\n[{now_ts()}] Seed={seed} 开始攻击\n{'='*72}")
        set_all_seeds(seed)
        rng = random.Random(seed)

        model = reload_model_for_seed(args.arch, args.ckpt, args.dataset, device)
        params = get_attack_params(model)
        param_map = dict(model.named_parameters())
        init_acc = evaluate_top1(model, test_loader, device)
        print(f"[{now_ts()}] 初始 Top-1: {init_acc:.2f}%")

        exclude_groups_set: Set[Tuple[str, int]] = set()
        exclude_groups_queue: deque[Tuple[str, int]] = deque(maxlen=20)
        forbidden_transitions_set: Set[Tuple[str, int, int, int]] = set()
        forbidden_transitions_queue: deque[Tuple[str, int, int, int]] = deque(maxlen=1000)

        attack_history = []

        for flip_idx in range(args.physical_budget):
            baseline_loss = compute_loss_and_gradients(model, calib_batches, criterion)

            selected_groups, stage_a = select_top_groups_by_grad(
                model=model,
                params=params,
                exclude_groups=exclude_groups_set,
                coarse_groups=coarse_groups,
            )
            if flip_idx == 0:
                selected_ratio = (
                    float(stage_a["selected_groups"]) / float(stage_a["eligible_groups"])
                    if stage_a["eligible_groups"] > 0 else 0.0
                )
                print(
                    f"[{now_ts()}] Stage-A coarse filter: selected_groups={stage_a['selected_groups']} "
                    f"eligible_groups={stage_a['eligible_groups']} total_groups={stage_a['total_groups']} "
                    f"ratio={selected_ratio:.4%}"
                )
            if not selected_groups:
                print(f"[Flip {flip_idx+1}] 无 coarse groups，提前停止。")
                break

            candidates, stage_a_candidates = generate_candidates_for_selected_groups(
                model=model,
                params=params,
                selected_groups=selected_groups,
                forbidden_transitions=forbidden_transitions_set,
            )
            if not candidates:
                print(f"[Flip {flip_idx+1}] 无候选，提前停止。")
                break

            best, stage_b = exact_verification_topk(
                model=model,
                param_map=param_map,
                candidates=candidates,
                verify_batches=calib_batches,
                criterion=criterion,
                baseline_loss=baseline_loss,
                top_k=args.top_k_verify,
            )
            if best is None:
                print(f"[Flip {flip_idx+1}] Top-{args.top_k_verify} 无正增益候选，提前停止。")
                break

            success = apply_pattern_change(model, param_map, best)
            if not success:
                print(f"[Flip {flip_idx+1}] 应用候选失败，提前停止。")
                break

            group_key = (best.param_name, best.group_idx)
            if group_key not in exclude_groups_set:
                if len(exclude_groups_queue) == exclude_groups_queue.maxlen:
                    popped = exclude_groups_queue.popleft()
                    exclude_groups_set.discard(popped)
                exclude_groups_queue.append(group_key)
                exclude_groups_set.add(group_key)

            forward_key = (best.param_name, best.group_idx, best.old_code, best.new_code)
            reverse_key = (best.param_name, best.group_idx, best.new_code, best.old_code)
            for t_key in (forward_key, reverse_key):
                if t_key in forbidden_transitions_set:
                    continue
                if len(forbidden_transitions_queue) == forbidden_transitions_queue.maxlen:
                    popped_t = forbidden_transitions_queue.popleft()
                    forbidden_transitions_set.discard(popped_t)
                forbidden_transitions_queue.append(t_key)
                forbidden_transitions_set.add(t_key)

            attack_history.append(
                {
                    "flip": flip_idx + 1,
                    "param_name": best.param_name,
                    "group": best.group_idx,
                    "old_pattern": best.old_pattern,
                    "new_pattern": best.new_pattern,
                    "proxy_score": best.proxy_score,
                    "stage_a_selected": stage_a["selected_groups"],
                    "stage_a_candidates": stage_a_candidates["candidates_valid"],
                    "stage_b_tested": stage_b["candidates_tested"],
                }
            )

            if (flip_idx + 1) % 10 == 0 or (flip_idx + 1) == args.physical_budget:
                print(f"[{now_ts()}] Flip {flip_idx+1}/{args.physical_budget} 已完成")

        final_acc = evaluate_top1(model, test_loader, device)
        print(f"[{now_ts()}] Seed={seed} 结束：Final Top-1={final_acc:.2f}%")

        all_results.append(
            {
                "seed": seed,
                "arch": args.arch,
                "initial_acc": init_acc,
                "final_acc": final_acc,
                "acc_drop": init_acc - final_acc,
                "num_flips": len(attack_history),
                "attack_history": attack_history,
                "config": {
                    "dataset": args.dataset,
                    "dataset_root": str(dataset_root),
                    "physical_budget": args.physical_budget,
                    "coarse_groups": coarse_groups,
                    "calib_count": calib_count,
                    "top_m_per_group": args.top_m_per_group,
                    "top_k_verify": args.top_k_verify,
                    "max_groups_per_layer": args.max_groups_per_layer,
                    "group_sampling": args.group_sampling,
                    "min_sampled_group_ratio": args.min_sampled_group_ratio,
                },
            }
        )

    csv_path = os.path.join(args.output_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "arch", "initial_acc", "final_acc", "acc_drop", "num_flips"])
        for r in all_results:
            w.writerow(
                [
                    r["seed"],
                    r["arch"],
                    f"{r['initial_acc']:.2f}",
                    f"{r['final_acc']:.2f}",
                    f"{r['acc_drop']:.2f}",
                    r["num_flips"],
                ]
            )

    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"[{now_ts()}] 结果已保存: {args.output_dir}")


if __name__ == "__main__":
    main()
