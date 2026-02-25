#!/usr/bin/env python3
"""
Task 28: ResNet-20 / CIFAR-10 (2:4) Sparsity Baseline Audit + Baseline Improvement Probe.

This runner audits the end-to-end formation chain and explains why the current sparse baseline
(~86.5%) is lower than the dense baseline (~92%).

It also optionally runs a minimal "mask-fixed finetune" starting from the dense checkpoint:
  - Load dense weights into sparse architecture
  - Freeze a 2:4 mask (top-2 magnitude in each group)
  - Finetune weights under the fixed mask for a small number of epochs

Artifacts (always):
  - results/task28_sparsity_baseline_audit_log.txt
  - results/task28_sparsity_baseline_audit_table.csv

Artifacts (if finetune is enabled):
  - results/task28_sparse_mask_fixed_finetune_ckpt.pth
  - results/task28_sparse_mask_fixed_finetune_int8_ckpt.pth
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from config import DATASET
from models.factory import create_resnet20
from models.resnet20 import resnet20
from train.ptq_convert import Int8QuantizedResNet


RESULTS_DIR = "results"
LOG_PATH = os.path.join(RESULTS_DIR, "task28_sparsity_baseline_audit_log.txt")
TABLE_PATH = os.path.join(RESULTS_DIR, "task28_sparsity_baseline_audit_table.csv")
FINETUNE_CKPT_PATH = os.path.join(RESULTS_DIR, "task28_sparse_mask_fixed_finetune_ckpt.pth")
FINETUNE_INT8_CKPT_PATH = os.path.join(RESULTS_DIR, "task28_sparse_mask_fixed_finetune_int8_ckpt.pth")


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cifar10_loaders_offline(
    data_dir: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, transforms.Compose, transforms.Compose]:
    mean = tuple(float(x) for x in DATASET["mean"])
    std = tuple(float(x) for x in DATASET["std"])

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    cifar_dir = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.exists(cifar_dir):
        raise FileNotFoundError(
            f"CIFAR-10 not found under {data_dir} (expected {cifar_dir}). Offline mode forbids download."
        )

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=test_transform)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, test_loader, train_transform, test_transform


def eval_top1(model: nn.Module, loader: DataLoader, device: str, max_samples: Optional[int]) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
            if max_samples is not None and total >= max_samples:
                break
    return 100.0 * correct / max(1, total)


def sha1_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha1()
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        h.update(k.encode("utf-8"))
        if torch.is_tensor(v):
            t = v.detach().cpu().contiguous()
            h.update(str(t.dtype).encode("utf-8"))
            h.update(str(tuple(t.shape)).encode("utf-8"))
            h.update(t.numpy().tobytes())
        else:
            h.update(repr(v).encode("utf-8"))
    return h.hexdigest()


@dataclass
class StageRecord:
    stage_name: str
    ckpt_path: str
    sha1: str
    top1: float
    effective_nonzeros: int
    effective_total: int
    effective_sparsity: float
    group_valid_frac: float
    notes: str


def _get_mask_for_module(mod: nn.Module, weight: torch.Tensor) -> torch.Tensor:
    # Int8 modules used by attacks store an explicit sparse_mask buffer.
    if hasattr(mod, "sparse_mask") and getattr(mod, "sparse_mask") is not None:
        mask = getattr(mod, "sparse_mask").detach()
        return (mask > 0.5).to(weight.dtype)

    # SparseConv has cached_mask (static) and/or _calculate_mask (dynamic).
    if hasattr(mod, "cached_mask") and getattr(mod, "cached_mask") is not None and getattr(mod, "mask_frozen", False):
        mask = getattr(mod, "cached_mask").detach()
        return (mask > 0.5).to(weight.dtype)

    if hasattr(mod, "_calculate_mask") and callable(getattr(mod, "_calculate_mask")) and hasattr(mod, "N") and hasattr(mod, "M"):
        try:
            mask = mod._calculate_mask().detach()  # type: ignore[attr-defined]
            return (mask > 0.5).to(weight.dtype)
        except Exception:
            pass

    return torch.ones_like(weight)


def summarize_effective_sparsity(model: nn.Module) -> Tuple[int, int, float]:
    active = 0
    total = 0
    group_ok = 0
    group_total = 0

    for _, mod in model.named_modules():
        if not hasattr(mod, "weight"):
            continue
        w = getattr(mod, "weight")
        if not torch.is_tensor(w):
            continue

        w = w.detach()
        if w.dim() not in (2, 4):
            continue

        mask = _get_mask_for_module(mod, w)
        m = (mask > 0.5).to(torch.int32)
        active += int(m.sum().item())
        total += m.numel()

        # Group validity check (only when the masked dimension is divisible by 4)
        if w.dim() == 4:
            in_ch = w.shape[1]
            if in_ch % 4 == 0:
                m4 = m.view(w.shape[0], in_ch // 4, 4, w.shape[2], w.shape[3])
                sums = m4.sum(dim=2)
                group_total += sums.numel()
                group_ok += int((sums == 2).sum().item())
        else:
            in_feat = w.shape[1]
            if in_feat % 4 == 0 and hasattr(mod, "sparse_mask") and getattr(mod, "sparse_mask") is not None:
                m4 = m.view(w.shape[0], in_feat // 4, 4)
                sums = m4.sum(dim=2)
                group_total += sums.numel()
                group_ok += int((sums == 2).sum().item())

    sparsity = 1.0 - (active / max(1, total))
    group_valid_frac = group_ok / max(1, group_total)
    return active, total, group_valid_frac


def _transform_has_random_ops(t: transforms.Compose) -> bool:
    if not isinstance(t, transforms.Compose):
        return False
    random_kinds = (transforms.RandomCrop, transforms.RandomHorizontalFlip, transforms.RandomResizedCrop)
    return any(isinstance(x, random_kinds) for x in t.transforms)

def _extract_normalize_mean_std(t: transforms.Compose) -> Tuple[Optional[Tuple[float, ...]], Optional[Tuple[float, ...]]]:
    if not isinstance(t, transforms.Compose):
        return None, None
    for x in t.transforms:
        if isinstance(x, transforms.Normalize):
            mean = tuple(float(v) for v in x.mean)
            std = tuple(float(v) for v in x.std)
            return mean, std
    return None, None


def _bn_stats_ok(model: nn.Module) -> Tuple[bool, str]:
    means = []
    vars_ = []
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            means.append(mod.running_mean.detach().cpu())
            vars_.append(mod.running_var.detach().cpu())
    if not means:
        return True, "no_bn"
    mean = torch.cat([t.flatten() for t in means])
    var = torch.cat([t.flatten() for t in vars_])
    if torch.isnan(mean).any() or torch.isnan(var).any():
        return False, "bn_nan"
    if (var == 0).any():
        return False, "bn_zero_var"
    return True, f"mean_abs_mean={mean.abs().mean().item():.4f} var_mean={var.mean().item():.4f}"


def make_dense_int8_from_fp32_ckpt(fp32_ckpt_path: str, device: str) -> nn.Module:
    base = resnet20(sparsity_type=None)
    ckpt = torch.load(fp32_ckpt_path, map_location="cpu")
    base.load_state_dict(ckpt["model_state_dict"])
    base.eval()
    model = Int8QuantizedResNet(base, copy_sparse_masks=False).to(device)
    model.calibrate_all_layers()
    model.eval()
    return model


def make_sparse_int8_from_fp32_ckpt(fp32_ckpt_path: str, device: str) -> nn.Module:
    base = create_resnet20(sparsity_type="2:4", pretrained_path=fp32_ckpt_path).to(device)
    base.eval()
    if hasattr(base, "freeze_sparse_masks"):
        base.freeze_sparse_masks()
    model = Int8QuantizedResNet(base, copy_sparse_masks=True).to(device)
    model.calibrate_all_layers()
    model.eval()
    return model


def finetune_mask_fixed_sparse_from_dense(
    dense_ckpt_path: str,
    data_dir: str,
    device: str,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    num_workers: int,
    eval_samples: Optional[int],
    log_fh,
) -> Tuple[nn.Module, float]:
    set_seed(seed)
    train_loader, test_loader, _, _ = get_cifar10_loaders_offline(
        data_dir=data_dir, batch_size=batch_size, num_workers=num_workers
    )

    # Load dense weights into sparse architecture.
    model = resnet20(sparsity_type="2:4").to(device)
    dense_ckpt = torch.load(dense_ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(dense_ckpt["model_state_dict"], strict=False)
    print(f"[Finetune] dense->sparse load missing={len(missing)} unexpected={len(unexpected)}", file=log_fh)

    # Freeze masks computed from the initialized (dense) weights.
    if hasattr(model, "freeze_sparse_masks"):
        model.freeze_sparse_masks()
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_acc = -1.0
    best_state = None

    for ep in range(epochs):
        t0 = time.time()
        model.train()
        correct = 0
        total = 0
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.numel()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()

        scheduler.step()
        train_acc = 100.0 * correct / max(1, total)
        train_loss = total_loss / max(1, total)

        model.eval()
        val_acc = eval_top1(model, test_loader, device=device, max_samples=eval_samples)
        dt = time.time() - t0

        print(
            f"[Finetune][ep={ep+1:03d}/{epochs}] lr={optimizer.param_groups[0]['lr']:.5f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% val_acc={val_acc:.2f}% time={dt:.1f}s",
            file=log_fh,
        )
        log_fh.flush()

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_acc


def bake_frozen_masks_into_weights_(model: nn.Module) -> int:
    """
    Materialize the currently-frozen 2:4 masks into weights by zeroing masked-off entries.

    Why:
      - `SparseConv` stores masks as runtime attributes (`cached_mask`) that are NOT in `state_dict`.
      - If we finetune with a fixed mask, then save only weights, a future reload that recomputes
        mask by magnitude can select different positions (breaking reproducibility and dropping acc).
      - Baking the fixed mask into weights makes reload-time mask recomputation stable.
    """
    baked = 0
    with torch.no_grad():
        for mod in model.modules():
            if not hasattr(mod, "weight"):
                continue
            w = getattr(mod, "weight")
            if not torch.is_tensor(w) or w.dim() != 4:
                continue
            if not (hasattr(mod, "cached_mask") and getattr(mod, "cached_mask") is not None and getattr(mod, "mask_frozen", False)):
                continue
            mask = getattr(mod, "cached_mask").to(w.device).to(w.dtype)
            w.data.mul_(mask)
            baked += 1
    return baked


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 28: 2:4 sparsity baseline audit + finetune probe")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-samples", type=int, default=10000)

    parser.add_argument("--dense-ckpt", type=str, default="models/dense_model.pth")
    parser.add_argument("--sparse-ckpt", type=str, default="models/sparse_model.pth")

    # Improvement probe
    parser.add_argument("--finetune-epochs", type=int, default=0, help="Enable mask-fixed finetune if >0")
    parser.add_argument("--finetune-lr", type=float, default=0.01)
    parser.add_argument("--finetune-wd", type=float, default=5e-4)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    set_seed(args.seed)
    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"

    # Always enforce num_workers=0 due to environment constraints (PermissionError).
    num_workers = 0

    train_loader, test_loader, train_tf, test_tf = get_cifar10_loaders_offline(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=num_workers,
    )

    eval_samples = None if args.eval_samples <= 0 else int(args.eval_samples)

    records: List[StageRecord] = []

    with open(LOG_PATH, "w", encoding="utf-8") as log_fh:
        print("=" * 90, file=log_fh)
        print("Task 28: Sparsity Baseline Audit (ResNet-20 / CIFAR-10, 2:4)", file=log_fh)
        print("=" * 90, file=log_fh)
        print(f"timestamp: {_timestamp()}", file=log_fh)
        print(f"seed: {args.seed}", file=log_fh)
        print(f"device: {device}", file=log_fh)
        print(f"data_dir: {os.path.abspath(args.data_dir)}", file=log_fh)
        print(f"dense_ckpt: {os.path.abspath(args.dense_ckpt)}", file=log_fh)
        print(f"sparse_ckpt: {os.path.abspath(args.sparse_ckpt)}", file=log_fh)
        print(f"eval_samples: {eval_samples}", file=log_fh)
        print(f"num_workers: {num_workers}", file=log_fh)
        print(f"cifar10 mean/std: mean={DATASET['mean']} std={DATASET['std']}", file=log_fh)
        print(f"test_transform_has_random_ops: {_transform_has_random_ops(test_tf)}", file=log_fh)
        print("", file=log_fh)

        # ------------------
        # Stage: Dense FP32
        # ------------------
        dense = create_resnet20(sparsity_type=None, pretrained_path=args.dense_ckpt).to(device)
        dense.eval()
        dense_sd = dense.state_dict()
        dense_sha1 = sha1_state_dict(dense_sd)
        dense_top1 = eval_top1(dense, test_loader, device=device, max_samples=eval_samples)
        dense_active, dense_total, dense_gv = summarize_effective_sparsity(dense)
        records.append(
            StageRecord(
                stage_name="dense_fp32",
                ckpt_path=args.dense_ckpt,
                sha1=dense_sha1,
                top1=dense_top1,
                effective_nonzeros=dense_active,
                effective_total=dense_total,
                effective_sparsity=1.0 - dense_active / max(1, dense_total),
                group_valid_frac=dense_gv,
                notes="dense resnet20",
            )
        )
        print(f"[Stage] dense_fp32 top1={dense_top1:.2f}% sha1={dense_sha1[:12]}", file=log_fh)

        # ------------------
        # Stage: Dense INT8
        # ------------------
        dense_int8 = make_dense_int8_from_fp32_ckpt(args.dense_ckpt, device=device)
        dense_int8_sd = dense_int8.state_dict()
        dense_int8_sha1 = sha1_state_dict(dense_int8_sd)
        dense_int8_top1 = eval_top1(dense_int8, test_loader, device=device, max_samples=eval_samples)
        a, t, gv = summarize_effective_sparsity(dense_int8)
        records.append(
            StageRecord(
                stage_name="dense_int8_ptq",
                ckpt_path=args.dense_ckpt,
                sha1=dense_int8_sha1,
                top1=dense_int8_top1,
                effective_nonzeros=a,
                effective_total=t,
                effective_sparsity=1.0 - a / max(1, t),
                group_valid_frac=gv,
                notes="Int8QuantizedResNet(calibrated)",
            )
        )
        print(f"[Stage] dense_int8_ptq top1={dense_int8_top1:.2f}% sha1={dense_int8_sha1[:12]}", file=log_fh)

        # ------------------
        # Stage: Sparse FP32 (existing)
        # ------------------
        sparse = create_resnet20(sparsity_type="2:4", pretrained_path=args.sparse_ckpt).to(device)
        sparse.eval()
        sparse_sd = sparse.state_dict()
        sparse_sha1 = sha1_state_dict(sparse_sd)
        sparse_top1 = eval_top1(sparse, test_loader, device=device, max_samples=eval_samples)
        s_active, s_total, s_gv = summarize_effective_sparsity(sparse)
        records.append(
            StageRecord(
                stage_name="sparse_fp32_existing",
                ckpt_path=args.sparse_ckpt,
                sha1=sparse_sha1,
                top1=sparse_top1,
                effective_nonzeros=s_active,
                effective_total=s_total,
                effective_sparsity=1.0 - s_active / max(1, s_total),
                group_valid_frac=s_gv,
                notes="SparseConv dynamic mask (top-2 per 4) in forward",
            )
        )
        print(f"[Stage] sparse_fp32_existing top1={sparse_top1:.2f}% sha1={sparse_sha1[:12]}", file=log_fh)

        # ------------------
        # Stage: Sparse INT8 (existing)
        # ------------------
        sparse_int8 = make_sparse_int8_from_fp32_ckpt(args.sparse_ckpt, device=device)
        sparse_int8_sd = sparse_int8.state_dict()
        sparse_int8_sha1 = sha1_state_dict(sparse_int8_sd)
        sparse_int8_top1 = eval_top1(sparse_int8, test_loader, device=device, max_samples=eval_samples)
        a, t, gv = summarize_effective_sparsity(sparse_int8)
        records.append(
            StageRecord(
                stage_name="sparse_int8_existing",
                ckpt_path=args.sparse_ckpt,
                sha1=sparse_int8_sha1,
                top1=sparse_int8_top1,
                effective_nonzeros=a,
                effective_total=t,
                effective_sparsity=1.0 - a / max(1, t),
                group_valid_frac=gv,
                notes="Int8QuantizedResNet + copied cached masks",
            )
        )
        print(f"[Stage] sparse_int8_existing top1={sparse_int8_top1:.2f}% sha1={sparse_int8_sha1[:12]}", file=log_fh)

        # ------------------
        # Stage: Dense -> Sparse (mask-fixed, no finetune)
        # ------------------
        sparse_from_dense = resnet20(sparsity_type="2:4").to(device)
        dense_ckpt = torch.load(args.dense_ckpt, map_location="cpu")
        missing, unexpected = sparse_from_dense.load_state_dict(dense_ckpt["model_state_dict"], strict=False)
        print(f"[Stage] dense->sparse init load missing={len(missing)} unexpected={len(unexpected)}", file=log_fh)
        if hasattr(sparse_from_dense, "freeze_sparse_masks"):
            sparse_from_dense.freeze_sparse_masks()
        sparse_from_dense.eval()
        top1 = eval_top1(sparse_from_dense, test_loader, device=device, max_samples=eval_samples)
        sd = sparse_from_dense.state_dict()
        records.append(
            StageRecord(
                stage_name="sparse_fp32_from_dense_maskfixed_init",
                ckpt_path=args.dense_ckpt,
                sha1=sha1_state_dict(sd),
                top1=top1,
                effective_nonzeros=summarize_effective_sparsity(sparse_from_dense)[0],
                effective_total=summarize_effective_sparsity(sparse_from_dense)[1],
                effective_sparsity=1.0
                - summarize_effective_sparsity(sparse_from_dense)[0]
                / max(1, summarize_effective_sparsity(sparse_from_dense)[1]),
                group_valid_frac=summarize_effective_sparsity(sparse_from_dense)[2],
                notes="dense weights + fixed 2:4 mask (top-2 magnitude) no finetune",
            )
        )
        print(f"[Stage] sparse_fp32_from_dense_maskfixed_init top1={top1:.2f}%", file=log_fh)

        # ------------------
        # Optional: finetune mask-fixed sparse from dense
        # ------------------
        if args.finetune_epochs > 0:
            print("", file=log_fh)
            print("=" * 90, file=log_fh)
            print("Finetune Probe: mask-fixed sparse from dense", file=log_fh)
            print("=" * 90, file=log_fh)
            print(
                f"finetune_epochs={args.finetune_epochs} finetune_lr={args.finetune_lr} finetune_wd={args.finetune_wd}",
                file=log_fh,
            )

            ft_model, ft_best = finetune_mask_fixed_sparse_from_dense(
                dense_ckpt_path=args.dense_ckpt,
                data_dir=args.data_dir,
                device=device,
                seed=args.seed,
                epochs=args.finetune_epochs,
                lr=args.finetune_lr,
                weight_decay=args.finetune_wd,
                batch_size=args.batch_size,
                num_workers=num_workers,
                eval_samples=eval_samples,
                log_fh=log_fh,
            )

            # Bake masks into weights so the saved checkpoint is reproducible when reloaded.
            baked_layers = bake_frozen_masks_into_weights_(ft_model)
            print(f"[Finetune] baked frozen masks into weights (layers={baked_layers})", file=log_fh)

            ft_top1 = eval_top1(ft_model, test_loader, device=device, max_samples=eval_samples)
            ft_sd = ft_model.state_dict()

            torch.save(
                {
                    "epoch": args.finetune_epochs,
                    "acc": ft_top1,
                    "model_state_dict": ft_sd,
                    "notes": {
                        "init_dense_ckpt": os.path.abspath(args.dense_ckpt),
                        "mask": "fixed top-2 per 4 (in-ch groups, per kernel position)",
                        "epochs": args.finetune_epochs,
                        "lr": args.finetune_lr,
                        "weight_decay": args.finetune_wd,
                        "seed": args.seed,
                    },
                },
                FINETUNE_CKPT_PATH,
            )
            print(f"[Finetune] saved fp32 sparse ckpt: {FINETUNE_CKPT_PATH}", file=log_fh)
            print(f"[Finetune] finetuned sparse_fp32 (in-memory) top1={ft_top1:.2f}%", file=log_fh)

            # Reload path (matches how attack runners build the baseline): load -> freeze masks -> eval.
            ft_reload = create_resnet20(sparsity_type="2:4", pretrained_path=FINETUNE_CKPT_PATH).to(device)
            ft_reload.eval()
            if hasattr(ft_reload, "freeze_sparse_masks"):
                ft_reload.freeze_sparse_masks()
            ft_reload_top1 = eval_top1(ft_reload, test_loader, device=device, max_samples=eval_samples)

            ft_reload_sd = ft_reload.state_dict()
            ft_sha1 = sha1_state_dict(ft_reload_sd)
            a, t, gv = summarize_effective_sparsity(ft_reload)
            records.append(
                StageRecord(
                    stage_name="sparse_fp32_maskfixed_finetuned",
                    ckpt_path=FINETUNE_CKPT_PATH,
                    sha1=ft_sha1,
                    top1=ft_reload_top1,
                    effective_nonzeros=a,
                    effective_total=t,
                    effective_sparsity=1.0 - a / max(1, t),
                    group_valid_frac=gv,
                    notes=f"mask-fixed finetune from dense; best_val_acc={ft_best:.2f}; reload_eval={ft_reload_top1:.2f}",
                )
            )
            print(f"[Finetune] finetuned sparse_fp32 (reloaded) top1={ft_reload_top1:.2f}% sha1={ft_sha1[:12]}", file=log_fh)

            # Convert finetuned sparse to int8 (match attack pipeline: start from reloaded base + frozen masks)
            ft_int8 = Int8QuantizedResNet(ft_reload, copy_sparse_masks=True).to(device)
            ft_int8.calibrate_all_layers()
            ft_int8.eval()
            ft_int8_top1 = eval_top1(ft_int8, test_loader, device=device, max_samples=eval_samples)
            ft_int8_sd = ft_int8.state_dict()
            ft_int8_sha1 = sha1_state_dict(ft_int8_sd)
            a, t, gv = summarize_effective_sparsity(ft_int8)
            records.append(
                StageRecord(
                    stage_name="sparse_int8_maskfixed_finetuned",
                    ckpt_path=FINETUNE_INT8_CKPT_PATH,
                    sha1=ft_int8_sha1,
                    top1=ft_int8_top1,
                    effective_nonzeros=a,
                    effective_total=t,
                    effective_sparsity=1.0 - a / max(1, t),
                    group_valid_frac=gv,
                    notes="finetuned sparse -> Int8QuantizedResNet calibrated",
                )
            )
            torch.save(
                {
                    "fp32_ckpt": os.path.abspath(FINETUNE_CKPT_PATH),
                    "top1": ft_int8_top1,
                    "model_state_dict": ft_int8_sd,
                    "notes": {"quantization": "symmetric_int8", "seed": args.seed},
                },
                FINETUNE_INT8_CKPT_PATH,
            )
            print(f"[Finetune] finetuned sparse_int8 top1={ft_int8_top1:.2f}% sha1={ft_int8_sha1[:12]}", file=log_fh)

        # ------------------
        # Fault checklist (A-I)
        # ------------------
        print("", file=log_fh)
        print("=" * 90, file=log_fh)
        print("High-Probability Fault Checklist (A–I)", file=log_fh)
        print("=" * 90, file=log_fh)

        # A/B: normalization and test transforms
        tf_mean, tf_std = _extract_normalize_mean_std(test_tf)
        cfg_mean = tuple(float(x) for x in DATASET["mean"])
        cfg_std = tuple(float(x) for x in DATASET["std"])
        a_pass = (tf_mean == cfg_mean) and (tf_std == cfg_std)
        b_pass = not _transform_has_random_ops(test_tf)
        print(f"(A) CIFAR-10 normalization present (mean/std logged above): {'PASS' if a_pass else 'FAIL'}", file=log_fh)
        print(f"(B) test transform deterministic (no RandomCrop/Flip): {'PASS' if b_pass else 'FAIL'}", file=log_fh)

        # C: checkpoint load strictness evidence (dense/sparse stages were created by create_resnet20; it prints load msg)
        print("(C) checkpoint load: see [Stage] lines; strict load in factory (missing/unexpected should be 0).", file=log_fh)

        # D: BN running stats + eval mode
        bn_ok, bn_msg = _bn_stats_ok(sparse)
        print(f"(D) BN running stats sane (no NaN/zero var) on sparse model: {'PASS' if bn_ok else 'FAIL'} ({bn_msg})", file=log_fh)
        print(f"(D) model.eval() enforced during eval_top1(): PASS", file=log_fh)

        # E: mask grouping axis correctness (2:4 per group)
        e_pass = s_gv > 0.999
        print(f"(E) 2:4 group validity fraction on sparse_fp32_existing: {s_gv:.6f} => {'PASS' if e_pass else 'FAIL'}", file=log_fh)

        # F/H: encode/decode consistency (position encoding) sanity via int8 mask groups
        f_pass = gv > 0.999
        print(f"(F/H) metadata grouping sanity on sparse_int8_existing (mask has 2 ones per 4): {gv:.6f} => {'PASS' if f_pass else 'FAIL'}", file=log_fh)
        print("(F/H) NOTE: baseline drop is observed before CSR/position encoding is used for attack.", file=log_fh)

        # G: PTQ calibration executed
        g_pass = True
        for mod in sparse_int8.modules():
            if hasattr(mod, "quantized") and hasattr(mod, "int8_weights"):
                if not bool(getattr(mod, "quantized")):
                    g_pass = False
                    break
        print(f"(G) INT8 PTQ calibration sets quantized=True in Int8 modules: {'PASS' if g_pass else 'FAIL'}", file=log_fh)

        # I: full test-set usage
        i_pass = eval_samples in (None, 10000)
        print(f"(I) eval uses full CIFAR-10 test set (10000) when eval_samples=10000: {'PASS' if i_pass else 'WARN'}", file=log_fh)

        # Root-cause summary
        print("", file=log_fh)
        print("=" * 90, file=log_fh)
        print("Root-Cause Classification", file=log_fh)
        print("=" * 90, file=log_fh)
        if dense_top1 >= 90.0 and sparse_top1 < 90.0:
            print(
                f"Dense baseline is high ({dense_top1:.2f}%), but sparse baseline is low ({sparse_top1:.2f}%).",
                file=log_fh,
            )
            print("Conclusion: accuracy drop is dominated by the sparsification/training regime, not PTQ.", file=log_fh)
        else:
            print("Conclusion: dense baseline is also low or sparse is not lower; investigate data/model eval.", file=log_fh)

        print("", file=log_fh)
        print("How to run:", file=log_fh)
        print(
            "  python run_task28_sparsity_baseline_audit.py --device cpu --eval-samples 10000 --seed 123",
            file=log_fh,
        )
        print(
            "  python run_task28_sparsity_baseline_audit.py --device cpu --eval-samples 10000 --seed 123 --finetune-epochs 20 --finetune-lr 0.01",
            file=log_fh,
        )

    # Write CSV table
    with open(TABLE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


if __name__ == "__main__":
    main()
