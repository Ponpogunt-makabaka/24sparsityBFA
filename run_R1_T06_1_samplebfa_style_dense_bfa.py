#!/usr/bin/env python3
"""
R1_T06.1 baseline model + sampleBFA-style dense attack.

Design goals:
- Use the same R1_T06.1 baseline INT8 sparse checkpoint loader.
- Keep sparse-gated forward semantics (never densify sparse_mask).
- Use sampleBFA search strategy:
  per-layer top-k by |grad|, sign-bit-style flip (q -> q-128 / q+128),
  exact loss verification on a fixed attack batch, global best apply.
- Do not re-quantize when the checkpoint is already INT8.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from run_R1_T06_1_sparse_gated_dense_view_exact import (
    FixedSubsetLoader,
    generate_fixed_indices,
    get_sparse_mask_stats,
    load_int8_sparse_model,
    set_all_seeds,
)


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def flip_sign_style(old_q: int) -> int:
    return old_q - 128 if old_q >= 0 else old_q + 128


def evaluate_subset(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
    max_samples: Optional[int] = None,
) -> Tuple[float, float, int]:
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = criterion(out, y)
            bs = int(x.size(0))
            total += bs
            total_loss += float(loss.item()) * bs
            correct += int((out.argmax(dim=1) == y).sum().item())
            if max_samples is not None and total >= max_samples:
                break
    if total <= 0:
        return 0.0, 0.0, 0
    return 100.0 * correct / total, total_loss / total, total


def build_attack_batch(
    train_dataset: torchvision.datasets.CIFAR10,
    attack_samples: int,
    seed: int,
    device: str,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    indices = generate_fixed_indices(train_dataset, attack_samples, seed)
    loader = FixedSubsetLoader(train_dataset, indices, batch_size=batch_size)
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    got = 0
    for x, y in loader:
        xs.append(x.to(device))
        ys.append(y.to(device))
        got += int(x.size(0))
        if got >= attack_samples:
            break
    x_attack = torch.cat(xs, dim=0)[:attack_samples]
    y_attack = torch.cat(ys, dim=0)[:attack_samples]
    return x_attack, y_attack, indices


def pick_attack_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    layers: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not (hasattr(module, "int8_weights") and hasattr(module, "scale") and hasattr(module, "weight")):
            continue
        # Match sampleBFA convention: only residual blocks, no downsample.
        if (name.startswith("layer1.") or name.startswith("layer2.") or name.startswith("layer3.")) and (
            "downsample" not in name
        ):
            layers.append((name, module))
    if not layers:
        for name, module in model.named_modules():
            if hasattr(module, "int8_weights") and hasattr(module, "scale") and hasattr(module, "weight"):
                layers.append((name, module))
    return layers


def read_int8_flat(module: nn.Module, flat_idx: int) -> int:
    with torch.no_grad():
        flat = module.int8_weights.view(-1)
        return int(flat[flat_idx].item())


def write_int8_flat(module: nn.Module, flat_idx: int, int8_value: int) -> None:
    with torch.no_grad():
        flat = module.int8_weights.view(-1)
        flat[flat_idx] = torch.tensor(int8_value, dtype=torch.int8, device=flat.device)


@dataclass
class TraceRow:
    step: int
    layer_name: str
    weight_idx: int
    old_int8: int
    new_int8: int
    grad_val: float
    attack_loss_before: float
    attack_loss_after: float
    delta_attack_loss: float
    eval_loss: float
    eval_acc: float
    n_layer_candidates: int
    n_global_candidates: int
    elapsed_sec: float


def run_samplebfa_style_attack(args: argparse.Namespace) -> Tuple[List[TraceRow], Dict[str, float]]:
    set_all_seeds(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, fallback to CPU")
        device = "cpu"

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    transform_train_attack = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=False, transform=transform_test)
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=False, transform=transform_train_attack
    )

    eval_indices = generate_fixed_indices(test_dataset, args.eval_samples, args.seed)
    eval_loader = FixedSubsetLoader(test_dataset, eval_indices, batch_size=args.eval_batch_size)

    model, is_int8_ckpt, _ = load_int8_sparse_model(args.ckpt, device)
    if not is_int8_ckpt:
        print("[Info] Checkpoint is not native INT8 ckpt; loader converted it to INT8 model.")

    mask_stats = get_sparse_mask_stats(model)
    print(
        "[Mask] layers={layers} total={total} ones={ones} zeros={zeros} density={density:.4f}".format(
            layers=int(mask_stats["layers"]),
            total=int(mask_stats["total"]),
            ones=int(mask_stats["ones"]),
            zeros=int(mask_stats["zeros"]),
            density=mask_stats["density"],
        )
    )

    attack_layers = pick_attack_layers(model)
    if not attack_layers:
        raise RuntimeError("No attackable INT8 layers found.")
    print(f"[Attack] sampleBFA-style layers: {len(attack_layers)}")

    criterion = nn.CrossEntropyLoss()
    eval_acc0, eval_loss0, _ = evaluate_subset(model, eval_loader, criterion, device, max_samples=args.eval_samples)
    print(f"[Step 0] Eval acc={eval_acc0:.2f}% loss={eval_loss0:.6f}")

    x_attack, y_attack, attack_indices = build_attack_batch(
        train_dataset=train_dataset,
        attack_samples=args.attack_samples,
        seed=args.seed + 17,
        device=device,
        batch_size=args.attack_batch_size,
    )
    print(f"[Attack Batch] n={x_attack.size(0)}")

    traces: List[TraceRow] = []
    applied_set: Set[Tuple[str, int]] = set()

    prev_eval_acc = eval_acc0
    prev_eval_loss = eval_loss0

    for step in range(1, args.n_iter + 1):
        step_t0 = time.time()

        model.eval()
        model.zero_grad(set_to_none=True)
        out = model(x_attack)
        attack_loss_before = float(criterion(out, y_attack).item())
        out = None
        attack_global_candidates = 0

        # Gradients for Stage A candidate scoring (sampleBFA style).
        loss = criterion(model(x_attack), y_attack)
        loss.backward()

        global_best = None

        for layer_name, module in attack_layers:
            grad = module.weight.grad
            if grad is None:
                continue

            grad_flat = grad.detach().view(-1)
            order = torch.argsort(torch.abs(grad_flat), descending=True)
            topk = min(args.topk, int(order.numel()))

            layer_best = None
            layer_candidates = 0

            for rank in range(topk):
                idx = int(order[rank].item())
                key = (layer_name, idx)
                if key in applied_set:
                    continue

                g_val = float(grad_flat[idx].item())
                old_q = read_int8_flat(module, idx)
                new_q = flip_sign_style(old_q)

                write_int8_flat(module, idx, new_q)
                with torch.no_grad():
                    new_loss = float(criterion(model(x_attack), y_attack).item())
                write_int8_flat(module, idx, old_q)

                delta = new_loss - attack_loss_before
                if delta <= 0:
                    continue

                layer_candidates += 1
                entry = {
                    "layer_name": layer_name,
                    "module": module,
                    "weight_idx": idx,
                    "old_int8": old_q,
                    "new_int8": new_q,
                    "grad_val": g_val,
                    "attack_loss_after": new_loss,
                    "delta_attack_loss": delta,
                }
                if layer_best is None or delta > layer_best["delta_attack_loss"]:
                    layer_best = entry

            if layer_best is not None:
                layer_best["n_layer_candidates"] = layer_candidates
                attack_global_candidates += layer_candidates
                if global_best is None or layer_best["delta_attack_loss"] > global_best["delta_attack_loss"]:
                    global_best = layer_best

            if module.weight.grad is not None:
                module.weight.grad = None

        if global_best is None:
            print(f"[Step {step}] no positive-delta candidate, stop.")
            break

        chosen_layer = global_best["layer_name"]
        chosen_module = global_best["module"]
        chosen_idx = int(global_best["weight_idx"])
        old_q = int(global_best["old_int8"])
        new_q = int(global_best["new_int8"])

        write_int8_flat(chosen_module, chosen_idx, new_q)
        applied_set.add((chosen_layer, chosen_idx))

        eval_acc, eval_loss, _ = evaluate_subset(model, eval_loader, criterion, device, max_samples=args.eval_samples)
        elapsed = time.time() - step_t0

        tr = TraceRow(
            step=step,
            layer_name=chosen_layer,
            weight_idx=chosen_idx,
            old_int8=old_q,
            new_int8=new_q,
            grad_val=float(global_best["grad_val"]),
            attack_loss_before=float(attack_loss_before),
            attack_loss_after=float(global_best["attack_loss_after"]),
            delta_attack_loss=float(global_best["delta_attack_loss"]),
            eval_loss=float(eval_loss),
            eval_acc=float(eval_acc),
            n_layer_candidates=int(global_best["n_layer_candidates"]),
            n_global_candidates=int(attack_global_candidates),
            elapsed_sec=float(elapsed),
        )
        traces.append(tr)

        print(
            f"[Step {step:02d}] layer={chosen_layer} idx={chosen_idx} q:{old_q}->{new_q} "
            f"delta_attack={tr.delta_attack_loss:+.6f} eval_acc={eval_acc:.2f}% eval_loss={eval_loss:.6f} "
            f"cands={attack_global_candidates} t={elapsed:.2f}s"
        )

        prev_eval_acc = eval_acc
        prev_eval_loss = eval_loss

    final_eval_acc = prev_eval_acc
    final_eval_loss = prev_eval_loss
    summary = {
        "seed": float(args.seed),
        "n_iter": float(args.n_iter),
        "topk": float(args.topk),
        "attack_samples": float(args.attack_samples),
        "eval_samples": float(args.eval_samples),
        "baseline_eval_acc": float(eval_acc0),
        "baseline_eval_loss": float(eval_loss0),
        "final_eval_acc": float(final_eval_acc),
        "final_eval_loss": float(final_eval_loss),
        "acc_drop": float(eval_acc0 - final_eval_acc),
        "loss_increase": float(final_eval_loss - eval_loss0),
        "applied_flips": float(len(traces)),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "attack_indices_seed.json"), "w") as f:
        json.dump(attack_indices, f)
    with open(os.path.join(args.out_dir, "eval_indices_seed.json"), "w") as f:
        json.dump(eval_indices, f)

    return traces, summary


def save_outputs(args: argparse.Namespace, traces: List[TraceRow], summary: Dict[str, float]) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    trace_csv = os.path.join(args.out_dir, "R1_T06_1_samplebfa_style_trace.csv")
    with open(trace_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "step",
                "layer_name",
                "weight_idx",
                "old_int8",
                "new_int8",
                "grad_val",
                "attack_loss_before",
                "attack_loss_after",
                "delta_attack_loss",
                "eval_loss",
                "eval_acc",
                "n_layer_candidates",
                "n_global_candidates",
                "elapsed_sec",
            ]
        )
        for t in traces:
            w.writerow(
                [
                    t.step,
                    t.layer_name,
                    t.weight_idx,
                    t.old_int8,
                    t.new_int8,
                    f"{t.grad_val:.8f}",
                    f"{t.attack_loss_before:.8f}",
                    f"{t.attack_loss_after:.8f}",
                    f"{t.delta_attack_loss:.8f}",
                    f"{t.eval_loss:.8f}",
                    f"{t.eval_acc:.4f}",
                    t.n_layer_candidates,
                    t.n_global_candidates,
                    f"{t.elapsed_sec:.4f}",
                ]
            )

    summary_csv = os.path.join(args.out_dir, "R1_T06_1_samplebfa_style_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "seed",
                "n_iter",
                "topk",
                "attack_samples",
                "eval_samples",
                "baseline_eval_acc",
                "baseline_eval_loss",
                "final_eval_acc",
                "final_eval_loss",
                "acc_drop",
                "loss_increase",
                "applied_flips",
            ]
        )
        w.writerow(
            [
                int(summary["seed"]),
                int(summary["n_iter"]),
                int(summary["topk"]),
                int(summary["attack_samples"]),
                int(summary["eval_samples"]),
                f"{summary['baseline_eval_acc']:.4f}",
                f"{summary['baseline_eval_loss']:.8f}",
                f"{summary['final_eval_acc']:.4f}",
                f"{summary['final_eval_loss']:.8f}",
                f"{summary['acc_drop']:.4f}",
                f"{summary['loss_increase']:.8f}",
                int(summary["applied_flips"]),
            ]
        )

    log_path = os.path.join(args.out_dir, "R1_T06_1_samplebfa_style_run_log.txt")
    with open(log_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("R1_T06.1 baseline + sampleBFA-style dense attack\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {now_ts()}\n")
        f.write(f"Checkpoint: {os.path.abspath(args.ckpt)}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"n_iter: {args.n_iter}\n")
        f.write(f"topk(per-layer): {args.topk}\n")
        f.write(f"attack_samples(train subset): {args.attack_samples}\n")
        f.write(f"eval_samples(test subset): {args.eval_samples}\n")
        f.write("\nResults:\n")
        f.write(f"baseline_eval_acc: {summary['baseline_eval_acc']:.4f}\n")
        f.write(f"final_eval_acc: {summary['final_eval_acc']:.4f}\n")
        f.write(f"acc_drop: {summary['acc_drop']:.4f}\n")
        f.write(f"baseline_eval_loss: {summary['baseline_eval_loss']:.8f}\n")
        f.write(f"final_eval_loss: {summary['final_eval_loss']:.8f}\n")
        f.write(f"loss_increase: {summary['loss_increase']:.8f}\n")
        f.write(f"applied_flips: {int(summary['applied_flips'])}\n")
        f.write("\nReproduce:\n")
        f.write(
            "python run_R1_T06_1_samplebfa_style_dense_bfa.py "
            f"--device {args.device} --seed {args.seed} --n-iter {args.n_iter} --topk {args.topk} "
            f"--attack-samples {args.attack_samples} --attack-batch-size {args.attack_batch_size} "
            f"--eval-samples {args.eval_samples} --eval-batch-size {args.eval_batch_size} "
            f"--ckpt {args.ckpt} --out-dir {args.out_dir}\n"
        )

    print(f"[Output] trace: {trace_csv}")
    print(f"[Output] summary: {summary_csv}")
    print(f"[Output] log: {log_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run sampleBFA-style dense attack on R1_T06.1 baseline model")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument(
        "--ckpt",
        type=str,
        default="results/legacy_L0/by_task/task28_sparse_mask_fixed_finetune_int8_ckpt.pth",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-iter", type=int, default=50)
    p.add_argument("--topk", type=int, default=5, help="per-layer top-k |grad| candidates")
    p.add_argument("--attack-samples", type=int, default=128)
    p.add_argument("--attack-batch-size", type=int, default=128)
    p.add_argument("--eval-samples", type=int, default=2000)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--out-dir", type=str, default="results/R1/R1_T06_1_samplebfa_style_dense_bfa")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    traces, summary = run_samplebfa_style_attack(args)
    save_outputs(args, traces, summary)
    print(
        "[Summary] flips={flips} baseline_acc={ba:.2f}% final_acc={fa:.2f}% drop={drop:.2f}%".format(
            flips=int(summary["applied_flips"]),
            ba=summary["baseline_eval_acc"],
            fa=summary["final_eval_acc"],
            drop=summary["acc_drop"],
        )
    )


if __name__ == "__main__":
    main()
