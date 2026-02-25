#!/usr/bin/env python3
"""
R1_T07: sampleBFA-style dense-view BFA on R1_T06.1 baseline model.

Key semantics:
- Baseline model: same checkpoint pipeline as R1_T06.1.
- Keep sparse-gated forward semantics (do NOT densify sparse masks).
- sampleBFA-style search:
  per-layer top-k by |grad|, sign-style int8 flip, exact loss verify on fixed attack batch.
- Run three modes in one command:
  global / zero_only / nonzero_only.
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

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from run_R1_T06_1_sparse_gated_dense_view_exact import (
    FixedSubsetLoader,
    detect_zero_point,
    generate_fixed_indices,
    get_sparse_mask_stats,
    load_int8_sparse_model,
    reload_model_for_mode,
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


def value_allowed(mode: str, q: int, zero_point: int) -> bool:
    if mode == "global":
        return True
    if mode == "zero_only":
        return q == zero_point
    if mode == "nonzero_only":
        return q != zero_point
    raise ValueError(f"Unknown mode: {mode}")


@dataclass
class StepTrace:
    step: int
    mode: str
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


@dataclass
class ModeResult:
    mode: str
    display_name: str
    baseline_acc: float
    baseline_loss: float
    final_acc: float
    final_loss: float
    applied_flips: int
    acc_history: List[float]
    loss_history: List[float]
    traces: List[StepTrace]


def run_mode(
    model: nn.Module,
    mode: str,
    display_name: str,
    zero_point: int,
    x_attack: torch.Tensor,
    y_attack: torch.Tensor,
    eval_loader,
    criterion: nn.Module,
    args: argparse.Namespace,
) -> ModeResult:
    attack_layers = pick_attack_layers(model)
    if not attack_layers:
        raise RuntimeError("No attackable INT8 layers found.")

    eval_acc0, eval_loss0, _ = evaluate_subset(model, eval_loader, criterion, args.device, max_samples=args.eval_samples)
    print(f"[{mode}] Step 0: eval_acc={eval_acc0:.2f}% eval_loss={eval_loss0:.6f}")

    traces: List[StepTrace] = []
    applied_set: Set[Tuple[str, int]] = set()
    acc_history = [eval_acc0]
    loss_history = [eval_loss0]

    prev_eval_acc = eval_acc0
    prev_eval_loss = eval_loss0

    for step in range(1, args.physical_budget + 1):
        step_t0 = time.time()
        model.eval()
        model.zero_grad(set_to_none=True)

        with torch.no_grad():
            attack_loss_before = float(criterion(model(x_attack), y_attack).item())

        loss = criterion(model(x_attack), y_attack)
        loss.backward()

        global_best = None
        global_candidates = 0

        for layer_name, module in attack_layers:
            grad = module.weight.grad
            if grad is None:
                continue
            grad_flat = grad.detach().view(-1)
            order = torch.argsort(torch.abs(grad_flat), descending=True)

            layer_best = None
            layer_candidates = 0
            eligible_seen = 0

            for idx_t in order:
                idx = int(idx_t.item())
                key = (layer_name, idx)
                if key in applied_set:
                    continue

                old_q = read_int8_flat(module, idx)
                if not value_allowed(mode, old_q, zero_point):
                    continue

                eligible_seen += 1
                if eligible_seen > args.topk:
                    break

                g_val = float(grad_flat[idx].item())
                new_q = flip_sign_style(old_q)

                write_int8_flat(module, idx, new_q)
                with torch.no_grad():
                    new_loss = float(criterion(model(x_attack), y_attack).item())
                write_int8_flat(module, idx, old_q)

                delta = new_loss - attack_loss_before
                if delta <= 0:
                    continue

                layer_candidates += 1
                cand = {
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
                    layer_best = cand

            if layer_best is not None:
                layer_best["n_layer_candidates"] = layer_candidates
                global_candidates += layer_candidates
                if global_best is None or layer_best["delta_attack_loss"] > global_best["delta_attack_loss"]:
                    global_best = layer_best

            if module.weight.grad is not None:
                module.weight.grad = None

        if global_best is None:
            print(f"[{mode}] Step {step:02d}: no positive-delta candidate, stop.")
            break

        chosen_layer = global_best["layer_name"]
        chosen_module = global_best["module"]
        chosen_idx = int(global_best["weight_idx"])
        old_q = int(global_best["old_int8"])
        new_q = int(global_best["new_int8"])

        write_int8_flat(chosen_module, chosen_idx, new_q)
        applied_set.add((chosen_layer, chosen_idx))

        eval_acc, eval_loss, _ = evaluate_subset(
            model, eval_loader, criterion, args.device, max_samples=args.eval_samples
        )
        elapsed = time.time() - step_t0

        tr = StepTrace(
            step=step,
            mode=mode,
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
            n_global_candidates=int(global_candidates),
            elapsed_sec=float(elapsed),
        )
        traces.append(tr)

        print(
            f"[{mode}] Step {step:02d}: layer={chosen_layer} idx={chosen_idx} q:{old_q}->{new_q} "
            f"delta_attack={tr.delta_attack_loss:+.6f} eval_acc={eval_acc:.2f}% eval_loss={eval_loss:.6f} "
            f"cands={global_candidates} t={elapsed:.2f}s"
        )

        acc_history.append(eval_acc)
        loss_history.append(eval_loss)
        prev_eval_acc = eval_acc
        prev_eval_loss = eval_loss

    return ModeResult(
        mode=mode,
        display_name=display_name,
        baseline_acc=eval_acc0,
        baseline_loss=eval_loss0,
        final_acc=prev_eval_acc,
        final_loss=prev_eval_loss,
        applied_flips=len(traces),
        acc_history=acc_history,
        loss_history=loss_history,
        traces=traces,
    )


def save_trace_csv(path: str, traces: List[StepTrace]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "step",
                "mode",
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
                    t.mode,
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


def save_summary(path: str, results: List[ModeResult], args: argparse.Namespace) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "mode",
                "display_name",
                "seed",
                "physical_budget",
                "topk_per_layer",
                "attack_samples",
                "eval_samples",
                "baseline_eval_acc",
                "final_eval_acc",
                "acc_drop",
                "baseline_eval_loss",
                "final_eval_loss",
                "loss_increase",
                "applied_flips",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.mode,
                    r.display_name,
                    args.seed,
                    args.physical_budget,
                    args.topk,
                    args.attack_samples,
                    args.eval_samples,
                    f"{r.baseline_acc:.4f}",
                    f"{r.final_acc:.4f}",
                    f"{(r.baseline_acc - r.final_acc):.4f}",
                    f"{r.baseline_loss:.8f}",
                    f"{r.final_loss:.8f}",
                    f"{(r.final_loss - r.baseline_loss):.8f}",
                    r.applied_flips,
                ]
            )


def save_combined_curve(path: str, results: List[ModeResult]) -> None:
    plt.figure(figsize=(14, 5.5))
    ax1 = plt.subplot(1, 2, 1)
    for r in results:
        ax1.plot(range(len(r.acc_history)), r.acc_history, marker="o", linewidth=2, markersize=3, label=r.mode)
    ax1.set_title("R1_T07 Accuracy Curves")
    ax1.set_xlabel("Physical Flips")
    ax1.set_ylabel("Top-1 Accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    for r in results:
        ax2.plot(range(len(r.loss_history)), r.loss_history, marker="o", linewidth=2, markersize=3, label=r.mode)
    ax2.set_title("R1_T07 Loss Curves")
    ax2.set_xlabel("Physical Flips")
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_run_log(path: str, args: argparse.Namespace, zero_point: int, mask_stats: Dict[str, float], results: List[ModeResult]) -> None:
    with open(path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("R1_T07 sampleBFA-style dense-view BFA\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {now_ts()}\n")
        f.write(f"Checkpoint: {os.path.abspath(args.ckpt)}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Modes: {','.join(args.modes)}\n")
        f.write(f"Physical budget: {args.physical_budget}\n")
        f.write(f"Topk per layer: {args.topk}\n")
        f.write(f"Attack samples (train subset): {args.attack_samples}\n")
        f.write(f"Eval samples (test subset): {args.eval_samples}\n")
        f.write(f"Zero point: {zero_point}\n")
        f.write(
            f"Mask stats: layers={int(mask_stats['layers'])}, total={int(mask_stats['total'])}, "
            f"ones={int(mask_stats['ones'])}, zeros={int(mask_stats['zeros'])}, density={mask_stats['density']:.4f}\n"
        )
        f.write("\nMode Results:\n")
        for r in results:
            f.write(
                f"- {r.mode}: baseline_acc={r.baseline_acc:.4f}, final_acc={r.final_acc:.4f}, "
                f"acc_drop={(r.baseline_acc-r.final_acc):.4f}, baseline_loss={r.baseline_loss:.8f}, "
                f"final_loss={r.final_loss:.8f}, applied_flips={r.applied_flips}\n"
            )
        f.write("\nReproduce:\n")
        f.write(
            "python run_R1_T07_samplebfa_style_dense_bfa.py "
            f"--device {args.device} --seed {args.seed} --physical-budget {args.physical_budget} "
            f"--modes {' '.join(args.modes)} "
            f"--topk {args.topk} --attack-samples {args.attack_samples} --attack-batch-size {args.attack_batch_size} "
            f"--eval-samples {args.eval_samples} --eval-batch-size {args.eval_batch_size} "
            f"--ckpt {args.ckpt} --out-dir {args.out_dir}\n"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R1_T07 sampleBFA-style dense-view BFA")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument(
        "--ckpt",
        type=str,
        default="results/legacy_L0/by_task/task28_sparse_mask_fixed_finetune_int8_ckpt.pth",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--modes",
        nargs="+",
        default=["global", "zero_only", "nonzero_only"],
        choices=["global", "zero_only", "nonzero_only"],
        help="Subset of modes to run",
    )
    p.add_argument("--physical-budget", type=int, default=50)
    p.add_argument("--topk", type=int, default=5, help="per-layer top-k eligible indices")
    p.add_argument("--attack-samples", type=int, default=128)
    p.add_argument("--attack-batch-size", type=int, default=128)
    p.add_argument("--eval-samples", type=int, default=2000)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--out-dir", type=str, default="results/R1/R1_T07_samplebfa_style_dense_bfa")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_all_seeds(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, fallback to CPU")
        device = "cpu"
    args.device = device

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
    with open(os.path.join(args.out_dir, "eval_indices_seed.json"), "w") as f:
        json.dump(eval_indices, f)

    x_attack, y_attack, attack_indices = build_attack_batch(
        train_dataset=train_dataset,
        attack_samples=args.attack_samples,
        seed=args.seed + 17,
        device=device,
        batch_size=args.attack_batch_size,
    )
    with open(os.path.join(args.out_dir, "attack_indices_seed.json"), "w") as f:
        json.dump(attack_indices, f)

    model_seed, is_int8_ckpt, state_dict_to_load = load_int8_sparse_model(args.ckpt, device)
    zero_point = detect_zero_point(model_seed)
    mask_stats = get_sparse_mask_stats(model_seed)
    del model_seed

    print(
        "[Mask] layers={layers} total={total} ones={ones} zeros={zeros} density={density:.4f}".format(
            layers=int(mask_stats["layers"]),
            total=int(mask_stats["total"]),
            ones=int(mask_stats["ones"]),
            zeros=int(mask_stats["zeros"]),
            density=mask_stats["density"],
        )
    )
    print(f"[Model] zero_point={zero_point}")

    criterion = nn.CrossEntropyLoss()
    mode_display = {
        "global": "global",
        "zero_only": "zero-only",
        "nonzero_only": "non-zero-only",
    }
    mode_task_id = {
        "global": 1,
        "zero_only": 2,
        "nonzero_only": 3,
    }
    mode_specs = [(m, mode_display[m]) for m in args.modes]
    all_results: List[ModeResult] = []

    for mode, display_name in mode_specs:
        print("\n" + "=" * 70)
        print(f"R1_T07 running mode {mode}")
        print("=" * 70)
        model = reload_model_for_mode(
            is_int8_ckpt=is_int8_ckpt,
            state_dict_to_load=state_dict_to_load,
            ckpt_path=args.ckpt,
            device=device,
        )
        mode_result = run_mode(
            model=model,
            mode=mode,
            display_name=display_name,
            zero_point=zero_point,
            x_attack=x_attack,
            y_attack=y_attack,
            eval_loader=eval_loader,
            criterion=criterion,
            args=args,
        )
        all_results.append(mode_result)
        trace_path = os.path.join(args.out_dir, f"R1_T07_task{mode_task_id[mode]}_{mode}_trace.csv")
        save_trace_csv(trace_path, mode_result.traces)
        print(f"[Output] trace: {trace_path}")

    summary_path = os.path.join(args.out_dir, "R1_T07_summary_table.csv")
    save_summary(summary_path, all_results, args)
    print(f"[Output] summary: {summary_path}")

    curve_path = os.path.join(args.out_dir, "R1_T07_acc_loss_3modes.png")
    save_combined_curve(curve_path, all_results)
    print(f"[Output] curve: {curve_path}")

    run_log_path = os.path.join(args.out_dir, "R1_T07_run_log.txt")
    save_run_log(run_log_path, args, zero_point, mask_stats, all_results)
    print(f"[Output] run log: {run_log_path}")

    print("\n[Summary] R1_T07 completed:")
    for r in all_results:
        print(
            f"  {r.mode}: {r.baseline_acc:.2f}% -> {r.final_acc:.2f}% "
            f"(drop {r.baseline_acc - r.final_acc:.2f}%), "
            f"loss {r.baseline_loss:.4f} -> {r.final_loss:.4f}, flips={r.applied_flips}"
        )


if __name__ == "__main__":
    main()
