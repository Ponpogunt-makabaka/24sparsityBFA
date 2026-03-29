#!/usr/bin/env python3
"""
Run zero-only BFA attack 10 times with different seeds,
then plot a Candlestick Chart showing accuracy distribution
across iterations.

Candlestick semantics (per iteration):
  - Body:  Q1 (25th percentile) to Q3 (75th percentile)
  - Wicks: min to max across 10 runs
  - Line:  median
  - Color: red if median dropped vs previous iteration, green otherwise
"""

import os
import sys
import copy
import json
import random
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# -- local imports --
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resnet20_nm import resnet20_nm
from sparse_ops import SparseConv, SparseLinear


# ── helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(ckpt_path: str):
    """Return clean state_dict from checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "model_state"):
            if key in ckpt:
                sd = ckpt[key]
                break
        else:
            sd = ckpt
    else:
        sd = ckpt
    return {k.replace("module.", ""): v for k, v in sd.items()}


def quantize_model(model):
    """Per-layer symmetric INT8 quantization. Returns layer_info dict."""
    layer_info = {}
    quant_types = (nn.Conv2d, nn.Linear, SparseConv, SparseLinear)
    for name, module in model.named_modules():
        if not isinstance(module, quant_types):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue
        w = module.weight.data.cpu()
        max_abs = float(w.abs().max().item())
        step = max_abs / 127.0 if max_abs > 0 else 1e-8
        q = torch.clamp(torch.round(w / step), -127, 127).to(torch.int8)
        module.weight.data = (q.float() * step)
        lname = name if name else module.__class__.__name__
        layer_info[lname] = {"step_size": float(step), "quantized_int8": q.cpu()}
    return layer_info


def evaluate_top1(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            correct += model(images).argmax(1).eq(targets).sum().item()
            total += targets.size(0)
    return 100.0 * correct / total if total else 0.0


# ── single attack run ────────────────────────────────────────────────────────

def run_zero_only_attack(
    seed: int,
    state_dict: dict,
    n_iter: int,
    topk: int,
    attack_sample_size: int,
    data_root: str,
    device: torch.device,
    test_loader: DataLoader,
) -> list:
    """
    Run one zero-only BFA attack.
    Returns list of length (n_iter + 1):
      [baseline_acc, acc_after_iter1, ..., acc_after_iter_n]
    """
    set_seed(seed)

    # fresh quantized model
    model = resnet20_nm()
    model.load_state_dict(copy.deepcopy(state_dict))
    layer_info = quantize_model(model)
    model = model.to(device).eval()

    criterion = nn.CrossEntropyLoss()

    # attack data loader (shuffled with current seed)
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    trainset = datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=train_tf
    )
    attack_loader = DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True
    )

    # determine attackable layers
    layer_modules = [
        k for k in layer_info
        if (k.startswith("layer1.") or k.startswith("layer2.") or k.startswith("layer3."))
        and "downsample" not in k
    ]
    if not layer_modules:
        layer_modules = ["layer1.0.conv1"]

    # small attack batch for candidate evaluation
    def sample_attack_batch():
        imgs, tgts, proc = [], [], 0
        for ib, tb in attack_loader:
            if proc >= attack_sample_size:
                break
            rem = attack_sample_size - proc
            imgs.append(ib[:rem])
            tgts.append(tb[:rem])
            proc += imgs[-1].size(0)
        return torch.cat(imgs).to(device), torch.cat(tgts).to(device)

    test_imgs, test_tgts = sample_attack_batch()

    # baseline accuracy
    baseline = evaluate_top1(model, test_loader, device)
    accuracies = [baseline]
    applied_set = set()

    for it in range(n_iter):
        # ── gradient computation ──
        model.eval()
        model.zero_grad()
        grad_imgs, grad_tgts = sample_attack_batch()
        outputs = model(grad_imgs)
        loss = criterion(outputs, grad_tgts)
        loss.backward()
        loss_val = loss.item()

        grads = {}
        for name, p in model.named_parameters():
            if p.grad is not None and "step_size" not in name:
                grads[name] = p.grad.detach().cpu().clone()

        # ── evaluate candidates across all layers ──
        global_entries = []
        for module_name in layer_modules:
            target_name = module_name + ".weight"
            if target_name not in grads or module_name not in layer_info:
                continue
            g = grads[target_name]
            q_weights = layer_info[module_name]["quantized_int8"]
            step_size = layer_info[module_name]["step_size"]
            g_flat = g.flatten()
            _, indices = torch.sort(g_flat.abs(), descending=True)
            tk = min(topk, indices.numel())

            for i in range(tk):
                idx = indices[i].item()
                dims = g.shape
                multi_idx, temp = [], idx
                for d in reversed(dims):
                    multi_idx.append(temp % d)
                    temp //= d
                multi_idx = tuple(reversed(multi_idx))

                if (module_name, multi_idx) in applied_set:
                    continue

                orig_q = q_weights[multi_idx].item()
                flipped_q = orig_q - 128 if orig_q >= 0 else orig_q + 128

                with torch.no_grad():
                    param = dict(model.named_parameters()).get(target_name)
                    if param is None:
                        continue
                    orig_val = param.data[multi_idx].item()
                    param.data[multi_idx] = float(flipped_q) * step_size
                    new_loss = criterion(model(test_imgs), test_tgts).item()
                    param.data[multi_idx] = orig_val
                    delta = new_loss - loss_val

                if delta <= 0:
                    continue
                global_entries.append({
                    "module_name": module_name,
                    "target_name": target_name,
                    "multi_idx": multi_idx,
                    "orig_q": int(orig_q),
                    "flipped_q": int(flipped_q),
                    "delta": delta,
                    "step_size": step_size,
                })

        # ── select best ZERO candidate ──
        if not global_entries:
            accuracies.append(accuracies[-1])
            continue

        sorted_entries = sorted(global_entries, key=lambda e: e["delta"], reverse=True)
        chosen = None
        for entry in sorted_entries:
            qcpu = layer_info[entry["module_name"]]["quantized_int8"]
            if int(qcpu[entry["multi_idx"]].item()) == 0:
                chosen = entry
                break

        if chosen is None:
            accuracies.append(accuracies[-1])
            continue

        # ── apply flip ──
        with torch.no_grad():
            param = dict(model.named_parameters()).get(chosen["target_name"])
            if param is not None:
                param.data[chosen["multi_idx"]] = (
                    float(chosen["flipped_q"]) * chosen["step_size"]
                )

        qcpu = layer_info[chosen["module_name"]]["quantized_int8"].clone()
        qcpu[chosen["multi_idx"]] = torch.tensor(chosen["flipped_q"], dtype=torch.int8)
        layer_info[chosen["module_name"]]["quantized_int8"] = qcpu
        applied_set.add((chosen["module_name"], chosen["multi_idx"]))

        acc = evaluate_top1(model, test_loader, device)
        accuracies.append(acc)

    return accuracies


# ── candlestick chart ─────────────────────────────────────────────────────────

def plot_candlestick(all_runs: dict, n_iter: int, output_path: str) -> None:
    """
    Draw a candlestick chart from multi-run attack results.

    For each iteration i (0 = baseline, 1..n_iter):
      body  = Q1 → Q3  (interquartile range)
      wicks = min → max
      line  = median
      color = red if median dropped vs previous, green otherwise
    """
    n_points = n_iter + 1  # include baseline at index 0
    iters = list(range(n_points))

    # collect per-iteration statistics
    medians, q1s, q3s, mins, maxs = [], [], [], [], []
    for i in iters:
        vals = []
        for run_id in sorted(all_runs.keys()):
            curve = all_runs[run_id]
            if i < len(curve):
                vals.append(curve[i])
        arr = np.array(vals)
        medians.append(np.median(arr))
        q1s.append(np.percentile(arr, 25))
        q3s.append(np.percentile(arr, 75))
        mins.append(np.min(arr))
        maxs.append(np.max(arr))

    fig, ax = plt.subplots(figsize=(12, 6))

    body_width = 0.45
    wick_lw = 1.2
    median_lw = 2.0

    for i in iters:
        x = i
        # color: red if median dropped, green otherwise
        if i == 0:
            color = "#2ca02c"  # green for baseline
        else:
            color = "#d62728" if medians[i] < medians[i - 1] else "#2ca02c"

        body_lo = q1s[i]
        body_hi = q3s[i]
        wick_lo = mins[i]
        wick_hi = maxs[i]

        # lower wick
        ax.plot([x, x], [wick_lo, body_lo], color="black", linewidth=wick_lw, zorder=2)
        # upper wick
        ax.plot([x, x], [body_hi, wick_hi], color="black", linewidth=wick_lw, zorder=2)
        # wick caps
        cap_w = body_width * 0.4
        ax.plot([x - cap_w, x + cap_w], [wick_lo, wick_lo], color="black", linewidth=wick_lw, zorder=2)
        ax.plot([x - cap_w, x + cap_w], [wick_hi, wick_hi], color="black", linewidth=wick_lw, zorder=2)

        # body rectangle
        rect = mpatches.FancyBboxPatch(
            (x - body_width / 2, body_lo),
            body_width,
            body_hi - body_lo,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor="black",
            linewidth=1.0,
            alpha=0.85,
            zorder=3,
        )
        ax.add_patch(rect)

        # median line
        ax.plot(
            [x - body_width / 2, x + body_width / 2],
            [medians[i], medians[i]],
            color="white",
            linewidth=median_lw,
            zorder=4,
        )

    # scatter individual run points (semi-transparent)
    for run_id in sorted(all_runs.keys()):
        curve = all_runs[run_id]
        xs = list(range(len(curve)))
        ax.scatter(
            xs, curve, s=12, alpha=0.35, color="steelblue", zorder=5, edgecolors="none"
        )

    # axis formatting
    ax.set_xticks(iters)
    ax.set_xticklabels(
        ["Base"] + [str(i) for i in range(1, n_points)], fontsize=10
    )
    ax.set_xlabel("Iteration (0 = Baseline, 1‥N = after flip)", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title(
        f"Zero-Only BFA Attack — Candlestick Chart ({len(all_runs)} runs)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(-0.6, n_iter + 0.6)
    y_lo = max(0, min(mins) - 5)
    y_hi = min(100, max(maxs) + 3)
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, alpha=0.3, linestyle="--")

    # legend
    legend_elements = [
        mpatches.Patch(facecolor="#d62728", edgecolor="black", label="Median dropped"),
        mpatches.Patch(facecolor="#2ca02c", edgecolor="black", label="Median stable/up"),
        Line2D([0], [0], color="white", linewidth=2, label="Median"),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="steelblue",
            markersize=6, alpha=0.5, label="Individual runs",
        ),
    ]
    ax.legend(
        handles=legend_elements, loc="upper right", fontsize=9,
        framealpha=0.9, edgecolor="gray",
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved candlestick chart → {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Zero-only BFA x10 runs → Candlestick Chart"
    )
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--attack_sample_size", type=int, default=128)
    parser.add_argument("--ckpt", default="1_sparse_finetune.pth")
    parser.add_argument("--data_root", default=os.path.expanduser("~/data/cifar10"))
    parser.add_argument(
        "--out_dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "reports", "zero_only_10runs",
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # resolve checkpoint path
    ckpt_path = args.ckpt
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ckpt_path)

    # load once
    state_dict = load_checkpoint(ckpt_path)

    # test loader (shared across runs)
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_loader = DataLoader(
        datasets.CIFAR10(
            root=args.data_root, train=False, download=False, transform=test_tf
        ),
        batch_size=128, shuffle=False, num_workers=0, pin_memory=True,
    )

    # run attacks
    all_runs = {}
    t0 = time.time()
    for run_id in range(args.n_runs):
        print(f"\n{'='*60}")
        print(f"  Run {run_id + 1}/{args.n_runs}  (seed={run_id})")
        print(f"{'='*60}")
        t_run = time.time()
        accs = run_zero_only_attack(
            seed=run_id,
            state_dict=state_dict,
            n_iter=args.n_iter,
            topk=args.topk,
            attack_sample_size=args.attack_sample_size,
            data_root=args.data_root,
            device=device,
            test_loader=test_loader,
        )
        elapsed = time.time() - t_run
        all_runs[run_id] = accs
        print(f"  Seed {run_id}: baseline={accs[0]:.2f}% → final={accs[-1]:.2f}%  "
              f"(drop={accs[0]-accs[-1]:.2f}pp, {elapsed:.1f}s)")

    total_time = time.time() - t0
    print(f"\nAll {args.n_runs} runs completed in {total_time:.1f}s")

    # save raw results
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "zero_only_10runs_results.json")
    json_data = {
        "config": {
            "n_runs": args.n_runs,
            "n_iter": args.n_iter,
            "topk": args.topk,
            "attack_sample_size": args.attack_sample_size,
            "ckpt": ckpt_path,
        },
        "runs": {str(k): v for k, v in all_runs.items()},
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved raw results → {json_path}")

    # print summary table
    print(f"\n{'Iter':<6}", end="")
    for r in range(args.n_runs):
        print(f"{'Run'+str(r):<9}", end="")
    print(f"{'Median':<9}{'Min':<9}{'Max':<9}")
    print("-" * (6 + 9 * args.n_runs + 27))
    for i in range(args.n_iter + 1):
        label = "Base" if i == 0 else str(i)
        print(f"{label:<6}", end="")
        vals = []
        for r in range(args.n_runs):
            v = all_runs[r][i] if i < len(all_runs[r]) else float("nan")
            vals.append(v)
            print(f"{v:<9.2f}", end="")
        arr = np.array(vals)
        print(f"{np.median(arr):<9.2f}{np.min(arr):<9.2f}{np.max(arr):<9.2f}")

    # plot
    chart_path = os.path.join(args.out_dir, "zero_only_10runs_candlestick.png")
    plot_candlestick(all_runs, args.n_iter, chart_path)


if __name__ == "__main__":
    main()
