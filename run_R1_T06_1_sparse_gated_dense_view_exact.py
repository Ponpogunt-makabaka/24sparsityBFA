#!/usr/bin/env python3
"""
R1_T06.1 (Rewritten): Sparse-Gated Weight-Bit BFA with Dense-View Candidate Search
and Top-K Exact Verification.

Latest requested semantics:
- Keep model physically sparse (sparse_mask stays original 0/1, never overwritten to all ones)
- Candidate generation uses dense-view global search over all weights
- Stage B exact verification evaluates real sparse-gated forward and naturally rejects
  ineffective candidates (e.g., masked-zero sites with no forward impact)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from models.factory import create_resnet20
from train.ptq_convert import Int8QuantizedResNet


# =============================================================================
# Utilities
# =============================================================================


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flip_int8_value(int8_val: int, bit_pos: int) -> int:
    u8 = int8_val & 0xFF
    u8 ^= (1 << bit_pos)
    return u8 - 256 if u8 >= 128 else u8


def compute_int8_delta(int8_val: int, bit_pos: int, scale: float) -> float:
    flipped_val = flip_int8_value(int8_val, bit_pos)
    return (flipped_val - int8_val) * scale


def detect_zero_point(model: torch.nn.Module) -> int:
    for _, module in model.named_modules():
        if hasattr(module, "zero_point"):
            zp = module.zero_point
            return int(zp.item()) if hasattr(zp, "item") else int(zp)
    return 0


def get_sparse_mask_stats(model: torch.nn.Module) -> Dict[str, float]:
    total = 0
    ones = 0
    layers = 0
    for _, module in model.named_modules():
        if hasattr(module, "sparse_mask") and module.sparse_mask is not None:
            mask = module.sparse_mask
            ones += int((mask > 0.5).sum().item())
            total += int(mask.numel())
            layers += 1
    zeros = total - ones
    density = (float(ones) / float(total)) if total > 0 else 1.0
    return {
        "layers": float(layers),
        "total": float(total),
        "ones": float(ones),
        "zeros": float(zeros),
        "density": density,
    }


def generate_fixed_indices(dataset: torchvision.datasets.CIFAR10, n_samples: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    total = len(dataset)
    n = min(n_samples, total)
    return sorted(rng.choice(total, size=n, replace=False).tolist())


class FixedSubsetLoader:
    def __init__(self, dataset: torchvision.datasets.CIFAR10, indices: List[int], batch_size: int = 256):
        self.dataset = dataset
        self.indices = indices
        self.batch_size = batch_size
        self.n_samples = len(indices)

    def __iter__(self):
        subset = torch.utils.data.Subset(self.dataset, self.indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        yield from loader

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class StepTrace:
    step: int
    mode: str
    flips_used: int
    cost: int
    layer_name: str
    weight_idx: int
    bit_pos: int
    old_int8: int
    new_int8: int
    delta_value_fp32: float
    proxy_score: float
    exact_score: float
    L_calib: float
    delta_L_calib: float
    L_eval: float
    delta_L_eval: float
    acc_eval: float
    delta_acc_eval: float
    topk_considered: int
    verification_time: float
    objective: float


@dataclass
class AttackResult:
    mode: str
    mode_display: str
    seed: int
    physical_budget: int
    calib_samples: int
    eval_samples: int
    topk: int
    initial_accuracy: float
    initial_loss_eval: float
    initial_loss_calib: float
    final_accuracy: float
    final_loss_eval: float
    final_loss_calib: float
    accuracy_history: List[float]
    loss_eval_history: List[float]
    loss_calib_history: List[float]
    traces: List[StepTrace]
    acc_increase_steps: List[int]
    loss_decrease_steps: List[int]
    wall_time_sec: float


# =============================================================================
# Attack Engine
# =============================================================================


class SparseGatedDenseViewBFAExact:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.flipped_bits: Set[Tuple[str, int, int]] = set()

        self.quantized_layers: List[Tuple[str, nn.Module]] = []
        for name, module in self.model.named_modules():
            if hasattr(module, "int8_weights") and hasattr(module, "scale"):
                self.quantized_layers.append((name, module))

    def _collect_batch(self, loader, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        n = 0
        for inputs, targets in loader:
            xs.append(inputs.to(self.device))
            ys.append(targets.to(self.device))
            n += int(inputs.size(0))
            if n >= n_samples:
                break
        x = torch.cat(xs, dim=0)[:n_samples]
        y = torch.cat(ys, dim=0)[:n_samples]
        return x, y

    def compute_gradients(self, calib_loader, calib_samples: int) -> bool:
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        x, y = self._collect_batch(calib_loader, calib_samples)

        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        loss = criterion(out, y)
        loss.backward()
        return True

    def compute_exact_loss(self, calib_loader, calib_samples: int) -> float:
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        x, y = self._collect_batch(calib_loader, calib_samples)
        with torch.no_grad():
            out = self.model(x)
            loss = criterion(out, y)
        return float(loss.item())

    def enumerate_candidates(
        self,
        weight_filter: Optional[Callable[[int], bool]] = None,
    ) -> List[Tuple[float, str, nn.Module, int, int, int, float]]:
        """
        Dense-view candidate search:
        - Iterate all int8 positions without sparse-mask-based filtering
        - Keep only positive proxy-score candidates
        """
        candidates: List[Tuple[float, str, nn.Module, int, int, int, float]] = []

        for layer_name, module in self.quantized_layers:
            if module.weight.grad is None:
                continue

            grad_flat = module.weight.grad.data.flatten()
            int8_flat = module.int8_weights.flatten()
            scale = float(module.scale.item())

            numel = int(int8_flat.numel())
            for idx in range(numel):
                int8_val = int(int8_flat[idx].item())

                if weight_filter is not None and not weight_filter(int8_val):
                    continue

                grad_val = float(grad_flat[idx].item())
                if abs(grad_val) < 1e-12:
                    continue

                for bit_pos in range(8):
                    key = (layer_name, idx, bit_pos)
                    if key in self.flipped_bits:
                        continue

                    delta = compute_int8_delta(int8_val, bit_pos, scale)
                    score = grad_val * delta
                    if score > 0:
                        candidates.append((score, layer_name, module, idx, bit_pos, int8_val, delta))

            if module.weight.grad is not None:
                module.weight.grad = None

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates

    @staticmethod
    def _save_value(module: nn.Module, weight_idx: int) -> int:
        with torch.no_grad():
            flat = module.int8_weights.flatten()
            return int(flat[weight_idx].item())

    @staticmethod
    def _restore_value(module: nn.Module, weight_idx: int, value: int) -> None:
        with torch.no_grad():
            flat = module.int8_weights.flatten()
            flat[weight_idx] = torch.tensor(value, dtype=torch.int8, device=flat.device)

    @staticmethod
    def _flip_inplace(module: nn.Module, weight_idx: int, bit_pos: int) -> None:
        with torch.no_grad():
            flat = module.int8_weights.flatten()
            old_val = int(flat[weight_idx].item())
            new_val = flip_int8_value(old_val, bit_pos)
            flat[weight_idx] = torch.tensor(new_val, dtype=torch.int8, device=flat.device)

    def flip_bit(self, layer_name: str, module: nn.Module, weight_idx: int, bit_pos: int) -> Tuple[int, int]:
        with torch.no_grad():
            flat = module.int8_weights.flatten()
            old_val = int(flat[weight_idx].item())
            new_val = flip_int8_value(old_val, bit_pos)
            flat[weight_idx] = torch.tensor(new_val, dtype=torch.int8, device=flat.device)
            self.flipped_bits.add((layer_name, weight_idx, bit_pos))
        return old_val, new_val

    def exact_verify_topk_candidates(
        self,
        candidates: List[Tuple[float, str, nn.Module, int, int, int, float]],
        calib_loader,
        calib_samples: int,
        baseline_loss: float,
        topk: int = 64,
    ) -> Tuple[Optional[Tuple[float, str, nn.Module, int, int, int, float]], float, int]:
        if not candidates:
            return None, baseline_loss, 0

        topk_candidates = candidates[: min(topk, len(candidates))]
        best_candidate = None
        best_exact_loss = baseline_loss

        for candidate in topk_candidates:
            _, _, module, weight_idx, bit_pos, _, _ = candidate
            old_value = self._save_value(module, weight_idx)
            self._flip_inplace(module, weight_idx, bit_pos)
            try:
                exact_loss = self.compute_exact_loss(calib_loader, calib_samples)
            finally:
                self._restore_value(module, weight_idx, old_value)

            if exact_loss > best_exact_loss:
                best_exact_loss = exact_loss
                best_candidate = candidate

        return best_candidate, best_exact_loss, len(topk_candidates)

    def evaluate(self, data_loader, criterion: nn.Module, max_samples: Optional[int] = None) -> Tuple[float, float, int]:
        self.model.eval()
        total = 0
        correct = 0
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                bs = int(inputs.size(0))
                total += bs
                total_loss += float(loss.item()) * bs
                pred = outputs.argmax(dim=1)
                correct += int((pred == targets).sum().item())

                if max_samples is not None and total >= max_samples:
                    break

        acc = 100.0 * float(correct) / float(total)
        avg_loss = float(total_loss) / float(total)
        return acc, avg_loss, total


# =============================================================================
# Runner
# =============================================================================


def run_attack_mode_exact(
    model: nn.Module,
    mode: str,
    mode_display: str,
    physical_budget: int,
    calib_loader: FixedSubsetLoader,
    eval_loader: FixedSubsetLoader,
    calib_samples: int,
    eval_samples: int,
    seed: int,
    zero_point: int,
    topk: int,
) -> AttackResult:
    print("\n" + "=" * 70)
    print(f"R1_T06.1 REWRITE: Running {mode.upper()} ({mode_display})")
    print("=" * 70)

    bfa = SparseGatedDenseViewBFAExact(model=model, device=next(model.parameters()).device)
    criterion = nn.CrossEntropyLoss()

    if mode == "global":
        weight_filter = None
    elif mode == "zero_only":
        weight_filter = lambda v, zp=zero_point: v == zp
    elif mode == "nonzero_only":
        weight_filter = lambda v, zp=zero_point: v != zp
    else:
        raise ValueError(f"Unknown mode: {mode}")

    acc_calib_init, loss_calib_init, _ = bfa.evaluate(calib_loader, criterion, max_samples=calib_samples)
    acc_eval_init, loss_eval_init, _ = bfa.evaluate(eval_loader, criterion, max_samples=eval_samples)
    print(
        f"[R1_T06.1_REWRITE][MODE={mode}][SEED={seed}][STEP=0][FLIPS_USED=0] "
        f"L_eval={loss_eval_init:.6f} acc_eval={acc_eval_init:.2f}"
    )

    accuracy_history = [acc_eval_init]
    loss_eval_history = [loss_eval_init]
    loss_calib_history = [loss_calib_init]
    traces: List[StepTrace] = []
    acc_increase_steps: List[int] = []
    loss_decrease_steps: List[int] = []

    prev_acc = acc_eval_init
    prev_loss_eval = loss_eval_init
    prev_loss_calib = loss_calib_init

    wall_t0 = time.time()

    for step in range(1, physical_budget + 1):
        step_t0 = time.time()

        bfa.compute_gradients(calib_loader, calib_samples)
        candidates = bfa.enumerate_candidates(weight_filter=weight_filter)

        if not candidates:
            print(
                f"[R1_T06.1_REWRITE][MODE={mode}][SEED={seed}][STEP={step}] "
                f"No candidates - stopping"
            )
            break

        verify_t0 = time.time()
        best_candidate, exact_loss, topk_verified = bfa.exact_verify_topk_candidates(
            candidates=candidates,
            calib_loader=calib_loader,
            calib_samples=calib_samples,
            baseline_loss=prev_loss_calib,
            topk=topk,
        )
        verification_time = time.time() - verify_t0

        if best_candidate is None:
            print(
                f"[R1_T06.1_REWRITE][MODE={mode}][SEED={seed}][STEP={step}] "
                f"No valid candidate after verification - stopping"
            )
            break

        exact_score = exact_loss - prev_loss_calib
        score, layer_name, module, weight_idx, bit_pos, int8_val, delta_value = best_candidate

        if mode == "zero_only" and int8_val != zero_point:
            raise RuntimeError("zero_only selected non-zero candidate")
        if mode == "nonzero_only" and int8_val == zero_point:
            raise RuntimeError("nonzero_only selected zero candidate")

        old_int8, new_int8 = bfa.flip_bit(layer_name, module, weight_idx, bit_pos)

        _, loss_calib_new, _ = bfa.evaluate(calib_loader, criterion, max_samples=calib_samples)
        acc_eval_new, loss_eval_new, _ = bfa.evaluate(eval_loader, criterion, max_samples=eval_samples)

        delta_L_calib = loss_calib_new - prev_loss_calib
        delta_L_eval = loss_eval_new - prev_loss_eval
        delta_acc = acc_eval_new - prev_acc

        if delta_acc > 0:
            acc_increase_steps.append(step)
        if delta_L_eval < 0:
            loss_decrease_steps.append(step)

        traces.append(
            StepTrace(
                step=step,
                mode=mode,
                flips_used=step,
                cost=1,
                layer_name=layer_name,
                weight_idx=int(weight_idx),
                bit_pos=int(bit_pos),
                old_int8=old_int8,
                new_int8=new_int8,
                delta_value_fp32=float(delta_value),
                proxy_score=float(score),
                exact_score=float(exact_score),
                L_calib=float(loss_calib_new),
                delta_L_calib=float(delta_L_calib),
                L_eval=float(loss_eval_new),
                delta_L_eval=float(delta_L_eval),
                acc_eval=float(acc_eval_new),
                delta_acc_eval=float(delta_acc),
                topk_considered=int(topk_verified),
                verification_time=float(verification_time),
                objective=float(delta_L_eval),
            )
        )

        print(
            f"[R1_T06.1_REWRITE][MODE={mode}][SEED={seed}][STEP={step}][FLIPS_USED={step}] "
            f"L_eval={loss_eval_new:.6f} acc_eval={acc_eval_new:.2f} "
            f"exact_L={exact_loss:.4f} (Δ{exact_score:+.4f}) "
            f"topk={topk_verified}/{len(candidates)} verify_t={verification_time:.2f}s"
        )

        accuracy_history.append(acc_eval_new)
        loss_eval_history.append(loss_eval_new)
        loss_calib_history.append(loss_calib_new)

        prev_acc = acc_eval_new
        prev_loss_eval = loss_eval_new
        prev_loss_calib = loss_calib_new

        _ = time.time() - step_t0

        if acc_eval_new < 12.0:
            print(f"[R1_T06.1_REWRITE] Near random accuracy at step {step}. Stopping early.")
            break

    wall_time = time.time() - wall_t0

    return AttackResult(
        mode=mode,
        mode_display=mode_display,
        seed=seed,
        physical_budget=physical_budget,
        calib_samples=calib_samples,
        eval_samples=eval_samples,
        topk=topk,
        initial_accuracy=accuracy_history[0],
        initial_loss_eval=loss_eval_history[0],
        initial_loss_calib=loss_calib_history[0],
        final_accuracy=accuracy_history[-1],
        final_loss_eval=loss_eval_history[-1],
        final_loss_calib=loss_calib_history[-1],
        accuracy_history=accuracy_history,
        loss_eval_history=loss_eval_history,
        loss_calib_history=loss_calib_history,
        traces=traces,
        acc_increase_steps=acc_increase_steps,
        loss_decrease_steps=loss_decrease_steps,
        wall_time_sec=wall_time,
    )


# =============================================================================
# Output Helpers
# =============================================================================


def save_trace_csv(result: AttackResult, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "step",
                "mode",
                "flips_used",
                "cost",
                "layer_name",
                "weight_idx",
                "bit_pos",
                "old_int8",
                "new_int8",
                "delta_value_fp32",
                "proxy_score",
                "exact_score",
                "L_calib",
                "delta_L_calib",
                "L_eval",
                "delta_L_eval",
                "acc_eval",
                "delta_acc_eval",
                "topk_considered",
                "verification_time",
                "objective",
            ]
        )
        for t in result.traces:
            w.writerow(
                [
                    t.step,
                    t.mode,
                    t.flips_used,
                    t.cost,
                    t.layer_name,
                    t.weight_idx,
                    t.bit_pos,
                    t.old_int8,
                    t.new_int8,
                    f"{t.delta_value_fp32:.6f}",
                    f"{t.proxy_score:.6f}",
                    f"{t.exact_score:.6f}",
                    f"{t.L_calib:.6f}",
                    f"{t.delta_L_calib:.6f}",
                    f"{t.L_eval:.6f}",
                    f"{t.delta_L_eval:.6f}",
                    f"{t.acc_eval:.4f}",
                    f"{t.delta_acc_eval:.4f}",
                    t.topk_considered,
                    f"{t.verification_time:.4f}",
                    f"{t.objective:.6f}",
                ]
            )


def generate_curves(results: List[AttackResult], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(12, 7))
    for r in results:
        x = list(range(len(r.accuracy_history)))
        plt.plot(x, r.accuracy_history, marker="o", linewidth=2.5, markersize=5, label=f"{r.mode_display} (K={r.topk})")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlabel("Physical Flips", fontsize=14, fontweight="bold")
    plt.ylabel("Top-1 Accuracy (%)", fontsize=14, fontweight="bold")
    plt.title("R1_T06.1 REWRITE: Sparse-Gated BFA + Dense-View Search + Top-K Exact\nAccuracy vs Physical Flips", fontsize=16, fontweight="bold")
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.axhline(y=10, color="gray", linestyle=":", linewidth=1.5, alpha=0.5, label="Random (10%)")
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "R1_T06_1_acc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 7))
    for r in results:
        x = list(range(len(r.loss_eval_history)))
        plt.plot(x, r.loss_eval_history, marker="o", linewidth=2.5, markersize=5, label=f"{r.mode_display} (K={r.topk})")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlabel("Physical Flips", fontsize=14, fontweight="bold")
    plt.ylabel("Cross-Entropy Loss", fontsize=14, fontweight="bold")
    plt.title("R1_T06.1 REWRITE: Sparse-Gated BFA + Dense-View Search + Top-K Exact\nLoss vs Physical Flips", fontsize=16, fontweight="bold")
    plt.xlim(left=0)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "R1_T06_1_loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


def generate_summary_table(results: List[AttackResult], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "mode",
                "mode_display",
                "seed",
                "topk",
                "baseline_acc",
                "final_acc",
                "acc_drop",
                "baseline_loss_eval",
                "final_loss_eval",
                "loss_increase",
                "n_acc_increase_steps",
                "n_loss_decrease_steps",
                "acc_increase_steps",
                "loss_decrease_steps",
                "wall_time_sec",
            ]
        )

        for r in results:
            w.writerow(
                [
                    r.mode,
                    r.mode_display,
                    r.seed,
                    r.topk,
                    f"{r.initial_accuracy:.2f}%",
                    f"{r.final_accuracy:.2f}%",
                    f"{r.initial_accuracy - r.final_accuracy:.2f}%",
                    f"{r.initial_loss_eval:.6f}",
                    f"{r.final_loss_eval:.6f}",
                    f"{r.final_loss_eval - r.initial_loss_eval:.6f}",
                    len(r.acc_increase_steps),
                    len(r.loss_decrease_steps),
                    str(r.acc_increase_steps),
                    str(r.loss_decrease_steps),
                    f"{r.wall_time_sec:.2f}",
                ]
            )


# =============================================================================
# Model Loading
# =============================================================================


def load_int8_sparse_model(ckpt_path: str, device: str) -> Tuple[nn.Module, bool, Dict[str, torch.Tensor]]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    ckpt_to_check = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    is_int8_ckpt = any("int8_weights" in k for k in ckpt_to_check.keys())

    state_dict_to_load = checkpoint.get("model_state_dict", checkpoint)

    if is_int8_ckpt:
        base_model = create_resnet20(sparsity_type="2:4", pretrained_path=None).to(device)
        base_model.eval()

        for name, module in base_model.named_modules():
            mask_key = f"{name}.sparse_mask"
            if mask_key in state_dict_to_load and hasattr(module, "register_buffer"):
                mask_tensor = state_dict_to_load[mask_key]
                if hasattr(module, "cached_mask"):
                    del module.cached_mask
                module.register_buffer("cached_mask", mask_tensor)

        model = Int8QuantizedResNet(base_model, copy_sparse_masks=True).to(device)
        filtered = {k: v for k, v in state_dict_to_load.items() if "sparse_mask" not in k}
        model.load_state_dict(filtered, strict=False)
        model.calibrate_all_layers()
        model.eval()
    else:
        base_model = create_resnet20(sparsity_type="2:4", pretrained_path=ckpt_path).to(device)
        base_model.eval()
        base_model.freeze_sparse_masks()
        model = Int8QuantizedResNet(base_model, copy_sparse_masks=True).to(device)
        model.calibrate_all_layers()
        model.eval()

    return model, is_int8_ckpt, state_dict_to_load


def reload_model_for_mode(
    is_int8_ckpt: bool,
    state_dict_to_load: Dict[str, torch.Tensor],
    ckpt_path: str,
    device: str,
) -> nn.Module:
    if is_int8_ckpt:
        base_model = create_resnet20(sparsity_type="2:4", pretrained_path=None).to(device)
        base_model.eval()
        for name, module in base_model.named_modules():
            mask_key = f"{name}.sparse_mask"
            if mask_key in state_dict_to_load and hasattr(module, "register_buffer"):
                mask_tensor = state_dict_to_load[mask_key]
                if hasattr(module, "cached_mask"):
                    del module.cached_mask
                module.register_buffer("cached_mask", mask_tensor)

        model = Int8QuantizedResNet(base_model, copy_sparse_masks=True).to(device)
        filtered = {k: v for k, v in state_dict_to_load.items() if "sparse_mask" not in k}
        model.load_state_dict(filtered, strict=False)
        model.calibrate_all_layers()
        model.eval()
        return model

    base_model = create_resnet20(sparsity_type="2:4", pretrained_path=ckpt_path).to(device)
    base_model.eval()
    base_model.freeze_sparse_masks()
    model = Int8QuantizedResNet(base_model, copy_sparse_masks=True).to(device)
    model.calibrate_all_layers()
    model.eval()
    return model


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="R1_T06.1 REWRITE: sparse-gated exact attack with dense-view search")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to sparse INT8 checkpoint")
    parser.add_argument("--seed", type=int, default=0, nargs="+", help="Seed(s), e.g. --seed 0 42")
    parser.add_argument("--physical-budget", type=int, default=50)
    parser.add_argument("--calib-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--out-dir", type=str, default="results/R1/R1_T06_1_sparse_gated_dense_view_exact")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["global", "zero_only", "nonzero_only"],
        choices=["global", "zero_only", "nonzero_only"],
        help="Subset of modes to run",
    )
    args = parser.parse_args()

    seeds = [args.seed] if isinstance(args.seed, int) else list(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=False, transform=transform)

    mode_display = {
        "global": "sparse-gated : global weight-bit flip (dense-view search)",
        "zero_only": "sparse-gated : zero-only (dense-view search)",
        "nonzero_only": "sparse-gated : non-zero only (dense-view search)",
    }
    mode_task_id = {"global": 1, "zero_only": 2, "nonzero_only": 3}
    modes = [(m, mode_display[m]) for m in args.modes]

    all_results: List[AttackResult] = []

    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"[R1_T06.1_REWRITE] Running with SEED={seed}")
        print("=" * 80)

        set_all_seeds(seed)

        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            print("[Warning] CUDA not available, using CPU")

        eval_indices = generate_fixed_indices(test_dataset, args.eval_samples, seed)
        eval_indices_path = os.path.join(args.out_dir, f"eval_indices_seed{seed}.json")
        with open(eval_indices_path, "w") as f:
            json.dump(eval_indices, f)

        calib_indices = generate_fixed_indices(test_dataset, args.calib_samples, seed + 1)
        calib_indices_path = os.path.join(args.out_dir, f"calib_indices_seed{seed}.json")
        with open(calib_indices_path, "w") as f:
            json.dump(calib_indices, f)

        eval_loader = FixedSubsetLoader(test_dataset, eval_indices, batch_size=256)
        calib_loader = FixedSubsetLoader(test_dataset, calib_indices, batch_size=256)

        model_seed, is_int8_ckpt, state_dict_to_load = load_int8_sparse_model(args.ckpt, device)
        zero_point = detect_zero_point(model_seed)
        print(f"[R1_T06.1_REWRITE] Zero point: {zero_point}")
        del model_seed

        for mode, mode_display in modes:
            model = reload_model_for_mode(
                is_int8_ckpt=is_int8_ckpt,
                state_dict_to_load=state_dict_to_load,
                ckpt_path=args.ckpt,
                device=device,
            )

            mask_stats = get_sparse_mask_stats(model)
            print(
                "[R1_T06.1_REWRITE] Sparse mask stats before attack: "
                f"layers={int(mask_stats['layers'])}, total={int(mask_stats['total'])}, "
                f"ones={int(mask_stats['ones'])}, zeros={int(mask_stats['zeros'])}, "
                f"density={mask_stats['density']:.4f}"
            )
            if mask_stats["total"] > 0 and mask_stats["zeros"] <= 0:
                raise RuntimeError("Sparse mask became fully dense; this violates requested semantics")

            result = run_attack_mode_exact(
                model=model,
                mode=mode,
                mode_display=mode_display,
                physical_budget=args.physical_budget,
                calib_loader=calib_loader,
                eval_loader=eval_loader,
                calib_samples=args.calib_samples,
                eval_samples=args.eval_samples,
                seed=seed,
                zero_point=zero_point,
                topk=args.topk,
            )
            all_results.append(result)

            trace_path = os.path.join(args.out_dir, f"R1_T06_1_task{mode_task_id[mode]}_{mode}_trace.csv")
            save_trace_csv(result, trace_path)
            print(f"[Output] Saved trace CSV to {trace_path}")

    generate_curves(all_results, args.out_dir)
    summary_path = os.path.join(args.out_dir, "R1_T06_1_summary_table.csv")
    generate_summary_table(all_results, summary_path)
    print(f"[Output] Saved summary table to {summary_path}")

    log_path = os.path.join(args.out_dir, "R1_T06_1_run_log.txt")
    with open(log_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("R1_T06.1 REWRITE: Sparse-Gated BFA with Dense-View Candidate Search + Top-K Exact\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {now_ts()}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Checkpoint: {os.path.abspath(args.ckpt)}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Physical budget: {args.physical_budget}\n")
        f.write(f"Calib samples: {args.calib_samples}\n")
        f.write(f"Eval samples: {args.eval_samples}\n")
        f.write(f"Top-K: {args.topk}\n")
        f.write(f"Modes: {','.join(args.modes)}\n")
        f.write("\nSemantics:\n")
        f.write("- sparse_mask is preserved (sparse-gated forward)\n")
        f.write("- candidate generation uses dense-view global search\n")
        f.write("- exact verification naturally rejects ineffective masked candidates\n\n")
        f.write("Results:\n")
        for r in all_results:
            f.write(f"\n{r.mode_display} (K={r.topk})\n")
            f.write(f"  Baseline: {r.initial_accuracy:.2f}% acc, {r.initial_loss_eval:.6f} loss\n")
            f.write(f"  Final: {r.final_accuracy:.2f}% acc, {r.final_loss_eval:.6f} loss\n")
            f.write(f"  Acc increase steps: {r.acc_increase_steps}\n")
            f.write(f"  Loss decrease steps: {r.loss_decrease_steps}\n")

        f.write("\nReproduction command:\n")
        f.write(
            f"python run_R1_T06_1_sparse_gated_dense_view_exact.py "
            f"--device {args.device} --seed {' '.join(map(str, seeds))} "
            f"--physical-budget {args.physical_budget} --calib-samples {args.calib_samples} "
            f"--eval-samples {args.eval_samples} --topk {args.topk} "
            f"--modes {' '.join(args.modes)} "
            f"--ckpt {args.ckpt} --out-dir {args.out_dir}\n"
        )

    result_pkl = os.path.join(args.out_dir, "R1_T06_1_results.pkl")
    with open(result_pkl, "wb") as f:
        pickle.dump(all_results, f)

    print(f"[Output] Saved run log to {log_path}")
    print(f"[Output] Saved results pickle to {result_pkl}")
    print("\n[Summary] R1_T06.1_REWRITE completed:")
    for r in all_results:
        drop = r.initial_accuracy - r.final_accuracy
        print(f"  {r.mode_display} (K={r.topk}): {r.initial_accuracy:.2f}% -> {r.final_accuracy:.2f}% (drop: {drop:.2f}%)")


if __name__ == "__main__":
    main()
