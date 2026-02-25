#!/usr/bin/env python3
"""
Task 9: Collision Handling Characterization

Goal:
  Empirically measure collision semantics in the current repo stack.

This script runs two microbenches:
  A) CSR collision microbench (Task4-style CSR decode path):
     - Uses models/sparse_csr.py:CSRSparseConv2d for observed behavior.
     - Also tests a minimal CSR-linear decode path implemented locally (mirrors CSRSparseConv2d decode).
     - Injects duplicate csr_column_indices within a row and classifies behavior vs oracles:
         * merge-add (sum contributions)
         * mask-first (keep first occurrence)
         * drop (drop all colliding contributions)
       We also report mask-last (keep last occurrence) to capture the current implementation if applicable.

  B) 2:4-group collision microbench (Task5-style 2:4 groups with 2-bit indices):
     - Forces a collision by applying an "unsafe" move into an occupied index using the same
       update logic as bfa/csr_non_collision_attack.py (but without collision checks).
     - Classifies observed output vs the same oracles.

Outputs (fixed filenames):
  - results/task9_collision_characterization_result.pkl
  - results/task9_collision_characterization_log.txt
  - results/task9_collision_behavior_table.png

Constraints:
  - CPU-only
  - No extra deps beyond torch/torchvision/matplotlib/pickle
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.sparse_csr import CSRSparseConv2d
from train.ptq_convert import Int8QuantizedLinear


@dataclass
class CollisionCaseResult:
    case_id: str
    kind: str  # "csr_conv" | "csr_linear" | "nm_linear"
    seed: int
    baseline_cols: Tuple[int, int]
    collision_cols: Tuple[int, int]
    values: Tuple[int, int]  # int8 values
    scale: float
    x: Tuple[float, float, float, float]
    y_baseline: float
    y_collision: float
    oracle: Dict[str, float]
    behavior: str
    l2_to_oracles: Dict[str, float]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _oracle_outputs(
    dup_col: int,
    values: Tuple[int, int],
    scale: float,
    x: torch.Tensor,
) -> Dict[str, float]:
    v0, v1 = values
    x_c = float(x[dup_col].item())
    return {
        "merge_add": float((v0 + v1) * scale * x_c),
        "mask_first": float(v0 * scale * x_c),
        "mask_last": float(v1 * scale * x_c),
        "drop": 0.0,
    }


def _classify(y_obs: float, oracle: Dict[str, float], tol: float = 1e-6) -> Tuple[str, Dict[str, float]]:
    d = {}
    for k, y in oracle.items():
        d[k] = float(abs(y_obs - y))  # scalar L2
    exact = [k for k, dist in d.items() if dist <= tol]
    if exact:
        # If multiple exact matches (rare), pick deterministic order.
        for k in ("merge_add", "mask_first", "mask_last", "drop"):
            if k in exact:
                return k, d
        return exact[0], d
    # No exact match -> closest
    best = min(d.items(), key=lambda kv: kv[1])[0]
    return f"other_closest:{best}", d


def _run_csr_conv(
    values: Tuple[int, int],
    cols: Tuple[int, int],
    scale: float,
    x: torch.Tensor,
) -> float:
    # A minimal 1x1 conv, in_channels=4 -> in_features=4 so indices in [0,3]
    conv = CSRSparseConv2d(in_channels=4, out_channels=1, kernel_size=1, bias=False)
    conv.eval()
    with torch.no_grad():
        conv.weight.zero_()
        conv.scale.fill_(float(scale))
        conv.csr_values = torch.tensor(list(values), dtype=torch.int8)
        conv.csr_column_indices = torch.tensor(list(cols), dtype=torch.int16)
        conv.csr_row_ptr = torch.tensor([0, 2], dtype=torch.int32)
        conv.original_shape = conv.weight.shape
        conv.quantized = True
        y = conv(x.view(1, 4, 1, 1).float())
    return float(y.view(-1)[0].item())


def _run_csr_linear_local_decode(
    values: Tuple[int, int],
    cols: Tuple[int, int],
    scale: float,
    x: torch.Tensor,
) -> float:
    # Local CSR decode consistent with models/sparse_csr.py: assignment semantics (last wins).
    w = torch.zeros(4, dtype=torch.float32)
    for v, c in zip(values, cols):
        w[int(c)] = float(v) * float(scale)
    y = float(torch.dot(w, x.float()).item())
    return y


def _make_nm_linear_model(
    values: Tuple[int, int],
    active: Tuple[int, int],
    scale: float,
) -> Int8QuantizedLinear:
    # Int8QuantizedLinear expects fp32 weight param but uses int8_weights when quantized=True.
    mask = torch.zeros(1, 4, dtype=torch.float32)
    mask[0, active[0]] = 1.0
    mask[0, active[1]] = 1.0
    mod = Int8QuantizedLinear(4, 1, bias=False, sparse_mask=mask)
    mod.eval()
    with torch.no_grad():
        mod.weight.zero_()
        mod.scale.fill_(float(scale))
        mod.int8_weights.zero_()
        mod.int8_weights[0, active[0]] = int(values[0])
        mod.int8_weights[0, active[1]] = int(values[1])
        mod.quantized = True
    return mod


def _unsafe_nm_collision_move_inplace(
    mod: Int8QuantizedLinear,
    old_idx: int,
    new_idx: int,
) -> None:
    """
    Force a collision by moving one active value into the other active index.
    Mirrors bfa/csr_non_collision_attack._apply_non_collision_move() update rule,
    but intentionally skips collision checks.
    """
    with torch.no_grad():
        w = mod.int8_weights  # [1,4]
        m = mod.sparse_mask   # [1,4]
        old_val = int(w[0, old_idx].item())
        w[0, new_idx] = torch.tensor(old_val, dtype=w.dtype)
        w[0, old_idx] = torch.tensor(0, dtype=w.dtype)

        # Toggle bits (collision causes both to toggle off if new_idx was already active).
        m[0, old_idx] = torch.tensor(float(int(m[0, old_idx].item()) ^ 1), dtype=m.dtype)
        m[0, new_idx] = torch.tensor(float(int(m[0, new_idx].item()) ^ 1), dtype=m.dtype)


def _run_nm_linear(
    values: Tuple[int, int],
    active: Tuple[int, int],
    scale: float,
    x: torch.Tensor,
    force_collision: bool,
) -> float:
    mod = _make_nm_linear_model(values=values, active=active, scale=scale)
    if force_collision:
        _unsafe_nm_collision_move_inplace(mod, old_idx=active[0], new_idx=active[1])
    with torch.no_grad():
        y = mod(x.view(1, 4).float())
    return float(y.view(-1)[0].item())


def _save_behavior_table_png(
    out_path: str,
    rows: List[str],
    behaviors: List[str],
    table_counts: List[List[int]],
    title: str,
) -> None:
    # Plot as annotated heatmap (counts).
    fig = plt.figure(figsize=(12, max(4, 0.45 * len(rows))))
    ax = fig.add_subplot(1, 1, 1)
    mat = torch.tensor(table_counts, dtype=torch.float32)
    im = ax.imshow(mat.numpy(), aspect="auto", cmap="Blues")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Observed Behavior")
    ax.set_ylabel("Pattern")
    ax.set_xticks(list(range(len(behaviors))))
    ax.set_xticklabels(behaviors, rotation=30, ha="right")
    ax.set_yticks(list(range(len(rows))))
    ax.set_yticklabels(rows)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{int(mat[i, j].item())}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 9: Collision Handling Characterization")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds to sweep (seed, seed+1, ...)")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--unsafe-collision", action="store_true",
                        help="Enable unsafe 2:4 collision injection for measurement (does not affect Task5).")
    args = parser.parse_args()

    device = "cpu"
    os.makedirs("results", exist_ok=True)

    log_path = "results/task9_collision_characterization_log.txt"
    out_pkl = "results/task9_collision_characterization_result.pkl"
    out_png = "results/task9_collision_behavior_table.png"

    seeds = [args.seed + i for i in range(args.num_seeds)]

    # Patterns focus on a single duplicate column, but vary the order of values to disambiguate first/last behavior.
    patterns = [
        ("dup_col0", (0, 2), (0, 0)),
        ("dup_col1", (1, 3), (1, 1)),
        ("dup_col2", (2, 3), (2, 2)),
        ("dup_col3", (0, 3), (3, 3)),
    ]

    behaviors_order = ["merge_add", "mask_first", "mask_last", "drop", "other"]

    all_results: List[CollisionCaseResult] = []
    freq: Dict[str, Dict[str, int]] = {
        "csr_conv": {b: 0 for b in behaviors_order},
        "csr_linear": {b: 0 for b in behaviors_order},
        "nm_linear": {b: 0 for b in behaviors_order},
    }
    # Per-pattern truth-table counts: (kind, pattern_id) -> behavior bucket counts
    per_pattern: Dict[Tuple[str, str], Dict[str, int]] = {}
    for kind in ("csr_conv", "csr_linear", "nm_linear"):
        for pat_id, _, _ in patterns:
            per_pattern[(kind, pat_id)] = {b: 0 for b in behaviors_order}

    with open(log_path, "w", encoding="utf-8") as log:
        log.write("=" * 100 + "\n")
        log.write("Task 9: Collision Handling Characterization\n")
        log.write("=" * 100 + "\n")
        log.write(f"Script: run_task9_collision_characterization.py\n")
        log.write(f"Timestamp: {_now_ts()}\n")
        log.write(f"Seed(base): {args.seed}\n")
        log.write(f"Seed sweep: {seeds}\n")
        log.write(f"Device: {device}\n")
        log.write("Dataset path: N/A (microbench)\n")
        log.write("Model checkpoint path: N/A (microbench)\n")
        log.write(f"Tolerance (exact match): {args.tol}\n")
        log.write(f"Unsafe 2:4 collision enabled: {bool(args.unsafe_collision)}\n")
        log.write("\nAssumptions:\n")
        log.write("- CSR decode path uses dense reconstruction followed by standard conv/linear.\n")
        log.write("- 2:4 group 'collision' is simulated by forcing a move into an occupied slot using Task5 update rules.\n")
        log.write("\n")

        for seed in seeds:
            _set_seed(seed)

            # Deterministic scale and input.
            scale = 0.5
            # Input: keep all non-zero to ensure signal; still deterministic per seed.
            x = torch.tensor([random.uniform(-1.0, 1.0) for _ in range(4)], dtype=torch.float32)
            # Ensure no all-zero.
            if float(x.abs().sum().item()) < 1e-6:
                x[0] = 1.0

            for pat_id, baseline_cols, collision_cols in patterns:
                dup_col = collision_cols[0]

                # Choose two distinct int8 values to disambiguate first/last.
                v0 = random.choice([5, 11, 23, 37, 61, -7, -19, -41])
                v1 = random.choice([3, 13, 29, 43, 79, -5, -17, -59])
                if v1 == v0:
                    v1 = -v0
                values = (int(v0), int(v1))

                # --- CSR conv ---
                y_b = _run_csr_conv(values=values, cols=baseline_cols, scale=scale, x=x)
                y_c = _run_csr_conv(values=values, cols=collision_cols, scale=scale, x=x)
                oracle = _oracle_outputs(dup_col=dup_col, values=values, scale=scale, x=x)
                behavior, d = _classify(y_c, oracle, tol=args.tol)
                bucket = behavior if behavior in behaviors_order else ("other" if behavior.startswith("other_") else "other")
                freq["csr_conv"][bucket] += 1
                per_pattern[("csr_conv", pat_id)][bucket] += 1
                all_results.append(CollisionCaseResult(
                    case_id=f"csr_conv:{pat_id}",
                    kind="csr_conv",
                    seed=seed,
                    baseline_cols=baseline_cols,
                    collision_cols=collision_cols,
                    values=values,
                    scale=scale,
                    x=tuple(float(v) for v in x.tolist()),
                    y_baseline=y_b,
                    y_collision=y_c,
                    oracle=oracle,
                    behavior=behavior,
                    l2_to_oracles=d,
                ))

                # --- CSR linear (local decode) ---
                y_b = _run_csr_linear_local_decode(values=values, cols=baseline_cols, scale=scale, x=x)
                y_c = _run_csr_linear_local_decode(values=values, cols=collision_cols, scale=scale, x=x)
                oracle = _oracle_outputs(dup_col=dup_col, values=values, scale=scale, x=x)
                behavior, d = _classify(y_c, oracle, tol=args.tol)
                bucket = behavior if behavior in behaviors_order else ("other" if behavior.startswith("other_") else "other")
                freq["csr_linear"][bucket] += 1
                per_pattern[("csr_linear", pat_id)][bucket] += 1
                all_results.append(CollisionCaseResult(
                    case_id=f"csr_linear:{pat_id}",
                    kind="csr_linear",
                    seed=seed,
                    baseline_cols=baseline_cols,
                    collision_cols=collision_cols,
                    values=values,
                    scale=scale,
                    x=tuple(float(v) for v in x.tolist()),
                    y_baseline=y_b,
                    y_collision=y_c,
                    oracle=oracle,
                    behavior=behavior,
                    l2_to_oracles=d,
                ))

                # --- 2:4 group collision (linear) ---
                # Baseline uses the baseline_cols as the two active indices (must be distinct).
                active = baseline_cols
                y_b = _run_nm_linear(values=values, active=active, scale=scale, x=x, force_collision=False)
                if args.unsafe_collision:
                    y_c = _run_nm_linear(values=values, active=active, scale=scale, x=x, force_collision=True)
                else:
                    y_c = float("nan")

                oracle = _oracle_outputs(dup_col=active[1], values=values, scale=scale, x=x)
                if args.unsafe_collision:
                    behavior, d = _classify(y_c, oracle, tol=args.tol)
                    bucket = behavior if behavior in behaviors_order else ("other" if behavior.startswith("other_") else "other")
                    freq["nm_linear"][bucket] += 1
                    per_pattern[("nm_linear", pat_id)][bucket] += 1
                else:
                    behavior, d = ("other", {k: float("nan") for k in oracle})
                all_results.append(CollisionCaseResult(
                    case_id=f"nm_linear:{pat_id}",
                    kind="nm_linear",
                    seed=seed,
                    baseline_cols=baseline_cols,
                    collision_cols=collision_cols,
                    values=values,
                    scale=scale,
                    x=tuple(float(v) for v in x.tolist()),
                    y_baseline=y_b,
                    y_collision=y_c,
                    oracle=oracle,
                    behavior=behavior,
                    l2_to_oracles=d,
                ))

        log.write("== Behavior Frequencies (counts) ==\n")
        for kind in ("csr_conv", "csr_linear", "nm_linear"):
            log.write(f"\n[{kind}]\n")
            total = sum(freq[kind].values())
            for b in behaviors_order:
                log.write(f"- {b:10s}: {freq[kind][b]:6d} / {total:6d}\n")

        log.write("\nHow to run:\n")
        log.write("  python run_task9_collision_characterization.py --unsafe-collision\n")

    # Save raw measurements
    result_obj = {
        "timestamp": _now_ts(),
        "seed_base": args.seed,
        "seeds": seeds,
        "tolerance": args.tol,
        "unsafe_collision_enabled": bool(args.unsafe_collision),
        "freq": freq,
        "per_pattern_freq": per_pattern,
        "results": [r.__dict__ for r in all_results],
    }
    with open(out_pkl, "wb") as f:
        pickle.dump(result_obj, f)

    # Build behavior table plot
    rows: List[str] = []
    table_counts: List[List[int]] = []
    for kind in ("csr_conv", "csr_linear", "nm_linear"):
        for pat_id, baseline_cols, collision_cols in patterns:
            # Keep row label compact but explicit.
            rows.append(f"{kind}:{pat_id} {baseline_cols}->{collision_cols}")
            table_counts.append([per_pattern[(kind, pat_id)][b] for b in behaviors_order])
    _save_behavior_table_png(
        out_path=out_png,
        rows=rows,
        behaviors=behaviors_order,
        table_counts=table_counts,
        title="Task 9: Collision Behavior Truth Table (pattern -> behavior)",
    )

    print(f"[Task9] Wrote: {out_pkl}")
    print(f"[Task9] Wrote: {log_path}")
    print(f"[Task9] Wrote: {out_png}")


if __name__ == "__main__":
    main()
