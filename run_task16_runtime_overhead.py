#!/usr/bin/env python3
"""
Task 16: Runtime/overhead characterization (attack + defense compute cost).
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, List, Optional

import torch

import run_task11_metadata_defense as t11
from bfa.csr_non_collision_attack import _apply_non_collision_move, _compute_candidates  # noqa: SLF001
from scripts.p012_17_utils import load_cifar10_loaders_offline, now_ts, set_all_seeds


def _profile_one(
    defense: str,  # none|parity|crc
    args,
    test_loader,
    calib_loader,
    device: str,
) -> Dict:
    set_all_seeds(args.seed)
    model = t11._load_sparse_int8_resnet20(device=device, ckpt_path=args.ckpt)  # noqa: SLF001
    model.eval()

    layers = t11._get_sparse_layers(model)  # noqa: SLF001
    integrity: Dict[str, t11.IntegrityState] = {}
    if defense in ("parity", "crc"):
        for name, module in layers:
            integrity[name] = t11._build_integrity_state_for_layer(module=module, line_bytes=args.line_bytes)  # noqa: SLF001

    flipped = set()
    counters = {"attempted": 0, "collisions": 0}
    search_ms: List[float] = []
    apply_ms: List[float] = []
    defense_ms: List[float] = []
    executed = 0

    wall_t0 = time.perf_counter()
    for _ in range(args.max_flips):
        t0 = time.perf_counter()
        candidates = _compute_candidates(
            model=model,
            calib_loader=calib_loader,
            device=device,
            calib_samples=args.calib_samples,
            flipped=flipped,
            counters=counters,
            max_groups_per_layer=args.max_groups_per_layer,
            allow_nonpositive=bool(args.allow_nonpositive),
        )
        search_ms.append((time.perf_counter() - t0) * 1000.0)
        if not candidates:
            break

        score, layer_name, module, g_idx, old_idx, new_idx = candidates[0]
        bit_pos = (old_idx ^ new_idx).bit_length() - 1
        flipped.add((layer_name, int(g_idx), int(old_idx), int(bit_pos)))

        t1 = time.perf_counter()
        ok = _apply_non_collision_move(module, int(g_idx), int(old_idx), int(new_idx))
        apply_ms.append((time.perf_counter() - t1) * 1000.0)
        if not ok:
            defense_ms.append(0.0)
            continue

        t2 = time.perf_counter()
        if defense in ("parity", "crc"):
            state = integrity.get(layer_name)
            if state is not None:
                w_flat, _ = t11._flatten_groups(module.int8_weights)  # noqa: SLF001
                m_flat, _ = t11._flatten_groups(module.sparse_mask)  # noqa: SLF001
                assert w_flat is not None and m_flat is not None
                if defense == "parity":
                    cur_p = t11._compute_current_parity_for_group(m_flat, int(g_idx))  # noqa: SLF001
                    exp_p = int(state.good_parity[int(g_idx)].item())
                    if cur_p != exp_p:
                        if args.mitigation == "revert":
                            t11._restore_group_from_good(w_flat, m_flat, state, int(g_idx))  # noqa: SLF001
                        else:
                            t11._drop_group_to_zero_and_commit(w_flat, m_flat, state, int(g_idx))  # noqa: SLF001
                    else:
                        t11._commit_current_group_as_good(w_flat, m_flat, state, int(g_idx))  # noqa: SLF001
                else:
                    line_idx = int(int(g_idx) // state.groups_per_line)
                    cur_crc = t11._compute_current_crc_for_line(m_flat, line_idx, state.groups_per_line)  # noqa: SLF001
                    exp_crc = int(state.good_crc[line_idx].item()) if line_idx < int(state.good_crc.numel()) else int(cur_crc)
                    if int(cur_crc) != int(exp_crc):
                        if args.mitigation == "revert":
                            t11._restore_line_from_good(w_flat, m_flat, state, line_idx)  # noqa: SLF001
                        else:
                            t11._drop_line_to_zero_and_commit(w_flat, m_flat, state, line_idx)  # noqa: SLF001
                    else:
                        t11._commit_current_line_as_good(w_flat, m_flat, state, line_idx)  # noqa: SLF001

                with torch.no_grad():
                    module.int8_weights.copy_(t11._restore_groups(w_flat, state.w_meta).clone())  # noqa: SLF001
                    module.sparse_mask.copy_(t11._restore_groups(m_flat, state.m_meta).clone())  # noqa: SLF001
        defense_ms.append((time.perf_counter() - t2) * 1000.0)
        executed += 1

    wall_sec = time.perf_counter() - wall_t0

    def _avg(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    avg_search = _avg(search_ms)
    avg_apply = _avg(apply_ms)
    avg_defense = _avg(defense_ms)
    avg_total = avg_search + avg_apply + avg_defense
    throughput = (1000.0 / avg_total) if avg_total > 0 else 0.0
    return {
        "defense": defense,
        "executed_flips": int(executed),
        "attempted_candidates": int(counters["attempted"]),
        "collisions_skipped": int(counters["collisions"]),
        "avg_search_ms": avg_search,
        "avg_apply_ms": avg_apply,
        "avg_defense_ms": avg_defense,
        "avg_total_ms": avg_total,
        "throughput_attempts_per_sec": throughput,
        "wall_sec": float(wall_sec),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 16: Runtime/overhead characterization")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--ckpt", type=str, default="models/sparse_model.pth")
    parser.add_argument("--max-flips", type=int, default=50)
    parser.add_argument("--calib-samples", type=int, default=256)
    parser.add_argument("--max-groups-per-layer", type=int, default=2000)
    parser.add_argument("--allow-nonpositive", action="store_true")
    parser.add_argument("--line-bytes", type=int, default=64)
    parser.add_argument("--mitigation", choices=["revert", "drop"], default="revert")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    out_csv = "results/task16_runtime_overhead_table.csv"
    out_log = "results/task16_runtime_overhead_log.txt"

    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    set_all_seeds(args.seed)
    _, test_loader = load_cifar10_loaders_offline(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=0,
    )
    calib_loader = test_loader

    prof_none = _profile_one("none", args, test_loader, calib_loader, device)
    prof_parity = _profile_one("parity", args, test_loader, calib_loader, device)
    prof_crc = _profile_one("crc", args, test_loader, calib_loader, device)
    rows = [prof_none, prof_parity, prof_crc]

    base = prof_none["avg_total_ms"] if prof_none["avg_total_ms"] > 0 else 1.0
    for r in rows:
        r["throughput_impact_pct_vs_none"] = 100.0 * (r["avg_total_ms"] - base) / base

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "defense",
                "executed_flips",
                "attempted_candidates",
                "collisions_skipped",
                "avg_search_ms",
                "avg_apply_ms",
                "avg_defense_ms",
                "avg_total_ms",
                "throughput_attempts_per_sec",
                "throughput_impact_pct_vs_none",
                "wall_sec",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "defense": r["defense"],
                "executed_flips": r["executed_flips"],
                "attempted_candidates": r["attempted_candidates"],
                "collisions_skipped": r["collisions_skipped"],
                "avg_search_ms": f"{r['avg_search_ms']:.6f}",
                "avg_apply_ms": f"{r['avg_apply_ms']:.6f}",
                "avg_defense_ms": f"{r['avg_defense_ms']:.6f}",
                "avg_total_ms": f"{r['avg_total_ms']:.6f}",
                "throughput_attempts_per_sec": f"{r['throughput_attempts_per_sec']:.6f}",
                "throughput_impact_pct_vs_none": f"{r['throughput_impact_pct_vs_none']:.6f}",
                "wall_sec": f"{r['wall_sec']:.6f}",
            })

    with open(out_log, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("Task 16: Runtime/Overhead Characterization\n")
        f.write("=" * 100 + "\n")
        f.write("Script: run_task16_runtime_overhead.py\n")
        f.write(f"Timestamp: {now_ts()}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Dataset path: {os.path.abspath(args.data_dir)}\n")
        f.write(f"Model checkpoint path: {os.path.abspath(args.ckpt)}\n")
        f.write("\nConfig:\n")
        for k, v in vars(args).items():
            f.write(f"- {k}: {v}\n")
        f.write("\nProfile summary:\n")
        for r in rows:
            f.write(
                f"- {r['defense']}: search={r['avg_search_ms']:.3f}ms apply={r['avg_apply_ms']:.3f}ms "
                f"defense={r['avg_defense_ms']:.3f}ms total={r['avg_total_ms']:.3f}ms "
                f"thr={r['throughput_attempts_per_sec']:.2f}/s impact={r['throughput_impact_pct_vs_none']:.2f}%\n"
            )
        f.write("\nOutputs:\n")
        f.write(f"- {out_csv}\n")
        f.write(f"- {out_log}\n")

    print(f"[Task16] Wrote: {out_csv}")
    print(f"[Task16] Wrote: {out_log}")


if __name__ == "__main__":
    main()

