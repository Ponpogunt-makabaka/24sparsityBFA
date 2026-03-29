#!/usr/bin/env python3
import argparse
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


AFTER_RE = re.compile(r"After flip:\s*([0-9]+\.?[0-9]*)%")
BEFORE_RE = re.compile(r"INT8 before:\s*(-?\d+)")


def parse_log(log_path):
    after_vals = []
    before_vals = []
    last_before = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m_before = BEFORE_RE.search(line)
            if m_before:
                last_before = int(m_before.group(1))
            m_after = AFTER_RE.search(line)
            if m_after:
                after_vals.append(float(m_after.group(1)))
                before_vals.append(last_before)
                last_before = None
    return after_vals, before_vals


def main():
    parser = argparse.ArgumentParser(description="Plot three BFA curves in one figure.")
    parser.add_argument("--dense_log", required=True)
    parser.add_argument("--zero_log", required=True)
    parser.add_argument("--nonzero_log", required=True)
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dense_after, dense_before = parse_log(args.dense_log)
    zero_after, _ = parse_log(args.zero_log)
    nonzero_after, _ = parse_log(args.nonzero_log)

    plt.figure(figsize=(10, 6))

    # 1) combine attack (dense): line + markers by INT8-before state
    xd = list(range(1, len(dense_after) + 1))
    plt.plot(xd, dense_after, color="C0", linestyle="-", linewidth=1.8, label="Combine attack (dense)")
    dense_zero_x = [x for i, x in enumerate(xd) if i < len(dense_before) and dense_before[i] == 0]
    dense_zero_y = [dense_after[i - 1] for i in dense_zero_x]
    dense_nonzero_x = [x for i, x in enumerate(xd) if i < len(dense_before) and dense_before[i] != 0]
    dense_nonzero_y = [dense_after[i - 1] for i in dense_nonzero_x]
    if dense_nonzero_x:
        plt.scatter(dense_nonzero_x, dense_nonzero_y, c="C0", s=52, marker="o", label="Combine: non-zero chosen")
    if dense_zero_x:
        plt.scatter(dense_zero_x, dense_zero_y, c="red", s=95, marker="*", label="Combine: zero chosen")

    # 2) zero-only attack
    xz = list(range(1, len(zero_after) + 1))
    if xz:
        plt.plot(xz, zero_after, color="C3", linestyle="-", marker="*", markersize=8, linewidth=1.5, label="Zero-only attack")

    # 3) non-zero-only attack
    xn = list(range(1, len(nonzero_after) + 1))
    if xn:
        plt.plot(xn, nonzero_after, color="C2", linestyle="-", marker="o", markersize=5.5, linewidth=1.5, label="Non-zero-only attack")

    plt.xlabel("Iteration")
    plt.ylabel("Top-1 Accuracy After Flip (%)")
    plt.title(f"After-Flip Top-1 vs Iteration (dense, n={args.n_iter}, k={args.topk})")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.set_ylim(0, 100)
    ax.set_xlim(1, args.n_iter)
    plt.grid(True, alpha=0.6)
    plt.legend(loc="best", fontsize=9)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    print(f"Wrote plot: {args.out}")


if __name__ == "__main__":
    main()
