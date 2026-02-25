#!/usr/bin/env python3
"""
Plot combined CSR non-collision attack curves for ImageNet expansion tasks (6-8).

Inputs (pickle):
- results/task6_resnet18_csr_non_collision_result.pkl
- results/task7_mobilenetv2_csr_non_collision_result.pkl
- results/task8_deit_tiny_csr_non_collision_result.pkl

Output:
- results/image_net_attack_curves.png
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt


def _load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    series = []
    for label, path in [
        ("ResNet-18", "results/task6_resnet18_csr_non_collision_result.pkl"),
        ("MobileNet-V2", "results/task7_mobilenetv2_csr_non_collision_result.pkl"),
        ("DeiT-Tiny", "results/task8_deit_tiny_csr_non_collision_result.pkl"),
    ]:
        if not os.path.exists(path):
            continue
        r = _load(path)
        series.append((label, r.accuracy_history))

    if not series:
        raise FileNotFoundError("No result pickles found under ./results for tasks 6-8.")

    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 6))

    for label, acc in series:
        flips = list(range(len(acc)))
        plt.plot(flips, acc, marker="o", linewidth=2, markersize=4, label=label)

    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlabel("Successful Flips", fontsize=12, fontweight="bold")
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title("CSR Index Attack (Non-Collision): ImageNet Expansion (Tasks 6-8)",
              fontsize=14, fontweight="bold")
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.legend(loc="best")
    plt.tight_layout()
    out_path = "results/image_net_attack_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved to {out_path}")


if __name__ == "__main__":
    main()
