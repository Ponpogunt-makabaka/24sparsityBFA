#!/usr/bin/env python3
"""
Task 7: Sparse INT8 MobileNet-V2 (ImageNet) - CSR Index Non-Collision Attack
Pointwise (1x1) conv + linear only. Depthwise conv excluded.
"""
import argparse
import os
import pickle

import torch
import matplotlib.pyplot as plt

from torchvision.models import mobilenet_v2

from models.factory import _compute_2_4_mask_conv, _compute_2_4_mask_linear, _load_local_weights
from train.imagenet_utils import get_imagenet_loader
from train.imagenet_pipeline_utils import (
    apply_sparse_mask,
    replace_with_int8_from_mask,
    calibrate_int8,
    run_bn_recalibration,
    evaluate,
)
from bfa.csr_non_collision_attack import run_csr_non_collision_attack


def main():
    parser = argparse.ArgumentParser(description="Task 7: MobileNet-V2 CSR Non-Collision Attack")
    parser.add_argument("--data-root", type=str, required=True, help="ImageNet root path")
    parser.add_argument("--subset-root", type=str, default=None, help="Optional ImageNet subset root")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--max-flips", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--weights-path", type=str, default=None,
                        help="Local MobileNet-V2 weights path (offline load)")
    parser.add_argument("--imagenette-root", type=str, default=None,
                        help="Local Imagenette root path (used if ImageNet is unavailable)")
    parser.add_argument("--max-groups-per-layer", type=int, default=2000,
                        help="Optional cap on groups sampled per layer for candidate search")
    parser.add_argument("--eval-samples", type=int, default=256,
                        help="Optional cap on evaluation samples per accuracy check")
    parser.add_argument("--allow-nonpositive", action="store_true",
                        help="Allow non-positive score candidates when positives are scarce")
    parser.add_argument("--bn-calib-steps", type=int, default=300)
    parser.add_argument("--bn-lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs("./results", exist_ok=True)

    batch_size = args.batch_size
    num_workers = 0 if args.num_workers > 0 else args.num_workers

    val_loader = get_imagenet_loader(
        root=args.data_root,
        split="val",
        batch_size=batch_size,
        num_workers=num_workers,
        subset_root=args.subset_root,
        shuffle=False,
        imagenette_root=args.imagenette_root
    )

    calib_loader = get_imagenet_loader(
        root=args.data_root,
        split="val",
        batch_size=batch_size,
        num_workers=num_workers,
        subset_root=args.subset_root,
        shuffle=True,
        imagenette_root=args.imagenette_root
    )

    train_loader = get_imagenet_loader(
        root=args.data_root,
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        subset_root=args.subset_root,
        shuffle=True,
        imagenette_root=args.imagenette_root
    )

    # Load Dense FP32
    model = mobilenet_v2(weights=None).to(device)
    if args.weights_path:
        _load_local_weights(model, args.weights_path, strict=True)
    model.eval()

    def is_pointwise_conv(mod) -> bool:
        return mod.kernel_size == (1, 1) and mod.groups == 1

    def conv_filter(mod, name: str) -> bool:
        if not is_pointwise_conv(mod):
            return False
        return mod.in_channels % 4 == 0

    def linear_filter(mod, name: str) -> bool:
        return True

    mask_map = apply_sparse_mask(
        model,
        compute_conv_mask=_compute_2_4_mask_conv,
        compute_linear_mask=_compute_2_4_mask_linear,
        conv_filter=conv_filter,
        linear_filter=linear_filter
    )

    bn_steps = run_bn_recalibration(
        model,
        train_loader=train_loader,
        device=device,
        max_steps=args.bn_calib_steps,
        lr=args.bn_lr
    )
    model.eval()
    post_bn_acc = evaluate(model, val_loader, device, max_samples=args.eval_samples)

    replace_with_int8_from_mask(model, mask_map)
    calibrate_int8(model)
    model.eval()

    result = run_csr_non_collision_attack(
        model=model,
        test_loader=val_loader,
        calib_loader=calib_loader,
        device=device,
        max_success=args.max_flips,
        calib_samples=args.calib_samples,
        log_interval=args.log_interval,
        max_groups_per_layer=args.max_groups_per_layer,
        eval_samples=args.eval_samples,
        allow_nonpositive=args.allow_nonpositive
    )

    # Save result
    result_path = "./results/task7_mobilenetv2_csr_non_collision_result.pkl"
    log_path = "./results/task7_mobilenetv2_csr_non_collision_log.txt"
    plot_path = "./results/task7_mobilenetv2_csr_non_collision.png"

    with open(result_path, "wb") as f:
        pickle.dump(result, f)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("Task 7: MobileNet-V2 CSR Non-Collision Attack\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Weights Path: {args.weights_path}\n")
        f.write(f"Imagenette Root: {args.imagenette_root}\n")
        f.write(f"BN Calibration Steps: {bn_steps}\n")
        f.write(f"BN Calibration LR: {args.bn_lr}\n")
        f.write(f"Post-BN Sparse FP32 Accuracy: {post_bn_acc:.2f}%\n")
        f.write(f"Eval Samples per Check: {args.eval_samples}\n")
        f.write(f"Max Groups per Layer (sampled): {args.max_groups_per_layer}\n")
        f.write(f"Allow Non-Positive Scores: {args.allow_nonpositive}\n")
        f.write(f"Initial Accuracy: {result.initial_accuracy:.2f}%\n")
        f.write(f"Final Accuracy: {result.final_accuracy:.2f}%\n")
        f.write(f"Successful Non-Collision Flips Executed: {result.total_flips}\n")
        f.write(f"Total Flips Attempted: {result.attempted_flips}\n")
        f.write(f"Collisions Avoided (Skipped): {result.collisions_skipped}\n")
        f.write(f"Accuracy Drop: {result.initial_accuracy - result.final_accuracy:.2f}%\n\n")

        f.write("-" * 100 + "\n")
        f.write(f"{'Flips':>8} | {'Accuracy':>10} | {'Loss':>10} | {'Move (layer:g:old->new)'}\n")
        f.write("-" * 100 + "\n")
        for entry in result.flip_log:
            flips = entry["flips"]
            acc = entry["accuracy"]
            loss = entry["loss"]
            move = entry["move"]
            if move is None:
                move_str = "(initial state)"
            else:
                layer_short = move["layer"].split(".")[-1]
                move_str = f"{layer_short}:g{move['group']}:{move['old_idx']}->{move['new_idx']}"
            f.write(f"{flips:8d} | {acc:9.2f}% | {loss:10.4f} | {move_str}\n")

    # Plot
    plt.figure(figsize=(10, 6))
    flips = list(range(len(result.accuracy_history)))
    plt.plot(flips, result.accuracy_history, marker='o', linewidth=2, markersize=5,
             label='MobileNet-V2 CSR Non-Collision')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('Successful Flips', fontsize=12, fontweight='bold')
    plt.ylabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Task 7: MobileNet-V2 CSR Index Attack (Non-Collision)', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[Task7] Results saved to {result_path}")
    print(f"[Task7] Detailed log saved to {log_path}")
    print(f"[Task7] Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
