#!/usr/bin/env python3
"""
Task 8: Sparse INT8 DeiT-Tiny (ImageNet) - CSR Index Non-Collision Attack
Targets Linear layers with 2:4 mask along in_features.
"""
import argparse
import os
import pickle

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models.factory import create_imagenet_deit_tiny_fp32, _compute_2_4_mask_linear
from train.imagenet_utils import get_imagenet_loader
from train.imagenet_pipeline_utils import apply_sparse_mask, replace_with_int8_from_mask, calibrate_int8, evaluate
from train.ptq_convert import Int8QuantizedMultiheadAttention
from bfa.csr_non_collision_attack import run_csr_non_collision_attack


def _replace_mha_with_int8_qkv_from_map(module: nn.Module, mask_map: dict[str, torch.Tensor], prefix: str = ""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.MultiheadAttention):
            mask = mask_map.get(full_name)
            new_attn = Int8QuantizedMultiheadAttention(
                embed_dim=child.embed_dim,
                num_heads=child.num_heads,
                dropout=child.dropout,
                bias=child.in_proj_bias is not None,
                add_bias_kv=False,
                add_zero_attn=child.add_zero_attn,
                kdim=child.kdim,
                vdim=child.vdim,
                batch_first=child.batch_first,
                sparse_mask=mask
            )
            new_attn.in_proj_weight.data.copy_(child.in_proj_weight.data)
            if child.in_proj_bias is not None and new_attn.in_proj_bias is not None:
                new_attn.in_proj_bias.data.copy_(child.in_proj_bias.data)
            new_attn.out_proj.weight.data.copy_(child.out_proj.weight.data)
            if child.out_proj.bias is not None and new_attn.out_proj.bias is not None:
                new_attn.out_proj.bias.data.copy_(child.out_proj.bias.data)
            setattr(module, name, new_attn)
        else:
            _replace_mha_with_int8_qkv_from_map(child, mask_map, prefix=full_name)


def main():
    parser = argparse.ArgumentParser(description="Task 8: DeiT-Tiny CSR Non-Collision Attack")
    parser.add_argument("--data-root", type=str, required=True, help="ImageNet root path")
    parser.add_argument("--subset-root", type=str, default=None, help="Optional ImageNet subset root")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--calib-samples", type=int, default=128,
                        help="Attack calibration samples (used for gradient-based candidate search)")
    parser.add_argument("--ptq-calib-samples", type=int, default=128,
                        help="PTQ calibration samples (forward-only pass to validate sparse INT8 path)")
    parser.add_argument("--max-flips", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--weights-path", type=str, default=None,
                        help="Local DeiT/ViT weights path (offline load)")
    parser.add_argument("--max-groups-per-layer", type=int, default=2000,
                        help="Optional cap on groups sampled per layer for candidate search")
    parser.add_argument("--eval-samples", type=int, default=256,
                        help="Optional cap on evaluation samples per accuracy check")
    parser.add_argument("--allow-nonpositive", action="store_true",
                        help="Allow non-positive score candidates when positives are scarce")
    parser.add_argument("--imagenette-root", type=str, default=None,
                        help="Local Imagenette root path (used if ImageNet is unavailable)")
    parser.add_argument("--min-dense-acc", type=float, default=70.0)
    parser.add_argument("--min-sparse-fp32-acc", type=float, default=60.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs("./results", exist_ok=True)

    batch_size = args.batch_size
    num_workers = 0 if args.num_workers > 0 else args.num_workers
    while True:
        try:
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

            # Step A: Ladder test (Dense FP32 -> Sparse FP32 -> Sparse INT8)
            model = create_imagenet_deit_tiny_fp32(device=device, weights_path=args.weights_path)
            dense_acc = evaluate(model, val_loader, device, max_samples=args.eval_samples)
            print(f"[Task8] Check 1 (Dense FP32) Acc: {dense_acc:.2f}%")
            if dense_acc < args.min_dense_acc:
                raise RuntimeError(
                    f"[Task8] Dense FP32 acc {dense_acc:.2f}% < threshold {args.min_dense_acc:.2f}%. "
                    "Weight loading/remapping likely incorrect."
                )

            # Apply 2:4 sparsity to QKV projections (MultiheadAttention.in_proj_weight).
            qkv_mask_map: dict[str, torch.Tensor] = {}
            for name, mod in model.named_modules():
                if not isinstance(mod, nn.MultiheadAttention):
                    continue
                if not (name.startswith("encoder.layers.encoder_layer_") and name.endswith(".self_attention")):
                    continue
                mask = _compute_2_4_mask_linear(mod.in_proj_weight)
                mod.in_proj_weight.data.mul_(mask)
                qkv_mask_map[name] = mask

            # Apply 2:4 sparsity only to MLP fc1/fc2 (skip patch embedding + head).
            def _linear_filter(mod: nn.Linear, name: str) -> bool:
                return (".mlp.0" in name) or (".mlp.3" in name)

            mask_map = apply_sparse_mask(
                model,
                compute_conv_mask=lambda w: None,
                compute_linear_mask=_compute_2_4_mask_linear,
                conv_filter=lambda m, n: False,
                linear_filter=_linear_filter
            )

            sparse_fp32_acc = evaluate(model, val_loader, device, max_samples=args.eval_samples)
            print(f"[Task8] Check 2 (Sparse FP32) Acc: {sparse_fp32_acc:.2f}%")
            if sparse_fp32_acc < args.min_sparse_fp32_acc:
                raise RuntimeError(
                    f"[Task8] Sparse FP32 acc {sparse_fp32_acc:.2f}% < threshold {args.min_sparse_fp32_acc:.2f}%. "
                    "Sparsity may be applied to sensitive layers or is too aggressive."
                )

            # Convert to INT8 modules (PTQ weights) and attach sparse masks.
            _replace_mha_with_int8_qkv_from_map(model, qkv_mask_map)
            replace_with_int8_from_mask(model, mask_map)
            calibrate_int8(model)
            for mod in model.modules():
                if isinstance(mod, Int8QuantizedMultiheadAttention):
                    mod.calibrate_quantization()
            model.eval()

            # PTQ calibration (activation pass) to validate the sparse INT8 forward path.
            with torch.no_grad():
                seen = 0
                for inputs, _ in calib_loader:
                    inputs = inputs.to(device)
                    _ = model(inputs)
                    seen += inputs.size(0)
                    if seen >= args.ptq_calib_samples:
                        break

            int8_acc = evaluate(model, val_loader, device, max_samples=args.eval_samples)
            print(f"[Task8] Check 3 (Sparse INT8) Acc: {int8_acc:.2f}%")

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
            break
        except PermissionError as exc:
            if num_workers > 0:
                print(f"[Task8] DataLoader worker error ({exc}). Falling back to num_workers=0.")
                num_workers = 0
                continue
            raise
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and device == "cuda":
                print(f"[Task8] CUDA OOM at batch_size={batch_size}, retrying with half.")
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                if batch_size == 1:
                    raise
                continue
            raise

    # Save result
    result_path = "./results/task8_deit_tiny_csr_non_collision_result.pkl"
    log_path = "./results/task8_deit_tiny_csr_non_collision_log.txt"
    plot_path = "./results/task8_deit_tiny_csr_non_collision.png"

    with open(result_path, "wb") as f:
        pickle.dump(result, f)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("Task 8: DeiT-Tiny CSR Non-Collision Attack\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Weights Path: {args.weights_path}\n")
        f.write(f"Imagenette Root: {args.imagenette_root}\n")
        f.write(f"Dense FP32 Acc: {dense_acc:.2f}%\n")
        f.write(f"Sparse FP32 Acc: {sparse_fp32_acc:.2f}%\n")
        f.write(f"Sparse INT8 Acc (Pre-Attack): {int8_acc:.2f}%\n")
        f.write(f"PTQ Calib Samples (fwd): {args.ptq_calib_samples}\n")
        f.write(f"Attack Calib Samples (grad): {args.calib_samples}\n")
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
             label='DeiT-Tiny CSR Non-Collision')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('Successful Flips', fontsize=12, fontweight='bold')
    plt.ylabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Task 8: DeiT-Tiny CSR Index Attack (Non-Collision)', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.xlim(left=0)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[Task8] Results saved to {result_path}")
    print(f"[Task8] Detailed log saved to {log_path}")
    print(f"[Task8] Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
