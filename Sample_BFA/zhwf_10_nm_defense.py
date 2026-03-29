import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from zhwf_06_nm_pos_attack import load_state
from resnet20_nm import resnet20_nm


def _encode_posenc_from_group_vals(group_vals: np.ndarray) -> int:
    """Encode top-2 active indices (by abs magnitude) in a 4-value group into 4-bit posenc."""
    abs_vals = np.abs(group_vals.astype(np.float64))
    top2 = np.argsort(abs_vals)[-2:]
    top2 = sorted([int(top2[0]), int(top2[1])])
    return (top2[0] << 2) | top2[1]


def _decode_posenc_to_pair(code: int) -> Tuple[int, int]:
    a = (int(code) >> 2) & 0b11
    b = int(code) & 0b11
    if a == b:
        # collision encoding should not exist in legal 2:4 position encoding.
        # fallback to canonical pair if encountered.
        return 0, 1
    return (a, b) if a < b else (b, a)


def _iter_conv_2to4_groups(weight: np.ndarray):
    """
    Iterate (out_i, group_i, pos_i, group_vals_view) over 2:4 groups of a conv weight.
    weight shape: (out_ch, in_ch, kH, kW)
    """
    if weight.ndim != 4:
        return
    out_ch, in_ch, k_h, k_w = weight.shape
    if in_ch < 4:
        return
    group_count = in_ch // 4
    if group_count <= 0:
        return
    k = k_h * k_w
    w_flat = weight.reshape(out_ch, in_ch, k)
    for out_i in range(out_ch):
        for gi in range(group_count):
            s = gi * 4
            e = s + 4
            for pi in range(k):
                yield out_i, gi, pi, w_flat[out_i, s:e, pi]


def _target_weight_keys(state: Dict[str, torch.Tensor], layer_prefixes: List[str]) -> List[str]:
    keys = []
    for k, v in state.items():
        if not k.endswith(".weight"):
            continue
        if "conv" not in k:
            continue
        if not any(k.startswith(p + ".") for p in layer_prefixes):
            continue
        if not torch.is_tensor(v):
            continue
        if v.ndim != 4:
            continue
        if v.shape[1] < 4:
            continue
        keys.append(k)
    return sorted(keys)


def repair_with_reference(
    attacked_state: Dict[str, torch.Tensor],
    reference_state: Dict[str, torch.Tensor],
    layer_prefixes: List[str],
    mode: str = "strict_copy",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """
    Compare attacked vs reference 2:4 position encoding per group; repair mismatches.

    mode:
      - strict_copy: copy the full 4-value group from reference for mismatched groups
      - remap_to_ref_pos: keep attacked magnitudes but move top-2 values into reference positions
    """
    repaired = {k: v.clone() if torch.is_tensor(v) else v for k, v in attacked_state.items()}
    keys = _target_weight_keys(repaired, layer_prefixes)

    stats = {
        "layers_scanned": len(keys),
        "groups_scanned": 0,
        "groups_mismatch": 0,
        "groups_repaired": 0,
    }

    for key in keys:
        if key not in reference_state:
            continue
        wa = repaired[key].detach().cpu().numpy().copy()
        wr = reference_state[key].detach().cpu().numpy().copy()
        if wa.shape != wr.shape:
            continue

        for out_i, gi, pi, ga in _iter_conv_2to4_groups(wa):
            s = gi * 4
            e = s + 4
            gr = wr.reshape(wr.shape[0], wr.shape[1], -1)[out_i, s:e, pi]
            code_a = _encode_posenc_from_group_vals(ga)
            code_r = _encode_posenc_from_group_vals(gr)
            stats["groups_scanned"] += 1
            if code_a == code_r:
                continue

            stats["groups_mismatch"] += 1
            if mode == "strict_copy":
                ga[:] = gr
            elif mode == "remap_to_ref_pos":
                cur = ga.copy()
                cur_abs = np.abs(cur.astype(np.float64))
                active_cur = np.argsort(cur_abs)[-2:]
                active_cur = sorted([int(active_cur[0]), int(active_cur[1])])
                v0 = float(cur[active_cur[0]])
                v1 = float(cur[active_cur[1]])
                r0, r1 = _decode_posenc_to_pair(code_r)
                new_g = np.zeros(4, dtype=ga.dtype)
                new_g[r0] = v0
                new_g[r1] = v1
                ga[:] = new_g
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            stats["groups_repaired"] += 1

        repaired[key] = torch.from_numpy(wa).type_as(repaired[key])

    return repaired, stats


def save_checkpoint_like(source_ckpt_path: str, state: Dict[str, torch.Tensor], out_path: str) -> None:
    """Save repaired state in a checkpoint format compatible with the source checkpoint."""
    ckpt = torch.load(source_ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        # keep original container keys if possible
        for key in ("state_dict", "model_state_dict", "model", "model_state"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt[key] = state
                break
        else:
            # plain dict state checkpoint
            ckpt = state
    else:
        ckpt = state

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(ckpt, out_path)


def evaluate_full_test_accuracy(state: Dict[str, torch.Tensor], batch_size: int = 128, num_workers: int = 0) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet20_nm()
    model.load_state_dict(state)
    model.to(device).eval()

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    testset = datasets.CIFAR10(
        root=os.path.expanduser("~/data/cifar10"),
        train=False,
        download=False,
        transform=test_transform,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(1)
            correct += int((pred == y).sum().item())
            total += y.size(0)
    return float(correct) / float(total) if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Defend zhwf_10 NM-pos attack via reference-guided group repair.")
    parser.add_argument("--attacked-ckpt", required=True, help="Potentially attacked checkpoint path")
    parser.add_argument("--reference-ckpt", default="1_sparse_finetune.pth", help="Trusted clean checkpoint path")
    parser.add_argument("--out-ckpt", required=True, help="Output repaired checkpoint path")
    parser.add_argument("--layers", default="layer1,layer2,layer3", help="Layer prefixes to protect")
    parser.add_argument(
        "--mode",
        default="strict_copy",
        choices=["strict_copy", "remap_to_ref_pos"],
        help="Repair mode",
    )
    parser.add_argument("--eval", action="store_true", help="Evaluate full CIFAR-10 test accuracy before/after")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    attacked_state = load_state(args.attacked_ckpt)
    reference_state = load_state(args.reference_ckpt)
    layer_prefixes = [x.strip() for x in args.layers.split(",") if x.strip()]

    repaired_state, stats = repair_with_reference(
        attacked_state=attacked_state,
        reference_state=reference_state,
        layer_prefixes=layer_prefixes,
        mode=args.mode,
    )
    save_checkpoint_like(args.attacked_ckpt, repaired_state, args.out_ckpt)

    print("Defense Summary")
    print(f"mode={args.mode}")
    print(f"layers={layer_prefixes}")
    print(f"layers_scanned={stats['layers_scanned']}")
    print(f"groups_scanned={stats['groups_scanned']}")
    print(f"groups_mismatch={stats['groups_mismatch']}")
    print(f"groups_repaired={stats['groups_repaired']}")
    print(f"saved={args.out_ckpt}")

    if args.eval:
        acc_before = evaluate_full_test_accuracy(attacked_state, batch_size=args.batch_size, num_workers=args.num_workers)
        acc_after = evaluate_full_test_accuracy(repaired_state, batch_size=args.batch_size, num_workers=args.num_workers)
        print(f"acc_before={acc_before:.6f}")
        print(f"acc_after={acc_after:.6f}")
        print(f"acc_gain={acc_after - acc_before:+.6f}")


if __name__ == "__main__":
    main()
