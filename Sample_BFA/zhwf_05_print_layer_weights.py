import os
import argparse
import sys
import torch
import numpy as np
from resnet20_nm import resnet20_nm


def load_state(ckpt_path: str):
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "model_state"):
            if key in ckpt:
                state = ckpt[key]
                break
        else:
            state = ckpt
    else:
        state = ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}
    return state


def analyze_conv_weight(w: torch.Tensor, name: str = "layer1.0.conv1.weight") -> None:
    # w shape: [out_channels, in_channels, kH, kW]
    w_np = w.cpu().numpy()
    shape = w_np.shape
    total = w_np.size
    zero_count = int((w_np == 0).sum())
    sparsity = 100.0 * zero_count / total

    print(f"Layer: {name}")
    print(f"Shape: {shape}")
    print(f"Dtype: {w_np.dtype}")
    print(f"Total elems: {total}")
    print(f"Zero elems: {zero_count} ({sparsity:.4f} %)")

    # Per-out-channel sparsity
    out_chan = shape[0]
    per_out = (w_np.reshape(out_chan, -1) == 0).sum(axis=1)
    per_out_pct = 100.0 * per_out / per_out.max()  # relative to max for visibility
    print("Per-output-channel zero counts (first 16):")
    for i, z in enumerate(per_out[:16]):
        print(f"  out[{i}]: zeros={int(z)} ({100.0*z/ (shape[1]*shape[2]*shape[3]):.2f}%)")

    # Check if zeros per output channel are constant (structured per-filter)
    nonzero_per_out = (w_np.reshape(out_chan, -1) != 0).sum(axis=1)
    unique_nonzero_counts = np.unique(nonzero_per_out)
    print(f"Unique nonzero counts per out-channel: {unique_nonzero_counts[:10]} (len={len(unique_nonzero_counts)})")

    # Check positional pattern across out channels
    mask = (w_np == 0).astype(np.int32)
    # frequency that a position (in,in_kh,kw) is zero across out channels
    pos_freq = mask.sum(axis=0)  # shape: [in, kH, kW]
    max_pos_freq = int(pos_freq.max())
    min_pos_freq = int(pos_freq.min())
    print(f"Position-wise zero frequency across output channels: min={min_pos_freq}, max={max_pos_freq}")

    # If some positions are always zero across all out channels -> structured kernel sparsity
    if max_pos_freq == out_chan:
        print("Conclusion hint: Some kernel positions are ZERO across all output channels (structured mask).")
    else:
        print("Conclusion hint: No kernel position is zero for all output channels (likely unstructured or per-filter patterns).")

    # Check for N:M style: identical number of nonzeros per contiguous groups?
    # Here check if per-out nonzero counts are identical or close
    mean_nz = nonzero_per_out.mean()
    std_nz = nonzero_per_out.std()
    print(f"Nonzero per-out mean={mean_nz:.2f}, std={std_nz:.2f}")
    if std_nz < 1e-3:
        print("Pattern: exact constant nonzeros per output channel (consistent structured pruning).")
    elif std_nz / (mean_nz + 1e-12) < 0.05:
        print("Pattern: nearly constant nonzeros per out-channel (likely N:M structured).")
    else:
        print("Pattern: varying nonzeros per out-channel (unstructured pruning likely).")

    # Print first output channel kernel values and mask overview
    print("\nExample: out[0] kernel values (flatten) and mask (0 indicates zero):")
    flat0 = w_np.reshape(out_chan, -1)[0]
    mask0 = (flat0 == 0).astype(np.int32)
    # show first 64 elements
    show_n = min(64, flat0.size)
    vals = ", ".join([f"{float(x):.3e}" for x in flat0[:show_n]])
    masks = ", ".join([str(int(x)) for x in mask0[:show_n]])
    print(vals)
    print(masks)


def print_out0_formatted(w: torch.Tensor) -> None:
    """Print out[0] flattened weights as 16 rows of 9 numbers each."""
    w_np = w.cpu().numpy()
    out0 = w_np.reshape(w_np.shape[0], -1)[0]
    flat = out0.flatten()
    # Expect 16*9 = 144 elements
    assert flat.size == 144, f"unexpected size {flat.size}, expected 144"
    for r in range(16):
        row = flat[r*9:(r+1)*9]
        row_str = "\t".join([f"{float(x):.6e}" for x in row])
        sys.stdout.write(row_str + "\n")
    sys.stdout.flush()


def print_groups_and_masks(w: torch.Tensor, print_mask: bool = True, print_pos: bool = False) -> None:
    """For out[0], group input channels into 4 consecutive groups of 4.
    For each group and each of 9 kernel positions print one line containing:
      - group weights: w1\tw2\tw3\tw4
      - optional mask: m1\tm2\tm3\tm4
      - optional pos encoding: 4-bit string (aabb) where aa/bb are 2-bit indices (0-3)

    Result: 36 lines (4 groups * 9 positions), one group-position per line.
    """
    w_np = w.cpu().numpy()
    # select out[0]: shape (in_channels, kH, kW) -> reshape to (16,9)
    out0 = w_np[0].reshape(w_np.shape[1], -1)
    rows, cols = out0.shape
    assert rows == 16 and cols == 9, f"unexpected shape {out0.shape}"

    groups = [(i*4, (i+1)*4) for i in range(4)]
    for g_idx, (s, e) in enumerate(groups):
        for pos in range(cols):
            vals = out0[s:e, pos]
            vals_list = [float(x) for x in vals]
            vals_str = "\t".join([f"{x:.6e}" for x in vals_list])

            parts = [f"[{vals_str}]"]

            if print_mask:
                masks = [(1 if v != 0 else 0) for v in vals_list]
                masks_str = "".join([str(m) for m in masks])
                parts.append(f"[{masks_str}]")

            if print_pos:
                # compute two nonzero indices (sorted) within the 4-group and encode as 4 bits
                idxs = [i for i, v in enumerate(vals_list) if v != 0]
                if len(idxs) >= 2:
                    a, b = sorted(idxs)[:2]
                elif len(idxs) == 1:
                    a = idxs[0]; b = 0
                else:
                    a = 0; b = 0
                binstr = f"{a:02b}{b:02b}"
                parts.append(f"[{binstr}]")

            # join parts with tab and print single line per group-position
            print("\t".join(parts))


def _load_weight_by_key(ckpt: str, key: str):
    state = load_state(ckpt)
    model = resnet20_nm()
    model.load_state_dict(state)
    if key in state:
        return state[key]
    for name, p in model.named_parameters():
        if name == key:
            return p.data
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="1_sparse_finetune.pth")
    parser.add_argument("--out0", action="store_true", help="Print out[0] as 16 rows x 9 cols and exit")
    parser.add_argument("--groups", action="store_true", help="Print grouped weights for out[0] (36 lines: 4 groups x 9 positions)")
    parser.add_argument("--mask", action="store_true", help="Also print mask for each group (1 if nonzero, 0 if zero)")
    parser.add_argument("--pos", action="store_true", help="Also print 4-bit position encoding for each group (two 2-bit indices)")
    args = parser.parse_args()

    key = "layer1.0.conv1.weight"
    w = _load_weight_by_key(args.ckpt, key)
    if w is None:
        print(f"Weight {key} not found in checkpoint or model.")
        raise SystemExit(1)

    if args.out0:
        print_out0_formatted(w)
        raise SystemExit(0)

    if args.groups:
        print_groups_and_masks(w, print_mask=args.mask, print_pos=args.pos)
        raise SystemExit(0)

    analyze_conv_weight(w, name=key)
