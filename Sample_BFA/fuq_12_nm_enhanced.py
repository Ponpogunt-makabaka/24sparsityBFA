import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from zhwf_06_nm_pos_attack import load_state
from resnet20_nm import resnet20_nm
from zhwf_04_nm_dense import forward_and_compute_gradients


# ---------------------------------------------------------------------------
# Shared helpers (from zhwf_10, with generalised group ranges)
# ---------------------------------------------------------------------------

def evaluate_full_test_accuracy(model: torch.nn.Module, device: torch.device, batch_size: int = 128) -> float:
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    testset = datasets.CIFAR10(root=os.path.expanduser('~/data/cifar10'), train=False, download=False, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            correct += int((preds == targets).sum().item())
            total += targets.size(0)
    return float(correct) / float(total) if total > 0 else 0.0


def groups_from_w(w: np.ndarray, out_idx: int = 0):
    """Extract 2:4 groups for a given output channel.

    Generalised: works for any in_ch divisible by 4 and any spatial size.
    """
    in_ch = w.shape[1]
    kH, kW = w.shape[2], w.shape[3]
    spatial = kH * kW
    out0 = w[out_idx].reshape(in_ch, -1)  # (in_ch, spatial)

    def _frac8_is_zero(x: float) -> bool:
        s = format(abs(float(x)), '.12f')
        if '.' not in s:
            return False
        frac = s.split('.')[-1]
        return frac[:8] == '00000000'

    n_groups = in_ch // 4
    groups = []
    for g in range(n_groups):
        s = g * 4
        e = s + 4
        group_positions = []
        for pos in range(spatial):
            vals = out0[s:e, pos].tolist()
            mask = [0 if _frac8_is_zero(v) else 1 for v in vals]
            if sum(mask) != 2:
                abs_vals = [abs(float(v)) for v in vals]
                top2 = sorted(range(len(vals)), key=lambda i: abs_vals[i], reverse=True)[:2]
                mask = [1 if i in top2 else 0 for i in range(len(vals))]
            idxs = [i for i, m in enumerate(mask) if m == 1]
            a, b = sorted(idxs)[:2]
            posenc = f"{a:02b}{b:02b}"
            group_positions.append({"vals": vals, "mask": mask, "posenc": posenc})
        groups.append(group_positions)
    return groups


# ---------------------------------------------------------------------------
# Collision set
# ---------------------------------------------------------------------------
COLLISION_SET = {'0000', '0101', '1010', '1111'}


# ---------------------------------------------------------------------------
# Step 1: Coarse Group Pre-filter (vectorized scoring)
# ---------------------------------------------------------------------------

def step1_coarse_group_prefilter(state, grads, target_keys, coarse_n, exclude_set):
    """For every group across all target layers, compute group_score = sum(|grad|).

    Scoring is vectorized via numpy; inner loops only append results.
    Returns top-N list of (score, key, out_i, gi, pi).
    """
    scored = []
    for key in target_keys:
        g_param = grads.get(key)
        if g_param is None:
            continue
        g_np = g_param.numpy()
        out_ch, in_ch, kH, kW = g_np.shape
        spatial = kH * kW
        n_groups = in_ch // 4
        # Vectorized: reshape to (out_ch, n_groups, 4, spatial), abs, sum over group-of-4
        g_reshaped = np.abs(g_np.reshape(out_ch, n_groups, 4, spatial)).sum(axis=2)
        # g_reshaped shape: (out_ch, n_groups, spatial)
        for out_i in range(out_ch):
            for gi in range(n_groups):
                for pi in range(spatial):
                    if (key, out_i, gi, pi) in exclude_set:
                        continue
                    scored.append((float(g_reshaped[out_i, gi, pi]), key, out_i, gi, pi))

    scored.sort(key=lambda x: -x[0])
    return scored[:coarse_n]


# ---------------------------------------------------------------------------
# Step 2+3: Candidate Generation + Directional Proxy Scoring (rank-based)
# ---------------------------------------------------------------------------

def step23_generate_and_score_candidates(selected, state, grads, forbidden_set):
    """For each selected group, enumerate single-bit posenc flips.

    Value transfer uses rank-based mapping (aligned with T10_Enhanced):
      old_active = sorted([a0, b0])
      new_active = sorted([a1, b1])
      new[new_active[rank]] = old_values[old_active[rank]]

    Proxy score = np.dot(g_slice, delta)  (signed, no abs, no normalisation).
    Returns list of candidate tuples.
    """
    candidates = []
    for _, key, out_i, gi, pi in selected:
        w = state[key].cpu().numpy()
        groups_before = groups_from_w(w, out_idx=out_i)
        entry = groups_before[gi][pi]
        orig_posenc = entry['posenc']
        orig_vals = np.array(entry['vals'], dtype=float)

        # Get gradient slice for this group position
        g_np = grads[key].numpy()
        in_ch = w.shape[1]
        out_grad = g_np[out_i].reshape(in_ch, -1)
        g_slice = out_grad[gi * 4:gi * 4 + 4, pi].astype(float)

        for bit_idx in range(4):
            flipped = list(orig_posenc)
            flipped[bit_idx] = '1' if orig_posenc[bit_idx] == '0' else '0'
            flipped_s = ''.join(flipped)
            if flipped_s == orig_posenc or flipped_s in COLLISION_SET:
                continue
            # check forbidden (both directions)
            if (key, out_i, gi, pi, orig_posenc, flipped_s) in forbidden_set:
                continue

            a0, b0 = int(orig_posenc[:2], 2), int(orig_posenc[2:], 2)
            a1, b1 = int(flipped_s[:2], 2), int(flipped_s[2:], 2)

            # Rank-based mapping: sort active positions, transfer by rank
            old_active = sorted([a0, b0])
            new_active = sorted([a1, b1])
            old_values = [orig_vals[old_active[0]], orig_vals[old_active[1]]]
            new_group_vals = np.zeros(4, dtype=float)
            for rank in range(2):
                new_group_vals[new_active[rank]] = old_values[rank]

            delta = new_group_vals - orig_vals
            proxy_score = float(np.dot(g_slice, delta))  # signed!
            candidates.append((proxy_score, key, out_i, gi, pi, orig_posenc, flipped_s, new_group_vals, orig_vals))

    return candidates


# ---------------------------------------------------------------------------
# Step 4: Global Top-K Selection
# ---------------------------------------------------------------------------

def step4_select_topk_candidates(candidates, topk):
    """Sort by proxy_score descending, return top-K."""
    candidates_sorted = sorted(candidates, key=lambda x: -x[0])
    return candidates_sorted[:topk]


# ---------------------------------------------------------------------------
# Step 5: Exact Forward Verification (Save-Apply-Restore)
# ---------------------------------------------------------------------------

def step5_exact_verification(state, candidates, model, inputs, targets, criterion, loss_orig, device):
    """Test each candidate with a real forward pass using save-apply-restore.

    Reuses a single model instance: modify state[key] -> load_state_dict -> eval -> restore.
    Returns the candidate with max positive delta.
    """
    best = None
    best_delta = -1e9

    for cand in candidates:
        proxy_score, key, out_i, gi, pi, orig_posenc, att_posenc, new_group_vals, orig_vals = cand

        # Save
        w_orig = state[key].clone()

        # Apply (rank-based new_group_vals already computed in step 2+3)
        w_mod = w_orig.cpu().numpy().copy()
        in_ch = w_mod.shape[1]
        s = gi * 4
        e = s + 4
        w_mod[out_i].reshape(in_ch, -1)[s:e, pi] = new_group_vals.astype(w_mod.dtype)
        state[key] = torch.from_numpy(w_mod).type_as(w_orig)
        model.load_state_dict(state)
        model.eval()

        # Evaluate
        with torch.no_grad():
            loss_att = float(criterion(model(inputs), targets).item())

        # Restore
        state[key] = w_orig

        delta = loss_att - loss_orig
        if delta > best_delta:
            best_delta = delta
            best = cand

    # Restore model to clean state
    model.load_state_dict(state)
    return best, best_delta


# ---------------------------------------------------------------------------
# Main enhanced NCA attack loop
# ---------------------------------------------------------------------------

def run_nca_enhanced(ckpt_path='sample_bfa_model.pth', coarse_n=1000, topk=64,
                     layer_prefixes=None, n_iter=1, print_candidates=False, seed=0):
    if layer_prefixes is None:
        layer_prefixes = ['layer1', 'layer2', 'layer3']

    # Deterministic seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    state = load_state(ckpt_path)
    target_keys = [k for k in state.keys()
                   if any(k.startswith(lp + '.') for lp in layer_prefixes)
                   and 'conv' in k and k.endswith('.weight')]
    if not target_keys:
        raise KeyError(f'No conv weight keys found for prefixes: {layer_prefixes}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet20_nm()
    st_init = {k: v.clone() for k, v in state.items()}
    model.load_state_dict(st_init)
    model.to(device).eval()

    # Deterministic test-set loader (no augmentation, no shuffle) — aligned with T10_Enhanced
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    testset = datasets.CIFAR10(root=os.path.expanduser('~/data/cifar10'), train=False, download=False, transform=test_transform)
    grad_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    criterion = nn.CrossEntropyLoss()

    # Verification batch: first 128 samples from test set (same loader, first batch)
    inputs, targets = next(iter(grad_loader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Sliding window dedup
    exclude_queue = deque(maxlen=20)
    exclude_set = set()
    forbidden_queue = deque(maxlen=1000)
    forbidden_set = set()

    for iter_i in range(n_iter):
        # Recompute gradients on current state (test set, 256 samples)
        st_cur = {k: v.clone() for k, v in state.items()}
        model.load_state_dict(st_cur)
        _, grads = forward_and_compute_gradients(model, grad_loader, criterion,
                                                 attack_sample_size=256, attack_batch_size=128,
                                                 device=str(device))

        # Step 1: coarse prefilter (vectorized)
        selected = step1_coarse_group_prefilter(state, grads, target_keys, coarse_n, exclude_set)
        if not selected:
            if print_candidates:
                print('No groups available; stopping.')
            break

        # Step 2+3: generate and score candidates (rank-based mapping)
        candidates = step23_generate_and_score_candidates(selected, state, grads, forbidden_set)
        if not candidates:
            if print_candidates:
                print('No candidates generated; stopping.')
            break

        # Step 4: global top-K
        topk_cands = step4_select_topk_candidates(candidates, topk)

        # Baseline loss
        st_base = {k: v.clone() for k, v in state.items()}
        model.load_state_dict(st_base)
        model.to(device).eval()
        with torch.no_grad():
            out_orig = model(inputs)
            loss_orig = float(criterion(out_orig, targets).item())

        # Step 5: exact verification (save-apply-restore on single model)
        best, best_delta_val = step5_exact_verification(
            state, topk_cands, model, inputs, targets, criterion, loss_orig, device)

        if best is None or best_delta_val <= 0:
            if print_candidates:
                print('No positive-delta candidate; stopping.')
            break

        proxy_score, param_key, out_i, gi, pi, orig_posenc, att_posenc, new_group_vals, orig_vals = best

        # Compute full-test accuracy before attack
        acc_before = evaluate_full_test_accuracy(model, device)

        # Apply the best attack permanently to state (rank-based)
        w = state[param_key].cpu().numpy().copy()
        in_ch = w.shape[1]
        s = gi * 4
        e = s + 4
        w[out_i].reshape(in_ch, -1)[s:e, pi] = new_group_vals.astype(w.dtype)
        state[param_key] = torch.from_numpy(w).type_as(state[param_key])

        # Evaluate full-test accuracy after attack
        st_after = {k: v.clone() for k, v in state.items()}
        model.load_state_dict(st_after)
        model.to(device).eval()
        acc_after = evaluate_full_test_accuracy(model, device)

        # Update sliding window dedup: exclude recently attacked group
        group_key = (param_key, out_i, gi, pi)
        if group_key not in exclude_set:
            if len(exclude_queue) == exclude_queue.maxlen:
                popped = exclude_queue.popleft()
                exclude_set.discard(popped)
            exclude_queue.append(group_key)
            exclude_set.add(group_key)

        # Forbid this transition and its reverse
        fwd = (param_key, out_i, gi, pi, orig_posenc, att_posenc)
        rev = (param_key, out_i, gi, pi, att_posenc, orig_posenc)
        for t_key in (fwd, rev):
            if t_key not in forbidden_set:
                if len(forbidden_queue) == forbidden_queue.maxlen:
                    popped = forbidden_queue.popleft()
                    forbidden_set.discard(popped)
                forbidden_queue.append(t_key)
                forbidden_set.add(t_key)

        # Print results (format compatible with zhwf_10_nm_plot.py)
        if print_candidates:
            if iter_i == 0:
                print('\nFirst attack:')
            else:
                print(f"\nBest attack (iter {iter_i}):")
            print('param_key,out_ch,group,pos,orig_posenc,att_posenc,proxy_score,loss_orig,loss_att,delta')
            print(f"{param_key},{out_i},{gi},{pi},{orig_posenc},{att_posenc},{proxy_score:.6e},{loss_orig:.6f},{loss_orig + best_delta_val:.6f},{best_delta_val:.6f}")
        print('acc_before')
        print(f"{acc_before:.6f}")
        print('acc_after')
        print(f"{acc_after:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NM enhanced attack (T10_Enhanced algorithm-aligned)')
    parser.add_argument('--nca-enhanced', action='store_true', help='Run enhanced NCA with verbose output')
    parser.add_argument('--nca-proxy-best-only', action='store_true', help='Run enhanced NCA with plot-compatible output (acc_after lines)')
    parser.add_argument('--ckpt', default='sample_bfa_model.pth', help='Checkpoint path')
    parser.add_argument('--layers', default='layer1,layer2,layer3', help='Comma-separated layer prefixes to search')
    parser.add_argument('--proxy-topG', type=int, default=1000, help='Step 1: number of groups to keep after coarse ranking')
    parser.add_argument('--per-group-K', type=int, default=64, help='Step 4: global top-K candidates for exact verification')
    parser.add_argument('--n-iter', type=int, default=1, help='Number of iterative attack rounds')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()

    if args.nca_enhanced:
        run_nca_enhanced(args.ckpt, coarse_n=args.proxy_topG, topk=args.per_group_K,
                         layer_prefixes=args.layers.split(','), n_iter=args.n_iter,
                         print_candidates=True, seed=args.seed)
    elif args.nca_proxy_best_only:
        run_nca_enhanced(args.ckpt, coarse_n=args.proxy_topG, topk=args.per_group_K,
                         layer_prefixes=args.layers.split(','), n_iter=args.n_iter,
                         print_candidates=False, seed=args.seed)
    else:
        print('Usage: specify --nca-enhanced or --nca-proxy-best-only')
        parser.print_help()
