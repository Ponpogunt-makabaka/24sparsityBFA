import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from zhwf_06_nm_pos_attack import load_state
from resnet20_nm import resnet20_nm
from zhwf_04_nm_dense import forward_and_compute_gradients


def evaluate_full_test_accuracy(model: torch.nn.Module, device: torch.device, batch_size: int = 128) -> float:
    """Evaluate model accuracy on the full CIFAR-10 test set and return accuracy (0-1)."""
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
    # replicate grouping logic from zhwf_06_nm_pos_attack.group_layer1_conv1
    # w: full weight array or per-out array; out_idx selects output channel
    in_ch = w.shape[1]
    out0 = w[out_idx].reshape(in_ch, -1)  # (in_ch, 9)

    def _frac8_is_zero(x: float) -> bool:
        s = format(abs(float(x)), '.12f')
        if '.' not in s:
            return False
        frac = s.split('.')[-1]
        return frac[:8] == '00000000'

    groups = []
    for g in range(4):
        s = g * 4
        e = s + 4
        group_positions = []
        for pos in range(9):
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


def compute_and_print(ckpt_path='1_sparse_finetune.pth', attack_group=None, attack_pos=None, orig_posenc='0110', new_posenc='0111'):
    state = load_state(ckpt_path)
    key = 'layer1.0.conv1.weight'
    if key not in state:
        raise KeyError(f'{key} not found')

    w = state[key].cpu().numpy().copy()

    # grouping before attack
    groups_before = groups_from_w(w)

    # if attack_group is provided, construct attacked weight copy and groups_after
    if attack_group is not None and attack_pos is not None:
        vals = groups_before[attack_group][attack_pos]['vals']
        a0, b0 = int(orig_posenc[:2], 2), int(orig_posenc[2:], 2)
        a1, b1 = int(new_posenc[:2], 2), int(new_posenc[2:], 2)
        stored = [vals[i] for i in [a0, b0]]
        new_group_vals = [0.0] * 4
        for i, idx in enumerate([a1, b1]):
            new_group_vals[idx] = stored[i]
        w_att = w.copy()
        s = attack_group * 4
        e = s + 4
        w_att[0].reshape(w.shape[1], -1)[s:e, attack_pos] = np.array(new_group_vals, dtype=w_att.dtype)
        groups_after = groups_from_w(w_att)
    else:
        # no attack: after == before
        groups_after = groups_before

    # prepare model and attack loader (reuse forward_and_compute_gradients)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_orig = resnet20_nm()
    st_orig = {k: v.clone() for k, v in state.items()} if isinstance(state, dict) else state
    model_orig.load_state_dict(st_orig)

    # prepare attack data loader (trainset sampler) matching zhwf_04_nm_dense defaults
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    trainset = datasets.CIFAR10(root=os.path.expanduser('~/data/cifar10'), train=True, download=False, transform=train_transform)
    attack_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()

    # compute grads (uses first 128 samples by default in helper)
    _, grads_before = forward_and_compute_gradients(model_orig, attack_loader, criterion, attack_sample_size=128, attack_batch_size=128, device=str(device))

    target_name = 'layer1.0.conv1.weight'
    g_before = grads_before.get(target_name)
    if g_before is None:
        raise RuntimeError('Gradients for target layer not found')

    # reshape to (in,9) for out[0]
    g_before_arr = g_before.numpy()[0].reshape(g_before.shape[1], -1)  # (16,9)

    # collect rows and print header
    rows = []
    print('group,pos,before_vals,before_posenc,after_vals,after_posenc,grad_sum_before')
    for gi in range(4):
        s = gi * 4
        e = s + 4
        for pi in range(9):
            before_entry = groups_before[gi][pi]
            after_entry = groups_after[gi][pi]
            before_vals = before_entry['vals']
            before_posenc = before_entry['posenc']
            after_vals = after_entry['vals']
            after_posenc = after_entry['posenc']
            grad_sum_b = float(np.abs(g_before_arr[s:e, pi]).sum())
            before_vals_str = ' '.join([f"{v:.3f}" for v in before_vals])
            after_vals_str = ' '.join([f"{v:.3f}" for v in after_vals])
            rows.append((gi, pi, before_vals_str, before_posenc, after_vals_str, after_posenc, grad_sum_b))
            # print(f"{gi},{pi}, [{before_vals_str}], {before_posenc}, [{after_vals_str}], {after_posenc}, {grad_sum_b:.6e}")

    # print top-10 by grad_sum_before
    rows_sorted = sorted(rows, key=lambda x: x[6], reverse=True)
    print('\nTop-10 entries by grad_sum_before:')
    print('rank,group,pos,before_vals,before_posenc,grad_sum_before')
    for rank, r in enumerate(rows_sorted[:10], start=1):
        gi, pi, bvals, bpos, avals, apos, gsum = r
        print(f"{rank},{gi},{pi}, [{bvals}], {bpos}, {gsum:.6e}")


def measure_loss_change_for_group_pos(ckpt_path: str = '1_sparse_finetune.pth', attack_group: int = 0, attack_pos: int = 0, orig_posenc: str = '0110', new_posencs=None):
    if new_posencs is None:
        new_posencs = ['1110', '0010', '0100', '0111']

    state = load_state(ckpt_path)
    key = 'layer1.0.conv1.weight'
    if key not in state:
        raise KeyError(f'{key} not found')

    w = state[key].cpu().numpy().copy()
    groups_before = groups_from_w(w)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # original model
    model_orig = resnet20_nm()
    st_orig = {k: v.clone() for k, v in state.items()} if isinstance(state, dict) else state
    model_orig.load_state_dict(st_orig)
    model_orig.to(device).eval()

    # attack data loader (single batch) — reuse same transform as other functions
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    trainset = datasets.CIFAR10(root=os.path.expanduser('~/data/cifar10'), train=True, download=False, transform=train_transform)
    attack_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()

    # take first batch
    inputs, targets = next(iter(attack_loader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        out_orig = model_orig(inputs)
        loss_orig = float(criterion(out_orig, targets).item())

    print('new_posenc,loss_orig,loss_att,delta')
    # perform each attack
    for new_posenc in new_posencs:
        # construct attacked weights copy
        vals = groups_before[attack_group][attack_pos]['vals']
        a0, b0 = int(orig_posenc[:2], 2), int(orig_posenc[2:], 2)
        a1, b1 = int(new_posenc[:2], 2), int(new_posenc[2:], 2)
        stored = [vals[i] for i in [a0, b0]]
        new_group_vals = [0.0] * 4
        for i, idx in enumerate([a1, b1]):
            new_group_vals[idx] = stored[i]
        w_att = w.copy()
        s = attack_group * 4
        e = s + 4
        w_att[0].reshape(w.shape[1], -1)[s:e, attack_pos] = np.array(new_group_vals, dtype=w_att.dtype)

        # build attacked state dict and model
        st_att = {k: v.clone() for k, v in state.items()} if isinstance(state, dict) else state
        st_att[key] = torch.from_numpy(w_att).type_as(st_att[key])
        model_att = resnet20_nm()
        model_att.load_state_dict(st_att)
        model_att.to(device).eval()

        with torch.no_grad():
            out_att = model_att(inputs)
            loss_att = float(criterion(out_att, targets).item())

        delta = loss_att - loss_orig
        print(f"{new_posenc},{loss_orig:.6f},{loss_att:.6f},{delta:.6f}")


def run_nca_all(ckpt_path: str = '1_sparse_finetune.pth') -> None:
    """For each of the 36 (group,pos) entries in out[0], enumerate all non-collision
    4-bit posenc targets (excluding original and collision encodings), perform the
    attack and print loss changes."""
    collision_set = {'0000', '0101', '1010', '1111'}
    all_posencs = [format(i, '04b') for i in range(16)]

    state = load_state(ckpt_path)
    key = 'layer1.0.conv1.weight'
    if key not in state:
        raise KeyError(f'{key} not found')
    w = state[key].cpu().numpy().copy()
    out_ch = w.shape[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_orig = resnet20_nm()
    st_orig = {k: v.clone() for k, v in state.items()} if isinstance(state, dict) else state
    model_orig.load_state_dict(st_orig)
    model_orig.to(device).eval()

    # single batch for evaluation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    trainset = datasets.CIFAR10(root=os.path.expanduser('~/data/cifar10'), train=True, download=False, transform=train_transform)
    attack_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    inputs, targets = next(iter(attack_loader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        out_orig = model_orig(inputs)
        loss_orig = float(criterion(out_orig, targets).item())

    print('out_ch,group,pos,orig_posenc,att_posenc,loss_orig,loss_att,delta')
    # iterate all output channels and their 36 positions
    for out_i in range(out_ch):
        groups_before = groups_from_w(w, out_idx=out_i)
        for gi in range(4):
            s = gi * 4
            e = s + 4
            for pi in range(9):
                entry = groups_before[gi][pi]
                orig_posenc = entry['posenc']
                # candidate targets: single-bit flips of orig_posenc, excluding collisions
                targets_posenc = []
                for bit_idx in range(4):
                    flipped = list(orig_posenc)
                    flipped[bit_idx] = '1' if orig_posenc[bit_idx] == '0' else '0'
                    flipped_s = ''.join(flipped)
                    if flipped_s != orig_posenc and flipped_s not in collision_set:
                        targets_posenc.append(flipped_s)
                for att in targets_posenc:
                    # construct attacked weight
                    vals = entry['vals']
                    a0, b0 = int(orig_posenc[:2], 2), int(orig_posenc[2:], 2)
                    a1, b1 = int(att[:2], 2), int(att[2:], 2)
                    stored = [vals[i] for i in [a0, b0]]
                    new_group_vals = [0.0] * 4
                    for i, idx in enumerate([a1, b1]):
                        new_group_vals[idx] = stored[i]
                    w_att = w.copy()
                    w_att[out_i].reshape(w.shape[1], -1)[s:e, pi] = np.array(new_group_vals, dtype=w_att.dtype)

                    # build attacked model
                    st_att = {k: v.clone() for k, v in state.items()} if isinstance(state, dict) else state
                    st_att[key] = torch.from_numpy(w_att).type_as(st_att[key])
                    model_att = resnet20_nm()
                    model_att.load_state_dict(st_att)
                    model_att.to(device).eval()

                    with torch.no_grad():
                        out_att = model_att(inputs)
                        loss_att = float(criterion(out_att, targets).item())

                    delta = loss_att - loss_orig
                    print(f"{out_i},{gi},{pi},{orig_posenc},{att},{loss_orig:.6f},{loss_att:.6f},{delta:.6f}")


def run_nca_layer_best(ckpt_path: str = '1_sparse_finetune.pth') -> None:
    """Run single-bit NCA over entire layer and print the single attack with maximum loss increase."""
    collision_set = {'0000', '0101', '1010', '1111'}

    state = load_state(ckpt_path)
    key = 'layer1.0.conv1.weight'
    if key not in state:
        raise KeyError(f'{key} not found')
    w = state[key].cpu().numpy().copy()
    out_ch = w.shape[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_orig = resnet20_nm()
    st_orig = {k: v.clone() for k, v in state.items()} if isinstance(state, dict) else state
    model_orig.load_state_dict(st_orig)
    model_orig.to(device).eval()

    # single batch
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    trainset = datasets.CIFAR10(root=os.path.expanduser('~/data/cifar10'), train=True, download=False, transform=train_transform)
    attack_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    inputs, targets = next(iter(attack_loader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        out_orig = model_orig(inputs)
        loss_orig = float(criterion(out_orig, targets).item())

    best = None
    best_delta = -1e9

    for out_i in range(out_ch):
        groups_before = groups_from_w(w, out_idx=out_i)
        for gi in range(4):
            s = gi * 4
            e = s + 4
            for pi in range(9):
                entry = groups_before[gi][pi]
                orig_posenc = entry['posenc']
                # single-bit flips
                for bit_idx in range(4):
                    flipped = list(orig_posenc)
                    flipped[bit_idx] = '1' if orig_posenc[bit_idx] == '0' else '0'
                    flipped_s = ''.join(flipped)
                    if flipped_s == orig_posenc or flipped_s in collision_set:
                        continue

                    # construct attacked weight
                    vals = entry['vals']
                    a0, b0 = int(orig_posenc[:2], 2), int(orig_posenc[2:], 2)
                    a1, b1 = int(flipped_s[:2], 2), int(flipped_s[2:], 2)
                    stored = [vals[i] for i in [a0, b0]]
                    new_group_vals = [0.0] * 4
                    for i, idx in enumerate([a1, b1]):
                        new_group_vals[idx] = stored[i]

                    w_att = w.copy()
                    w_att[out_i].reshape(w.shape[1], -1)[s:e, pi] = np.array(new_group_vals, dtype=w_att.dtype)

                    # build attacked model
                    st_att = {k: v.clone() for k, v in state.items()} if isinstance(state, dict) else state
                    st_att[key] = torch.from_numpy(w_att).type_as(st_att[key])
                    model_att = resnet20_nm()
                    model_att.load_state_dict(st_att)
                    model_att.to(device).eval()

                    with torch.no_grad():
                        out_att = model_att(inputs)
                        loss_att = float(criterion(out_att, targets).item())

                    delta = loss_att - loss_orig
                    if delta > best_delta:
                        best_delta = delta
                        best = (out_i, gi, pi, orig_posenc, flipped_s, loss_orig, loss_att, delta)

    if best is None:
        print('No valid single-bit non-collision attacks found')
    else:
        out_i, gi, pi, orig_posenc, att, loss_o, loss_a, delta = best
        print('out_ch,group,pos,orig_posenc,att_posenc,loss_orig,loss_att,delta')
        print(f"{out_i},{gi},{pi},{orig_posenc},{att},{loss_o:.6f},{loss_a:.6f},{delta:.6f}")


def run_nca_proxy(ckpt_path: str = '1_sparse_finetune.pth', topG: int = 200, per_group_k: int = 2, print_candidates: bool = False, layer_prefixes=None, n_iter: int = 1) -> None:
    """Two-stage proxy over multiple layers: (1) rank groups by sum(|grad|) across the 4 weights; (2) for top-G groups compute candidate score = |g^T delta|/||delta|| and evaluate top-K candidates precisely.

    `layer_prefixes` is a list of parameter-name prefixes to search (e.g. ['layer1','layer2','layer3']).
    When ``print_candidates`` is False the function will suppress per-candidate output and only print the final Best attack.
    """
    collision_set = {'0000', '0101', '1010', '1111'}

    if layer_prefixes is None:
        layer_prefixes = ['layer1', 'layer2', 'layer3']

    state = load_state(ckpt_path)
    # collect target parameter keys matching the requested layer prefixes
    target_keys = [k for k in state.keys() if any(k.startswith(lp + '.') for lp in layer_prefixes) and 'conv' in k and k.endswith('.weight')]
    if not target_keys:
        raise KeyError(f'No conv weight keys found for prefixes: {layer_prefixes}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_orig = resnet20_nm()
    st_orig = {k: v.clone() for k, v in state.items()} if isinstance(state, dict) else state
    model_orig.load_state_dict(st_orig)
    model_orig.to(device).eval()

    # prepare single batch and criterion
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    trainset = datasets.CIFAR10(root=os.path.expanduser('~/data/cifar10'), train=True, download=False, transform=train_transform)
    attack_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    inputs, targets = next(iter(attack_loader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    # iterative attack loop: run up to n_iter rounds, skipping already-attacked (param_key,out_i,gi,pi)
    attacked_groups = set()  # set of (param_key,out_i,gi,pi) already attacked
    for iter_i in range(n_iter):
        # recompute grads on current model/state
        st_orig = {k: v.clone() for k, v in state.items()} if isinstance(state, dict) else state
        model_orig.load_state_dict(st_orig)
        _, grads = forward_and_compute_gradients(model_orig, attack_loader, criterion, attack_sample_size=128, attack_batch_size=128, device=str(device))

        # build list of groups across all selected target_keys with group-level score = sum(abs(g_slice))
        groups = []  # (score, param_key, out_i, gi, pi, g_slice, orig_vals, orig_posenc)
        for key in target_keys:
            g_param = grads.get(key)
            if g_param is None:
                # skip keys without gradients
                continue
            g_np = g_param.numpy()
            w = state[key].cpu().numpy().copy()
            out_ch = w.shape[0]
            for out_i in range(out_ch):
                for gi in range(4):
                    for pi in range(9):
                        if (key, out_i, gi, pi) in attacked_groups:
                            # skip groups already attacked in previous iterations
                            continue
                        out_grad = g_np[out_i].reshape(g_np.shape[1], -1)  # (in_ch, 9)
                        g_slice = out_grad[gi*4:gi*4+4, pi].astype(float)
                        score_group = float(np.abs(g_slice).sum())
                        groups_before = groups_from_w(w, out_idx=out_i)
                        entry = groups_before[gi][pi]
                        groups.append((score_group, key, out_i, gi, pi, g_slice, np.array(entry['vals'], dtype=float), entry['posenc']))

        groups_sorted = sorted(groups, key=lambda x: x[0], reverse=True)
        topG = min(topG, len(groups_sorted))
        selected_groups = groups_sorted[:topG]

        # candidate scoring per selected group
        candidates = []  # (cand_score, param_key, out_i, gi, pi, orig_posenc, cand_posenc, delta_vec, orig_vals)
        for _, key, out_i, gi, pi, g_slice, orig_vals, orig_posenc in selected_groups:
            for bit_idx in range(4):
                flipped = list(orig_posenc)
                flipped[bit_idx] = '1' if orig_posenc[bit_idx] == '0' else '0'
                flipped_s = ''.join(flipped)
                if flipped_s == orig_posenc or flipped_s in collision_set:
                    continue
                a0, b0 = int(orig_posenc[:2], 2), int(orig_posenc[2:], 2)
                a1, b1 = int(flipped_s[:2], 2), int(flipped_s[2:], 2)
                stored = [orig_vals[i] for i in [a0, b0]]
                new_group_vals = np.zeros(4, dtype=float)
                for i, idx in enumerate([a1, b1]):
                    new_group_vals[idx] = stored[i]
                delta = new_group_vals - orig_vals
                denom = np.linalg.norm(delta) + 1e-12
                cand_score = abs(float(np.dot(g_slice, delta))) / denom
                candidates.append((cand_score, key, out_i, gi, pi, orig_posenc, flipped_s, delta, orig_vals))

        # pick top per-group K
        # group candidates by (param_key,out_i,gi,pi)
        from collections import defaultdict
        group_cands = defaultdict(list)
        for item in candidates:
            _, key, out_i, gi, pi, *_ = item
            group_cands[(key, out_i, gi, pi)].append(item)

        selected_candidates = []
        for keyg, clist in group_cands.items():
            clist_sorted = sorted(clist, key=lambda x: x[0], reverse=True)
            for item in clist_sorted[:per_group_k]:
                selected_candidates.append(item)

        # evaluate selected candidates precisely
        best = None
        best_delta = -1e9
        if print_candidates:
            print('out_ch,group,pos,orig_posenc,att_posenc,proxy_score,loss_orig,loss_att,delta')

        with torch.no_grad():
            out_orig = model_orig(inputs)
            loss_orig = float(criterion(out_orig, targets).item())

        for cand in selected_candidates:
            cand_score, key, out_i, gi, pi, orig_posenc, att_posenc, delta_vec, orig_vals = cand
            # construct attacked weight
            s = gi * 4
            e = s + 4
            # copy the parameter-specific weights and apply delta
            w = state[key].cpu().numpy().copy()
            w_att = w.copy()
            w_att[out_i].reshape(w.shape[1], -1)[s:e, pi] = (orig_vals + delta_vec).astype(w_att.dtype)

            st_att = {k: v.clone() for k, v in state.items()} if isinstance(state, dict) else state
            st_att[key] = torch.from_numpy(w_att).type_as(st_att[key])
            model_att = resnet20_nm()
            model_att.load_state_dict(st_att)
            model_att.to(device).eval()

            with torch.no_grad():
                out_att = model_att(inputs)
                loss_att = float(criterion(out_att, targets).item())

            delta = loss_att - loss_orig
            if print_candidates:
                print(f"{out_i},{gi},{pi},{orig_posenc},{att_posenc},{cand_score:.6e},{loss_orig:.6f},{loss_att:.6f},{delta:.6f}")
            if delta > best_delta:
                best_delta = delta
                best = (key, out_i, gi, pi, orig_posenc, att_posenc, cand_score, loss_orig, loss_att, delta)
        # end of candidate evaluation for this iteration
        if best is not None:
            # unpack best attack which includes the parameter key (param_key, out_i, gi, pi, ...)
            param_key, out_i, gi, pi, orig_posenc, att_posenc, cand_score, loss_o, loss_a, delta = best

            # compute full-test accuracy before attack
            acc_before = evaluate_full_test_accuracy(model_orig, device)

            # build attacked weights for the best attack and evaluate full-test accuracy after attack
            s = gi * 4
            e = s + 4
            w = state[param_key].cpu().numpy().copy()
            entry = groups_from_w(w, out_idx=out_i)[gi][pi]
            orig_vals = np.array(entry['vals'], dtype=w.dtype)
            a0, b0 = int(orig_posenc[:2], 2), int(orig_posenc[2:], 2)
            a1, b1 = int(att_posenc[:2], 2), int(att_posenc[2:], 2)
            stored = [orig_vals[i] for i in [a0, b0]]
            new_group_vals = np.zeros(4, dtype=orig_vals.dtype)
            for i, idx in enumerate([a1, b1]):
                new_group_vals[idx] = stored[i]

            w_att = w.copy()
            w_att[out_i].reshape(w.shape[1], -1)[s:e, pi] = new_group_vals.astype(w_att.dtype)

            st_att = {k: v.clone() for k, v in state.items()} if isinstance(state, dict) else state
            st_att[param_key] = torch.from_numpy(w_att).type_as(st_att[param_key])
            model_best_att = resnet20_nm()
            model_best_att.load_state_dict(st_att)
            model_best_att.to(device).eval()
            acc_after = evaluate_full_test_accuracy(model_best_att, device)

            # commit the attack to `state` so subsequent iterations use updated weights
            state[param_key] = torch.from_numpy(w_att).type_as(state[param_key])
            # record that this group's been attacked so it's skipped next iterations
            attacked_groups.add((param_key, out_i, gi, pi))

            # Print iteration-labeled Best attack results.
            # If this is the first iteration, label it explicitly as the First attack.
            # First line: attack metadata (param_key, channel, group, pos, posencs, proxy score, losses, delta).
            # Following lines: full-testset inference accuracies printed on separate lines
            # for clarity (acc_before then acc_after).
            if iter_i == 0:
                print('\nFirst attack:')
            else:
                print(f"\nBest attack (iter {iter_i}):")
            print('param_key,out_ch,group,pos,orig_posenc,att_posenc,proxy_score,loss_orig,loss_att,delta')
            print(f"{param_key},{out_i},{gi},{pi},{orig_posenc},{att_posenc},{cand_score:.6e},{loss_o:.6f},{loss_a:.6f},{delta:.6f}")
            # Full-testset accuracies (separate lines):
            print('acc_before')
            print(f"{acc_before:.6f}")
            print('acc_after')
            print(f"{acc_after:.6f}")
        else:
            print('No candidates evaluated')
            # if nothing found this iteration, break early
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NM positive attack tools')
    parser.add_argument('--print-groups', action='store_true', help='Only print weight groups and position encodings for output channel 0')
    parser.add_argument('--ckpt', default='1_sparse_finetune.pth', help='Checkpoint path')
    parser.add_argument('--attack-00', action='store_true', help='Run the four specified attacks on group 0 pos 0 and print loss changes')
    parser.add_argument('--nca-all', action='store_true', help='Run non-collision attacks for all 36 (group,pos) of out[0] and print losses')
    parser.add_argument('--nca-layer-best', action='store_true', help='Run single-bit NCA across entire layer and print the single attack with max loss increase')
    parser.add_argument('--nca-proxy', action='store_true', help='Run two-stage proxy NCA and only print the final Best attack')
    parser.add_argument('--nca-proxy-best-only', action='store_true', help='Run two-stage proxy NCA but only print the final Best attack')
    parser.add_argument('--layers', default='layer1,layer2,layer3', help='Comma-separated layer prefixes to search (e.g. layer1,layer2,layer3)')
    parser.add_argument('--proxy-topG', type=int, default=200, help='Number of groups to keep after group-level ranking')
    parser.add_argument('--per-group-K', type=int, default=2, help='Number of candidates per selected group to evaluate')
    parser.add_argument('--n-iter', type=int, default=1, help='Number of iterative attack rounds to run; skip groups already attacked')
    args = parser.parse_args()

    if args.print_groups:
        def print_groups_only(ckpt_path: str = '1_sparse_finetune.pth') -> None:
            state = load_state(ckpt_path)
            key = 'layer1.0.conv1.weight'
            if key not in state:
                raise KeyError(f'{key} not found')

            w = state[key].cpu().numpy().copy()
            groups = groups_from_w(w)
            print('group,pos,posenc,vals')
            for gi, group in enumerate(groups):
                for pi, entry in enumerate(group):
                    posenc = entry['posenc']
                    vals = entry['vals']
                    vals_str = ' '.join([f"{v:.3f}" for v in vals])
                    print(f"{gi},{pi},{posenc},[{vals_str}]")

        print_groups_only(args.ckpt)
    elif args.attack_00:
        measure_loss_change_for_group_pos(args.ckpt, attack_group=0, attack_pos=0, orig_posenc='0110', new_posencs=['1110', '0010', '0100', '0111'])
    elif args.nca_all:
        run_nca_all(args.ckpt)
    elif args.nca_layer_best:
        run_nca_layer_best(args.ckpt)
    elif args.nca_proxy:
        run_nca_proxy(args.ckpt, topG=args.proxy_topG, per_group_k=args.per_group_K, layer_prefixes=args.layers.split(','), n_iter=args.n_iter)
    elif args.nca_proxy_best_only:
        run_nca_proxy(args.ckpt, topG=args.proxy_topG, per_group_k=args.per_group_K, print_candidates=False, layer_prefixes=args.layers.split(','), n_iter=args.n_iter)
    else:
        compute_and_print()
