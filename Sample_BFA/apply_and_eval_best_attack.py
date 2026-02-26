import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from zhwf_06_nm_pos_attack import load_state
from resnet20_nm import resnet20_nm
from zhwf_10_nm_pos_attack import groups_from_w


collision_set = {'0000', '0101', '1010', '1111'}


def find_best_single_bit_attack(ckpt_path: str = '1_sparse_finetune.pth', device: str = 'cpu'):
    state = load_state(ckpt_path)
    key = 'layer1.0.conv1.weight'
    if key not in state:
        raise KeyError(f'{key} not found')
    w = state[key].cpu().numpy().copy()
    out_ch = w.shape[0]

    device_t = torch.device(device)
    model_orig = resnet20_nm()
    st_orig = {k: v.clone() for k, v in state.items()} if isinstance(state, dict) else state
    model_orig.load_state_dict(st_orig)
    model_orig.to(device_t).eval()

    # single batch (same as nca code)
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
    inputs = inputs.to(device_t)
    targets = targets.to(device_t)

    with torch.no_grad():
        out_orig = model_orig(inputs)
        loss_orig = float(criterion(out_orig, targets).item())

    best = None
    best_delta = -1e9

    # search identical to run_nca_layer_best but return best tuple
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
                    model_att.to(device_t).eval()

                    with torch.no_grad():
                        out_att = model_att(inputs)
                        loss_att = float(criterion(out_att, targets).item())

                    delta = loss_att - loss_orig
                    if delta > best_delta:
                        best_delta = delta
                        best = dict(out_ch=out_i, group=gi, pos=pi, orig_posenc=orig_posenc, att_posenc=flipped_s, loss_orig=loss_orig, loss_att=loss_att, delta=delta)

    return best


def evaluate_full_test_accuracy(model: torch.nn.Module, device: torch.device):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    testset = datasets.CIFAR10(root=os.path.expanduser('~/data/cifar10'), train=False, download=False, transform=test_transform)
    loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            out = model(inputs)
            preds = out.argmax(dim=1)
            correct += int((preds == targets).sum().item())
            total += targets.size(0)
    return correct / total


def apply_attack_and_save(ckpt_path: str, best: dict, out_path: str = None, device: str = 'cpu'):
    state = load_state(ckpt_path)
    key = 'layer1.0.conv1.weight'
    w = state[key].cpu().numpy().copy()

    out_i = best['out_ch']
    gi = best['group']
    pi = best['pos']
    orig_posenc = best['orig_posenc']
    att_posenc = best['att_posenc']

    groups_before = groups_from_w(w, out_idx=out_i)
    entry = groups_before[gi][pi]
    vals = entry['vals']

    a0, b0 = int(orig_posenc[:2], 2), int(orig_posenc[2:], 2)
    a1, b1 = int(att_posenc[:2], 2), int(att_posenc[2:], 2)
    stored = [vals[i] for i in [a0, b0]]
    new_group_vals = [0.0] * 4
    for i, idx in enumerate([a1, b1]):
        new_group_vals[idx] = stored[i]

    s = gi * 4
    e = s + 4
    w_att = w.copy()
    w_att[out_i].reshape(w.shape[1], -1)[s:e, pi] = np.array(new_group_vals, dtype=w_att.dtype)

    st_att = {k: v.clone() for k, v in state.items()} if isinstance(state, dict) else state
    st_att[key] = torch.from_numpy(w_att).type_as(st_att[key])

    if out_path is None:
        base = os.path.splitext(ckpt_path)[0]
        out_path = base + '_attacked.pth'

    torch.save(st_att, out_path)
    return out_path, entry['vals'], new_group_vals


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='1_sparse_finetune.pth')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = args.device
    print('Searching for best single-bit non-collision attack (single-batch estimate)...')
    best = find_best_single_bit_attack(args.ckpt, device=device)
    if best is None:
        print('No valid attack found')
        raise SystemExit(1)

    print('Best candidate (single-batch estimate):')
    print(best)

    # prepare original model for full-test eval
    state = load_state(args.ckpt)
    model_orig = resnet20_nm()
    st_orig = {k: v.clone() for k, v in state.items()} if isinstance(state, dict) else state
    model_orig.load_state_dict(st_orig)
    model_orig.to(device)

    acc_before = evaluate_full_test_accuracy(model_orig, torch.device(device))

    # apply attack and save
    out_path, before_vals, after_vals = apply_attack_and_save(args.ckpt, best, device=device)

    # load attacked model and eval
    st_att = torch.load(out_path)
    model_att = resnet20_nm()
    model_att.load_state_dict(st_att)
    model_att.to(device)
    acc_after = evaluate_full_test_accuracy(model_att, torch.device(device))

    # print required info
    print('\n--- Result Summary ---')
    print('Checkpoint:', args.ckpt)
    print('Applied attack: out_ch=%d, group=%d, pos=%d' % (best['out_ch'], best['group'], best['pos']))
    print('orig_posenc -> att_posenc: %s -> %s' % (best['orig_posenc'], best['att_posenc']))
    print('Before attack group 4 weights:', ' '.join([f"{v:.6f}" for v in before_vals]))
    print('After attack group 4 weights:', ' '.join([f"{v:.6f}" for v in after_vals]))
    print('Single-batch loss_orig (used during search):', best['loss_orig'])
    print('Single-batch loss_att  (used during search):', best['loss_att'])
    print('Estimated delta (single-batch):', best['delta'])
    print(f'Full-test accuracy before attack: {acc_before*100:.2f}%')
    print(f'Full-test accuracy after  attack: {acc_after*100:.2f}%')
    print('Attacked checkpoint saved to:', out_path)
