import os
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn

from zhwf_06_nm_pos_attack import load_state, group_layer1_conv1
from resnet20_nm import resnet20_nm


def simulate_flip_full(ckpt_path='1_sparse_finetune.pth', group=0, pos=0, orig_posenc='0110', new_posenc='0111', batch_size=128, data_root='~/data/cifar10'):
    state = load_state(ckpt_path)
    key = 'layer1.0.conv1.weight'
    if key not in state:
        raise KeyError(f'{key} not found in checkpoint')

    w = state[key].cpu().numpy().copy()
    out_ch, in_ch, kH, kW = w.shape

    groups = group_layer1_conv1(ckpt_path, pos_flag=False, mask_flag=False)
    vals = groups[group][pos]['vals']

    a0, b0 = int(orig_posenc[:2], 2), int(orig_posenc[2:], 2)
    a1, b1 = int(new_posenc[:2], 2), int(new_posenc[2:], 2)
    orig_idxs = [a0, b0]
    new_idxs = [a1, b1]

    stored = [vals[i] for i in orig_idxs]
    new_group_vals = [0.0] * 4
    for i, idx in enumerate(new_idxs):
        new_group_vals[idx] = stored[i]

    w_att = w.copy()
    s = group * 4
    e = s + 4
    w_att[0].reshape(in_ch, -1)[s:e, pos] = np.array(new_group_vals, dtype=w_att.dtype)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_orig = resnet20_nm()
    model_att = resnet20_nm()

    st_orig = {k: v.clone() for k, v in state.items()} if isinstance(state, dict) else state
    st_att = {k: v.clone() for k, v in state.items()} if isinstance(state, dict) else state
    st_att[key] = torch.from_numpy(w_att)
    model_orig.load_state_dict(st_orig)
    model_att.load_state_dict(st_att)
    model_orig = model_orig.to(device).eval()
    model_att = model_att.to(device).eval()

    data_root = os.path.expanduser(data_root)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    testset = datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform)
    loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    total = 0
    correct_orig = 0
    correct_att = 0
    loss_orig_sum = 0.0
    loss_att_sum = 0.0

    with torch.no_grad():
        for imgs, tgts in loader:
            imgs = imgs.to(device)
            tgts = tgts.to(device)
            out_o = model_orig(imgs)
            out_a = model_att(imgs)

            loss_orig_sum += float(criterion(out_o, tgts).item())
            loss_att_sum += float(criterion(out_a, tgts).item())

            pred_o = out_o.argmax(1)
            pred_a = out_a.argmax(1)
            correct_orig += int(pred_o.eq(tgts).sum().item())
            correct_att += int(pred_a.eq(tgts).sum().item())
            total += imgs.size(0)

    acc_orig = 100.0 * correct_orig / total
    acc_att = 100.0 * correct_att / total
    loss_orig = loss_orig_sum / total
    loss_att = loss_att_sum / total

    print('Full-dataset simulation: flip posenc', orig_posenc, '->', new_posenc, f'for Group{group},pos{pos}')
    print(f'Total samples: {total} Device: {device}')
    print(f'Original loss={loss_orig:.6f} acc={acc_orig:.2f}%')
    print(f'Attacked  loss={loss_att:.6f} acc={acc_att:.2f}%')

    return {
        'loss_orig': loss_orig,
        'acc_orig': acc_orig,
        'loss_att': loss_att,
        'acc_att': acc_att,
    }


if __name__ == '__main__':
    res = simulate_flip_full()
    print('Result:', res)
