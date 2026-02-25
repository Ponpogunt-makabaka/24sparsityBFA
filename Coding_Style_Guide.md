# Coding Style Guide: sample_bfa Reference

**Based on:** `sample_bfa/` directory analysis
**Date:** 2026-02-24
**Purpose:** Define coding conventions for all new `run_R1_Txx*.py` scripts

---

## Table of Contents

1. [File Structure and Imports](#1-file-structure-and-imports)
2. [Naming Conventions](#2-naming-conventions)
3. [Type Hinting and Docstrings](#3-type-hinting-and-docstrings)
4. [Argument Parsing](#4-argument-parsing)
5. [Device Management](#5-device-management)
6. [Random Seed Handling](#6-random-seed-handling)
7. [Logging and Output](#7-logging-and-output)
8. [Exact Verification Pattern](#8-exact-verification-pattern)
9. [Gradient Computation](#9-gradient-computation)
10. [Data Loading](#10-data-loading)
11. [Checkpoint Loading](#11-checkpoint-loading)

---

## 1. File Structure and Imports

### 1.1 Import Order

```python
# 1. Standard library imports
import os
import copy
import random

# 2. Third-party imports
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader

# 3. Local imports (from same project)
from resnet20_nm import resnet20_nm
from sparse_ops import SparseConv, SparseLinear
from config import MODEL, DATASET
```

### 1.2 File Header

```python
"""
Brief description of what this script does.

More detailed explanation if needed.
"""
```

### 1.3 Import Flexibility

Use try-except for optional local imports:

```python
try:
    from resnet20_nm import quan_Conv2d, quan_Linear, quantize
except Exception:
    from models.quantization import quan_Conv2d, quan_Linear, quantize
```

---

## 2. Naming Conventions

### 2.1 Variables

- **snake_case** for all variables
- Descriptive names preferred over abbreviations

```python
attack_sample_size = 128
attack_batch_size = 128
layer_modules = []
applied_history = []
applied_set = set()  # Use descriptive set names
```

### 2.2 Functions

- **snake_case** for function names
- Verb-noun structure for actions

```python
def forward_and_compute_gradients(model, data_loader, criterion, attack_sample_size, device):
    pass

def load_and_quantize_ckpt(ckpt_path, data_root, batch_size, device):
    pass

def evaluate_top1(model, loader, device):
    pass
```

### 2.3 Classes

- **PascalCase** for class names

```python
class AverageMeter(object):
    pass

class RecorderMeter(object):
    pass

class BFA(object):
    pass
```

### 2.4 Constants

- **UPPER_CASE** for module-level constants

```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
```

---

## 3. Type Hinting and Docstrings

### 3.1 Type Hints

Type hints use simple `:` syntax (not from `typing` module where possible):

```python
def forward_and_compute_gradients(
    model,
    data_loader,
    criterion,
    attack_sample_size: int = 128,
    attack_batch_size: int = 128,
    device: str = "cuda",
):
    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    # ...
```

For class constructors:

```python
class SparseLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, N=2, M=4, decay = 0.0002, **kwargs):
        # ...
```

### 3.2 Docstrings

Use triple-quoted strings with brief description:

```python
def int2bin(input, num_bits):
    '''
    convert the signed integer value into unsigned integer (2's complement equivalently).
    Note that, the conversion is different depends on number of bit used.
    '''
    output = input.clone()
    if num_bits == 1:
        output = output/2 + .5
    # ...
```

For classes:

```python
class SparseConv(nn.Conv2d):
    """" implement N:M sparse convolution layer """
    pass
```

---

## 4. Argument Parsing

### 4.1 Argparse Structure

Create parser at the bottom of the script (in `if __name__ == "__main__"`):

```python
if __name__ == "__main__":
    # parse args for iterative attacks
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', type=int, default=10, help='number of attack iterations')
    parser.add_argument('--attack_sample_size', type=int, default=128, help='number of samples used to evaluate flips')
    parser.add_argument('--topk', type=int, default=20, help='number of top gradient indices per layer')
    args = parser.parse_args()
    n_iter = args.n_iter
    attack_sample_size = args.attack_sample_size
```

### 4.2 Argument Naming Convention

- Use `--snake-case` for argument names
- Include `help` parameter for all arguments
- Specify `type` explicitly

```python
parser.add_argument('--n_iter', type=int, default=10, help='number of attack iterations')
parser.add_argument('--attack_sample_size', type=int, default=128, help='number of samples used to evaluate flips')
parser.add_argument('--topk', type=int, default=20, help='top-k candidates per layer')
```

---

## 5. Device Management

### 5.1 Device Detection Pattern

Always check CUDA availability before using:

```python
device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
model = model.to(device)
```

### 5.2 Fallback Pattern

```python
device = "cuda"
if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"
```

### 5.3 Model Device Placement

```python
model_fp = model.to(device).eval()
```

---

## 6. Random Seed Handling

### 6.1 Config-Based Seed

Seeds are defined in `config.py`:

```python
# config.py
SEED = 42
```

### 6.2 Manual Seed Setting

When reproducibility is needed:

```python
import random
import torch
import numpy as np

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
```

---

## 7. Logging and Output

### 7.1 Print Statement Format

Use f-strings with clear formatting:

```python
print(f"Loaded checkpoint: {ckpt_path}\nDevice: {device}\nSamples: {total}\nFP32 Top-1: {fp32_top1:.2f}%\nINT8 Top-1: {int8_top1:.2f}%")
```

### 7.2 Iteration Progress

```python
for it in range(n_iter):
    print(f"\n=== Iteration {it+1}/{n_iter} ===")
    # ...
    print(f"Global best flip: module={module_name} index={multi_idx} delta={delta:+.6f}")
    print(f"INT8 before: {orig_int8} | INT8 (after flip): {cand_int8}")
    print(f"Applied flip. Baseline Top-1: {baseline_top1:.2f}%, After flip: {flipped_top1:.2f}%, Drop: {baseline_top1 - flipped_top1:.2f}%")
```

### 7.3 Section Headers

```python
print("\n=== Iteration {it+1}/{n_iter} ===")
print('No positive-delta candidate flips found this iteration; stopping.')
```

---

## 8. Exact Verification Pattern

### 8.1 Save-Apply-Restore Pattern

```python
# Save original value
orig_val = param.data[multi_idx].item()

# Apply candidate flip
param.data[multi_idx] = new_val
new_output = model(test_imgs)
new_loss = criterion(new_output, test_tgts).item()

# Restore original value
param.data[multi_idx] = orig_val

# Compute delta
delta = new_loss - loss_val
```

### 8.2 Using torch.no_grad()

```python
with torch.no_grad():
    new_val = float(flipped_q) * step_size
    param = None
    for name, p in model_attack.named_parameters():
        if name == target_name:
            param = p
            break

    if param is not None:
        orig_val = param.data[multi_idx].item()
        param.data[multi_idx] = new_val
        # ... verification ...
        param.data[multi_idx] = orig_val
```

### 8.3 Candidate Evaluation Loop

```python
layer_entries = []
topk = min(args.topk, indices.numel())
for i in range(topk):
    idx = indices[i].item()
    grad_val = g_flat[idx].item()

    # Compute multi-dimensional index
    dims = g.shape
    multi_idx = []
    temp_idx = idx
    for d in reversed(dims):
        multi_idx.append(temp_idx % d)
        temp_idx //= d
    multi_idx = tuple(reversed(multi_idx))

    # Exact verification
    orig_q = q_weights[multi_idx].item()
    flipped_q = orig_q - 128 if orig_q >= 0 else orig_q + 128

    # Save-apply-restore pattern
    with torch.no_grad():
        new_val = float(flipped_q) * step_size
        param.data[multi_idx] = new_val
        new_output = model(test_imgs)
        new_loss = criterion(new_output, test_tgts).item()
        param.data[multi_idx] = orig_val

    delta = new_loss - loss_val

    # Filter candidates
    key = (module_name, tuple(multi_idx))
    if key in applied_set:
        continue
    if delta <= 0:
        continue

    layer_entries.append({...})
```

---

## 9. Gradient Computation

### 9.1 Forward and Compute Gradients Function

```python
def forward_and_compute_gradients(
    model,
    data_loader,
    criterion,
    attack_sample_size: int = 128,
    attack_batch_size: int = 128,
    device: str = "cuda",
):
    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model = model.to(device)
    model.eval()
    model.zero_grad()

    # Collect batch
    all_imgs = []
    all_tgts = []
    processed = 0
    for images, targets in data_loader:
        if processed >= attack_sample_size:
            break
        rem = attack_sample_size - processed
        all_imgs.append(images[:rem])
        all_tgts.append(targets[:rem])
        processed += all_imgs[-1].size(0)

    full_images = torch.cat(all_imgs).to(device)
    full_targets = torch.cat(all_tgts).to(device)

    # Forward and backward
    outputs = model(full_images)
    loss = criterion(outputs, full_targets)
    loss.backward()

    acc_loss = loss.item()

    # Extract gradients
    grads = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if "step_size" in name:
            continue
        grads[name] = p.grad.detach().cpu().clone()

    return acc_loss, grads
```

### 9.2 Gradient-Based Ranking

```python
g_flat = g.flatten()
abs_sorted, indices = torch.sort(g_flat.abs(), descending=True)
```

---

## 10. Data Loading

### 10.1 Dataset Transforms

```python
# Test transform (no augmentation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

# Train/Attack transform (with augmentation)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
```

### 10.2 DataLoader Pattern

```python
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

loader = DataLoader(
    datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform),
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
```

### 10.3 Batch Collection Pattern

```python
all_imgs = []
all_tgts = []
processed = 0
for images, targets in data_loader:
    if processed >= attack_sample_size:
        break
    rem = attack_sample_size - processed
    all_imgs.append(images[:rem])
    all_tgts.append(targets[:rem])
    processed += all_imgs[-1].size(0)

full_images = torch.cat(all_imgs).to(device)
full_targets = torch.cat(all_tgts).to(device)
```

---

## 11. Checkpoint Loading

### 11.1 Flexible Checkpoint Loading

```python
ckpt = torch.load(ckpt_path, map_location="cpu")
if isinstance(ckpt, dict):
    for key in ("state_dict", "model_state_dict", "model", "model_state"):
        if key in ckpt:
            state_dict = ckpt[key]
            break
    else:
        state_dict = ckpt
else:
    state_dict = ckpt

# Remove "module." prefix if present
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
```

### 11.2 Quantization Pattern

```python
layer_info = {}
quant_types = (torch.nn.Conv2d, torch.nn.Linear, SparseConv, SparseLinear)
for name, module in model_q.named_modules():
    if isinstance(module, quant_types) and hasattr(module, 'weight') and module.weight is not None:
        w = module.weight.data.cpu()
        max_abs = float(w.abs().max().item())
        step = max_abs / 127.0 if max_abs > 0 else 1e-8
        q = torch.clamp(torch.round(w / step), -127, 127).to(torch.int8)
        module.weight.data = (q.float() * step).to(module.weight.data.device)
        layer_name = name if name else module.__class__.__name__
        layer_info[layer_name] = {'step_size': float(step), 'quantized_int8': q.cpu()}
```

---

## 12. Attack Loop Pattern

### 12.1 Iterative Attack Structure

```python
# Initialize
applied_history = []
applied_set = set()  # set of (module_name, multi_idx) already flipped

for it in range(n_iter):
    print(f"\n=== Iteration {it+1}/{n_iter} ===")

    # Compute gradients
    loss_val, grads = forward_and_compute_gradients(model_attack, attack_loader, criterion, attack_sample_size, attack_batch_size)

    # Ensure test batch is on same device as model
    param_dev = next(model_attack.parameters()).device
    test_imgs = test_imgs.to(param_dev)
    test_tgts = test_tgts.to(param_dev)

    # Collect candidates
    global_entries = []
    for module_name in layer_modules:
        # ... per-layer candidate enumeration ...
        if len(layer_entries) > 0:
            layer_max = max(layer_entries, key=lambda e: abs(e['delta']))
            global_entries.append(layer_max)

    # Check if any candidates found
    if len(global_entries) == 0:
        print('No positive-delta candidate flips found this iteration; stopping.')
        break

    # Select global best
    global_max = max(global_entries, key=lambda e: e['delta'])

    # Evaluate baseline
    baseline_top1 = evaluate_top1(model_attack, test_loader, device)

    # Apply flip permanently
    # ... apply code ...

    # Evaluate after flip
    flipped_top1 = evaluate_top1(model_attack, test_loader, device)
    print(f"Applied flip. Baseline Top-1: {baseline_top1:.2f}%, After flip: {flipped_top1:.2f}%, Drop: {baseline_top1 - flipped_top1:.2f}%")
```

### 12.2 Sign-Flip Pattern

```python
orig_q = q_weights[multi_idx].item()
flipped_q = orig_q - 128 if orig_q >= 0 else orig_q + 128
```

---

## 13. Evaluation Pattern

### 13.1 Top-1 Accuracy Evaluation

```python
def evaluate_top1(model, loader, device):
    model = model.to(device).eval()
    correct = total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            preds = model(images).argmax(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return 100.0 * correct / total if total else 0.0
```

---

## 14. Complete Script Template

```python
"""
R1_TXX: [Brief description of what this attack does]
"""

import os
import copy
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from resnet20_nm import resnet20_nm
from sparse_ops import SparseConv, SparseLinear
import torch.nn as nn
import argparse


def forward_and_compute_gradients(
    model,
    data_loader,
    criterion,
    attack_sample_size: int = 128,
    attack_batch_size: int = 128,
    device: str = "cuda",
):
    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model = model.to(device)
    model.eval()
    model.zero_grad()

    all_imgs = []
    all_tgts = []
    processed = 0
    for images, targets in data_loader:
        if processed >= attack_sample_size:
            break
        rem = attack_sample_size - processed
        all_imgs.append(images[:rem])
        all_tgts.append(targets[:rem])
        processed += all_imgs[-1].size(0)

    full_images = torch.cat(all_imgs).to(device)
    full_targets = torch.cat(all_tgts).to(device)

    outputs = model(full_images)
    loss = criterion(outputs, full_targets)
    loss.backward()

    acc_loss = loss.item()

    grads = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if "step_size" in name:
            continue
        grads[name] = p.grad.detach().cpu().clone()

    return acc_loss, grads


def load_and_quantize_ckpt(ckpt_path="model.pth", data_root="~/data/cifar10", batch_size=128, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    data_root = os.path.expanduser(data_root)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    loader = DataLoader(datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = resnet20_nm()
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "model_state"):
            if key in ckpt:
                state_dict = ckpt[key]
                break
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    model_fp = model.to(device).eval()
    correct = total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            preds = model_fp(images).argmax(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    fp32_top1 = 100.0 * correct / total if total else 0.0

    model_q = resnet20_nm()
    model_q.load_state_dict(state_dict)

    layer_info = {}
    quant_types = (torch.nn.Conv2d, torch.nn.Linear, SparseConv, SparseLinear)
    for name, module in model_q.named_modules():
        if isinstance(module, quant_types) and hasattr(module, 'weight') and module.weight is not None:
            w = module.weight.data.cpu()
            max_abs = float(w.abs().max().item())
            step = max_abs / 127.0 if max_abs > 0 else 1e-8
            q = torch.clamp(torch.round(w / step), -127, 127).to(torch.int8)
            module.weight.data = (q.float() * step).to(module.weight.data.device)
            layer_name = name if name else module.__class__.__name__
            layer_info[layer_name] = {'step_size': float(step), 'quantized_int8': q.cpu()}

    model_q = model_q.to(device).eval()
    correct_q = total_q = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            preds = model_q(images).argmax(1)
            correct_q += preds.eq(targets).sum().item()
            total_q += targets.size(0)
    int8_top1 = 100.0 * correct_q / total_q if total_q else 0.0

    print(f"Loaded checkpoint: {ckpt_path}\nDevice: {device}\nSamples: {total}\nFP32 Top-1: {fp32_top1:.2f}%\nINT8 Top-1: {int8_top1:.2f}%")

    return fp32_top1, int8_top1, layer_info


def evaluate_top1(model, loader, device):
    model = model.to(device).eval()
    correct = total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            preds = model(images).argmax(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return 100.0 * correct / total if total else 0.0


if __name__ == "__main__":
    fp, iq, info = load_and_quantize_ckpt()

    attack_sample_size = 128
    attack_batch_size = 128

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    trainset = datasets.CIFAR10(root=os.path.expanduser("~/data/cifar10"), train=True, download=False, transform=train_transform)
    attack_loader = DataLoader(trainset, batch_size=attack_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model_attack = resnet20_nm()
    # ... load checkpoint ...

    criterion = nn.CrossEntropyLoss()

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', type=int, default=10, help='number of attack iterations')
    parser.add_argument('--attack_sample_size', type=int, default=128, help='number of samples used to evaluate flips')
    parser.add_argument('--topk', type=int, default=20, help='number of top gradient indices per layer')
    args = parser.parse_args()

    # ... attack loop ...
```

---

## 15. Key Patterns Summary

| Pattern | Description |
|---------|-------------|
| `model.eval()` | Always set model to eval mode before attack |
| `model.zero_grad()` | Clear gradients before backward pass |
| `with torch.no_grad():` | Use for inference and value modifications |
| `p.grad.detach().cpu().clone()` | Extract gradients to CPU for processing |
| `orig_q - 128 if orig_q >= 0 else orig_q + 128` | Sign-flip operation |
| `set()` tracking | Use set to track already-applied flips |
| f-strings | Use f-strings for formatted output |
| `:+.6f` | Format for signed float with 6 decimals |
| `{it+1}/{n_iter}` | Show iteration progress |
| `device` fallback | Always check CUDA availability |

---

**End of Coding Style Guide**
