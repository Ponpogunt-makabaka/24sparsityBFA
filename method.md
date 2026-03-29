# Method: Bit-Flip Attack on 2:4 Structured Sparse Neural Networks

## Table of Contents

- [1. Attack Theory](#1-attack-theory)
  - [1.1 Threat Model](#11-threat-model)
  - [1.2 2:4 Structured Sparsity Representation](#12-24-structured-sparsity-representation)
  - [1.3 INT8 Quantization Model](#13-int8-quantization-model)
  - [1.4 Two Attack Surfaces](#14-two-attack-surfaces)
  - [1.5 Gradient-Guided Search Principle](#15-gradient-guided-search-principle)
  - [1.6 Directional Proxy Scoring](#16-directional-proxy-scoring)
- [2. Attack Algorithm Design](#2-attack-algorithm-design)
  - [2.1 Algorithm Overview](#21-algorithm-overview)
  - [2.2 Classic BFA: Weight Magnitude Bit-Flip (T04)](#22-classic-bfa-weight-magnitude-bit-flip-t04)
  - [2.3 Metadata BFA: Index Encoding Attack (T08/T09)](#23-metadata-bfa-index-encoding-attack-t08t09)
  - [2.4 Enhanced Hybrid Search: T10_Enhanced](#24-enhanced-hybrid-search-t10_enhanced)
  - [2.5 Attack Variants](#25-attack-variants)
- [3. Implementation Details](#3-implementation-details)
  - [3.1 Index Encoding Scheme](#31-index-encoding-scheme)
  - [3.2 Group Flattening (K-Dimension Semantics)](#32-group-flattening-k-dimension-semantics)
  - [3.3 Candidate Generation and Filtering](#33-candidate-generation-and-filtering)
  - [3.4 Save-Apply-Restore Verification Pattern](#34-save-apply-restore-verification-pattern)
  - [3.5 State Tracking and Anti-Cycling](#35-state-tracking-and-anti-cycling)
  - [3.6 Architecture-Aware Layer Selection](#36-architecture-aware-layer-selection)
- [4. Sparsification Pipeline (T09)](#4-sparsification-pipeline-t09)
  - [4.1 Mask Computation](#41-mask-computation)
  - [4.2 Fine-Tuning with Mask Enforcement](#42-fine-tuning-with-mask-enforcement)
- [5. Defense Mechanism](#5-defense-mechanism)
  - [5.1 Reference-Guided Position Repair](#51-reference-guided-position-repair)
  - [5.2 Zero-Only Attack Constraint](#52-zero-only-attack-constraint)
- [6. Experimental Matrix](#6-experimental-matrix)

---

## 1. Attack Theory

### 1.1 Threat Model

The attacker has **physical access to the memory** where neural network weights are stored (e.g., DRAM). Using techniques such as Rowhammer, the attacker can flip individual bits in the binary representation of quantized weights. The attack goal is to **maximize model accuracy degradation** with a minimal number of bit-flips (physical budget).

**Key assumptions**:
- The model is deployed with INT8 quantization (weights stored as 8-bit integers).
- The model uses 2:4 structured sparsity, where each group of 4 consecutive weights along the K-dimension has exactly 2 non-zero values.
- The attacker has access to a small calibration dataset (e.g., 256 samples) to guide the search.
- The attacker can perform forward and backward passes on the model.

### 1.2 2:4 Structured Sparsity Representation

NVIDIA Ampere GPUs natively accelerate 2:4 sparsity through Sparse Tensor Cores. In this format, each contiguous group of 4 weights along the reduction dimension (K) contains exactly 2 non-zero and 2 zero values. The hardware stores:

1. **Non-zero values**: The 2 magnitude values (INT8).
2. **Index metadata**: A compact encoding indicating **which 2 of the 4 positions** are active.

This creates two distinct attack surfaces: the magnitude values themselves, and the index metadata that encodes the sparsity pattern.

```
Physical Memory Layout (per group of 4):
  ┌─────────────────────────────────┐
  │  value_0 (INT8)  │  value_1 (INT8)  │  index_metadata (4-bit)  │
  └─────────────────────────────────┘

Example: group = [0, 3.5, 0, -1.2]
  Active positions: (1, 3)
  Stored values: [3.5, -1.2]
  Index code: encode(1, 3) = (3 << 2) | 1 = 0b1101 = 13
```

### 1.3 INT8 Quantization Model

Weights are quantized using per-layer symmetric quantization:

```
scale = max(|W|) / 127
W_int8 = clamp(round(W / scale), -127, 127)
W_reconstructed = W_int8 * scale
```

A single bit-flip in INT8 two's complement representation can cause a large magnitude change. For example, flipping the MSB (sign bit) of value `q` maps `q → q ± 128`:

```
If q >= 0: flipped_q = q - 128
If q < 0:  flipped_q = q + 128
```

This is the **maximum damage** a single bit can inflict on a weight value.

### 1.4 Two Attack Surfaces

This project implements attacks on both surfaces:

| Attack Surface | Target | Effect | Constraint |
|---|---|---|---|
| **Weight Magnitude** (Classic BFA) | INT8 value bits | Changes magnitude of non-zero weight | Sparsity pattern unchanged |
| **Index Metadata** (Metadata BFA) | 4-bit index encoding | Relocates non-zero values to different positions | Magnitudes preserved, pattern changes |

**Metadata attack** is the primary focus of this project. By flipping a single bit in the 4-bit index encoding, the attacker changes which positions within a group of 4 are active, effectively "moving" the non-zero weights to different positions in the weight tensor.

### 1.5 Gradient-Guided Search Principle

The core principle is to use loss gradients to identify which bit-flip will cause the largest increase in model loss (and thus the largest accuracy degradation).

Given the loss function $\mathcal{L}$, the first-order Taylor approximation of loss change from a weight perturbation $\Delta w$ is:

$$\Delta \mathcal{L} \approx \nabla_w \mathcal{L} \cdot \Delta w$$

This means:
- Weights with **large gradient magnitude** are sensitive — small changes cause large loss changes.
- The **sign alignment** between gradient and perturbation direction matters — perturbations aligned with the gradient direction increase loss.

### 1.6 Directional Proxy Scoring

For metadata attacks, the perturbation is not a simple magnitude change but a **pattern relocation**. The proxy score for a candidate pattern transition is:

```
proxy_score = grad · (w_tilde_new - w_tilde_old)
```

Where:
- `w_tilde_old`: The current reconstructed dense group (non-zero values at current positions, zeros elsewhere)
- `w_tilde_new`: The candidate dense group (same values moved to new positions)
- `grad`: The loss gradient for this group

A **positive proxy score** means this pattern change is predicted to increase loss (desirable for the attacker).

---

## 2. Attack Algorithm Design

### 2.1 Algorithm Overview

The project implements three generations of attack algorithms, each improving search efficiency and attack effectiveness:

```
T04 (Classic BFA)          T08/T09 (Metadata BFA)         T10_Enhanced (Hybrid)
┌──────────────┐           ┌──────────────────┐           ┌─────────────────────┐
│ Gradient      │           │ Coarse group      │           │ Coarse group filter  │
│ → Top-K weights│          │ → Candidates      │           │ → Candidate gen      │
│ → MSB flip    │           │ → Exact verify    │           │ → Proxy scoring      │
│ → Apply best  │           │ → Apply best      │           │ → Top-K by proxy     │
└──────────────┘           └──────────────────┘           │ → Exact verify       │
                                                           │ → Apply best         │
                                                           └─────────────────────┘
```

### 2.2 Classic BFA: Weight Magnitude Bit-Flip (T04)

**File**: `Sample_BFA/zhwf_04_nm_dense.py`

This is the baseline attack that targets INT8 weight values directly.

**Algorithm per iteration**:

```
1. Forward pass on calibration data → compute loss L
2. Backward pass → obtain gradients for all layers
3. For each attackable layer:
   a. Flatten gradient, sort by |grad| descending
   b. For top-K gradient positions:
      - Read current INT8 value q
      - Compute flipped value: q' = q ± 128 (MSB flip)
      - Temporarily apply flip, compute new loss L'
      - Record delta = L' - L (only keep positive deltas)
   c. Collect all positive-delta candidates from this layer
4. Select global best candidate across all layers by max delta
5. Permanently apply the flip and update INT8 records
6. Evaluate full test accuracy
```

> **Note on selection variants**: The original `zhwf_04_nm_dense.py` uses a per-layer max (best single candidate from each layer) before global selection. The `zhwf_04_nm_dense_non_zero.py` and `zhwf_04_nm_dense_zero_only.py` variants pool all candidates from all layers globally (`global_entries.extend(layer_entries)`) before selecting the overall best. The zero-only variant additionally filters to only accept candidates where the INT8-quantized value is currently 0.

**Key characteristics**:
- Targets any weight position regardless of current value
- MSB flip provides maximum magnitude change per bit
- Greedy: each iteration selects the single best flip
- O(n_layers × top_K) forward passes per iteration

### 2.3 Metadata BFA: Index Encoding Attack (T08/T09)

**File**: `data/T09_ImageNet_Scale/engine/run_R1_T08_metadata_improved.py`

This attack targets the 2:4 sparsity pattern metadata instead of weight magnitudes. It operates on the index encoding that specifies which 2 of 4 positions are active.

**Two-stage search per iteration**:

```
Stage A — Coarse Group Selection:
  1. Compute gradients on calibration data
  2. For each 2:4 group across all layers:
     group_score = sum(|grad[0..3]|)
  3. Sort globally, select Top-N groups (N=1000 default, 3000 for DeiT)

Stage B — Candidate Generation + Exact Verification:
  1. For each selected group:
     a. Read current pattern code (4-bit)
     b. Generate 1-bit flip candidates (up to 4 per group)
     c. Filter: reject collisions (i==j), no-change, forbidden transitions
     d. Compute proxy_score = grad · delta_w_tilde
  2. Sort all candidates by proxy_score descending
  3. Exact verification of Top-K candidates (K=64):
     - Save original state
     - Apply candidate pattern change
     - Forward pass → compute actual loss
     - Restore original state
  4. Select candidate with maximum actual loss increase
  5. Permanently apply the winning candidate
```

**Scaling to large models (T09)**:
- Supports ResNet-18, MobileNetV2, DeiT-Tiny
- Datasets: Imagenette (10-class ImageNet subset), CIFAR-100
- Architecture-aware layer selection (skip depthwise, first conv, etc.)
- Balanced calibration sampling across classes

### 2.4 Enhanced Hybrid Search: T10_Enhanced

**File**: `engine/run_R1_T10_enhanced_hybrid.py`

T10_Enhanced combines the coarse filtering efficiency of T10 with the directional proxy scoring precision of T08, organized as a 5-step pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     T10_Enhanced 5-Step Pipeline                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Step 1: Coarse Group Pre-filter                                       │
│  ┌──────────────────────────────────────────────┐                      │
│  │ For each 2:4 group in all sparse layers:     │                      │
│  │   group_score = sum(|grad[0..3]|)            │                      │
│  │ Sort globally → Top-N (N=1000)               │                      │
│  └──────────────────────────────────────────────┘                      │
│                          │                                              │
│                          ▼                                              │
│  Step 2: Candidate Generation                                          │
│  ┌──────────────────────────────────────────────┐                      │
│  │ For each Top-N group:                        │                      │
│  │   current_code = pattern_to_code(pattern)    │                      │
│  │   For bit_pos in 0..3:                       │                      │
│  │     candidate_code = current_code ^ (1 << bit_pos)                  │
│  │     Filter: collision? no-change? forbidden?  │                     │
│  │ Output: ~2700 valid candidates               │                      │
│  └──────────────────────────────────────────────┘                      │
│                          │                                              │
│                          ▼                                              │
│  Step 3: Fine Directional Proxy Scoring                                │
│  ┌──────────────────────────────────────────────┐                      │
│  │ For each candidate:                          │                      │
│  │   w_tilde_old = int8_val * scale * old_mask  │                      │
│  │   w_tilde_new = reassigned_val * scale * new_mask                   │
│  │   proxy_score = grad · (w_tilde_new - w_tilde_old)                  │
│  └──────────────────────────────────────────────┘                      │
│                          │                                              │
│                          ▼                                              │
│  Step 4: Global Top-K Selection                                        │
│  ┌──────────────────────────────────────────────┐                      │
│  │ Sort all candidates by proxy_score descending │                     │
│  │ Select Top-K (K=64)                          │                      │
│  └──────────────────────────────────────────────┘                      │
│                          │                                              │
│                          ▼                                              │
│  Step 5: Exact Forward Verification                                    │
│  ┌──────────────────────────────────────────────┐                      │
│  │ For each Top-K candidate:                    │                      │
│  │   Save(mask, int8_weights)                   │                      │
│  │   Apply candidate pattern change             │                      │
│  │   Forward pass → actual_loss                 │                      │
│  │   Restore(mask, int8_weights)                │                      │
│  │   delta = actual_loss - baseline_loss        │                      │
│  │ Select candidate with max positive delta     │                      │
│  └──────────────────────────────────────────────┘                      │
│                          │                                              │
│                          ▼                                              │
│                   Apply Best Flip                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Comparison of search heuristics**:

| Aspect | T08 | T10 (Original) | T10_Enhanced |
|---|---|---|---|
| Coarse groups N | 1000 | 64 | **1000** |
| Candidate pool | ~2700 | ~170 | **~2700** |
| Proxy scoring (Stage B) | Yes | No | **Yes** |
| Exact verification K | 64 | 100 | **64** |
| Directionality | Continuous proxy | Binary threshold | **Continuous proxy** |
| Key improvement | — | Fast coarse filter | **Coarse filter + proxy ranking** |

### 2.5 Attack Variants

The project also explores constrained attack modes:

| Variant | File | Constraint | Purpose |
|---|---|---|---|
| **Dense (all weights)** | `zhwf_04_nm_dense.py` | None | Baseline upper bound |
| **Non-zero only** | `zhwf_04_nm_dense_non_zero.py` | Only flip INT8 values != 0 | Realistic hardware fault model |
| **Zero only** | `zhwf_04_nm_dense_zero_only.py` | Only flip INT8 values == 0 (0→non-zero) | Defense analysis |
| **Metadata only** | `run_R1_T08_metadata_improved.py` | Only flip index encoding | Primary research focus |

---

## 3. Implementation Details

### 3.1 Index Encoding Scheme

Each 2:4 group has 2 active positions out of 4. The positions are encoded as a 4-bit value:

```python
def encode_index_to_4bit(i: int, j: int) -> int:
    """Encode two active positions into 4-bit code.
    Lower 2 bits: first position (i)
    Upper 2 bits: second position (j)
    """
    return (j << 2) | i    # 4-bit: [j1 j0 i1 i0]

def decode_4bit_to_index(code: int) -> Tuple[int, int]:
    return (code & 0x3, (code >> 2) & 0x3)
```

**Valid patterns** (6 total, i < j):

| Pattern (i, j) | Code | Binary |
|---|---|---|
| (0, 1) | 0b0100 = 4 | `01 00` |
| (0, 2) | 0b1000 = 8 | `10 00` |
| (0, 3) | 0b1100 = 12 | `11 00` |
| (1, 2) | 0b1001 = 9 | `10 01` |
| (1, 3) | 0b1101 = 13 | `11 01` |
| (2, 3) | 0b1110 = 14 | `11 10` |

**Collision codes** (invalid, i == j): 0b0000(0), 0b0101(5), 0b1010(10), 0b1111(15)

> **Note**: The attack engines (T08, T10_Enhanced) use `(j << 2) | i` with canonical ordering `i < j`. The defense module (`zhwf_10_nm_defense.py`) uses the **reversed** encoding `(top2[0] << 2) | top2[1]` where `top2[0] < top2[1]`, placing the smaller index in the upper bits. For a pair `(0, 3)`: attack encodes as 12 (`0b1100`), defense encodes as 3 (`0b0011`). This discrepancy means the defense and attack use **incompatible** code spaces — the defense's mismatch detection relies solely on whether the two encodings differ, not on the absolute code value, so it still functions correctly for detection, but the codes themselves are not interchangeable.

A **1-bit flip** on a valid code produces a candidate code. The candidate is accepted only if:
1. It does not decode to a collision (i == j)
2. It produces a different pattern than the current one
3. It is not in the forbidden transitions set

### 3.2 Group Flattening (K-Dimension Semantics)

The 2:4 sparsity pattern is defined along the **K-dimension** (matrix reduction dimension). The flattening strategy differs by layer type:

**Conv2d (4D tensor `[out_ch, in_ch, kH, kW]`)**:
```python
# K = in_ch × kH × kW (reduction dimension in GEMM)
# NHWC memory order: permute to [out_ch, kH, kW, in_ch]
w_perm = weight.permute(0, 2, 3, 1).contiguous()
flat = w_perm.view(-1, 4)   # Each row is one 2:4 group
# K must be divisible by 4
```

**Linear (2D tensor `[out_features, in_features]`)**:
```python
# K = in_features
flat = weight.contiguous().view(-1, 4)
```

**Restoring from flat back to original shape**:
```python
def restore_groups(flat, meta):
    if meta[0] == "conv":
        t_perm = flat.view(meta[2])               # permuted shape
        return t_perm.permute(0, 3, 1, 2).contiguous()  # back to NCHW
    return flat.view(meta[1])                      # linear: original shape
```

### 3.3 Candidate Generation and Filtering

For each selected group, candidates are generated by flipping each of the 4 bits in the index code:

```python
for bit_pos in range(4):
    candidate_code = current_code ^ (1 << bit_pos)

    # Filter 1: collision check (i == j after decode)
    candidate_pattern = code_to_pattern(candidate_code)
    if candidate_pattern is None:          # collision
        continue

    # Filter 2: no-change check
    if candidate_pattern == current_pattern:
        continue

    # Filter 3: forbidden transition check (anti-cycling)
    if (layer_name, g_idx, current_code, candidate_code) in forbidden_transitions:
        continue

    # Proxy scoring: reconstruct weight delta
    w_new_group = torch.zeros_like(w_group)
    for rank, dst_pos in enumerate(candidate_pattern):
        w_new_group[dst_pos] = old_values[rank]     # relocate values
    delta_w_tilde = w_new_group.float() * scale - w_tilde_current
    proxy_score = torch.dot(grad_group, delta_w_tilde)
```

The proxy scoring uses the **same value assignment** as the actual application: the two values from the old active positions are placed into the new pattern positions in rank order (`old_values[0]` → `new_pattern[0]`, `old_values[1]` → `new_pattern[1]`).

> **Note on `scale` factor**: The `scale` multiplication applies only to T10_Enhanced, which operates on INT8-quantized models (`w_tilde = int8_val * scale * mask`). T08/T09 operates on FP32 weight values directly, so its proxy formula is simply `proxy = grad · (w_new - w_old)` without any scale factor. Both formulas are mathematically equivalent under their respective quantization contexts.

### 3.4 Save-Apply-Restore Verification Pattern

Exact verification requires actually modifying the model weights, running a forward pass, and then restoring the original state. This pattern is critical for correctness:

```python
for candidate in topk_candidates:
    # --- Save ---
    orig_mask = m_group.clone()
    orig_weights = w_group.clone()

    # --- Apply ---
    w_group.zero_()
    m_group.zero_()
    for rank, dst_pos in enumerate(new_pattern):
        w_group[dst_pos] = old_values[rank]
        m_group[dst_pos] = 1.0
    module.sparse_mask.copy_(restore_groups(m_flat, m_meta))
    module.int8_weights.copy_(restore_groups(w_flat, w_meta))

    # --- Evaluate ---
    new_loss = criterion(model(verify_imgs), verify_tgts).item()
    delta = new_loss - baseline_loss

    # --- Restore ---
    m_group.copy_(orig_mask)
    w_group.copy_(orig_weights)
    module.sparse_mask.copy_(restore_groups(m_flat, m_meta))
    module.int8_weights.copy_(restore_groups(w_flat, w_meta))

    # --- Integrity check ---
    assert compute_model_hash(model) == model_hash_before
```

### 3.5 State Tracking and Anti-Cycling

To prevent the search from oscillating between the same pattern transitions:

**Exclude groups** (FIFO, maxlen=20): Recently attacked groups are temporarily excluded from the coarse selection to encourage diversity.

**Forbidden transitions** (FIFO, maxlen=1000): Both the forward and reverse transitions are recorded. This prevents the search from undoing a previous flip.

```python
# After applying a flip:
exclude_groups_queue.append((layer_name, group_idx))       # maxlen=20
forbidden_transitions_queue.append(forward_transition)      # maxlen=1000
forbidden_transitions_queue.append(reverse_transition)

# FIFO eviction: oldest entries are removed when queue is full,
# allowing eventual re-visit if the search gets stuck.
```

### 3.6 Architecture-Aware Layer Selection

Not all layers are eligible for 2:4 sparsity. The selection rules ensure hardware compliance:

| Architecture | Sparsified Layers | Skipped Layers | Reason |
|---|---|---|---|
| **ResNet-18** | 20 layers (conv + fc) | conv1 (in_ch=3) | K=3×7×7=147, not divisible by 4 |
| **MobileNetV2** | 35 layers (pointwise) | 17 depthwise + first conv | Depthwise: groups=in_ch, grouped conv breaks K-dim semantics |
| **DeiT-Tiny** | 49 layers (QKV, MLP, out_proj, **heads.head**) | conv_proj, cls_token, pos_embed | conv_proj skipped (in_ch=3); cls_token/pos_embed skipped implicitly (dim != 2 or 4); heads.head is **included** as a legitimate 2:4 attack surface |

The decision function (simplified composite — actual implementation has per-architecture branches for ResNet-18, MobileNetV2, DeiT-Tiny with `arch` parameter):
```python
def _should_sparsify(name, param, model, arch):
    # Skip 1: only 2D (Linear) or 4D (Conv2d) tensors
    if param.dim() not in (2, 4):
        return False
    # Skip 2 (Conv2d): depthwise convolution (groups > 1)
    if isinstance(module, nn.Conv2d) and module.groups > 1:
        return False
    # Skip 3 (Conv2d): first conv with in_channels=3 (K not divisible by 4)
    if isinstance(module, nn.Conv2d) and module.in_channels == 3:
        return False
    # Skip 4: K-dimension not divisible by 4
    if K % 4 != 0:
        return False
    # Include: classification heads (fc, classifier, heads.head) are legitimate
    return True
```

> **Note on attack-time eligibility**: The T08/T09 attack engine uses a simpler check `p.numel() % 4 != 0` (total element count) rather than the stricter K-dimension check used during sparsification. This works in practice because all eligible architectures satisfy both conditions, but the two checks are not theoretically equivalent.

---

## 4. Sparsification Pipeline (T09)

### 4.1 Mask Computation

The 2:4 mask is computed by selecting the top-2 values (by absolute magnitude) in each group of 4:

```python
def compute_2_4_mask_conv(weight):
    """Conv2d: NHWC K-dimension grouping.
    Actual implementation prunes bottom-2 (shown here as equivalent top-2 keep).
    """
    w_perm = weight.permute(0, 2, 3, 1).contiguous()
    flat = w_perm.view(-1, 4)
    flat_abs = flat.abs()
    # Actual code: find bottom-2 to prune, then invert to get keep mask
    prune_idx = torch.argsort(flat_abs, dim=1)[:, :2]
    keep_mask = torch.ones_like(flat_abs, dtype=torch.bool)
    keep_mask.scatter_(dim=1, index=prune_idx, value=False)
    mask_flat = keep_mask.to(flat.dtype)
    # Restore to original shape
    mask_perm = mask_flat.view(w_perm.shape)
    return mask_perm.permute(0, 3, 1, 2).contiguous()

def compute_2_4_mask_linear(weight):
    """Linear: K = in_features."""
    flat = weight.view(-1, 4)
    flat_abs = flat.abs()
    prune_idx = torch.argsort(flat_abs, dim=1)[:, :2]
    keep_mask = torch.ones_like(flat_abs, dtype=torch.bool)
    keep_mask.scatter_(dim=1, index=prune_idx, value=False)
    return keep_mask.to(flat.dtype).view(weight.shape)
```

### 4.2 Fine-Tuning with Mask Enforcement

After initial mask computation, the model is fine-tuned to recover accuracy lost from pruning. The mask is enforced after every optimizer step:

```python
# Training loop (simplified)
for epoch in range(num_epochs):
    for images, targets in train_loader:
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()

        # Optional: zero out gradients at masked positions
        apply_mask_to_grads(model, mask_map)

        optimizer.step()

        # Critical: re-apply mask to prevent gradient leakage
        apply_masks(model, mask_map)  # masked weights = 0.0 exactly
```

**Mask enforcement** ensures that:
1. Masked positions remain exactly 0.0 (no floating-point drift)
2. Gradients at masked positions do not affect weight updates
3. The 2:4 sparsity structure is maintained throughout training

---

## 5. Defense Mechanism

### 5.1 Reference-Guided Position Repair

**File**: `Sample_BFA/zhwf_10_nm_defense.py`

The defense assumes access to a trusted clean checkpoint (reference). It compares the position encoding of each group between the attacked model and the reference, then repairs mismatches.

**Two repair modes**:

**Mode A — strict_copy**: Replace the entire group with reference values.
```python
if code_attacked != code_reference:
    group_attacked[:] = group_reference[:]    # full copy
```

**Mode B — remap_to_ref_pos**: Keep the attacked magnitudes but move them to reference positions. This preserves any legitimate magnitude changes while fixing position corruption.
```python
if code_attacked != code_reference:
    v0, v1 = attacked_values_at_active_positions
    ref_pos0, ref_pos1 = decode(code_reference)
    new_group = [0, 0, 0, 0]
    new_group[ref_pos0] = v0
    new_group[ref_pos1] = v1
```

**Trade-off**: Mode A provides stronger repair but discards any legitimate weight adaptation; Mode B preserves magnitudes but is less effective if magnitudes were also corrupted.

> **Known limitation**: The defense uses **in_ch-dimension grouping** (`weight.reshape(out_ch, in_ch, k)`, then stride-4 slices along `in_ch` for each spatial position), which differs from the attack engines' NHWC-permuted K-dimension grouping (`weight.permute(0,2,3,1).view(-1,4)`). These produce identical groups for 1×1 convolutions (where `kH*kW=1`), but for larger kernels (e.g., 3×3), the 4-element groups are different. This means the defense may produce false positives or miss actual attacks on non-1×1 conv layers. Future work should align the defense grouping with the attack semantics.

### 5.2 Zero-Only Attack Constraint

**File**: `Sample_BFA/zhwf_04_nm_dense_zero_only.py`

This variant restricts the attacker to only flip weights that are currently zero (0→non-zero transitions). The key difference from the standard attack:

```python
# Standard: choose the flip with highest delta
global_max = max(global_entries, key=lambda e: e['delta'])

# Zero-only: choose the flip with highest delta AMONG zero-valued weights
sorted_entries = sorted(global_entries, key=lambda e: e['delta'], reverse=True)
chosen = None
for entry in sorted_entries:
    orig_int8 = info[entry['module_name']]['quantized_int8'][entry['multi_idx']]
    if orig_int8 == 0:    # only accept zero-valued positions
        chosen = entry
        break
```

This models a scenario where sparse zeros provide a natural defense: as zeros are flipped to non-zero, the layer sparsity decreases, and the attack eventually exhausts available zero positions.

---

## 6. Experimental Matrix

### Models and Datasets

| Model | Parameters | Dataset | Sparse Layers | Eligible Groups |
|---|---|---|---|---|
| ResNet-20 | 0.27M | CIFAR-10 | 18 conv | ~700 |
| ResNet-18 | 11.7M | Imagenette / CIFAR-100 | 20 | ~36,000 |
| MobileNetV2 | 3.4M | Imagenette / CIFAR-100 | 35 | ~28,000 |
| DeiT-Tiny | 5.7M | Imagenette / CIFAR-100 | 49 | ~115,000 |

### Attack Results Summary

**T10_Enhanced on ResNet-20 (CIFAR-10, 50 flips)**:
- Initial accuracy: 92.21% → Final: 12.43% (drop: **79.78%**)
- Average time: 4.07s/flip, total: ~3.4 minutes

**T08/T09 Metadata BFA on large models (50 flips)**:

| Model | Dataset | Initial Acc | Final Acc | Drop |
|---|---|---|---|---|
| ResNet-18 | Imagenette | 98.47% | 70.57% | 27.90% |
| MobileNetV2 | Imagenette | 95.57% | 10.42% | **85.15%** |
| DeiT-Tiny | Imagenette | 93.89% | 82.65% | 11.24% |
| MobileNetV2 | CIFAR-100 | 73.91% | 1.04% | **72.87%** |
| ResNet-18 | CIFAR-100 | 78.89% | 63.66% | 15.23% |
| DeiT-Tiny | CIFAR-100 | 82.32% | 79.67% | 2.65% |

**Key findings**:
- **MobileNetV2** is the most vulnerable architecture (85%+ drop), likely due to the narrow pointwise convolution bottleneck structure.
- **DeiT-Tiny** (Vision Transformer) is the most robust (2-11% drop), benefiting from the redundancy in multi-head self-attention.
- **ResNet-18** shows moderate vulnerability (15-28% drop).

### Time Complexity per Iteration (T10_Enhanced)

| Step | Time | Proportion | Operations |
|---|---|---|---|
| Gradient computation | ~0.5s | 12% | 1 forward + 1 backward pass |
| Step 1 (Coarse filter) | ~1.0s | 25% | Sort all groups by gradient magnitude |
| Step 2+3 (Candidates + proxy) | ~0.25s | 6% | ~2700 dot products (O(1) each) |
| Step 4 (Top-K sort) | ~0.001s | <1% | Sort ~2700 entries |
| Step 5 (Exact verify) | ~2.5s | **60%** | 64 forward passes with save/restore |
| **Total** | **~4.07s** | 100% | — |

Step 5 dominates because each of the 64 candidates requires a full save/restore cycle plus a forward pass on the verification batch.
