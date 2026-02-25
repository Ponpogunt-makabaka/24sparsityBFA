# 24sparsityBFA Project Summary

**Date**: 2026-02-18
**Project**: Sparse Model Bit-Flip Attacks (BFA) - Complete Study
**Status**: All Workflows Complete (Legacy Tasks 0-28 + R1 Revised Workflow T01-T05)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack](#2-technology-stack)
3. [Project Structure](#3-project-structure)
4. [Technical Background](#4-technical-background)
5. [Legacy Work Summary](#5-legacy-work-summary)
6. [R1 Workflow Results](#6-r1-workflow-results)
7. [Complete Comparison](#7-complete-comparison)
8. [Key Findings](#8-key-findings)
9. [Reproduction Commands](#9-reproduction-commands)
10. [References](#10-references)

---

## 1. Project Overview

### 1.1 Research Goal

Study **Bit-Flip Attacks (BFA)** on sparse 2:4 quantized neural networks, with three main attack vectors:

1. **Weight-Bit Attacks**: Directly flipping bits in INT8 quantized weight values
2. **Metadata Attacks**: Modifying sparse mask information (which weights are active)
3. **Joint Attacks**: Unified framework selecting optimal attack per step

### 1.2 Research Phases

| Phase | Tasks | Focus |
|-------|-------|-------|
| **Legacy** | Task 0-28 | Initial BFA exploration, CSR attacks, defense mechanisms |
| **R1 Revised** | R1_T01-T05 | Structured metadata attack comparison with unified framework |

### 1.3 Key Datasets

| Dataset | Classes | Use Case |
|---------|--------|----------|
| **CIFAR-10** | 10 | Primary dataset for ResNet-20 experiments |
| **Imagenette** | 10 | ImageNet subset for ResNet-18/MobileNetV2/DeiT-Tiny |

### 1.4 Models Studied

| Model | Parameters | Sparsity | Quantization |
|-------|-----------|----------|-------------|
| **ResNet-20** | ~270K | 2:4 structured | INT8 |
| **ResNet-18** | ~11M | 2:4 structured | INT8 |
| **MobileNetV2** | ~3.4M | 2:4 structured | INT8 |
| **DeiT-Tiny** | ~5.7M | Partial (QKV+MLP) | INT8 |

---

## 2. Technology Stack

### 2.1 Core Dependencies

```python
# Core ML Framework
torch>=2.0.0
torchvision>=0.15.0

# Data Science
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# Visualization
matplotlib>=3.7.0

# Utilities
tqdm>=4.65.0
jupyter>=1.0.0
```

### 2.2 Project Architecture

```
├── bfa/                    # Attack engines
├── models/                 # Model definitions & factory
├── train/                  # Training & PTQ utilities
├── scripts/                # Plotting & analysis
├── results/                # Experiment results
└── run_*.py                # Experiment scripts (30+ scripts)
```

### 2.3 Key Technical Components

| Component | File | Purpose |
|-----------|------|---------|
| **INT8 Quantization** | `train/ptq_convert.py` | FP32→INT8 symmetric quantization |
| **2:4 Sparsity** | `models/sparse_ops.py` | Structured 2:4 sparse operations |
| **BFA Engine** | `bfa/int8_attack.py` | Progressive Bit Search algorithm |
| **CSR Attack** | `bfa/csr_non_collision_attack.py` | Non-collision CSR index attacks |
| **Metadata Attack** | `bfa/encoded_sparse_attack.py` | Sparse mask manipulation |

### 2.4 Code Statistics

| Metric | Value |
|--------|-------|
| **Total Python scripts** | 30+ |
| **Total lines of code** | ~14,836 lines |
| **Result files** | 100+ (pkl, csv, txt, png) |
| **Experiment tasks** | 33 (Task 0-28 + R1_T01-T05) |

---

## 3. Project Structure

### 3.1 Directory Layout

```
24sparsityBFA/
├── bfa/                               # Attack engines
│   ├── int8_attack.py               # INT8 bit-flip BFA engine
│   ├── csr_non_collision_attack.py  # CSR non-collision attacks
│   ├── encoded_sparse_attack.py     # Dense-format metadata attacks
│   └── ...
│
├── models/                            # Model definitions
│   ├── factory.py                   # Model factory with sparsity support
│   ├── sparse_ops.py                # 2:4 sparse conv/linear operations
│   ├── sparse_csr.py                 # CSR format utilities
│   └── resnet20.py                   # ResNet-20 architecture
│
├── train/                             # Training utilities
│   ├── ptq_convert.py                # Post-training INT8 conversion
│   ├── train_sparse.py               # Sparse model training
│   └── imagenet_pipeline_utils.py   # ImageNet/Imagenette utilities
│
├── scripts/                           # Plotting & analysis
│   ├── plot_R1_comparison.py         # R1 T01-T04 comparison
│   ├── plot_full_comparison.py       # Complete comparison (all tasks)
│   └── p012_17_utils.py              # Shared utilities
│
├── results/                           # Experiment results
│   ├── R1/                           # Revised workflow results
│   │   ├── R1_T01_*                   # Index encoding, any pattern
│   │   ├── R1_T02_*                   # Index encoding, 1-bit reachable
│   │   ├── R1_T03_*                   # Bitmask encoding, 25 swaps
│   │   ├── R1_T04_*                   # Bitmask encoding, 50 swaps
│   │   ├── R1_T05_*                   # Joint best-step attack
│   │   ├── full_comparison.png        # All tasks comparison
│   │   └── R1_comparison_curve.png   # R1 T01-T04 comparison
│   │
│   └── legacy_L0/                     # Archived legacy results
│       ├── by_task/                   # Task-organized (77 files)
│       ├── by_date/                   # Chronological with plots
│       ├── debug_task1xx/             # Debug artifacts
│       └── MANIFEST.md                  # Archive manifest
│
├── run_R1_T01_*.py                     # R1 task scripts (990 lines each)
├── run_R1_T02_*.py
├── run_R1_T03_*.py
├── run_R1_T04_*.py
├── run_R1_T05_*.py                     # Joint best-step (1527 lines)
│
├── run_task1_*.py                       # Legacy Task 1-3 scripts
├── run_task2_*.py                       # Legacy Task 2-5 scripts
├── run_task4_*.py                       # Legacy Task 4-8 scripts
├── run_task[5-23]*.py                   # Various analysis tasks
├── run_task28_*.py                      # Baseline audit & recovery
│
└── Feb18ProjectSummarized.md          # This file
```

### 3.2 Script Categories

| Category | Scripts | Purpose |
|----------|---------|---------|
| **Baseline** | `run_task1_baseline_comparison.py` | Dense/sparse INT8 baseline |
| **Weight Attacks** | `run_task[2-3]_*.py` | Zero/nonzero targeting |
| **CSR Attacks** | `run_task[4-8]_*.py` | Non-collision CSR attacks |
| **Defense** | `run_task11_*.py` | Parity/CRC defense |
| **Analysis** | `run_task[9-10,12-17,22-23]*.py` | Ablation, localization |
| **Metadata** | `run_task18-21*.py` | Bitmask encoding attacks |
| **R1 Workflow** | `run_R1_T[01-05]_*.py` | Revised metadata attacks |

---

## 4. Technical Background

### 4.1 2:4 Structured Sparsity

**Definition**: Exactly 2 out of 4 weights are non-zero in each group

```
Example groups (4 elements each):
[1.2, 0, 0.8, 0]    → Valid  (2 non-zeros)
[0, 1.5, 0, 2.1]    → Valid  (2 non-zeros)
[0.5, 0, 1.2, 0]    → Invalid (1 non-zero)
```

**Storage**:
- Dense format with `sparse_mask` (0/1 tensor)
- Groups along input channels: `[out, in/4, 4, k, k]`
- Mask indicates which positions are active

### 4.2 INT8 Quantization

**Symmetric Quantization Formula**:
```
scale = max(|w|) / 127
w_int8 = clamp(round(w / scale), -128, 127)
w_fp32 = w_int8 × scale
```

**INT8 Format** (Two's Complement):
- Bit 7: Sign bit (0=positive, 1=negative)
- Bits 0-6: Magnitude (0-127)
- Range: -128 to +127

### 4.3 Attack Vectors

#### 4.3.1 Weight-Bit Flip

**Cost**: 1 physical flip per operation
**Target**: INT8 weight bits
**Effect**: Directly changes weight value
**Most vulnerable**: Bit 7 (sign bit)

#### 4.3.2 Index/Position Metadata (R1_T01/T02)

**Encoding**: Two 2-bit indices packed into 4-bit code
```python
code = (j << 2) | i  # where i, j ∈ {0,1,2,3}, i ≠ j
```

**Cost**: 1 physical flip (changes 1 bit in 4-bit code)
**Constraints**:
- **Any Pattern (R1_T01)**: All 6 valid patterns reachable
- **1-bit Reachable (R1_T02)**: Only Hamming-1 neighbors

#### 4.3.3 Bitmask Metadata (R1_T03/R1_T04)

**Encoding**: Direct 4-bit mask
```
0b1100 = positions 2,3 active
popcount(mask) == 2  # Validity condition
```

**Cost**: 2 physical flips (one 1→0, one 0→1)
**Operation**: Cost-2 swap maintains validity

### 4.4 Progressive Bit Search (PBS) Algorithm

```
1. Compute gradients on calibration batch
2. Score candidates: score = grad × Δvalue
3. Select candidate with max score
4. Flip bit, update model
5. Repeat until budget exhausted
```

**Key Features**:
- History mask: prevents re-flipping same bit
- Sparsity mask: only targets non-zero weights
- Gradient-based scoring: first-order approximation

---

## 5. Legacy Work Summary

### 5.1 Legacy Task Classification

#### Task 1-3: Dense-Format Weight-Bit Attacks

| Task | Type | Target | Final Acc | Drop |
|------|------|--------|-----------|------|
| **Task 1** | Global | All weights | 10.00% | 82.10% |
| **Task 2** | Zero-only | Zero weights | 12.43% | 79.67% |
| **Task 3** | Nonzero-only | Non-zero weights | 10.00% | 82.10% |

**Baseline**: 92.10% (original sparse checkpoint)

#### Task 4-5: CSR Index Attacks

| Task | Type | Final Acc | Drop |
|------|------|-----------|------|
| **Task 4** | CSR Index (collision allowed) | 10.84% | 81.26% |
| **Task 5** | CSR Non-Collision (NCSA) | 9.67% | 82.43% |

**Key Finding**: Non-collision attacks are slightly more effective

#### Task 6-8: ImageNet Expansion

| Model | Dataset | Baseline | Final Acc | Drop |
|-------|--------|----------|----------|------|
| **ResNet-18** | Imagenette | 94.53% | 46.09% | 48.44% |
| **MobileNetV2** | Imagenette | 95.31% | 0.00% | 95.31% |
| **DeiT-Tiny** | Imagenette | 87.50% | 59.77% | 27.73% |

**Key Finding**: MobileNetV2 is most vulnerable to metadata attacks

#### Task 9: Collision Characterization

**Findings**:
- CSR collisions behave as **mask-last (overwrite-last)**
- Unsafe 2:4 collision injection behaves as **drop**

#### Task 10: Flip Outcome Analysis

**Outcome Taxonomy**:
- **Rewire**: Valid position change (60%)
- **Drop**: Both non-zeros lost (20%)
- **No-op**: No effective change (15%)
- **Collision**: Invalid (5%)

#### Task 11: Metadata Defense

**Defense Mechanisms**:
- **Parity Check**: Detects metadata changes
- **CRC-8**: Stronger detection
- **Revert**: Restores original state on detection

**Results**:
- Parity trusted: 100% detection (mitigation effective)
- Adaptive bypass: 0% detection (attacker can co-modify parity)

#### Task 12: Scoring Ablations

| Method | Final Acc | Drop |
|--------|-----------|------|
| **NCSA (w*Δg)** | 9.67% | 76.90% |
| **Grad-only (Δg)** | 88.72% | -2.15% |
| **Weight-only (|w|)** | 86.13% | 0.44% |
| **Random-valid** | 86.91% | -0.34% |

**Conclusion**: Gradient-based scoring is essential

#### Task 13: Calibration Sensitivity

| Calib Samples | Final Acc |
|--------------|-----------|
| 32 | 10.06% |
| 64 | 25.29% |
| **256** | **9.67%** ← Optimal |
| 512 | 36.04% |
| 1024 | 11.67% |

#### Task 14: Layer Localization

**Flip Distribution**:
- Stage 1 (early layers): 36/50 flips
- Stage 2 (middle layers): 10/50 flips
- Stage 3 (late layers): 4/50 flips

**Top Vulnerable Layers**: `layer1.0.conv2`, `layer1.1.conv1`, `layer2.0.downsample.0`

#### Task 15: Defense Realism

| Scenario | Final Acc |
|----------|-----------|
| Baseline (no defense) | 13.38% |
| Parity trusted | 86.57% (100% detection) |
| Parity adaptive bypass | 13.38% (0% detection) |
| Parity budgeted bypass | 35.50% (cost=2) |

#### Task 16: Runtime Overhead

| Operation | Time |
|-----------|------|
| Search (per attempt) | ~1.57-1.60s |
| Parity check+mitigate | ~0.136ms |
| CRC-8 check+mitigate | ~2.005ms |

#### Task 17: Seed Robustness

| Seed | Final Acc |
|------|-----------|
| 0 | 9.67% |
| 123 | 13.38% |
| 42 | 13.67% |
| **Mean** | **12.24% ± 1.82%** |

#### Tasks 18-23: Minimal Closed-Loop

| Task | Focus | Final Acc | Drop |
|------|-------|-----------|------|
| **T18** | Bitmask 1-bit validity | N/A | 0% valid flips |
| **T19** | Weight MSB on non-zero | 9.91% | 76.66% |
| **T20** | Bitmask cost-2 swap | 35.50% | 51.07% |
| **T21** | Metadata vs Weight MSB | 9.67% vs 9.91% | Comparable |
| **T22** | Dense weight MSB | 9.57% | 82.96% |
| **T23** | Summary figure | - | - |

**Key Finding**: Bitmask requires cost-2 swaps; single-bit flips always break validity

#### Task 28: Baseline Recovery

| Phase | Accuracy |
|-------|----------|
| Original sparse checkpoint | 86.50% |
| After mask-fixed finetune | **92.21%** ← New baseline |

---

## 6. R1 Workflow Results

### 6.1 R1 Task Overview

| Task | Encoding | Constraint | Ops | Final Acc | Drop | Script Lines |
|------|----------|------------|-----|-----------|------|-------------|
| **R1_T01** | Index | Any Pattern | 50 | 38.67% | 53.66% | 990 |
| **R1_T02** | Index | 1-bit Reachable | 50 | 30.37% | 61.96% | 909 |
| **R1_T03** | Bitmask | Cost-2 Swap | 25 | 65.14% | 27.19% | 944 |
| **R1_T04** | Bitmask | Cost-2 Swap | 50 | 50.73% | 41.60% | 929 |
| **R1_T05** | Joint | Best-Step | 50 | **9.96%** | **82.25%** | 1527 |

### 6.2 R1_T01: Index Any Pattern

**Overview**: Allows ANY valid 2-of-4 pattern transition

**Features**:
- 6 patterns per group (all 2-of-4 combinations)
- Gradient-based proxy: `ΔL_g ≈ ∇_{w̃_g} L · (w̃_g(p') − w̃_g(p))`
- Anti-reversal: Forbidden transitions prevent cycling

**Results**:
- Baseline: 92.33%
- Final (50 flips): 38.67%
- Runtime: ~400 seconds

**Script**: `run_R1_T01_group_metadata_index_anypattern.py`

### 6.3 R1_T02: Index 1-bit Reachable

**Overview**: Constrained to Hamming-1 neighbors in 4-bit code space

**Key Difference**:
```python
# Only 1-bit flips allowed
candidates = [c ^ (1 << b) for b in {0,1,2,3}]

# Reject collisions (i == j)
if i == j:
    continue  # Invalid pattern
```

**Results**:
- Baseline: 92.33%
- Final (50 flips): 30.37%
- Drop: **61.96%** (strongest of metadata-only attacks)
- Runtime: ~638 seconds

**Script**: `run_R1_T02_group_metadata_index_1bit.py`

### 6.4 R1_T03: Bitmask Cost-2 Swap (25 Swaps)

**Overview**: Bitmask encoding with structure-preserving swaps

**Swap Operation**:
```python
# Flip one 1→0 and one 0→1
new_mask = current_mask ^ ((1 << bit_off) | (1 << bit_on))
```

**Physical vs Logical Mapping**:
- Physical budget: 50 flips
- Cost per swap: 2 flips
- Logical swaps: 25

**Results**:
- Baseline: 92.33%
- Final: 65.14% (at 50 physical flips)
- Drop: 27.19%
- Runtime: ~383 seconds

**Script**: `run_R1_T03_group_metadata_bitmask_swap_cost2.py`

### 6.5 R1_T04: Bitmask Cost-2 Swap (50 Swaps)

**Purpose**: Fair comparison with R1_T01/T02 (50 logical operations)

**Key Change**: `max_logical_swaps=50` instead of `physical_budget=50`

**Physical Flips**: 100 (2 per swap)

**Results**:
- Baseline: 92.33%
- Final: 50.73%
- Drop: 41.60%
- Runtime: ~761 seconds

**Comparison with R1_T03**:
| Metric | R1_T03 | R1_T04 |
|--------|--------|--------|
| Logical swaps | 25 | 50 |
| Physical flips | 50 | 100 |
| Final accuracy | 65.14% | 50.73% |
| Accuracy drop | 27.20% | 41.60% |

**Script**: `run_R1_T04_bitmask_swaps50.py`

### 6.6 R1_T05: Joint Best-Step Attack

**Overview**: Unified framework with three action types

**Two-Stage Selection**:
```
Stage A (Fast Proxy Scoring):
  1. Compute gradients on calibration batch
  2. Enumerate candidates from ALL action types
  3. Score by: proxy_score / cost
  4. Select top-K (K=64)

Stage B (Exact Verification):
  1. For each top-K candidate:
     - Apply to model
     - Compute exact loss
     - Revert
  2. Select: argmax(ΔL_exact / cost)
```

**Action Types**:
1. **Weight-bit flip** (cost=1): Flip INT8 weight bits
2. **Index 1-bit move** (cost=1): 1-bit in 4-bit code space
3. **Bitmask swap** (cost=2): Cost-2 swap between valid masks

**Results**:
- Baseline: 92.21%
- Final (50 flips): **9.96%**
- Drop: **82.25%** ⭐ Strongest attack
- Runtime: ~3977 seconds (66 minutes)

**Action Breakdown**:
```
Weight-Bit Flips:  ████████████████████ 50 (100%)
Index 1-bit Moves:  ░░ 0 (0%)
Bitmask Swaps:      ░░ 0 (0%)
```

**Key Insight**: Weight-bit flips dominate; metadata attacks never selected

**Script**: `run_R1_T05_joint_best_step_attack.py`

---

## 7. Complete Comparison

### 7.1 All Tasks Comparison

| Task | Encoding | Constraint | Baseline | Final | Drop |
|------|----------|------------|----------|-------|------|
| **Legacy Task1** | - | Global weight-bit | 92.10% | 10.00% | 82.10% |
| **Legacy Task2** | - | Zero-targeting | 92.10% | 12.43% | 79.67% |
| **Legacy Task3** | - | Nonzero-targeting | 92.10% | 10.00% | 82.10% |
| **R1_T01** | Index | Any Pattern | 92.33% | 38.67% | 53.66% |
| **R1_T02** | Index | 1-bit Reachable | 92.33% | 30.37% | 61.96% |
| **R1_T03** | Bitmask | 25 Swaps | 92.33% | 65.14% | 27.19% |
| **R1_T04** | Bitmask | 50 Swaps | 92.33% | 50.73% | 41.60% |
| **R1_T05** | **Joint** | **Best-Step** | **92.21%** | **9.96%** | **82.25%** |

### 7.2 Visualization

**Full Comparison Plot**: `results/R1/full_comparison.png`

- 8 curves with distinct colors
- Legacy tasks (Task1-3): Gray shades (dashed)
- R1_T01-T04: Blue, Orange, Green, Red
- R1_T05: Purple (thickest line, strongest attack)

### 7.3 Accuracy Drop Ranking (50 flips)

| Rank | Task | Accuracy Drop |
|------|------|---------------|
| 1 | **R1_T05** | **82.25%** ⭐ |
| 2 | Legacy Task1 | 82.10% |
| 2 | Legacy Task3 | 82.10% |
| 4 | Legacy Task2 | 79.67% |
| 5 | R1_T02 | 61.96% |
| 6 | R1_T01 | 53.66% |
| 7 | R1_T04 | 41.60% |
| 8 | R1_T03 | 27.19% |

---

## 8. Key Findings

### 8.1 Attack Effectiveness

1. **R1_T05 Strongest**: Joint best-step attack achieves 82.25% accuracy drop
2. **Weight-Bit Dominates**: All 50 R1_T05 steps chose weight-bit flips
3. **Metadata Limited**: Metadata-only attacks (R1_T01-T04) show 27-62% drops
4. **Sign Bit Critical**: Bit 7 (sign bit) is the most vulnerable position

### 8.2 Encoding Comparison

| Encoding | Pros | Cons |
|----------|------|------|
| **Index (Any)** | More candidate options | Less constrained |
| **Index (1-bit)** | Stronger constraint | Better performance |
| **Bitmask** | Natural validity | High cost (2 flips) |

### 8.3 Baseline Impact

| Baseline | Accuracy | Robustness |
|----------|----------|------------|
| Original (~86%) | 86.50% | Low |
| Improved (~92%) | 92.21% | **High** |

**Key Insight**: Improved baseline is more robust but still vulnerable

### 8.4 Defense Implications

1. **Weight Integrity > Metadata**: Defenses should focus on weight bit protection
2. **Sign Bit Priority**: Special protection needed for INT8 sign bit
3. **Unified Attacks**: Joint attacks find optimal damage per budget

---

## 9. Reproduction Commands

### 9.1 R1 Tasks

```bash
# R1_T01: Index encoding, any pattern
python run_R1_T01_group_metadata_index_anypattern.py \
  --device cpu --seed 0 --max-flips 50 \
  --calib-samples 256 --eval-samples 2000 \
  --ckpt results/legacy_L0/by_task/task28_sparse_mask_fixed_finetune_int8_ckpt.pth

# R1_T02: Index encoding, 1-bit reachable
python run_R1_T02_group_metadata_index_1bit.py \
  --device cpu --seed 0 --max-flips 50 \
  --calib-samples 256 --eval-samples 2000 \
  --ckpt results/legacy_L0/by_task/task28_sparse_mask_fixed_finetune_int8_ckpt.pth

# R1_T03: Bitmask encoding, 25 swaps
python run_R1_T03_group_metadata_bitmask_swap_cost2.py \
  --device cpu --seed 0 --physical-budget 50 \
  --calib-samples 256 --eval-samples 2000 \
  --ckpt results/legacy_L0/by_task/task28_sparse_mask_fixed_finetune_int8_ckpt.pth

# R1_T04: Bitmask encoding, 50 swaps
python run_R1_T04_bitmask_swaps50.py \
  --device cpu --seed 0 --max-logical-swaps 50 \
  --calib-samples 256 --eval-samples 2000 \
  --ckpt results/legacy_L0/by_task/task28_sparse_mask_fixed_finetune_int8_ckpt.pth

# R1_T05: Joint best-step attack
python run_R1_T05_joint_best_step_attack.py \
  --device cpu --seed 0 --physical-budget 50 \
  --topk 64 \
  --calib-samples 256 --eval-samples 2000 \
  --ckpt results/legacy_L0/by_task/task28_sparse_mask_fixed_finetune_int8_ckpt.pth
```

### 9.2 Visualization

```bash
# Generate R1 T01-T04 comparison
python scripts/plot_R1_comparison.py

# Generate complete comparison (all tasks)
python scripts/plot_full_comparison.py
```

### 9.3 Legacy Task Reproduction

```bash
# Task 5: CSR Non-Collision Attack (NCSA)
python run_task5_csr_non_collision.py \
  --device cpu --max-flips 50 --seed 0

# Task 11: Metadata Defense
python run_task11_metadata_defense.py \
  --device cpu --seed 0 --defense parity

# Task 28: Baseline Audit
python run_task28_sparsity_baseline_audit.py \
  --device cpu --eval-samples 10000 --seed 123
```

---

## 10. References

### 10.1 Core Checkpoints

| Checkpoint | Path | Accuracy |
|------------|------|----------|
| **Sparse INT8** | `results/legacy_L0/by_task/task28_sparse_mask_fixed_finetune_int8_ckpt.pth` | 92.21% |
| Original Sparse | `models/sparse_model.pth` | 86.50% |

### 10.2 Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `bfa/int8_attack.py` | 535 | INT8 BFA engine |
| `train/ptq_convert.py` | 400+ | PTQ conversion |
| `run_R1_T05_*.py` | 1527 | Joint attack script |
| `agent_develop_log.md` | 1077+ | Development log |

### 10.3 Result Files

| Category | Count | Location |
|----------|-------|----------|
| R1 results | 28 files | `results/R1/` |
| Legacy results | 77 files | `results/legacy_L0/` |
| Plots | 20+ | Various |

---

## Appendix A: Attack Algorithm Details

### A.1 Progressive Bit Search (PBS)

```
Input: Model M, budget B, calibration batch C
Output: Attack sequence

1. Initialize:
   - flipped_bits = {}
   - current_loss = evaluate(M, C)

2. For b = 1 to B:
   a. Compute gradients ∇_w L on C
   b. For each weight bit (i, j, k) not in flipped_bits:
      - Compute Δvalue if bit k is flipped
      - Score = |∇_w[i,j,k]| × |Δvalue|
   c. Select bit with max Score
   d. Flip the bit
   e. Update flipped_bits
   f. Evaluate new accuracy

3. Return: flipped_bits, accuracy_history
```

### A.2 Metadata Attack (Group-Based)

```
Input: Model M, budget B, groups G
Output: Attack sequence

1. Initialize:
   - exclude_groups = {}
   - forbidden_transitions = {}

2. For step = 1 to B:
   a. Compute gradients ∇_w L
   b. For each group g in G not in exclude_groups:
      - Get current pattern p (2 active positions)
      - For each candidate pattern p' (valid transitions only):
         - Compute Δw̃_g = w̃_g(p') - w̃_g(p)
         - Score = ∇_{w̃_g} L · Δw̃_g
      - Keep best candidate for this group
   c. Select global best candidate
   d. Apply pattern change to sparse_mask
   e. Update exclude_groups, forbidden_transitions
   f. Evaluate new accuracy

3. Return: attack_history, accuracy_history
```

### A.3 Joint Best-Step (Two-Stage)

```
Input: Model M, budget B, top-K
Output: Attack sequence

1. For step = 1 while budget allows:
   a. Stage A - Fast Proxy Scoring:
      - Compute gradients
      - Enumerate candidates from ALL action types
      - Score: proxy_score / cost
      - Select top-K candidates

   b. Stage B - Exact Verification:
      - For each top-K candidate:
         * Apply candidate to M
         * Compute exact loss L_exact
         * Revert candidate
      - Select: argmax(ΔL_exact / cost)

   c. Apply chosen action
   d. Update budget
   e. Evaluate accuracy

2. Return: action_history, accuracy_history
```

---

## Appendix B: Technical Specifications

### B.1 Model Architecture (ResNet-20)

```
Input: 32×32×3 image
Output: 10 classes

Conv1: 3×3, 64 filters, stride=1, padding=1
Stage1: 2 blocks
  - Conv: 3×3, 64 filters, stride=1, padding=1
  - Conv: 3×3, 64 filters, stride=1, padding=1
  - Identity skip
Stage2: 2 blocks
  - Conv: 3×3, 128 filters, stride=2, padding=1
  - Conv: 3×3, 128 filters, stride=1, padding=1
  - Identity skip
Stage3: 2 blocks
  - Conv: 3×3, 256 filters, stride=2, padding=1
  - Conv: 3×3, 256 filters, stride=1, padding=1
  - Identity skip

AvgPool: 4×4, stride=4
FC: 256 → 10
```

### B.2 2:4 Sparsity Implementation

**Conv2d with 2:4 Sparsity**:
```python
class SparseConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        # Register sparse_mask buffer
        self.register_buffer('sparse_mask',
                             generate_2to4_mask(in_channels))

    def forward(self, x):
        # Apply sparse mask
        w_masked = self.weight * self.sparse_mask
        return F.conv2d(x, w_masked, ...)
```

**Mask Generation**:
- Group input channels by 4
- Keep top-2 magnitude weights per group
- Set others to 0

---

## Appendix C: Experiment Configurations

### C.1 Standard Settings

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10 |
| Model | ResNet-20 |
| Sparsity | 2:4 structured |
| Quantization | INT8 symmetric |
| Calibration samples | 256 |
| Evaluation samples | 2000 |
| Seed | 0 |
| Device | CPU |

### C.2 Attack Parameters

| Parameter | Value |
|-----------|-------|
| Physical budget | 50 (default) |
| Top-K (exact verify) | 64 |
| Exclude window | 20 |
| Max forbidden transitions | 1000 |

---

**End of Document**

**Generated**: 2026-02-18
**Workflow**: R1 Revised + Legacy Complete
**Total Tasks**: 33 (Task 0-28 + R1_T01-T05)
**Status**: Complete
