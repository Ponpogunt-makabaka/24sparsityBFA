# R1 Latest Version Inventory

**Generated**: 2026-02-23
**Purpose**: List all current R1 tasks after version cleanup

---

## Version Cleanup Summary

**Deleted Old Versions**:
- T02 (proxy-only) → Deleted, replaced by T02.1 (exact verification)
- T03 (proxy-only) → Deleted, replaced by T03.1 (exact verification)

**Current Status**:
- No more T02/T02.1 or T03/T03.1 confusion
- All analysis now based on exact verification versions where applicable

---

## Current R1 Tasks (Latest Versions)

| Task ID | Script | Type | Key Features | Baseline Acc | Final Acc | Notes |
|---------|--------|------|--------------|--------------|-----------|-------|
| **T01** | `run_R1_T01_group_metadata_index_anypattern.py` | Metadata | Any pattern position encoding | 92.33% | 16.75% | No .1 version |
| **T02.1** | `run_R1_T02_1_group_metadata_index_1bit_exact.py` | Metadata | 1-bit reachable + exact verification | 92.33% | 11.67% | **NEW BASELINE** |
| **T03.1** | `run_R1_T03_1_group_metadata_bitmask_swap_cost2_exact.py` | Metadata | Cost-2 swap + exact verification | 92.33% | 44.34% | **NEW BASELINE** |
| **T04** | `run_R1_T04_bitmask_swaps50.py` | Metadata | Extended swap (50 swaps) | 92.33% | ~30% | No .1 version |
| **T05** | `run_R1_T05_joint_best_step_attack.py` | Joint | Weight + index + bitmask | 92.33% | 9.96% | No .1 version |
| **T06.1** | `run_R1_T06_1_sparse_gated_dense_view_exact.py` | Weight | Dense-format global + exact verification | 92.40% | 10.20% (global) | **NEW BASELINE** |
| **T06.1** | `run_R1_T06_1_samplebfa_style_dense_bfa.py` | Weight | sampleBFA-style variant | 92.40% | ~10% | Wrapper for above |
| **T07** | `run_R1_T07_samplebfa_style_dense_bfa.py` | Baseline | sampleBFA comparison | 92.40% | ~10% | No .1 version |

---

## Key Method Upgrades (T02.1, T03.1, T06.1)

### Top-K Exact Verification
All `.1` versions use two-stage verification:
1. **Stage 1 (Proxy Screening)**: First-order Taylor expansion: `score = w_fp * (g_new - g_old)`
2. **Stage 2 (Exact Verification)**: Real forward pass on calib subset (256 samples)
3. **Revert**: Model state restored after each verification

**Impact**:
- **T02.1**: 0 loss decreases (vs 7 in old T02)
- **T03.1**: Smooth loss progression
- **T06.1**: 0 loss decreases across all modes

---

## Mode-Level Results (T06.1)

| Mode | Baseline Acc | Final Acc | Acc Drop | Loss Increase | Notes |
|------|--------------|-----------|----------|---------------|-------|
| **global** | 92.40% | 10.20% | 82.20% | 17.31 | All weights, sparse-gated forward |
| **zero_only** | 92.40% | 89.90% | 2.50% | 0.09 | Attacks sparse_mask==0 positions |
| **nonzero_only** | 92.40% | 10.15% | 82.25% | 14.51 | Attacks sparse_mask==1 positions |

---

## Attack Type Classification

### Metadata Attacks (T01-T04)
- **Target**: Sparse structure encoding (index/bitmask)
- **Forward**: Sparse-gated semantics (sparse_mask preserved)
- **Effectiveness**: T02.1 (80.66% drop) > T01 (75.58% drop) > T03.1 (48% drop)

### Joint Attack (T05)
- **Target**: Both metadata (index/bitmask) AND weight bits
- **Selection**: Joint top-K across all action types
- **Result**: 100% weight_bit selection (0 index, 0 bitmask)

### Weight Attacks (T06.1, T07)
- **Target**: INT8 weight bits directly
- **Forward**: Sparse-gated semantics (sparse_mask preserved in T06.1)
- **Effectiveness**: Global (82.20%) ≈ nonzero_only (82.25%) >> zero_only (2.50%)

---

## File Locations

### Scripts
```
/home/lab-2010/24sparsityBFA/run_R1_T01_group_metadata_index_anypattern.py
/home/lab-2010/24sparsityBFA/run_R1_T02_1_group_metadata_index_1bit_exact.py
/home/lab-2010/24sparsityBFA/run_R1_T03_1_group_metadata_bitmask_swap_cost2_exact.py
/home/lab-2010/24sparsityBFA/run_R1_T04_bitmask_swaps50.py
/home/lab-2010/24sparsityBFA/run_R1_T05_joint_best_step_attack.py
/home/lab-2010/24sparsityBFA/run_R1_T06_1_sparse_gated_dense_view_exact.py
/home/lab-2010/24sparsityBFA/run_R1_T06_1_samplebfa_style_dense_bfa.py
/home/lab-2010/24sparsityBFA/run_R1_T07_samplebfa_style_dense_bfa.py
```

### Result Directories
```
/home/lab-2010/24sparsityBFA/results/R1/R1_T01_*
/home/lab-2010/24sparsityBFA/results/R1/R1_T02_1_*
/home/lab-2010/24sparsityBFA/results/R1/R1_T03_1_*
/home/lab-2010/24sparsityBFA/results/R1/R1_T04_*
/home/lab-2010/24sparsityBFA/results/R1/R1_T05_*
/home/lab-2010/24sparsityBFA/results/R1/R1_T06_1_*
/home/lab-2010/24sparsityBFA/results/R1/R1_T07_*
```

---

## Critical Fixes Applied

### quantized Bug Fix
All R1 scripts correctly call `calibrate_all_layers()` after checkpoint load to set `quantized=True`. This ensures sparse_mask is used during forward pass.

### Exact Verification Implementation
All `.1` versions implement proper state revert after exact verification:
```python
def restore_model_state(...):
    module.sparse_mask.copy_(m_new.clone())
    module.int8_weights.copy_(w_new.clone())
```

---

## Notes for Analysis

1. **T02.1 is now the reference** for 1-bit reachable metadata attack
2. **T03.1 is now the reference** for cost-2 swap metadata attack
3. **T06.1 is now the reference** for dense-format weight-bit attack
4. **Old T02/T03 results should NOT be used** for any new analysis
5. **T05 remains unchanged** (no .1 version exists)
6. **T01, T04, T07 remain unchanged** (no .1 version exists)
