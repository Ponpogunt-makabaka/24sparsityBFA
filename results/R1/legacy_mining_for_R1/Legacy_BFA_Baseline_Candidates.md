# Legacy BFA Baseline Candidates for R1

**Generated**: 2026-02-23
**Purpose**: Identify legacy artifacts suitable as R1_T07 and R1_T06 comparison baselines

---

## R1_T07 Baseline: Traditional BFA Reference

**Purpose**: R1_T07 (sampleBFA-style) needs traditional BFA baseline for comparison

### Recommended Primary Baseline

**Legacy Task 1: Dense INT8 BFA**

| Attribute | Value | Notes |
|-----------|-------|-------|
| **File** | `results/legacy_L0/by_task/task1_dense_int8_log.txt` | Complete attack trace |
| **Model** | Dense INT8 ResNet20 | No sparse mask |
| **Checkpoint** | Legacy dense checkpoint | Different from R1 |
| **Result** | 92.46% → 10.00% (50 flips) | Strong baseline |
| **Attack pattern** | Global bit selection | MSB (bit7) dominated |
| **Bit format** | `layer:idx:bit` | Matches R1_T05 |

**Why Recommended**:
1. Most traditional BFA formulation (global bit flip)
2. Complete 50-step trace for curve comparison
3. Dense model provides "upper bound" on attack effectiveness

**Caveats**:
1. Uses different checkpoint (not Task28)
2. Dense semantics differ from sparse-gated R1
3. MSB dominance may differ from R1_T07 sign-style flip

---

### Recommended Secondary Baseline

**Legacy Task 3: Non-Zero-Only Sparse Attack**

| Attribute | Value | Notes |
|-----------|-------|-------|
| **File** | `results/legacy_L0/by_task/task3_sparse_dense_nonzero_log.txt` | Sparse model attack |
| **Model** | Sparse INT8 ResNet20 | With sparse_mask |
| **Result** | 92.10% → 10.00% (50 flips) | Matches Task 1 |
| **Attack mode** | Non-zero-only (sparse_mask==1) | Closest to R1_T07 |

**Why Recommended**:
1. Uses sparse model (like R1)
2. Non-zero-only ≈ R1_T07 nonzero_only mode
3. Achieves same effectiveness as dense (validates sparse vulnerability)

**Caveats**:
1. Legacy semantics may differ (possible densification)
2. Different checkpoint (92.10% vs R1 92.40%)
3. Not exact same as sampleBFA approach

---

## R1_T06 Baseline: Dense-Format Global Reference

**Purpose**: R1_T06 (dense-view search) needs dense-format legacy comparison

### Recommended Primary Baseline

**Legacy Task 1: Dense INT8 BFA**

**Same as above for R1_T07**

**Why Recommended for R1_T06**:
1. Dense-format global search (same concept)
2. Bit-level granularity matches R1_T06 weight-bit flip
3. Provides "oracle" comparison (no sparse constraints)

**Caveats**:
1. R1_T06 uses sparse-gated forward, Task1 uses dense forward
2. This tests: how much does sparse-gating protect against BFA?

---

### Recommended Mode-Specific Baselines

**For R1_T06 global mode**:
- **Baseline**: Legacy Task 1
- **Comparison**: Dense (Task1) vs Sparse-gated (R1_T06)
- **Expected**: R1_T06 less effective due to sparse constraint

**For R1_T06 zero_only mode**:
- **Baseline**: Legacy Task 2 (zero-only)
- **Legacy result**: 92.10% → 12.43% (79.67% drop)
- **R1 result**: 92.40% → 89.90% (2.50% drop)
- **Discrepancy**: ⚠️ MAJOR - investigate semantics!

**For R1_T06 nonzero_only mode**:
- **Baseline**: Legacy Task 3 (non-zero-only)
- **Legacy result**: 92.10% → 10.00% (82.10% drop)
- **R1 result**: 92.40% → 10.15% (82.25% drop)
- **Agreement**: ✅ EXCELLENT match

---

## Metadata Attack Baselines (R1_T01-T05)

### R1_T01: Any Pattern Attack

**Recommended Baseline**: Legacy Task 5 (NCSA)

| Attribute | Value |
|-----------|-------|
| **File** | `results/legacy_L0/by_task/task5_csr_non_collision_log.txt` |
| **Result** | 92.10% → 11.10% (50 flips) |
| **Semantics** | Non-collision move space |

**Comparison**:
- Task5: CSR encoding, NCSA
- R1_T01: Position encoding, any pattern
- Both achieve strong effectiveness with collision avoidance

---

### R1_T02: 1-Bit Reachable

**Recommended Baseline**: Legacy Task 5 (NCSA) + Task 18 (validity)

| Attribute | Value |
|-----------|-------|
| **File** | `results/legacy_L0/by_task/task18_bitmask_validity_log.txt` |
| **Result** | 0% valid single-bit flips (popcount constraint) |

**Key Insight**:
- Task18 proves single-bit flips break 4-bit bitmask validity
- This justifies R1_T02's 1-bit reachable constraint on position encoding (not bitmask)

---

### R1_T03: Cost-2 Swap

**Recommended Baseline**: Legacy Task 20 (swap)

| Attribute | Value |
|-----------|-------|
| **File** | `results/legacy_L0/by_task/task20_bitmask_swap_log.txt` |
| **Result** | 86.57% → 35.50% (25 swaps = 50 flips) |

**Comparison**:
- Task20: Cost-2 swap on bitmask
- R1_T03: Cost-2 swap on bitmask
- Direct conceptual predecessor

**Note**: Different baseline accuracy (86.57% vs R1 92.40%)

---

### R1_T04: Extended Swap

**Recommended Baseline**: Legacy Task 20 (reference)

**Same as R1_T03** - R1_T04 extends to 50 swaps (100 flips)

---

### R1_T05: Joint Attack

**Recommended Baseline**: Legacy Task 21 (compare)

| Attribute | Value |
|-----------|-------|
| **File** | `results/legacy_L0/by_task/task21_position_compare_log.txt` |
| **Results** | metadata_ncsa: 86.57% → 9.67%, weight_msb: 86.57% → 9.91% |

**Key Finding**: Metadata and weight attacks achieve similar effectiveness
**R1_T05 finding**: Joint attack chose weight_bit 100% of time

---

## Baseline Suitability Summary

| R1 Task | Legacy Baseline | Suitability | Match Quality |
|---------|----------------|-------------|---------------|
| R1_T07 | Task 1 (dense) | High | Traditional BFA reference |
| R1_T07 | Task 3 (nonzero) | High | Sparse model comparison |
| R1_T06 global | Task 1 (dense) | High | Dense vs sparse-gated |
| R1_T06 zero_only | Task 2 (zero) | ⚠️ Low | Major discrepancy |
| R1_T06 nonzero | Task 3 (nonzero) | ✅ High | Excellent match |
| R1_T01 | Task 5 (NCSA) | Medium | Different encoding |
| R1_T02 | Task 18 (validity) | High | Justifies constraint |
| R1_T03 | Task 20 (swap) | High | Direct predecessor |
| R1_T04 | Task 20 (swap) | High | Extended version |
| R1_T05 | Task 21 (compare) | Medium | Shows metadata ~ weight |

---

## Critical Discrepancy: Zero-Only Mode

**Issue**: Legacy Task 2 vs R1_T06 zero_only results differ dramatically

| Metric | Legacy Task 2 | R1_T06 zero_only |
|--------|---------------|------------------|
| Baseline | 92.10% | 92.40% |
| Final | 12.43% | 89.90% |
| Drop | 79.67% | 2.50% |

**Possible Explanations**:
1. **Semantic difference**: Legacy "dense-zero" vs R1 "sparse-gated zero"
2. **Candidate generation**: Legacy attacks masked positions differently
3. **Forward path**: Legacy may densify before attack
4. **Checkpoint**: Different model states

**Investigation Needed**:
```python
# Hypothesis 1: Legacy densifies before zero-only attack
# Test: Check if Task2 preserves sparse_mask during forward

# Hypothesis 2: Different candidate definition
# Test: Compare which positions are attacked in each
```

**Impact**: Cannot claim "zero-only ineffective" without resolving this

---

## Reproduction Commands

### Legacy Task 1 (Dense BFA)
```bash
# Note: Original script may not be available
# Use results file for reference
cat results/legacy_L0/by_task/task1_dense_int8_log.txt
```

### Legacy Task 3 (Non-Zero-Only)
```bash
cat results/legacy_L0/by_task/task3_sparse_dense_nonzero_log.txt
```

### Legacy Task 20 (Cost-2 Swap)
```bash
cat results/legacy_L0/by_task/task20_bitmask_swap_log.txt
```

### Legacy Task 28 (Checkpoint Source)
```bash
# Verify checkpoint structure
python -c "
import torch
ckpt = torch.load('results/legacy_L0/by_task/task28_sparse_mask_fixed_finetune_int8_ckpt.pth', map_location='cpu')
print('Keys:', list(ckpt.keys()) if isinstance(ckpt, dict) else 'direct state_dict')
"
```
