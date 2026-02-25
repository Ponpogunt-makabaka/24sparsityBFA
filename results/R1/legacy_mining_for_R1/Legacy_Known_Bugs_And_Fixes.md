# Legacy Known Bugs and Fixes

**Generated**: 2026-02-23
**Purpose**: Document legacy bugs that affect R1 semantic correctness
**Impact**: Critical for understanding R1 results and avoiding repeated mistakes

---

## Critical Bugs

### Bug #1: quantized Flag Not Set After Checkpoint Load

**Severity**: P0 - Critical
**Status**: FIXED in legacy, documented for R1

**Phenomenon**:
- Loading INT8 checkpoint does NOT set `module.quantized = True`
- Forward path uses `if self.quantized:` to decide whether to call `get_dequantized_weights()`
- When `quantized=False`, sparse_mask is completely ignored
- Metadata attacks appear to have no effect

**Evidence**:
```
Before fix:
  loss_delta = 0.0
  logits diff = 0.0
  Attack had NO effect despite modifying sparse_mask

After fix:
  loss_delta = 0.0033
  Attack effect confirmed
```
Source: `debug_task1xx/sanity_one_step_log.txt`

**Root Cause**:
```python
# WRONG (legacy bug)
model.load_state_dict(filtered_state_dict, strict=False)
model.eval()  # quantized still False!

# CORRECT (fix)
model.load_state_dict(filtered_state_dict, strict=False)
model.calibrate_all_layers()  # Sets quantized=True
model.eval()
```

**Fix Location**: `run_task1xx_group_metadata_attack.py` model loading logic

**Impact on R1**:
- **Affects**: All R1 metadata attacks (T01-T05)
- **Check**: Verify R1 scripts call `calibrate_all_layers()` after loading
- **Risk**: If missed, metadata attacks will show false-negative results

**Reference**: `debug_task1xx/final_root_cause_report.md`

---

### Bug #2: Sparse Mask Code Path Verification

**Severity**: P1 - High
**Status**: VERIFIED (not a bug, but documented)

**Phenomenon**:
- Initial concern: sparse_mask might not be used in forward path
- Investigation confirmed: sparse_mask IS used when quantized=True

**Evidence**:
```python
# Verified path:
def get_dequantized_weights(self):
    w_dequantized = self.int8_weights.float() * self.scale
    if self.sparse_mask is not None:
        w_dequantized = w_dequantized * self.sparse_mask  # USED HERE
    return w_dequantized
```
Source: `debug_task1xx/code_path_audit.md`

**Impact on R1**:
- Confirms R1 sparse-gated semantics are correctly implemented
- No action needed for R1

---

## Semantic Differences (Legacy vs R1)

### Difference #1: Zero-Only Attack Effectiveness

**Observation**:
- Legacy Task2 zero-only: 92.10% → 12.43% (79.67% drop)
- R1_T06 zero_only: 92.40% → 89.90% (2.50% drop)

**Possible Explanations**:
1. **Dense-view semantics**: Legacy may have densified before zero-only attack
2. **Forward path difference**: Legacy vs R1 sparse-gated implementation
3. **Candidate generation**: Different approach to selecting "zero" positions

**Evidence**:
- Task2 log: `Model Type: sparse_int8_dense_zero`
- R1_T06 log: `sparse-gated : zero-only (dense-view search)`

**Impact on R1**:
- **Affects**: Interpretation of R1_T06/T07 zero_only results
- **Action**: Investigate semantic difference before claiming "zero-only ineffective"

**Verification Needed**:
```python
# Test: Does legacy Task2 actually preserve sparse_mask during forward?
# Compare: Apply same zero-only flip on both models, compare loss delta
```

---

### Difference #2: Checkpoint Baseline Accuracy

**Observation**:
- Legacy Task1-3: 92.10% baseline
- Legacy Task4-27: 86.50-86.57% baseline
- R1 all tasks: 92.33-92.40% baseline

**Explanation**:
- Task28 checkpoint: `task28_sparse_mask_fixed_finetune_int8_ckpt.pth`
- Legacy earlier tasks: `models/sparse_model.pth`
- Task28 is improved checkpoint (finetuned with mask fixed)

**Impact on R1**:
- R1 uses Task28 checkpoint (higher accuracy)
- Legacy Task4-27 results not directly comparable to R1

**Reference**: `debug_task1xx/code_path_audit.md` baseline table

---

### Difference #3: CSR vs 2:4 Encoding

**Observation**:
- Legacy Task4-5: CSR index encoding (column indices into sparse matrix)
- R1_T01-T05: 2:4 bitmask/position encoding

**Key Differences**:
| Aspect | CSR (Legacy) | 2:4 (R1) |
|--------|--------------|----------|
| Encoding | Column indices | 4-bit bitmask |
| Group size | Variable (NNZ per row) | Fixed (4 weights) |
| Collision | Possible (multiple indices → same position) | N/A (position encoding) |

**Impact on R1**:
- Cannot directly compare Task4/5 results to R1_T01-T05
- Flip taxonomy (Task10) needs adaptation for 2:4

---

## Known Limitations

### Limitation #1: Proxy Scoring vs Exact Verification

**Legacy**: Used Taylor approximation proxy scoring
```python
score = w_fp * (g_new - g_old)  # First-order Taylor
```

**R1**: Top-K exact verification (T02.1, T03.1, T06.1)
```python
# Apply flip, compute real loss, revert
new_loss = criterion(model(x), y)
delta = new_loss - baseline_loss
```

**Impact**:
- Legacy results may have non-monotonic loss
- R1 exact verification ensures monotonicity

**Evidence**:
- R1_T02: 7 loss decreases (proxy only)
- R1_T02.1: 0 loss decreases (exact verification)

---

### Limitation #2: Calib/Eval Sample Sizes

**Legacy**: Various configurations
- Task1-3: Not documented (likely 256/2000)
- Task13: Swept calib from 32 to 256

**R1**: Standardized
- calib_samples: 256
- eval_samples: 2000

**Impact**:
- Direct comparison requires matching sample sizes
- Task13 shows sensitivity to calib size

---

## Verification Checklist for R1

Before trusting R1 results, verify:

- [ ] All R1 scripts call `calibrate_all_layers()` after checkpoint load
- [ ] R1 zero_only uses same semantics as intended (not legacy "dense-zero")
- [ ] R1 nonzero_only preserves sparse_mask in forward
- [ ] R1 metadata attacks use exact verification (not proxy only)
- [ ] R1 uses Task28 checkpoint consistently
- [ ] R1 log format matches legacy for comparison

---

## Summary Table

| Bug/Diff | Category | Affects R1 | Action |
|----------|----------|------------|--------|
| quantized=False | Bug | T01-T05 | Verify calibrate_all_layers() |
| Zero-only effectiveness | Diff | T06/T07 | Investigate semantics |
| Baseline accuracy | Diff | Comparison | Use Task28 consistently |
| CSR vs 2:4 | Diff | T01-T05 vs Legacy | Note encoding difference |
| Proxy vs exact | Limitation | All | R1 uses exact (advantage) |
