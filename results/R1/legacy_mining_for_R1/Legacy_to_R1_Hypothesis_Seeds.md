# Legacy to R1 Hypothesis Seeds

**Generated**: 2026-02-23
**Purpose**: Provide testable hypotheses for further investigation based on legacy mining
**Format**: 10+ hypotheses with supporting evidence and verification approach

---

## Hypothesis #1: Legacy Task2 Zero-Only Used Different Semantics

**Claim**: Legacy Task2 (zero-only: 92.10% → 12.43%) did NOT preserve sparse_mask during forward, unlike R1_T06 (zero_only: 92.40% → 89.90%)

**Supporting Evidence**:
1. Legacy result shows 79.67% accuracy drop
2. R1 result shows only 2.50% drop
3. Legacy log shows "Model Type: sparse_int8_dense_zero"
4. The word "dense" in model type suggests densification

**Verification Approach** (minimal experiment):
```python
# Load legacy checkpoint (if available)
# Apply one zero-only flip
# Check: does forward use sparse_mask?
# Compare loss delta with/without sparse_mask
```

**Expected Outcome**:
- If true: Legacy zero-only bypassed sparse_mask
- Impact: R1 zero_only result is CORRECT for sparse-gated semantics

**Files to Reference**:
- `results/legacy_L0/by_task/task2_sparse_dense_zero_log.txt`
- `Legacy_Known_Bugs_And_Fixes.md` (Difference #1)

---

## Hypothesis #2: R1_T02 Loss Non-Monotonicity Caused by Proxy Scoring

**Claim**: R1_T02's 7 loss decreases are due to Taylor approximation proxy scoring, not exact loss verification

**Supporting Evidence**:
1. R1_T02: 7 loss decreases documented
2. R1_T02.1 (exact verification): 0 loss decreases
3. Legacy Task5 used proxy scoring exclusively
4. Taylor approximation: `score = w_fp * (g_new - g_old)` ignores higher-order terms

**Verification Approach**:
```python
# Re-run R1_T02 candidates with exact verification
# Compare: proxy_score vs exact_delta
# Count: how many proxy-positive flips are exact-negative?
```

**Expected Outcome**:
- Confirms: Exact verification eliminates loss non-monotonicity
- Impact: R1_T02.1 is the correct baseline

**Files to Reference**:
- R1_T02 results vs R1_T02.1 results
- `Legacy_BFA_Baseline_Candidates.md` (Limitation #1)

---

## Hypothesis #3: Task18 Bitmask Validity Implies R1_T02 Must Use Position Encoding

**Claim**: Task18 proved 0% of single-bit flips preserve 4-bit bitmask popcount=2, therefore R1_T02's "1-bit reachable" MUST use position encoding, not bitmask

**Supporting Evidence**:
1. Task18: "valid_flips: 0 (0.0000%)"
2. Task18: Post-flip popcount always 1 or 3, never 2
3. R1_T02 uses position encoding (two 2-bit indices)
4. Single-bit flip on position encoding preserves one index, changes other

**Verification Approach**:
```python
# Test: Flip one bit in position encoding (e.g., 14(0b1110) -> 6(0b0110))
# Verify: popcount maintained in derived bitmask?
# Verify: New pattern is valid (no collision)?
```

**Expected Outcome**:
- Confirms: Position encoding enables 1-bit reachable transitions
- Explains: Why R1_T02 works despite Task18 findings

**Files to Reference**:
- `results/legacy_L0/by_task/task18_bitmask_validity_log.txt`
- R1_T02 log (shows position encoding usage)

---

## Hypothesis #4: Task5 NCSA Effectiveness Matches R1_T01 Any Pattern

**Claim**: Legacy Task5 (NCSA: 92.10% → 11.10%) and R1_T01 (any pattern: 92.33% → 16.75%) achieve similar effectiveness despite encoding differences

**Supporting Evidence**:
1. Both achieve ~80% accuracy drop
2. Both use non-collision constraint
3. Task5: CSR column indices
4. R1_T01: Position encoding

**Verification Approach**:
```python
# Compare per-step accuracy curves
# Measure: correlation between flip sequences
# Analyze: Do both attack similar layer/group patterns?
```

**Expected Outcome**:
- Confirms: Non-collision constraint is key, not encoding
- Impact: Metadata attack effectiveness is encoding-agnostic

**Files to Reference**:
- `results/legacy_L0/by_task/task5_csr_non_collision_log.txt`
- R1_T01 results

---

## Hypothesis #5: R1_T05 Joint Attack Always Choosing Weight_Bit Matches Task21

**Claim**: R1_T05 (100% weight_bit choice) validates Task21 finding that weight MSB ≈ metadata NCSA in effectiveness

**Supporting Evidence**:
1. R1_T05: All 50 steps chose weight_bit, 0 chose index/bitmask
2. Task21: metadata_ncsa (86.57% → 9.67%) ≈ weight_msb (86.57% → 9.91%)
3. When equally effective, simpler action preferred

**Verification Approach**:
```python
# Compare R1_T05 weight_bit flips to Task21 weight_msb flips
# Check: Are same layers/bits targeted?
# Analyze: Why does metadata lose in joint selection?
```

**Expected Outcome**:
- Confirms: Weight-bit attack has slight advantage in selection
- Explains: Joint attack's unanimous weight_bit choice

**Files to Reference**:
- R1_T05 results
- `results/legacy_L0/by_task/task21_position_compare_log.txt`

---

## Hypothesis #6: quantized=False Bug May Affect Legacy Task4-27 Results

**Claim**: Legacy Task4-27 may have had `quantized=False` bug, explaining weaker metadata attack results compared to Task28+R1

**Supporting Evidence**:
1. Task4-27 baseline: 86.50-86.57%
2. Task28/R1 baseline: 92.33-92.40%
3. Task4 result: 86.54% → 53.94% (weak)
4. Task5 result: 92.10% → 11.10% (strong, uses different checkpoint)
5. Bug discovered and fixed in task1xx series

**Verification Approach**:
```python
# Check: Do Task4-27 scripts call calibrate_all_layers()?
# If not: Re-run Task4 with quantized=True
# Compare: Attack effectiveness with/without fix
```

**Expected Outcome**:
- If true: Legacy Task4-27 metadata attacks underestimated
- Impact: R1_T01-T05 are more reliable than Task4-27

**Files to Reference**:
- `debug_task1xx/final_root_cause_report.md`
- Legacy Task4-5 comparison

---

## Hypothesis #7: Task10 Flip Taxonomy "Delete" Explains Task4 Weakness

**Claim**: Task4's low effectiveness (26.40% drop) is due to 26/50 flips being "delete" outcomes (weight becomes inaccessible)

**Supporting Evidence**:
1. Task10 Task4 breakdown: 11 rewire, 26 delete, 13 clamp-noop
2. Task4 accuracy: 86.54% → 53.94% (only 32.60% drop)
3. Task5 (NCSA, no delete): 92.10% → 11.10% (81.00% drop)

**Verification Approach**:
```python
# Re-analyze Task4 flip outcomes
# Count: How many "delete" flips occurred?
# Test: Remove "delete" flips, recompute effectiveness
```

**Expected Outcome**:
- Confirms: Collision → delete → reduced effectiveness
- Validates: NCSA (Task5, R1_T01/T02) necessity

**Files to Reference**:
- `results/legacy_L0/by_task/task10_flip_outcome_log.txt`
- Task4 vs Task5 comparison

---

## Hypothesis #8: Task13 Calib Sweep Shows Optimal calib=256

**Claim**: Task13 calibration sweep demonstrates that calib_samples=256 (used by R1) is near-optimal for attack effectiveness

**Supporting Evidence**:
1. R1 uses calib_samples=256 consistently
2. Task13 sweeps calib from 32 to 256
3. Need to verify: does effectiveness saturate at 256?

**Verification Approach**:
```python
# Extract Task13 data
# Plot: Attack effectiveness vs calib_samples
# Check: Diminishing returns above 256?
```

**Expected Outcome**:
- If true: R1's calib=256 is well-justified
- If false: R1 may be suboptimal

**Files to Reference**:
- `results/legacy_L0/by_task/task13_calib_sweep_log.txt`
- `results/legacy_L0/by_task/task13_calib_sweep_table.csv`

---

## Hypothesis #9: Task20 Cost-2 Swap Effectiveness Limited by Proxy Scoring

**Claim**: Legacy Task20 (swap: 86.57% → 35.50%, 51.07% drop) would be more effective with exact verification like R1_T03.1

**Supporting Evidence**:
1. Task20 uses proxy scoring: `w_fp * (g_new - g_old)`
2. R1_T03.1 uses exact verification (proxy + exact)
3. Need comparison: Task20 vs R1_T03.1

**Verification Approach**:
```python
# Compare Task20 swap effectiveness to R1_T03.1
# Normalize: Compare per-flip accuracy drop
# Test: Re-run Task20 with exact verification
```

**Expected Outcome**:
- If true: Exact verification improves swap attack
- Impact: R1_T03.1 supersedes Task20

**Files to Reference**:
- `results/legacy_L0/by_task/task20_bitmask_swap_log.txt`
- R1_T03.1 results (when available)

---

## Hypothesis #10: R1_T06/T07 Nonzero-Only Matches Task3 Due to Same Target Set

**Claim**: R1_T06 nonzero_only (82.25% drop) matches Task3 (82.10% drop) because both attack the same critical weights (sparse_mask==1 positions)

**Supporting Evidence**:
1. Task3: 92.10% → 10.00% (82.10% drop)
2. R1_T06 nonzero: 92.40% → 10.15% (82.25% drop)
3. Nearly identical effectiveness despite different implementations
4. Both target non-masked weights

**Verification Approach**:
```python
# Compare flip sequences: Task3 vs R1_T06 nonzero
# Check: Do both attack similar layers/positions?
# Analyze: Correlation between flip rankings
```

**Expected Outcome**:
- Confirms: Sparse vulnerability is in non-zero weights
- Validates: Zero-only should be weak (if semantics match)

**Files to Reference**:
- `results/legacy_L0/by_task/task3_sparse_dense_nonzero_log.txt`
- R1_T06 results

---

## Hypothesis #11: Task11 Defense Overhead Analysis Applies to R1 Metadata

**Claim**: Task11 defense overhead analysis (parity: +25%, CRC: +1.56%) provides realistic cost estimates for defending R1 metadata attacks

**Supporting Evidence**:
1. Task11 computed per-group overhead
2. Metadata storage: 4 bits per 2:4 group
3. Parity: +1 bit/group = 25% metadata overhead
4. CRC: +0.0625 bits/group = 1.56% overhead

**Verification Approach**:
```python
# Compute R1 metadata storage requirements
# Compare: R1 position encoding (4 bits) vs Task11 bitmask (4 bits)
# Apply: Task11 overhead formulas to R1
```

**Expected Outcome**:
- Confirms: Defense overhead comparable for R1
- Impact: R1 metadata attacks defendable with low cost

**Files to Reference**:
- `results/legacy_L0/by_task/task11_defense_log.txt`

---

## Hypothesis #12: Task17 Seed Robustness Shows R1 Results Are Seed-Dependent

**Claim**: Task17 seed robustness analysis suggests R1 single-seed results may vary across different random seeds

**Supporting Evidence**:
1. R1 all tasks use seed=0 only
2. Task17 tested multiple seeds
3. Need to check: Task17 variance magnitude

**Verification Approach**:
```python
# Extract Task17 seed variance data
# Compute: Std dev of final accuracy across seeds
# Apply: Variance bounds to R1 results
```

**Expected Outcome**:
- If high variance: R1 should report multi-seed results
- If low variance: Single-seed R1 sufficient

**Files to Reference**:
- `results/legacy_L0/by_task/task17_seed_robustness_log.txt`
- `results/legacy_L0/by_task/task17_seed_robustness_table.csv`

---

## Summary of Hypotheses

| ID | Claim | Impact | Verification Effort |
|----|-------|--------|---------------------|
| #1 | Task2 zero-only semantic diff | High | Medium (code analysis) |
| #2 | Proxy scoring causes non-monotonicity | High | Low (data comparison) |
| #3 | Task18 → R1_T02 position encoding | High | Low (logical deduction) |
| #4 | Task5 ≈ R1_T01 effectiveness | Medium | Medium (curve comparison) |
| #5 | R1_T05 validates Task21 | Medium | Low (result comparison) |
| #6 | quantized=False bug in Task4-27 | High | High (re-run needed) |
| #7 | Delete outcome explains Task4 | Medium | Low (outcome analysis) |
| #8 | calib=256 optimal | Low | Low (data extraction) |
| #9 | Exact verification improves Task20 | Medium | High (re-run needed) |
| #10 | Task3 ≈ R1_T06 nonzero | Medium | Medium (sequence comparison) |
| #11 | Task11 defense applies to R1 | Low | Low (calculation) |
| #12 | Seed dependency affects R1 | Medium | Medium (variance analysis) |

---

## Recommended Priority for Investigation

1. **Hypothesis #1** (Task2 semantics) - resolves major discrepancy
2. **Hypothesis #2** (Proxy vs exact) - validates R1 methodology
3. **Hypothesis #7** (Delete outcomes) - explains attack mechanisms
4. **Hypothesis #6** (quantized bug in legacy) - validates legacy data
5. **Hypothesis #3** (Position encoding necessity) - design justification
