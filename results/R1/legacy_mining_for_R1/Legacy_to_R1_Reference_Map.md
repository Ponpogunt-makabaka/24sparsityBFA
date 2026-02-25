# Legacy to R1 Reference Map

**Generated**: 2026-02-23
**Purpose**: Systematic mapping of legacy Task1-28 artifacts to R1_T01-R1_T07 reference needs
**Method**: File-based inventory + content analysis + semantic mapping

---

## Legend

| Priority | Meaning |
|----------|---------|
| **High** | Directly reusable, high confidence, low risk |
| **Medium** | Reusable with adaptation, moderate risk |
| **Low** | Indirect reference only, high risk of misuse |

---

## Task Mapping (by Legacy Task ID)

### Task 1: Dense INT8 BFA Attack

**File**: `task1_dense_int8_log.txt`

**Purpose**: Traditional BFA on dense INT8 ResNet20 - baseline reference

**Related R1 Tasks**:
- R1_T06 (dense-format BFA baseline)
- R1_T07 (sampleBFA-style comparison)

**Reusable Content**:
- **Log format**: Flips-Accuracy-Loss progression table
- **Bit naming**: `layer:idx:bit` format (e.g., `0:393:7`)
- **Attack pattern**: Global bit selection, MSB (bit7) dominance
- **Result**: 92.46% → 10.00% (50 flips)

**Key Findings**:
- Bit7 (MSB) flips dominate early attacks
- Accuracy plateaus at ~10% after ~15 flips
- Loss continues to increase even after accuracy plateau

**Risk Points**:
- Dense model semantics differ from sparse-gated R1
- No sparse_mask involved
- Legacy checkpoint path differs from R1

**Priority**: **High** (for R1_T07 baseline comparison)

---

### Task 2: Sparse INT8 Zero-Only Attack

**File**: `task2_sparse_dense_zero_log.txt`

**Purpose**: BFA on sparse INT8, attacking ONLY masked positions

**Related R1 Tasks**:
- R1_T06 zero_only mode
- R1_T07 zero_only mode

**Reusable Content**:
- **Zero-only definition**: Attack bits where `sparse_mask==0`
- **Result**: 92.10% → 12.43% (50 flips)
- **Insight**: Zero-only attacks still effective but less than global

**Key Findings**:
- Zero-only can still drop accuracy significantly
- Contradicts R1_T06/T07 findings (zero-only ineffective)

**Risk Points**:
- Legacy used `sparse_int8_dense_zero` (different forward semantics?)
- Possible semantic difference: legacy may have densified before attack

**Priority**: **High** (investigate semantic discrepancy with R1)

---

### Task 3: Sparse INT8 Non-Zero Attack

**File**: `task3_sparse_dense_nonzero_log.txt`

**Purpose**: BFA on sparse INT8, attacking ONLY non-masked positions

**Related R1 Tasks**:
- R1_T06 nonzero_only mode
- R1_T07 nonzero_only mode

**Reusable Content**:
- **Non-zero-only definition**: Attack bits where `sparse_mask==1`
- **Result**: 92.10% → 10.00% (50 flips)
- **Insight**: Non-zero-only as effective as global attack

**Key Findings**:
- Non-zero-only matches Task 1 (dense) effectiveness
- Confirms that sparse-gated weights are the vulnerability

**Risk Points**:
- Legacy semantics may differ from R1 sparse-gated forward

**Priority**: **High** (baseline for nonzero_only effectiveness)

---

### Task 4: CSR Index Attack (Collision-Allowed)

**File**: `task4_sparse_csr_index_log.txt`

**Purpose**: Attack CSR column indices with collision allowed

**Related R1 Tasks**:
- R1_T01 (any pattern metadata attack)
- R1_T02 (1-bit reachable index attack)

**Reusable Content**:
- **CSR index format**: `layer:csr_index:bit`
- **Collision semantics**: Multiple indices map to same physical position
- **Result**: 86.54% → 53.94% (50 flips)

**Key Findings**:
- Early flips can INCREASE accuracy (e.g., step 1: 86.54% → 89.76%)
- Collision causes unexpected behavior (some flips have zero effect)

**Risk Points**:
- Legacy Task4 used different checkpoint (86.54% baseline vs 92.40% R1)
- CSR encoding differs from 2:4 position encoding

**Priority**: **Medium** (collision effect reference for R1_T01)

---

### Task 5: CSR Index Attack (Non-Collision)

**File**: `task5_csr_non_collision_log.txt`

**Purpose**: Attack CSR indices with collision avoidance

**Related R1 Tasks**:
- R1_T02 (1-bit reachable, NCSA semantics)
- R1_T01 (any pattern, but with collision check)

**Reusable Content**:
- **NCSA pattern**: Non-Collision Space Attack
- **Result**: 92.10% → 11.10% (50 flips)
- **Effectiveness**: Matches Task 1 (dense) attack

**Key Findings**:
- Avoiding collision dramatically improves attack effectiveness
- Step format: `layer:g:old->new` (group-level moves)

**Risk Points**:
- Group encoding differs from R1 position encoding
- CSR indices vs 2:4 bitmask encoding

**Priority**: **High** (NCSA effectiveness reference)

---

### Task 10: Flip Outcome Taxonomy

**File**: `task10_flip_outcome_log.txt`

**Purpose**: Categorize flip effects (rewire, delete, merge, clamp-noop)

**Related R1 Tasks**:
- All metadata attacks (R1_T01-T05)

**Reusable Content**:
- **Flip taxonomy**:
  - `rewire`: Valid position change with effect
  - `delete`: One weight becomes inaccessible
  - `merge`: Two weights map to same position
  - `clamp-noop`: No effective change
- **Task4 outcome breakdown**: 11 rewire, 26 delete, 13 clamp-noop

**Key Findings**:
- Collision causes "delete" outcomes (26 out of 50 in Task4)
- High delete rate explains Task4's lower effectiveness

**Risk Points**:
- Taxonomy defined for CSR encoding, may need adaptation for 2:4

**Priority**: **Medium** (outcome analysis framework)

---

### Task 11: Metadata Integrity Defense

**File**: `task11_defense_log.txt`

**Purpose**: Parity/CRC defense against metadata attacks

**Related R1 Tasks**:
- All metadata attacks (defense perspective)

**Reusable Content**:
- **Defense mechanisms**:
  - `none`: No protection (86.57% → 13.38%)
  - `parity`: +1 bit/group, 100% detection
  - `crc`: +0.0625 bits/group, 100% detection
- **Overhead analysis**: Storage vs metadata-only overhead

**Key Findings**:
- Both parity and CRC achieve 100% detection rate
- Parity: +25% metadata-only overhead
- CRC: +1.56% metadata-only overhead

**Priority**: **Low** (defense context, not directly relevant to attack R1)

---

### Task 18: Bitmask Validity Under 1-Bit Flip

**File**: `task18_bitmask_validity_log.txt`

**Purpose**: Test if single-bit flips preserve bitmask popcount=2

**Related R1 Tasks**:
- R1_T03 (bitmask cost-2 swap)
- R1_T04 (bitmask swaps)

**Reusable Content**:
- **Critical finding**: 0% of single-bit flips preserve popcount=2
- **Post-flip breakdown**: 50% popcount=1, 50% popcount=3
- **Conclusion**: Cost-2 swap required for bitmask attacks

**Key Findings**:
- Single-bit flips on 4-bit bitmask ALWAYS invalidate metadata
- This justifies Task 20/Task 3's cost-2 approach

**Priority**: **High** (justifies R1_T03/T04 design)

---

### Task 20: Bitmask Metadata Swap (Cost=2)

**File**: `task20_bitmask_swap_log.txt`

**Purpose**: Attack bitmask with swap (flip 1→0 + 0→1)

**Related R1 Tasks**:
- R1_T03 (cost-2 swap baseline)
- R1_T04 (extended swap attack)

**Reusable Content**:
- **Swap semantics**: One bit 1→0, one bit 0→1 (preserves popcount=2)
- **Physical budget**: 50 flips = 25 logical swaps
- **Result**: 86.57% → 35.50% (25 swaps)

**Key Findings**:
- Cost-2 constraint reduces but doesn't eliminate effectiveness
- Scoring: `w_fp * (g_new - g_old)` (Taylor approximation)

**Risk Points**:
- Legacy used different baseline accuracy (86.57% vs 92.40%)
- Proxy scoring vs exact verification

**Priority**: **High** (R1_T03 direct predecessor)

---

### Task 21: Position Encoding Compare

**File**: `task21_position_compare_log.txt`

**Purpose**: Compare metadata NCSA vs weight MSB attack

**Related R1 Tasks**:
- R1_T01 (position encoding attack)
- R1_T02 (1-bit reachable)

**Reusable Content**:
- **Comparison results**:
  - metadata_ncsa: 86.57% → 9.67%
  - weight_msb: 86.57% → 9.91%
- **Insight**: Metadata and weight attacks achieve similar effectiveness

**Key Findings**:
- Position-based metadata attack nearly matches weight MSB
- NCSA (non-collision) is critical for metadata effectiveness

**Priority**: **Medium** (position encoding validation)

---

### Task 28: Closed Loop Sanity Check

**File**: `task28_closed_loop_sanity_log.txt`, `task28_sparsity_baseline_audit_log.txt`

**Purpose**: Verify checkpoint compatibility and baseline accuracy

**Related R1 Tasks**:
- All R1 tasks (checkpoint source)

**Reusable Content**:
- **Checkpoint paths**:
  - `task28_sparse_mask_fixed_finetune_int8_ckpt.pth` (R1 uses this)
  - `task28_sparse_mask_fixed_finetune_ckpt.pth`
- **Baseline audit**: Sparsity structure verification
- **Sanity check**: dense vs sparse attack comparison

**Key Findings**:
- Checkpoint contains int8_weights, scale, sparse_mask
- Baseline accuracy: 92.33% (matches R1)
- **Critical**: Must call `calibrate_all_layers()` to set `quantized=True`

**Risk Points**:
- **KNOWN BUG**: Loading checkpoint doesn't set `quantized=True`
- See `debug_task1xx/final_root_cause_report.md`

**Priority**: **High** (checkpoint source + critical bug documentation)

---

## Summary by Category

### Checkpoint & Baseline

| Legacy Task | Artifact | R1 Relevance |
|-------------|----------|--------------|
| Task 28 | INT8 checkpoint | Direct use |
| Task 28 | Baseline audit log | Verification |

### Traditional BFA

| Legacy Task | Artifact | R1 Relevance |
|-------------|----------|--------------|
| Task 1 | Dense INT8 attack log | R1_T07 baseline |
| Task 2 | Zero-only attack log | R1_T06/T07 comparison |
| Task 3 | Non-zero-only attack log | R1_T06/T07 validation |

### Metadata Attacks

| Legacy Task | Artifact | R1 Relevance |
|-------------|----------|--------------|
| Task 4 | CSR collision log | R1_T01 collision reference |
| Task 5 | NCSA log | R1_T01/T02 effectiveness |
| Task 20 | Cost-2 swap log | R1_T03 baseline |
| Task 21 | Position compare | R1_T01/T02 validation |

### Analysis Tools

| Legacy Task | Artifact | R1 Relevance |
|-------------|----------|--------------|
| Task 10 | Flip taxonomy | Outcome analysis |
| Task 11 | Defense overhead | Defense context |
| Task 18 | Bitmask validity | R1_T03/T04 justification |

### Critical Bugs

| Bug | Location | Impact | Fix |
|-----|----------|--------|-----|
| `quantized=False` after load | debug_task1xx/ | Metadata ignored | Call `calibrate_all_layers()` |

---

## Recommended Reading Order for R1 Agent

1. **Task 28 audit** → Understand checkpoint structure
2. **debug_task1xx/** → Learn critical bug
3. **Task 1** → Traditional BFA baseline
4. **Task 18** → Bitmask validity constraints
5. **Task 20** → Cost-2 swap semantics
6. **Task 10** → Flip outcome taxonomy
