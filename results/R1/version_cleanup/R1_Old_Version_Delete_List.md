# R1 Old Version Delete List

**Generated**: 2026-02-23
**Purpose**: List old Txx versions to be deleted (Txx.1 exists)

---

## Delete Policy

Only delete old version files when a corresponding `.1` version exists AND the `.1` version is the exact verification upgrade.

---

## Task 02: Delete Old T02 (Keep T02.1)

**Reason**: T02.1 adds Top-K Exact Verification, eliminating loss non-monotonicity

### Scripts to Delete
- `run_R1_T02_group_metadata_index_1bit.py`

### Results to Delete
- `results/R1/R1_T02_group_metadata_index_1bit_curve.png`
- `results/R1/R1_T02_group_metadata_index_1bit_log.txt`
- `results/R1/R1_T02_group_metadata_index_1bit_result.pkl`
- `results/R1/R1_T02_group_metadata_index_1bit_table.csv`
- `results/R1/R1_T02_group_metadata_index_1bit_rerun_curve.png`
- `results/R1/R1_T02_group_metadata_index_1bit_rerun_log.txt`
- `results/R1/R1_T02_group_metadata_index_1bit_rerun_result.pkl`
- `results/R1/R1_T02_group_metadata_index_1bit_rerun_table.csv`
- `results/R1/R1_T02_loss_change_detailed.png`

### Replacement (Keep)
- `run_R1_T02_1_group_metadata_index_1bit_exact.py`
- `results/R1/R1_T02_1_group_metadata_index_1bit_exact_*`

---

## Task 03: Delete Old T03 (Keep T03.1)

**Reason**: T03.1 adds Top-K Exact Verification for cost-2 swap attack

### Scripts to Delete
- `run_R1_T03_group_metadata_bitmask_swap_cost2.py`

### Results to Delete
- `results/R1/R1_T03_group_metadata_bitmask_swap_cost2_curve.png`
- `results/R1/R1_T03_group_metadata_bitmask_swap_cost2_log.txt`
- `results/R1/R1_T03_group_metadata_bitmask_swap_cost2_result.pkl`
- `results/R1/R1_T03_group_metadata_bitmask_swap_cost2_table.csv`

### Replacement (Keep)
- `run_R1_T03_1_group_metadata_bitmask_swap_cost2_exact.py`
- `results/R1/R1_T03_1_group_metadata_bitmask_swap_cost2_exact_*`

---

## Summary

| Task | Old Files | New Files | Action |
|------|-----------|-----------|--------|
| T02 | 9 files | 4 files (T02.1) | Delete old |
| T03 | 4 files | 4 files (T03.1) | Delete old |

**Total Files to Delete**: 13 files

---

## Files to KEEP (No .1 version exists)

- `run_R1_T01_group_metadata_index_anypattern.py` + results (no T01.1)
- `run_R1_T04_bitmask_swaps50.py` + results (no T04.1)
- `run_R1_T05_joint_best_step_attack.py` + results (no T05.1)
- `run_R1_T06_1_sparse_gated_dense_view_exact.py` + results (T06.1 is latest)
- `run_R1_T06_1_samplebfa_style_dense_bfa.py` (wrapper for T06.1, keep as variant)
- `run_R1_T07_samplebfa_style_dense_bfa.py` + results (no T07.1)
