# R1 Post-Delete Verification

**Generated**: 2026-02-23
**Status**: COMPLETED - All old versions successfully removed

---

## Deletion Results

### Successfully Deleted (13 files)

#### T02 Old Version (9 files)
- `run_R1_T02_group_metadata_index_1bit.py` ✓
- `results/R1/R1_T02_group_metadata_index_1bit_curve.png` ✓
- `results/R1/R1_T02_group_metadata_index_1bit_log.txt` ✓
- `results/R1/R1_T02_group_metadata_index_1bit_result.pkl` ✓
- `results/R1/R1_T02_group_metadata_index_1bit_table.csv` ✓
- `results/R1/R1_T02_group_metadata_index_1bit_rerun_curve.png` ✓
- `results/R1/R1_T02_group_metadata_index_1bit_rerun_log.txt` ✓
- `results/R1/R1_T02_group_metadata_index_1bit_rerun_result.pkl` ✓
- `results/R1/R1_T02_group_metadata_index_1bit_table.csv` ✓
- `results/R1/R1_T02_loss_change_detailed.png` ✓

#### T03 Old Version (4 files)
- `run_R1_T03_group_metadata_bitmask_swap_cost2.py` ✓
- `results/R1/R1_T03_group_metadata_bitmask_swap_cost2_curve.png` ✓
- `results/R1/R1_T03_group_metadata_bitmask_swap_cost2_log.txt` ✓
- `results/R1/R1_T03_group_metadata_bitmask_swap_cost2_result.pkl` ✓
- `results/R1/R1_T03_group_metadata_bitmask_swap_cost2_table.csv` ✓

### No Deletion Failures
All files were successfully removed.

---

## Remaining R1 Scripts (Latest Versions)

| Script | Version | Type |
|--------|---------|------|
| `run_R1_T01_group_metadata_index_anypattern.py` | T01 | Metadata: any pattern |
| `run_R1_T02_1_group_metadata_index_1bit_exact.py` | **T02.1** | Metadata: 1-bit reachable + exact |
| `run_R1_T03_1_group_metadata_bitmask_swap_cost2_exact.py` | **T03.1** | Metadata: cost-2 swap + exact |
| `run_R1_T04_bitmask_swaps50.py` | T04 | Metadata: extended swap |
| `run_R1_T05_joint_best_step_attack.py` | T05 | Joint: metadata + weight |
| `run_R1_T06_1_sparse_gated_dense_view_exact.py` | **T06.1** | Weight: dense format + exact |
| `run_R1_T06_1_samplebfa_style_dense_bfa.py` | T06.1 variant | Weight: sampleBFA style |
| `run_R1_T07_samplebfa_style_dense_bfa.py` | T07 | Baseline: sampleBFA comparison |

---

## Version Conflicts Resolved

- ✓ No more T02 (old) vs T02.1 confusion
- ✓ No more T03 (old) vs T03.1 confusion
- ✓ All analysis now based on exact verification versions

---

## Reports That Need Updating

The following reports in `results/R1/R1_logic_alignment/` may reference old T02/T03:
- `R1_T01-T05_Logical_Refactor_Plan.md`
- `R1_T01_to_T05_Diagnostic_Sheets.md`
- `R1_T05_vs_T06_Global_Equivalence_Conditions.md`

**Action**: These will be superseded by new re-diagnosis reports.
