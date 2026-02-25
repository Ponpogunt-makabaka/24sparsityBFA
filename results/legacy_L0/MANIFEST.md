# Legacy Archive Manifest

**Date**: 2026-02-16
**Purpose**: Archive all legacy results (0.xx series, Task1-23, Task4-8, debug artifacts, etc.) to maintain traceability while organizing the results directory.

---

## Archive Structure

```
results/legacy_0/
├── by_task/              # Task-organized legacy results
├── by_date/             # Chronological legacy results
├── task1_3_sparse_dense_comparison.png  # Original comparison plot
├── collision_impact_comparsion.png
└── ... (other legacy results)
```

---

## Legacy Categories

### 1. Original Task Series (0.xx numbering)
- **Task1-3**: Baseline BFA comparisons on Dense/Sparse INT8
- **Task4-8**: Sparse CSR index attacks, ImageNet expansion
- **Task9-17**: Extended analysis (collision, defense, robustness)
- **Task18-23**: Minimal closed-loop experiments
- **Task28**: Baseline audit and finetune

### 2. Debug Artifacts
- `debug_task1xx/` - Task1xx debugging experiments

### 3. Old New/Placeholder Results
- `new/` - Original Task1xx results (before fix)

---

## File Migration Log

### Moved to legacy_0/by_date/

All files dated before 2026-02-16 14:00:
- `task1_*.png`
- `task2_*.png`
- ...
- `task23_*.png`

### Moved to legacy_0/by_task/

Organized by task number:
- `task1_*/` → `legacy_0/by_task/task1/`
- `task2_*/` → `legacy_0/by_task/task2/`
- ...
- `task23_*/` → `legacy_0/by_task/task23/`

---

## Soft Links / Compatibility

For backward compatibility with scripts that hardcode old paths:
- Original Task1-3 comparison plot: `results/task1_3_sparse_dense_comparison.png` → `results/legacy_0/by_date/task1_3_sparse_dense_comparison.png`
- Task1-2 pkl files: preserved in `legacy_0/by_task/task1/`

---

## New Naming Convention (Task 1.n)

Starting from Task 1.1:
- **New pattern**: `task1.n_<taskname>.<ext>`
- **Example**: `task1.1_group_metadata_attack_curve.png`
- **Directory**: `results/task1_1/` for organized new results

---

## Notes

- All legacy results remain fully intact in `legacy_0/`
- Original `results/task1_3_sparse_dense_comparison.png` preserved for reference
- New comparison plots in `results/task1_1/` provide updated views
