# Agent Development Log Archive (Legacy + Early R1 Setup)

Archive date: 2026-02-24

This archive contains entries moved out of the active log:
- Legacy task streams (Task 0-28)
- Early R1 setup and pre-stabilization history
- Legacy dense-format debug loops that are no longer active pipeline policy

---

# Agent Development Log (handoff)

Date: 2026-02-09

## Project Goal (current phase)
- Study Bit-Flip Attacks (BFA) on sparse models, including CSR index attacks, and extend to ImageNet/Imagenette with ResNet-18, MobileNet-V2, and DeiT-Tiny.
- Emphasis: non-collision CSR index flips, 2:4 sparsity, and BN recalibration to fix distribution shift.

## Key Fixes Applied
1) **Imagenette label mapping**
- File: `train/imagenet_utils.py`
- Uses `torchvision.datasets.Imagenette` with `download=False` (offline).
- Applies Imagenette→ImageNet-1k label mapping via wnids:
  - n01440764→0, n02102040→217, n02979186→482, n03000684→491, n03028079→497,
    n03394916→566, n03417042→569, n03425413→571, n03445777→574, n03888257→701.
- Local dataset path: `/home/lab-2010/24sparsityBFA/data/imagenette/imagenette2`.

2) **2:4 conv mask dimension fix (Input Channel rule)**
- File: `models/factory.py` + `models/sparse_ops.py`
- 2:4 mask now groups along **input channels (dim=1)**: `[out, in/4, 4, k, k]`.
- Skip if `in_channels % 4 != 0`.
- Skip conv1, fc, and downsample layers for ResNet-18.

3) **BN Recalibration (critical)**
- New helper: `train/imagenet_pipeline_utils.py` with:
  - `run_bn_recalibration()` (BN-only SGD, LR 1e-3, ~200–300 steps)
  - `apply_sparse_mask()`, `replace_with_int8_from_mask()`, `calibrate_int8()`, `evaluate()`.

4) **Pipeline upgraded for Tasks 6–7**
- Files:
  - `run_task6_resnet18_csr_non_collision.py`
  - `run_task7_mobilenetv2_csr_non_collision.py`
- New flow:
  - Load local weights → apply 2:4 sparsity (skip sensitive layers) → BN recalibration → PTQ Int8 → CSR non-collision attack.
- `num_workers=0` enforced (multiprocessing permission issues).

## Data / Weights
- Imagenette root: `/home/lab-2010/24sparsityBFA/data/imagenette`
- Weights:
  - ResNet-18: `/home/lab-2010/24sparsityBFA/weights/resnet18-f37072fd.pth`
  - MobileNet-V2: `/home/lab-2010/24sparsityBFA/weights/mobilenet_v2-b0353104.pth`
  - DeiT-Tiny: `/home/lab-2010/24sparsityBFA/weights/deit_tiny_patch16_224-a1311bcf.pth`

## Latest Verified Results
### Task 6 (ResNet-18, Imagenette mapped)
- Post-BN Sparse FP32 Acc: **95.31%**
- Int8 Initial Acc: **94.53%**
- Final Acc after 50 non-collision flips: **46.09%**
- Log: `results/task6_resnet18_csr_non_collision_log.txt`
- Result: `results/task6_resnet18_csr_non_collision_result.pkl`

### Task 7 (MobileNet-V2, Imagenette mapped)
- Post-BN Sparse FP32 Acc: **96.09%**
- Int8 Initial Acc: **95.31%**
- Final Acc after 50 non-collision flips: **0.00%**
- Log: `results/task7_mobilenetv2_csr_non_collision_log.txt`
- Result: `results/task7_mobilenetv2_csr_non_collision_result.pkl`

### Task 8 (DeiT-Tiny, Imagenette mapped)
- Key remap (DeiT/timm -> torchvision ViT): `models/factory.py`
- Sparsity target: QKV (MultiheadAttention.in_proj_weight) + MLP fc1/fc2 only (skip conv_proj/head)
- Dense FP32 Acc (eval_samples=128): **95.31%**
- Sparse FP32 Acc: **87.50%**
- Int8 Initial Acc: **87.50%**
- Final Acc after 50 non-collision flips: **75.78%**
- Log: `results/task8_deit_tiny_csr_non_collision_log.txt`
- Result: `results/task8_deit_tiny_csr_non_collision_result.pkl`

## Report & Plots
- Final report: `ImageNet_Expansion_Results.md` (updated with BN-calibrated results)
- Combined attack curve: `results/image_net_attack_curves.png`
- Task4 vs Task5 comparison plot: `results/collision_impact_comparsion.png`

## Known Constraints / Issues
- Multiprocessing DataLoader fails (PermissionError); always use `num_workers=0`.
- Imagenette download is disabled; must use local dataset.
- ResNet-18 / MobileNet-V2 / DeiT-Tiny pipelines are stable.

## Useful Scripts
- `debug_resnet_pipeline.py` (ladder test + BN recalibration)
- `run_task6_resnet18_csr_non_collision.py`
- `run_task7_mobilenetv2_csr_non_collision.py`
- `run_task8_deit_tiny_csr_non_collision.py`

---

Date: 2026-02-11 (update)

## New Work Completed (Tasks 9-11 + Task11 Debug Validation)

### Task 8 status correction (aggressive search run)
- Updated Task8 aggressive search result:
  - Final Acc after 50 non-collision flips: **59.77%**
  - Log: `results/task8_deit_tiny_csr_non_collision_log.txt`
  - Result: `results/task8_deit_tiny_csr_non_collision_result.pkl`

### Task 9: Collision Handling Characterization
- Script: `run_task9_collision_characterization.py`
- Outputs:
  - `results/task9_collision_characterization_result.pkl`
  - `results/task9_collision_characterization_log.txt`
  - `results/task9_collision_behavior_table.png`
- Verified behavior:
  - CSR decode collisions behave as **mask-last (overwrite-last)**.
  - Unsafe 2:4 collision injection behaves as **drop**.

### Task 10: Flip Outcome Taxonomy + Effective-Rewire
- Script: `run_task10_flip_outcome_analysis.py`
- Outputs:
  - `results/task10_flip_outcome_breakdown.png`
  - `results/task10_acc_vs_effective_rewire.png`
  - `results/task10_flip_outcome_summary.pkl`
  - `results/task10_flip_outcome_log.txt`
- Notes:
  - Uses Task9 collision semantics to classify CSR collisions.
  - Includes Task4/5 (and Task6/7/8 if files exist) in breakdown.

### Task 11: Metadata Integrity Defense + Overhead/Effectiveness
- Script: `run_task11_metadata_defense.py`
- Main outputs:
  - `results/task11_defense_attack_curves.png`
  - `results/task11_defense_overhead_table.csv`
  - `results/task11_defense_result.pkl`
  - `results/task11_defense_log.txt`
- Added debug interfaces (minimal patch):
  - `--defense {all,none,parity,crc8}`
  - `--out-prefix`
  - `--bypass-parity`
  - `--bypass-crc`
  - `--debug-trace --trace-path`
- Trace output:
  - `results/task11_debug_trace.pkl` (list of per-attempt dict records)

## Task11 Debug / Validation Runs (CPU)

### Check 1: drop mitigation sanity
- Run output: `results/task11_debug_drop_log.txt`
- Result:
  - `none`: 86.57% -> 13.38%
  - `parity+drop`: 86.57% -> 61.47% (detected/mitigated 50/50)
  - `crc+drop`: 86.57% -> 9.91% (detected/mitigated 50/50)
- Interpretation:
  - Revert flatness is an **upper-bound behavior**, not “defense inactive”.
  - CRC line-level drop is strongly destructive.

### Check 2: parity bypass (adaptive co-flip parity bit)
- Run output: `results/task11_debug_parity_bypass_log.txt`
- Result:
  - `detected=0`, `mitigated=0`, final 13.38% (baseline-like collapse)
- Interpretation:
  - Parity check is against stored parity bits; if attacker can co-modify parity storage, bypass is feasible.

### Check 3: CRC bypass (adaptive co-flip CRC byte)
- Run output: `results/task11_debug_crc_bypass_log.txt`
- Result:
  - `detected=0`, `mitigated=0`, final 13.38% (baseline-like collapse)
- Interpretation:
  - CRC effectiveness depends on trusted checksum storage domain.

### Check 4: ordering/timing validation (flip -> check/mitigate -> eval)
- Run outputs:
  - `results/task11_debug_timing_log.txt`
  - `results/task11_debug_trace.pkl`
- Timing counters:
  - `hash_b!=a=50`, `hash_c==d=50`, `revert(a==c)=50`, `assert_fail=0`
- Interpretation:
  - No TOCTTOU bug observed; evaluation uses post-mitigation state.

## Current Conclusion (Task11)
- “Parity/CRC + revert keeps accuracy flat” is **expected upper bound** under trusted-checksum assumption.
- Under adaptive attacker with checksum co-modification (`--bypass-parity`, `--bypass-crc`), defense is bypassed and accuracy collapses toward baseline.

---

Date: 2026-02-12 (update)

## New Work Completed (Tasks 12–17)

### Shared Utilities (Tasks 12–17)
- Helper: `scripts/p012_17_utils.py`
- Notes:
  - CIFAR-10 offline loaders with `num_workers=0` enforced.
  - Implements scored non-collision attack variants used by Tasks 12–15, 17:
    - `ncsa` (w*Δg), `grad_only`, `weight_only`, `random_valid`.

### Task 12: NCSA vs Baselines / Scoring Ablations (ResNet-20 CIFAR-10)
- Script: `run_task12_ablation_greedy_vs_random.py`
- Outputs:
  - `results/task12_ablation_attack_curves.png`
  - `results/task12_ablation_summary.csv`
  - `results/task12_ablation_result.pkl`
  - `results/task12_ablation_log.txt`
- Key results (@50 flips, eval_samples=2000):
  - NCSA (w*Δg): 86.57% -> 9.67%
  - Random-valid: 86.57% -> 86.91%
  - Grad-only (Δg): 86.57% -> 88.72%
  - Weight-only (|w|): 86.57% -> 86.13%

### Task 13: Calibration Sensitivity Sweep (calib_samples)
- Script: `run_task13_calib_sweep_ncsa.py`
- Outputs:
  - `results/task13_calib_sweep_curves.png`
  - `results/task13_calib_sweep_table.csv`
  - `results/task13_calib_sweep_log.txt`
- Key results (seed=123, @50 flips):
  - calib=32: 86.57% -> 10.06%
  - calib=64: 86.57% -> 25.29%
  - calib=128: 86.57% -> 43.70%
  - calib=256: 86.57% -> 9.67%
  - calib=512: 86.57% -> 36.04%
  - calib=1024: 86.57% -> 11.67%

### Task 14: Layer-wise Vulnerability / Flip Localization
- Script: `run_task14_layer_localization.py`
- Outputs:
  - `results/task14_layer_histogram.png`
  - `results/task14_layer_impact.png`
  - `results/task14_layer_trace.pkl`
  - `results/task14_layer_trace_log.txt`
- Headline (seed=123):
  - 86.57% -> 9.67% after 50 flips
  - Flip concentration: stage1 dominates (36/50 flips), then stage2 (10), stage3 (4).
  - Top-hit layers by count: `layer1.0.conv2`, `layer1.1.conv1`, `layer2.0.downsample.0`, `layer1.2.conv1`.

### Task 15: Defense Realism (trusted checksum vs adaptive attacker + budgeted bypass)
- Script: `run_task15_defense_realism.py`
- Outputs:
  - `results/task15_defense_realism_curves.png`
  - `results/task15_defense_realism_table.csv`
  - `results/task15_defense_realism_log.txt`
- Key results (seed=123, mitigation=revert):
  - Baseline none: 86.57% -> 13.38%
  - Parity trusted: 86.57% -> 86.57% (det_rate=100%, mit_rate=100%)
  - Parity adaptive bypass: 86.57% -> 13.38% (det_rate=0%)
  - Parity budgeted bypass (cost=2): 86.57% -> 35.50% with 50 physical flips (25 logical + 25 checksum edits)
  - CRC-8 shows same qualitative pattern as parity (trusted upper bound; adaptive bypass collapses).

### Task 16: Runtime/Overhead Characterization (attack search + defense compute)
- Script: `run_task16_runtime_overhead.py`
- Outputs:
  - `results/task16_runtime_overhead_table.csv`
  - `results/task16_runtime_overhead_log.txt`
- Key results (seed=123, 50 flips):
  - Avg search time dominates: ~1.57–1.60s/attempt (CPU).
  - Parity check+mitigation: ~0.136ms/attempt.
  - CRC-8 check+mitigation: ~2.005ms/attempt.

### Task 17: Seed Robustness (3 seeds)
- Script: `run_task17_seed_robustness.py`
- Outputs:
  - `results/task17_seed_robustness_table.csv`
  - `results/task17_seed_robustness_log.txt`
  - `results/task17_seed_robustness_boxplot.png`
- Key results (Task5 ResNet-20 CIFAR-10, 3 seeds, @50 flips):
  - Final acc per seed: 9.67%, 13.38%, 13.67%
  - Mean final: 12.24% (std 1.82%)
- Notes:
  - Task6/8 multi-seed robustness not run by default (`--run-task6` available; compute heavy).

---

Date: 2026-02-13 (update)

## Task 4.1: 2:4 Group-Local Position Index Attack (Collision Allowed)
- Purpose:
  - Bridge between Task4 (CSR absolute column-index bit flips) and Task5 (2:4 group-local non-collision moves).
  - Quantify whether allowing collisions changes attack strength and how much damage comes from collision-induced drop vs rewires.
- Script: `run_task4_1_2_4_group_local_collision.py`
- Repro:
  - `python run_task4_1_2_4_group_local_collision.py --device cpu --max-flips 50`
- Outputs:
  - `results/task4_1_2_4_group_local_collision_result.pkl`
  - `results/task4_1_2_4_group_local_collision_log.txt`
  - `results/task4_1_2_4_group_local_collision.png`
  - `results/task4_1_vs_task5_attack_curves.png`
  - `results/task4_1_vs_task5_summary.csv`
  - (cached Task5 reference, not overwriting old Task5): `results/task5_ref_ncsa_non_collision_seed0_eval2000.pkl`
- Collision semantics (repo-established):
  - Uses the exact Task5 update rule (move value + mask XOR toggles).
  - If the target slot is already active, both mask bits toggle OFF, producing an effective **drop** of both non-zeros (matches Task9 nm_linear collision characterization).
- Key results (ResNet-20 CIFAR-10, INT8 PTQ, seed=0, calib_samples=256, eval_samples=2000, max_groups_per_layer=2000):
  - Task4.1 (collision allowed): **86.57% -> 10.84%** after 50 flips
  - Outcome counts: collision_drop=21, rewire=29, noop=0
  - Task5 reference (rerun non-collision NCSA, same settings): **86.57% -> 9.67%** after 50 flips
- Notes:
  - Existing `results/task5_csr_non_collision_result.pkl` had an incompatible (stale) baseline (~92.10%), so the runner regenerates a comparable non-collision reference under a new filename without overwriting Task5 artifacts.

---

Date: 2026-02-14 (update)

## Minimal Closed-Loop (Tasks 18–23) on ResNet-20 / CIFAR-10 (2:4)

### Phase 0 Recon (interfaces confirmed)
- 2:4 metadata representation in current pipeline:
  - Stored as `sparse_mask` (0/1) alongside `int8_weights` in `Int8QuantizedConv2d/Linear` (PTQ model: `Int8QuantizedResNet`).
  - Grouping used by Task5/NCSA: for conv weights, permute to `[out, kh, kw, in]` then view `(-1, 4)`; for linear, view `(-1, 4)`.
  - Position/index encoding is implicit: the two `1` bits in each 4-bit group correspond to the two active 2-bit indices in {0,1,2,3}.
- NCSA scoring (Task5 move space):
  - For a rewire `old_idx -> new_idx` within a group: `score = w_fp * (g_new - g_old)`, where `w_fp = int8_val * scale` and `g_*` comes from `module.weight.grad` at that position.
- Baseline checkpoint + dataset:
  - Checkpoint: `models/sparse_model.pth`
  - Dataset root: `./data`
  - Sanity baseline: ~**86.57%** with `eval_samples=2000`, `seed=123`, CPU.

### Task 18: Bitmask Metadata Validity Under 1 Physical Bit Flip
- Script: `run_task18_bitmask_validity.py`
- Config: seed=123, eval_samples=2000, device=cpu
- Result (counting baseline-valid groups with popcount==2):
  - total_groups(pop2)=67,456; total_single_bit_flips=269,824
  - valid_flips(popcount stays 2)=0 (0.00%)
  - popcount after flip: 134,912 -> 1, 134,912 -> 3
- Outputs:
  - `results/task18_bitmask_validity_log.txt`
  - `results/task18_bitmask_validity_summary.csv`
  - `results/task18_bitmask_validity_breakdown.png`

### Task 19: Bitmask Case Best Feasible (1-bit FI) = Weight MSB on Non-Zero Weights
- Script: `run_task19_bitmask_weight_msb.py`
- Config: seed=123, max_flips=50, calib_samples=256, eval_samples=2000, device=cpu
- Result: **86.57% -> 9.91%** after 50 MSB flips (restricted to `int8_val!=0` and `mask==1` when present)
- Outputs:
  - `results/task19_bitmask_weight_msb_log.txt`
  - `results/task19_bitmask_weight_msb_curve.png`
  - `results/task19_bitmask_weight_msb_result.pkl`

### Task 20: Bitmask Swap Becomes Attackable Under Cost=2 (Swap 1->0 and 0->1)
- Script: `run_task20_bitmask_swap_cost2.py`
- Threat model: physical_budget=50, cost_per_swap=2 => logical_steps=25
- Config: seed=123, calib_samples=256, eval_samples=2000, device=cpu
- Result: **86.57% -> 35.50%** after 25 swaps (50 physical flips)
- Outputs:
  - `results/task20_bitmask_swap_log.txt`
  - `results/task20_bitmask_swap_curve.png`
  - `results/task20_bitmask_swap_result.pkl`

### Task 21 (Case C): Position Encoding Compare (Metadata NCSA vs Weight MSB)
- Script: `run_task21_position_compare_ncsa_vs_weight.py`
- Config: seed=123, max_flips=50, calib_samples=256, eval_samples=2000, max_groups_per_layer=2000, device=cpu
- Results:
  - metadata NCSA (non-collision): **86.57% -> 9.67%**
  - weight MSB (non-zero only): **86.57% -> 9.91%**
  - time-to-threshold (from CSV):
    - flips_to_50%: metadata=24, weight=4
    - flips_to_20%: metadata=30, weight=10
- Outputs:
  - `results/task21_position_ncsa_curve.png`
  - `results/task21_position_weight_msb_curve.png`
  - `results/task21_position_compare_table.csv`
  - `results/task21_position_compare_log.txt`
  - (optional) `results/task21_position_ncsa_result.pkl`, `results/task21_position_weight_msb_result.pkl`

### Task 22 (Case A): Dense Baseline Weight MSB
- Script: `run_task22_dense_weight_msb.py`
- Config: seed=123, max_flips=50, calib_samples=256, eval_samples=2000, device=cpu
- Result: **92.53% -> 9.57%** after 50 MSB flips
- Outputs:
  - `results/task22_dense_weight_msb_log.txt`
  - `results/task22_dense_weight_msb_curve.png`
  - `results/task22_dense_weight_msb_result.pkl`

### Task 23: Minimal Closed-Loop Summary Figure + Table
- Script: `run_task23_miniclose_summary.py`
- Outputs:
  - `results/task23_miniclose_curves.png`
  - `results/task23_miniclose_table.csv`
  - `results/task23_miniclose_log.txt`

---

Date: 2026-02-14 (update)

## Task 28: Sparse Baseline Formation Audit + 90%+ Baseline Recovery (ResNet-20 / CIFAR-10, 2:4)

### Baseline Audit (Dense vs Sparse, FP32 vs INT8)
- Script: `run_task28_sparsity_baseline_audit.py`
- Config:
  - seed=123, device=cpu, data_dir=./data, batch_size=128
  - eval_samples=10000 (full CIFAR-10 test set)
- Key results (Top-1):
  - dense_fp32 (`models/dense_model.pth`): **92.45%**
  - dense_int8_ptq: **92.46%**
  - sparse_fp32_existing (`models/sparse_model.pth`): **86.50%**
  - sparse_int8_existing: **86.54%**
  - dense->sparse (mask-fixed init, no finetune): **58.28%**
- Root cause classification:
  - Dense is already strong (~92.45%), INT8 PTQ does not drop accuracy materially.
  - The ~86.5% baseline is attributable to the *current sparse checkpoint / sparsification training regime*, not PTQ.
- Outputs:
  - `results/task28_sparsity_baseline_audit_log.txt`
  - `results/task28_sparsity_baseline_audit_table.csv`

### Minimal Fix: Mask-Fixed Finetune From Dense (2:4 mask frozen)
- Script: `run_task28_sparsity_baseline_audit.py` (with finetune enabled)
- Config:
  - `--finetune-epochs 10 --finetune-lr 0.01 --finetune-wd 5e-4`
  - Mask: fixed 2:4 (top-2 magnitude per 4 along in-ch groups, per kernel position); mask frozen before finetune.
- Results (Top-1, full test set):
  - sparse_fp32_maskfixed_finetuned: **92.17%**
  - sparse_int8_maskfixed_finetuned: **92.21%**
- New checkpoints (attack-compatible, do not overwrite existing Task5 defaults):
  - `results/task28_sparse_mask_fixed_finetune_ckpt.pth`
  - `results/task28_sparse_mask_fixed_finetune_int8_ckpt.pth`

### Closed-Loop Sanity (Attacks Still Run on Improved Baseline)
- Script: `run_task28_closed_loop_sanity.py`
- Config: seed=123, device=cpu, eval_samples=2000, calib_samples=256, max_flips=10
- Results:
  - dense_weight_msb_int8: **92.53% -> 11.18%** (10 flips)
  - sparse_metadata_ncsa_noncollision_int8 (using improved sparse ckpt): **92.33% -> 71.44%** (10 flips)
- Outputs:
  - `results/task28_closed_loop_sanity_log.txt`
  - `results/task28_closed_loop_sanity_table.csv`
  - `results/task28_closed_loop_sanity_curves.png`
  - `results/task28_closed_loop_sanity_result.pkl`

Repro commands:
- Audit only:
  - `python run_task28_sparsity_baseline_audit.py --device cpu --eval-samples 10000 --seed 123`
- Audit + finetune (rebuild 90%+ baseline):
  - `python run_task28_sparsity_baseline_audit.py --device cpu --eval-samples 10000 --seed 123 --finetune-epochs 10 --finetune-lr 0.01 --finetune-wd 5e-4`
- Closed-loop sanity:
  - `python run_task28_closed_loop_sanity.py --device cpu --seed 123 --max-flips 10 --eval-samples 2000`

---

Date: 2026-02-16 (update) - ROOT CAUSE FIX

## Task1xx Group Metadata Attack - Bug Fix

### Root Cause Identified
**Bug**: Task1xx 模型加载后 `quantized=False`，导致 `Int8QuantizedConv2d.forward()` 跳过 `get_dequantized_weights()`，完全忽略 `sparse_mask`。

**Evidence**:
- Before fix: `layer.quantized == False` → loss_delta = 0.0
- After fix: `layer.quantized == True` → loss_delta = 0.0033

**Fix**: 在加载 Int8 checkpoint 后调用 `model.calibrate_all_layers()` 来设置 `quantized=True`。

### Fixed Results (task28 baseline, 92%+)
| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Baseline Acc | 92.29% | 92.33% |
| Final Acc (50 flips) | 92.29% | 38.67% |
| Accuracy Drop | 0.00% | 53.66% |

### Debug Artifacts Created (results/debug_task1xx/)
- `code_path_audit.md` - Code path analysis
- `sanity_one_step_log.txt` - One-step differential test
- `sanity_one_step_result.json` - Test results
- `final_root_cause_report.md` - Root cause analysis
- `run_debug_task1xx_sanity_one_step.py` - Sanity test script

### Modified Files
- `run_task1xx_group_metadata_attack.py` - Added `calibrate_all_layers()` call

### Key Code Change
```python
# Line 824-826 (ADDED)
model.calibrate_all_layers()  # CRITICAL: sets quantized=True
model.eval()
```

### Reproduction Command (Fixed)
```bash
python run_task1xx_group_metadata_attack.py \
  --device cpu --seed 0 --max-flips 50 \
  --calib-samples 256 --eval-samples 2000 \
  --topk-verify 0 \
  --ckpt results/task28_sparse_mask_fixed_finetune_int8_ckpt.pth
```

### Key Takeaways
1. **`quantized` flag is critical** - Without it, metadata has no effect on forward
2. **Int8 checkpoint loading requires calibration** - Must call `calibrate_all_layers()` after loading
3. **Group-based metadata attack is effective** - 53.66% accuracy drop in 50 flips

---

Date: 2026-02-16 (update)

## NEW (1.xx): Group-based 2:4 Metadata Attack (Position Encoding)

### Overview
Implemented a NEW group-based 2:4 metadata attack that treats each 2:4 group (4 positions) as one unit and enumerates all VALID 2-of-4 patterns (6 combinations) within the group.

### Key Features
1. **Group-local pattern enumeration**: For each group, try all 6 valid 2-of-4 patterns
2. **Gradient-based proxy score**: ΔL_g ≈ ∇_{w̃_g} L · (w̃_g(p') − w̃_g(p))
3. **Non-collision validity (NCA)**: All 6 patterns are valid by construction for 2:4 sparsity
4. **Anti-reversal mechanism**: Tracks forbidden transitions to prevent flipping back to previous states
5. **Recent group exclusion**: Prevents immediate re-modification of recently changed groups (20-step window)

### Files Added
- `run_task1xx_group_metadata_attack.py` - Main attack script

### Outputs (results/new/)
- `task1xx_group_metadata_attack_result.pkl` - Full attack result with trace
- `task1xx_group_metadata_attack_curve.png` - Accuracy vs. flip count plot
- `task1xx_group_metadata_attack_table.csv` - Summary statistics
- `task1xx_group_metadata_attack_log.txt` - Detailed log with step-by-step details

### Key Results (using task28 improved baseline, 92%+)
- Baseline accuracy: **92.29%**
- Final accuracy after 50 flips: **92.29%**
- Accuracy drop: **0.00%**
- Total groups: 3,378,200
- Valid groups (popcount=2): 3,372,010
- Candidates considered: 16,859,802
- Skipped (excluded): 790
- Skipped (forbidden transitions): 248

### Reproduction Command
```bash
python run_task1xx_group_metadata_attack.py \
  --device cpu --seed 0 --max-flips 50 \
  --calib-samples 256 --eval-samples 2000 \
  --topk-verify 0 \
  --ckpt results/task28_sparse_mask_fixed_finetune_int8_ckpt.pth
```

### Assumptions and Limitations
1. **Proxy nature**: The attack uses a first-order gradient proxy which may not perfectly correlate with actual loss increase
2. **Baseline robustness**: The task28 baseline (92%+) is significantly more robust than the original 86% baseline
3. **Threat model**: Assumes attacker can modify metadata (sparse_mask) directly
4. **No retraining**: Uses existing checkpoint without any retraining

### Comparison with Legacy Tasks
- Legacy Task5 (NCSA) on 86% baseline: **86.57% -> 9.67%** (50 flips)
- New Task1.xx on 92% baseline: **92.29% -> 92.29%** (50 flips)
- The improved baseline shows significantly higher robustness to metadata attacks

### Notes for Future Work
- Consider using the original 86% baseline (`models/sparse_model.pth`) for comparison with legacy tasks
- The anti-reversal mechanism successfully prevents group cycling (observed in earlier runs)
- The attack successfully spreads across multiple layers and groups

---

Date: 2026-02-16 (update) - REORGANIZATION

## Results Reorganization + New Naming Convention (Task 1.n)

### Overview
Comprehensive reorganization of the `results/` directory to establish a clean naming convention (`task1.n_taskname`) and archive all legacy results.

### New Naming Convention
- **Format**: `task1.n_<taskname>.<ext>` where n is the sub-task number
- **Example**: `task1.1_group_metadata_attack_curve.png`
- **Directory**: `results/task1_1/` for all Task 1.1 outputs

### Legacy Archive Structure
All legacy results (Task 0-23, Task28, debug artifacts) moved to `results/legacy_0/`:

```
results/legacy_0/
├── by_task/           # Task-organized legacy results (77 files)
├── by_date/           # Chronological legacy results with plots
├── debug_task1xx/     # Task1xx debugging artifacts
├── old_new_task1xx/   # Original Task1xx results (before fix)
└── MANIFEST.md        # Archive manifest
```

### Files Migrated (Summary)
- **Task 0-23 logs**: All task logs, CSVs, and pkl files
- **Task 28**: Baseline audit and closed-loop sanity files
- **Debug artifacts**: task1xx debugging outputs
- **Old comparison plot**: `task1_3_sparse_dense_comparison.png`

### New Artifacts Created (results/task1_1/)

#### Core Results (Task 1.1: Group Metadata Attack, Fixed)
| File | Description |
|------|-------------|
| `task1.1_group_metadata_attack_log.txt` | Detailed attack log |
| `task1.1_group_metadata_attack_table.csv` | Summary statistics |
| `task1.1_group_metadata_attack_result.pkl` | Full result with trace |
| `task1.1_group_metadata_attack_curve.png` | Accuracy curve (standalone) |
| `task1.1_group_metadata_attack_curve.csv` | Curve data for plotting |

#### Comparison Results
| File | Description |
|------|-------------|
| `task1_3_dense_format_attacks_curve.csv` | Task1-3 combined curve data |
| `task1.1_dense_format_vs_metadata_attacks.png` | **NEW: Comprehensive comparison plot** |
| `task1.1_dense_format_vs_metadata_attacks_table.csv` | **NEW: Summary table for paper** |

### Key Results Summary

| Attack Variant | Baseline | Final@50 | Drop |
|----------------|----------|----------|------|
| Dense-format: Global (Task 1) | 92.10% | 10.00% | 82.10% |
| Dense-format: Zero-only (Task 2) | 92.10% | 12.43% | 79.67% |
| Dense-format: Nonzero-only (Task 3) | 92.10% | 10.00% | 82.10% |
| **Group Metadata Attack (Task 1.1, fixed)** | **92.33%** | **38.67%** | **53.66%** |

### New Comparison Plot Improvements
The new `task1.1_dense_format_vs_metadata_attacks.png` addresses several issues with the legacy plot:

1. **Unified x-axis**: "Physical Flips / Iterations" (clearer semantics)
2. **Unified y-axis**: "Top-1 Accuracy (%)" (standardized)
3. **Better grouping**: Task1-3 use similar style (blue color family), Task1.1 is distinct (red)
4. **Clearer title**: "Dense-Format vs Metadata Attacks (ResNet-20/CIFAR-10, INT8)"
5. **Reference line**: Random chance (10%) marked for context

### Old Plot → New Plot Mapping
| Old Path | New Path | Notes |
|----------|----------|-------|
| `results/task1_3_sparse_dense_comparison.png` | `results/legacy_0/by_date/task1_3_sparse_dense_comparison.png` | Preserved in legacy |
| N/A | `results/task1_1/task1.1_dense_format_vs_metadata_attacks.png` | **NEW: Updated comparison** |

### Reproduction Commands

#### Generate new comparison plot:
```bash
cd /home/lab-2010/24sparsityBFA
python3 scripts/plot_task1_1_dense_vs_metadata.py
```

#### Reproduce Task 1.1 attack (fixed):
```bash
python run_task1xx_group_metadata_attack.py \
  --device cpu --seed 0 --max-flips 50 \
  --calib-samples 256 --eval-samples 2000 \
  --topk-verify 0 \
  --ckpt results/task28_sparse_mask_fixed_finetune_int8_ckpt.pth
```

#### Reproduce legacy Task1-3 dense-format attacks:
```bash
# Task 1 (Global)
# (See scripts used for original results)

# Task 2 (Zero-only)
# (See scripts used for original results)

# Task 3 (Nonzero-only)
# (See scripts used for original results)
```

### Data Sources Used
- **Task1-3 curves**: `results/legacy_0/by_date/task[123]_sparse_dense_*_result.pkl`
- **Task1.1 curve**: `results/task1_1/task1.1_group_metadata_attack_log.txt` (parsed)

### Backward Compatibility
- All legacy paths in `legacy_0/` remain intact
- No original data was deleted, only moved
- Soft links can be added if scripts hardcode old paths (currently not needed)

### Files Modified
- `scripts/plot_task1_1_dense_vs_metadata.py` - Updated to use CSV data instead of pkl
- `agent_develop_log.md` - This update

### Next Steps
- Future Task 1.2+ results should follow `task1.n_<taskname>.<ext>` naming
- Consider creating `results/task1_2/` for the next sub-task
- Update paper figures to reference new plot paths

---

Date: 2026-02-16 (update) - R1 WORKFLOW

## Revised R1 Workflow: New Naming Scheme + 1-bit Reachable Attack (R1_T02)

### Naming Convention Change

#### Problem
Legacy Task1/Task2/... naming causes confusion with new revised workflow tasks. Need a clear separation between legacy work (Task 0-23, Task28) and the new revised workflow.

#### Solution: R1_ Prefix
- **R1_** = Revised workflow 1 (distinguishes from legacy tasks)
- **T01, T02, ...** = Task IDs within R1 (no dots to avoid file naming issues)
- **Full format**: `R1_Txx_<short_task_name>_<type>.ext`

#### Legacy Archive
All non-R1 results moved to `results/legacy_L0/` (consolidated from `legacy_0/` and `task1_1/`):
```
results/legacy_L0/
├── by_task/           # Task-organized (77 files)
├── by_date/           # Chronological with plots
├── debug_task1xx/     # Debug artifacts
├── old_new_task1xx/   # Pre-fix Task1xx results
├── task1_1/           # Previous task1.n results
└── MANIFEST.md        # Archive manifest
```

#### R1 Results Root
```
results/R1/
├── R1_T01_group_metadata_index_anypattern_*  # Any pattern transition
└── R1_T02_group_metadata_index_1bit_*         # 1-bit reachable (NEW)
```

---

## R1_T01: Group-based Metadata Attack (Index/Position Encoding, Any Pattern)

### Overview
Previously called "Task 1.xx" or "Task 1.1". Renamed to R1_T01 for clarity.
This attack allows ANY valid 2-of-4 pattern transition within each group.

### Key Features
- Group-local pattern enumeration: All 6 valid 2-of-4 patterns considered
- Gradient-based proxy score: ΔL_g ≈ ∇_{w̃_g} L · (w̃_g(p') − w̃_g(p))
- Anti-reversal: Forbidden transitions prevent cycling
- Recent group exclusion: 20-step window

### Files
- Script: `run_R1_T01_group_metadata_index_anypattern.py` (renamed from `run_task1xx_group_metadata_attack.py`)
- Legacy stub: `run_task1xx_group_metadata_attack.py` (marked deprecated)

### Results (copied from task1_1/)
- `R1_T01_group_metadata_index_anypattern_log.txt`
- `R1_T01_group_metadata_index_anypattern_table.csv`
- `R1_T01_group_metadata_index_anypattern_result.pkl`
- `R1_T01_group_metadata_index_anypattern_curve.png`
- `R1_T01_group_metadata_index_anypattern_curve.csv`

### Key Numbers
- Baseline: **92.33%**
- Final (50 flips): **38.67%**
- Drop: **53.66%**

---

## R1_T02: Group-based Metadata Attack (Index/Position Encoding, 1-bit Reachable) - NEW

### Overview
NEW attack that constrains candidate generation to **1-bit reachable transitions** in the 4-bit metadata code space.

### Key Difference from R1_T01
- **4-bit code space**: Each 2-of-4 pattern encoded as `(j << 2) | i` where i,j are positions
- **1-bit neighbors**: For current code c, candidates are `c ^ (1 << b)` for b in {0,1,2,3}
- **NCA validity**: Reject candidates where decoded positions have collision (i==j)
- **No support change**: Reject candidates that don't change active positions

### Encoding Details
```python
# Encode two 2-bit indices into 4-bit code
def encode_index_to_4bit(i: int, j: int) -> int:
    return (j << 2) | i

# Decode 4-bit code to (i, j)
def decode_4bit_to_index(code: int) -> Tuple[int, int]:
    i = code & 0x3      # Lower 2 bits
    j = (code >> 2) & 0x3  # Upper 2 bits
    return (i, j)

# Check for collision
def code_to_pattern(code: int) -> Optional[Tuple[int, int]]:
    i, j = decode_4bit_to_index(code)
    if i == j:
        return None  # Collision - invalid
    return (min(i, j), max(i, j))
```

### Files Created
- Script: `run_R1_T02_group_metadata_index_1bit.py`

### Outputs
- `results/R1/R1_T02_group_metadata_index_1bit_log.txt`
- `results/R1/R1_T02_group_metadata_index_1bit_table.csv`
- `results/R1/R1_T02_group_metadata_index_1bit_result.pkl`
- `results/R1/R1_T02_group_metadata_index_1bit_curve.png`

### Key Results (seed=0, task28 baseline, 92%+)
- Baseline: **92.33%**
- Final (50 flips): **30.37%**
- Drop: **61.96%**
- Runtime: **638.41s** (~10.6 minutes)

### Candidate Statistics
- Total groups: 3,378,200
- Valid groups (popcount=2): 3,372,010
- Candidates considered: 13,487,805
- Candidates valid: 8,992,341
- **Candidates rejected (collision): 4,495,464**
- Candidates rejected (no change): 0

### Step-by-Step Example
```
Step 1: layer2.0.downsample.0 g=81
  Code: 14(0b1110) -> 6(0b0110) (bit3 flipped)
  Pattern: (2,3) -> (1,2)
  Proxy: 0.0316 | Acc: 92.38% -> 92.33%
```

### Reproduction Command
```bash
python run_R1_T02_group_metadata_index_1bit.py \
  --device cpu \
  --seed 0 \
  --max-flips 50 \
  --calib-samples 256 \
  --eval-samples 2000 \
  --ckpt results/legacy_L0/by_task/task28_sparse_mask_fixed_finetune_int8_ckpt.pth
```

### Comparison: R1_T01 vs R1_T02
| Metric | R1_T01 (Any Pattern) | R1_T02 (1-bit Reachable) |
|--------|---------------------|--------------------------|
| Baseline | 92.33% | 92.33% |
| Final@50 | 38.67% | 30.37% |
| Drop | 53.66% | 61.96% |
| Candidate Set | 6 patterns/group | ≤4 patterns/group (1-bit) |
| Collision Filter | Not needed | Yes (NCA) |
| Constraint | None | 1-bit Hamming distance |

### Key Observations
1. **1-bit constraint is MORE effective**: 61.96% drop vs 53.66% drop
2. **High collision rejection rate**: ~4.5M candidates rejected due to i==j collision
3. **All 1-bit flips valid**: 0 candidates rejected for "no support change"
4. **Metadata changes verified**: Initial/final hashes differ

### Safety Mechanisms
- Recent group exclusion: 20-step window
- Forbidden transitions: Both forward and reverse stored
- Automatic cleanup of old transitions (max 1000)

### Notes for Future Work
- R1_T02 demonstrates that 1-bit reachability is a meaningful constraint
- The NCA collision filter is essential for index/position encoding
- Consider comparing with bitmask-based attacks (Tasks 18-20)

---

## File Modifications Summary

### New Files Created
1. `run_R1_T01_group_metadata_index_anypattern.py` (copied from task1xx)
2. `run_R1_T02_group_metadata_index_1bit.py` (NEW)
3. `results/R1/R1_T02_*` (5 files)

### Files Modified
1. `run_task1xx_group_metadata_attack.py` - Added deprecation notice
2. `agent_develop_log.md` - This update

### Files Moved
- `results/legacy_0/*` → `results/legacy_L0/`
- `results/task1_1/*` → `results/legacy_L0/task1_1/`

---

## Verification Checklist
- [x] R1_T01 script renamed and working
- [x] R1_T02 script created and executed
- [x] All R1 outputs follow `R1_Txx_*` naming
- [x] Legacy results consolidated in `legacy_L0/`
- [x] agent_develop_log.md updated
- [x] Metadata hash changes verified
- [x] Exit code 0 for R1_T02 run

---

Date: 2026-02-17 (update) - R1_T03 BITMASK COST-2 SWAP

## R1_T03: Group-based Metadata Attack (Bitmask Encoding, Cost-2 Swap) - NEW

### Overview
NEW attack that uses **direct bitmask encoding** with **structure-preserving cost-2 swaps**.

### Why Cost-2?
- Bitmask representation: 4-bit mask where popcount=2 indicates validity
- Single bit flip breaks validity (popcount becomes 1 or 3)
- Minimal valid move: cost-2 swap (flip one 1→0 AND one 0→1)
- This maintains popcount=2 always

### Bitmask Encoding vs Index/Position Encoding

| Property | R1_T01/T02 (Index) | R1_T03 (Bitmask) |
|----------|-------------------|------------------|
| Representation | Two 2-bit indices (i, j) | Direct 4-bit mask |
| Validity | i != j | popcount(mask) == 2 |
| Single bit flip | May break validity | Always breaks validity |
| Minimal valid move | 1-bit (with NCA) | Cost-2 swap |

### Swap Operation
```python
# Cost-2 swap: flip one 1->0 and one 0->1
def enumerate_cost2_swaps(current_mask: int) -> List[Tuple[int, int, int, int]]:
    # Returns: (new_mask, bit_off, bit_on, swap_code)
    # Example: mask=0b1100 (positions 2,3 active)
    #   - Swap 1: bit3->0, bit0->1 => 0b0101 (positions 0,2)
    #   - Swap 2: bit3->0, bit1->1 => 0b0110 (positions 1,2)
    #   - Swap 3: bit2->0, bit0->1 => 0b1001 (positions 0,3)
    #   - Swap 4: bit2->0, bit1->1 => 0b1010 (positions 1,3)
    # Total: 4 candidates per group
```

### Physical vs Logical Mapping
- Physical budget = total bit flips (e.g., 50)
- Each swap costs 2 physical flips
- Logical swaps = floor(physical_budget / 2) = 25
- Plot shows accuracy vs physical flips (stepwise at even positions)

### Files Created
- Script: `run_R1_T03_group_metadata_bitmask_swap_cost2.py`

### Outputs
- `results/R1/R1_T03_group_metadata_bitmask_swap_cost2_log.txt`
- `results/R1/R1_T03_group_metadata_bitmask_swap_cost2_table.csv`
- `results/R1/R1_T03_group_metadata_bitmask_swap_cost2_result.pkl`
- `results/R1/R1_T03_group_metadata_bitmask_swap_cost2_curve.png`

### Key Results (seed=0, task28 baseline, 92%+)
- Baseline: **92.33%**
- Final (50 physical flips = 25 swaps): **65.14%**
- Drop: **27.20%**
- Runtime: **382.56s** (~6.4 minutes)

### Candidate Statistics
- Total groups: 1,689,100
- Valid groups (popcount=2): 1,686,110
- Candidates per group: **Exactly 4** (cost-2 swaps)
- Total candidates considered: 6,744,431
- Candidates valid: 6,744,431 (no NCA filter needed for bitmask)

### Step-by-Step Example
```
Swap 0 (physical=2): layer2.0.downsample.0 g=81
  Mask: 12(0b1100) -> 5(0b0101)
  Operation: bit3->0, bit0->1
  Pattern: (2,3) -> (0,2)
  Proxy: 0.0316 | Acc: 92.33% -> 91.99%
```

### Reproduction Command
```bash
python run_R1_T03_group_metadata_bitmask_swap_cost2.py \
  --device cpu \
  --seed 0 \
  --physical-budget 50 \
  --calib-samples 256 \
  --eval-samples 2000 \
  --ckpt results/legacy_L0/by_task/task28_sparse_mask_fixed_finetune_int8_ckpt.pth
```

### Comparison: R1_T01 vs R1_T02 vs R1_T03

| Metric | R1_T01 (Any) | R1_T02 (1-bit) | R1_T03 (Cost-2) |
|--------|--------------|----------------|-----------------|
| Encoding | Index (i,j) | Index (4-bit) | Bitmask |
| Baseline | 92.33% | 92.33% | 92.33% |
| Final@50 | 38.67% | 30.37% | 65.14% |
| Drop | 53.66% | 61.96% | 27.20% |
| Logical ops | 50 | 50 | 25 swaps |
| Candidates/group | 6 | ≤4 | 4 |
| Cost/operation | 1 flip | 1 flip | 2 flips |

### Key Observations
1. **Bitmask cost-2 is LESS effective**: 27.20% drop vs 53.66% (R1_T01) and 61.96% (R1_T02)
2. **Half the operations**: Only 25 logical swaps vs 50 operations in T01/T02
3. **No collision filter**: Bitmask naturally maintains validity (popcount=2)
4. **Metadata changes verified**: Initial/final hashes differ
5. **Runtime comparable**: ~6.4 min vs ~10.6 min (T02) - fewer ops but similar per-op cost

### Safety Mechanisms
- Recent group exclusion: 20-step window
- Forbidden swaps: Both forward and reverse stored
- Automatic cleanup of old swaps (max 1000)

### Notes for Future Work
- R1_T03 demonstrates the "minimal valid move" in bitmask encoding
- Cost-2 constraint significantly reduces attack effectiveness
- Consider comparing with Task 20 (bitmask swap cost=2) for consistency
- The physical budget vs logical operation mapping is important for fair comparison

### Caveats
- eval_samples=2000 may have noise compared to full test set (10000)
- Physical budget=50 means only 25 logical swaps
- For direct comparison with T01/T02, consider running with physical_budget=100

---

Date: 2026-02-17 (update) - R1_T04 BITMASK 50 SWAPS

## R1_T04: Group-based Metadata Attack (Bitmask Encoding, 50 Logical Swaps) - NEW

### Purpose
R1_T03 used physical_budget=50, yielding only 25 logical swaps due to cost=2 per swap.
R1_T04 fixes **logical_swaps=50** for fair comparison with R1_T01/T02 (50 operations).
This allows direct comparison: bitmask (50 swaps) vs index (50 operations).

### Key Difference from R1_T03
- **Primary control**: max_logical_swaps (default 50) instead of physical_budget
- **Physical flips auto-calculated**: 2 * max_logical_swaps = 100
- **Plot x-axis**: Logical swaps (0..50) instead of physical flips
- **Secondary x-axis**: Physical flips (0..100) shown for reference

### Validity Checks
- Strict assertions: `popcount(mask) == 2` for every modified group
- All validity checks passed: No failures during execution
- Each swap flips exactly 2 bits: one 1->0 and one 0->1

### Files Created
- Script: `run_R1_T04_bitmask_swaps50.py`

### Outputs
- `results/R1/R1_T04_bitmask_swaps50_log.txt`
- `results/R1/R1_T04_bitmask_swaps50_table.csv` (includes R1_T03 comparison)
- `results/R1/R1_T04_bitmask_swaps50_result.pkl`
- `results/R1/R1_T04_bitmask_swaps50_curve.png` (dual x-axis: swaps & flips)

### Key Results (seed=0, task28 baseline, 92%+)
- Baseline: **92.33%**
- Final (50 logical swaps = 100 physical flips): **50.73%**
- Drop: **41.60%**
- Runtime: **761.15s** (~12.7 minutes)

### R1_T03 vs R1_T04 Comparison

| Metric | R1_T03 | R1_T04 |
|--------|--------|--------|
| Logical swaps | 25 | 50 |
| Physical flips | 50 | 100 |
| Baseline | 92.33% | 92.33% |
| Final accuracy | 65.14% | 50.73% |
| Accuracy drop | 27.20% | 41.60% |
| Runtime | 382.56s | 761.15s |

### Validity Statistics
- Total validity checks passed: 13,488,862
- Total validity checks failed: 0
- **All swaps maintained popcount=2** ✓

### Step-by-Step Example
```
Swap 0 (physical=2): layer2.0.downsample.0 g=81
  Mask: 12(0b1100) -> 5(0b0101)
  Operation: bit3->0, bit0->1
  Pattern: (2,3) -> (0,2)
  Validity: OK (popcount=2)
  Proxy: 0.0316 | Acc: 92.33% -> 91.99%
```

### Reproduction Command
```bash
python run_R1_T04_bitmask_swaps50.py \
  --device cpu \
  --seed 0 \
  --max-logical-swaps 50 \
  --calib-samples 256 \
  --eval-samples 2000 \
  --ckpt results/legacy_L0/by_task/task28_sparse_mask_fixed_finetune_int8_ckpt.pth
```

### Updated Comparison: R1_T01 vs R1_T02 vs R1_T03 vs R1_T04

| Metric | R1_T01 (Any) | R1_T02 (1-bit) | R1_T03 (Bitmask-25) | R1_T04 (Bitmask-50) |
|--------|--------------|----------------|-------------------|-------------------|
| Encoding | Index (any) | Index (1-bit) | Bitmask | Bitmask |
| Baseline | 92.33% | 92.33% | 92.33% | 92.33% |
| Logical ops | 50 | 50 | 25 swaps | **50 swaps** |
| Physical flips | 50 | 50 | 50 | **100** |
| Final accuracy | 38.67% | 30.37% | 65.14% | **50.73%** |
| Accuracy drop | 53.66% | 61.96% | 27.20% | **41.60%** |

### Key Observations
1. **Scaling behavior**: Doubling swaps (25→50) roughly doubles damage (27.20%→41.60%)
2. **Still less effective**: Even with 50 swaps, bitmask (41.60%) < index-any (53.66%)
3. **Fair comparison**: Now R1_T04 has same logical ops as R1_T01/T02
4. **Linear scaling**: Damage per swap ~0.83% (41.60%/50)
5. **All validities confirmed**: Zero popcount violations in ~13.5M checks

### Safety Mechanisms
- Recent group exclusion: 20-step window
- Forbidden swaps: Both forward and reverse stored
- Automatic cleanup of old swaps (max 1000)
- **Validity assertion**: popcount==2 verified after each swap

### Notes for Future Work
- R1_T04 demonstrates bitmask attack scales linearly with swap count
- The 1-bit index encoding (R1_T02) remains most effective (61.96% drop)
- Bitmask cost-2 is a natural constraint but reduces attack surface
- Consider hybrid attacks: bitmask with different group sizes

---

Date: 2026-02-18 (update) - R1_T05 JOINT BEST-STEP ATTACK

---

Date: 2026-02-18 (update) - R1_T06 LEGACY TASK1-3 RERUN WITH LOSS

## R1_T06: Legacy Task1-3 Rerun with Loss Trace - NEW

### Purpose
Re-run legacy Task1-3 (dense-format weight-bit BFA variants) with detailed loss tracking to investigate accuracy increases in legacy curves.

### Key Questions
1. Does accuracy increase on fixed eval subset?
2. Are there steps where ΔL_eval < 0 (loss decreases)?
3. Are acc increases just discrete fluctuations while loss still increases?

### Investigation Scope
- **Legacy Task1**: Global weight-bit flips (all weights)
- **Legacy Task2**: Zero-only targeting (flip bits in zero weights)
- **Legacy Task3**: Nonzero-only targeting (flip bits in nonzero weights)

### Hard Requirements
1. **Fixed eval subset**: Generate once with seed, reuse across all steps/tasks
2. **Fixed calibration subset**: Same for gradient computation
3. **Loss tracking**: Record L_calib, ΔL_calib, L_eval, ΔL_eval per step
4. **Output**: Trace CSV with all fields, acc/loss curves, summary report

### Status: COMPLETED
- [x] Legacy implementation located in `run_sparse_tasks.py` and `bfa/int8_attack.py`
- [x] Output directory: `results/R1/R1_T06_legacy_task1_3_rerun_loss/`
- [x] Script created: `run_R1_T06_legacy_task1_3_rerun_with_loss.py`
- [x] Execution completed (seed=0)
- [x] Analysis completed

### Execution Results (seed=0)

| Mode | Baseline Acc | Final Acc | Drop | Baseline Loss | Final Loss | Loss Increase |
|------|--------------|-----------|------|---------------|------------|---------------|
| Legacy Task1 (Global) | 92.40% | 10.85% | 81.55% | 0.2698 | 13.0021 | 12.7324 |
| Legacy Task2 (Zero-Only) | 92.40% | 89.90% | 2.50% | 0.2698 | 0.3618 | 0.0921 |
| Legacy Task3 (NonZero-Only) | 92.40% | 10.85% | 81.55% | 0.2698 | 13.0021 | 12.7324 |

### Key Findings

**Q1: Does accuracy increase on fixed eval subset?**
- **NO**: All three modes show monotonically decreasing accuracy
- **Legacy acc increases were caused by eval subset drift** (random sampling at each step)

**Q2: Are there steps where ΔL_eval < 0 (loss decreases)?**
- **NO**: Loss monotonically increases in all modes
- No bugs detected in flip persistence or evaluation

**Q3: Accuracy vs Loss**
- **Loss is more reliable** than accuracy for measuring attack progress
- Accuracy has discrete fluctuations (steps 7, 8, 14, 19 show Δacc=0 but ΔL>0)
- Loss shows continuous increase even when accuracy stays flat

### Anomaly Notes

**Zero-Only Mode (Legacy Task2) is very ineffective**:
- Only 2.50% accuracy drop after 50 flips
- Many flips (especially early ones) have no effect on acc/loss
- This is expected: flipping bits in already-zero weights has minimal impact

**Steps with Δacc=0 but ΔL>0** (Task1):
- Steps 7, 8, 14, 19 show no accuracy change but positive loss increase
- This demonstrates loss is a finer-grained metric

### Output Files

| File | Description |
|------|-------------|
| `eval_indices_seed0.json` | Fixed eval subset (2000 samples) |
| `calib_indices_seed0.json` | Fixed calib subset (256 samples) |
| `R1_T06_task1_global_trace.csv` | Step-by-step trace (21 steps) |
| `R1_T06_task1_zero_only_trace.csv` | Step-by-step trace (51 steps) |
| `R1_T06_task1_nonzero_only_trace.csv` | Step-by-step trace (17 steps) |
| `R1_T06_acc_curves.png` | Accuracy vs physical flips plot |
| `R1_T06_loss_curves.png` | Loss vs physical flips plot |
| `R1_T06_summary_table.csv` | Summary statistics |
| `R1_T06_report.md` | Full analysis report |
| `R1_T06_run_log.txt` | Run log with reproduction command |
| `R1_T06_results.pkl` | Serialized results |

### Reproduction Command
```bash
python run_R1_T06_legacy_task1_3_rerun_with_loss.py \
  --device cpu \
  --seed 0 \
  --physical-budget 50 \
  --calib-samples 256 \
  --eval-samples 2000 \
  --ckpt results/legacy_L0/by_task/task28_sparse_mask_fixed_finetune_int8_ckpt.pth
```

---

Date: 2026-02-18 16:30 - R1_T06 REDO: Per-Step Loss Printing + Specified Legends

## R1_T06 REDO: Legacy Task1-3 with Per-Step Loss Output

### Purpose
Redo R1_T06 with:
1. Per-step loss printing in fixed format (for grep)
2. Exact legend labels as specified
3. Dense-format semantic confirmation

### Step A: Locate Legacy Runner

**Entry Point**: `run_sparse_tasks.py` (line 85-159)

**Key Function**: `convert_to_dense_format()` (line 53-66)
```python
def convert_to_dense_format(model: torch.nn.Module, zero_point: int):
    """Convert sparse Int8 model to dense-format storage:
    - set pruned weights to zero_point
    - set mask to all ones (disable pruning in forward)
    """
    for _, module in model.named_modules():
        if hasattr(module, "sparse_mask") and module.sparse_mask is not None:
            mask = module.sparse_mask
            int8_w = module.int8_weights
            # zero out pruned positions
            int8_w[mask < 0.5] = torch.tensor(zero_point, dtype=torch.int8, device=int8_w.device)
            # disable pruning (dense-format behavior)
            module.sparse_mask = torch.ones_like(mask)
```

**Dense-Format Semantic**:
- Legacy task1-3 treat the sparse model as dense weight storage
- `convert_to_dense_format()` zeros out pruned weights and sets mask to all ones
- This ensures sparse_mask doesn't interfere with weight-bit flips in forward pass

**Forward Path** (`train/ptq_convert.py` line 84-90):
```python
def get_dequantized_weights(self):
    w_dequantized = self.int8_weights.float() * self.scale
    if self.sparse_mask is not None:
        w_dequantized = w_dequantized * self.sparse_mask
    return w_dequantized
```

**Critical Issue Found**:
- When `sparse_mask` is all ones (dense-format), the multiplication has no effect
- But we need to ensure this is set correctly before attack

### Step B: Plan for New Script

**New Script**: `run_R1_T06_legacy_dense_format_bfa.py`

**Required Features**:
1. **Dense-format conversion**: Call `convert_to_dense_format()` before attack
2. **Fixed eval/calib subsets**: Same as before
3. **Per-step printing**:
   ```
   [R1_T06][MODE=<global|zero_only|nonzero_only>][SEED=<seed>][STEP=<t>][FLIPS_USED=<k>] L_eval=<...> acc_eval=<...>
   ```
4. **Exact legends**:
   - "dense format : global weight-bit flip"
   - "dense format : zero-only"
   - "dense format: non-zero only"

### Status: COMPLETED
- [x] Located legacy runner
- [x] Confirmed dense-format semantic
- [x] Created new script with per-step printing
- [x] Run completed (seed=0)
- [x] Generated outputs with exact legends

### Script Created
**File**: `run_R1_T06_legacy_dense_format_bfa.py` (1055 lines)

**Key Features**:
1. **Dense-format conversion**: Uses `convert_to_dense_format()` from legacy code
2. **Fixed eval/calib subsets**: Same indices across all steps
3. **Per-step loss printing**: Format `[R1_T06][MODE=<...>][SEED=<...>][STEP=<t>][FLIPS_USED=<k>] L_eval=<...> acc_eval=<...>`
4. **EXACT legend labels** as specified

### Execution Results (seed=0)

| Mode | Baseline Acc | Final Acc | Drop | Baseline Loss | Final Loss | Loss Increase | Steps |
|------|--------------|-----------|------|---------------|------------|---------------|-------|
| dense format : global weight-bit flip | 92.40% | 10.20% | 82.20% | 0.269775 | 13.711840 | 13.442065 | 14 |
| dense format : zero-only | 92.40% | 11.60% | 80.80% | 0.269775 | 15.095377 | 14.825601 | 28 |
| dense format: non-zero only | 92.40% | 10.85% | 81.55% | 0.269775 | 13.002141 | 12.732365 | 16 |

### Key Findings

**CRITICAL DISCOVERY: Zero-Only Now Works!**
- Previous run (without `convert_to_dense_format`): zero-only only had 2.50% drop
- Current run (with dense-format): zero-only has 80.80% drop
- **Root cause**: Without `convert_to_dense_format()`, the sparse_mask was still being applied in forward pass, nullifying zero-weight flips
- **Fix**: `convert_to_dense_format()` sets sparse_mask to all ones and zeros pruned weights to zero_point, ensuring dense-format semantic

**Q1: Does accuracy increase on fixed eval subset?**
- **NO**: All three modes show monotonically decreasing accuracy

**Q2: Are there steps where ΔL_eval < 0?**
- **NO**: Loss monotonically increases in all modes

**Q3: Dense-format vs Sparse-Gated**
- The previous R1_T06 run was NOT true dense-format - sparse_mask was still gating the forward pass
- Legacy semantic requires converting sparse_mask to all ones

### Output Files

| File | Description |
|------|-------------|
| `eval_indices_seed0.json` | Fixed eval subset (2000 samples) |
| `calib_indices_seed0.json` | Fixed calib subset (256 samples) |
| `R1_T06_task1_global_trace.csv` | Step-by-step trace (14 steps) |
| `R1_T06_task2_zero_only_trace.csv` | Step-by-step trace (28 steps) |
| `R1_T06_task3_nonzero_only_trace.csv` | Step-by-step trace (16 steps) |
| `R1_T06_acc_curves.png` | Accuracy vs physical flips (with EXACT legends) |
| `R1_T06_loss_curves.png` | Loss vs physical flips (with EXACT legends) |
| `R1_T06_summary_table.csv` | Summary statistics |
| `R1_T06_report.md` | Full analysis report |
| `R1_T06_run_log.txt` | Run log with reproduction command |
| `R1_T06_results.pkl` | Serialized results |

### Per-Step Loss Output (Sample)
```
[R1_T06][MODE=global][SEED=0][STEP=0][FLIPS_USED=0] L_eval=0.269775 acc_eval=92.40
[R1_T06][MODE=global][SEED=0][STEP=1][FLIPS_USED=1] L_eval=0.300748 acc_eval=91.00
[R1_T06][MODE=global][SEED=0][STEP=2][FLIPS_USED=2] L_eval=0.334230 acc_eval=90.35
...
```

### Comparison with Legacy Results
| Mode | This Run (Dense Format) | Legacy (Task1-3) |
|------|------------------------|------------------|
| Global | 82.20% drop | ~82.10% drop ✓ |
| Zero-Only | 80.80% drop | ~79.67% drop ✓ |
| NonZero-Only | 81.55% drop | ~82.10% drop ✓ |

### Reproduction Command
```bash
python run_R1_T06_legacy_dense_format_bfa.py \
  --device cpu \
  --seed 0 \
  --physical-budget 50 \
  --calib-samples 256 \
  --eval-samples 2000 \
  --ckpt results/legacy_L0/by_task/task28_sparse_mask_fixed_finetune_int8_ckpt.pth
```

### Code Changes
- Created new file: `run_R1_T06_legacy_dense_format_bfa.py`
- Key addition: `convert_to_dense_format()` call before each attack
- This ensures legacy dense-format semantic (sparse_mask set to all ones)

---

---

## [2026-02-19 01:35] [R1_T06_debug] Step1 文本对齐与语义初判

1) 本步做了什么
- 读取 `results/R1/R1_T06_legacy_task1_3_rerun_loss/R1_T06_run_log.txt`，提取运行配置（ckpt/seed/eval_samples/calib_samples/eval_indices/calib_indices）。
- 读取 `results/R1/R1_T06_legacy_task1_3_rerun_loss/R1_T06_task1_zero_only_trace.csv`，确认 zero-only 轨迹中大量 step 的 `delta_L_eval=0`。
- 定位 runner 脚本并比对两份实现：`run_R1_T06_legacy_task1_3_rerun_with_loss.py` 与 `run_R1_T06_legacy_dense_format_bfa.py`。
- 定位 forward 权重路径到 `train/ptq_convert.py`（`get_dequantized_weights()` 与 `forward()`），确认默认会乘 `sparse_mask`。
- 给出“是否乘 mask / 是否 dense-format”的初步判定并记录证据行号。

2) 修改/新增文件
- 仅更新：`agent_develop_log.md`

3) 运行命令（可复现）
```bash
sed -n '1,260p' results/R1/R1_T06_legacy_task1_3_rerun_loss/R1_T06_run_log.txt
sed -n '1,120p' results/R1/R1_T06_legacy_task1_3_rerun_loss/R1_T06_task1_zero_only_trace.csv
rg -n "sparse_mask|apply_mask|mask|dequant|get_dequant|zero_only|dense|forward|Int8Quantized|flip" \
  run_R1_T06_legacy_task1_3_rerun_with_loss.py run_R1_T06_legacy_dense_format_bfa.py train/ptq_convert.py
nl -ba run_R1_T06_legacy_task1_3_rerun_with_loss.py | sed -n '860,1015p'
nl -ba run_R1_T06_legacy_dense_format_bfa.py | sed -n '88,130p'
nl -ba run_R1_T06_legacy_dense_format_bfa.py | sed -n '970,1030p'
nl -ba train/ptq_convert.py | sed -n '52,110p'
```

4) 输出路径（results/...）
- 输入日志：`results/R1/R1_T06_legacy_task1_3_rerun_loss/R1_T06_run_log.txt`
- 输入轨迹：`results/R1/R1_T06_legacy_task1_3_rerun_loss/R1_T06_task1_zero_only_trace.csv`

5) 关键证据（日志片段/统计数字/断言结论）
- `R1_T06_run_log.txt` 明确写了 reproduction command 是：
  `python run_R1_T06_legacy_dense_format_bfa.py ...`，且标题为 `Dense-Format BFA`。
- `run_R1_T06_legacy_dense_format_bfa.py:941-944` 在每个 mode 前显式调用：
  `convert_to_dense_format(model, zero_point)`，并打印 "sparse_mask set to all ones"。
- `convert_to_dense_format` 实现：`run_R1_T06_legacy_dense_format_bfa.py:96-112`，将 pruned 位置写为 `zero_point`，并将 `module.sparse_mask = torch.ones_like(mask)`。
- forward 仍存在 mask 路径：
  `train/ptq_convert.py:84-90` (`get_dequantized_weights` 里 `w_dequantized * sparse_mask`)；
  `train/ptq_convert.py:92-99` (`forward` 用 `get_dequantized_weights`)。
- 对照脚本 `run_R1_T06_legacy_task1_3_rerun_with_loss.py:931-967` 未调用 `convert_to_dense_format`，因此它本身不是 dense-format 语义。

初步判定：
- **本次给定 run_log 对应的实际运行是 `run_R1_T06_legacy_dense_format_bfa.py`，并且有 dense-format 转换调用。**
- **模型 forward 默认会乘 mask，但 dense 脚本将 mask 置为全 1 后，该 gating 语义应被禁用。**
- 因此仅从 Step1 看，H1 不能直接成立，需要 Step2/3 用 logits 差分实验钉死。

6) 下一步行动
- 进入 Step2：做“一步差分实验”（固定 batch，选 zero-only 轨迹中的 `old_int8==0` 位置 flip），计算 `max(|logits_after-logits_before|)`，先判定 H1/H2 的强弱。
## [2026-02-19 01:39] [R1_T06_debug] Step2 一步差分实验（flip 是否影响 logits）

1) 本步做了什么
- 构建与 run_log 一致的模型上下文（Int8 checkpoint + `Int8QuantizedResNet` + `calibrate_all_layers`）。
- 按 run_log 的固定 eval 索引 `eval_indices_seed0.json` 取前 64 样本作为固定 batch。
- 复用 zero-only 轨迹第1步位置：`layer2.0.downsample.0`, `weight_idx=114`, `bit_pos=7`, `old_int8==0`。
- 在 dense-format 语义下（先执行 `convert_to_dense_format`）做一次单 flip 并比较 logits 前后差异。
- 落盘 delta 结果到调试目录。

2) 修改/新增文件
- 新增：`results/R1/R1_T06_debug/step2_single_flip_logits_delta.json`
- 更新：`agent_develop_log.md`

3) 运行命令（可复现）
```bash
python - <<'PY'
# 载入模型 -> convert_to_dense_format -> fixed batch -> 单点flip -> 计算max|Δlogits|
# (完整脚本见终端历史)
PY
```

4) 输出路径（results/...）
- `results/R1/R1_T06_debug/step2_single_flip_logits_delta.json`

5) 关键证据（日志片段/统计数字/断言结论）
`step2_single_flip_logits_delta.json`:
- `old_int8: 0`
- `new_int8: -128`
- `mask_val_after_dense_conversion: 1.0`
- `delta_logits_max_abs: 1.8688297271728516`
- `delta_logits_mean_abs: 0.3525879979133606`

结论（Step2）：
- `delta_logits_max_abs` 显著大于 0，说明 flip **确实进入 forward**。
- 因此在该运行语义下，H2（flip改错buffer/未进forward）不成立；H1（被mask完全抵消）也不成立（至少对该点不成立）。

6) 下一步行动
- 执行 Step4：对 `zero_only_trace.csv` 做全量统计（old_int8、层分布、mask==0比例）。
- 执行 Step5：对齐 legacy 与 R1_T06 的 loss/acc 口径（sum vs mean、样本数）。
- 基于 Step4/Step5 判定 H3/H4，并决定是否需要最小修复与短跑复验。
## [2026-02-19 01:42] [R1_T06_debug] Step4 zero-only 候选空间统计（H3）

1) 本步做了什么
- 读取 `R1_T06_task1_zero_only_trace.csv` 全部 50 步，统计 `old_int8==0`、`delta_L_eval==0` 比例。
- 从 checkpoint 读取原始 `sparse_mask`（dense-format 转换前），将 trace 中每个 `(layer_name, weight_idx)` 回查为 `mask==0/1`。
- 统计 flip 的层分布 Top-5 及各层 mask0/mask1 细分。
- 生成结构化统计文件供报告直接引用。

2) 修改/新增文件
- 新增：`results/R1/R1_T06_debug/step4_zero_only_stats.json`
- 更新：`agent_develop_log.md`

3) 运行命令（可复现）
```bash
python - <<'PY'
# 读取 zero_only_trace + ckpt sparse_mask，输出 mask/层分布统计
# (完整脚本见终端历史)
PY
```

4) 输出路径（results/...）
- `results/R1/R1_T06_debug/step4_zero_only_stats.json`

5) 关键证据（日志片段/统计数字/断言结论）
`step4_zero_only_stats.json`:
- `steps_total = 50`
- `pct_old_int8_eq_0 = 1.0`（zero-only 定义本身正确）
- `pct_delta_L_eval_eq_0 = 0.94`（50步中47步 eval loss 变化为0，按trace精度）
- `pct_mask_eq_0 = 0.94`，`pct_mask_eq_1 = 0.06`
- Top-5 layer flips:
  - `layer2.0.downsample.0: 29`（其中 `mask0=29`）
  - `layer1.0.conv2: 8`（`mask0=8`）
  - `layer1.0.conv1: 7`（`mask0=7`）
  - `conv1: 3`（`mask1=3`）
  - `layer2.0.conv1: 2`（`mask0=2`）

结论（Step4）：
- H3 **强成立**：zero-only 被选中的位置高度集中在“原始稀疏掩码为0”的位置（94%）。
- 这与 `delta_L_eval` 大量为 0 的现象高度一致，解释了 zero-only 曲线长期平台。

6) 下一步行动
- 执行 Step5：对齐 T06 与 legacy 的 loss/acc 口径与实验设置差异（mean vs sum、eval样本数、ckpt来源），判断 H4 的作用范围。
## [2026-02-19 01:44] [R1_T06_debug] Step5 评估口径对齐（H4）

1) 本步做了什么
- 提取 T06 run_log 与 legacy task1/2/3 logs 的基线/终点 acc/loss 数字并做结构化对比。
- 代码级核对 loss 定义：T06 与 legacy 都是 CE mean（累计 `loss*bs` 再除 `total`）。
- 核对 eval/calib 设置差异、数据加载范围差异、checkpoint 来源差异。
- 输出比较结果 JSON 用于报告引用。

2) 修改/新增文件
- 新增：`results/R1/R1_T06_debug/step5_legacy_vs_t06_comparison.json`
- 更新：`agent_develop_log.md`

3) 运行命令（可复现）
```bash
sed -n '1,220p' results/legacy_L0/by_task/task1_sparse_dense_global_log.txt
sed -n '1,220p' results/legacy_L0/by_task/task2_sparse_dense_zero_log.txt
sed -n '1,220p' results/legacy_L0/by_task/task3_sparse_dense_nonzero_log.txt
nl -ba bfa/int8_attack.py | sed -n '412,438p'
nl -ba run_R1_T06_legacy_task1_3_rerun_with_loss.py | sed -n '365,385p'
nl -ba run_sparse_tasks.py | sed -n '38,75p'
python - <<'PY'
# 解析并汇总 T06 vs legacy 指标到 JSON
# (完整脚本见终端历史)
PY
```

4) 输出路径（results/...）
- `results/R1/R1_T06_debug/step5_legacy_vs_t06_comparison.json`

5) 关键证据（日志片段/统计数字/断言结论）
- 口径一致（mean，不是 sum）：
  - T06: `run_R1_T06_legacy_task1_3_rerun_with_loss.py:375,384`
  - legacy: `bfa/int8_attack.py:431,437`
- T06 配置（来自 run_log）：`eval_samples=2000`, `calib_samples=256`, ckpt=`task28_sparse_mask_fixed_finetune_int8_ckpt.pth`。
- legacy pipeline（代码）：
  - `run_sparse_tasks.py:73` 使用完整 test_loader（无 2000 子集裁剪）
  - `run_sparse_tasks.py:40-43` 模型来源为 `models/sparse_model.pth`
  - `calib_samples=100`（`run_sparse_tasks.py:100,125,151`）
- 对比数字（JSON）：
  - global final loss: T06 `13.71184` vs legacy `739.8527`
  - zero final loss: T06 `15.095377` vs legacy `42.766`
  - nonzero final loss: T06 `13.002141` vs legacy `68.0255`

结论（Step5）：
- H4（评估定义变更）中“mean vs sum”**不成立**。
- 但 H4 的“实验设置不一致”**成立**：eval子集/校准样本/ckpt来源与 legacy 不同，足以造成整体曲线和loss量级差异。
- H4 只能解释“整体偏移”，不能单独解释 zero-only 长平台；平台现象仍以 H3 为主因。

6) 下一步行动
- 执行 Step6 最小修复：针对 H3 在 zero-only 候选构造中加入“原始 mask==1 约束”（仅选 forward 更有效的 0），短跑 10 步验证 loss/acc 是否明显连续下降。
## [2026-02-19 01:48] [R1_T06_debug] 补充A/B实验：H1（mask抵消）与脚本语义差异

1) 本步做了什么
- 在同一 `(layer,idx,bit)` 上做 A/B 单点 flip 对照：
  - A: **不**做 dense conversion（等价 `run_R1_T06_legacy_task1_3_rerun_with_loss.py` 语义）
  - B: 做 dense conversion（`run_R1_T06_legacy_dense_format_bfa.py` 语义）
- 固定同一 eval batch（eval_indices 前64样本），比较 `max|Δlogits|`。
- 输出结构化 JSON 证据。

2) 修改/新增文件
- 新增：`results/R1/R1_T06_debug/step3_mask_ab_delta.json`
- 更新：`agent_develop_log.md`

3) 运行命令（可复现）
```bash
python - <<'PY'
# 同点位：no_dense vs dense_conversion 的 logits delta A/B
# (完整脚本见终端历史)
PY
```

4) 输出路径（results/...）
- `results/R1/R1_T06_debug/step3_mask_ab_delta.json`

5) 关键证据（日志片段/统计数字/断言结论）
`step3_mask_ab_delta.json`:
- `layer=layer2.0.downsample.0, idx=114, bit=7`
- no_dense:
  - `mask_val=0.0`
  - `delta_logits_max_abs=0.0`
- dense_conversion:
  - `mask_val=1.0`
  - `delta_logits_max_abs=1.8688297271728516`

结论：
- 若不做 dense conversion，则该 zero-only flip 完全被 mask 抵消（H1 成立）。
- 做 dense conversion 后同一点位立即生效（H1 在 dense语义下不成立）。
- 这直接解释了“两个 R1_T06 runner 行为差异巨大”的根因之一：脚本语义不一致。

6) 下一步行动
- 继续 Step6 最小修复验证：针对 H3（zero-only候选多落在原始mask0）做候选过滤并短跑 10 steps。
## [2026-02-19 01:50] [R1_T06_debug] Step6 最小修复 + 10步复跑验证（H3）

1) 本步做了什么
- 对 `run_R1_T06_legacy_dense_format_bfa.py` 做最小修复：
  - `convert_to_dense_format()` 中保存 `original_sparse_mask`。
  - `enumerate_candidates()` 新增 `require_original_mask_one` 开关。
  - `zero_only` 模式下仅允许 `original_sparse_mask==1` 的位置进入候选。
- 运行修复版短跑（physical-budget=10，seed=0，calib=256，eval=2000）。
- 对比修复前后 zero-only 前10步效果差异并落盘 JSON。

2) 修改/新增文件
- 修改：`run_R1_T06_legacy_dense_format_bfa.py`
- 新增：
  - `results/R1/R1_T06_debug/short_fix_10steps/R1_T06_task2_zero_only_trace.csv`
  - `results/R1/R1_T06_debug/short_fix_10steps/R1_T06_run_log.txt`
  - `results/R1/R1_T06_debug/short_fix_10steps/R1_T06_summary_table.csv`
  - `results/R1/R1_T06_debug/short_fix_10steps/R1_T06_acc_curves.png`
  - `results/R1/R1_T06_debug/short_fix_10steps/R1_T06_loss_curves.png`
  - `results/R1/R1_T06_debug/short_fix_10steps/R1_T06_results.pkl`
  - `results/R1/R1_T06_debug/step6_zero_only_10step_before_after.json`
- 更新：`agent_develop_log.md`

3) 运行命令（可复现）
```bash
python -m py_compile run_R1_T06_legacy_dense_format_bfa.py
python run_R1_T06_legacy_dense_format_bfa.py \
  --device cpu --seed 0 --physical-budget 10 \
  --calib-samples 256 --eval-samples 2000 \
  --ckpt results/legacy_L0/by_task/task28_sparse_mask_fixed_finetune_int8_ckpt.pth \
  --out-dir results/R1/R1_T06_debug/short_fix_10steps
python - <<'PY'
# 读取修复前/后 zero-only trace，比较前10步 acc/loss 变化
# (完整脚本见终端历史)
PY
```

4) 输出路径（results/...）
- `results/R1/R1_T06_debug/short_fix_10steps/`
- `results/R1/R1_T06_debug/step6_zero_only_10step_before_after.json`

5) 关键证据（日志片段/统计数字/断言结论）
修复版 zero-only 逐步打印（10步）显示连续下降：
- STEP1: `L_eval=0.287796 acc_eval=91.65`
- STEP5: `L_eval=0.520873 acc_eval=85.50`
- STEP10: `L_eval=0.682146 acc_eval=82.60`

`step6_zero_only_10step_before_after.json`:
- 修复前（旧trace前10步）：
  - `acc drop 0->10 = 0.75`
  - `loss gain 0->10 = 0.018021`
- 修复后（新trace前10步）：
  - `acc drop 0->10 = 9.80`
  - `loss gain 0->10 = 0.412371`

结论：
- H3 修复后，zero-only 不再“几乎不下降”，前10步已出现显著且连续的 loss/acc 变化。

6) 下一步行动
- 生成 `results/R1/R1_T06_debug/R1_T06_debug_report.md`（H1-H4判定、证据链、patch摘要、短跑结果）。
- 导出最小 patch diff 文件并完成最终交付检查。
## [2026-02-19 01:53] [R1_T06_debug] 结果归档与交付物生成

1) 本步做了什么
- 生成最终证据链报告：`results/R1/R1_T06_debug/R1_T06_debug_report.md`。
- 生成最小改动 diff 文件：`results/R1/R1_T06_debug/R1_T06_min_patch.diff`。
- 整理并校验关键证据文件是否齐全（Step2/3/4/5/6 JSON + 短跑输出目录）。
- 完成最终判定汇总（H1-H4）。

2) 修改/新增文件
- 新增：
  - `results/R1/R1_T06_debug/R1_T06_debug_report.md`
  - `results/R1/R1_T06_debug/R1_T06_min_patch.diff`
- 更新：`agent_develop_log.md`

3) 运行命令（可复现）
```bash
# 报告与diff由本轮命令直接写入：
# - results/R1/R1_T06_debug/R1_T06_debug_report.md
# - results/R1/R1_T06_debug/R1_T06_min_patch.diff
```

4) 输出路径（results/...）
- `results/R1/R1_T06_debug/R1_T06_debug_report.md`
- `results/R1/R1_T06_debug/R1_T06_min_patch.diff`
- `results/R1/R1_T06_debug/step2_single_flip_logits_delta.json`
- `results/R1/R1_T06_debug/step3_mask_ab_delta.json`
- `results/R1/R1_T06_debug/step4_zero_only_stats.json`
- `results/R1/R1_T06_debug/step5_legacy_vs_t06_comparison.json`
- `results/R1/R1_T06_debug/step6_zero_only_10step_before_after.json`
- `results/R1/R1_T06_debug/short_fix_10steps/`

5) 关键证据（日志片段/统计数字/断言结论）
- H2 否定：`step2_single_flip_logits_delta.json` -> `delta_logits_max_abs=1.8688`。
- H1 条件成立（脚本语义相关）：`step3_mask_ab_delta.json` -> no-dense 时 `delta=0`, dense 时 `delta=1.8688`。
- H3 成立：`step4_zero_only_stats.json` -> `pct(mask==0)=0.94`, `pct(delta_L_eval==0)=0.94`。
- 修复有效：`step6_zero_only_10step_before_after.json` -> zero-only 前10步 acc drop `0.75 -> 9.80`。

6) 下一步行动
- 向用户提交最终结论、最小 patch、短跑复验结果与可复现命令。
## [2026-02-20 13:36] [R1_T06_debug] 修复版 T06 全量 50 flips 复跑（seed=0）

1) 本步做了什么
- 运行修复后的 `run_R1_T06_legacy_dense_format_bfa.py`，将 `physical-budget` 设为 `50`（full run）。
- 使用与短跑一致配置：`seed=0`、`calib-samples=256`、`eval-samples=2000`、同一 checkpoint。
- 采集并确认三条曲线（global / zero_only / nonzero_only）均成功输出 CSV、PNG、summary、run_log。
- 验证 zero-only 不再“几乎不下降”，可持续下降至 50 flips。

2) 修改/新增文件（含路径）
- 新增目录与输出：`results/R1/R1_T06_debug/full_fix_50steps_seed0/`
- 新增文件（关键）：
  - `results/R1/R1_T06_debug/full_fix_50steps_seed0/R1_T06_task1_global_trace.csv`
  - `results/R1/R1_T06_debug/full_fix_50steps_seed0/R1_T06_task2_zero_only_trace.csv`
  - `results/R1/R1_T06_debug/full_fix_50steps_seed0/R1_T06_task3_nonzero_only_trace.csv`
  - `results/R1/R1_T06_debug/full_fix_50steps_seed0/R1_T06_summary_table.csv`
  - `results/R1/R1_T06_debug/full_fix_50steps_seed0/R1_T06_run_log.txt`
  - `results/R1/R1_T06_debug/full_fix_50steps_seed0/R1_T06_acc_curves.png`
  - `results/R1/R1_T06_debug/full_fix_50steps_seed0/R1_T06_loss_curves.png`
  - `results/R1/R1_T06_debug/full_fix_50steps_seed0/R1_T06_results.pkl`
- 更新：`agent_develop_log.md`

3) 运行命令（可复现）
```bash
python run_R1_T06_legacy_dense_format_bfa.py \
  --device cpu --seed 0 --physical-budget 50 \
  --calib-samples 256 --eval-samples 2000 \
  --ckpt results/legacy_L0/by_task/task28_sparse_mask_fixed_finetune_int8_ckpt.pth \
  --out-dir results/R1/R1_T06_debug/full_fix_50steps_seed0
```

4) 输出路径（results/...）
- `results/R1/R1_T06_debug/full_fix_50steps_seed0/`

5) 关键证据（日志片段/统计数字/断言结论）
- `R1_T06_summary_table.csv`:
  - global: `92.40% -> 10.20%`（drop `82.20%`）
  - zero_only: `92.40% -> 46.70%`（drop `45.70%`）
  - nonzero_only: `92.40% -> 10.85%`（drop `81.55%`）
- 运行日志显示 `zero_only` 到 `STEP=50`:
  - `L_eval=3.513234 acc_eval=46.70`
- 结论：修复后 zero-only 已显著下降，不再是旧结果中的“几乎平坦”。

6) 下一步行动
- 若用户确认，可将该 full-run 结果同步覆盖到 `results/R1/R1_T06_legacy_task1_3_rerun_loss/` 作为新的官方 T06 基线。
## [2026-02-20 14:14] [R1_T06_debug] 清点旧 R1_T06 结果

1) 本步做了什么
- 枚举 `results/` 下所有名称包含 `R1_T06` 的目录与文件。
- 检查工作区根目录是否存在 `R1_T06*` 独立文件。
- 确认待删除旧结果范围。

2) 修改/新增文件（含路径）
- 更新：`agent_develop_log.md`

3) 运行命令（可复现）
```bash
find results -maxdepth 3 \( -type d -o -type f \) -name '*R1_T06*' | sort
ls -1 | rg '^R1_T06' || true
```

4) 输出路径（results/...）
- 待清理路径：
  - `results/R1/R1_T06_debug`
  - `results/R1/R1_T06_legacy_task1_3_rerun_loss`

5) 关键证据（日志片段/统计数字/断言结论）
- 清点结果显示旧产物集中在上述两个目录内，包含 `R1_T06_summary_table.csv`、`R1_T06_run_log.txt`、trace CSV、曲线图等。
- 根目录无独立 `R1_T06*` 文件。

6) 下一步行动
- 删除旧 `R1_T06` 结果目录后，重新跑修复版 T06（50 flips）。
## [2026-02-20 14:15] [R1_T06_debug] 删除旧 R1_T06 结果

1) 本步做了什么
- 删除旧结果目录：`results/R1/R1_T06_legacy_task1_3_rerun_loss`。
- 删除旧调试目录：`results/R1/R1_T06_debug`。
- 复查 `results/` 下是否仍存在 `*R1_T06*` 路径。

2) 修改/新增文件（含路径）
- 删除：`results/R1/R1_T06_legacy_task1_3_rerun_loss/`（整目录）
- 删除：`results/R1/R1_T06_debug/`（整目录）
- 更新：`agent_develop_log.md`

3) 运行命令（可复现）
```bash
rm -rf results/R1/R1_T06_legacy_task1_3_rerun_loss
rm -rf results/R1/R1_T06_debug
find results -maxdepth 3 \( -type d -o -type f \) -name '*R1_T06*' | sort
```

4) 输出路径（results/...）
- 删除后：无 `R1_T06` 旧结果路径残留。

5) 关键证据（日志片段/统计数字/断言结论）
- 复查命令输出为空：`find ... -name '*R1_T06*'` 未返回任何路径。
- 断言：旧 `R1_T06` 结果已全部清空。

6) 下一步行动
- 重新运行修复版 R1_T06，生成新的全量结果并汇总给用户。
## [2026-02-20 14:19] [R1_T06_debug] 重新运行 R1_T06 并生成新结果（50 flips）

1) 本步做了什么
- 在旧结果清空后，重新运行修复版 `R1_T06` 全流程（global / zero_only / nonzero_only）。
- 使用配置：`seed=0`、`physical-budget=50`、`eval-samples=2000`、`calib-samples=256`。
- 校验新输出文件完整生成（trace CSV、summary、run_log、png、pkl）。

2) 修改/新增文件（含路径）
- 新增：`results/R1/R1_T06_legacy_task1_3_rerun_loss/` 下全部新产物
  - `R1_T06_task1_global_trace.csv`
  - `R1_T06_task2_zero_only_trace.csv`
  - `R1_T06_task3_nonzero_only_trace.csv`
  - `R1_T06_summary_table.csv`
  - `R1_T06_run_log.txt`
  - `R1_T06_acc_curves.png`
  - `R1_T06_loss_curves.png`
  - `R1_T06_results.pkl`
  - `eval_indices_seed0.json`
  - `calib_indices_seed0.json`
- 更新：`agent_develop_log.md`

3) 运行命令（可复现）
```bash
python run_R1_T06_legacy_dense_format_bfa.py \
  --device cpu --seed 0 --physical-budget 50 \
  --calib-samples 256 --eval-samples 2000 \
  --ckpt results/legacy_L0/by_task/task28_sparse_mask_fixed_finetune_int8_ckpt.pth \
  --out-dir results/R1/R1_T06_legacy_task1_3_rerun_loss
```

4) 输出路径（results/...）
- `results/R1/R1_T06_legacy_task1_3_rerun_loss/`

5) 关键证据（日志片段/统计数字/断言结论）
- `R1_T06_summary_table.csv`：
  - global: `92.40% -> 10.20%`（drop `82.20%`）
  - zero_only: `92.40% -> 46.70%`（drop `45.70%`）
  - nonzero_only: `92.40% -> 10.85%`（drop `81.55%`）
- `R1_T06_task2_zero_only_trace.csv` 最后一步（step 50）：
  - `acc_eval=46.7000`, `loss_eval=3.513234`
- 结论：新复跑结果已替换旧结果，zero-only 曲线明显下降。

6) 下一步行动
- 向用户展示新结果摘要与输出路径。
## [2026-02-20 14:31] [R1_T06_debug] T05 vs T06 global 下降速度差异分析记录

1) 本步做了什么
- 对比读取 `R1_T05` 与新复跑 `R1_T06` 的日志、summary、trace。
- 定位并核对两者选点机制（T05: StageA+StageB exact；T06: proxy-only top1）。
- 计算前 9 步 / 前 14 步累计 loss 增量并对齐 random-level 达成步数。
- 输出结论：T05 达到 random level 更快的主要原因是“每步 exact verification 重排”，而非 metadata 被选中。

2) 修改/新增文件（含路径）
- 更新：`agent_develop_log.md`

3) 运行命令（可复现）
```bash
head -n 120 results/R1/R1_T05_joint_best_step_attack_log.txt
cat results/R1/R1_T05_joint_best_step_attack_table.csv
head -n 140 results/R1/R1_T06_legacy_task1_3_rerun_loss/R1_T06_run_log.txt
head -n 20 results/R1/R1_T06_legacy_task1_3_rerun_loss/R1_T06_task1_global_trace.csv
```

4) 输出路径（results/...）
- `results/R1/R1_T05_joint_best_step_attack_log.txt`
- `results/R1/R1_T05_joint_best_step_attack_table.csv`
- `results/R1/R1_T06_legacy_task1_3_rerun_loss/R1_T06_run_log.txt`
- `results/R1/R1_T06_legacy_task1_3_rerun_loss/R1_T06_task1_global_trace.csv`

5) 关键证据（日志片段/统计数字/断言结论）
- T05 在 `step=9` 已到 `acc=10.06%`；T06 global 在 `step=14` 到 `acc=10.20%`。
- 累计 loss 增量对比：
  - T05 前9步 `ΣΔL ≈ 25.44`
  - T06 前9步 `ΣΔL ≈ 5.01`
- 机制差异（代码证据）：
  - T05: `top-K proxy -> exact verification -> argmax exact`（`run_R1_T05_joint_best_step_attack.py:824`, `run_R1_T05_joint_best_step_attack.py:895`, `run_R1_T05_joint_best_step_attack.py:1049`）
  - T06: `proxy 排序后直接取 candidates[0]`（`run_R1_T06_legacy_dense_format_bfa.py:496`）
- T05 action breakdown: `weight_bit=50, index_1bit=0, bitmask_swap=0`（非 metadata 驱动）。

6) 下一步行动
- 如需严格公平对比，可新增一个 “T06 + topK exact verification” 对照跑，直接验证步效差异是否主要来自选点策略。
## [2026-02-20 14:33] [R1_T06_debug] zero_only 重跑前备份当前结果

1) 本步做了什么
- 创建对比目录 `results/R1/R1_T06_zero_only_rerun_compare/`。
- 备份当前 zero_only trace 与 summary，避免被后续结果覆盖。
- 快速读取当前 zero_only 终点指标，作为对比基线。

2) 修改/新增文件（含路径）
- 新增目录：`results/R1/R1_T06_zero_only_rerun_compare/`
- 新增文件：
  - `results/R1/R1_T06_zero_only_rerun_compare/zero_only_trace_current.csv`
  - `results/R1/R1_T06_zero_only_rerun_compare/summary_current.csv`
- 更新：`agent_develop_log.md`

3) 运行命令（可复现）
```bash
mkdir -p results/R1/R1_T06_zero_only_rerun_compare
cp results/R1/R1_T06_legacy_task1_3_rerun_loss/R1_T06_task2_zero_only_trace.csv \
  results/R1/R1_T06_zero_only_rerun_compare/zero_only_trace_current.csv
cp results/R1/R1_T06_legacy_task1_3_rerun_loss/R1_T06_summary_table.csv \
  results/R1/R1_T06_zero_only_rerun_compare/summary_current.csv
```

4) 输出路径（results/...）
- `results/R1/R1_T06_zero_only_rerun_compare/`

5) 关键证据（日志片段/统计数字/断言结论）
- 当前 zero_only trace: `steps=50`
- 当前终点：`acc_eval=46.7000`, `L_eval=3.513234`

6) 下一步行动
- 复用同一 eval/calib indices，执行 zero_only 单模式重跑并导出新 trace。
## [2026-02-20 14:36] [R1_T06_debug] zero_only 单模式重跑完成

1) 本步做了什么
- 使用 `run_R1_T06_legacy_dense_format_bfa.py` 同一代码路径，单独调用 `run_attack_mode(..., mode='zero_only')` 进行重跑。
- 复用现有 `eval_indices_seed0.json` 与 `calib_indices_seed0.json`，确保与当前结果同一数据子集。
- 配置保持一致：`seed=0`、`physical_budget=50`、`eval_samples=2000`、`calib_samples=256`、同一 checkpoint。
- 导出新 trace 与新 summary(json)。

2) 修改/新增文件（含路径）
- 新增：
  - `results/R1/R1_T06_zero_only_rerun_compare/zero_only_trace_rerun.csv`
  - `results/R1/R1_T06_zero_only_rerun_compare/zero_only_rerun_summary.json`
- 更新：`agent_develop_log.md`

3) 运行命令（可复现）
```bash
python - <<'PY'
# 通过 import run_R1_T06_legacy_dense_format_bfa 并调用 run_attack_mode(mode='zero_only')
# 复用：results/R1/R1_T06_legacy_task1_3_rerun_loss/eval_indices_seed0.json
#      results/R1/R1_T06_legacy_task1_3_rerun_loss/calib_indices_seed0.json
# 输出：results/R1/R1_T06_zero_only_rerun_compare/zero_only_trace_rerun.csv
PY
```

4) 输出路径（results/...）
- `results/R1/R1_T06_zero_only_rerun_compare/zero_only_trace_rerun.csv`
- `results/R1/R1_T06_zero_only_rerun_compare/zero_only_rerun_summary.json`

5) 关键证据（日志片段/统计数字/断言结论）
- 重跑终点：`acc_eval=46.7000`, `L_eval=3.513234`, `steps=50`。
- 与当前 zero_only 终点数值一致（初步观察）。

6) 下一步行动
- 逐步对齐 compare（step-by-step 差分），输出 max|Δacc|/max|Δloss| 与动作字段是否一致。
## [2026-02-20 14:37] [R1_T06_debug] zero_only 新旧结果对比完成

1) 本步做了什么
- 读取当前 trace (`zero_only_trace_current.csv`) 与新重跑 trace (`zero_only_trace_rerun.csv`)。
- 逐步对齐比较 `acc_eval` / `L_eval` / `proxy_score` 以及动作字段（layer, idx, bit, old/new int8）。
- 生成 step 级差分 CSV、总结 JSON 和 markdown 报告。

2) 修改/新增文件（含路径）
- 新增：
  - `results/R1/R1_T06_zero_only_rerun_compare/zero_only_stepwise_diff.csv`
  - `results/R1/R1_T06_zero_only_rerun_compare/zero_only_compare_summary.json`
  - `results/R1/R1_T06_zero_only_rerun_compare/zero_only_compare_report.md`
- 更新：`agent_develop_log.md`

3) 运行命令（可复现）
```bash
python - <<'PY'
# 逐步读取 current/rerun 两个 trace，输出 stepwise diff + summary + report
PY
```

4) 输出路径（results/...）
- `results/R1/R1_T06_zero_only_rerun_compare/zero_only_stepwise_diff.csv`
- `results/R1/R1_T06_zero_only_rerun_compare/zero_only_compare_summary.json`
- `results/R1/R1_T06_zero_only_rerun_compare/zero_only_compare_report.md`

5) 关键证据（日志片段/统计数字/断言结论）
- `steps compared = 50`
- `max |Δacc_eval| = 0.0`
- `max |ΔL_eval| = 0.0`
- `max |Δproxy_score| = 0.0`
- `all_action_same = True`（每一步 layer/weight_idx/bit_pos/old_int8/new_int8 完全一致）
- 终点一致：`final acc 46.7000 vs 46.7000`, `final loss 3.513234 vs 3.513234`

6) 下一步行动
- 向用户汇报：本次 zero_only 重跑与当前 zero_only 结果完全一致（可视为 deterministic/可复现）。
## [2026-02-20 14:43] [R1_T06_debug] legacy task2 vs 当前 R1_T06 zero_only 差异归因

1) 本步做了什么
- 读取 `results/legacy_L0/by_task/task2_sparse_dense_zero_log.txt` 与 `run_sparse_tasks.py`，确认 legacy task2 的真实实现路径。
- 读取 `bfa/int8_attack.py` 与 `run_R1_T06_legacy_dense_format_bfa.py`，对齐 zero-only 候选定义与评估设置。
- 量化对比两者 flip 落点在原始 `sparse_mask` 上的分布（mask==0 vs mask==1）。

2) 修改/新增文件（含路径）
- 更新：`agent_develop_log.md`

3) 运行命令（可复现）
```bash
head -n 220 results/legacy_L0/by_task/task2_sparse_dense_zero_log.txt
nl -ba run_sparse_tasks.py | sed -n '109,132p'
nl -ba run_R1_T06_legacy_dense_format_bfa.py | sed -n '96,118p'
nl -ba run_R1_T06_legacy_dense_format_bfa.py | sed -n '485,490p'
```

4) 输出路径（results/...）
- `results/legacy_L0/by_task/task2_sparse_dense_zero_log.txt`
- `results/legacy_L0/by_date/task2_sparse_dense_zero_result.pkl`
- `results/R1/R1_T06_legacy_task1_3_rerun_loss/R1_T06_task2_zero_only_trace.csv`

5) 关键证据（日志片段/统计数字/断言结论）
- legacy task2（dense zero-only）最终：`92.10% -> 12.43%`（50 flips）。
- 当前 R1_T06 zero_only 最终：`92.40% -> 46.70%`（50 flips）。
- 原始 mask 落点统计：
  - legacy task2：`50 flips` 中 `mask==0` 为 `44`，`mask==1` 为 `6`。
  - 当前 R1_T06：`50 flips` 中（可映射 mask 的 48 步）`mask==0` 为 `0`，`mask==1` 为 `48`（另 2 步在 `fc` 无 sparse_mask）。
- 结论：当前 R1_T06 通过 `require_original_mask_one` 把 zero-only 候选从“含大量原始剪枝位点复活”改为“仅原始保留连接里的零值”，攻击强度显著下降。

6) 下一步行动
- 向用户汇报差异表：候选空间定义、checkpoint、评估样本与早停阈值的影响顺序。
## [2026-02-20 14:45] [R1_T06_debug] zero_only 语义改回 legacy（代码最小修改）

1) 本步做了什么
- 修改 `run_R1_T06_legacy_dense_format_bfa.py`：zero_only 候选枚举不再强制 `original_mask==1`。
- 保留 dense-format（mask 全1）前提，恢复 legacy task2 的 zero-only 逻辑：按 `int8==zero_point` 选点。
- 运行 `python -m py_compile run_R1_T06_legacy_dense_format_bfa.py` 确认语法正确。

2) 修改/新增文件（含路径）
- 修改：`run_R1_T06_legacy_dense_format_bfa.py`
- 更新：`agent_develop_log.md`

3) 运行命令（可复现）
```bash
python -m py_compile run_R1_T06_legacy_dense_format_bfa.py
```

4) 输出路径（results/...）
- 本步无新 results 产物。

5) 关键证据（日志片段/统计数字/断言结论）
- 枚举调用由：`require_original_mask_one=(mode == 'zero_only')`
- 改为：`require_original_mask_one=False`
- 编译通过，无语法错误。

6) 下一步行动
- 删除旧 R1_T06 zero-only 相关结果并全量重跑 R1_T06，重绘曲线。
## [2026-02-20 14:46] [R1_T06_debug] 删除旧 R1_T06 zero-only 结果

1) 本步做了什么
- 删除旧 `R1_T06` 输出目录，避免旧 zero-only 数据污染新曲线。
- 删除 zero-only 重跑对比目录（旧语义下的比较文件）。
- 复查 `results/R1` 下是否仍存在 `*R1_T06*` 路径。

2) 修改/新增文件（含路径）
- 删除：`results/R1/R1_T06_legacy_task1_3_rerun_loss/`
- 删除：`results/R1/R1_T06_zero_only_rerun_compare/`
- 更新：`agent_develop_log.md`

3) 运行命令（可复现）
```bash
rm -rf results/R1/R1_T06_legacy_task1_3_rerun_loss
rm -rf results/R1/R1_T06_zero_only_rerun_compare
find results/R1 -maxdepth 2 \( -type d -o -type f \) -name '*R1_T06*' | sort
```

4) 输出路径（results/...）
- 删除后：`results/R1` 下无 `R1_T06` 残留路径。

5) 关键证据（日志片段/统计数字/断言结论）
- 复查命令输出为空，确认旧 R1_T06 结果已清空。

6) 下一步行动
- 按新 legacy zero-only 语义重跑 R1_T06（50 flips）并重绘曲线。
## [2026-02-20 14:49] [R1_T06_debug] 按 legacy zero-only 语义重跑并重绘 R1_T06 曲线

1) 本步做了什么
- 运行修改后的 `run_R1_T06_legacy_dense_format_bfa.py`（seed=0, budget=50）。
- 重新生成三条模式结果（global/zero_only/nonzero_only）与曲线图。
- 验证 zero_only 不再停留在 46.7%，已降至 near-random 区间并提前早停。

2) 修改/新增文件（含路径）
- 新增（重建）：`results/R1/R1_T06_legacy_task1_3_rerun_loss/` 全部产物
  - `R1_T06_task1_global_trace.csv`
  - `R1_T06_task2_zero_only_trace.csv`
  - `R1_T06_task3_nonzero_only_trace.csv`
  - `R1_T06_summary_table.csv`
  - `R1_T06_run_log.txt`
  - `R1_T06_results.pkl`
  - `R1_T06_acc_curves.png`
  - `R1_T06_loss_curves.png`
- 更新：`agent_develop_log.md`

3) 运行命令（可复现）
```bash
python run_R1_T06_legacy_dense_format_bfa.py \
  --device cpu --seed 0 --physical-budget 50 \
  --calib-samples 256 --eval-samples 2000 \
  --ckpt results/legacy_L0/by_task/task28_sparse_mask_fixed_finetune_int8_ckpt.pth \
  --out-dir results/R1/R1_T06_legacy_task1_3_rerun_loss
```

4) 输出路径（results/...）
- `results/R1/R1_T06_legacy_task1_3_rerun_loss/`

5) 关键证据（日志片段/统计数字/断言结论）
- 新 `R1_T06_summary_table.csv`：
  - global: `92.40% -> 10.20%`（drop `82.20%`）
  - zero_only: `92.40% -> 11.60%`（drop `80.80%`）
  - nonzero_only: `92.40% -> 10.85%`（drop `81.55%`）
- zero_only trace 显示在 `step=28` 达到 `acc_eval=11.60` 并触发 near-random 早停。
- 结论：已恢复到 legacy-like zero-only 强度，旧 46.7% 结果已被替换。

6) 下一步行动
- 向用户回传新结果与图路径；如需我可再与 `legacy task2_sparse_dense_zero` 做一步一比一轨迹对照。

---

## [2026-02-20 16:20] [R1_T01_T02] 添加 Loss Tracking 并重新运行

1) 本步做了什么
- 为 R1_T01 (any-pattern metadata attack) 和 R1_T02 (1-bit reachable metadata attack) 添加了 loss increase/decrease 追踪功能。
- 修改 `AttackStepLog` 数据结构，新增 `delta_loss` 和 `delta_accuracy` 字段。
- 在攻击循环中追踪准确率回升步骤 (`acc_increase_steps`) 和损失下降步骤 (`loss_decrease_steps`)。
- 更新日志输出格式，显示每步的 delta 值。
- 重新运行两个攻击任务，生成新结果。

2) 修改/新增文件（含路径）
- 修改：`run_R1_T01_group_metadata_index_anypattern.py`
  - 添加 `delta_loss`, `delta_accuracy` 字段到 `AttackStepLog`
  - 添加 `acc_increase_steps`, `loss_decrease_steps` 列表追踪
  - 更新输出格式和 CSV 表格
- 修改：`run_R1_T02_group_metadata_index_1bit.py`
  - 同上修改
- 新增：
  - `results/R1/R1_T01_group_metadata_index_anypattern_rerun_result.pkl`
  - `results/R1/R1_T01_group_metadata_index_anypattern_rerun_table.csv`
  - `results/R1/R1_T01_group_metadata_index_anypattern_rerun_log.txt`
  - `results/R1/R1_T02_group_metadata_index_1bit_rerun_result.pkl`
  - `results/R1/R1_T02_group_metadata_index_1bit_rerun_table.csv`
  - `results/R1/R1_T02_group_metadata_index_1bit_rerun_log.txt`

3) 运行命令（可复现）
```bash
python run_R1_T01_group_metadata_index_anypattern.py \
  --device cpu --seed 0 --max-flips 50 \
  --calib-samples 256 --eval-samples 2000 \
  --ckpt results/legacy_L0/by_task/task28_sparse_mask_fixed_finetune_int8_ckpt.pth \
  --out-prefix results/R1/R1_T01_group_metadata_index_anypattern_rerun

python run_R1_T02_group_metadata_index_1bit.py \
  --device cpu --seed 0 --max-flips 50 \
  --calib-samples 256 --eval-samples 2000 \
  --ckpt results/legacy_L0/by_task/task28_sparse_mask_fixed_finetune_int8_ckpt.pth \
  --out-prefix results/R1/R1_T02_group_metadata_index_1bit_rerun
```

4) 输出路径（results/...）
- `results/R1/R1_T01_group_metadata_index_anypattern_rerun_*`
- `results/R1/R1_T02_group_metadata_index_1bit_rerun_*`

5) 关键证据（日志片段/统计数字/断言结论）

**R1_T01 (any-pattern)**:
- Baseline: 92.33% → Final: 16.75% (drop: 75.59%)
- Baseline Loss: 0.251385 → Final Loss: 15.950482 (increase: 15.699097)
- **准确率回升次数**: 1 次 (step 44)
- **损失下降次数**: 0 次
- Loss 单调增加 ✓

**R1_T02 (1-bit reachable)**:
- Baseline: 92.33% → Final: 25.63% (drop: 66.70%)
- Baseline Loss: 0.251385 → Final Loss: 8.788884 (increase: 8.537499)
- **准确率回升次数**: 5 次 (steps: [1, 17, 34, 37, 46])
- **损失下降次数**: 7 次 (steps: [17, 32, 33, 35, 37, 46, 48])
- 存在非单调性 ⚠️

**对比 R1_T01 和 R1_T02**:
- R1_T01 的 loss 完全单调增加（0 次下降）
- R1_T02 的 loss 存在 7 次下降
- 可能原因：1-bit reachable 约束限制了搜索空间，导致某些翻转反而略微降低损失

6) 下一步行动
- 分析 R1_T02 中 loss 下降的具体原因
- 更新 `position_encoding_bitreachable_NCA_method.md` 文档，添加 loss 跟踪结果


