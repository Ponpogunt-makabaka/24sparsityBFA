# R1_T05 vs R1_T06 Global: Equivalence Conditions

## 结论先行
- 只有在“同 threat model + 同候选空间 + 同选择器 + 同数据口径 + 同停止规则”同时满足时，T05 与 T06 global 才应接近一致。
- 当前实现下，这些条件大部分不满足，因此“不能直接要求两者曲线一致”。

## 严格等价条件清单

| 条件ID | 何时应接近一致 | 当前状态 | 证据 |
|---|---|---|---|
| EQ1 | 同 baseline checkpoint | `满足` | T05 日志与 T06.1 运行都使用 task28 int8 sparse ckpt（`results/R1/R1_T05_joint_best_step_attack_log.txt`，`results/R1/R1_T06_1_sparse_gated_dense_view_exact/R1_T06_1_run_log.txt`） |
| EQ2 | 同攻击面（都只允许 weight-bit） | `不满足` | T05 默认 joint 三动作（`run_R1_T05_joint_best_step_attack.py:1208`），T06 global 是 weight-only 模式 |
| EQ3 | 同 weight 候选池定义 | `不满足` | T05 只枚举活跃非零权重（`run_R1_T05_joint_best_step_attack.py:405`, `run_R1_T05_joint_best_step_attack.py:409`）；T06 dense-view 枚举全部 int8 位点（`run_R1_T06_1_sparse_gated_dense_view_exact.py:226`） |
| EQ4 | 同 Stage-A 候选裁剪策略 | `不满足` | T05 有每层上限与跨 action 合并 top-K（`run_R1_T05_joint_best_step_attack.py:856`, `run_R1_T05_joint_best_step_attack.py:879`）；T06 先全排序再截 top-K（`run_R1_T06_1_sparse_gated_dense_view_exact.py:264`, `run_R1_T06_1_sparse_gated_dense_view_exact.py:307`） |
| EQ5 | 同 Stage-B exact 验证口径 | `不满足` | T05 Stage-B 用 `calib_loader` 且样本数参数是 `eval_samples`（`run_R1_T05_joint_best_step_attack.py:900`, `run_R1_T05_joint_best_step_attack.py:923`）；T06 Stage-B 用 `calib_samples`（`run_R1_T06_1_sparse_gated_dense_view_exact.py:422`, `run_R1_T06_1_sparse_gated_dense_view_exact.py:426`） |
| EQ6 | 同数据拆分（calib/eval） | `不满足` | T05: `calib_loader=test_loader`（`run_R1_T05_joint_best_step_attack.py:1310`）；T06: 固定索引分开采样（`run_R1_T06_1_sparse_gated_dense_view_exact.py:807`, `run_R1_T06_1_sparse_gated_dense_view_exact.py:812`） |
| EQ7 | 同状态约束（forbidden/exclude） | `不满足` | T05 同时维护 flipped_bits+metadata 禁忌集（`run_R1_T05_joint_best_step_attack.py:1003`~`run_R1_T05_joint_best_step_attack.py:1006`）；T06 仅跟踪 flipped_bits（`run_R1_T06_1_sparse_gated_dense_view_exact.py:253`, `run_R1_T06_1_sparse_gated_dense_view_exact.py:293`） |
| EQ8 | 同停止规则（跑满预算或同阈值停） | `不满足` | T06 在 `acc<12%` 提前停（`run_R1_T06_1_sparse_gated_dense_view_exact.py:503`）；T05 默认跑满预算 |
| EQ9 | 同物理预算计费规则 | `部分满足` | weight 动作均 cost=1；但 T05 允许 cost=2 bitmask（joint 模式） |
| EQ10 | 同模式下的同类对照（T05 weight_only vs T06 nonzero/global） | `当前未执行` | 需按 `R1_Minimal_Verification_Experiments_Table.csv` 的 E03/E08 追加短跑验证 |

## 现有数据能说明什么

- T05 当前结果：`92.33% -> 9.96%`，50/50 全部选择 `weight_bit`（`results/R1/R1_T05_joint_best_step_attack_table.csv`）。
- T05 的 metadata 候选不是 0：Step1 候选量 index=50047、bitmask=50047；但 top-K 为 `63/1/0`，且 50 步均未选择 bitmask。
- T06.1（sparse-gated dense-view exact）global：约 6 步到 near-random，zero_only 仅小幅下降（`92.40% -> 89.90%`），说明 mask 门控语义在工作（`results/R1/R1_T06_1_sparse_gated_dense_view_exact/R1_T06_1_summary_table.csv`）。

## 何时“应接近一致”
仅当你执行如下对齐实验时，才可合理期待 T05 与 T06 global 接近：
- T05 使用 `weight_only` 模式。
- T05 与 T06 使用同一 eval/calib indices、同一 exact batch size。
- 两边统一停止规则（都跑满 50，或都用同阈值 early-stop）。
- 两边统一 Stage-A 候选裁剪（是否 dense-view 全量、是否每层 cap）。

