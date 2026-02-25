# R1 Logical Consistency Criteria

## 0. 目标定义
本标准用于判断 R1_T01-R1_T05 的实现是否“logical”，并以 `R1_T07`（sampleBFA 风格、传统 value-only BFA）作为参照框架。

- 参照点（T07）关键语义：value-only、逐层 top-k 代理筛选、单点 exact 验证后应用、保留 sparse-gated forward，不改 `sparse_mask`（`run_R1_T07_samplebfa_style_dense_bfa.py:172`, `run_R1_T07_samplebfa_style_dense_bfa.py:240`, `run_R1_T07_samplebfa_style_dense_bfa.py:243`）。
- R1_T01-T05 是 metadata 或 joint threat model，不要求和 T07 曲线一致，但要求“命名-语义-实现”一致。

## 1. 一致性判据（必须先定义再评估）

| ID | 维度 | Pass 标准 | Fail 信号 |
|---|---|---|---|
| C1 | Threat model 一致性 | 任务名与攻击面一致：`weight` / `index` / `bitmask` / `joint`；cost 定义一致并进入选择目标 | 任务名写 metadata，但实际只在改权重；或 cost 口径前后不一致 |
| C2 | Forward 语义一致性 | sparse-gated 任务中 `sparse_mask` 保持生效；dense-view 仅用于候选搜索，不应强制 densify 结构 | `sparse_mask` 被覆盖为全 1，或 flip 写到非 forward tensor |
| C3 | Candidate space 一致性 | `global/zero_only/nonzero_only/joint` 的过滤规则与命名一致，且候选非空原因可解释 | `zero_only` 选到 non-zero；候选长期为 0 且无解释 |
| C4 | Selection policy 一致性 | 明确是 `proxy-only` 还是 `topK+exact`；Stage A/Stage B 的 score 与 cost 使用一致 | 标称 exact 但实际未做 apply-eval-revert；或 score/cost 在两阶段口径不一致 |
| C5 | Eval/Calib 数据一致性 | 报告明确 calib 与 eval 关系（同集/分集/重叠比例），并解释其对曲线的影响 | 默认为“泛化评估”但 calib/eval 实为同一批样本 |
| C6 | 指标解释一致性 | 允许 acc 局部回升；应结合 loss、exact/proxy 偏差、候选约束判断；不能把“非单调”直接判 bug | 仅凭一两个回升点直接断言实现错误 |

## 2. 每个维度的可执行检查

### C1 Threat model
- 检查 action 定义与命名是否对应：
  - T01/T02：index metadata（`run_R1_T02_group_metadata_index_1bit.py:304`）。
  - T03/T04：bitmask swap metadata（`run_R1_T03_group_metadata_bitmask_swap_cost2.py:329`, `run_R1_T04_bitmask_swaps50.py:469`）。
  - T05：joint 三类 action + cost（`run_R1_T05_joint_best_step_attack.py:441`, `run_R1_T05_joint_best_step_attack.py:550`, `run_R1_T05_joint_best_step_attack.py:660`, `run_R1_T05_joint_best_step_attack.py:662`）。

### C2 Forward semantics
- sparse-gated 语义应保持：
  - T06.1 已加硬检查：若 mask 全 1 则报错（`run_R1_T06_1_sparse_gated_dense_view_exact.py:840`）。
  - T06.1 候选搜索 dense-view 但不按 mask 过滤（`run_R1_T06_1_sparse_gated_dense_view_exact.py:226`）。

### C3 Candidate space
- 命名过滤一致性：
  - T06.1 `zero_only/nonzero_only` 用 `weight_filter` 明确定义（`run_R1_T06_1_sparse_gated_dense_view_exact.py:381`）。
  - T07 `value_allowed` 明确定义三模式（`run_R1_T07_samplebfa_style_dense_bfa.py:129`）。

### C4 Selection policy
- proxy-only：T02/T03/T04（`run_R1_T02_group_metadata_index_1bit.py:582`, `run_R1_T03_group_metadata_bitmask_swap_cost2.py:594`, `run_R1_T04_bitmask_swaps50.py:529`）。
- topK+exact：T01(可选)、T02.1、T03.1、T05、T06.1（`run_R1_T01_group_metadata_index_anypattern.py:635`, `run_R1_T02_1_group_metadata_index_1bit_exact.py:736`, `run_R1_T03_1_group_metadata_bitmask_swap_cost2_exact.py:760`, `run_R1_T05_joint_best_step_attack.py:1041`, `run_R1_T06_1_sparse_gated_dense_view_exact.py:422`）。

### C5 Eval/Calib
- T01-T05 当前实现：`calib_loader=test_loader`（如 `run_R1_T05_joint_best_step_attack.py:1310`），属于“同数据源切片”。
- T06.1：fixed indices 分开采样（`run_R1_T06_1_sparse_gated_dense_view_exact.py:807`, `run_R1_T06_1_sparse_gated_dense_view_exact.py:812`），但存在部分重叠（seed0 overlap=44/256）。
- T07：attack batch 来自 train，eval 来自 test（`run_R1_T07_samplebfa_style_dense_bfa.py:535`, `run_R1_T07_samplebfa_style_dense_bfa.py:545`）。

### C6 指标解释
- 允许局部回升：代理目标与评估目标不完全一致时会出现（尤其 proxy-only）。
- 仅当出现“持续大幅反向 + exact 也无法改善 + flip 无效证据”时才升级为实现问题。

## 3. 当前 R1 状态（基于已有结果的初判）

- T02 proxy-only 出现回升：`n_acc_increase_steps=5`, `n_loss_decrease_steps=7`（`results/R1/R1_T02_group_metadata_index_1bit_rerun_table.csv`）。
- T02.1 exact 后明显收敛：`n_acc_increase_steps=1`, `n_loss_decrease_steps=0`（`results/R1/R1_T02_1_group_metadata_index_1bit_exact_table.csv`）。
- T03 proxy-only 也有回升（从日志解析：acc 回升 3 次，loss 回落 2 次），T03.1 exact 后降为 acc 回升 1 次、loss 回落 0 次。
- T05 在当前 run 中 50/50 选 `weight_bit`，但 metadata 并非“未进入候选”，而是长期在 top-K 处于劣势（详见诊断与等价条件文档）。

## 4. 低成本输入吸收情况

已读取并纳入以下低成本证据：
- `results/R1/legacy_mining_for_R1/Legacy_to_R1_Reference_Map.md`
- `results/R1/legacy_mining_for_R1/Legacy_Known_Bugs_And_Fixes.md`
- `results/R1/legacy_mining_for_R1/Legacy_to_R1_Hypothesis_Seeds.md`

其中对本轮判据影响最大的三点是：
- `quantized=False` 历史 bug 要求所有 R1 任务在 load 后必须 `calibrate_all_layers()`。
- T02/T03 的非单调现象优先解释为 proxy 局限，exact 版本用于主结论更 logical。
- T05 vs T06 不可直接等价，需在同 threat model/候选空间/数据口径下做对齐实验。
