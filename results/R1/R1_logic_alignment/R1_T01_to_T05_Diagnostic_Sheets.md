# R1_T01 ~ R1_T05 Diagnostic Sheets (参照 R1_T07)

## 总览结论
- `R1_T07` 是传统 sampleBFA 风格（value-only, per-layer topk, 单点 exact 验证后应用）。
- `R1_T01~T05` 大多是 metadata / joint threat model，不能直接要求与 T07 曲线重合。
- 当前最主要的“不够 logical”点不是单一代码 bug，而是“任务定义、选择策略、数据口径”混用导致可比性叙事不清。
- 诊断已吸收低成本文档：`Legacy_to_R1_Reference_Map.md`、`Legacy_Known_Bugs_And_Fixes.md`、`Legacy_to_R1_Hypothesis_Seeds.md`。

---

## T01 诊断（`run_R1_T01_group_metadata_index_anypattern.py`）

- 任务名与实现是否一致：`基本一致`。
- 攻击对象：index metadata（任意 2-of-4 模式迁移）。
- 选择策略：`proxy + optional topK exact`（默认 `topk_verify=64`，见 `run_R1_T01_group_metadata_index_anypattern.py:776`，`run_R1_T01_group_metadata_index_anypattern.py:635`）。
- 与 T07 的关系：`不同 threat model`（T07 是 value-only）。
- 结果证据：`92.33% -> 16.75%`，`n_acc_increase_steps=1`，`n_loss_decrease_steps=0`（`results/R1/R1_T01_group_metadata_index_anypattern_rerun_table.csv`）。

主要风险点：
- `Medium`：calib/eval 都来自 `test_loader`（`run_R1_T01_group_metadata_index_anypattern.py:889`），叙事上容易被误读为“独立评估”。
- `Low`：`exclude_groups` 用 set 截断（非有序）会引入微小不稳定性。

最小修正建议：
- 引入固定 `eval_indices` 和 `calib_indices`（允许重叠但需在日志报告 overlap 比例）。
- 将 `exclude_groups` 改为有序队列（deque）以提升可复现性。

---

## T02 诊断（`run_R1_T02_group_metadata_index_1bit.py`）

- 任务名与实现是否一致：`一致`。
- 攻击对象：index metadata，仅 1-bit reachable 转移，含 collision reject（`run_R1_T02_group_metadata_index_1bit.py:412`）。
- 选择策略：`proxy-only`（`run_R1_T02_group_metadata_index_1bit.py:582`）。
- 与 T07 的关系：`不同 threat model`，且 T02 没有 exact stage。
- 结果证据：
  - `92.33% -> 25.63%`（`results/R1/R1_T02_group_metadata_index_1bit_rerun_table.csv`）
  - `n_acc_increase_steps=5`, `n_loss_decrease_steps=7`（同表；日志 step 明确出现回弹）。

对“单调上升/回弹”的判定：
- 本例优先归因为 `proxy-only 近似误差 + 小样本梯度目标与评估目标不一致`，不是直接 bug。
- 支撑证据：T02.1 仅替换为 topK+exact 后，回弹显著减少。

主要风险点：
- `High`：用于对外结论时，proxy-only 曲线波动会被误读为实现问题。
- `Medium`：calib/eval 同源（`run_R1_T02_group_metadata_index_1bit.py:812`）。

最小修正建议：
- 默认切到 T02.1 逻辑（exact 选择）或至少加 `--selection {proxy,exact}` 开关并默认 exact。
- 保留 proxy 仅作为 ablation，不作为主结论曲线。

---

## T03 诊断（`run_R1_T03_group_metadata_bitmask_swap_cost2.py`）

- 任务名与实现是否一致：`一致`。
- 攻击对象：bitmask metadata，cost-2 swap（`run_R1_T03_group_metadata_bitmask_swap_cost2.py:105`）。
- 选择策略：`proxy-only`（`run_R1_T03_group_metadata_bitmask_swap_cost2.py:594`）。
- 物理约束：保持 popcount=2（swap 定义与 pattern 解析共同保障）。
- 与 T07 的关系：`不同 threat model`。
- 结果证据：`92.33% -> 65.14%`（`results/R1/R1_T03_group_metadata_bitmask_swap_cost2_table.csv`）。
- 回弹统计（由日志解析）：acc 回升 3 次、loss 回落 2 次。

对“单调上升/回弹”的判定：
- 与 T02 同类：优先解释为 proxy-only 近似误差，而非必然实现错误。
- 支撑证据：T03.1 exact 后变为 acc 回升 1 次、loss 回落 0 次，且 final acc 从 65.14% 降至 44.34%。

主要风险点：
- `Medium`：50 physical flips 仅等于 25 logical swaps，容易与 T01/T02 的“50 次操作”错比。

最小修正建议：
- 对外主图使用 T03.1 或 T04（50 logical swaps）做公平比较。
- 图标题强制同时标注 logical 与 physical 预算。

---

## T04 诊断（`run_R1_T04_bitmask_swaps50.py`）

- 任务名与实现是否一致：`一致`（专门做“50 logical swaps 公平比较”，`run_R1_T04_bitmask_swaps50.py:10`）。
- 攻击对象：bitmask metadata（cost-2 swap），仍是 proxy-only。
- 与 T07 的关系：`不同 threat model`，但可作为“bitmask 家族内部公平版”基线。
- 结果证据：`92.33% -> 50.73%`，logical=50，physical=100（`results/R1/R1_T04_bitmask_swaps50_table.csv`）。

主要风险点：
- `Medium`：若读者只看“50”不看单位，会与 T03/T05 产生错误对齐。

最小修正建议：
- 统一输出两条轴信息：`logical_step` 与 `physical_flips`。

---

## T05 诊断（`run_R1_T05_joint_best_step_attack.py`）

- 任务名与实现是否一致：`总体一致`（joint: weight/index/bitmask 三类 action，Stage A+Stage B）。
- 攻击对象：
  - weight-bit（cost=1），仅活跃非零权重（`run_R1_T05_joint_best_step_attack.py:405`, `run_R1_T05_joint_best_step_attack.py:409`）。
  - index_1bit（cost=1）。
  - bitmask_swap（cost=2，`score_proxy_norm=proxy/2`，`run_R1_T05_joint_best_step_attack.py:662`）。
- 选择策略：`topK + exact verify`（`run_R1_T05_joint_best_step_attack.py:1030`, `run_R1_T05_joint_best_step_attack.py:1041`）。
- 与 T07 的关系：`强化攻击者`（跨 action 家族联合搜索），不是传统 sampleBFA baseline。
- 结果证据：`92.33% -> 9.96%`，action breakdown=`50/0/0`（`results/R1/R1_T05_joint_best_step_attack_table.csv`）。

关键发现（已验证不是“metadata 死代码”）：
- Step1 候选总量：weight=10,500，index=50,047，bitmask=50,047。
- Step1 top-K：weight=63，index=1，bitmask=0。
- 全 50 步 top-K 平均：weight=63.04，index=0.96，bitmask=0；index 在 39/50 步进入过 top-K，bitmask 从未进入。
- 结论：metadata 并非未枚举，而是 proxy/cost 排序长期劣后。

主要风险点：
- `Medium`：Stage B `apply->eval->revert` 没有 `try/finally`（`run_R1_T05_joint_best_step_attack.py:936`~`run_R1_T05_joint_best_step_attack.py:950`），异常时存在状态污染风险。
- `Medium`：Stage B exact 用 `calib_loader` + `eval_samples`（2000），而 Stage A 梯度常用 calib_samples=256，口径不对称。
- `Medium`：calib/eval 同源（`run_R1_T05_joint_best_step_attack.py:1310`）。

最小修正建议：
- 给 Stage B 增加 `try/finally + hash assert`，保证异常也能 revert。
- 增加 `--stratified-topk` 调试开关（如 32/16/16）仅用于自检，不改默认主逻辑。
- 与 T06 对齐比较时，新增 `--weight-only` 运行配置，去除 metadata 干扰。

---

## 特别结论 1：T02/T03 的“上升现象”是否一定是 bug？

- 判定：`不是直接 bug`，优先视为代理近似导致的局部非单调。
- 证据链：
  - T02 proxy-only：5 次 acc 回升 / 7 次 loss 回落。
  - T02.1 exact：降到 1 次 / 0 次。
  - T03 proxy-only：3 次 / 2 次；T03.1 exact：1 次 / 0 次。
- 解释：exact verification 把“proxy 误判”的动作筛掉后，曲线更稳定，这更像方法差异而非实现错误。

## 特别结论 2：T05 vs T06 global 为什么不能直接要求一致？

- 必须同时满足“同 threat model + 同候选空间 + 同选择口径 + 同数据口径 + 同停止条件”才可近似一致。
- 当前并不满足（详见 `R1_T05_vs_T06_Global_Equivalence_Conditions.md`）。
