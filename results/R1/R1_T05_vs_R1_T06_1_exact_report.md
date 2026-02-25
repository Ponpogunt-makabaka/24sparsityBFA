# R1_T05 vs R1_T06.1 Exact Verification 对比分析报告

## 数据来源
- `results/R1/R1_T06_1_legacy_dense_format_bfa_exact/R1_T06_1_task1_global_trace.csv`
- `results/R1/R1_T06_1_legacy_dense_format_bfa_exact/R1_T06_1_task2_zero_only_trace.csv`
- `results/R1/R1_T06_1_legacy_dense_format_bfa_exact/R1_T06_1_task3_nonzero_only_trace.csv`
- `results/R1/R1_T06_1_legacy_dense_format_bfa_exact/R1_T06_1_summary_table.csv`
- `results/R1/R1_T05_joint_best_step_attack_result.pkl`
- `results/R1/R1_T05_joint_best_step_attack_table.csv`

## 结论先行
- 当前这批结果中，`R1_T06.1 global` 在 **第 6 步达到 10.2%**（near-random），明显快于 `R1_T05` 在 **第 12 步达到 9.96%**。
- 前 5 步累计 `ΔL_exact`：T06.1 global=11.778424，T05=10.541862，T06.1 高 **11.73%**。
- 本次数据里，T06.1 global 前 6 步全部打在 `old_int8 != 0`；但 `zero_only` 轨迹证明“被剪枝后置零位点”一旦在 dense-format 中暴露，仍有显著破坏力。

## 1) 动作落点分布比对 (Action Target Distribution)

| 轨迹 | 分析步数 | old_int8==0 次数 | old_int8!=0 次数 | old_int8==0 占比 |
|---|---:|---:|---:|---:|
| T06.1 global (前15步窗口，实际仅6步) | 6 | 0 | 6 | 0.00% |
| T06.1 zero_only (全程) | 12 | 12 | 0 | 100.00% |
| T06.1 nonzero_only (全程) | 5 | 0 | 5 | 0.00% |
| T05 joint (历史日志) | 50 | N/A | N/A | action 全为 `weight_bit` (50/50) |

对比解释：
- T06.1 global 在当前 run 中并没有优先选择 `old_int8==0` 位点（前 6 步全是非零）。
- T05 历史结果 `50/50` 全是 `weight_bit`，且该任务语义限定在 sparse-gated 活跃权重域内搜索。

## 2) 破坏力步效分析 (Step-wise Lethality)

| 指标 | R1_T05 | R1_T06.1 global |
|---|---:|---:|
| 达到 acc<=10.5% 的步数 | 8 | 6 |
| 达到 acc<=10.0% 的步数 | 12 | 未在已记录步数内达到 |
| 第5步累计 `ΔL_exact` | 10.541862 | 11.778424 |
| 第5步累计 `ΔL_exact` 比值 (T06.1/T05) | - | 1.1173x |

补充：T06.1 global 仅记录到 6 步，是因为脚本在 `acc<12%` 时提前停止；第 6 步 `acc_eval=10.20%`。

## 3) 搜索空间的数学优势 (Search Space Advantage)

### 3.1 T05 的候选池/Top-K 竞争（Step 1）

| 指标 | 数值 |
|---|---:|
| Stage-A 候选总数 | 110594 |
| weight_bit 候选数 | 10500 (9.49%) |
| index_1bit 候选数 | 50047 |
| bitmask_swap 候选数 | 50047 |
| Top-K 组成 (K=64) | weight=63, index=1, bitmask=0 |

### 3.2 T06.1 三种模式的 exact 强度对比（来自 CSV）

| 模式 | 步数 | mean(ΔL_exact) | max(ΔL_exact) | 最终 acc |
|---|---:|---:|---:|---:|
| T06.1 global | 6 | 2.830441 | 5.204223 | 10.20% |
| T06.1 zero_only | 12 | 0.672236 | 1.483211 | 11.95% |
| T06.1 nonzero_only | 5 | 2.828733 | 5.352670 | 10.15% |

### 3.3 前5步逐步对比（同起点，不同模式各自贪心）

| step | global ΔL_exact | zero_only ΔL_exact | nonzero_only ΔL_exact |
|---:|---:|---:|---:|
| 1 | 0.212009 | 0.097579 | 0.212009 |
| 2 | 1.082937 | 0.230070 | 1.082937 |
| 3 | 2.668168 | 0.399045 | 3.062389 |
| 4 | 3.871228 | 0.515610 | 5.352670 |
| 5 | 3.944082 | 0.540608 | 4.433660 |

数据解释：
- 当前 run 中，global 的高破坏动作主要与 nonzero_only 通道同量级；zero_only 的单步 `ΔL_exact` 明显更小。
- 但 zero_only 能独立将准确率打到 11.95%，说明 dense-format 暴露的“原0权重位点”不是死区，而是有效附加攻击面。

## 4) 密集格式搜索的语义漏洞 (Semantic Loophole)

结合本次 CSV 证据与 T06.1 机制，可得：
1. `zero_only` 全程 `old_int8==0`（100%），却在 12 步内把 acc 从 92.40% 降到 11.95%，loss 显著上升。
2. 这说明 dense-format 下，原本被 sparse mask 隐藏的连接一旦被翻成大幅值（如 64/-128），会直接进入前向传播并改变激活分布。
3. 对已在稀疏拓扑上适配的模型来说，这等价于“向未训练过的路径注入强噪声权重”，会造成分布偏移和级联误差。

量化支撑（zero_only）：
- 最终 acc drop: 80.45%
- `L_eval` 增量（末步 - baseline）: 8.774923

## 结语
- 这组结果支持“在同为 Exact Verification 下，T06.1 比 T05 更快崩溃”的现象。
- 但就当前 CSV 而言，T06.1 global 的快崩并非来自“前几步大量打 zero 位点”，而是来自更强的非零位点候选；zero 位点提供的是额外后备破坏通道。
