# R1_T01-T05 Logical Refactor Plan

## Layer 1（必须修）

### L1-0 量化状态硬断言（防止历史 quantized=False 复发）
- 问题描述：Legacy 已出现过 load 后未量化激活导致 metadata 攻击“看起来无效”的 P0 问题（见 `Legacy_Known_Bugs_And_Fixes.md`）。
- 为什么不 logical：如果 `quantized=False`，forward 语义会偏离任务定义，结论直接失效。
- 最小修复方案：
  - 在 T01-T05 load 完成后统一断言关键层 `quantized=True`。
  - 若不满足则强制 `calibrate_all_layers()` 后再断言并记录日志。
- 验证方法：单步 flip + delta-logits 断言（5 分钟内完成）。
- 预期影响：避免“攻击无效但程序仍继续”的假阴性结论。

### L1-1 统一 Eval/Calib 口径（避免结论误读）
- 问题描述：T01-T05 当前均使用 `calib_loader=test_loader`（例如 `run_R1_T05_joint_best_step_attack.py:1310`）。
- 为什么不 logical：优化目标与报告指标在同一数据源，容易被误读为“泛化评估”；也会放大 proxy/selection 对该子集的偶然拟合。
- 最小修复方案：
  - 抽取 `FixedSubsetLoader + generate_fixed_indices` 到公共 util。
  - 新增参数：`--eval-seed`, `--calib-seed`, `--save-indices`。
  - 日志新增 `overlap_count/overlap_ratio`。
- 验证方法：5~10 steps 快跑，比较 old/new 曲线的波动幅度与重跑一致性。
- 预期影响：不一定改变最终最优值，但能显著提升“可比性与可解释性”。

### L1-2 T05 Stage-B 强化 Revert 安全性（防状态污染）
- 问题描述：T05 Stage-B 目前 `apply -> eval -> revert` 未用 `try/finally` 包裹（`run_R1_T05_joint_best_step_attack.py:936`~`run_R1_T05_joint_best_step_attack.py:950`）。
- 为什么不 logical：一旦中途异常，模型可能带脏状态继续验证后续候选。
- 最小修复方案：
  - 用 `try/finally` 强制 revert。
  - 每次候选后追加 hash assert（`int8_hash`, `metadata_hash`）可选开关 `--strict-revert-check`。
- 验证方法：在 Stage-B 人为注入一次异常（第3候选）并确认后续 hash 不漂移。
- 预期影响：结论稳定性提升，避免“隐性污染导致虚假最优”。

### L1-3 T05 增加“threat model preset”避免命名误导
- 问题描述：用户常把“全选 weight”误解为“与 dense global 等价”。
- 为什么不 logical：`all-action joint` 与 `weight-only` 是不同 threat model；需要显式可切换。
- 最小修复方案：
  - 新增 `--mode {joint,weight_only,metadata_only}`，内部映射到三个 disable flags。
  - 日志头部强制打印 mode + enabled_actions。
- 验证方法：每个 mode 跑 5 步，确认候选与 action breakdown 行为符合预期。
- 预期影响：减少语义误会；便于与 T06/T07 做公平对照。

---

## Layer 2（建议修）

### L2-1 默认对外结果优先 T02.1/T03.1（exact）
- 问题描述：T02/T03 proxy-only 有明显局部回弹，容易触发“实现错误”误判。
- 为什么不 logical：当存在更强的 exact 版本时，主结论继续用 proxy-only 可解释性偏弱。
- 最小修复方案：
  - 保留 T02/T03 为 ablation。
  - 对外主图切换为 T02.1/T03.1；报告注明 proxy-only 仅方法对照。
- 验证方法：5~10步对照跑 proxy vs exact，统计 `n_acc_increase_steps/n_loss_decrease_steps`。
- 预期影响：曲线更稳定，审稿/复核时争议更小。

### L2-2 统一停止准则（Stop rule）并记录触发原因
- 问题描述：不同任务的停止条件不统一（如 T06.1 有 near-random 提前停，T05 跑满预算）。
- 为什么不 logical：步数比较会混入“停止策略差异”。
- 最小修复方案：
  - 新增统一参数：`--stop-acc-threshold`（默认禁用），所有任务通用。
  - 日志写入 `stop_reason`。
- 验证方法：固定预算 10 步测试启用/禁用 stop rule。
- 预期影响：跨任务“到 random level 的步数”比较更公平。

### L2-3 标准化候选审计日志
- 问题描述：T01-T04 缺少与 T05 同等级的每步候选分解（按 action/type/topK）。
- 为什么不 logical：无法快速区分“算法真实偏好”与“候选空间偏置”。
- 最小修复方案：
  - 统一输出字段：`N_candidates_by_type`, `TopK_by_type`, `best_proxy_by_type`, `best_exact_by_type`。
- 验证方法：跑 5 步检查日志字段完整性。
- 预期影响：debug 成本显著下降，证据链更完整。

### L2-4 约束集合改用有序结构
- 问题描述：`exclude_groups = set(list(...)[-20:])` 依赖 set 顺序，复现性弱。
- 为什么不 logical：相同 seed 也可能在不同 Python 运行态出现微差异。
- 最小修复方案：
  - 用 `collections.deque(maxlen=20)` 存历史，再维护一个 membership set。
- 验证方法：同 seed 重跑 3 次，比较 step trace 一致性。
- 预期影响：复现实验稳定性提升。

---

## Layer 3（可选增强）

### L3-1 Replay Test（共享 flip 序列复放）
- 问题描述：当前多任务对比常混入“搜索器差异”和“语义差异”。
- 为什么不 logical：没有分离变量，难归因。
- 最小修复方案：
  - 保存 flip/action 序列，支持在另一个 runner 上 replay。
- 验证方法：将 T05 前 10 步 replay 到 T06 语义，比较曲线偏差。
- 预期影响：可精确拆分“搜索策略影响”和“forward语义影响”。

### L3-2 Shared-flips Transfer（T05 vs T06）
- 问题描述：用户经常问“为什么两者不一致”。
- 为什么不 logical：缺少统一动作集合下的 A/B forward 对照。
- 最小修复方案：
  - 固定一组 weight flips，在 T05 与 T06 语义各自评估同一序列。
- 验证方法：5步短跑，输出逐步 `delta_logits / delta_loss`。
- 预期影响：差异归因更可解释。

### L3-3 Stratified Top-K 诊断开关
- 问题描述：联合搜索中某类动作长期不进 top-K 时难判断是否纯实力差。
- 为什么不 logical：缺少控制实验，无法排除“候选数量挤占”偏置。
- 最小修复方案：
  - 仅在 debug 模式下启用 `--topk-stratified w:i:b`。
- 验证方法：10步对照 `normal vs stratified`，比较最终 chosen action type。
- 预期影响：快速确认“全 weight”是 dominance 还是 selection bias。
