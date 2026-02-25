# Task1xx Metadata Attack - Code Path Audit Report

**Date**: 2026-02-16
**Purpose**: 定位为什么 Task1xx 攻击修改了 metadata 但准确率不变

---

## 1. 数据流表

| 组件 | 变量名 | Shape | 存储位置 | 读写性质 |
|------|--------|-------|----------|----------|
| **被修改对象** | `module.sparse_mask` | 变化 (各层不同) | `Int8QuantizedConv2d.register_buffer` | **写**：Task1xx `apply_pattern_change()` |
| **被修改对象** | `module.int8_weights` | 变化 | `Int8QuantizedConv2d.register_buffer` | **写**：Task1xx `apply_pattern_change()` (移动权重值) |
| **forward读取对象** | `module.sparse_mask` | 同上 | 同上 | **读**：`get_dequantized_weights()` 第88行 |
| **forward读取对象** | `module.int8_weights` | 同上 | 同上 | **读**：`get_dequantized_weights()` 第86行 |
| **forward使用对象** | `dequantized_weight` | 临时变量 | stack (临时) | `forward()` 第96-97行 |

### 关键代码路径

**写入路径** (Task1xx `apply_pattern_change()`):
```python
# run_task1xx_group_metadata_attack.py:481-482
module.sparse_mask.copy_(m_new.clone())    # 直接修改 sparse_mask
module.int8_weights.copy_(w_new.clone())   # 同时移动 int8 权重值
```

**读取路径** (Int8QuantizedConv2d `forward()`):
```python
# train/ptq_convert.py:92-100
def forward(self, x):
    if self.quantized:
        original_weight = self.weight.data
        dequantized_weight = self.get_dequantized_weights()  # 调用下面
        self.weight.data = dequantized_weight
        output = super().forward(x)
        self.weight.data = original_weight
        return output

def get_dequantized_weights(self):
    w_dequantized = self.int8_weights.float() * self.scale
    if self.sparse_mask is not None:
        w_dequantized = w_dequantized * self.sparse_mask  # sparse_mask 参与计算
    return w_dequantized
```

### 结论 (A) = (B)
- ✅ `module.sparse_mask` 是同一个对象（register_buffer，存在于模块状态中）
- ✅ 写入路径和读取路径操作**同一个 buffer**
- ✅ 不存在缓存/副本导致写入无效的问题

---

## 2. 潜在问题分析

### P0.1: mask_fixed 检查
检查 `task28_sparse_mask_fixed_finetune_int8_ckpt.pth` 命名：
- ✅ 文件名包含 `mask_fixed`，但这只是训练策略描述（mask在finetune期间保持固定）
- ✅ sparse_mask **确实存储在 checkpoint 中**（通过 `load_state_dict` 加载）
- ✅ forward **确实使用** sparse_mask（见上述代码路径）

**结论**: mask_fixed ≠ metadata 不参与 forward。metadata (sparse_mask) 确实在 forward 中被使用。

### P0.2: 权重移动逻辑问题
在 `apply_pattern_change()` 中：
```python
# 第462-475行
old_values = w_flat[g_idx, old_active_indices].clone()
w_flat[g_idx, :] = 0
m_flat[g_idx, :] = 0
for i, pos in enumerate(new_pattern):
    if i < len(old_values):
        w_flat[g_idx, pos] = old_values[i]
        m_flat[g_idx, pos] = 1
```

**潜在问题**:
1. 权重值从旧位置移动到新位置，但 `int8_weights` 值本身没有重新量化
2. 这是 **metadata-only 攻击**，不改变 int8 权重值，只改变它们的位置

### P0.3: Δw̃_g 构造问题
```python
# compute_dense_reconstruction() 函数
w_tilde = w_group.float() * scale * m_group
```
这里 `m_group` 来自 `module.sparse_mask`，所以 Δw̃ 构造正确。

---

## 3. 需要进一步验证的问题

1. **一步差分测试** (Step 2): 验证修改 metadata 是否真的影响 logits/loss
2. **随机破坏测试** (Step 3): 验证大规模 metadata 破坏是否影响准确率
3. **Task1xx 插桩** (Step 4): 验证每步修改是否真的生效

---

## 4. 基线模型信息

| 模型 | 路径 | Top-1 Acc |
|------|------|-----------|
| Task28 Int8 | `results/task28_sparse_mask_fixed_finetune_int8_ckpt.pth` | 92.21% |
| Legacy Sparse | `models/sparse_model.pth` | 86.50% |

**注意**: Task28 基线准确率显著高于 Legacy 基线（+5.7%），这可能影响攻击效果。
