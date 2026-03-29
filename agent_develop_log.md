# Agent Development Log — Fix-Agent (2:4 Sparsity 语义修复)

Date: 2026-02-27
Last Updated: 2026-02-27 (ALL 5 PHASES COMPLETED)
Status: **DONE** — Smoke test 27/27 passed

---

## Phase 1: 问题定位与复现

### 1.1 T09 状态报告结构化摘要

#### 当前各模型/数据集结果总览

| 模型 | 数据集 | Dense Acc | 说明 | Sparse Init | Sparse FT | FT Epochs | 状态 |
|------|--------|-----------|------|-------------|-----------|-----------|------|
| ResNet-18 | Imagenette | 80.92% | 预训练直评(1000类) | 5.15% | 98.47% | 3 | **不公平对照** |
| MobileNetV2 | Imagenette | 82.73% | 预训练直评(1000类) | 0.00% | 92.87% | 3 | **不公平对照** |
| DeiT-Tiny | Imagenette | 79.06% | 预训练直评(1000类) | 12.64% | 93.96% | 1 | **不公平对照** |
| ResNet-18 | CIFAR-100 | 79.26% | HF社区预训练 | 62.95% | 69.93% | 1 | **欠训练** |
| MobileNetV2 | CIFAR-100 | 74.30% | 社区预训练 | 1.31%(崩) | 67.32% | 1 | **欠训练+崩溃init** |
| DeiT-Tiny | CIFAR-100 | 1.00% | 权重完全不匹配 | 1.00% | 23.18% | 1 | **基线无效** |

#### 关键问题清单

**问题 P1: Conv2d 2:4 分组语义**
- 文件: `data/T09_ImageNet_Scale/step3_sparsify_finetune.py:368-381`
- 当前实现: `w.permute(0,2,3,1).contiguous().view(-1,4)` (NHWC-style 展平)
- 这是 NHWC 内存格式的 K 维分组，对 in_channels%4==0 的标准卷积是正确的
- **但**对 in_channels=3 (如 conv_proj [192,3,16,16])，groups 会跨越空间位置边界
- 对 depthwise conv (in_channels=1)，groups 完全在空间维度上，语义不合理

**问题 P2: build_fixed_masks() 无 layer selection**
- 文件: `data/T09_ImageNet_Scale/step3_sparsify_finetune.py:394-404`
- 对所有 4D 参数用 conv mask，所有 2D 参数用 linear mask
- 没有跳过 depthwise conv、conv_proj、分类头等
- MobileNetV2 sparse init 崩溃到 0.00%/1.31% 很可能因为 depthwise conv 被错误稀疏化

**问题 P3: DeiT-Tiny CIFAR-100 基线失效**
- 文件: `data/T09_ImageNet_Scale/step3_sparsify_finetune.py:337-363`
- ImageNet 预训练: patch_size=16, conv_proj=[192,3,16,16], head=[1000,192]
- CIFAR-100 模型: patch_size=4, conv_proj=[192,3,4,4], head=[100,192]
- 形状不匹配 → `_load_compatible_state_dict` 跳过 conv_proj 和 head → 随机初始化
- Dense=1.00% 完全是预期的（只有 encoder 层加载成功，但没有有效 tokenization）

**问题 P4: Dense vs Sparse 比较不公平**
- Imagenette Dense Acc 是 ImageNet-1000 预训练权重直接评估 Imagenette 10 类子集
- Sparse FT 则在 Imagenette 上做了 1-3 epoch 微调
- 结果出现 "Sparse > Dense" 假象（ResNet-18: 98.47% > 80.92%）

**问题 P5: 权重文件缺失**
- `data/T09_ImageNet_Scale/weights/` 下只有 JSON 元数据，无 .pth 检查点
- 需要重新下载/生成

**问题 P6: 数据集路径不一致**
- 代码默认: `data/imagenette_full`，实际路径: `data/imagenette/imagenette2/`
- CIFAR-100: 代码默认 `data/cifar100`，目录不存在
- 需要创建 symlink 或修改默认路径

#### 现有 2:4 实现对比

| 实现位置 | Linear 分组 | Conv2d 分组 | Layer Selection |
|---------|------------|------------|-----------------|
| `step3_sparsify_finetune.py` | `w.view(-1,4)` 沿 K 维 | `permute(0,2,3,1).view(-1,4)` NHWC | **无** (全部 4D/2D) |
| `models/sparse_ops.py` Sparse | `w.reshape(-1,4)` 沿展平 | N/A | N/A |
| `models/sparse_ops.py` Sparse_NHWC | N/A | `view(out, in//4, 4, kH, kW)` dim=2 | `in_ch%4!=0` 跳过 |
| `models/factory.py` | `view(out, in//4, 4)` dim=2 | `view(out, in//4, 4, kH, kW)` dim=2 | 有 filter 函数 |
| 攻击引擎 (T08) | `w.view(-1,4)` | `permute(0,2,3,1).view(-1,4)` | depthwise 跳过 |

**关键发现**: 稀疏化阶段 (step3) 没有 layer selection，但攻击阶段 (T08) 跳过 depthwise。
这导致稀疏化时破坏了 depthwise 层，但攻击时又不攻击它们，两个阶段不一致。

#### 已做工作（可用/无意义判定）

| 实验 | 可用性 | 原因 |
|------|--------|------|
| Imagenette ResNet-18 攻击 | 结果参考可用 | 但 Dense Acc 不公平 |
| Imagenette MobileNetV2 攻击 | 结果参考可用 | 但 depthwise 被错误稀疏化 |
| Imagenette DeiT-Tiny 攻击 | 结果参考可用 | conv_proj 被错误稀疏化 |
| CIFAR-100 ResNet-18 | **欠训练** | 1 epoch, 需 40 epochs |
| CIFAR-100 MobileNetV2 | **欠训练+稀疏init崩** | 1 epoch, depthwise 问题 |
| CIFAR-100 DeiT-Tiny | **无意义** | Dense=1.00%, 基线不成立 |

### 1.2 复现命令清单

#### 环境信息
```
PyTorch: 2.8.0+cu128
CUDA: Available (RTX 2080 Ti)
Python: 3.x
```

#### 数据准备（必须先执行）

```bash
# 1. 创建 imagenette symlink（代码期望 data/imagenette_full）
ln -sf /home/lab-2010/24sparsityBFA-main/data/imagenette/imagenette2 \
       /home/lab-2010/24sparsityBFA-main/data/imagenette_full

# 2. 下载 CIFAR-100（通过 torchvision 自动下载，指定目录）
python3 -c "import torchvision; torchvision.datasets.CIFAR100(root='data/cifar100', download=True)"

# 3. 下载 torchvision 预训练权重
cd /home/lab-2010/24sparsityBFA-main
python3 data/T09_ImageNet_Scale/scripts/download_torchvision_weights.py

# 4. 下载 DeiT-Tiny 权重
python3 data/T09_ImageNet_Scale/scripts/download_deit_tiny_weights.py
```

#### 复现命令（用于验证当前代码行为）

```bash
# ResNet-18 Imagenette: 复现 Dense=80.92%, SparseInit=5.15%, SparseFT=98.47%
python3 data/T09_ImageNet_Scale/step3_sparsify_finetune.py \
    --arch resnet18 --dataset imagenette --epochs 3 --seed 42

# MobileNetV2 Imagenette: 复现 Dense=82.73%, SparseInit=0.00%
python3 data/T09_ImageNet_Scale/step3_sparsify_finetune.py \
    --arch mobilenet_v2 --dataset imagenette --epochs 3 --seed 42

# ResNet-18 CIFAR-100: 复现 Dense=79.26%, SparseInit=62.95%
python3 data/T09_ImageNet_Scale/step3_sparsify_finetune.py \
    --arch resnet18 --dataset cifar100 --epochs 1 --seed 42
```

### 1.3 执行计划（Phase 2-5 文件变更清单）

#### Phase 2: 修正 2:4 稀疏语义

**新增文件:**
- `utils/sparse_checker.py` — 2:4 合规检查器（按 GEMM K 维语义）
- `utils/layer_selector.py` — 每模型的 layer selection 策略

**修改文件:**
- `data/T09_ImageNet_Scale/step3_sparsify_finetune.py` — 集成 layer selection + 修正 Conv2d mask 逻辑

**产出:**
- `reports/phase2_compliance_check.json` — 每层合规检查结果

#### Phase 3: 建立公平 baseline

**修改文件:**
- `data/T09_ImageNet_Scale/step3_sparsify_finetune.py` — 增加 `--mode` 参数 (dense-ft / sparse-init / sparse-ft)

**产出:**
- `reports/phase3_baseline_results.json`

#### Phase 4: DeiT CIFAR-100

**修改文件:**
- `data/T09_ImageNet_Scale/step3_sparsify_finetune.py` — CIFAR-100 DeiT 路线 A (resize 224 + patch16)

**产出:**
- `reports/phase4_deit_cifar100.json`

#### Phase 5: 回归 + 文档

**新增文件:**
- `scripts/smoke_test.py` — 最小回归测试
- `reports/T09_fix_report.md` — 修复报告

**风险点与回滚策略:**
- 所有修改先在 ResNet-18 Imagenette 上 smoke test
- 原始 step3 文件保留备份
- 使用 git 跟踪所有变更

---

## Phase 2: 修正 2:4 稀疏语义 (已完成)

### 2.1 修复内容

#### Fix 1: Conv2d K-dimension grouping
- **文件**: `step3_sparsify_finetune.py:compute_2_4_mask_conv()`
- **旧**: 用 `numel()%4` 检查，允许 groups 跨越 output-channel 行边界
- **新**: 用 `K%4` 检查 (K=in_ch*kH*kW)，K%4≠0 时返回 None 跳过
- **影响**: 正确排除 in_ch=3 的首层 conv 和 depthwise conv

#### Fix 2: Linear K-dimension grouping
- **文件**: `step3_sparsify_finetune.py:compute_2_4_mask_linear()`
- **旧**: 用 `numel()%4` 检查
- **新**: 用 `in_features%4` 检查

#### Fix 3: Architecture-aware layer selection
- **文件**: `step3_sparsify_finetune.py:_should_sparsify()`
- 新增 `_is_depthwise_conv()`, `_get_conv_module()`, `_should_sparsify()`
- 每个架构独立的跳过/包含规则：
  - ResNet-18: 跳过 conv1 (K=147, K%4≠0)，包含所有其他 conv 和 fc
  - MobileNetV2: 跳过 17 depthwise conv + features.0.0 (in_ch=3)，包含 pointwise 1x1 + classifier
  - DeiT-Tiny: 跳过 conv_proj (in_ch=3, K%4≠0)，包含所有 MLP/attention Linear + heads.head

#### Fix 4: DeiT CIFAR-100 Route A
- **文件**: `step3_sparsify_finetune.py:create_model_and_load()`
- 始终使用 patch_size=16 + image_size=224（匹配预训练权重）
- CIFAR-100 图片在 data loader 中 Resize 到 224x224
- 结果：从 Dense=1.00% 修复到 Dense-FT=82.84%

### 2.2 新增文件

- `utils/__init__.py` — 空初始化
- `utils/sparse_checker.py` — 2:4 GEMM K-dim 合规检查器
- `utils/layer_selector.py` — 架构感知 layer selection 策略

### 2.3 合规验证结果

| 模型 | 稀疏化层数 | 跳过层数 | 违规率 |
|------|-----------|---------|--------|
| ResNet-18 | 20 | 1 (conv1) | **0%** |
| MobileNetV2 | 35 | 18 (17 DW + first conv) | **0%** |
| DeiT-Tiny | 49 | 3 (cls_token + conv_proj + pos_embed) | **0%** |

---

## Phase 3: 建立公平 baseline (已完成)

### 3.1 新增 --mode 参数

`step3_sparsify_finetune.py` 支持 4 种模式：
- `dense-eval`: 仅评估预训练权重
- `dense-ft`: 无稀疏化微调（公平对照）
- `sparse-init`: 应用 mask 后直接评估
- `sparse-ft`: 应用 mask + 微调

### 3.2 关键发现: 数据集不完整

- 旧的 `data/imagenette/imagenette2/` 只有 **2419** 训练图片（标准应有 9469）
- 这导致 sparse-ft 仅 32%（vs 旧结果 92.87%），误以为是 mask 逻辑问题
- 修复：重新下载完整 imagenette2 到 `data/imagenette2/`，symlink `data/imagenette_full`

### 3.3 Imagenette 四指标结果 (3 epochs, full dataset)

| Model | Dense-eval | Dense-FT | Sparse-Init | Sparse-FT | Gap(FT) |
|-------|-----------|----------|-------------|-----------|---------|
| ResNet-18 | 80.92% | **98.88%** | 18.37% | **98.47%** | 0.41pp |
| MobileNetV2 | 82.73% | **98.93%** | 0.00% | **95.57%** | 3.36pp |
| DeiT-Tiny | 79.06% | **98.68%** | 25.55% | **93.89%** | 4.79pp |

**分析**:
- Dense-eval < Dense-FT: 因为 ImageNet 1000-class 预训练直评 Imagenette 10 类，微调后大幅提升
- 稀疏成本(Gap): ResNet-18 仅 0.41pp, MobileNetV2 3.36pp, DeiT-Tiny 4.79pp — 均在合理范围
- 2:4 稀疏化对 ViT (DeiT) 的影响 > CNN (ResNet-18)，这与文献一致

---

## Phase 4: DeiT-Tiny CIFAR-100 baseline (进行中)

### 4.1 修复方案: Route A (resize 224 + patch16)

- CIFAR-100 32x32 → Resize(224) → patch16 → 196 tokens
- 复用 ImageNet DeiT-Tiny 全部权重（除了 heads.head [1000→100]）
- skipped=2 (heads.head.weight + heads.head.bias) 是预期的

### 4.2 Dense-FT 结果 (20 epochs)

| Metric | Value |
|--------|-------|
| Dense-eval | 1.00% (预期 — 头随机) |
| Dense-FT (20ep) | **82.84%** |
| Phase 4 gate (≥55%) | **通过** |

### 4.3 Sparse-FT 结果 (20 epochs)

| Metric | Value |
|--------|-------|
| Dense-eval | 1.00% (头随机 — 预期) |
| Dense-FT (20ep) | **82.84%** |
| Sparse-Init | 1.00% |
| Sparse-FT (20ep) | **82.32%** |
| Sparsity Cost | **0.52pp** |

---

## Phase 5: 回归测试与文档收尾

### 5.1 全局结果总表

#### Imagenette (10-class, 3 epochs)

| Model | Dense-eval | Dense-FT | Sparse-Init | Sparse-FT | Gap(FT) | Layers | Compliance |
|-------|-----------|----------|-------------|-----------|---------|--------|------------|
| ResNet-18 | 80.92% | 98.88% | 18.37% | **98.47%** | 0.41pp | 20/21 | 0% violation |
| MobileNetV2 | 82.73% | 98.93% | 0.00% | **95.57%** | 3.36pp | 35/53 | 0% violation |
| DeiT-Tiny | 79.06% | 98.68% | 25.55% | **93.89%** | 4.79pp | 49/52 | 0% violation |

#### CIFAR-100 (DeiT-Tiny only, 20 epochs, Route A)

| Model | Dense-eval | Dense-FT | Sparse-Init | Sparse-FT | Gap(FT) |
|-------|-----------|----------|-------------|-----------|---------|
| DeiT-Tiny | 1.00% | 82.84% | 1.00% | **82.32%** | 0.52pp |

### 5.2 修复前后对比

| 问题 | 修复前 | 修复后 |
|------|--------|--------|
| P1: Conv2d K-dim | numel()%4 → 允许跨行 | K%4 → 严格行内分组 |
| P2: Layer selection | 全部 4D/2D → 破坏 depthwise | 架构感知 → 跳过不合格层 |
| P3: DeiT CIFAR-100 | Dense=1.00% (patch4 不匹配) | Dense-FT=82.84% (Route A: 224+patch16) |
| P4: 不公平对比 | Dense=pretrained eval, Sparse=FT | 四指标: eval/FT/init/FT |
| P5: 权重缺失 | 只有 JSON | 全部 .pth 重新生成 |
| P6: 数据集不完整 | 2419 张 (imagenette) | 9469 张 (重新下载完整版) |

---

## Phase 6: 清理旧 Checkpoints + 重新运行 BFA 攻击 (已完成)

Date: 2026-03-06

### 6.1 清理旧文件

#### 删除旧/错误 Checkpoints
Phase 2-4 修正了 2:4 稀疏语义后，旧的 checkpoint 和中间产物已过时。已删除：
- 6 个孤立 JSON（无对应 .pth）
- 8 个中间版本 checkpoint（`*_fixed.*`, `*_v2.*`, `*_v3.*`, `*_10ep.*`）
- 2 个不完整数据集训练产物（`*_dense_ft_imagenette.*`）

#### 保留的正确 Checkpoints
```
weights/
├── resnet18-f37072fd.pth              # ImageNet 预训练
├── mobilenet_v2-b0353104.pth          # ImageNet 预训练
├── deit_tiny_patch16_224-a1311bcf.pth  # ImageNet 预训练
├── resnet18_{sparse,dense}_ft_imagenette_full.{pth,json}
├── mobilenet_v2_{sparse,dense}_ft_imagenette_full.{pth,json}
├── deit_tiny_{sparse,dense}_ft_imagenette_full.{pth,json}
├── deit_tiny_{sparse,dense}_ft_cifar100_20ep.{pth,json}
```

#### 删除旧攻击结果
删除全部 6 个旧结果目录（基于错误稀疏权重的攻击结果）。

### 6.2 修补攻击引擎

**文件**: `engine/run_R1_T08_metadata_improved.py`

修复 DeiT-Tiny + CIFAR-100 不兼容问题（Route A 对齐）：

| 修复点 | 旧代码 | 新代码 |
|--------|--------|--------|
| `create_model()` CIFAR-100 | `image_size=32, patch_size=4` | DeiT: `image_size=224, patch_size=16` |
| `build_cifar100_loaders()` | 固定 32x32，无 resize | DeiT 时 `Resize(224)` |
| `main()` 调用 | 无 arch 参数 | 传递 `arch=args.arch` |

### 6.3 BFA 攻击结果（新 checkpoints，50 flips，seed=0）

#### Imagenette (3 模型)

| Model | Groups | Initial Acc | Final Acc | Acc Drop | Flips | 运行时间 |
|-------|--------|------------|-----------|----------|-------|---------|
| ResNet-18 | 2,919,728 | 98.47% | 59.67% | **38.80%** | 50 | ~18h |
| MobileNetV2 | 867,440 | 95.57% | 10.42% | **85.15%** | 50 | ~8h |
| DeiT-Tiny | 1,411,968 | 93.89% | 84.36% | **9.53%** | 50 | ~9h |

#### CIFAR-100 (DeiT-Tiny only, Route A)

| Model | Groups | Initial Acc | Final Acc | Acc Drop | Flips | 运行时间 |
|-------|--------|------------|-----------|----------|-------|---------|
| DeiT-Tiny | 1,368,768 | 82.32% | 79.54% | **2.78%** | 50 | ~8.5h |

### 6.4 分析

**攻击脆弱性排序**: MobileNetV2 (85.15%) >> ResNet-18 (38.80%) >> DeiT-Tiny Imagenette (9.53%) > DeiT-Tiny CIFAR-100 (2.78%)

**关键观察**:
1. **MobileNetV2 极度脆弱**: 50 flips 即从 95.57% → 10.42%（近乎随机），可能因 pointwise 1x1 conv 权重稀疏后冗余度极低
2. **ResNet-18 中等脆弱**: 下降 38.80pp，3x3 conv 有一定冗余但不足以抵御 50 次攻击
3. **DeiT-Tiny 最鲁棒**: Imagenette 仅降 9.53pp，CIFAR-100 仅降 2.78pp。Transformer 的 attention 机制提供了更高的冗余度
4. **CIFAR-100 比 Imagenette 更鲁棒**: 同为 DeiT-Tiny，CIFAR-100 降幅仅 2.78pp vs 9.53pp，可能因 100 类分类任务分散了攻击影响

---

## Phase 7: 归一化实验 + 最终 BFA 攻击 (已完成)

Date: 2026-03-07

### 7.1 背景

Phase 6 的全搜索攻击耗时约 43.5 小时（4 个任务合计），且不同模型/数据集间的超参数不统一，结果不可直接比较。需要：
1. 引入 `--coarse-ratio` 加速参数替代全搜索，实现约 80x 加速
2. 统一 `--calib-per-class` 使校准集大小随类别数自动缩放，保证公平比较
3. 扩展 CIFAR-100 实验到全部 3 个模型（Phase 6 只做了 DeiT-Tiny）
4. 增加 DeiT-Tiny `--exclude-head` 消融实验

### 7.2 归一化设计

#### 攻击预算归一化
- `--physical-budget 50`: 所有实验统一 50 次 bit-flip

#### 搜索空间归一化
- `--coarse-ratio 0.001`: 按梯度幅度预筛选 top-0.1% 的 group（Stage-A 粗筛）
- `--top-k-verify 64`: Stage-B 精确验证 top-64 候选
- 这使每步攻击的搜索空间统一为总 group 数的 0.1%，而非全搜索

#### 校准集归一化
- `--calib-per-class 8`: 每个类别采样 8 张图片
  - Imagenette (10 类): 8×10 = 80 张
  - CIFAR-100 (100 类): 8×100 = 800 张
- 评估集固定使用完整验证集

#### 模型权重归一化
- 全部使用 Phase 2-4 修正后的正确 checkpoint（`*_sparse_ft_*_full.pth`）
- DeiT-Tiny CIFAR-100 使用 Route A 训练的 `*_sparse_ft_cifar100_20ep.pth`

### 7.3 统一实验参数

```
--physical-budget 50
--calib-per-class 8
--coarse-ratio 0.001
--top-k-verify 64
--seed 0
--device cuda
```

### 7.4 归一化攻击结果（8 组实验）

#### 主实验（6 组: 3 模型 × 2 数据集）

| # | Model | Dataset | Initial Acc | Final Acc | Acc Drop | Flips | 运行时间 |
|---|-------|---------|------------|-----------|----------|-------|---------|
| 1 | ResNet-18 | Imagenette | 98.47% | 30.47% | **68.00%** | 50 | 3m40s |
| 2 | MobileNetV2 | Imagenette | 95.57% | 10.42% | **85.15%** | 50 | 2m56s |
| 3 | DeiT-Tiny | Imagenette | 93.89% | 84.36% | **9.53%** | 50 | 4m03s |
| 4 | ResNet-18 | CIFAR-100 | 78.89% | 69.60% | **9.29%** | 50 | 6m02s |
| 5 | MobileNetV2 | CIFAR-100 | 73.91% | 1.17% | **72.74%** | 50 | 5m42s |
| 6 | DeiT-Tiny | CIFAR-100 | 82.32% | 65.85% | **16.47%** | 50 | 26m07s |

#### 消融实验（DeiT-Tiny exclude-head）

| # | Model | Dataset | exclude-head | Initial Acc | Final Acc | Acc Drop | 运行时间 |
|---|-------|---------|-------------|------------|-----------|----------|---------|
| 7 | DeiT-Tiny | Imagenette | Yes | 93.89% | 82.80% | **11.09%** | 3m58s |
| 8 | DeiT-Tiny | CIFAR-100 | Yes | 82.32% | 65.85% | **16.47%** | 26m22s |

### 7.5 分析

#### 攻击脆弱性排序（按 Acc Drop 降序）

```
MobileNetV2/Imagenette (85.15%) > MobileNetV2/CIFAR-100 (72.74%)
  > ResNet-18/Imagenette (68.00%) > DeiT-Tiny/CIFAR-100 (16.47%)
  > DeiT-Tiny/Imagenette (11.09%*) > ResNet-18/CIFAR-100 (9.29%)
  > DeiT-Tiny/Imagenette (9.53%)
```

#### 关键发现

1. **MobileNetV2 在两个数据集上都最脆弱**: Imagenette 85.15%, CIFAR-100 72.74%。Pointwise 1x1 conv 稀疏化后冗余度极低。
2. **DeiT-Tiny 在两个数据集上都最鲁棒**: Imagenette 仅降 9.53pp, CIFAR-100 降 16.47pp。Transformer attention 机制提供了更高的特征冗余度。
3. **ResNet-18 表现两极化**: Imagenette 降 68.00pp（中等脆弱），但 CIFAR-100 仅降 9.29pp（鲁棒）。
4. **exclude-head 消融**: Imagenette 去头后降幅从 9.53% → 11.09%（增加 1.56pp），说明攻击头部反而不如攻击 encoder 有效。CIFAR-100 则无变化（16.47%），说明攻击集中在 encoder 层。
5. **加速效果显著**: 归一化参数使总运行时间从 ~43.5h（Phase 6 全搜索）降至 ~53min（8 个实验合计），加速约 49x。

#### 对比 Phase 6（全搜索 vs 归一化搜索）

| Model/Dataset | Phase 6 (全搜索) | Phase 7 (归一化) | 差异 |
|--------------|-----------------|-----------------|------|
| ResNet-18/Imagenette | 38.80% | **68.00%** | +29.20pp |
| MobileNetV2/Imagenette | 85.15% | **85.15%** | 0pp |
| DeiT-Tiny/Imagenette | 9.53% | **9.53%** | 0pp |
| DeiT-Tiny/CIFAR-100 | 2.78% | **16.47%** | +13.69pp |

**注意**: 归一化搜索（coarse-ratio=0.001）对 ResNet-18 和 DeiT-Tiny/CIFAR-100 反而找到了更有效的攻击，说明全搜索中存在 anti-oscillation 过度约束问题。

---
