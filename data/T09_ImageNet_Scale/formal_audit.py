#!/usr/bin/env python3
"""
独立正式审计脚本 - Route A (T09) 实验验证

此脚本对以下内容进行严格审计：
1. 基线和数据一致性
2. 稀疏约束验证
3. 攻击轨迹和物理约束审计
4. 指标一致性检查
"""

import os
import sys
import torch
import torch.nn as nn
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# =============================================================================
# 任务 1: 基线和数据一致性验证
# =============================================================================

def extract_transforms_from_file(file_path: str) -> Dict:
    """从Python文件中提取数据转换配置"""
    result = {
        "crop_size": None,
        "resize_size": None,
        "mean": None,
        "std": None,
        "found": False
    }

    if not Path(file_path).exists():
        return result

    content = Path(file_path).read_text()

    # 查找常见的转换模式
    # 1. 查找 crop_size
    crop_matches = re.findall(r'crop_size\s*=\s*(\d+)', content)
    if crop_matches:
        result["crop_size"] = int(crop_matches[0])

    # 2. 查找 transforms 中的 Resize
    resize_matches = re.findall(r'Resize\((\d+)\)', content)
    if resize_matches:
        result["resize_size"] = int(resize_matches[0])

    # 3. 查找 CenterCrop
    centercrop_matches = re.findall(r'CenterCrop\((\d+)\)', content)
    if centercrop_matches:
        result["crop_size"] = int(centercrop_matches[0])

    # 4. 查找 normalization 均值和标准差
    mean_matches = re.findall(r'mean\s*=\s*\[([^\]]+)\]', content)
    if mean_matches:
        mean_str = mean_matches[0]
        result["mean"] = [float(x.strip()) for x in mean_str.split(',')]

    std_matches = re.findall(r'std\s*=\s*\[([^\]]+)\]', content)
    if std_matches:
        std_str = std_matches[0]
        result["std"] = [float(x.strip()) for x in std_str.split(',')]

    # 5. 查找 Normalize 参数
    normalize_matches = re.findall(r'Normalize\(\s*mean=\[([^\]]+)\]\s*,\s*std=\[([^\]]+)\]', content)
    if normalize_matches:
        mean_str, std_str = normalize_matches[0]
        result["mean"] = [float(x.strip()) for x in mean_str.split(',')]
        result["std"] = [float(x.strip()) for x in std_str.split(',')]
        result["found"] = True

    # 检查是否找到任何转换
    if any([result["crop_size"], result["resize_size"], result["mean"], result["std"]]):
        result["found"] = True

    return result


def audit_task1_baseline_consistency() -> Dict:
    """任务 1: 验证基线和数据一致性"""
    print("\n" + "="*70)
    print("任务 1: 基线和数据一致性验证")
    print("="*70)

    result = {
        "task": "Task_1_Baseline_Consistency",
        "status": "UNKNOWN",
        "findings": [],
        "details": {}
    }

    base_dir = Path("data/T09_ImageNet_Scale")

    # 1.1 检查微调脚本的转换配置
    print("\n[1.1] 检查微调脚本 (step3_sparsify_finetune.py) 的转换配置...")
    finetune_script = base_dir / "step3_sparsify_finetune.py"
    finetune_transforms = extract_transforms_from_file(str(finetune_script))

    print(f"  Crop Size: {finetune_transforms['crop_size']}")
    print(f"  Resize Size: {finetune_transforms['resize_size']}")
    print(f"  Mean: {finetune_transforms['mean']}")
    print(f"  Std: {finetune_transforms['std']}")

    result["details"]["finetune_transforms"] = finetune_transforms

    # 1.2 检查攻击脚本的转换配置
    print("\n[1.2] 检查攻击脚本的转换配置...")
    attack_script = base_dir / "engine" / "run_R1_T08_metadata_improved.py"

    if not attack_script.exists():
        # 尝试其他可能的路径
        attack_script = base_dir / "run_R1_T08_metadata_improved.py"

    attack_transforms = extract_transforms_from_file(str(attack_script))

    print(f"  Crop Size: {attack_transforms['crop_size']}")
    print(f"  Resize Size: {attack_transforms['resize_size']}")
    print(f"  Mean: {attack_transforms['mean']}")
    print(f"  Std: {attack_transforms['std']}")

    result["details"]["attack_transforms"] = attack_transforms

    # 1.3 对比转换配置
    print("\n[1.3] 对比转换配置...")

    transforms_match = True
    findings = []

    if finetune_transforms["crop_size"] != attack_transforms["crop_size"]:
        transforms_match = False
        finding = f"FAIL: Crop size 不一致 - 微调: {finetune_transforms['crop_size']}, 攻击: {attack_transforms['crop_size']}"
        findings.append(finding)
        print(f"  ✗ {finding}")
    else:
        print(f"  ✓ Crop size 一致: {finetune_transforms['crop_size']}")

    if finetune_transforms["mean"] and attack_transforms["mean"]:
        if finetune_transforms["mean"] != attack_transforms["mean"]:
            transforms_match = False
            finding = f"FAIL: Mean 不一致 - 微调: {finetune_transforms['mean']}, 攻击: {attack_transforms['mean']}"
            findings.append(finding)
            print(f"  ✗ {finding}")
        else:
            print(f"  ✓ Mean 一致: {finetune_transforms['mean']}")

    if finetune_transforms["std"] and attack_transforms["std"]:
        if finetune_transforms["std"] != attack_transforms["std"]:
            transforms_match = False
            finding = f"FAIL: Std 不一致 - 微调: {finetune_transforms['std']}, 攻击: {attack_transforms['std']}"
            findings.append(finding)
            print(f"  ✗ {finding}")
        else:
            print(f"  ✓ Std 一致: {finetune_transforms['std']}")

    result["findings"].extend(findings)

    # 1.4 检查数据集配置
    print("\n[1.4] 检查数据集配置...")

    # 查找 imagenette 配置
    config_path = base_dir / "config.py"
    if config_path.exists():
        config_content = config_path.read_text()

        # 查找类别数
        num_classes_matches = re.findall(r'num_classes\s*=\s*(\d+)', config_content)
        if num_classes_matches:
            num_classes = int(num_classes_matches[0])
            print(f"  配置中的类别数: {num_classes}")
            result["details"]["config_num_classes"] = num_classes

        # 查找 imagenette 路径
        imagenette_matches = re.findall(r'imagenette[^"]*"', config_content, re.IGNORECASE)
        if imagenette_matches:
            print(f"  Imagenette 路径配置: {imagenette_matches}")

    # 检查 JSON 日志中的验证样本数
    print("\n[1.5] 检查 JSON 日志中的验证集信息...")
    json_files = list((base_dir / "weights").glob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            if "val_samples" in data:
                print(f"  {json_file.name}: val_samples = {data['val_samples']}")
                result["details"][f"{json_file.name}_val_samples"] = data["val_samples"]

            if "num_classes" in data:
                print(f"  {json_file.name}: num_classes = {data['num_classes']}")
                result["details"][f"{json_file.name}_num_classes"] = data["num_classes"]
        except:
            pass

    result["status"] = "PASS" if transforms_match else "FAIL"

    print(f"\n[任务 1 结果] {result['status']}")
    if result["status"] == "FAIL":
        for finding in findings:
            print(f"  - {finding}")

    return result


# =============================================================================
# 任务 2: 稀疏约束验证
# =============================================================================

def verify_strict_zeros(checkpoint_path: str) -> Dict:
    """严格验证检查点中的零值"""
    result = {
        "path": checkpoint_path,
        "exists": False,
        "loadable": False,
        "max_pruned_value": 0.0,
        "max_pruned_layer": None,
        "total_pruned_elements": 0,
        "strict_zero_count": 0,
        "non_zero_pruned_count": 0,
        "layers_checked": 0,
        "layers_with_leakage": 0,
        "status": "UNKNOWN"
    }

    if not Path(checkpoint_path).exists():
        return result

    result["exists"] = True

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        return result

    result["loadable"] = True

    # 提取 state dict
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    else:
        state_dict = checkpoint

    max_pruned_val = 0.0
    max_pruned_layer = None
    total_pruned = 0
    strict_zeros = 0
    non_zero_pruned = 0

    print(f"\n    审计: {Path(checkpoint_path).name}")
    print(f"    {'Layer':<50} {'Pruned':<10} {'Max|Val|':<12} {'StrictZero':<12} {'NonZero':<10}")
    print(f"    {'-'*100}")

    for key in sorted(state_dict.keys()):
        if "weight" not in key:
            continue

        weight = state_dict[key]

        if weight.dim() < 2:
            continue

        # 检查是否有 sparse_mask
        mask_key = key.replace("weight", "sparse_mask")
        if mask_key in state_dict:
            mask = state_dict[mask_key]
            # 检查被掩码的权重值
            pruned_indices = (mask == 0)
            if pruned_indices.any():
                pruned_values = weight[pruned_indices]
                max_val = pruned_values.abs().max().item()
                num_pruned = pruned_indices.sum().item()
                num_strict_zero = (pruned_values == 0.0).sum().item()
                num_non_zero = num_pruned - num_strict_zero

                total_pruned += num_pruned
                strict_zeros += num_strict_zero
                non_zero_pruned += num_non_zero

                if max_val > max_pruned_val:
                    max_pruned_val = max_val
                    max_pruned_layer = key

                layer_status = "PASS" if max_val == 0.0 else f"LEAK({max_val:.2e})"
                print(f"    {key:<50} {num_pruned:<10} {max_val:<12.2e} {num_strict_zero:<12} {num_non_zero:<10} {layer_status}")

                result["layers_checked"] += 1
                if max_val > 0.0:
                    result["layers_with_leakage"] += 1
        else:
            # 没有 sparse_mask，尝试检查权重是否本身就是稀疏的
            # 对于 2:4 稀疏，检查是否有零值模式
            if weight.dim() == 4:
                # Conv2d: permute(0, 2, 3, 1).view(-1, 4)
                out_ch, in_ch, kh, kw = weight.shape
                if in_ch % 4 == 0:
                    w_perm = weight.permute(0, 2, 3, 1).contiguous()
                    flat = w_perm.view(-1, 4)
                    # 检查每组中的零值
                    for i in range(flat.shape[0]):
                        group = flat[i]
                        zeros = (group == 0.0).sum().item()
                        if zeros > 0:
                            zero_values = group[group == 0.0]
                            max_val = zero_values.abs().max().item()
                            if max_val > max_pruned_val:
                                max_pruned_val = max_val
                                max_pruned_layer = f"{key}_group{i}"

    result["max_pruned_value"] = max_pruned_val
    result["max_pruned_layer"] = max_pruned_layer
    result["total_pruned_elements"] = total_pruned
    result["strict_zero_count"] = strict_zeros
    result["non_zero_pruned_count"] = non_zero_pruned

    # 判断状态
    if result["layers_checked"] == 0:
        result["status"] = "NO_MASK_FOUND"
    elif max_pruned_val == 0.0:
        result["status"] = "PASS"
    elif max_pruned_val < 1e-10:
        result["status"] = "WARN_TINY_LEAK"
    else:
        result["status"] = "FAIL_GRADIENT_LEAKAGE"

    return result


def audit_task2_sparsity_constraints() -> Dict:
    """任务 2: 稀疏约束验证"""
    print("\n" + "="*70)
    print("任务 2: 稀疏约束验证 (严格零值检查)")
    print("="*70)

    result = {
        "task": "Task_2_Sparsity_Constraints",
        "status": "UNKNOWN",
        "models": {},
        "findings": []
    }

    weights_dir = Path("data/T09_ImageNet_Scale/weights")
    models = [
        ("resnet18", "resnet18_2_4_sparse_imagenette.pth"),
        ("mobilenet_v2", "mobilenet_v2_2_4_sparse_imagenette.pth"),
        ("deit_tiny", "deit_tiny_2_4_sparse_imagenette.pth"),
    ]

    overall_pass = True

    for model_name, filename in models:
        checkpoint_path = weights_dir / filename
        model_result = verify_strict_zeros(str(checkpoint_path))
        result["models"][model_name] = model_result

        print(f"\n  [{model_name.upper()}] 状态: {model_result['status']}")
        print(f"    最大掩码值: {model_result['max_pruned_value']:.2e}")
        print(f"    最差层: {model_result['max_pruned_layer']}")

        if model_result["status"] == "PASS":
            print(f"    ✓ 所有被掩码的权重严格为 0.0")
        elif model_result["status"] == "WARN_TINY_LEAK":
            print(f"    ⚠ 警告: 检测到极小泄漏值 (< 1e-10)")
            overall_pass = False
        elif model_result["status"] == "FAIL_GRADIENT_LEAKAGE":
            print(f"    ✗ 失败: 检测到梯度泄漏！")
            result["findings"].append(f"{model_name}: 梯度泄漏 - 最大值 {model_result['max_pruned_value']:.2e}")
            overall_pass = False
        else:
            print(f"    ? 未找到 sparse_mask")

    result["status"] = "PASS" if overall_pass else "FAIL"
    print(f"\n[任务 2 结果] {result['status']}")

    return result


# =============================================================================
# 任务 3: 攻击轨迹和物理约束审计
# =============================================================================

def parse_attack_results(results_dir: str) -> Dict:
    """解析攻击结果目录"""
    results_path = Path(results_dir)
    if not results_path.exists():
        return {}

    attack_results = {}

    # 查找所有 CSV 结果文件
    csv_files = list(results_path.glob("**/*T08*.csv"))
    csv_files.extend(results_path.glob("**/*attack*.csv"))
    csv_files.extend(results_path.glob("**/results.csv"))

    for csv_file in csv_files:
        # 确定模型类型
        if "resnet" in csv_file.name.lower() or "resnet" in str(csv_file).lower():
            model_type = "resnet18"
        elif "mobile" in csv_file.name.lower():
            model_type = "mobilenet_v2"
        elif "deit" in csv_file.name.lower() or "vit" in str(csv_file).lower():
            model_type = "deit_tiny"
        else:
            continue

        try:
            with open(csv_file, 'r') as f:
                content = f.read()

            # 解析 CSV 内容
            lines = content.strip().split('\n')
            headers = lines[0] if lines else ""

            attack_results[model_type] = {
                "file": str(csv_file),
                "headers": headers,
                "content": content,
                "lines": lines
            }
        except:
            pass

    # 同时检查 JSON 和日志文件
    json_files = list(results_path.glob("**/*T08*.json"))
    log_files = list(results_path.glob("**/*T08*.log"))
    log_files.extend(results_path.glob("**/*T08*.txt"))

    for log_file in log_files:
        if "resnet" in log_file.name.lower():
            model_type = "resnet18"
        elif "mobile" in log_file.name.lower():
            model_type = "mobilenet_v2"
        elif "deit" in log_file.name.lower():
            model_type = "deit_tiny"
        else:
            continue

        try:
            with open(log_file, 'r') as f:
                content = f.read()

            if model_type not in attack_results:
                attack_results[model_type] = {}

            attack_results[model_type]["log_file"] = str(log_file)
            attack_results[model_type]["log_content"] = content
        except:
            pass

    return attack_results


def audit_attack_trace(model_name: str, attack_data: Dict) -> Dict:
    """审计单个模型的攻击轨迹"""
    result = {
        "model": model_name,
        "file_found": False,
        "total_steps": 0,
        "steps_completed": 0,
        "action_types": defaultdict(int),
        "collision_errors": 0,
        "exact_50_steps": False,
        "only_index_1bit": False,
        "status": "UNKNOWN",
        "findings": []
    }

    if not attack_data:
        result["findings"].append("未找到攻击结果文件")
        return result

    result["file_found"] = True

    # 从日志内容中提取信息
    log_content = attack_data.get("log_content", "")
    csv_content = attack_data.get("content", "")

    content_to_parse = log_content if log_content else csv_content

    # 统计步数
    step_pattern = r'Step\s+(\d+)'
    steps = re.findall(step_pattern, content_to_parse)
    if steps:
        result["steps_completed"] = len(steps)
        result["total_steps"] = int(steps[-1]) if steps else 0

    # 检查是否正好 50 步
    if result["steps_completed"] == 50:
        result["exact_50_steps"] = True

    # 检查动作类型
    index_1bit_count = len(re.findall(r'action_type["\s:]+index_1bit', content_to_parse, re.IGNORECASE))
    index_1bit_count += len(re.findall(r'index_1bit\s+\|', content_to_parse, re.IGNORECASE))
    index_1bit_count += len(re.findall(r'\bindex_1bit\b', content_to_parse, re.IGNORECASE))

    weight_bit_count = len(re.findall(r'action_type["\s:]+weight_bit', content_to_parse, re.IGNORECASE))
    weight_bit_count += len(re.findall(r'weight_bit\s+\|', content_to_parse, re.IGNORECASE))

    bitmask_swap_count = len(re.findall(r'action_type["\s:]+bitmask_swap', content_to_parse, re.IGNORECASE))
    bitmask_swap_count += len(re.findall(r'bitmask_swap\s+\|', content_to_parse, re.IGNORECASE))

    result["action_types"]["index_1bit"] = index_1bit_count
    result["action_types"]["weight_bit"] = weight_bit_count
    result["action_types"]["bitmask_swap"] = bitmask_swap_count

    # 检查是否只有 index_1bit
    total_actions = sum(result["action_types"].values())
    if total_actions > 0 and result["action_types"]["index_1bit"] == total_actions:
        result["only_index_1bit"] = True

    # 检查碰撞错误
    collision_patterns = [
        r'collision',
        r'invalid.*pattern',
        r'cannot.*decode',
        r'error.*code'
    ]

    for pattern in collision_patterns:
        matches = re.findall(pattern, content_to_parse, re.IGNORECASE)
        result["collision_errors"] += len(matches)

    # 判断状态
    if result["exact_50_steps"] and result["only_index_1bit"] and result["collision_errors"] == 0:
        result["status"] = "PASS"
    elif not result["file_found"]:
        result["status"] = "NO_FILE"
    else:
        issues = []
        if not result["exact_50_steps"]:
            issues.append(f"步数不是50 (实际: {result['steps_completed']})")
        if not result["only_index_1bit"]:
            issues.append(f"动作类型不纯 (index_1bit: {result['action_types']['index_1bit']}, weight_bit: {result['action_types']['weight_bit']}, bitmask_swap: {result['action_types']['bitmask_swap']})")
        if result["collision_errors"] > 0:
            issues.append(f"发现 {result['collision_errors']} 个碰撞错误")

        result["findings"].extend(issues)
        result["status"] = "FAIL"

    return result


def audit_task3_attack_trace() -> Dict:
    """任务 3: 攻击轨迹和物理约束审计"""
    print("\n" + "="*70)
    print("任务 3: 攻击轨迹和物理约束审计")
    print("="*70)

    result = {
        "task": "Task_3_Attack_Trace",
        "status": "UNKNOWN",
        "models": {},
        "findings": []
    }

    results_dir = Path("data/T09_ImageNet_Scale/results")
    attack_data = parse_attack_results(str(results_dir))

    models = ["resnet18", "mobilenet_v2", "deit_tiny"]
    all_pass = True

    print(f"\n    检查结果目录: {results_dir}")
    print(f"    找到 {len(attack_data)} 个模型的攻击结果")

    for model_name in models:
        model_result = audit_attack_trace(model_name, attack_data.get(model_name, {}))
        result["models"][model_name] = model_result

        print(f"\n  [{model_name.upper()}]")
        print(f"    文件找到: {model_result['file_found']}")
        print(f"    完成步数: {model_result['steps_completed']}")
        print(f"    正好50步: {model_result['exact_50_steps']}")
        print(f"    动作类型: {dict(model_result['action_types'])}")
        print(f"    只有index_1bit: {model_result['only_index_1bit']}")
        print(f"    碰撞错误: {model_result['collision_errors']}")
        print(f"    状态: {model_result['status']}")

        if model_result["status"] == "PASS":
            print(f"    ✓ 通过")
        elif model_result["status"] == "NO_FILE":
            print(f"    ✗ 未找到攻击结果文件")
            all_pass = False
            result["findings"].append(f"{model_name}: 未找到攻击结果文件")
        else:
            print(f"    ✗ 失败")
            all_pass = False
            for finding in model_result["findings"]:
                result["findings"].append(f"{model_name}: {finding}")

    result["status"] = "PASS" if all_pass else "FAIL"
    print(f"\n[任务 3 结果] {result['status']}")

    return result


# =============================================================================
# 任务 4: 指标一致性检查
# =============================================================================

def audit_task4_metrics_coherence() -> Dict:
    """任务 4: 指标一致性检查"""
    print("\n" + "="*70)
    print("任务 4: 指标一致性检查")
    print("="*70)

    result = {
        "task": "Task_4_Metrics_Coherence",
        "status": "UNKNOWN",
        "models": {},
        "findings": []
    }

    weights_dir = Path("data/T09_ImageNet_Scale/weights")
    results_dir = Path("data/T09_ImageNet_Scale/results")

    models = [
        ("resnet18", "resnet18_2_4_sparse_imagenette"),
        ("mobilenet_v2", "mobilenet_v2_2_4_sparse_imagenette"),
        ("deit_tiny", "deit_tiny_2_4_sparse_imagenette"),
    ]

    all_pass = True

    for model_name, checkpoint_base in models:
        print(f"\n  [{model_name.upper()}]")

        model_result = {
            "json_acc": None,
            "log_initial_acc": None,
            "log_final_acc": None,
            "log_acc_drop": None,
            "match": False,
            "acc_drop_match": False,
            "status": "UNKNOWN"
        }

        # 4.1 读取 JSON 中的最终验证准确率
        json_file = weights_dir / f"{checkpoint_base}.json"

        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)

                json_acc = json_data.get("final_val_top1")
                if json_acc is None:
                    json_acc = json_data.get("val_top1")

                model_result["json_acc"] = json_acc
                print(f"    JSON 最终准确率: {json_acc:.2f}%" if json_acc else "    JSON 最终准确率: N/A")
            except:
                print(f"    无法读取 JSON 文件")
        else:
            print(f"    JSON 文件不存在: {json_file}")

        # 4.2 读取攻击日志中的准确率
        # 查找相关的日志文件
        log_files = list(results_dir.glob(f"**/*{checkpoint_base}*.log"))
        log_files.extend(results_dir.glob(f"**/*{checkpoint_base}*.txt"))

        initial_acc = None
        final_acc = None
        acc_drop = None

        for log_file in log_files:
            try:
                content = log_file.read_text()

                # 查找初始准确率 (Step 0)
                initial_match = re.search(r'(?:initial|baseline|step\s+0)[^\n]*?(\d+\.?\d*)%?', content, re.IGNORECASE)
                if initial_match:
                    try:
                        initial_acc = float(initial_match.group(1))
                    except:
                        pass

                # 查找最终准确率
                final_match = re.search(r'(?:final|step\s+50|last)[^\n]*?(\d+\.?\d*)%?', content, re.IGNORECASE)
                if final_match:
                    try:
                        final_acc = float(final_match.group(1))
                    except:
                        pass

                # 查找准确率下降
                drop_match = re.search(r'(?:drop|decrease)[^\n]*?(\d+\.?\d*)%?', content, re.IGNORECASE)
                if drop_match:
                    try:
                        acc_drop = float(drop_match.group(1))
                    except:
                        pass

                # 更精确的模式匹配
                acc_lines = re.findall(r'Acc[^\d]+(\d+\.\d+)%', content)
                if acc_lines:
                    if not initial_acc:
                        initial_acc = float(acc_lines[0])
                    if not final_acc:
                        final_acc = float(acc_lines[-1])

                    if len(acc_lines) >= 2:
                        acc_drop = initial_acc - final_acc

                break  # 找到一个日志文件就够了
            except:
                pass

        model_result["log_initial_acc"] = initial_acc
        model_result["log_final_acc"] = final_acc
        model_result["log_acc_drop"] = acc_drop

        print(f"    日志初始准确率: {initial_acc:.2f}%" if initial_acc else "    日志初始准确率: N/A")
        print(f"    日志最终准确率: {final_acc:.2f}%" if final_acc else "    日志最终准确率: N/A")
        print(f"    日志准确率下降: {acc_drop:.2f}%" if acc_drop else "    日志准确率下降: N/A")

        # 4.3 验证一致性
        if json_acc is not None and final_acc is not None:
            if abs(json_acc - final_acc) < 0.01:  # 允许 0.01% 的舍入误差
                model_result["match"] = True
                print(f"    ✓ JSON 和日志最终准确率匹配")
            else:
                diff = abs(json_acc - final_acc)
                print(f"    ✗ JSON 和日志最终准确率不匹配 (差异: {diff:.2f}%)")
                all_pass = False
                result["findings"].append(f"{model_name}: JSON准确率({json_acc:.2f}%) != 日志准确率({final_acc:.2f}%)")

        # 4.4 验证准确率下降计算
        if initial_acc is not None and final_acc is not None and acc_drop is not None:
            expected_drop = initial_acc - final_acc
            if abs(expected_drop - acc_drop) < 0.05:  # 允许 0.05% 的误差
                model_result["acc_drop_match"] = True
                print(f"    ✓ 准确率下降计算正确 ({initial_acc:.2f}% - {final_acc:.2f}% = {acc_drop:.2f}%)")
            else:
                print(f"    ✗ 准确率下降计算不匹配 (期望: {expected_drop:.2f}%, 实际: {acc_drop:.2f}%)")
                all_pass = False
                result["findings"].append(f"{model_name}: 准确率下降计算错误")
        elif initial_acc and final_acc:
            # 重新计算
            acc_drop = initial_acc - final_acc
            model_result["log_acc_drop"] = acc_drop
            print(f"    ✓ 准确率下降: {initial_acc:.2f}% - {final_acc:.2f}% = {acc_drop:.2f}%")
            model_result["acc_drop_match"] = True

        # 判断状态
        if model_result["match"] or (json_acc and final_acc and abs(json_acc - final_acc) < 0.1):
            model_result["status"] = "PASS"
        elif json_acc is None and final_acc is None:
            model_result["status"] = "NO_DATA"
        else:
            model_result["status"] = "FAIL"

        result["models"][model_name] = model_result

    result["status"] = "PASS" if all_pass else "FAIL"
    print(f"\n[任务 4 结果] {result['status']}")

    return result


# =============================================================================
# 主审计函数
# =============================================================================

def run_formal_audit() -> Dict:
    """运行完整的正式审计"""
    print("\n" + "="*70)
    print(" "*15 + "ROUTE A (T09) 正式审计报告")
    print("="*70)
    print("审计日期:", __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)

    results = {
        "audit_date": __import__('datetime').datetime.now().isoformat(),
        "tasks": {}
    }

    # 运行四个任务
    results["tasks"]["task1"] = audit_task1_baseline_consistency()
    results["tasks"]["task2"] = audit_task2_sparsity_constraints()
    results["tasks"]["task3"] = audit_task3_attack_trace()
    results["tasks"]["task4"] = audit_task4_metrics_coherence()

    # 生成总结
    print("\n" + "="*70)
    print("审计总结")
    print("="*70)

    task_results = [
        ("任务1: 基线和数据一致性", results["tasks"]["task1"]["status"]),
        ("任务2: 稀疏约束验证", results["tasks"]["task2"]["status"]),
        ("任务3: 攻击轨迹审计", results["tasks"]["task3"]["status"]),
        ("任务4: 指标一致性", results["tasks"]["task4"]["status"]),
    ]

    for task_name, status in task_results:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {task_name}: {status}")

    all_pass = all(t["status"] == "PASS" for t in results["tasks"].values())

    print("\n" + "="*70)
    if all_pass:
        print(" "*25 + "🟢 审计结果: 全面通过 🟢")
    else:
        fail_count = sum(1 for t in results["tasks"].values() if t["status"] == "FAIL")
        print(f" "*20 + f"🟡 审计结果: {fail_count}/4 任务未通过 🟡")
    print("="*70)

    results["overall_status"] = "PASS" if all_pass else "FAIL"

    # 保存报告
    report_path = "data/T09_ImageNet_Scale/formal_audit_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[报告已保存到: {report_path}]")

    return results


if __name__ == "__main__":
    results = run_formal_audit()
