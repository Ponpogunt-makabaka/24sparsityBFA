"""
FP32 Bit-Flip Attack (BFA) Engine for Dense and Sparse Models.

This module implements the Progressive Bit Search (PBS) algorithm for FP32 models,
supporting both dense and sparse (2:4) models with multiple attack modes.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm

from bfa.utils_attack import (
    float_to_bits, bits_to_float, flip_bit,
    IEEE754_RANGES, compute_bit_flip_magnitude
)


@dataclass
class AttackResult:
    """Result of a BFA attack."""
    initial_accuracy: float
    final_accuracy: float
    total_flips: int
    rounds: int
    accuracy_history: List[float]
    loss_history: List[float]
    flips_per_round: List[int]
    bit_positions: List[Tuple[str, int, int]]  # (layer_name, weight_idx, bit_pos)
    detailed_log: List[Dict]  # 每 N 次翻转的详细记录


class FP32BFA:
    """
    FP32 Bit-Flip Attack engine.

    Supports:
    - Dense models (standard FP32)
    - Sparse models (2:4 structured sparsity)
    - Mode A (Dynamic Masking): Mask recalculated each forward pass
    - Mode B (Static Masking): Mask frozen before attack
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str = 'dense',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the BFA engine.

        Args:
            model: PyTorch model to attack
            mode: Attack mode
                - 'dense': Standard dense model attack
                - 'dynamic': Sparse model with dynamic masking (Mode A)
                - 'static': Sparse model with static masking (Mode B)
            device: Device to run on
        """
        self.model = model.to(device)
        self.mode = mode
        self.device = device
        self.original_weights = {}
        self.attack_log = []

        # Store original weights
        self._store_original_weights()

        # For Mode B, freeze masks before attack
        if mode == 'static':
            if hasattr(model, 'freeze_sparse_masks'):
                model.freeze_sparse_masks()
                print("[BFA] Sparse masks frozen (Mode B)")
        else:
            if hasattr(model, 'unfreeze_sparse_masks'):
                model.unfreeze_sparse_masks()
                print("[BFA] Sparse masks unfrozen (Mode A)")

    def _store_original_weights(self):
        """Store original model weights for reference."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:  # Only Conv/Linear
                self.original_weights[name] = param.data.clone()

    def _get_attackable_layers(self) -> List[Tuple[str, nn.Parameter]]:
        """
        Get list of layers to attack.

        For v3: Attack ALL layers (no exceptions).
        """
        layers = []
        for name, param in self.model.named_parameters():
            # Only attack weight tensors (not biases)
            if param.requires_grad and len(param.shape) >= 2:
                layers.append((name, param))
        return layers

    def compute_sensitivity(
        self,
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        num_samples: int = 100
    ) -> List[Tuple[float, str, int, int]]:
        """
        Compute sensitivity score for each bit.

        Score = |∇wL × (w_flipped - w_original)|

        Args:
            data_loader: Data loader for calibration samples
            criterion: Loss function
            num_samples: Number of samples to use for gradient computation

        Returns:
            List of (score, layer_name, weight_idx, bit_pos) sorted by score (descending)
        """
        self.model.eval()
        sensitivity_scores = []

        # Get calibration batch
        calib_data = []
        calib_targets = []
        for inputs, targets in data_loader:
            calib_data.append(inputs.to(self.device))
            calib_targets.append(targets.to(self.device))
            if len(calib_data) * inputs.size(0) >= num_samples:
                break

        calib_inputs = torch.cat(calib_data, dim=0)[:num_samples]
        calib_targets = torch.cat(calib_targets, dim=0)[:num_samples]

        # Compute gradients for all layers
        layers = self._get_attackable_layers()

        for layer_name, param in layers:
            # Enable gradient for this parameter
            param.requires_grad = True

            # Forward pass
            outputs = self.model(calib_inputs)
            loss = criterion(outputs, calib_targets)

            # Backward pass
            self.model.zero_grad()
            loss.backward()

            # Get gradient
            grad = param.grad.data

            # For Mode B (Static): Zero out gradients for pruned weights
            if self.mode == 'static':
                # Apply mask to gradients (only consider non-pruned weights)
                with torch.no_grad():
                    if hasattr(self.model, 'sparsity_type') and self.model.sparsity_type == "2:4":
                        # Get the sparse mask by matching parameter name
                        for module in self.model.modules():
                            if hasattr(module, 'cached_mask') and module.cached_mask is not None:
                                # Use id() for reliable comparison
                                if hasattr(module, 'weight') and id(module.weight) == id(param):
                                    mask = module.cached_mask
                                    grad = grad * mask
                                    break

            # Compute sensitivity for each weight and each bit
            weight_flat = param.data.flatten()
            grad_flat = grad.flatten()

            for idx in range(weight_flat.numel()):
                w_val = weight_flat[idx].item()
                grad_val = grad_flat[idx].item()

                # Skip if gradient is zero (no sensitivity)
                if abs(grad_val) < 1e-10:
                    continue

                # For Mode A (Dynamic): Consider all weights
                # For Mode B (Static): Skip if weight is pruned
                if self.mode == 'static' and abs(w_val) < 1e-10:
                    continue

                # Compute sensitivity for each bit
                for bit_pos in range(32):
                    w_flipped = flip_bit(w_val, bit_pos)
                    delta_w = w_flipped - w_val
                    score = grad_val * delta_w

                    # Keep only flips that increase loss (positive first-order change)
                    if score > 0:
                        sensitivity_scores.append((score, layer_name, idx, bit_pos))

            param.requires_grad = False

        # Sort by sensitivity (descending)
        sensitivity_scores.sort(key=lambda x: x[0], reverse=True)
        return sensitivity_scores

    def flip_bit_in_model(
        self,
        layer_name: str,
        weight_idx: int,
        bit_pos: int
    ) -> bool:
        """
        Flip a specific bit in the model.

        Args:
            layer_name: Name of the layer parameter
            weight_idx: Flat index of the weight
            bit_pos: Bit position to flip (0-31)

        Returns:
            True if successful, False otherwise
        """
        for name, param in self.model.named_parameters():
            if name == layer_name:
                with torch.no_grad():
                    weight_flat = param.data.flatten()
                    w_val = weight_flat[weight_idx].item()
                    w_flipped = flip_bit(w_val, bit_pos)
                    weight_flat[weight_idx] = w_flipped
                return True
        return False

    def progressive_bit_search(
        self,
        test_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        calib_loader: Optional[torch.utils.data.DataLoader] = None,
        max_flips: int = 50,
        target_accuracy: float = 0.1,
        bits_per_round: int = 1,
        calib_samples: int = 100,
        log_interval: int = 1  # 每 N 次翻转记录一次
    ) -> AttackResult:
        """
        Run Progressive Bit Search (PBS) attack.

        Following the paper specification: each iteration flips exactly 1 bit,
        gradients are recomputed each iteration.

        Args:
            test_loader: Data loader for evaluation
            criterion: Loss function
            calib_loader: Data loader for sensitivity computation
            max_flips: Maximum number of bit flips (iterations)
            target_accuracy: Stop when accuracy drops below this
            bits_per_round: Number of bits to flip per round (default: 1 for paper compliance)
            calib_samples: Number of samples for sensitivity computation
            log_interval: Log interval in iterations (default: 1 for detailed logging)

        Returns:
            AttackResult with attack statistics
        """
        if calib_loader is None:
            calib_loader = test_loader

        # Initial evaluation
        initial_acc, initial_loss = self._evaluate(test_loader, criterion)
        print(f"[BFA] Initial accuracy: {initial_acc:.2f}%, loss: {initial_loss:.4f}")
        print(f"[BFA] {'Flips':>8} | {'Accuracy':>10} | {'Loss':>10} | {'Flipped Bits (layer:idx:bit)'}")
        print(f"[BFA] {'-'*8}-+-{'-'*11}-+-{'-'*11}-+-{'-'*60}")

        # 记录初始状态
        detailed_log = [{
            'flips': 0,
            'accuracy': initial_acc,
            'loss': initial_loss,
            'bit_flips': []
        }]

        accuracy_history = [initial_acc]
        loss_history = [initial_loss]
        flips_per_round = []
        bit_positions = []

        total_flips = 0
        round_num = 0
        next_log_flip = log_interval  # 下一次记录的翻转次数

        with tqdm(total=max_flips, desc="BFA Progress") as pbar:
            while total_flips < max_flips:
                # Compute sensitivities
                sensitivities = self.compute_sensitivity(
                    calib_loader, criterion, calib_samples
                )

                if not sensitivities:
                    print("[BFA] No more sensitive bits found")
                    break

                # Get top-k bits to flip
                top_k = min(bits_per_round, len(sensitivities), max_flips - total_flips)
                top_bits = sensitivities[:top_k]

                # 记录本轮翻转的比特
                round_bit_flips = []
                for score, layer_name, weight_idx, bit_pos in top_bits:
                    if self.flip_bit_in_model(layer_name, weight_idx, bit_pos):
                        bit_positions.append((layer_name, weight_idx, bit_pos))
                        round_bit_flips.append({
                            'layer': layer_name,
                            'idx': weight_idx,
                            'bit': bit_pos,
                            'score': score
                        })
                        total_flips += 1

                flips_per_round.append(len(top_bits))
                round_num += 1

                # Evaluate
                current_acc, current_loss = self._evaluate(test_loader, criterion)
                accuracy_history.append(current_acc)
                loss_history.append(current_loss)

                # 检查是否需要记录日志
                if total_flips >= next_log_flip:
                    bit_summary = ", ".join([f"{b['layer'].split('.')[-1]}:{b['idx']}:{b['bit']}" for b in round_bit_flips[:5]])
                    if len(round_bit_flips) > 5:
                        bit_summary += f" ... (+{len(round_bit_flips)-5} more)"
                    print(f"[BFA] {total_flips:8d} | {current_acc:9.2f}% | {current_loss:10.4f} | {bit_summary}")

                    detailed_log.append({
                        'flips': total_flips,
                        'accuracy': current_acc,
                        'loss': current_loss,
                        'bit_flips': round_bit_flips
                    })
                    next_log_flip += log_interval

                pbar.update(len(top_bits))
                pbar.set_postfix({
                    'acc': f'{current_acc:.1f}%',
                    'flips': total_flips
                })

                # Check if target reached
                if current_acc < target_accuracy * 100:
                    print(f"[BFA] Target accuracy reached: {current_acc:.2f}%")
                    break

        # Final evaluation
        final_acc, final_loss = self._evaluate(test_loader, criterion)
        print(f"[BFA] {'-'*8}-+-{'-'*11}-+-{'-'*11}-+-{'-'*60}")
        print(f"[BFA] Final accuracy: {final_acc:.2f}%, loss: {final_loss:.4f} after {total_flips} flips")

        return AttackResult(
            initial_accuracy=initial_acc,
            final_accuracy=final_acc,
            total_flips=total_flips,
            rounds=round_num,
            accuracy_history=accuracy_history,
            loss_history=loss_history,
            flips_per_round=flips_per_round,
            bit_positions=bit_positions,
            detailed_log=detailed_log
        )

    def _evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        Evaluate model accuracy and loss.

        Returns:
            (accuracy, loss) tuple
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100.0 * correct / total
        avg_loss = total_loss / total
        return accuracy, avg_loss


def run_bfa_attack(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    mode: str = 'dense',
    max_flips: int = 50,
    target_accuracy: float = 0.1,
    bits_per_round: int = 1,
    calib_samples: int = 100,
    log_interval: int = 1,
    save_path: Optional[str] = None,
    save_log_path: Optional[str] = None,
    save_csv_path: Optional[str] = None,
    model_type: str = 'dense',
    attack_mode: str = 'initial'
) -> AttackResult:
    """
    Convenience function to run a BFA attack.

    Following paper specification: 1 bit per iteration, max 50 iterations.

    Args:
        model: PyTorch model to attack
        test_loader: Test data loader
        mode: Attack mode ('dense', 'dynamic', 'static')
        max_flips: Maximum number of bit flips (default: 50 for paper compliance)
        target_accuracy: Target accuracy to stop (default: 0.1 = 10%)
        bits_per_round: Bits per round (default: 1 for paper compliance)
        calib_samples: Calibration samples for gradient computation
        log_interval: Log interval in iterations (default: 1 for detailed logging)
        save_path: Path to save results (pkl)
        save_log_path: Path to save detailed log (txt)
        save_csv_path: Path to save CSV report
        model_type: Model type identifier for CSV ('dense' or 'sparse')
        attack_mode: Attack mode identifier for CSV

    Returns:
        AttackResult object
    """
    criterion = nn.CrossEntropyLoss()

    bfa = FP32BFA(model, mode=mode)
    result = bfa.progressive_bit_search(
        test_loader=test_loader,
        criterion=criterion,
        max_flips=max_flips,
        target_accuracy=target_accuracy,
        bits_per_round=bits_per_round,
        calib_samples=calib_samples,
        log_interval=log_interval
    )

    if save_path:
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"[BFA] Results saved to {save_path}")

    if save_log_path:
        # 保存详细日志到文本文件
        with open(save_log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("BFA Attack Detailed Log\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Initial Accuracy: {result.initial_accuracy:.2f}%\n")
            f.write(f"Final Accuracy: {result.final_accuracy:.2f}%\n")
            f.write(f"Total Flips: {result.total_flips}\n")
            f.write(f"Rounds: {result.rounds}\n\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Flips':>8} | {'Accuracy':>10} | {'Loss':>10} | {'Flipped Bits (layer:idx:bit)'}\n")
            f.write("-" * 100 + "\n")

            for entry in result.detailed_log:
                flips = entry['flips']
                acc = entry['accuracy']
                loss = entry['loss']
                bit_flips = entry['bit_flips']

                if bit_flips:
                    bit_summary = ", ".join([f"{b['layer'].split('.')[-1]}:{b['idx']}:{b['bit']}" for b in bit_flips[:10]])
                    if len(bit_flips) > 10:
                        bit_summary += f" ... (+{len(bit_flips)-10} more)"
                else:
                    bit_summary = "(initial state)"

                f.write(f"{flips:8d} | {acc:9.2f}% | {loss:10.4f} | {bit_summary}\n")

            f.write("-" * 100 + "\n")
        print(f"[BFA] Detailed log saved to {save_log_path}")

    if save_csv_path:
        import csv
        import os

        # 准备 CSV 数据
        csv_data = []
        for entry in result.detailed_log:
            flips = entry['flips']
            acc = entry['accuracy']
            loss = entry['loss']
            bit_flips = entry['bit_flips']

            if bit_flips:
                bit_summary = ";".join([f"{b['layer'].split('.')[-1]}:{b['idx']}:{b['bit']}" for b in bit_flips])
            else:
                bit_summary = ""

            csv_data.append({
                'flips': flips,
                'accuracy': f"{acc:.2f}",
                'loss': f"{loss:.4f}",
                'model_type': model_type,
                'attack_mode': attack_mode,
                'bit_positions': bit_summary
            })

        # 写入 CSV 文件
        file_exists = os.path.exists(save_csv_path)
        with open(save_csv_path, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['flips', 'accuracy', 'loss', 'model_type', 'attack_mode', 'bit_positions']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for row in csv_data:
                writer.writerow(row)

        print(f"[BFA] CSV report saved to {save_csv_path}")

    return result


if __name__ == '__main__':
    print("FP32 BFA Engine")
    print("This module provides the core BFA functionality.")
    print("Import and use run_bfa_attack() to run attacks.")
