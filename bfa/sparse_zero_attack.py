#!/usr/bin/env python3
"""
Task 2: Sparse Zero vs Non-Zero Weight Attack

Compares two attack scenarios on Sparse Int8 models:
- Scenario A (Attack Non-Zeros): Only flip bits in non-zero weights (w ≠ 0)
- Scenario B (Attack Zeros): Only flip bits in zero weights (w = 0)

This tests whether value corruption (A) or sparsity structure destruction (B)
is more critical for model stability.
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class SparseAttackResult:
    """Result of a sparse zero vs non-zero attack."""
    initial_accuracy: float
    final_accuracy: float
    total_flips: int
    scenario: str  # 'nonzero' or 'zero'
    accuracy_history: List[float]
    loss_history: List[float]
    bit_positions: List[Tuple[str, int, int, bool]]  # (layer, idx, bit, was_zero)
    detailed_log: List[Dict]


class SparseZeroAttack:
    """
    Sparse Int8 BFA engine with zero vs non-zero targeting.

    Scenarios:
    - Scenario A: Attack only non-zero weights (value corruption)
    - Scenario B: Attack only zero weights (sparsity structure destruction)

    Key features:
    - history_mask: Prevents re-flipping the same bit
    - sparsity_mask: Only targets effective (non-zero) weights when targeting non-zeros
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize the Sparse Zero Attack engine."""
        self.model = model.to(device)
        self.device = device
        self.original_weights = {}
        self.attack_log = []

        # History mask: set of (layer_name, weight_idx, bit_pos) already flipped
        self.flipped_bits: Set[Tuple[str, int, int]] = set()

        # Store original Int8 weights
        self._store_original_int8_weights()

    def _store_original_int8_weights(self):
        """Store original Int8 weights for reference."""
        for name, module in self.model.named_modules():
            if hasattr(module, 'int8_weights'):
                self.original_weights[name] = module.int8_weights.clone()

    def _get_quantized_layers(self) -> List[Tuple[str, nn.Module]]:
        """Get list of layers with Int8 weights to attack."""
        layers = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'int8_weights') and hasattr(module, 'scale'):
                layers.append((name, module))
        return layers

    def flip_int8_bit(
        self,
        layer_name: str,
        module: nn.Module,
        weight_idx: int,
        bit_pos: int
    ) -> bool:
        """Flip a specific bit in the Int8 weight storage."""
        with torch.no_grad():
            int8_weights_flat = module.int8_weights.flatten()
            current_val = int8_weights_flat[weight_idx].item()

            # Flip in uint8 space to avoid Python's negative int XOR semantics
            new_val = self._flip_int8_value(current_val, bit_pos)

            int8_weights_flat[weight_idx] = torch.tensor(new_val, dtype=torch.int8)

            # Add to flipped_bits history
            self.flipped_bits.add((layer_name, weight_idx, bit_pos))
        return True

    @staticmethod
    def _flip_int8_value(int8_val: int, bit_pos: int) -> int:
        """Flip a bit in a signed int8 value using proper 8-bit two's complement."""
        u8 = int8_val & 0xFF
        u8 ^= (1 << bit_pos)
        return u8 - 256 if u8 >= 128 else u8

    def is_zero_weight(self, module: nn.Module, weight_idx: int) -> bool:
        """
        Check if a weight is zero in the sparse representation.

        For 2:4 sparse models, a weight is zero if:
        1. The int8_weights value is 0, OR
        2. The sparse_mask is 0 for that position
        """
        int8_weights_flat = module.int8_weights.flatten()

        # Check sparse mask if available
        if hasattr(module, 'sparse_mask') and module.sparse_mask is not None:
            mask_flat = module.sparse_mask.flatten()
            return mask_flat[weight_idx].item() < 0.5

        # Otherwise check if int8 value is zero
        return int8_weights_flat[weight_idx].item() == 0

    def compute_targeted_sensitivity(
        self,
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        num_samples: int = 100,
        target_zeros: bool = False,
        flipped_bits: Optional[Set[Tuple[str, int, int]]] = None
    ) -> List[Tuple[float, str, nn.Module, int, int]]:
        """
        Compute sensitivity score for targeted weight type (zero or non-zero).

        IMPORTANT:
        - Excludes bits already in flipped_bits (history_mask)
        - For sparse models targeting non-zeros: zeros out gradients for pruned weights (sparsity_mask)

        Args:
            data_loader: Data loader for calibration samples
            criterion: Loss function
            num_samples: Number of samples for gradient computation
            target_zeros: If True, target zero weights. If False, target non-zero weights.
            flipped_bits: Set of (layer_name, weight_idx, bit_pos) to exclude

        Returns:
            List of (score, layer_name, module, weight_idx, bit_pos) sorted by score
        """
        if flipped_bits is None:
            flipped_bits = self.flipped_bits

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

        # Get quantized layers
        layers = self._get_quantized_layers()

        for layer_name, module in layers:
            # Forward pass
            outputs = self.model(calib_inputs)
            loss = criterion(outputs, calib_targets)

            # Backward pass
            self.model.zero_grad()
            loss.backward()

            # Get gradient from the dequantized weight
            if module.weight.grad is not None:
                grad = module.weight.grad.data
            else:
                continue

            # Get Int8 weights and scale
            int8_weights = module.int8_weights
            scale = module.scale.item()

            # Flatten for iteration
            int8_flat = int8_weights.flatten()
            grad_flat = grad.flatten()

            # Get sparse mask if available
            is_sparse = hasattr(module, 'sparse_mask') and module.sparse_mask is not None
            if is_sparse:
                mask_flat = module.sparse_mask.flatten()
                # SPARSITY_MASK: Zero out gradients for pruned weights
                # This ensures we only target effective (non-zero) weights
                if not target_zeros:
                    grad_flat = grad_flat * mask_flat
            else:
                mask_flat = None

            for idx in range(int8_flat.numel()):
                # Check if this is a zero or non-zero weight
                if is_sparse and mask_flat is not None:
                    is_zero = mask_flat[idx].item() < 0.5
                else:
                    is_zero = int8_flat[idx].item() == 0

                # Filter based on target_zeros parameter
                if target_zeros and not is_zero:
                    continue  # Skip non-zero weights when targeting zeros
                if not target_zeros and is_zero:
                    continue  # Skip zero weights when targeting non-zeros

                grad_val = grad_flat[idx].item()

                # Skip if gradient is zero
                if abs(grad_val) < 1e-10:
                    continue

                int8_val = int8_flat[idx].item()

                # Compute sensitivity for each bit (signed loss increase)
                for bit_pos in range(8):
                    # HISTORY_MASK: Skip already-flipped bits
                    bit_key = (layer_name, idx, bit_pos)
                    if bit_key in flipped_bits:
                        continue  # This bit was already flipped, skip it

                    # For zero weights, some bits have no effect
                    if is_zero:
                        # For zero weights, signed delta is the flipped value * scale
                        delta_value = self._compute_int8_delta_for_zero(bit_pos, scale)
                    else:
                        delta_value = self._compute_int8_delta(int8_val, bit_pos, scale)

                    score = grad_val * delta_value
                    if score > 0:
                        sensitivity_scores.append((score, layer_name, module, idx, bit_pos))

            # Clear gradients
            if module.weight.grad is not None:
                module.weight.grad = None

        # Sort by sensitivity (descending)
        sensitivity_scores.sort(key=lambda x: x[0], reverse=True)
        return sensitivity_scores

    def _compute_int8_delta(self, int8_val: int, bit_pos: int, scale: float) -> float:
        """Compute the signed change in dequantized value when flipping a bit."""
        # Flip in uint8 space to ensure correct sign-bit behavior
        flipped_val = self._flip_int8_value(int8_val, bit_pos)

        # Compute dequantized values
        original_fp32 = int8_val * scale
        flipped_fp32 = flipped_val * scale

        return flipped_fp32 - original_fp32

    def _compute_int8_delta_for_zero(self, bit_pos: int, scale: float) -> float:
        """
        Compute the signed change when flipping a bit of a zero weight.

        For weight value = 0:
        - Flipping bit 7 (sign): 0 → 128 (which is -128 in signed) → Δ = |128| × scale
        - Flipping bit k (magnitude): 0 → 2^k → Δ = 2^k × scale
        """
        if bit_pos == 7:
            # 0 → 128 → -128 in signed int8
            flipped_val = -128
        else:
            flipped_val = (1 << bit_pos)

        return flipped_val * scale

    def targeted_progressive_search(
        self,
        test_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        calib_loader: Optional[torch.utils.data.DataLoader] = None,
        max_flips: int = 50,
        target_accuracy: float = 0.1,
        log_interval: int = 1,
        calib_samples: int = 100,
        target_zeros: bool = False,
        scenario_name: str = 'unknown'
    ) -> SparseAttackResult:
        """
        Run Progressive Bit Search attack targeting zero or non-zero weights.

        Args:
            test_loader: Data loader for evaluation
            criterion: Loss function
            calib_loader: Data loader for sensitivity computation
            max_flips: Maximum number of bit flips
            target_accuracy: Stop when accuracy drops below this
            log_interval: Log interval in iterations
            calib_samples: Number of samples for sensitivity computation
            target_zeros: If True, attack zero weights. If False, attack non-zero weights.
            scenario_name: Name of the scenario ('nonzero' or 'zero')

        Returns:
            SparseAttackResult with attack statistics
        """
        if calib_loader is None:
            calib_loader = test_loader

        # Reset flipped_bits history
        self.flipped_bits = set()

        # Initial evaluation
        initial_acc, initial_loss = self._evaluate(test_loader, criterion)
        target_type = "Zero Weights" if target_zeros else "Non-Zero Weights"
        print(f"[Sparse {scenario_name}] Initial accuracy: {initial_acc:.2f}%, loss: {initial_loss:.4f}")
        print(f"[Sparse {scenario_name}] Targeting: {target_type}")
        print(f"[Sparse {scenario_name}] {'Flips':>8} | {'Accuracy':>10} | {'Loss':>10} | {'Flipped Bit'}")
        print(f"[Sparse {scenario_name}] {'-'*8}-+-{'-'*11}-+-{'-'*11}-+-{'-'*60}")

        # Record initial state
        detailed_log = [{
            'flips': 0,
            'accuracy': initial_acc,
            'loss': initial_loss,
            'bit_flips': []
        }]

        accuracy_history = [initial_acc]
        loss_history = [initial_loss]
        bit_positions = []

        total_flips = 0
        round_num = 0
        consecutive_no_bits = 0
        max_consecutive = 5  # Stop if no bits found this many times

        with tqdm(total=max_flips, desc=f"{scenario_name} Attack") as pbar:
            while total_flips < max_flips:
                # Compute sensitivities (passing flipped_bits for history_mask)
                sensitivities = self.compute_targeted_sensitivity(
                    calib_loader, criterion, calib_samples, target_zeros=target_zeros,
                    flipped_bits=self.flipped_bits
                )

                if not sensitivities:
                    consecutive_no_bits += 1
                    if consecutive_no_bits >= max_consecutive:
                        print(f"[Sparse {scenario_name}] No more sensitive bits found after {consecutive_no_bits} attempts")
                        break
                    continue

                consecutive_no_bits = 0

                # Get top bit to flip
                score, layer_name, module, weight_idx, bit_pos = sensitivities[0]

                # Verify this bit hasn't been flipped yet
                bit_key = (layer_name, weight_idx, bit_pos)
                if bit_key in self.flipped_bits:
                    print(f"[Sparse {scenario_name}] WARNING: Bit {bit_key} was already flipped! Skipping.")
                    continue

                # Check if the weight matches our target
                is_zero = self.is_zero_weight(module, weight_idx)
                if target_zeros and not is_zero:
                    continue
                if not target_zeros and is_zero:
                    continue

                # Flip the bit
                if self.flip_int8_bit(layer_name, module, weight_idx, bit_pos):
                    bit_positions.append((layer_name, weight_idx, bit_pos, is_zero))
                    total_flips += 1
                    round_num += 1

                    # Record flipped bit
                    round_bit_flips = [{
                        'layer': layer_name,
                        'idx': weight_idx,
                        'bit': bit_pos,
                        'score': score,
                        'was_zero': is_zero
                    }]

                # Evaluate
                current_acc, current_loss = self._evaluate(test_loader, criterion)
                accuracy_history.append(current_acc)
                loss_history.append(current_loss)

                # Log at specified intervals
                if total_flips % log_interval == 0:
                    layer_short = layer_name.split('.')[-1]
                    zero_indicator = "Z" if is_zero else "N"
                    bit_summary = f"{layer_short}:{weight_idx}:{bit_pos}({zero_indicator})"
                    print(f"[Sparse {scenario_name}] {total_flips:8d} | {current_acc:9.2f}% | {current_loss:10.4f} | {bit_summary}")

                    detailed_log.append({
                        'flips': total_flips,
                        'accuracy': current_acc,
                        'loss': current_loss,
                        'bit_flips': round_bit_flips
                    })

                pbar.update(1)
                pbar.set_postfix({
                    'acc': f'{current_acc:.1f}%',
                    'flips': total_flips,
                    'history_size': len(self.flipped_bits)
                })

                # Check if target reached
                if current_acc < target_accuracy * 100:
                    print(f"[Sparse {scenario_name}] Target accuracy reached: {current_acc:.2f}%")
                    break

        # Final evaluation
        final_acc, final_loss = self._evaluate(test_loader, criterion)
        print(f"[Sparse {scenario_name}] {'-'*8}-+-{'-'*11}-+-{'-'*11}-+-{'-'*60}")
        print(f"[Sparse {scenario_name}] Final accuracy: {final_acc:.2f}%, loss: {final_loss:.4f} after {total_flips} flips")
        print(f"[Sparse {scenario_name}] Total unique bits flipped: {len(self.flipped_bits)}")

        return SparseAttackResult(
            initial_accuracy=initial_acc,
            final_accuracy=final_acc,
            total_flips=total_flips,
            scenario=scenario_name,
            accuracy_history=accuracy_history,
            loss_history=loss_history,
            bit_positions=bit_positions,
            detailed_log=detailed_log
        )

    def _evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Evaluate model accuracy and loss."""
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


def run_sparse_zero_attack(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    scenario: str = 'nonzero',  # 'nonzero' or 'zero'
    max_flips: int = 50,
    target_accuracy: float = 0.1,
    calib_samples: int = 100,
    log_interval: int = 1,
    save_path: Optional[str] = None,
    save_log_path: Optional[str] = None
) -> SparseAttackResult:
    """
    Convenience function to run a sparse zero vs non-zero attack.

    Args:
        model: Sparse Int8 PyTorch model
        test_loader: Test data loader
        scenario: 'nonzero' (attack non-zero weights) or 'zero' (attack zero weights)
        max_flips: Maximum number of bit flips
        target_accuracy: Target accuracy to stop
        calib_samples: Calibration samples
        log_interval: Log interval
        save_path: Path to save results (pkl)
        save_log_path: Path to save detailed log (txt)

    Returns:
        SparseAttackResult object
    """
    import pickle

    criterion = nn.CrossEntropyLoss()

    attack = SparseZeroAttack(model)

    target_zeros = (scenario == 'zero')
    scenario_name = "Zero-Attack" if target_zeros else "NonZero-Attack"

    result = attack.targeted_progressive_search(
        test_loader=test_loader,
        criterion=criterion,
        max_flips=max_flips,
        target_accuracy=target_accuracy,
        calib_samples=calib_samples,
        log_interval=log_interval,
        target_zeros=target_zeros,
        scenario_name=scenario_name
    )

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"[Task 2] Results saved to {save_path}")

    if save_log_path:
        with open(save_log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(f"Task 2: Sparse Zero vs Non-Zero Attack Log\n")
            f.write(f"Scenario: {scenario_name}\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Initial Accuracy: {result.initial_accuracy:.2f}%\n")
            f.write(f"Final Accuracy: {result.final_accuracy:.2f}%\n")
            f.write(f"Total Flips: {result.total_flips}\n")
            f.write(f"Target Type: {'Zero Weights' if target_zeros else 'Non-Zero Weights'}\n\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Flips':>8} | {'Accuracy':>10} | {'Loss':>10} | {'Flipped Bit'}\n")
            f.write("-" * 100 + "\n")

            for entry in result.detailed_log:
                flips = entry['flips']
                acc = entry['accuracy']
                loss = entry['loss']
                bit_flips = entry['bit_flips']

                if bit_flips:
                    b = bit_flips[0]
                    zero_ind = "Z" if b['was_zero'] else "N"
                    bit_summary = f"{b['layer'].split('.')[-1]}:{b['idx']}:{b['bit']}({zero_ind})"
                else:
                    bit_summary = "(initial state)"

                f.write(f"{flips:8d} | {acc:9.2f}% | {loss:10.4f} | {bit_summary}\n")

            f.write("-" * 100 + "\n")
        print(f"[Task 2] Detailed log saved to {save_log_path}")

    return result


if __name__ == '__main__':
    print("Sparse Zero vs Non-Zero Attack Engine")
    print("This module provides targeted attack functionality for sparse Int8 models.")
