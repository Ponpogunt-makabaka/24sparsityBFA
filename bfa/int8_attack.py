#!/usr/bin/env python3
"""
Int8 Bit-Flip Attack (BFA) Engine.

Implements Progressive Bit Search (PBS) for Int8 quantized models.

Int8 Format (two's complement):
- Bit 7: Sign bit (0=positive, 1=negative)
- Bits 0-6: Magnitude (0-127, with bit 6 = 64)

Attack strategy:
- Flip bits based on gradient sensitivity
- Score = |grad| × |Δvalue| where Δvalue is the change from bit flip
- EXCLUDE already-flipped bits (history_mask)
- EXCLUDE zero weights in sparse models (sparsity_mask)
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Set, Callable
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class Int8AttackResult:
    """Result of an Int8 BFA attack."""
    initial_accuracy: float
    final_accuracy: float
    total_flips: int
    rounds: int
    accuracy_history: List[float]
    loss_history: List[float]
    bit_positions: List[Tuple[str, int, int]]  # (layer_name, weight_idx, bit_pos)
    detailed_log: List[Dict]


class Int8BFA:
    """
    Int8 Bit-Flip Attack engine.

    Supports Int8 quantized models created by ptq_convert.py.

    Key features:
    - history_mask: Prevents re-flipping the same bit
    - sparsity_mask: Only targets effective (non-zero) weights in sparse models
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the Int8 BFA engine.

        Args:
            model: Int8 quantized PyTorch model
            device: Device to run on
        """
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
        """
        Flip a specific bit in the Int8 weight storage.

        Int8 format (two's complement):
        - Bit 7: Sign bit
        - Bits 0-6: Magnitude

        Args:
            layer_name: Name of the layer
            module: The module containing int8_weights
            weight_idx: Flat index of the weight
            bit_pos: Bit position to flip (0-7)

        Returns:
            True if successful
        """
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

    def compute_int8_sensitivity(
        self,
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        num_samples: int = 100,
        flipped_bits: Optional[Set[Tuple[str, int, int]]] = None,
        weight_filter: Optional[Callable[[int], bool]] = None
    ) -> List[Tuple[float, str, nn.Module, int, int]]:
        """
        Compute sensitivity score for each Int8 bit.

        Score = grad × Δvalue (first-order loss increase)

        Where Δvalue is the signed change in dequantized value when the bit is flipped.

        IMPORTANT:
        - Excludes bits already in flipped_bits (history_mask)
        - Excludes zero weights in sparse models (sparsity_mask)

        Args:
            data_loader: Data loader for calibration samples
            criterion: Loss function
            num_samples: Number of samples for gradient computation
            flipped_bits: Set of (layer_name, weight_idx, bit_pos) to exclude
            weight_filter: Optional filter on int8 value (True=keep)

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
                # If no gradient, skip this layer
                continue

            # Get Int8 weights and scale
            int8_weights = module.int8_weights
            scale = module.scale.item()

            # Flatten for iteration
            int8_flat = int8_weights.flatten()
            grad_flat = grad.flatten()

            # For sparse models, get sparsity mask
            is_sparse = hasattr(module, 'sparse_mask') and module.sparse_mask is not None
            if is_sparse:
                mask_flat = module.sparse_mask.flatten()
                # SPARSITY_MASK: Zero out gradients for pruned weights
                grad_flat = grad_flat * mask_flat
            else:
                mask_flat = None

            for idx in range(int8_flat.numel()):
                # SPARSITY_MASK: Skip sparse zero weights
                if is_sparse and mask_flat is not None:
                    if mask_flat[idx].item() < 0.5:
                        continue  # This weight is pruned, skip it

                int8_val = int8_flat[idx].item()
                if weight_filter is not None and not weight_filter(int8_val):
                    continue
                grad_val = grad_flat[idx].item()

                # Skip if gradient is zero
                if abs(grad_val) < 1e-10:
                    continue

                # Compute sensitivity for each bit
                for bit_pos in range(8):
                    # HISTORY_MASK: Skip already-flipped bits
                    bit_key = (layer_name, idx, bit_pos)
                    if bit_key in flipped_bits:
                        continue  # This bit was already flipped, skip it

                    # Compute signed value change from flipping this bit
                    delta_value = self._compute_int8_delta(int8_val, bit_pos, scale)
                    score = grad_val * delta_value

                    # Keep only flips that increase loss (positive first-order change)
                    if score > 0:
                        sensitivity_scores.append((score, layer_name, module, idx, bit_pos))

            # Clear gradients
            if module.weight.grad is not None:
                module.weight.grad = None

        # Sort by sensitivity (descending)
        sensitivity_scores.sort(key=lambda x: x[0], reverse=True)
        return sensitivity_scores

    def _compute_int8_delta(self, int8_val: int, bit_pos: int, scale: float) -> float:
        """
        Compute the change in dequantized value when flipping a bit.

        Args:
            int8_val: Current Int8 value
            bit_pos: Bit position to flip (0-7)
            scale: Quantization scale

        Returns:
            Absolute change in dequantized value
        """
        # Flip in uint8 space to ensure correct sign-bit behavior
        flipped_val = self._flip_int8_value(int8_val, bit_pos)

        # Compute dequantized values
        original_fp32 = int8_val * scale
        flipped_fp32 = flipped_val * scale

        return flipped_fp32 - original_fp32

    def progressive_bit_search(
        self,
        test_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        calib_loader: Optional[torch.utils.data.DataLoader] = None,
        max_flips: int = 50,
        target_accuracy: float = 0.1,
        log_interval: int = 1,
        calib_samples: int = 100,
        weight_filter: Optional[Callable[[int], bool]] = None
    ) -> Int8AttackResult:
        """
        Run Progressive Bit Search (PBS) attack on Int8 model.

        Args:
            test_loader: Data loader for evaluation
            criterion: Loss function
            calib_loader: Data loader for sensitivity computation
            max_flips: Maximum number of bit flips
            target_accuracy: Stop when accuracy drops below this
            log_interval: Log interval in iterations
            calib_samples: Number of samples for sensitivity computation

        Returns:
            Int8AttackResult with attack statistics
        """
        if calib_loader is None:
            calib_loader = test_loader

        # Reset flipped_bits history
        self.flipped_bits = set()

        # Initial evaluation
        initial_acc, initial_loss = self._evaluate(test_loader, criterion)
        print(f"[Int8 BFA] Initial accuracy: {initial_acc:.2f}%, loss: {initial_loss:.4f}")
        print(f"[Int8 BFA] {'Flips':>8} | {'Accuracy':>10} | {'Loss':>10} | {'Flipped Bit (layer:idx:bit)'}")
        print(f"[Int8 BFA] {'-'*8}-+-{'-'*11}-+-{'-'*11}-+-{'-'*60}")

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

        with tqdm(total=max_flips, desc="Int8 BFA Progress") as pbar:
            while total_flips < max_flips:
                # Compute sensitivities (passing flipped_bits for history_mask)
                sensitivities = self.compute_int8_sensitivity(
                    calib_loader, criterion, calib_samples, self.flipped_bits, weight_filter
                )

                if not sensitivities:
                    consecutive_no_bits += 1
                    if consecutive_no_bits >= max_consecutive:
                        print(f"[Int8 BFA] No more sensitive bits found after {consecutive_no_bits} attempts")
                        break
                    continue

                consecutive_no_bits = 0

                # Get top bit to flip (only 1 bit per iteration for PBS)
                score, layer_name, module, weight_idx, bit_pos = sensitivities[0]

                # Verify this bit hasn't been flipped yet
                bit_key = (layer_name, weight_idx, bit_pos)
                if bit_key in self.flipped_bits:
                    print(f"[Int8 BFA] WARNING: Bit {bit_key} was already flipped! Skipping.")
                    continue

                # Flip the bit
                if self.flip_int8_bit(layer_name, module, weight_idx, bit_pos):
                    bit_positions.append((layer_name, weight_idx, bit_pos))
                    total_flips += 1
                    round_num += 1

                    # Record flipped bit
                    round_bit_flips = [{
                        'layer': layer_name,
                        'idx': weight_idx,
                        'bit': bit_pos,
                        'score': score
                    }]

                # Evaluate
                current_acc, current_loss = self._evaluate(test_loader, criterion)
                accuracy_history.append(current_acc)
                loss_history.append(current_loss)

                # Log at specified intervals
                if total_flips % log_interval == 0:
                    layer_short = layer_name.split('.')[-1]
                    bit_summary = f"{layer_short}:{weight_idx}:{bit_pos}"
                    print(f"[Int8 BFA] {total_flips:8d} | {current_acc:9.2f}% | {current_loss:10.4f} | {bit_summary}")

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
                    print(f"[Int8 BFA] Target accuracy reached: {current_acc:.2f}%")
                    break

        # Final evaluation
        final_acc, final_loss = self._evaluate(test_loader, criterion)
        print(f"[Int8 BFA] {'-'*8}-+-{'-'*11}-+-{'-'*11}-+-{'-'*60}")
        print(f"[Int8 BFA] Final accuracy: {final_acc:.2f}%, loss: {final_loss:.4f} after {total_flips} flips")
        print(f"[Int8 BFA] Total unique bits flipped: {len(self.flipped_bits)}")

        return Int8AttackResult(
            initial_accuracy=initial_acc,
            final_accuracy=final_acc,
            total_flips=total_flips,
            rounds=round_num,
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


def run_int8_bfa_attack(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    max_flips: int = 50,
    target_accuracy: float = 0.1,
    calib_samples: int = 100,
    log_interval: int = 1,
    save_path: Optional[str] = None,
    save_log_path: Optional[str] = None,
    model_type: str = 'int8',
    weight_filter: Optional[Callable[[int], bool]] = None
) -> Int8AttackResult:
    """
    Convenience function to run an Int8 BFA attack.

    Args:
        model: PyTorch Int8 quantized model
        test_loader: Test data loader
        max_flips: Maximum number of bit flips
        target_accuracy: Target accuracy to stop
        calib_samples: Calibration samples
        log_interval: Log interval
        save_path: Path to save results (pkl)
        save_log_path: Path to save detailed log (txt)
        model_type: Model type identifier for logs

    Returns:
        Int8AttackResult object
    """
    import pickle

    criterion = nn.CrossEntropyLoss()

    bfa = Int8BFA(model)
    result = bfa.progressive_bit_search(
        test_loader=test_loader,
        criterion=criterion,
        max_flips=max_flips,
        target_accuracy=target_accuracy,
        calib_samples=calib_samples,
        log_interval=log_interval,
        weight_filter=weight_filter
    )

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"[Int8 BFA] Results saved to {save_path}")

    if save_log_path:
        with open(save_log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("Int8 BFA Attack Detailed Log\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Initial Accuracy: {result.initial_accuracy:.2f}%\n")
            f.write(f"Final Accuracy: {result.final_accuracy:.2f}%\n")
            f.write(f"Total Flips: {result.total_flips}\n")
            f.write(f"Total Unique Bits: {len(result.bit_positions)}\n")
            f.write(f"Rounds: {result.rounds}\n\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Flips':>8} | {'Accuracy':>10} | {'Loss':>10} | {'Flipped Bit (layer:idx:bit)'}\n")
            f.write("-" * 100 + "\n")

            for entry in result.detailed_log:
                flips = entry['flips']
                acc = entry['accuracy']
                loss = entry['loss']
                bit_flips = entry['bit_flips']

                if bit_flips:
                    b = bit_flips[0]
                    bit_summary = f"{b['layer'].split('.')[-1]}:{b['idx']}:{b['bit']}"
                else:
                    bit_summary = "(initial state)"

                f.write(f"{flips:8d} | {acc:9.2f}% | {loss:10.4f} | {bit_summary}\n")

            f.write("-" * 100 + "\n")

            # Write all flipped bits
            f.write("\nAll Flipped Bits:\n")
            f.write("-" * 100 + "\n")
            for i, (layer, idx, bit) in enumerate(result.bit_positions, 1):
                f.write(f"  {i:3d}. {layer.split('.')[-1]}:{idx}:{bit}\n")

        print(f"[Int8 BFA] Detailed log saved to {save_log_path}")

    return result


if __name__ == '__main__':
    print("Int8 BFA Engine")
    print("This module provides Int8 BFA functionality for quantized models.")
