#!/usr/bin/env python3
"""
Task 3: Encoded Sparse Model Attack (Position vs MSB)

Implements BFA attacks on CSR-encoded sparse models:
- Vector A: Attack Value MSB (weight value corruption)
- Vector B: Attack Index bits (position corruption causing wrong activation access)

This tests whether shifting a weight to wrong position (index flip) or
negating the weight value (MSB flip) causes faster collapse.
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class EncodedAttackResult:
    """Result of an encoded sparse attack."""
    initial_accuracy: float
    final_accuracy: float
    total_flips: int
    attack_type: str  # 'value_msb' or 'index_position'
    accuracy_history: List[float]
    loss_history: List[float]
    bit_positions: List[Tuple[str, int, int]]  # (layer, csr_idx, bit_pos)
    detailed_log: List[Dict]


class CSREncodedAttack:
    """
    BFA engine for CSR-encoded sparse models.

    Attack vectors:
    - Vector A (Value MSB): Target bit 7 of values array (sign flip)
    - Vector B (Position): Target bits of column_indices array
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize the CSR Encoded Attack engine."""
        self.model = model.to(device)
        self.device = device
        self.original_weights = {}
        self.attack_log = []
        # History mask to avoid re-flipping the same bit
        self.flipped_bits = set()

    def _get_csr_layers(self) -> List[Tuple[str, nn.Module]]:
        """Get list of CSR-encoded layers."""
        layers = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'csr_values') and len(module.csr_values) > 0:
                layers.append((name, module))
        return layers

    def flip_csr_value_bit(
        self,
        layer_name: str,
        module: nn.Module,
        csr_idx: int,
        bit_pos: int
    ) -> bool:
        """Flip a bit in the CSR value array."""
        with torch.no_grad():
            if csr_idx < len(module.csr_values):
                current_val = module.csr_values[csr_idx].item()
                new_val = self._flip_int8_value(current_val, bit_pos)

                module.csr_values[csr_idx] = torch.tensor(new_val, dtype=torch.int8)
                return True
        return False

    def flip_csr_index_bit(
        self,
        layer_name: str,
        module: nn.Module,
        csr_idx: int,
        bit_pos: int
    ) -> bool:
        """Flip a bit in the CSR column_indices array."""
        with torch.no_grad():
            if csr_idx < len(module.csr_column_indices):
                in_features = module.weight.shape[1] * module.weight.shape[2] * module.weight.shape[3]
                current_val = module.csr_column_indices[csr_idx].item()
                new_val = simulate_csr_index_attack(current_val, bit_pos, in_features)

                module.csr_column_indices[csr_idx] = torch.tensor(new_val, dtype=torch.int16)
                return True
        return False

    def compute_csr_sensitivity(
        self,
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        num_samples: int = 100,
        attack_indices: bool = False,
        flipped_bits: Optional[set] = None
    ) -> List[Tuple[float, str, nn.Module, int, int]]:
        """
        Compute sensitivity scores for CSR-encoded model.

        Args:
            data_loader: Data loader for calibration samples
            criterion: Loss function
            num_samples: Number of samples for gradient computation
            attack_indices: If True, attack indices. If False, attack values.

        Returns:
            List of (score, layer_name, module, csr_idx, bit_pos) sorted by score
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

        # Get CSR layers
        layers = self._get_csr_layers()

        # Performance note:
        # The original implementation recomputed forward/backward once per CSR layer,
        # which is extremely slow on CPU (and unnecessary). Gradients do not depend
        # on which layer we later read, so compute them once and reuse.
        outputs = self.model(calib_inputs)
        loss = criterion(outputs, calib_targets)
        self.model.zero_grad()
        loss.backward()

        for layer_name, module in layers:

            # Get gradient from the dequantized weight
            if module.weight.grad is not None:
                grad = module.weight.grad.data
            else:
                continue

            # Get scale
            scale = module.scale.item()

            # Reconstruct dense gradient
            out_channels = module.weight.shape[0]
            in_features = module.weight.shape[1] * module.weight.shape[2] * module.weight.shape[3]
            grad_2d = grad.view(out_channels, in_features)

            # For each CSR value, compute sensitivity
            for row in range(out_channels):
                start_idx = module.csr_row_ptr[row].item()
                end_idx = module.csr_row_ptr[row + 1].item()

                for csr_idx in range(int(start_idx), int(end_idx)):
                    # Get the column index
                    col = module.csr_column_indices[csr_idx].item()

                    # Get gradient at this position
                    grad_val = grad_2d[row, col].item()

                    if abs(grad_val) < 1e-10:
                        continue

                    # Get value
                    value = module.csr_values[csr_idx].item()

                    if attack_indices:
                        # Compute sensitivity for index bit flips
                        # An index flip moves the weight to a different column
                        for bit_pos in range(min(16, in_features.bit_length())):
                            bit_key = (layer_name, csr_idx, bit_pos)
                            if bit_key in flipped_bits:
                                continue
                            # Simulate index flip
                            flipped_col = simulate_csr_index_attack(col, bit_pos, in_features)
                            if 0 <= flipped_col < in_features:
                                # Estimate gradient at new position
                                new_grad_val = grad_2d[row, flipped_col].item() if flipped_col < grad_2d.shape[1] else 0
                                # Score based on gradient difference
                                score = abs(new_grad_val - grad_val) * abs(value) * scale
                                sensitivity_scores.append((score, layer_name, module, csr_idx, bit_pos))
                    else:
                        # Compute sensitivity for value bit flips (signed loss increase)
                        for bit_pos in range(8):
                            bit_key = (layer_name, csr_idx, bit_pos)
                            if bit_key in flipped_bits:
                                continue
                            delta_value = self._compute_int8_delta(value, bit_pos, scale)
                            score = grad_val * delta_value
                            if score > 0:
                                sensitivity_scores.append((score, layer_name, module, csr_idx, bit_pos))

            # Clear gradients
            if module.weight.grad is not None:
                module.weight.grad = None

        # Sort by sensitivity (descending)
        sensitivity_scores.sort(key=lambda x: x[0], reverse=True)
        return sensitivity_scores

    def _compute_int8_delta(self, int8_val: int, bit_pos: int, scale: float) -> float:
        """Compute the signed change in dequantized value when flipping a bit."""
        flipped_val = self._flip_int8_value(int8_val, bit_pos)

        original_fp32 = int8_val * scale
        flipped_fp32 = flipped_val * scale

        return flipped_fp32 - original_fp32

    @staticmethod
    def _flip_int8_value(int8_val: int, bit_pos: int) -> int:
        """Flip a bit in a signed int8 value using proper 8-bit two's complement."""
        u8 = int8_val & 0xFF
        u8 ^= (1 << bit_pos)
        return u8 - 256 if u8 >= 128 else u8

    def progressive_csr_attack(
        self,
        test_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        calib_loader: Optional[torch.utils.data.DataLoader] = None,
        max_flips: int = 50,
        target_accuracy: float = 0.1,
        log_interval: int = 1,
        calib_samples: int = 100,
        attack_indices: bool = False,
        attack_name: str = 'unknown'
    ) -> EncodedAttackResult:
        """
        Run Progressive Bit Search attack on CSR-encoded model.

        Args:
            test_loader: Data loader for evaluation
            criterion: Loss function
            calib_loader: Data loader for sensitivity computation
            max_flips: Maximum number of bit flips
            target_accuracy: Stop when accuracy drops below this
            log_interval: Log interval in iterations
            calib_samples: Number of samples for sensitivity computation
            attack_indices: If True, attack indices. If False, attack values.
            attack_name: Name of the attack

        Returns:
            EncodedAttackResult with attack statistics
        """
        if calib_loader is None:
            calib_loader = test_loader

        target_type = "Index Position" if attack_indices else "Value MSB"
        # Reset history for each run
        self.flipped_bits = set()

        # Initial evaluation
        initial_acc, initial_loss = self._evaluate(test_loader, criterion)
        print(f"[CSR {attack_name}] Initial accuracy: {initial_acc:.2f}%, loss: {initial_loss:.4f}")
        print(f"[CSR {attack_name}] Targeting: {target_type}")
        print(f"[CSR {attack_name}] {'Flips':>8} | {'Accuracy':>10} | {'Loss':>10} | {'Flipped Bit'}")
        print(f"[CSR {attack_name}] {'-'*8}-+-{'-'*11}-+-{'-'*11}-+-{'-'*60}")

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
        max_consecutive = 5

        with tqdm(total=max_flips, desc=f"{attack_name} Attack") as pbar:
            while total_flips < max_flips:
                # Compute sensitivities
                sensitivities = self.compute_csr_sensitivity(
                    calib_loader, criterion, calib_samples,
                    attack_indices=attack_indices,
                    flipped_bits=self.flipped_bits
                )

                if not sensitivities:
                    consecutive_no_bits += 1
                    if consecutive_no_bits >= max_consecutive:
                        print(f"[CSR {attack_name}] No more sensitive bits found")
                        break
                    continue

                consecutive_no_bits = 0

                # Get top bit to flip
                score, layer_name, module, csr_idx, bit_pos = sensitivities[0]

                bit_key = (layer_name, csr_idx, bit_pos)
                if bit_key in self.flipped_bits:
                    continue

                # Flip the bit
                if attack_indices:
                    success = self.flip_csr_index_bit(layer_name, module, csr_idx, bit_pos)
                else:
                    success = self.flip_csr_value_bit(layer_name, module, csr_idx, bit_pos)

                if success:
                    self.flipped_bits.add(bit_key)
                    bit_positions.append((layer_name, csr_idx, bit_pos))
                    total_flips += 1
                    round_num += 1

                    round_bit_flips = [{
                        'layer': layer_name,
                        'csr_idx': csr_idx,
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
                    bit_summary = f"{layer_short}:csr{csr_idx}:b{bit_pos}"
                    print(f"[CSR {attack_name}] {total_flips:8d} | {current_acc:9.2f}% | {current_loss:10.4f} | {bit_summary}")

                    detailed_log.append({
                        'flips': total_flips,
                        'accuracy': current_acc,
                        'loss': current_loss,
                        'bit_flips': round_bit_flips
                    })

                pbar.update(1)
                pbar.set_postfix({
                    'acc': f'{current_acc:.1f}%',
                    'flips': total_flips
                })

                # Check if target reached
                if current_acc < target_accuracy * 100:
                    print(f"[CSR {attack_name}] Target accuracy reached: {current_acc:.2f}%")
                    break

        # Final evaluation
        final_acc, final_loss = self._evaluate(test_loader, criterion)
        print(f"[CSR {attack_name}] {'-'*8}-+-{'-'*11}-+-{'-'*11}-+-{'-'*60}")
        print(f"[CSR {attack_name}] Final accuracy: {final_acc:.2f}%, loss: {final_loss:.4f} after {total_flips} flips")

        return EncodedAttackResult(
            initial_accuracy=initial_acc,
            final_accuracy=final_acc,
            total_flips=total_flips,
            attack_type=attack_name,
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


def run_csr_encoded_attack(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    attack_type: str = 'value_msb',  # 'value_msb' or 'index_position'
    max_flips: int = 50,
    target_accuracy: float = 0.1,
    calib_samples: int = 100,
    log_interval: int = 1,
    save_path: Optional[str] = None,
    save_log_path: Optional[str] = None
) -> EncodedAttackResult:
    """
    Convenience function to run a CSR-encoded sparse attack.

    Args:
        model: CSR-encoded sparse model
        test_loader: Test data loader
        attack_type: 'value_msb' (attack values) or 'index_position' (attack indices)
        max_flips: Maximum number of bit flips
        target_accuracy: Target accuracy to stop
        calib_samples: Calibration samples
        log_interval: Log interval
        save_path: Path to save results (pkl)
        save_log_path: Path to save detailed log (txt)

    Returns:
        EncodedAttackResult object
    """
    import pickle

    criterion = nn.CrossEntropyLoss()

    attack = CSREncodedAttack(model)

    attack_indices = (attack_type == 'index_position')
    attack_name = "Index-Position" if attack_indices else "Value-MSB"

    result = attack.progressive_csr_attack(
        test_loader=test_loader,
        criterion=criterion,
        max_flips=max_flips,
        target_accuracy=target_accuracy,
        calib_samples=calib_samples,
        log_interval=log_interval,
        attack_indices=attack_indices,
        attack_name=attack_name
    )

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"[Task 3] Results saved to {save_path}")

    if save_log_path:
        with open(save_log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(f"Task 3: CSR Encoded Sparse Attack Log\n")
            f.write(f"Attack Type: {attack_name}\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Initial Accuracy: {result.initial_accuracy:.2f}%\n")
            f.write(f"Final Accuracy: {result.final_accuracy:.2f}%\n")
            f.write(f"Total Flips: {result.total_flips}\n")
            f.write(f"Target Type: {'Column Indices' if attack_indices else 'Value MSB'}\n\n")
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
                    bit_summary = f"{b['layer'].split('.')[-1]}:csr{b['csr_idx']}:b{b['bit']}"
                else:
                    bit_summary = "(initial state)"

                f.write(f"{flips:8d} | {acc:9.2f}% | {loss:10.4f} | {bit_summary}\n")

            f.write("-" * 100 + "\n")
        print(f"[Task 3] Detailed log saved to {save_log_path}")

    return result


if __name__ == '__main__':
    print("CSR Encoded Sparse Attack Engine")
    print("This module provides BFA functionality for CSR-encoded sparse models.")


def simulate_csr_index_attack(col_index: int, bit_pos: int, in_features: int) -> int:
    """
    Helper to simulate CSR index bit flip.

    Args:
        col_index: Original column index
        bit_pos: Bit position to flip
        in_features: Upper bound for valid indices

    Returns:
        New column index within [0, in_features-1]
    """
    new_val = col_index ^ (1 << bit_pos)
    return max(0, min(in_features - 1, new_val))
