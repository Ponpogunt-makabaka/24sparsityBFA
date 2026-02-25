#!/usr/bin/env python3
"""
Task 4: Bitmask-based Sparse Encoding Attack

Simulates bit flips on sparse masks (bitmask encoding) while keeping values fixed.
Each flip toggles whether a weight is active (mask 1) or pruned (mask 0).
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
from tqdm import tqdm


@dataclass
class BitmaskAttackResult:
    """Result of a bitmask-based attack."""
    initial_accuracy: float
    final_accuracy: float
    total_flips: int
    accuracy_history: List[float]
    loss_history: List[float]
    bit_positions: List[Tuple[str, int]]  # (layer_name, mask_idx)
    detailed_log: List[Dict]


class BitmaskBFA:
    """
    Bitmask BFA engine for sparse Int8 models.

    Targets sparse masks (bitmask encoding) instead of weight values.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.flipped_bits: Set[Tuple[str, int]] = set()

    def _get_masked_layers(self) -> List[Tuple[str, nn.Module]]:
        """Get layers that carry a sparse mask."""
        layers = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'sparse_mask') and module.sparse_mask is not None:
                if hasattr(module, 'int8_weights') and hasattr(module, 'scale'):
                    layers.append((name, module))
        return layers

    def flip_mask_bit(self, layer_name: str, module: nn.Module, mask_idx: int) -> bool:
        """Toggle a bit in the sparse mask."""
        with torch.no_grad():
            mask_flat = module.sparse_mask.flatten()
            current_val = mask_flat[mask_idx].item()
            new_val = 1.0 - current_val
            mask_flat[mask_idx] = torch.tensor(new_val, dtype=module.sparse_mask.dtype)
            self.flipped_bits.add((layer_name, mask_idx))
        return True

    def compute_mask_sensitivity(
        self,
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        num_samples: int = 100,
        flipped_bits: Optional[Set[Tuple[str, int]]] = None
    ) -> List[Tuple[float, str, nn.Module, int]]:
        """
        Compute sensitivity scores for mask bit flips.

        Score = grad * delta (signed first-order loss increase).
        delta = (+w) when mask 0->1, or (-w) when mask 1->0.
        """
        if flipped_bits is None:
            flipped_bits = self.flipped_bits

        self.model.eval()
        sensitivity_scores = []

        # Calibration batch
        calib_data = []
        calib_targets = []
        for inputs, targets in data_loader:
            calib_data.append(inputs.to(self.device))
            calib_targets.append(targets.to(self.device))
            if len(calib_data) * inputs.size(0) >= num_samples:
                break

        calib_inputs = torch.cat(calib_data, dim=0)[:num_samples]
        calib_targets = torch.cat(calib_targets, dim=0)[:num_samples]

        # Masked layers
        layers = self._get_masked_layers()

        for layer_name, module in layers:
            outputs = self.model(calib_inputs)
            loss = criterion(outputs, calib_targets)

            self.model.zero_grad()
            loss.backward()

            if module.weight.grad is None:
                continue

            grad = module.weight.grad.data
            int8_weights = module.int8_weights
            scale = module.scale.item()
            mask = module.sparse_mask

            grad_flat = grad.flatten()
            int8_flat = int8_weights.flatten()
            mask_flat = mask.flatten()

            for idx in range(int8_flat.numel()):
                bit_key = (layer_name, idx)
                if bit_key in flipped_bits:
                    continue

                grad_val = grad_flat[idx].item()
                if abs(grad_val) < 1e-10:
                    continue

                w_val = int8_flat[idx].item() * scale
                if abs(w_val) < 1e-12:
                    continue

                mask_val = mask_flat[idx].item()
                delta = w_val if mask_val < 0.5 else -w_val
                score = grad_val * delta

                if score > 0:
                    sensitivity_scores.append((score, layer_name, module, idx))

            if module.weight.grad is not None:
                module.weight.grad = None

        sensitivity_scores.sort(key=lambda x: x[0], reverse=True)
        return sensitivity_scores

    def progressive_mask_search(
        self,
        test_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        calib_loader: Optional[torch.utils.data.DataLoader] = None,
        max_flips: int = 50,
        target_accuracy: float = 0.1,
        log_interval: int = 1,
        calib_samples: int = 100
    ) -> BitmaskAttackResult:
        """Run progressive search on mask bits."""
        if calib_loader is None:
            calib_loader = test_loader

        self.flipped_bits = set()

        initial_acc, initial_loss = self._evaluate(test_loader, criterion)
        print(f"[Bitmask BFA] Initial accuracy: {initial_acc:.2f}%, loss: {initial_loss:.4f}")
        print(f"[Bitmask BFA] {'Flips':>8} | {'Accuracy':>10} | {'Loss':>10} | {'Flipped Mask (layer:idx)'}")
        print(f"[Bitmask BFA] {'-'*8}-+-{'-'*11}-+-{'-'*11}-+-{'-'*60}")

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
        consecutive_no_bits = 0
        max_consecutive = 5

        with tqdm(total=max_flips, desc="Bitmask BFA Progress") as pbar:
            while total_flips < max_flips:
                sensitivities = self.compute_mask_sensitivity(
                    calib_loader, criterion, calib_samples, self.flipped_bits
                )

                if not sensitivities:
                    consecutive_no_bits += 1
                    if consecutive_no_bits >= max_consecutive:
                        print("[Bitmask BFA] No more sensitive mask bits found")
                        break
                    continue

                consecutive_no_bits = 0
                score, layer_name, module, mask_idx = sensitivities[0]

                bit_key = (layer_name, mask_idx)
                if bit_key in self.flipped_bits:
                    continue

                if self.flip_mask_bit(layer_name, module, mask_idx):
                    bit_positions.append((layer_name, mask_idx))
                    total_flips += 1

                    round_bit_flips = [{
                        'layer': layer_name,
                        'idx': mask_idx,
                        'score': score
                    }]

                current_acc, current_loss = self._evaluate(test_loader, criterion)
                accuracy_history.append(current_acc)
                loss_history.append(current_loss)

                if total_flips % log_interval == 0:
                    layer_short = layer_name.split('.')[-1]
                    bit_summary = f"{layer_short}:{mask_idx}"
                    print(f"[Bitmask BFA] {total_flips:8d} | {current_acc:9.2f}% | {current_loss:10.4f} | {bit_summary}")

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

                if current_acc < target_accuracy * 100:
                    print(f"[Bitmask BFA] Target accuracy reached: {current_acc:.2f}%")
                    break

        final_acc, final_loss = self._evaluate(test_loader, criterion)
        print(f"[Bitmask BFA] {'-'*8}-+-{'-'*11}-+-{'-'*11}-+-{'-'*60}")
        print(f"[Bitmask BFA] Final accuracy: {final_acc:.2f}%, loss: {final_loss:.4f} after {total_flips} flips")
        print(f"[Bitmask BFA] Total unique mask bits flipped: {len(self.flipped_bits)}")

        return BitmaskAttackResult(
            initial_accuracy=initial_acc,
            final_accuracy=final_acc,
            total_flips=total_flips,
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


def run_bitmask_attack(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    max_flips: int = 50,
    target_accuracy: float = 0.1,
    calib_samples: int = 100,
    log_interval: int = 1,
    save_path: Optional[str] = None,
    save_log_path: Optional[str] = None
) -> BitmaskAttackResult:
    """Convenience wrapper to run bitmask BFA."""
    import pickle

    criterion = nn.CrossEntropyLoss()
    attack = BitmaskBFA(model)

    result = attack.progressive_mask_search(
        test_loader=test_loader,
        criterion=criterion,
        max_flips=max_flips,
        target_accuracy=target_accuracy,
        calib_samples=calib_samples,
        log_interval=log_interval
    )

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"[Bitmask BFA] Results saved to {save_path}")

    if save_log_path:
        with open(save_log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("Bitmask BFA Attack Detailed Log\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Initial Accuracy: {result.initial_accuracy:.2f}%\n")
            f.write(f"Final Accuracy: {result.final_accuracy:.2f}%\n")
            f.write(f"Total Flips: {result.total_flips}\n")
            f.write(f"Total Unique Mask Bits: {len(result.bit_positions)}\n\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Flips':>8} | {'Accuracy':>10} | {'Loss':>10} | {'Flipped Mask (layer:idx)'}\n")
            f.write("-" * 100 + "\n")

            for entry in result.detailed_log:
                flips = entry['flips']
                acc = entry['accuracy']
                loss = entry['loss']
                bit_flips = entry['bit_flips']

                if bit_flips:
                    b = bit_flips[0]
                    bit_summary = f"{b['layer'].split('.')[-1]}:{b['idx']}"
                else:
                    bit_summary = "(initial state)"

                f.write(f"{flips:8d} | {acc:9.2f}% | {loss:10.4f} | {bit_summary}\n")

            f.write("-" * 100 + "\n\n")

            f.write("All Flipped Mask Bits:\n")
            f.write("-" * 100 + "\n")
            for i, (layer, idx) in enumerate(result.bit_positions, 1):
                f.write(f"  {i:3d}. {layer.split('.')[-1]}:{idx}\n")

        print(f"[Bitmask BFA] Detailed log saved to {save_log_path}")

    return result


if __name__ == '__main__':
    print("Bitmask BFA Engine")
    print("This module provides bitmask-targeted BFA for sparse Int8 models.")
