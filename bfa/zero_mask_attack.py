#!/usr/bin/env python3
"""
Task 2.5: Zero-Mask Revival Attack

Only flips sparse masks from 0 -> 1 (revive pruned weights),
keeping weight values unchanged. Uses signed first-order loss increase:
score = grad * delta, where delta = w (for mask 0 -> 1).
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
from tqdm import tqdm


@dataclass
class ZeroMaskAttackResult:
    """Result of zero-mask revival attack."""
    initial_accuracy: float
    final_accuracy: float
    total_flips: int
    accuracy_history: List[float]
    loss_history: List[float]
    mask_positions: List[Tuple[str, int]]  # (layer_name, mask_idx)
    detailed_log: List[Dict]


class ZeroMaskRevivalAttack:
    """
    Attack that only flips sparse mask bits from 0 -> 1.
    This simulates "reviving" pruned weights.
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
        """Get layers that carry a sparse mask and int8 weights."""
        layers = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'sparse_mask') and module.sparse_mask is not None:
                # Ensure binary mask (0/1)
                module.sparse_mask = (module.sparse_mask > 0.5).to(module.sparse_mask.dtype)
                if hasattr(module, 'int8_weights') and hasattr(module, 'scale'):
                    layers.append((name, module))
        return layers

    def flip_zero_mask_bit(self, layer_name: str, module: nn.Module, mask_idx: int) -> bool:
        """Flip a sparse mask bit only if it is 0 (revive)."""
        with torch.no_grad():
            mask_flat = module.sparse_mask.flatten()
            current_val = int(mask_flat[mask_idx].item())
            if current_val != 0:
                return False
            # Integer/boolean flip: 0 -> 1
            new_val = current_val ^ 1
            mask_flat[mask_idx] = torch.tensor(new_val, dtype=module.sparse_mask.dtype, device=mask_flat.device)
            self.flipped_bits.add((layer_name, mask_idx))
        return True

    def compute_zero_mask_sensitivity(
        self,
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        num_samples: int = 100,
        flipped_bits: Optional[Set[Tuple[str, int]]] = None
    ) -> List[Tuple[float, str, nn.Module, int]]:
        """
        Compute sensitivity for reviving zero-mask positions.

        score = grad * delta, where delta = w (for mask 0 -> 1).
        Only considers mask positions with current mask==0.
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

                # Only revive zeros
                if int(mask_flat[idx].item()) != 0:
                    continue

                grad_val = grad_flat[idx].item()
                if abs(grad_val) < 1e-10:
                    continue

                w_val = int8_flat[idx].item() * scale
                if abs(w_val) < 1e-12:
                    continue

                score = grad_val * w_val
                if score > 0:
                    sensitivity_scores.append((score, layer_name, module, idx))

            if module.weight.grad is not None:
                module.weight.grad = None

        sensitivity_scores.sort(key=lambda x: x[0], reverse=True)
        return sensitivity_scores

    def progressive_zero_mask_search(
        self,
        test_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        calib_loader: Optional[torch.utils.data.DataLoader] = None,
        max_flips: int = 50,
        target_accuracy: float = 0.1,
        log_interval: int = 1,
        calib_samples: int = 100
    ) -> ZeroMaskAttackResult:
        """Run progressive search that only revives zero mask bits."""
        if calib_loader is None:
            calib_loader = test_loader

        self.flipped_bits = set()

        initial_acc, initial_loss = self._evaluate(test_loader, criterion)
        print(f"[ZeroMask] Initial accuracy: {initial_acc:.2f}%, loss: {initial_loss:.4f}")
        print(f"[ZeroMask] {'Flips':>8} | {'Accuracy':>10} | {'Loss':>10} | {'Revived Mask (layer:idx)'}")
        print(f"[ZeroMask] {'-'*8}-+-{'-'*11}-+-{'-'*11}-+-{'-'*60}")

        detailed_log = [{
            'flips': 0,
            'accuracy': initial_acc,
            'loss': initial_loss,
            'mask_flips': []
        }]

        accuracy_history = [initial_acc]
        loss_history = [initial_loss]
        mask_positions = []

        total_flips = 0
        consecutive_no_bits = 0
        max_consecutive = 5

        with tqdm(total=max_flips, desc="ZeroMask BFA Progress") as pbar:
            while total_flips < max_flips:
                sensitivities = self.compute_zero_mask_sensitivity(
                    calib_loader, criterion, calib_samples, self.flipped_bits
                )

                if not sensitivities:
                    consecutive_no_bits += 1
                    if consecutive_no_bits >= max_consecutive:
                        print("[ZeroMask] No more sensitive zero-mask bits found")
                        break
                    continue

                consecutive_no_bits = 0
                score, layer_name, module, mask_idx = sensitivities[0]

                if self.flip_zero_mask_bit(layer_name, module, mask_idx):
                    mask_positions.append((layer_name, mask_idx))
                    total_flips += 1

                    round_mask_flips = [{
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
                    print(f"[ZeroMask] {total_flips:8d} | {current_acc:9.2f}% | {current_loss:10.4f} | {bit_summary}")

                    detailed_log.append({
                        'flips': total_flips,
                        'accuracy': current_acc,
                        'loss': current_loss,
                        'mask_flips': round_mask_flips
                    })

                pbar.update(1)
                pbar.set_postfix({
                    'acc': f'{current_acc:.1f}%',
                    'flips': total_flips,
                    'history_size': len(self.flipped_bits)
                })

                if current_acc < target_accuracy * 100:
                    print(f"[ZeroMask] Target accuracy reached: {current_acc:.2f}%")
                    break

        final_acc, final_loss = self._evaluate(test_loader, criterion)
        print(f"[ZeroMask] {'-'*8}-+-{'-'*11}-+-{'-'*11}-+-{'-'*60}")
        print(f"[ZeroMask] Final accuracy: {final_acc:.2f}%, loss: {final_loss:.4f} after {total_flips} flips")
        print(f"[ZeroMask] Total unique revived masks: {len(self.flipped_bits)}")

        return ZeroMaskAttackResult(
            initial_accuracy=initial_acc,
            final_accuracy=final_acc,
            total_flips=total_flips,
            accuracy_history=accuracy_history,
            loss_history=loss_history,
            mask_positions=mask_positions,
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


def run_zero_mask_attack(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    max_flips: int = 50,
    target_accuracy: float = 0.1,
    calib_samples: int = 100,
    log_interval: int = 1,
    save_path: Optional[str] = None,
    save_log_path: Optional[str] = None
) -> ZeroMaskAttackResult:
    """Convenience wrapper to run zero-mask revival attack."""
    import pickle

    criterion = nn.CrossEntropyLoss()
    attack = ZeroMaskRevivalAttack(model)

    result = attack.progressive_zero_mask_search(
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
        print(f"[ZeroMask] Results saved to {save_path}")

    if save_log_path:
        with open(save_log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("Task 2.5: Zero-Mask Revival Attack Log\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Initial Accuracy: {result.initial_accuracy:.2f}%\n")
            f.write(f"Final Accuracy: {result.final_accuracy:.2f}%\n")
            f.write(f"Total Flips: {result.total_flips}\n")
            f.write(f"Total Unique Revived Masks: {len(result.mask_positions)}\n\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Flips':>8} | {'Accuracy':>10} | {'Loss':>10} | {'Revived Mask (layer:idx)'}\n")
            f.write("-" * 100 + "\n")

            for entry in result.detailed_log:
                flips = entry['flips']
                acc = entry['accuracy']
                loss = entry['loss']
                mask_flips = entry['mask_flips']

                if mask_flips:
                    b = mask_flips[0]
                    bit_summary = f"{b['layer'].split('.')[-1]}:{b['idx']}"
                else:
                    bit_summary = "(initial state)"

                f.write(f"{flips:8d} | {acc:9.2f}% | {loss:10.4f} | {bit_summary}\n")

            f.write("-" * 100 + "\n\n")
            f.write("All Revived Masks:\n")
            f.write("-" * 100 + "\n")
            for i, (layer, idx) in enumerate(result.mask_positions, 1):
                f.write(f"  {i:3d}. {layer.split('.')[-1]}:{idx}\n")

        print(f"[ZeroMask] Detailed log saved to {save_log_path}")

    return result


if __name__ == '__main__':
    print("Zero-Mask Revival Attack Engine")
