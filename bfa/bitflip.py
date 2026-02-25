"""
Core BFA sensitivity computation.
Computes gradient-based sensitivity for each bit in model weights.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import heapq

from bfa.utils_attack import flip_bit, compute_bit_flip_magnitude


class BitFlipAttack:
    """
    Bit-Flip Attack using gradient-based sensitivity computation.

    The attack computes the sensitivity of each bit in the model weights
    by estimating how much the loss would change if that bit were flipped.
    """

    def __init__(self, model: nn.Module, device: str = 'cuda',
                 bit_width: int = 8):
        """
        Args:
            model: The model to attack
            device: Device to use ('cuda' or 'cpu')
            bit_width: Quantization bit width (for reference)
        """
        self.model = model
        self.device = device
        self.bit_width = bit_width
        self.flipped_bits = []  # Track flipped bits: [(layer_name, idx, bit_pos), ...]

        # Original loss for sensitivity computation
        self.baseline_loss = None
        self.baseline_acc = None

    def compute_gradient_sensitivity(
        self,
        data_loader,
        num_samples: int = 1000,
        criterion: nn.Module = None
    ) -> Dict[Tuple[str, int, int], float]:
        """
        Compute sensitivity for each bit using gradient information.

        For each weight parameter w and bit position b:
            sensitivity[b] ≈ |∂L/∂w| × |∂w/∂b|

        Where |∂w/∂b| is the magnitude change from flipping bit b.

        Args:
            data_loader: Data loader for calibration samples
            num_samples: Maximum number of samples to use
            criterion: Loss function (default: CrossEntropyLoss)

        Returns:
            Dictionary mapping (layer_name, weight_idx, bit_pos) → sensitivity
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.model.eval()

        # Collect gradients for all weights
        sensitivities = {}
        sample_count = 0

        print("Computing gradients on calibration set...")
        for images, labels in data_loader:
            if sample_count >= num_samples:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            self.model.zero_grad()
            loss.backward()

            # Compute sensitivity for each parameter
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.grad is not None:
                    grad = param.grad.data

                    # For each element in the weight tensor
                    for idx in range(param.numel()):
                        grad_magnitude = abs(grad.flatten()[idx].item())

                        # For each bit position
                        for bit_pos in range(32):
                            # Sensitivity = |grad| × |bit_flip_magnitude|
                            bit_magnitude = compute_bit_flip_magnitude(bit_pos)

                            # For mantissa bits, the actual magnitude depends on the weight value
                            # For sign/exponent bits, use the pre-computed magnitude
                            if bit_pos < 23:  # Mantissa
                                # The actual impact depends on the exponent (weight scale)
                                weight_val = param.flatten()[idx].item()
                                if abs(weight_val) > 1e-8:
                                    import math
                                    exponent = math.floor(math.log2(abs(weight_val)))
                                    actual_magnitude = grad_magnitude * (2 ** (bit_pos - 23)) * (2 ** exponent)
                                else:
                                    actual_magnitude = 0.0
                            else:
                                actual_magnitude = grad_magnitude * bit_magnitude

                            key = (name, idx, bit_pos)
                            sensitivities[key] = sensitivities.get(key, 0.0) + actual_magnitude

            sample_count += images.size(0)

        # Average over samples
        for key in sensitivities:
            sensitivities[key] /= min(sample_count, num_samples)

        print(f"Computed sensitivities for {len(sensitivities)} bits")
        return sensitivities

    def rank_bits_by_sensitivity(
        self,
        sensitivities: Dict[Tuple[str, int, int], float],
        top_k: Optional[int] = None
    ) -> List[Tuple[Tuple[str, int, int], float]]:
        """
        Rank bits by their sensitivity (descending order).

        Args:
            sensitivities: Dictionary of bit → sensitivity mapping
            top_k: If specified, return only top-k bits

        Returns:
            List of ((layer_name, idx, bit_pos), sensitivity) sorted by sensitivity
        """
        # Sort by sensitivity (descending)
        sorted_bits = sorted(
            sensitivities.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if top_k is not None:
            sorted_bits = sorted_bits[:top_k]

        return sorted_bits

    def flip_bits(self, bits_to_flip: List[Tuple[str, int, int]]):
        """
        Apply bit flips to model weights.

        Args:
            bits_to_flip: List of (layer_name, weight_idx, bit_pos) to flip
        """
        # Group flips by layer for efficiency
        layer_flips = {}
        for layer_name, weight_idx, bit_pos in bits_to_flip:
            if layer_name not in layer_flips:
                layer_flips[layer_name] = []
            layer_flips[layer_name].append((weight_idx, bit_pos))

        # Apply flips
        for name, param in self.model.named_parameters():
            if name in layer_flips:
                param_flat = param.data.view(-1)

                for weight_idx, bit_pos in layer_flips[name]:
                    original_value = param_flat[weight_idx].item()
                    flipped_value = flip_bit(original_value, bit_pos)
                    param_flat[weight_idx] = flipped_value

                    # Track flipped bit
                    self.flipped_bits.append((name, weight_idx, bit_pos))

                param.data = param_flat.view(param.data.shape)

    def compute_baseline(
        self,
        data_loader,
        num_samples: int = 1000,
        criterion: nn.Module = None
    ):
        """
        Compute baseline loss and accuracy on calibration set.

        Args:
            data_loader: Data loader for calibration samples
            num_samples: Maximum number of samples to use
            criterion: Loss function
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        sample_count = 0

        with torch.no_grad():
            for images, labels in data_loader:
                if sample_count >= num_samples:
                    break

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                sample_count += images.size(0)

        self.baseline_loss = total_loss / sample_count
        self.baseline_acc = 100.0 * correct / total

        print(f"Baseline - Loss: {self.baseline_loss:.4f}, Acc: {self.baseline_acc:.2f}%")

    def estimate_loss_change(
        self,
        bits_to_flip: List[Tuple[str, int, int]],
        data_loader,
        num_samples: int = 100,
        criterion: nn.Module = None
    ) -> float:
        """
        Estimate the loss change from flipping specific bits.

        This temporarily flips the bits, computes the loss, then flips them back.

        Args:
            bits_to_flip: List of bits to flip
            data_loader: Data loader for evaluation
            num_samples: Number of samples to evaluate on
            criterion: Loss function

        Returns:
            Estimated change in loss (positive = loss increases)
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Save original weights
        original_weights = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                original_weights[name] = param.data.clone()

        # Flip bits
        self.flip_bits(bits_to_flip)

        # Compute new loss
        self.model.eval()
        total_loss = 0.0
        sample_count = 0

        with torch.no_grad():
            for images, labels in data_loader:
                if sample_count >= num_samples:
                    break

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                sample_count += images.size(0)

        new_loss = total_loss / sample_count

        # Restore original weights
        for name, param in self.model.named_parameters():
            if name in original_weights:
                param.data.copy_(original_weights[name])

        # Clear flipped bits tracking since we restored
        self.flipped_bits = [b for b in self.flipped_bits if b not in bits_to_flip]

        if self.baseline_loss is not None:
            return new_loss - self.baseline_loss
        return new_loss

    def get_flipped_bits_count(self) -> int:
        """Return total number of flipped bits."""
        return len(self.flipped_bits)

    def save_flipped_model(self, path: str):
        """Save the attacked model checkpoint."""
        torch.save(self.model.state_dict(), path)
        print(f"Attacked model saved to: {path}")

    def get_bit_flip_summary(self) -> Dict[str, Dict[int, int]]:
        """
        Get a summary of flipped bits by layer and bit position.

        Returns:
            Dict mapping layer_name → {bit_pos: count}
        """
        summary = {}
        for layer_name, weight_idx, bit_pos in self.flipped_bits:
            if layer_name not in summary:
                summary[layer_name] = {}
            summary[layer_name][bit_pos] = summary[layer_name].get(bit_pos, 0) + 1
        return summary


if __name__ == '__main__':
    # Test the BitFlipAttack class
    import sys
    sys.path.append('..')
    from models.resnet20 import resnet20
    from models.quantized_model import QuantizedResNet
    from train.train_utils import get_cifar10_loaders

    print("Testing BitFlipAttack...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create a small model for testing
    base_model = resnet20()
    model = QuantizedResNet(base_model, bit_width=8).to(device)

    # Get data
    _, test_loader = get_cifar10_loaders(batch_size=32)

    # Create attack instance
    bfa = BitFlipAttack(model, device=device)

    # Compute baseline
    bfa.compute_baseline(test_loader, num_samples=100)

    # Compute sensitivities on a small sample
    print("Computing sensitivities...")
    sensitivities = bfa.compute_gradient_sensitivity(test_loader, num_samples=100)

    # Get top 10 bits
    top_bits = bfa.rank_bits_by_sensitivity(sensitivities, top_k=10)
    print(f"\nTop 10 most sensitive bits:")
    for (layer_name, idx, bit_pos), sensitivity in top_bits:
        print(f"  {layer_name}[{idx}], bit {bit_pos}: {sensitivity:.6f}")

    print("\nBitFlipAttack test completed!")
