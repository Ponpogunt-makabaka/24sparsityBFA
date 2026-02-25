"""
Evaluation and analysis functions for BFA attack results.
"""

import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np


def evaluate_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda'
) -> float:
    """
    Compute top-1 accuracy on test set.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to use

    Returns:
        Top-1 accuracy as percentage
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


def evaluate_loss(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: str = 'cuda'
) -> float:
    """
    Compute average loss on test set.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        criterion: Loss function
        device: Device to use

    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total += labels.size(0)

    return total_loss / total


def compute_accuracy_degradation(original_acc: float, attacked_acc: float) -> Dict[str, float]:
    """
    Compute accuracy degradation metrics.

    Args:
        original_acc: Original model accuracy (%)
        attacked_acc: Attacked model accuracy (%)

    Returns:
        Dictionary with degradation metrics
    """
    return {
        'original_accuracy': original_acc,
        'attacked_accuracy': attacked_acc,
        'accuracy_drop': original_acc - attacked_acc,
        'retained_accuracy': 100.0 * attacked_acc / original_acc if original_acc > 0 else 0.0,
    }


def generate_attack_report(
    history: Dict,
    flip_summary: Optional[Dict] = None,
    save_path: Optional[str] = None
) -> str:
    """
    Generate a text report of the attack results.

    Args:
        history: Attack history dictionary from ProgressiveBitSearch
        flip_summary: Optional summary of flipped bits by layer
        save_path: Optional path to save the report

    Returns:
        Report text
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Bit-Flip Attack Report")
    lines.append("=" * 60)
    lines.append("")

    # Summary statistics
    if history['accuracy']:
        lines.append("Attack Summary:")
        lines.append(f"  Initial accuracy: {history['accuracy'][0]:.2f}%")
        lines.append(f"  Final accuracy: {history['accuracy'][-1]:.2f}%")
        lines.append(f"  Accuracy drop: {history['accuracy'][0] - history['accuracy'][-1]:.2f}%")
        lines.append(f"  Initial loss: {history['loss'][0]:.4f}")
        lines.append(f"  Final loss: {history['loss'][-1]:.4f}")
        lines.append(f"  Total flips: {history['flips'][-1]}")
        lines.append(f"  Total rounds: {history['rounds'][-1]}")
        lines.append("")

    # Per-round statistics
        lines.append("Per-Round Statistics:")
        lines.append(f"{'Round':<6} {'Flips':<10} {'Accuracy':<12} {'Loss':<12}")
        lines.append("-" * 42)
        for i in range(len(history['rounds'])):
            lines.append(
                f"{history['rounds'][i]:<6} "
                f"{history['flips'][i]:<10} "
                f"{history['accuracy'][i]:<12.2f} "
                f"{history['loss'][i]:<12.4f}"
            )
        lines.append("")

    # Bit flip distribution
    if flip_summary:
        lines.append("Bit Flip Distribution by Layer:")
        for layer_name, bit_counts in flip_summary.items():
            total_flips = sum(bit_counts.values())
            lines.append(f"  {layer_name}: {total_flips} flips")

            # Show distribution by bit position
            sign_flips = sum(bit_counts.get(b, 0) for b in [31])
            exp_flips = sum(bit_counts.get(b, 0) for b in range(23, 31))
            mant_flips = sum(bit_counts.get(b, 0) for b in range(23))

            lines.append(f"    Sign bit: {sign_flips}")
            lines.append(f"    Exponent bits: {exp_flips}")
            lines.append(f"    Mantissa bits: {mant_flips}")
        lines.append("")

    lines.append("=" * 60)

    report = "\n".join(lines)

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {save_path}")

    return report


def compare_bit_distribution(
    original_weights: Dict[str, torch.Tensor],
    attacked_weights: Dict[str, torch.Tensor]
) -> Dict[str, Dict[int, int]]:
    """
    Analyze bit-level changes between original and attacked weights.

    Args:
        original_weights: Dict mapping layer names to weight tensors
        attacked_weights: Dict mapping layer names to attacked weight tensors

    Returns:
        Dictionary mapping layer_name → {bit_position: flip_count}
    """
    from bfa.utils_attack import float_to_bits

    flip_distribution = {}

    for layer_name in original_weights:
        if layer_name not in attacked_weights:
            continue

        original = original_weights[layer_name].flatten().cpu().numpy()
        attacked = attacked_weights[layer_name].flatten().cpu().numpy()

        if original.shape != attacked.shape:
            continue

        flip_distribution[layer_name] = {bit: 0 for bit in range(32)}

        for i in range(len(original)):
            orig_bits = float_to_bits(original[i])
            att_bits = float_to_bits(attacked[i])

            # Count differing bits
            diff = orig_bits ^ att_bits
            for bit_pos in range(32):
                if (diff >> bit_pos) & 1:
                    flip_distribution[layer_name][bit_pos] += 1

    return flip_distribution


def load_attack_log(log_path: str) -> Dict:
    """
    Load attack log from pickle file.

    Args:
        log_path: Path to attack log

    Returns:
        Dictionary with history and metadata
    """
    with open(log_path, 'rb') as f:
        return pickle.load(f)


def get_attack_metrics(log_path: str) -> Dict[str, float]:
    """
    Get key attack metrics from log file.

    Args:
        log_path: Path to attack log

    Returns:
        Dictionary with metrics
    """
    log = load_attack_log(log_path)
    history = log['history']

    metrics = {
        'initial_accuracy': history['accuracy'][0] if history['accuracy'] else 0.0,
        'final_accuracy': history['accuracy'][-1] if history['accuracy'] else 0.0,
        'accuracy_drop': (history['accuracy'][0] - history['accuracy'][-1]) if history['accuracy'] else 0.0,
        'total_flips': history['flips'][-1] if history['flips'] else 0,
        'total_rounds': history['rounds'][-1] if history['rounds'] else 0,
        'initial_loss': history['loss'][0] if history['loss'] else 0.0,
        'final_loss': history['loss'][-1] if history['loss'] else 0.0,
    }

    return metrics


def plot_attack_history(
    history: Dict,
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plot attack history (accuracy and loss vs flips).

    Args:
        history: Attack history dictionary
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy vs flips
    ax1.plot(history['flips'], history['accuracy'], 'b-', linewidth=2)
    ax1.set_xlabel('Cumulative Bit Flips')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy vs Bit Flips')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=10, color='r', linestyle='--', label='Target (10%)')
    ax1.legend()

    # Loss vs flips
    ax2.plot(history['flips'], history['loss'], 'r-', linewidth=2)
    ax2.set_xlabel('Cumulative Bit Flips')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss vs Bit Flips')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_bit_flip_heatmap(
    flip_summary: Dict[str, Dict[int, int]],
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plot a heatmap of bit flips by layer and bit position.

    Args:
        flip_summary: Dictionary mapping layer_name → {bit_pos: count}
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    import matplotlib.pyplot as plt

    # Organize data for heatmap
    layers = list(flip_summary.keys())
    bit_positions = list(range(32))

    # Create matrix
    data = np.zeros((len(layers), 32))
    for i, layer in enumerate(layers):
        for bit_pos in bit_positions:
            data[i, bit_pos] = flip_summary[layer].get(bit_pos, 0)

    # Plot
    fig, ax = plt.subplots(figsize=(14, max(6, len(layers) * 0.5)))

    im = ax.imshow(data, aspect='auto', cmap='YlOrRd')

    # Set ticks
    ax.set_xticks(bit_positions)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)

    # Add bit region labels
    ax.axvline(x=-0.5, color='white', linewidth=2)
    ax.axvline(x=22.5, color='white', linewidth=2)
    ax.axvline(x=30.5, color='white', linewidth=2)

    ax.text(11, -1, 'Mantissa', ha='center', va='bottom')
    ax.text(26.5, -1, 'Exponent', ha='center', va='bottom')
    ax.text(31, -1, 'S', ha='center', va='bottom')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Flips')

    ax.set_xlabel('Bit Position')
    ax.set_ylabel('Layer')
    ax.set_title('Bit Flip Distribution by Layer and Position')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    # Test evaluation functions
    print("Evaluation module loaded successfully")
    print("Available functions:")
    print("  - evaluate_accuracy()")
    print("  - evaluate_loss()")
    print("  - compute_accuracy_degradation()")
    print("  - generate_attack_report()")
    print("  - plot_attack_history()")
    print("  - plot_bit_flip_heatmap()")
