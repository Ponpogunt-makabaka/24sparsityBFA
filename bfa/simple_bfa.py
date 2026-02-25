"""
Simplified BFA attack that works correctly.
The issue with the original implementation was that the quantized model's
forward pass was re-quantizing weights each time, which interfered with bit flips.
"""

import os
import csv
import torch
import torch.nn as nn
from tqdm import tqdm
from models.resnet20 import resnet20
from models.quantized_model import QuantizedResNet, flip_bit
from train.train_utils import get_cifar10_loaders


def simple_bfa_attack(
    model_path='./models/trained_model.pth',
    max_flips=10000,
    target_accuracy=10.0,
    bits_per_round=100,
    calibration_samples=1000,
    device='cuda',
    output_csv='./results/original_report.csv'
):
    """
    Run BFA attack and save results to CSV.

    Args:
        model_path: Path to trained model
        max_flips: Maximum bit flips
        target_accuracy: Stop when accuracy drops below this (%)
        bits_per_round: Bits to flip per round
        calibration_samples: Samples for sensitivity computation
        device: Device to use
        output_csv: Path to save CSV results
    """
    # Load model
    print(f'Loading model from: {model_path}')
    base_model = resnet20()
    model = QuantizedResNet(base_model, bit_width=8)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Get data loaders
    _, test_loader = get_cifar10_loaders(batch_size=128)

    criterion = nn.CrossEntropyLoss()

    # Initial evaluation
    print("Computing initial accuracy...")
    initial_acc, initial_loss = evaluate_model(model, test_loader, criterion, device)
    print(f"Initial - Accuracy: {initial_acc:.2f}%, Loss: {initial_loss:.4f}")

    # Save original weights (for proper quantization)
    original_weights = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            original_weights[name] = param.data.clone()

    # Storage for results
    results = [{
        'round': 0,
        'cumulative_flips': 0,
        'accuracy': initial_acc,
        'loss': initial_loss
    }]

    current_acc = initial_acc
    total_flips = 0
    round_num = 0

    # Main attack loop
    while current_acc > target_accuracy and total_flips < max_flips:
        round_num += 1

        # Compute gradients on calibration set for sensitivity
        print(f"\n--- Round {round_num} ---")
        print(f"Current flips: {total_flips}")
        print(f"Current accuracy: {current_acc:.2f}%")
        print("Computing bit sensitivities...")

        sensitivities = compute_sensitivities(model, test_loader, calibration_samples, criterion, device)

        # Select top-k bits
        top_bits = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)[:bits_per_round]

        if not top_bits:
            print("No more bits to flip!")
            break

        # Flip the bits
        print(f"Flipping {len(top_bits)} bits...")
        for (layer_name, weight_idx, bit_pos), _ in top_bits:
            # Find the parameter and flip the bit
            for name, param in model.named_parameters():
                if name == layer_name:
                    param_flat = param.data.view(-1)
                    original_value = param_flat[weight_idx].item()
                    flipped_value = flip_bit(original_value, bit_pos)
                    param_flat[weight_idx] = flipped_value
                    break
            total_flips += 1

        # Evaluate new accuracy
        current_acc, current_loss = evaluate_model(model, test_loader, criterion, device)

        # Record results
        results.append({
            'round': round_num,
            'cumulative_flips': total_flips,
            'accuracy': current_acc,
            'loss': current_loss
        })

        print(f"After round {round_num}:")
        print(f"  Cumulative flips: {total_flips}")
        print(f"  Accuracy: {current_acc:.2f}%")
        print(f"  Loss: {current_loss:.4f}")

        # Early stopping if accuracy is very low
        if current_acc < 1.0:
            print("\nAccuracy dropped below 1%. Attack successful!")
            break

    # Save results to CSV
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['round', 'cumulative_flips', 'accuracy', 'loss'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_csv}")

    # Final summary
    print("\n" + "=" * 50)
    print("Attack Complete!")
    print("=" * 50)
    print(f"Initial accuracy: {initial_acc:.2f}%")
    print(f"Final accuracy: {current_acc:.2f}%")
    print(f"Accuracy drop: {initial_acc - current_acc:.2f}%")
    print(f"Total flips: {total_flips}")
    print(f"Total rounds: {round_num}")

    return results


def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model accuracy and loss."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating', leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / total
    return accuracy, avg_loss


def compute_sensitivities(model, data_loader, num_samples, criterion, device):
    """
    Compute sensitivity for each bit using gradient information.
    Returns dict of (layer_name, weight_idx, bit_pos) -> sensitivity
    """
    model.eval()
    sensitivities = {}
    sample_count = 0

    for images, labels in data_loader:
        if sample_count >= num_samples:
            break

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Compute sensitivity for each parameter
        for name, param in model.named_parameters():
            if 'weight' in name and param.grad is not None:
                grad = param.grad.data

                # For each element in the weight tensor
                for idx in range(param.numel()):
                    grad_magnitude = abs(grad.flatten()[idx].item())

                    # For each bit position
                    for bit_pos in range(32):
                        # Sensitivity = |grad| × |bit_flip_magnitude|
                        import math
                        if bit_pos < 23:  # Mantissa
                            weight_val = param.flatten()[idx].item()
                            if abs(weight_val) > 1e-8:
                                exponent = math.floor(math.log2(abs(weight_val)))
                                actual_magnitude = grad_magnitude * (2 ** (bit_pos - 23)) * (2 ** exponent)
                            else:
                                actual_magnitude = 0.0
                        else:  # Exponent or sign
                            actual_magnitude = grad_magnitude * (2 ** (bit_pos - 23))

                        key = (name, idx, bit_pos)
                        sensitivities[key] = sensitivities.get(key, 0.0) + actual_magnitude

        sample_count += images.size(0)

    # Average over samples
    for key in sensitivities:
        sensitivities[key] /= min(sample_count, num_samples)

    print(f"Computed sensitivities for {len(sensitivities)} bits")
    return sensitivities


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    results = simple_bfa_attack()
