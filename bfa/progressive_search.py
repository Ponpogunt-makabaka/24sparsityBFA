"""
Progressive Bit Search (PBS) algorithm for BFA.
Implements the iterative bit-flip attack strategy.
"""

import os
import pickle
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from bfa.bitflip import BitFlipAttack


class ProgressiveBitSearch:
    """
    Progressive Bit Search (PBS) algorithm.

    The algorithm iteratively:
    1. Computes sensitivity for all bits using gradient information
    2. Selects top-k most sensitive bits
    3. Flips those bits
    4. Evaluates new accuracy/loss
    5. Repeats until accuracy drops below threshold or flip budget exhausted
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader,
        device: str = 'cuda',
        max_flips: int = 10000,
        target_accuracy: float = 0.1,
        calibration_samples: int = 1000
    ):
        """
        Args:
            model: The model to attack
            test_loader: Data loader for evaluation
            device: Device to use
            max_flips: Maximum number of bits to flip
            target_accuracy: Attack until accuracy drops below this
            calibration_samples: Number of samples for sensitivity computation
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.max_flips = max_flips
        self.target_accuracy = target_accuracy
        self.calibration_samples = calibration_samples

        # Initialize BFA helper
        self.bfa = BitFlipAttack(model, device=device)

        # Attack history
        self.history = {
            'rounds': [],
            'flips': [],
            'accuracy': [],
            'loss': [],
            'top_bit_sensitivity': [],
        }

    def evaluate_model(self, criterion: nn.Module = None) -> Tuple[float, float]:
        """
        Evaluate model accuracy and loss on test set.

        Returns:
            (accuracy, loss) tuple
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Evaluating', leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        avg_loss = total_loss / total

        return accuracy, avg_loss

    def attack(
        self,
        bits_per_round: int = 100,
        verbose: bool = True,
        criterion: nn.Module = None,
        save_interval: Optional[int] = None
    ) -> Dict[str, List]:
        """
        Run the Progressive Bit Search attack.

        Args:
            bits_per_round: Number of bits to flip each round
            verbose: Whether to print progress
            criterion: Loss function
            save_interval: Save checkpoint every N rounds (None = no saving)

        Returns:
            Dictionary with attack history
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Compute baseline
        if verbose:
            print("Computing baseline metrics...")
        initial_acc, initial_loss = self.evaluate_model(criterion)
        self.bfa.compute_baseline(self.test_loader, self.calibration_samples, criterion)

        # Record initial state
        self.history['rounds'].append(0)
        self.history['flips'].append(0)
        self.history['accuracy'].append(initial_acc)
        self.history['loss'].append(initial_loss)
        self.history['top_bit_sensitivity'].append(0.0)

        if verbose:
            print(f"\nInitial - Accuracy: {initial_acc:.2f}%, Loss: {initial_loss:.4f}")
            print(f"Target accuracy: {self.target_accuracy * 100:.2f}%")
            print(f"Max flips: {self.max_flips}")
            print(f"Bits per round: {bits_per_round}\n")

        round_num = 0
        current_acc = initial_acc

        # Main attack loop
        while current_acc > self.target_accuracy * 100 and self.bfa.get_flipped_bits_count() < self.max_flips:
            round_num += 1

            if verbose:
                print(f"\n--- Round {round_num} ---")
                print(f"Current flips: {self.bfa.get_flipped_bits_count()}")
                print(f"Current accuracy: {current_acc:.2f}%")

            # Step 1: Compute sensitivities
            if verbose:
                print("Computing bit sensitivities...")
            sensitivities = self.bfa.compute_gradient_sensitivity(
                self.test_loader,
                num_samples=self.calibration_samples,
                criterion=criterion
            )

            # Step 2: Select top-k bits
            top_bits = self.bfa.rank_bits_by_sensitivity(
                sensitivities,
                top_k=bits_per_round
            )

            if not top_bits:
                print("No more bits to flip!")
                break

            top_sensitivity = top_bits[0][1] if top_bits else 0.0

            # Step 3: Flip the bits
            if verbose:
                print(f"Flipping {len(top_bits)} bits...")

            bits_to_flip = [bit for bit, _ in top_bits]
            self.bfa.flip_bits(bits_to_flip)

            # Step 4: Evaluate new accuracy
            current_acc, current_loss = self.evaluate_model(criterion)

            # Record state
            self.history['rounds'].append(round_num)
            self.history['flips'].append(self.bfa.get_flipped_bits_count())
            self.history['accuracy'].append(current_acc)
            self.history['loss'].append(current_loss)
            self.history['top_bit_sensitivity'].append(top_sensitivity)

            if verbose:
                print(f"After round {round_num}:")
                print(f"  Cumulative flips: {self.bfa.get_flipped_bits_count()}")
                print(f"  Accuracy: {current_acc:.2f}%")
                print(f"  Loss: {current_loss:.4f}")

            # Save checkpoint if requested
            if save_interval and round_num % save_interval == 0:
                checkpoint_path = f'./results/checkpoint_round_{round_num}.pth'
                self.save_checkpoint(checkpoint_path)

            # Early stopping if accuracy is very low
            if current_acc < 1.0:
                if verbose:
                    print("\nAccuracy dropped below 1%. Attack successful!")
                break

        # Final summary
        if verbose:
            print("\n" + "=" * 50)
            print("Attack Complete!")
            print("=" * 50)
            print(f"Initial accuracy: {self.history['accuracy'][0]:.2f}%")
            print(f"Final accuracy: {self.history['accuracy'][-1]:.2f}%")
            print(f"Total flips: {self.history['flips'][-1]}")
            print(f"Total rounds: {round_num}")
            print(f"Accuracy drop: {self.history['accuracy'][0] - self.history['accuracy'][-1]:.2f}%")

        return self.history

    def save_attack_log(self, path: str):
        """
        Save attack history to a pickle file.

        Args:
            path: Path to save the log
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        log_data = {
            'history': self.history,
            'flipped_bits': self.bfa.flipped_bits,
            'flip_summary': self.bfa.get_bit_flip_summary(),
        }

        with open(path, 'wb') as f:
            pickle.dump(log_data, f)

        print(f"Attack log saved to: {path}")

    def load_attack_log(self, path: str):
        """
        Load attack history from a pickle file.

        Args:
            path: Path to the log file
        """
        with open(path, 'rb') as f:
            log_data = pickle.load(f)

        self.history = log_data['history']
        print(f"Attack log loaded from: {path}")

        return log_data

    def save_checkpoint(self, path: str):
        """
        Save model checkpoint during attack.

        Args:
            path: Path to save checkpoint
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'flipped_bits': self.bfa.flipped_bits,
            'history': self.history,
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to: {path}")

    def load_checkpoint(self, path: str):
        """
        Load model checkpoint during attack.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.bfa.flipped_bits = checkpoint['flipped_bits']
        self.history = checkpoint['history']

        print(f"Checkpoint loaded from: {path}")
        print(f"Loaded {len(self.bfa.flipped_bits)} flipped bits")

    def get_attack_summary(self) -> Dict:
        """Get a summary of the attack results."""
        if not self.history['rounds']:
            return {}

        return {
            'initial_accuracy': self.history['accuracy'][0],
            'final_accuracy': self.history['accuracy'][-1],
            'accuracy_drop': self.history['accuracy'][0] - self.history['accuracy'][-1],
            'total_flips': self.history['flips'][-1],
            'total_rounds': self.history['rounds'][-1],
            'initial_loss': self.history['loss'][0],
            'final_loss': self.history['loss'][-1],
        }


def load_and_attack(
    model_path: str,
    test_loader,
    device: str = 'cuda',
    max_flips: int = 10000,
    target_accuracy: float = 0.1,
    bits_per_round: int = 100,
    save_path: Optional[str] = None
) -> Dict:
    """
    Convenience function to load a model and run BFA attack.

    Args:
        model_path: Path to trained model checkpoint
        test_loader: Test data loader
        device: Device to use
        max_flips: Maximum bit flips
        target_accuracy: Target accuracy
        bits_per_round: Bits to flip per round
        save_path: Path to save attack log

    Returns:
        Attack history dictionary
    """
    import sys
    sys.path.append('..')
    from models.resnet20 import resnet20
    from models.quantized_model import QuantizedResNet

    # Load model
    print(f"Loading model from: {model_path}")
    base_model = resnet20()
    model = QuantizedResNet(base_model, bit_width=8)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)

    # Run attack
    pbs = ProgressiveBitSearch(
        model, test_loader, device=device,
        max_flips=max_flips,
        target_accuracy=target_accuracy
    )

    history = pbs.attack(bits_per_round=bits_per_round)

    # Save results
    if save_path:
        pbs.save_attack_log(save_path)

    return history


if __name__ == '__main__':
    # Test the ProgressiveBitSearch class
    import sys
    sys.path.append('..')
    from models.resnet20 import resnet20
    from models.quantized_model import QuantizedResNet
    from train.train_utils import get_cifar10_loaders

    print("Testing ProgressiveBitSearch...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model
    base_model = resnet20()
    model = QuantizedResNet(base_model, bit_width=8).to(device)

    # Get data
    _, test_loader = get_cifar10_loaders(batch_size=32)

    # Create PBS instance
    pbs = ProgressiveBitSearch(
        model, test_loader, device=device,
        max_flips=100,  # Small number for testing
        target_accuracy=0.1,
        calibration_samples=100
    )

    # Run a few rounds
    history = pbs.attack(bits_per_round=10, verbose=True)

    print("\nTest completed!")
    print(f"Rounds: {len(history['rounds']) - 1}")
