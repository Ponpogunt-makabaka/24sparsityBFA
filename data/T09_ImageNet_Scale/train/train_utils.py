"""
Training utilities for ResNet-20 on CIFAR-10.
"""

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Dict, Any
import json


def get_cifar10_loaders(batch_size: int = 128,
                        data_dir: str = './data',
                        num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-10 train and test data loaders.

    Args:
        batch_size: Batch size for training
        data_dir: Directory to store/load dataset
        num_workers: Number of worker processes for data loading (0 for single-process)

    Returns:
        (train_loader, test_loader) tuple
    """
    # CIFAR-10 normalization values
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)

    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Create datasets
    os.makedirs(data_dir, exist_ok=True)

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> Dict[int, float]:
    """
    Compute the accuracy over the k top predictions.

    Args:
        output: Model predictions (logits) of shape (batch_size, num_classes)
        target: Ground truth labels of shape (batch_size,)
        topk: Tuple of k values (e.g., (1, 5) for top-1 and top-5)

    Returns:
        Dictionary mapping k to accuracy
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[k] = correct_k.mul_(100.0 / batch_size).item()
        return res


def train_epoch(model: nn.Module, train_loader: DataLoader,
                criterion: nn.Module, optimizer: torch.optim.Optimizer,
                device: str = 'cuda', epoch: int = 0) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use ('cuda' or 'cpu')
        epoch: Current epoch number (for progress bar)

    Returns:
        (average_loss, average_accuracy) tuple
    """
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for images, target in pbar:
        images = images.to(device)
        target = target.to(device)

        # Forward pass
        output = model(images)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        acc = accuracy(output, target, topk=(1,))[1]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.2f}%'
        })

    return loss_meter.avg, acc_meter.avg


def validate(model: nn.Module, test_loader: DataLoader,
             criterion: nn.Module, device: str = 'cuda') -> Tuple[float, float]:
    """
    Validate the model.

    Args:
        model: Model to validate
        test_loader: Test data loader
        criterion: Loss function
        device: Device to use

    Returns:
        (average_loss, average_accuracy) tuple
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    with torch.no_grad():
        for images, target in tqdm(test_loader, desc='Validation'):
            images = images.to(device)
            target = target.to(device)

            # Forward pass
            output = model(images)
            loss = criterion(output, target)

            # Metrics
            acc = accuracy(output, target, topk=(1,))[1]
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))

    return loss_meter.avg, acc_meter.avg


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, acc: float, loss: float, path: str,
                    is_best: bool = False):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        acc: Validation accuracy
        loss: Validation loss
        path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'acc': acc,
        'loss': loss,
    }

    torch.save(checkpoint, path)

    if is_best:
        best_path = path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)


def load_checkpoint(path: str, model: nn.Module,
                    optimizer: torch.optim.Optimizer = None) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)

    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return {
        'epoch': checkpoint.get('epoch', 0),
        'acc': checkpoint.get('acc', 0.0),
        'loss': checkpoint.get('loss', float('inf')),
    }


def save_training_log(log: Dict[str, list], path: str):
    """
    Save training log to JSON file.

    Args:
        log: Dictionary with training history
        path: Path to save log
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    log_serializable = {}
    for key, value in log.items():
        if isinstance(value, list):
            log_serializable[key] = [float(v) for v in value]
        else:
            log_serializable[key] = value

    with open(path, 'w') as f:
        json.dump(log_serializable, f, indent=2)


def load_training_log(path: str) -> Dict[str, list]:
    """Load training log from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # Test data loading
    set_seed(42)

    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Test a batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
        break

    print("Data loader test passed!")
