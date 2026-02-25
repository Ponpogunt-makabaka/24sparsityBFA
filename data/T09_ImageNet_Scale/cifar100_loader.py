#!/usr/bin/env python3
"""
CIFAR-100 dataloader utilities for T09.

Standard 32x32 CIFAR transforms:
- Train: RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize
- Test:  Normalize only
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_cifar100_loaders(
    batch_size: int = 128,
    data_dir: str = "data/cifar100",
    num_workers: int = 0,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/test DataLoaders for CIFAR-100.
    """
    data_root = Path(data_dir)
    data_root.mkdir(parents=True, exist_ok=True)

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=str(data_root),
        train=True,
        download=download,
        transform=train_transform,
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=str(data_root),
        train=False,
        download=download,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def quick_batch_test(
    batch_size: int = 32,
    data_dir: str = "data/cifar100",
    num_workers: int = 0,
    download: bool = True,
) -> None:
    train_loader, test_loader = get_cifar100_loaders(
        batch_size=batch_size,
        data_dir=data_dir,
        num_workers=num_workers,
        download=download,
    )
    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))

    assert x_train.ndim == 4 and x_train.shape[1:] == (3, 32, 32), (
        f"Train batch shape invalid: {tuple(x_train.shape)}"
    )
    assert x_test.ndim == 4 and x_test.shape[1:] == (3, 32, 32), (
        f"Test batch shape invalid: {tuple(x_test.shape)}"
    )
    assert y_train.ndim == 1 and y_test.ndim == 1, "Labels must be 1D."
    print(
        "CIFAR100 loader quick test passed: "
        f"train={tuple(x_train.shape)}, test={tuple(x_test.shape)}"
    )


if __name__ == "__main__":
    tr, te = get_cifar100_loaders(batch_size=64, download=True)
    print(f"train_batches={len(tr)} test_batches={len(te)}")
    quick_batch_test(batch_size=32, download=True)
