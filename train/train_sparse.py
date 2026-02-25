#!/usr/bin/env python3
"""
Training script for Dense and Sparse ResNet-20 on CIFAR-10.

Supports:
- Dense models (standard FP32)
- Sparse models (2:4 structured sparsity)
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet20 import resnet20
from models.factory import create_resnet20
from train.train_utils import (
    get_cifar10_loaders, train_epoch, validate,
    save_checkpoint, save_training_log, set_seed
)
from config import TRAINING, DATASET, PATHS, EXPERIMENT


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ResNet-20 on CIFAR-10')

    # Dataset arguments
    parser.add_argument('--data-dir', type=str, default=PATHS['data'],
                        help='Directory to store dataset')
    parser.add_argument('--batch-size', type=int, default=TRAINING['batch_size'],
                        help='Batch size for training')

    # Model arguments
    parser.add_argument('--sparsity', type=str, default=None,
                        choices=['None', '2:4'],
                        help='Sparsity type: None (dense) or 2:4 (sparse)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=TRAINING['epochs'],
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=TRAINING['lr'],
                        help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=TRAINING['momentum'],
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=TRAINING['weight_decay'],
                        help='Weight decay')
    parser.add_argument('--milestones', type=int, nargs='+', default=TRAINING['lr_milestones'],
                        help='Learning rate decay milestones')
    parser.add_argument('--gamma', type=float, default=TRAINING['lr_gamma'],
                        help='Learning rate decay factor')

    # Misc arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save-dir', type=str, default=PATHS['models'],
                        help='Directory to save checkpoints')
    parser.add_argument('--save-freq', type=int, default=20,
                        help='Save checkpoint every N epochs')

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse sparsity argument
    sparsity_type = None if args.sparsity == 'None' else args.sparsity

    # Set random seed
    set_seed(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create data loaders
    print('Loading CIFAR-10 dataset...')
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )

    # Create model
    model_name = f"ResNet-20 {sparsity_type if sparsity_type else 'Dense'}"
    print(f'Creating {model_name}...')
    model = create_resnet20(
        sparsity_type=sparsity_type,
        num_classes=DATASET['num_classes']
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    # Define learning rate scheduler
    scheduler = MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    training_log = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc = checkpoint.get('acc', 0.0)

    # Training loop
    print('Starting training...')
    for epoch in range(start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]['lr']

        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        # Log metrics
        training_log['epoch'].append(epoch)
        training_log['train_loss'].append(train_loss)
        training_log['train_acc'].append(train_acc)
        training_log['val_loss'].append(val_loss)
        training_log['val_acc'].append(val_acc)
        training_log['lr'].append(current_lr)

        # Print summary
        print(f'Epoch {epoch}/{args.epochs - 1} | '
              f'LR: {current_lr:.4f} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc:.2f}%')

        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            print(f'New best accuracy: {best_acc:.2f}%')

        if (epoch + 1) % args.save_freq == 0 or is_best or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_{sparsity_type or "dense"}_epoch_{epoch}.pth')
            save_checkpoint(
                model, optimizer, epoch, val_acc, val_loss,
                checkpoint_path, is_best=is_best
            )

    # Determine save path
    if sparsity_type == '2:4':
        final_path = os.path.join(args.save_dir, 'sparse_model.pth')
    else:
        final_path = os.path.join(args.save_dir, 'dense_model.pth')

    # Save final model
    save_checkpoint(model, optimizer, args.epochs - 1, val_acc, val_loss, final_path)
    print(f'Final model saved to: {final_path}')

    # Save training log
    log_path = os.path.join(args.save_dir, f'training_log_{sparsity_type or "dense"}.json')
    save_training_log(training_log, log_path)
    print(f'Training log saved to: {log_path}')

    print(f'Training complete! Best accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()
