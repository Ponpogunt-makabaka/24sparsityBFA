#!/bin/bash
# BFA Attack script for Sparse ResNet-20 (Mode A: Dynamic Masking)

MODEL_PATH="./models/sparse_model.pth"
DATA_DIR="./data"
RESULT_DIR="./results"
MAX_FLIPS=10000
BITS_PER_ROUND=100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --result-dir)
            RESULT_DIR="$2"
            shift 2
            ;;
        --max-flips)
            MAX_FLIPS="$2"
            shift 2
            ;;
        --bits-per-round)
            BITS_PER_ROUND="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$RESULT_DIR"

echo "======================================"
echo "BFA Attack on Sparse ResNet-20 (Mode A: Dynamic)"
echo "======================================"
echo "Model path:   $MODEL_PATH"
echo "Data dir:     $DATA_DIR"
echo "Result dir:   $RESULT_DIR"
echo "Max flips:    $MAX_FLIPS"
echo "Bits/round:   $BITS_PER_ROUND"
echo "Mode:         Dynamic (Mask recalculated each pass)"
echo "======================================"

python -c "
import torch
from models.factory import create_resnet20
from train.train_utils import get_cifar10_loaders
from bfa.fp32_attack import run_bfa_attack

# Load model
model = create_resnet20(sparsity_type='2:4', pretrained_path='$MODEL_PATH')
model.eval()

# Unfreeze masks for Mode A
model.unfreeze_sparse_masks()
print('[BFA] Masks unfrozen - Mode A (Dynamic Masking)')

# Load data
_, test_loader = get_cifar10_loaders(batch_size=128, data_dir='$DATA_DIR')

# Run attack
result = run_bfa_attack(
    model=model,
    test_loader=test_loader,
    mode='dynamic',
    max_flips=$MAX_FLIPS,
    bits_per_round=$BITS_PER_ROUND,
    save_path='$RESULT_DIR/attack_sparse_A_result.pkl'
)

print(f'Initial accuracy: {result.initial_accuracy:.2f}%')
print(f'Final accuracy: {result.final_accuracy:.2f}%')
print(f'Total flips: {result.total_flips}')
"

echo ""
echo "Attack complete!"
echo "Results saved to: $RESULT_DIR/attack_sparse_A_result.pkl"
