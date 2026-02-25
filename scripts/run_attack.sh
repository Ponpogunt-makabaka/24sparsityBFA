#!/bin/bash
# BFA attack script for ResNet-20

# Set default arguments
MODEL_PATH="./models/trained_model.pth"
MAX_FLIPS=10000
BITS_PER_ROUND=100
TARGET_ACCURACY=0.1
CALIBRATION_SAMPLES=1000
SAVE_PATH="./results/attack_log.pkl"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
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
        --target-acc)
            TARGET_ACCURACY="$2"
            shift 2
            ;;
        --calibration-samples)
            CALIBRATION_SAMPLES="$2"
            shift 2
            ;;
        --save-path)
            SAVE_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "======================================"
echo "Bit-Flip Attack on ResNet-20"
echo "======================================"
echo "Model path:           $MODEL_PATH"
echo "Max flips:            $MAX_FLIPS"
echo "Bits per round:       $BITS_PER_ROUND"
echo "Target accuracy:      $TARGET_ACCURACY"
echo "Calibration samples:  $CALIBRATION_SAMPLES"
echo "Save path:            $SAVE_PATH"
echo "======================================"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    echo "Please train the model first using: bash scripts/run_train.sh"
    exit 1
fi

# Run BFA attack
python -c "
import sys
sys.path.append('.')
import torch
from models.resnet20 import resnet20
from models.quantized_model import QuantizedResNet
from bfa.progressive_search import ProgressiveBitSearch
from train.train_utils import get_cifar10_loaders

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load model
print(f'Loading model from: $MODEL_PATH')
base_model = resnet20()
model = QuantizedResNet(base_model, bit_width=8)

checkpoint = torch.load('$MODEL_PATH', map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

# Get test data loader
_, test_loader = get_cifar10_loaders(batch_size=128)

# Run BFA attack
pbs = ProgressiveBitSearch(
    model, test_loader, device=device,
    max_flips=$MAX_FLIPS,
    target_accuracy=$TARGET_ACCURACY,
    calibration_samples=$CALIBRATION_SAMPLES
)

history = pbs.attack(
    bits_per_round=$BITS_PER_ROUND,
    verbose=True
)

# Save results
pbs.save_attack_log('$SAVE_PATH')

# Print summary
summary = pbs.get_attack_summary()
print('')
print('=' * 50)
print('Attack Summary')
print('=' * 50)
print(f'Initial accuracy: {summary[\"initial_accuracy\"]:.2f}%')
print(f'Final accuracy: {summary[\"final_accuracy\"]:.2f}%')
print(f'Accuracy drop: {summary[\"accuracy_drop\"]:.2f}%')
print(f'Total flips: {summary[\"total_flips\"]}')
print(f'Total rounds: {summary[\"total_rounds\"]}')
"

echo ""
echo "Attack complete!"
echo "Results saved to: $SAVE_PATH"
