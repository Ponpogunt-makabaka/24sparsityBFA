#!/bin/bash
# Training script for ResNet-20 on CIFAR-10

# Set default arguments
EPOCHS=200
BATCH_SIZE=128
BIT_WIDTH=8
DATA_DIR="./data"
SAVE_DIR="./models"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --bit-width)
            BIT_WIDTH="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create save directory if it doesn't exist
mkdir -p "$SAVE_DIR"

echo "======================================"
echo "Training ResNet-20 on CIFAR-10"
echo "======================================"
echo "Epochs:       $EPOCHS"
echo "Batch size:   $BATCH_SIZE"
echo "Bit width:    $BIT_WIDTH"
echo "Data dir:     $DATA_DIR"
echo "Save dir:     $SAVE_DIR"
echo "======================================"

# Run training
python train/train_quantized.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --bit-width $BIT_WIDTH \
    --data-dir "$DATA_DIR" \
    --save-dir "$SAVE_DIR"

echo ""
echo "Training complete!"
echo "Model saved to: $SAVE_DIR/trained_model.pth"
