#!/bin/bash
# Training script for Dense ResNet-20 on CIFAR-10

EPOCHS=200
BATCH_SIZE=128
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

mkdir -p "$SAVE_DIR"

echo "======================================"
echo "Training Dense ResNet-20 on CIFAR-10"
echo "======================================"
echo "Epochs:       $EPOCHS"
echo "Batch size:   $BATCH_SIZE"
echo "Data dir:     $DATA_DIR"
echo "Save dir:     $SAVE_DIR"
echo "======================================"

python train/train_sparse.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --sparsity None \
    --data-dir "$DATA_DIR" \
    --save-dir "$SAVE_DIR"

echo ""
echo "Training complete!"
echo "Model saved to: $SAVE_DIR/dense_model.pth"
