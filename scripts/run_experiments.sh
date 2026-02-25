#!/bin/bash
# Master script to run all BFA experiments
# Compares Dense vs Sparse (Mode A & Mode B) models under BFA

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default arguments
TRAIN_EPOCHS=200
BATCH_SIZE=128
DATA_DIR="$PROJECT_DIR/data"
MODEL_DIR="$PROJECT_DIR/models"
RESULT_DIR="$PROJECT_DIR/results"
MAX_FLIPS=10000
BITS_PER_ROUND=100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train)
            TRAIN="true"
            shift
            ;;
        --skip-train)
            TRAIN="false"
            shift
            ;;
        --attack)
            ATTACK="true"
            shift
            ;;
        --skip-attack)
            ATTACK="false"
            shift
            ;;
        --epochs)
            TRAIN_EPOCHS="$2"
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
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--train|--skip-train] [--attack|--skip-attack] [--epochs N] [--max-flips N] [--bits-per-round N] [--data-dir PATH]"
            exit 1
            ;;
    esac
done

# Default behavior
TRAIN="${TRAIN:-true}"
ATTACK="${ATTACK:-true}"

mkdir -p "$MODEL_DIR"
mkdir -p "$RESULT_DIR"

echo "========================================"
echo "BFA Experiments: 2:4 Sparsity Research"
echo "========================================"
echo "Train models:       $TRAIN"
echo "Run attacks:        $ATTACK"
echo "Training epochs:    $TRAIN_EPOCHS"
echo "Max bit flips:      $MAX_FLIPS"
echo "Bits per round:     $BITS_PER_ROUND"
echo "Data directory:     $DATA_DIR"
echo "========================================"
echo ""

# Phase 1: Training
if [ "$TRAIN" = "true" ]; then
    echo "================================================"
    echo "Phase 1: Training Models"
    echo "================================================"

    echo ""
    echo "[1/2] Training Dense ResNet-20..."
    bash "$SCRIPT_DIR/run_train_dense.sh" \
        --epochs "$TRAIN_EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --data-dir "$DATA_DIR" \
        --save-dir "$MODEL_DIR"

    echo ""
    echo "[2/2] Training 2:4 Sparse ResNet-20..."
    bash "$SCRIPT_DIR/run_train_sparse.sh" \
        --epochs "$TRAIN_EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --data-dir "$DATA_DIR" \
        --save-dir "$MODEL_DIR"

    echo ""
    echo "Training complete!"
    echo ""
fi

# Phase 2: BFA Attacks
if [ "$ATTACK" = "true" ]; then
    echo "================================================"
    echo "Phase 2: Running BFA Attacks"
    echo "================================================"

    echo ""
    echo "[1/3] Attack on Dense model..."
    bash "$SCRIPT_DIR/run_attack_dense.sh" \
        --model-path "$MODEL_DIR/dense_model.pth" \
        --data-dir "$DATA_DIR" \
        --result-dir "$RESULT_DIR" \
        --max-flips "$MAX_FLIPS" \
        --bits-per-round "$BITS_PER_ROUND"

    echo ""
    echo "[2/3] Attack on Sparse model (Mode A: Dynamic)..."
    bash "$SCRIPT_DIR/run_attack_A.sh" \
        --model-path "$MODEL_DIR/sparse_model.pth" \
        --data-dir "$DATA_DIR" \
        --result-dir "$RESULT_DIR" \
        --max-flips "$MAX_FLIPS" \
        --bits-per-round "$BITS_PER_ROUND"

    echo ""
    echo "[3/3] Attack on Sparse model (Mode B: Static)..."
    bash "$SCRIPT_DIR/run_attack_B.sh" \
        --model-path "$MODEL_DIR/sparse_model.pth" \
        --data-dir "$DATA_DIR" \
        --result-dir "$RESULT_DIR" \
        --max-flips "$MAX_FLIPS" \
        --bits-per-round "$BITS_PER_ROUND"

    echo ""
    echo "All attacks complete!"
    echo ""
fi

# Summary
echo "================================================"
echo "Experiment Summary"
echo "================================================"
echo ""

if [ "$ATTACK" = "true" ]; then
    echo "Results saved to: $RESULT_DIR"
    echo ""
    echo "Result files:"
    ls -1 "$RESULT_DIR"/attack_*.pkl 2>/dev/null || echo "  (No results found)"
    echo ""
fi

echo "========================================"
echo "All experiments completed!"
echo "========================================"
