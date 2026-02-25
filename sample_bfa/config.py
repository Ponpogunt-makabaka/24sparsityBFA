"""
Central configuration for ResNet20-BFA project.
"""

# Training configuration
TRAINING = {
    'epochs': 200,
    'batch_size': 128,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'lr_milestones': [100, 150],
    'lr_gamma': 0.1,
}

# Quantization configuration
QUANTIZATION = {
    'bit_width': 8,
    'quantize_bias': False,
    'quantize_method': 'uniform',  # 'uniform' or 'minmax'
}

# BFA (Bit-Flip Attack) configuration
BFA = {
    'max_flips': 10000,
    'target_accuracy': 0.1,  # Attack until accuracy drops below 10%
    'bits_per_round': 100,   # Number of bits to flip per round
    'calibration_samples': 1000,  # Number of samples for sensitivity computation
    'sensitivity_method': 'gradient',  # 'gradient' or 'finite_diff'
    'attack_all_layers': True,  # If True, attack ALL layers (v3 requirement)
}

# Dataset configuration
DATASET = {
    'name': 'CIFAR10',
    'num_classes': 10,
    'img_size': 32,
    'mean': (0.4914, 0.4822, 0.4465),  # CIFAR-10 mean
    'std': (0.2471, 0.2435, 0.2616),   # CIFAR-10 std
}

# Model configuration
MODEL = {
    'name': 'ResNet20',
    'depth': 20,
    'num_classes': 10,
    'input_channels': 3,
}

# Experiment configuration for v3 (2:4 Sparsity & FP32 BFA)
EXPERIMENT = {
    # Model type: 'dense' or 'sparse'
    'model_type': 'sparse',

    # Sparsity configuration (only used when model_type='sparse')
    'sparsity': {
        'N': 2,  # Number of non-zero elements per group
        'M': 4,  # Group size
    },

    # Attack mode for sparse models: 'dynamic' (Mode A) or 'static' (Mode B)
    # Mode A (Dynamic): Faults injected into dense weights can change which weights are pruned
    # Mode B (Static): Sparsity mask is frozen before attack starts
    # Ignored for dense models
    'attack_mode': 'dynamic',
}

# IEEE 754 FP32 bit ranges
IEEE754 = {
    'sign': [31],
    'exponent': list(range(23, 31)),
    'mantissa': list(range(23)),
}

# File paths
PATHS = {
    'data': './data',
    'models': './models',
    'results': './results',
    'logs': './logs',
    'checkpoint': './models/trained_model.pth',
    'checkpoint_dense': './models/dense_model.pth',
    'checkpoint_sparse': './models/sparse_model.pth',
    'attack_log': './results/attack_log.pkl',
}

# Device configuration
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Random seed for reproducibility
SEED = 42
