"""
ResNet-20 2:4 sparsity for CIFAR-10 dataset.
Adapted from PyTorch official implementation for 32x32 images.

Supports both Dense (standard FP32) and Sparse (2:4 structured sparsity) variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MODEL

# Import sparse operations (same directory)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparse_ops import SparseConv

import math
try:
    from .quantization_nm import *
except Exception:
    # Fallback stubs if quantization_nm is missing (allow inference with standard layers)
    try:
        # try package-level import
        from pth.quantization_nm import *
    except Exception:
        # define minimal fallbacks used in the repo so import won't fail
        import torch.nn as _nn

        def quan_Conv2d(*args, **kwargs):
            return _nn.Conv2d(*args, **kwargs)

        def quan_Linear(*args, **kwargs):
            return _nn.Linear(*args, **kwargs)

        def quantize(x):
            return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, sparsity_type=None):
        super(BasicBlock, self).__init__()
        self.sparsity_type = sparsity_type

        # Choose convolution type based on sparsity
        if sparsity_type == "2:4":
            ConvLayer = SparseConv
            conv_kwargs = {'N': 2, 'M': 4}
        else:
            ConvLayer = nn.Conv2d
            conv_kwargs = {}

        self.conv1 = ConvLayer(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False, **conv_kwargs
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvLayer(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=1, bias=False, **conv_kwargs
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Args:
        sparsity_type: None for dense (standard FP32), "2:4" for 2:4 sparse
    """
    def __init__(self, block, layers, num_classes=10, in_channels=3, sparsity_type=None):
        super(ResNet, self).__init__()
        self.sparsity_type = sparsity_type
        self.in_channels = 16  # Initial number of filters

        # Choose convolution type based on sparsity
        if sparsity_type == "2:4":
            ConvLayer = SparseConv
            conv_kwargs = {'N': 2, 'M': 4}
        else:
            ConvLayer = nn.Conv2d
            conv_kwargs = {}

        # Initial convolution (no maxpool for CIFAR-10)
        self.conv1 = ConvLayer(
            in_channels, 16, kernel_size=3, stride=1,
            padding=1, bias=False, **conv_kwargs
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # ResNet stages: [16, 32, 64] filters
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        # Average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            if self.sparsity_type == "2:4":
                downsample = nn.Sequential(
                    SparseConv(
                        self.in_channels, out_channels * block.expansion,
                        kernel_size=1, stride=stride, bias=False, N=2, M=4
                    ),
                    nn.BatchNorm2d(out_channels * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.in_channels, out_channels * block.expansion,
                        kernel_size=1, stride=stride, bias=False
                    ),
                    nn.BatchNorm2d(out_channels * block.expansion),
                )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, self.sparsity_type))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, sparsity_type=self.sparsity_type))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, SparseConv)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def freeze_sparse_masks(self):
        """
        Freeze all sparse masks in the network.
        Used for Mode B (Static Masking) BFA attacks.
        Only effective when sparsity_type="2:4".
        """
        if self.sparsity_type == "2:4":
            for m in self.modules():
                if isinstance(m, SparseConv):
                    m.freeze_mask()

    def unfreeze_sparse_masks(self):
        """
        Unfreeze all sparse masks in the network.
        Used for Mode A (Dynamic Masking) BFA attacks.
        Only effective when sparsity_type="2:4".
        """
        if self.sparsity_type == "2:4":
            for m in self.modules():
                if isinstance(m, SparseConv):
                    m.unfreeze_mask()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet20_nm(pretrained: bool = False, sparsity_type: str = None, **kwargs) -> ResNet:
    """
    Construct ResNet-20 for CIFAR-10.

    ResNet-20 has:
    - 1 initial conv
    - 3 stages × 3 blocks = 9 blocks
    - Each block has 2 convolutions
    - 1 + 9×2 + 1 (fc) = 20 layers

    Args:
        pretrained: If True, load pretrained weights (not implemented)
        sparsity_type: None for dense (standard FP32), "2:4" for 2:4 sparse
        **kwargs: Additional arguments passed to ResNet

    Returns:
        ResNet-20 model
    """
    model = ResNet(BasicBlock, [3, 3, 3], sparsity_type=sparsity_type, **kwargs)
    return model


def count_parameters(model):
    """Count total and trainable parameters in model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    # Test model
    model = resnet20()
    total_params, trainable_params = count_parameters(model)

    print(f"ResNet-20")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Expected: ~270K parameters
    assert total_params > 260000 and total_params < 280000, "Unexpected parameter count"
    assert y.shape == (2, 10), "Unexpected output shape"
    print("All tests passed!")
