#!/usr/bin/env python3
"""
Post-Training Quantization (PTQ) Conversion Script

Converts trained FP32 models to Int8 using symmetric quantization.

Usage:
    python train/ptq_convert.py --model_type dense --input models/dense_model.pth --output models/dense_int8_model.pth
    python train/ptq_convert.py --model_type sparse --input models/sparse_model.pth --output models/sparse_int8_model.pth
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import argparse
import numpy as np
from tqdm import tqdm

from models.resnet20 import resnet20
from train.train_utils import get_cifar10_loaders


class Int8QuantizedConv2d(nn.Conv2d):
    """
    Conv2d layer with Int8 symmetric quantization.

    Symmetric Quantization Formula:
        scale = max(|w|) / 127
        w_int8 = clamp(round(w / scale), -128, 127)
        w_fp32 = w_int8 * scale
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, sparse_mask=None):
        super(Int8QuantizedConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias
        )

        # Quantization parameters
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('int8_weights', torch.zeros_like(self.weight).to(torch.int8))

        # Sparse mask support (for sparse models)
        if sparse_mask is not None:
            self.register_buffer('sparse_mask', sparse_mask)
        else:
            self.sparse_mask = None

        # Flag to enable/disable quantization
        self.quantized = False

    def calibrate_quantization(self):
        """
        Calculate quantization parameters for symmetric Int8 quantization.

        For sparse models, only considers non-zero weights for scale calculation.
        """
        w = self.weight.data

        # Apply sparse mask if present
        if self.sparse_mask is not None:
            w_masked = w * self.sparse_mask
            w_max = w_masked.abs().max()
        else:
            w_max = w.abs().max()

        if w_max < 1e-8:
            self.scale.fill_(1.0)
        else:
            # Symmetric quantization: scale = max(|w|) / 127
            scale = w_max / 127.0
            self.scale.fill_(scale)

        # Quantize to Int8
        w_quantized = torch.clamp(
            (w / self.scale).round(),
            -128, 127
        )
        self.int8_weights.copy_(w_quantized.to(torch.int8))
        self.quantized = True

    def get_dequantized_weights(self):
        """Get dequantized FP32 weights from Int8 storage."""
        w_dequantized = self.int8_weights.float() * self.scale
        # Apply sparse mask if present
        if self.sparse_mask is not None:
            w_dequantized = w_dequantized * self.sparse_mask
        return w_dequantized

    def forward(self, x):
        if self.quantized:
            # Use dequantized weights for forward pass
            original_weight = self.weight.data
            dequantized_weight = self.get_dequantized_weights()
            self.weight.data = dequantized_weight
            output = super().forward(x)
            self.weight.data = original_weight
            return output
        else:
            return super().forward(x)

    def flip_int8_bit(self, weight_idx: int, bit_pos: int):
        """
        Flip a specific bit in the Int8 weight storage.

        Int8 format (two's complement):
        Bit 7: Sign bit (0=positive, 1=negative)
        Bits 0-6: Magnitude

        Args:
            weight_idx: Flat index of the weight to flip
            bit_pos: Bit position (0-7)
        """
        # Get current Int8 value
        int8_weights_flat = self.int8_weights.view(-1)
        current_val = int8_weights_flat[weight_idx].item()

        # Flip in uint8 space to avoid Python's negative int XOR semantics
        u8 = current_val & 0xFF
        u8 ^= (1 << bit_pos)
        new_val = u8 - 256 if u8 >= 128 else u8

        int8_weights_flat[weight_idx] = new_val


class Int8QuantizedLinear(nn.Linear):
    """Linear layer with Int8 symmetric quantization (optional sparse mask)."""
    def __init__(self, in_features, out_features, bias=True, sparse_mask=None):
        super(Int8QuantizedLinear, self).__init__(in_features, out_features, bias)

        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('int8_weights', torch.zeros_like(self.weight).to(torch.int8))
        if sparse_mask is not None:
            self.register_buffer('sparse_mask', sparse_mask)
        else:
            self.sparse_mask = None
        self.quantized = False

    def calibrate_quantization(self):
        """Calculate quantization parameters."""
        w = self.weight.data
        if self.sparse_mask is not None:
            w_masked = w * self.sparse_mask
            w_max = w_masked.abs().max()
        else:
            w_max = w.abs().max()

        if w_max < 1e-8:
            self.scale.fill_(1.0)
        else:
            scale = w_max / 127.0
            self.scale.fill_(scale)

        w_quantized = torch.clamp(
            (w / self.scale).round(),
            -128, 127
        )
        self.int8_weights.copy_(w_quantized.to(torch.int8))
        self.quantized = True

    def get_dequantized_weights(self):
        """Get dequantized FP32 weights from Int8 storage."""
        w_dequantized = self.int8_weights.float() * self.scale
        if self.sparse_mask is not None:
            w_dequantized = w_dequantized * self.sparse_mask
        return w_dequantized

    def forward(self, x):
        if self.quantized:
            original_weight = self.weight.data
            dequantized_weight = self.get_dequantized_weights()
            self.weight.data = dequantized_weight
            output = super().forward(x)
            self.weight.data = original_weight
            return output
        else:
            return super().forward(x)

    def flip_int8_bit(self, weight_idx: int, bit_pos: int):
        """Flip a specific bit in the Int8 weight storage."""
        int8_weights_flat = self.int8_weights.view(-1)
        current_val = int8_weights_flat[weight_idx].item()

        u8 = current_val & 0xFF
        u8 ^= (1 << bit_pos)
        new_val = u8 - 256 if u8 >= 128 else u8

        int8_weights_flat[weight_idx] = new_val


class Int8QuantizedMultiheadAttention(nn.MultiheadAttention):
    """
    MultiheadAttention with Int8 symmetric quantization for the QKV projection matrix.

    Notes:
    - Torchvision ViT uses `nn.MultiheadAttention` where QKV are stored as a single
      parameter `in_proj_weight` with shape [3*embed_dim, embed_dim].
    - For CSR index attacks we treat `in_proj_weight` as the "weight" matrix and apply
      a 2:4 mask along the input-feature dimension (dim=1).
    - Out projection (`out_proj`) remains a child module and can be converted separately.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = False,
        sparse_mask: torch.Tensor | None = None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
        )

        # Quantization parameters for in_proj_weight only.
        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("int8_weights", torch.zeros_like(self.in_proj_weight).to(torch.int8))

        # Sparse mask support (for 2:4 structured sparsity on in_proj_weight)
        if sparse_mask is not None:
            self.register_buffer("sparse_mask", sparse_mask)
        else:
            self.sparse_mask = None

        self.quantized = False

    @property
    def weight(self) -> torch.nn.Parameter:  # type: ignore[override]
        # Expose a `weight` attribute for attack code that expects module.weight.grad.
        return self.in_proj_weight

    def calibrate_quantization(self):
        """Calculate quantization parameters for symmetric Int8 quantization (in_proj_weight only)."""
        w = self.in_proj_weight.data

        if self.sparse_mask is not None:
            w_masked = w * self.sparse_mask
            w_max = w_masked.abs().max()
        else:
            w_max = w.abs().max()

        if w_max < 1e-8:
            self.scale.fill_(1.0)
        else:
            self.scale.fill_(w_max / 127.0)

        w_quantized = torch.clamp((w / self.scale).round(), -128, 127)
        self.int8_weights.copy_(w_quantized.to(torch.int8))
        self.quantized = True

    def get_dequantized_weights(self) -> torch.Tensor:
        w = self.int8_weights.float() * self.scale
        if self.sparse_mask is not None:
            w = w * self.sparse_mask
        return w

    def forward(self, query, key, value, **kwargs):  # noqa: ANN001
        if self.quantized:
            original_w = self.in_proj_weight.data
            self.in_proj_weight.data = self.get_dequantized_weights()
            out = super().forward(query, key, value, **kwargs)
            self.in_proj_weight.data = original_w
            return out
        return super().forward(query, key, value, **kwargs)


class Int8QuantizedResNet(nn.Module):
    """
    ResNet-20 with Int8 symmetric quantization.

    This model stores weights as Int8 and dequantizes on-the-fly during forward pass.
    Supports both dense and sparse (2:4) models.
    """
    def __init__(self, base_model, copy_sparse_masks=False):
        super(Int8QuantizedResNet, self).__init__()

        # Copy structure
        # Get sparse mask for conv1 if exists
        conv1_mask = None
        if copy_sparse_masks and hasattr(base_model.conv1, 'cached_mask'):
            conv1_mask = base_model.conv1.cached_mask

        self.conv1 = Int8QuantizedConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False,
                                        sparse_mask=conv1_mask)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu

        self.layer1 = self._convert_block(base_model.layer1, copy_sparse_masks)
        self.layer2 = self._convert_block(base_model.layer2, copy_sparse_masks)
        self.layer3 = self._convert_block(base_model.layer3, copy_sparse_masks)

        self.avgpool = base_model.avgpool
        self.fc = Int8QuantizedLinear(64, 10)

        # Copy weights from base model
        self._copy_weights(base_model)

    def _convert_block(self, block, copy_sparse_masks=False):
        """Convert ResNet block to Int8 quantized."""
        converted = nn.Sequential()
        for i, layer in enumerate(block):
            # Handle downsample
            downsample_layers = None
            if hasattr(layer, 'downsample') and layer.downsample is not None:
                downsample_list = []
                for downsample_layer in layer.downsample:
                    if isinstance(downsample_layer, nn.Conv2d):
                        # Get sparse mask if exists
                        downsample_mask = None
                        if copy_sparse_masks and hasattr(downsample_layer, 'cached_mask'):
                            downsample_mask = downsample_layer.cached_mask

                        quant_conv = Int8QuantizedConv2d(
                            downsample_layer.in_channels,
                            downsample_layer.out_channels,
                            kernel_size=downsample_layer.kernel_size,
                            stride=downsample_layer.stride,
                            padding=downsample_layer.padding,
                            bias=False,
                            sparse_mask=downsample_mask
                        )
                        quant_conv.weight.data.copy_(downsample_layer.weight.data)
                        downsample_list.append(quant_conv)
                    else:
                        downsample_list.append(downsample_layer)
                if downsample_list:
                    downsample_layers = nn.Sequential(*downsample_list)

            # Get sparse masks for conv layers
            conv1_mask = None
            conv2_mask = None
            if copy_sparse_masks:
                if hasattr(layer.conv1, 'cached_mask'):
                    conv1_mask = layer.conv1.cached_mask
                if hasattr(layer.conv2, 'cached_mask'):
                    conv2_mask = layer.conv2.cached_mask

            # Create new block
            new_block = BasicBlock(
                layer.conv1.in_channels,
                layer.conv1.out_channels,
                stride=layer.conv1.stride,
                downsample=downsample_layers,
                conv1_mask=conv1_mask,
                conv2_mask=conv2_mask
            )

            # Copy weights
            new_block.conv1.weight.data.copy_(layer.conv1.weight.data)
            new_block.bn1.load_state_dict(layer.bn1.state_dict())
            new_block.conv2.weight.data.copy_(layer.conv2.weight.data)
            new_block.bn2.load_state_dict(layer.bn2.state_dict())

            converted.add_module(str(i), new_block)
        return converted

    def _copy_weights(self, base_model):
        """Copy weights from base model."""
        self.conv1.weight.data.copy_(base_model.conv1.weight.data)
        self.bn1.load_state_dict(base_model.bn1.state_dict())
        self.fc.weight.data.copy_(base_model.fc.weight.data)
        if base_model.fc.bias is not None:
            self.fc.bias.data.copy_(base_model.fc.bias.data)

    def calibrate_all_layers(self, calib_loader=None):
        """
        Calibrate quantization parameters for all layers.

        Args:
            calib_loader: Calibration data loader (optional, for calibration-time quantization)
        """
        print("[PTQ] Calibrating quantization parameters...")

        # Calibrate conv1
        self.conv1.calibrate_quantization()

        # Calibrate all conv layers in blocks
        for module in self.modules():
            if isinstance(module, Int8QuantizedConv2d):
                module.calibrate_quantization()
            elif isinstance(module, Int8QuantizedLinear):
                module.calibrate_quantization()

        print("[PTQ] Calibration complete!")

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

    def evaluate(self, test_loader, device='cpu'):
        """Evaluate model accuracy."""
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = self(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total


class BasicBlock(nn.Module):
    """ResNet Basic Block with Int8 quantized convolutions."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 conv1_mask=None, conv2_mask=None):
        super(BasicBlock, self).__init__()
        self.conv1 = Int8QuantizedConv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False, sparse_mask=conv1_mask
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Int8QuantizedConv2d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=1, bias=False, sparse_mask=conv2_mask
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


def convert_to_int8(model_path, output_path, model_type='dense'):
    """
    Convert FP32 model to Int8 using PTQ.

    Args:
        model_path: Path to FP32 model checkpoint
        output_path: Path to save Int8 model
        model_type: 'dense' or 'sparse'
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"[PTQ] Loading FP32 model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Load base model
    if model_type == 'dense':
        base_model = resnet20(sparsity_type=None).to(device)
        copy_masks = False
    else:
        base_model = resnet20(sparsity_type="2:4").to(device)
        copy_masks = True  # Copy sparse masks for sparse models

    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()

    # For sparse models, freeze masks to ensure they are computed
    if copy_masks and hasattr(base_model, 'freeze_sparse_masks'):
        base_model.freeze_sparse_masks()
        print("[PTQ] Sparse masks frozen for conversion")

    print(f"[PTQ] Base model loaded. Best accuracy: {checkpoint.get('best_acc', 'N/A')}%")

    # Create Int8 quantized model
    print("[PTQ] Creating Int8 quantized model...")
    int8_model = Int8QuantizedResNet(base_model, copy_sparse_masks=copy_masks).to(device)

    # Get test loader for evaluation
    _, test_loader = get_cifar10_loaders(batch_size=128)

    # Evaluate FP32 model for comparison
    print("[PTQ] Evaluating FP32 model...")
    fp32_acc = base_model.eval()
    fp32_correct = 0
    fp32_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = base_model(inputs)
            _, predicted = outputs.max(1)
            fp32_total += targets.size(0)
            fp32_correct += predicted.eq(targets).sum().item()
    fp32_acc = 100.0 * fp32_correct / fp32_total
    print(f"[PTQ] FP32 Model Accuracy: {fp32_acc:.2f}%")

    # Calibrate quantization
    int8_model.calibrate_all_layers()

    # Evaluate Int8 model
    print("[PTQ] Evaluating Int8 model...")
    int8_acc = int8_model.evaluate(test_loader, device)
    print(f"[PTQ] Int8 Model Accuracy: {int8_acc:.2f}%")
    print(f"[PTQ] Accuracy Drop: {fp32_acc - int8_acc:.2f}%")

    # Save Int8 model
    save_dict = {
        'model_state_dict': int8_model.state_dict(),
        'fp32_accuracy': fp32_acc,
        'int8_accuracy': int8_acc,
        'accuracy_drop': fp32_acc - int8_acc,
        'quantization': 'symmetric_int8'
    }

    torch.save(save_dict, output_path)
    print(f"[PTQ] Int8 model saved to {output_path}")

    # Print quantization statistics
    print("\n[PTQ] Quantization Statistics:")
    print("-" * 60)
    for name, module in int8_model.named_modules():
        if isinstance(module, (Int8QuantizedConv2d, Int8QuantizedLinear)):
            scale = module.scale.item()
            int8_min = module.int8_weights.min().item()
            int8_max = module.int8_weights.max().item()
            print(f"{name:40s} scale={scale:.6f} range=[{int8_min}, {int8_max}]")
    print("-" * 60)

    return int8_model, fp32_acc, int8_acc


def main():
    parser = argparse.ArgumentParser(description='PTQ Conversion for ResNet-20')
    parser.add_argument('--model_type', type=str, default='dense', choices=['dense', 'sparse'],
                        help='Model type: dense or sparse')
    parser.add_argument('--input', type=str, default='models/dense_model.pth',
                        help='Input FP32 model path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output Int8 model path (default: models/{dense,sparse}_int8_model.pth)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        args.output = f'models/{args.model_type}_int8_model.pth'

    # Convert model
    convert_to_int8(args.input, args.output, args.model_type)


if __name__ == '__main__':
    main()
