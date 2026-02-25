"""
Quantized model wrapper for ResNet-20.
Provides uniform quantization for weights - critical for BFA effectiveness.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import struct


class QuantizedConv2d(nn.Conv2d):
    """
    Conv2d layer with quantized weights.
    Quantization is applied during forward pass but weights are stored in FP32.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, bit_width=8):
        super(QuantizedConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias
        )
        self.bit_width = bit_width
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))

    def quantize_weights(self):
        """
        Apply uniform quantization to weights.

        Formula:
        scale = (w.max - w.min) / (2^bit_width - 1)
        w_q = round((w - w.min) / scale)
        w_dq = w_q * scale + w.min
        """
        w = self.weight.data
        w_min = w.min()
        w_max = w.max()

        # Handle case where weights are constant
        if w_max - w_min < 1e-8:
            self.scale.fill_(1.0)
            self.zero_point.fill_(0.0)
            return w

        q_min = 0
        q_max = (1 << self.bit_width) - 1

        # Compute scale and zero point
        scale = (w_max - w_min) / (q_max - q_min)
        zero_point = q_min - (w_min / scale).round()

        self.scale.fill_(scale)
        self.zero_point.fill_(zero_point)

        # Quantize and dequantize
        w_quantized = torch.clamp(
            (w / scale).round() + zero_point,
            q_min, q_max
        )
        w_dequantized = (w_quantized - zero_point) * scale

        return w_dequantized

    def forward(self, x):
        # Apply weight quantization before using weights
        original_weight = self.weight.data.clone()
        quantized_weight = self.quantize_weights()
        self.weight.data = quantized_weight
        output = super().forward(x)
        self.weight.data = original_weight
        return output

    def get_quantized_weights(self) -> torch.Tensor:
        """Return quantized weights without modifying the layer."""
        with torch.no_grad():
            return self.quantize_weights()

    def set_weight_from_bits(self, bit_view: torch.Tensor):
        """
        Set weights from a bit representation (for BFA attack).

        Args:
            bit_view: Tensor with same shape as weight, containing FP32 values
                      after bit flips.
        """
        with torch.no_grad():
            self.weight.data.copy_(bit_view)


class QuantizedLinear(nn.Linear):
    """
    Linear layer with quantized weights.
    Used for the final classification layer.
    """
    def __init__(self, in_features, out_features, bias=True, bit_width=8):
        super(QuantizedLinear, self).__init__(in_features, out_features, bias)
        self.bit_width = bit_width
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))

    def quantize_weights(self):
        """Apply uniform quantization to weights."""
        w = self.weight.data
        w_min = w.min()
        w_max = w.max()

        if w_max - w_min < 1e-8:
            self.scale.fill_(1.0)
            self.zero_point.fill_(0.0)
            return w

        q_min = 0
        q_max = (1 << self.bit_width) - 1

        scale = (w_max - w_min) / (q_max - q_min)
        zero_point = q_min - (w_min / scale).round()

        self.scale.fill_(scale)
        self.zero_point.fill_(zero_point)

        w_quantized = torch.clamp(
            (w / scale).round() + zero_point,
            q_min, q_max
        )
        w_dequantized = (w_quantized - zero_point) * scale

        return w_dequantized

    def forward(self, x):
        original_weight = self.weight.data.clone()
        quantized_weight = self.quantize_weights()
        self.weight.data = quantized_weight
        output = super().forward(x)
        self.weight.data = original_weight
        return output

    def get_quantized_weights(self) -> torch.Tensor:
        """Return quantized weights."""
        with torch.no_grad():
            return self.quantize_weights()


class QuantizedResNet(nn.Module):
    """
    Wrapper that converts ResNet-20 to use quantized layers.
    This is the main model class used for BFA attacks.
    """
    def __init__(self, base_model, bit_width=8):
        """
        Args:
            base_model: Standard ResNet-20 model
            bit_width: Quantization bit width (typically 8)
        """
        super(QuantizedResNet, self).__init__()
        self.bit_width = bit_width

        # Copy structure from base model
        self.conv1 = QuantizedConv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False,
            bit_width=bit_width
        )
        self.bn1 = base_model.bn1
        self.relu = base_model.relu

        # Convert layer1
        self.layer1 = self._convert_block(base_model.layer1, bit_width)
        self.layer2 = self._convert_block(base_model.layer2, bit_width)
        self.layer3 = self._convert_block(base_model.layer3, bit_width)

        self.avgpool = base_model.avgpool

        # Convert final FC layer
        self.fc = QuantizedLinear(64, 10, bit_width=bit_width)

        # Copy weights from base model
        self._copy_weights(base_model)

    def _convert_block(self, block, bit_width):
        """Convert a ResNet block to use quantized convolutions."""
        converted = nn.Sequential()
        for i, layer in enumerate(block):
            # Build downsample layers first if needed
            downsample_layers = None
            if hasattr(layer, 'downsample') and layer.downsample is not None:
                downsample_list = []
                for downsample_layer in layer.downsample:
                    if isinstance(downsample_layer, nn.Conv2d):
                        quant_conv = QuantizedConv2d(
                            downsample_layer.in_channels,
                            downsample_layer.out_channels,
                            kernel_size=downsample_layer.kernel_size,
                            stride=downsample_layer.stride,
                            padding=downsample_layer.padding,
                            bias=False,
                            bit_width=bit_width
                        )
                        quant_conv.weight.data.copy_(downsample_layer.weight.data)
                        downsample_list.append(quant_conv)
                    else:
                        downsample_list.append(downsample_layer)
                if downsample_list:
                    downsample_layers = nn.Sequential(*downsample_list)

            # Create new BasicBlock
            new_block = BasicBlock(
                layer.conv1.in_channels,
                layer.conv1.out_channels,
                stride=layer.conv1.stride,
                downsample=downsample_layers,
                bit_width=bit_width
            )

            # Copy weights
            new_block.conv1.weight.data.copy_(layer.conv1.weight.data)
            new_block.bn1.load_state_dict(layer.bn1.state_dict())
            new_block.conv2.weight.data.copy_(layer.conv2.weight.data)
            new_block.bn2.load_state_dict(layer.bn2.state_dict())

            converted.add_module(str(i), new_block)
        return converted

    def _copy_weights(self, base_model):
        """Copy weights from base model to quantized layers."""
        self.conv1.weight.data.copy_(base_model.conv1.weight.data)
        self.bn1.load_state_dict(base_model.bn1.state_dict())

        # Copy FC weights
        self.fc.weight.data.copy_(base_model.fc.weight.data)
        if base_model.fc.bias is not None:
            self.fc.bias.data.copy_(base_model.fc.bias.data)

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

    def get_all_weights(self) -> List[Tuple[str, torch.Tensor]]:
        """
        Get all weight tensors that can be attacked.
        Returns list of (name, tensor) tuples.
        """
        weights = []
        for name, param in self.named_parameters():
            if 'weight' in name and param.requires_grad:
                weights.append((name, param))
        return weights

    def set_weight_bits(self, layer_name: str, bit_position: int, indices: torch.Tensor):
        """
        Flip specific bits in a weight tensor.

        Args:
            layer_name: Name of the layer parameter
            bit_position: Which bit to flip (0-31)
            indices: Tensor indices where bits should be flipped
        """
        for name, param in self.named_parameters():
            if name == layer_name:
                param_flat = param.data.view(-1)

                for idx in indices:
                    original_value = param_flat[idx].item()
                    flipped_value = flip_bit(original_value, bit_position)
                    param_flat[idx] = flipped_value

                param.data = param_flat.view(param.data.shape)
                break


def flip_bit(value: float, bit_position: int) -> float:
    """
    Flip a single bit in IEEE 754 FP32 representation.

    Args:
        value: Original float value
        bit_position: Bit position to flip (0-31)
                     0-22: mantissa, 23-30: exponent, 31: sign

    Returns:
        Float value with specified bit flipped
    """
    # Convert float to bytes, then to int
    packed = struct.pack('>f', value)
    as_int = struct.unpack('>I', packed)[0]

    # Flip the bit
    mask = 1 << bit_position
    flipped_int = as_int ^ mask

    # Convert back to float
    flipped_packed = struct.pack('>I', flipped_int)
    return struct.unpack('>f', flipped_packed)[0]


class BasicBlock(nn.Module):
    """ResNet Basic Block with quantized convolutions."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, bit_width=8):
        super(BasicBlock, self).__init__()
        self.conv1 = QuantizedConv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False, bit_width=bit_width
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = QuantizedConv2d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=1, bias=False, bit_width=bit_width
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


if __name__ == '__main__':
    # Test quantization
    import sys
    sys.path.append('..')
    from models.resnet20 import resnet20

    base_model = resnet20()
    quantized_model = QuantizedResNet(base_model, bit_width=8)

    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    y = quantized_model(x)
    print(f"Quantized ResNet-20 output shape: {y.shape}")

    # Test bit flip
    test_value = 3.14159
    flipped = flip_bit(test_value, 0)
    print(f"Original: {test_value}, Flipped bit 0: {flipped}")

    # Get all weights
    weights = quantized_model.get_all_weights()
    print(f"Number of weight tensors: {len(weights)}")
    print("Quantization test passed!")
