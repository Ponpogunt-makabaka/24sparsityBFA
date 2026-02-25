#!/usr/bin/env python3
"""
Task 3: CSR (Compressed Sparse Row) Sparse Encoding

Implements hardware-friendly CSR sparse format with:
- values: Int8[] - Non-zero weight values
- column_indices: Int16[] - Column indices for each value
- row_ptr: Int32[] - Row pointers (starting index for each row)

This allows simulating index corruption vs value corruption.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CSRFormat:
    """CSR format representation of a sparse weight tensor."""
    values: torch.Tensor  # Int8: Non-zero weight values
    column_indices: torch.Tensor  # Int16: Column indices
    row_ptr: torch.Tensor  # Int32: Row pointers
    original_shape: Tuple[int, ...]  # Original dense shape


def dense_to_csr(weight: torch.Tensor, sparsity_mask: Optional[torch.Tensor] = None) -> CSRFormat:
    """
    Convert a dense weight tensor to CSR format.

    Args:
        weight: Dense weight tensor [out_channels, in_channels, kh, kw]
        sparsity_mask: Optional sparsity mask (1=keep, 0=prune)

    Returns:
        CSRFormat with values, column_indices, row_ptr
    """
    device = weight.device
    dtype = weight.dtype

    # Apply sparsity mask if provided
    if sparsity_mask is not None:
        masked_weight = weight * sparsity_mask
    else:
        masked_weight = weight

    # Flatten spatial dimensions: [out_c, in_c, kh, kw] -> [out_c, in_c * kh * kw]
    out_channels = weight.shape[0]
    in_features = weight.shape[1] * weight.shape[2] * weight.shape[3]
    weight_2d = masked_weight.view(out_channels, in_features)

    # Convert to CSR
    # values: non-zero elements
    # column_indices: column positions of non-zero elements
    # row_ptr: cumulative count of non-zeros per row
    values = []
    column_indices = []
    row_ptr = [0]

    for row in range(out_channels):
        for col in range(in_features):
            val = weight_2d[row, col].item()
            if abs(val) > 1e-8:  # Non-zero
                values.append(val)
                column_indices.append(col)
        row_ptr.append(len(values))

    # Convert to tensors
    values_tensor = torch.tensor(values, dtype=dtype, device=device)
    column_indices_tensor = torch.tensor(column_indices, dtype=torch.int16, device=device)
    row_ptr_tensor = torch.tensor(row_ptr, dtype=torch.int32, device=device)

    return CSRFormat(
        values=values_tensor,
        column_indices=column_indices_tensor,
        row_ptr=row_ptr_tensor,
        original_shape=weight.shape
    )


def csr_to_dense(csr: CSRFormat, dtype=torch.float32) -> torch.Tensor:
    """Convert CSR format back to dense tensor."""
    out_channels, in_features = csr.original_shape[0], csr.original_shape[1] * csr.original_shape[2] * csr.original_shape[3]
    dense = torch.zeros(out_channels, in_features, dtype=dtype, device=csr.values.device)

    for row in range(out_channels):
        start_idx = csr.row_ptr[row].item()
        end_idx = csr.row_ptr[row + 1].item()

        for i in range(start_idx, end_idx):
            col = csr.column_indices[i].item()
            val = csr.values[i].item()
            dense[row, col] = val

    return dense.view(csr.original_shape)


class CSRSparseConv2d(nn.Conv2d):
    """
    Conv2d layer with CSR sparse encoding.

    Stores weights in hardware-friendly CSR format:
    - values: Int8 weight values
    - column_indices: Int16 column positions
    - row_ptr: Int32 row pointers

    This allows simulating hardware attacks on:
    1. Value array (MSB flips)
    2. Index array (position corruption)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, sparse_mask=None):
        super(CSRSparseConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias
        )

        # CSR format storage
        self.register_buffer('csr_values', torch.zeros(0, dtype=torch.int8))
        self.register_buffer('csr_column_indices', torch.zeros(0, dtype=torch.int16))
        self.register_buffer('csr_row_ptr', torch.zeros(0, dtype=torch.int32))

        # Quantization scale for dequantization
        self.register_buffer('scale', torch.ones(1))

        # Original sparsity mask
        if sparse_mask is not None:
            self.register_buffer('sparse_mask', sparse_mask)
        else:
            self.sparse_mask = None

        self.quantized = False

    def convert_to_csr(self):
        """Convert current weight to CSR format with Int8 quantization."""
        w = self.weight.data

        # Apply sparse mask
        if self.sparse_mask is not None:
            w_masked = w * self.sparse_mask
        else:
            w_masked = w

        # Calculate scale for quantization
        w_max = w_masked.abs().max()
        if w_max < 1e-8:
            self.scale.fill_(1.0)
        else:
            scale = w_max / 127.0
            self.scale.fill_(scale)

        # Quantize to Int8
        w_quantized = torch.clamp(
            (w_masked / self.scale).round(),
            -128, 127
        ).to(torch.int8)

        # Convert to CSR
        csr_format = dense_to_csr(w_quantized, self.sparse_mask)

        self.csr_values = csr_format.values
        self.csr_column_indices = csr_format.column_indices
        self.csr_row_ptr = csr_format.row_ptr

        self.quantized = True

    def load_csr_from_int8(
        self,
        int8_weights: torch.Tensor,
        scale: torch.Tensor,
        sparse_mask: Optional[torch.Tensor] = None
    ):
        """
        Build CSR buffers directly from pre-quantized Int8 weights.

        This is used to preserve pre-attack Int8 accuracy when converting to CSR.
        """
        # Align device/dtype
        int8_w = int8_weights.detach().to(self.weight.device).to(torch.int8)

        # Apply sparse mask (keep zeros pruned)
        if sparse_mask is not None:
            mask = sparse_mask.to(self.weight.device)
            int8_w = int8_w.clone()
            int8_w[mask < 0.5] = 0
            if hasattr(self, "sparse_mask"):
                self.sparse_mask = mask
            else:
                self.register_buffer("sparse_mask", mask)

        # Use existing scale from Int8 model
        if isinstance(scale, torch.Tensor):
            self.scale.copy_(scale.detach().to(self.weight.device))
        else:
            self.scale.fill_(float(scale))

        # Build CSR from int8 weights (no extra quantization)
        csr_format = dense_to_csr(int8_w, None)
        self.csr_values = csr_format.values
        self.csr_column_indices = csr_format.column_indices
        self.csr_row_ptr = csr_format.row_ptr
        self.original_shape = csr_format.original_shape
        self.quantized = True

    def get_dequantized_weights(self, corrupt_indices: Optional[List[Tuple[int, int]]] = None) -> torch.Tensor:
        """
        Get dequantized weights with optional index corruption.

        Args:
            corrupt_indices: List of (csr_idx, bit_flip) to corrupt indices

        Returns:
            Dequantized weight tensor with applied corruptions
        """
        # Reconstruct from CSR.
        # NOTE: This must match the collision semantics of the original Python loops:
        # if (row, col) appears multiple times, the last write wins ("mask_last").
        out_channels = int(self.csr_row_ptr.numel() - 1)
        in_features = int(self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3])

        if int(self.csr_values.numel()) == 0:
            return torch.zeros_like(self.weight, dtype=torch.float32)

        # Create column indices copy for corruption simulation.
        indices = self.csr_column_indices.clone()

        # Apply index corruption if specified.
        if corrupt_indices is not None:
            for csr_idx, bit_pos in corrupt_indices:
                if csr_idx < int(indices.numel()):
                    current_val = int(indices[csr_idx].item())
                    flipped_val = current_val ^ (1 << int(bit_pos))
                    flipped_val = max(0, min(in_features - 1, int(flipped_val)))
                    indices[csr_idx] = torch.tensor(flipped_val, dtype=torch.int16, device=indices.device)

        # Build row indices from row_ptr.
        row_ptr = self.csr_row_ptr.to(dtype=torch.long)
        counts = (row_ptr[1:] - row_ptr[:-1]).clamp(min=0)
        row_idx = torch.repeat_interleave(
            torch.arange(out_channels, device=row_ptr.device, dtype=torch.long),
            counts,
        )
        col_idx = indices.to(dtype=torch.long)
        vals = self.csr_values.to(dtype=torch.float32) * float(self.scale.item())

        dense = torch.zeros(out_channels, in_features, dtype=torch.float32, device=self.weight.device)
        dense[row_idx, col_idx] = vals  # last write wins for duplicate indices (mask_last)
        return dense.view(self.weight.shape)

    def corrupt_value_bit(self, csr_idx: int, bit_pos: int):
        """
        Flip a bit in the value array.

        Args:
            csr_idx: Index in CSR value array
            bit_pos: Bit position (0-7)
        """
        if csr_idx < len(self.csr_values):
            current_val = self.csr_values[csr_idx].item()
            new_val = self._flip_int8_value(current_val, bit_pos)
            self.csr_values[csr_idx] = torch.tensor(new_val, dtype=torch.int8)

    def corrupt_index_bit(self, csr_idx: int, bit_pos: int):
        """
        Flip a bit in the column index array.

        This simulates position corruption where a weight points to
        the wrong input activation column.

        Args:
            csr_idx: Index in CSR column_indices array
            bit_pos: Bit position to flip
        """
        if csr_idx < len(self.csr_column_indices):
            in_features = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
            current_val = self.csr_column_indices[csr_idx].item()
            new_val = current_val ^ (1 << bit_pos)
            # Ensure valid range
            new_val = max(0, min(in_features - 1, new_val))
            self.csr_column_indices[csr_idx] = torch.tensor(new_val, dtype=torch.int16)

    @staticmethod
    def _flip_int8_value(int8_val: int, bit_pos: int) -> int:
        """Flip a bit in a signed int8 value using proper 8-bit two's complement."""
        u8 = int8_val & 0xFF
        u8 ^= (1 << bit_pos)
        return u8 - 256 if u8 >= 128 else u8

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


class CSRSparseResNet(nn.Module):
    """
    ResNet-20 with CSR sparse encoding.

    This model stores weights in CSR format to enable
    position vs value attack simulations.
    """

    def __init__(self, base_model):
        super(CSRSparseResNet, self).__init__()

        def _get_mask(conv):
            if hasattr(conv, 'sparse_mask') and conv.sparse_mask is not None:
                return conv.sparse_mask
            if hasattr(conv, 'cached_mask'):
                return conv.cached_mask
            return None

        # Get sparse masks
        conv1_mask = _get_mask(base_model.conv1)

        self.conv1 = CSRSparseConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False,
                                     sparse_mask=conv1_mask)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu

        self.layer1 = self._convert_block(base_model.layer1, _get_mask)
        self.layer2 = self._convert_block(base_model.layer2, _get_mask)
        self.layer3 = self._convert_block(base_model.layer3, _get_mask)

        self.avgpool = base_model.avgpool

        # For FC, we use dense (no CSR needed for small layers)
        self.fc = base_model.fc

        # Copy weights
        self._copy_weights(base_model)

    def _convert_block(self, block, mask_fn):
        """Convert ResNet block to CSR sparse."""
        converted = nn.Sequential()
        for i, layer in enumerate(block):
            # Handle downsample
            downsample_layers = None
            if hasattr(layer, 'downsample') and layer.downsample is not None:
                downsample_list = []
                for downsample_layer in layer.downsample:
                    if isinstance(downsample_layer, nn.Conv2d):
                        downsample_mask = mask_fn(downsample_layer)

                        csr_conv = CSRSparseConv2d(
                            downsample_layer.in_channels,
                            downsample_layer.out_channels,
                            kernel_size=downsample_layer.kernel_size,
                            stride=downsample_layer.stride,
                            padding=downsample_layer.padding,
                            bias=False,
                            sparse_mask=downsample_mask
                        )
                        csr_conv.weight.data.copy_(downsample_layer.weight.data)
                        downsample_list.append(csr_conv)
                    else:
                        downsample_list.append(downsample_layer)
                if downsample_list:
                    downsample_layers = nn.Sequential(*downsample_list)

            # Get sparse masks
            conv1_mask = mask_fn(layer.conv1)
            conv2_mask = mask_fn(layer.conv2)

            # Create new block with CSR conv
            new_block = CSRBasicBlock(
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

    def load_int8_csr_weights(self, int8_model: nn.Module):
        """
        Load CSR buffers from a pre-quantized Int8 model to preserve accuracy.
        """
        int8_modules = {
            name: module for name, module in int8_model.named_modules()
            if hasattr(module, 'int8_weights') and hasattr(module, 'scale')
        }

        for name, module in self.named_modules():
            if isinstance(module, CSRSparseConv2d):
                if name not in int8_modules:
                    continue
                src = int8_modules[name]
                sparse_mask = src.sparse_mask if hasattr(src, "sparse_mask") else None
                module.load_csr_from_int8(src.int8_weights, src.scale, sparse_mask)

    def _copy_weights(self, base_model):
        """Copy weights from base model."""
        self.conv1.weight.data.copy_(base_model.conv1.weight.data)
        self.bn1.load_state_dict(base_model.bn1.state_dict())
        if base_model.fc.bias is not None:
            self.fc.bias.data.copy_(base_model.fc.bias.data)

    def convert_all_to_csr(self):
        """Convert all conv layers to CSR format."""
        print("[CSR] Converting all layers to CSR format...")
        self.conv1.convert_to_csr()

        for module in self.modules():
            if isinstance(module, CSRSparseConv2d):
                module.convert_to_csr()

        print("[CSR] Conversion complete!")

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


class CSRBasicBlock(nn.Module):
    """ResNet Basic Block with CSR sparse convolutions."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 conv1_mask=None, conv2_mask=None):
        super(CSRBasicBlock, self).__init__()
        self.conv1 = CSRSparseConv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False, sparse_mask=conv1_mask
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = CSRSparseConv2d(
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

def create_csr_model_from_sparse(device='cuda'):
    """Create a CSR sparse model from the trained sparse model."""
    from models.resnet20 import resnet20
    from train.ptq_convert import Int8QuantizedResNet

    # Load sparse base model
    base_model = resnet20(sparsity_type="2:4").to(device)
    checkpoint = torch.load('models/sparse_model.pth', map_location=device)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()
    base_model.freeze_sparse_masks()

    # Build Int8 model to preserve pre-attack accuracy
    int8_model = Int8QuantizedResNet(base_model, copy_sparse_masks=True).to(device)
    int8_model.calibrate_all_layers()
    int8_model.eval()

    # Create CSR model and load Int8 weights into CSR buffers
    csr_model = CSRSparseResNet(int8_model).to(device)
    csr_model.load_int8_csr_weights(int8_model)
    csr_model.eval()

    return csr_model


if __name__ == '__main__':
    print("CSR Sparse Encoding Module")
    print("This module provides CSR format for sparse weight storage.")
