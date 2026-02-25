"""
Bit manipulation utilities for BFA attack.
Provides IEEE 754 FP32 bit-level operations.
"""

import struct
import math
import torch
import numpy as np
from typing import Dict, List, Tuple, Union


# IEEE 754 Single Precision (FP32) bit layout
# [31] Sign [30:23] Exponent [22:0] Mantissa
IEEE754_RANGES = {
    'sign': [31],
    'exponent': list(range(23, 31)),
    'mantissa': list(range(23)),
}


def float_to_bits(x: Union[float, np.float32, np.float64]) -> int:
    """
    Convert a float to its 32-bit integer representation.

    Args:
        x: Float value to convert

    Returns:
        32-bit integer representing the IEEE 754 bits

    Example:
        >>> float_to_bits(3.1415927410e+00)
        1078530011  # 0x40490FDB in hex
    """
    # Pack as big-endian float32, then unpack as unsigned int32
    packed = struct.pack('>f', float(x))
    return struct.unpack('>I', packed)[0]


def bits_to_float(bits: int) -> float:
    """
    Convert a 32-bit integer to its float representation.

    Args:
        bits: 32-bit integer representing IEEE 754 bits

    Returns:
        Float value

    Example:
        >>> bits_to_float(0x40490FDB)
        3.1415927410125732
    """
    packed = struct.pack('>I', bits & 0xFFFFFFFF)
    return struct.unpack('>f', packed)[0]


def flip_bit(x: float, bit_position: int) -> float:
    """
    Flip a single bit in the IEEE 754 FP32 representation of a float.

    Args:
        x: Original float value
        bit_position: Which bit to flip (0-31)
                     - 0-22: mantissa (LSB to MSB)
                     - 23-30: exponent (LSB to MSB)
                     - 31: sign bit

    Returns:
        Float value with the specified bit flipped

    Example:
        >>> flip_bit(1.0, 31)  # Flip sign bit
        -1.0
        >>> flip_bit(1.0, 23)  # Flip LSB of exponent
        0.5
    """
    as_int = float_to_bits(x)
    mask = 1 << bit_position
    flipped_int = as_int ^ mask
    return bits_to_float(flipped_int)


def flip_bits_tensor(tensor: torch.Tensor, bit_position: int) -> torch.Tensor:
    """
    Flip a specific bit in all elements of a tensor.

    Args:
        tensor: Input tensor of floats
        bit_position: Which bit to flip (0-31)

    Returns:
        New tensor with the specified bit flipped in all elements
    """
    # Convert to int view
    tensor_flat = tensor.flatten().cpu().numpy()

    result = np.zeros_like(tensor_flat, dtype=np.float32)

    for i, val in enumerate(tensor_flat):
        result[i] = flip_bit(val, bit_position)

    return torch.from_numpy(result).view(tensor.shape).to(tensor.device)


def get_bit_ranges() -> Dict[str, List[int]]:
    """
    Get IEEE 754 bit range definitions.

    Returns:
        Dictionary with 'sign', 'exponent', 'mantissa' keys
        containing lists of bit positions
    """
    return IEEE754_RANGES.copy()


def compute_bit_flip_magnitude(bit_position: int) -> float:
    """
    Estimate the maximum magnitude change from flipping a bit at a given position.

    For a normalized FP32 number f = (-1)^s * 2^(e-127) * (1.m):
    - Sign bit (31): Can flip the sign → change of 2*|f|
    - Exponent bits (23-30): Change the scale by powers of 2
    - Mantissa bits (0-22): Change the precision proportionally to 2^(bit_pos - 23)

    Args:
        bit_position: Which bit position (0-31)

    Returns:
        Approximate maximum relative impact of flipping this bit
    """
    if bit_position == 31:
        return float('inf')  # Sign flip can double the magnitude
    elif 23 <= bit_position <= 30:
        # Exponent bit: affects the 2^e scale
        return 2.0 ** (bit_position - 23)
    else:
        # Mantissa bit: affects precision
        return 2.0 ** (bit_position - 23)


class BitIterator:
    """
    Iterator over all bits in a weight tensor.

    Yields tuples of (flat_index, bit_position, current_bit_value).
    """

    def __init__(self, tensor: torch.Tensor, bit_range: List[int] = None):
        """
        Args:
            tensor: Weight tensor to iterate over
            bit_range: List of bit positions to iterate (default: all 32 bits)
        """
        self.tensor = tensor
        self.num_elements = tensor.numel()
        self.bit_range = bit_range if bit_range is not None else list(range(32))

    def __iter__(self):
        """Iterate over all bits in the tensor."""
        for idx in range(self.num_elements):
            value = self.tensor.flatten()[idx].item()
            for bit_pos in self.bit_range:
                bit_value = (float_to_bits(value) >> bit_pos) & 1
                yield (idx, bit_pos, bit_value)

    def total_bits(self) -> int:
        """Return total number of bits to iterate over."""
        return self.num_elements * len(self.bit_range)


def verify_bit_flip():
    """
    Verify that bit flip operations work correctly.
    Useful for testing and debugging.
    """
    print("Testing bit manipulation utilities...")

    # Test 1: Round-trip conversion
    test_values = [0.0, 1.0, -1.0, 3.14159, 1e-10, 1e10, float('inf'), float('-inf')]
    for val in test_values:
        bits = float_to_bits(val)
        recovered = bits_to_float(bits)
        if val == 0.0:
            # Handle -0.0 case
            is_zero = (recovered == 0.0)
            is_nan = (recovered != recovered)
            assert is_zero or is_nan
        # For non-NaN values, check accuracy
        if val == val:
            assert abs(recovered - val) < 1e-6
    print("  Round-trip conversion: PASSED")

    # Test 2: Single bit flip
    val = 1.0
    for bit_pos in range(32):
        flipped = flip_bit(val, bit_pos)
        original_bits = float_to_bits(val)
        flipped_bits = float_to_bits(flipped)
        # Check exactly one bit changed
        diff = original_bits ^ flipped_bits
        assert diff == (1 << bit_pos)
    print("  Single bit flip: PASSED")

    # Test 3: Known bit flips
    assert flip_bit(1.0, 31) == -1.0
    assert flip_bit(-1.0, 31) == 1.0
    print("  Sign bit flip: PASSED")

    # Test 4: Bit range consistency
    ranges = get_bit_ranges()
    assert len(ranges['sign']) == 1
    assert len(ranges['exponent']) == 8
    assert len(ranges['mantissa']) == 23
    assert len(set(ranges['sign'] + ranges['exponent'] + ranges['mantissa'])) == 32
    print("  Bit ranges: PASSED")

    print("All bit manipulation tests PASSED!")


if __name__ == '__main__':
    verify_bit_flip()

    # Print bit flip magnitudes
    print("\nBit flip magnitudes:")
    for category in ['sign', 'exponent', 'mantissa']:
        print(f"  {category}:")
        for bit_pos in IEEE754_RANGES[category]:
            magnitude = compute_bit_flip_magnitude(bit_pos)
            print(f"    Bit {bit_pos:2d}: {magnitude}")
