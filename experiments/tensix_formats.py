"""Tensix number formats rounded on the host, shared by the Tensix experiments.

Round-to-nearest-even emulation of what the Tensix matrix engine (FPU)
consumes: TF32 (10 mantissa bits, the width of its source registers), BF16
(7), and the block floats BFP8_B/BFP4_B, where 16 consecutive values of a row
share the largest exponent and keep 7/3 magnitude bits.  NumPy only, so the
module imports in both the project venv and the ttnn venv.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

BLOCK = 16  # values sharing one exponent in a Tensix block-float row


def round_mantissa(values: np.ndarray, bits: int) -> np.ndarray:
    """FP32 values rounded to ``bits`` explicit mantissa bits, nearest-even."""

    data = np.ascontiguousarray(values, dtype=np.float32).copy()
    raw = data.view(np.uint32)
    shift = np.uint32(23 - bits)
    half = np.uint32(1 << (23 - bits - 1))
    odd = (raw >> shift) & np.uint32(1)
    raw += half - np.uint32(1) + odd
    raw &= ~np.uint32((1 << (23 - bits)) - 1)
    return data


def round_block_float(values: np.ndarray, magnitude_bits: int) -> np.ndarray:
    """Block float along the last axis: 16 values share the largest exponent."""

    data = np.asarray(values, dtype=np.float64)
    shape = data.shape
    width = shape[-1]
    padded_width = -(-width // BLOCK) * BLOCK
    padded = np.zeros((*shape[:-1], padded_width))
    padded[..., :width] = data
    blocks = padded.reshape(*shape[:-1], padded_width // BLOCK, BLOCK)
    peak = np.abs(blocks).max(axis=-1, keepdims=True)
    _, exponent = np.frexp(np.where(peak > 0.0, peak, 1.0))
    step = np.ldexp(1.0, exponent - magnitude_bits)
    limit = 2.0**magnitude_bits - 1.0
    quantised = np.clip(np.rint(blocks / step), -limit, limit) * step
    return quantised.reshape(*shape[:-1], padded_width)[..., :width].astype(np.float32)


def split_bf16(values: np.ndarray, terms: int) -> list[np.ndarray]:
    """``terms`` BF16 pieces, largest first, that one would feed to the FPU in turn."""

    remainder = np.asarray(values, dtype=np.float32)
    pieces = []
    for _ in range(terms):
        piece = round_mantissa(remainder, 7)
        pieces.append(piece)
        remainder = remainder - piece  # exact in FP32: the piece shares the leading bits
    return pieces


@dataclass(frozen=True)
class MatrixFormat:
    """How the coarse GEMV operands reach the FPU."""

    name: str
    matrix: Callable[[np.ndarray], np.ndarray]
    vector: Callable[[np.ndarray], np.ndarray]
    bytes_per_value: float


def _fp32(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _tf32(values: np.ndarray) -> np.ndarray:
    return round_mantissa(values, 10)


def _bf16(values: np.ndarray) -> np.ndarray:
    return round_mantissa(values, 7)


MATRIX_FORMATS = {
    "fp32": MatrixFormat("fp32", _fp32, _fp32, 4.0),
    "tf32": MatrixFormat("tf32", _tf32, _tf32, 4.0),
    "bf16": MatrixFormat("bf16", _bf16, _bf16, 2.0),
    "bfp8b": MatrixFormat("bfp8b", lambda m: round_block_float(m, 7), _bf16, 1.0 + 1.0 / BLOCK),
    "bfp4b": MatrixFormat("bfp4b", lambda m: round_block_float(m, 3), _bf16, 0.5 + 1.0 / BLOCK),
}

STENCIL_FORMATS: dict[str, Callable[[np.ndarray], list[np.ndarray]]] = {
    "tf32": lambda v: [_tf32(v)],
    "bf16": lambda v: [_bf16(v)],
    "bf16x2": lambda v: split_bf16(v, 2),
    "bf16x3": lambda v: split_bf16(v, 3),
}
