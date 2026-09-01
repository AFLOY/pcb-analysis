"""Incremental quadratic DICE-PEEC scoring for local PCB reroutes.

The implementation uses a scalar translation-invariant kernel. A production
PEEC implementation replaces it with the tensor kernels for partial
inductance and potential coefficients while preserving the same identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.fft import irfftn, next_fast_len, rfftn


@dataclass(frozen=True)
class SparseDelta:
    """Sparse change vector for one routing candidate."""

    rows: np.ndarray
    cols: np.ndarray
    values: np.ndarray

    @classmethod
    def from_changes(
        cls, changes: Iterable[tuple[int, int, float]]
    ) -> "SparseDelta":
        merged: dict[tuple[int, int], float] = {}
        for row, col, value in changes:
            key = (int(row), int(col))
            merged[key] = merged.get(key, 0.0) + float(value)
        merged = {key: value for key, value in merged.items() if value != 0.0}
        keys = list(merged)
        return cls(
            rows=np.asarray([key[0] for key in keys], dtype=np.int32),
            cols=np.asarray([key[1] for key in keys], dtype=np.int32),
            values=np.asarray([merged[key] for key in keys], dtype=np.float64),
        )

    @property
    def size(self) -> int:
        return int(self.values.size)

    def dense(self, shape: tuple[int, int]) -> np.ndarray:
        out = np.zeros(shape, dtype=np.float64)
        np.add.at(out, (self.rows, self.cols), self.values)
        return out


class FFTInteraction2D:
    """Linear-convolution operator for a softened 1/r interaction kernel."""

    def __init__(self, shape: tuple[int, int], softening: float = 0.75):
        self.shape = tuple(int(v) for v in shape)
        self.softening = float(softening)
        # Full convolution of an n-sized vector with a (2n-1)-sized kernel.
        self.fft_shape = tuple(next_fast_len(3 * n - 2) for n in self.shape)
        self.kernel_fft = rfftn(self._embedded_kernel(), self.fft_shape)

    def kernel_value(self, dr: np.ndarray, dc: np.ndarray) -> np.ndarray:
        return 1.0 / np.sqrt(dr * dr + dc * dc + self.softening**2)

    def _embedded_kernel(self) -> np.ndarray:
        nr, nc = self.shape
        row_offsets = np.arange(-(nr - 1), nr, dtype=np.float64)
        col_offsets = np.arange(-(nc - 1), nc, dtype=np.float64)
        dr, dc = np.meshgrid(row_offsets, col_offsets, indexing="ij")
        return self.kernel_value(dr, dc)

    def apply(self, vector: np.ndarray) -> np.ndarray:
        if vector.shape != self.shape:
            raise ValueError(f"expected {self.shape}, got {vector.shape}")
        padded = np.zeros(self.fft_shape, dtype=np.float64)
        padded[: self.shape[0], : self.shape[1]] = vector
        result = irfftn(rfftn(padded) * self.kernel_fft, self.fft_shape)
        # Crop the centered "same" block from the full linear convolution.
        starts = (self.shape[0] - 1, self.shape[1] - 1)
        return result[
            starts[0] : starts[0] + self.shape[0],
            starts[1] : starts[1] + self.shape[1],
        ]

    def energy(self, vector: np.ndarray) -> float:
        return float(np.vdot(vector, self.apply(vector)).real)


class DeltaQuadraticScorer:
    """Cache Kx and score local changes without a full-grid FFT."""

    def __init__(self, operator: FFTInteraction2D, base: np.ndarray):
        self.operator = operator
        self.base = np.asarray(base, dtype=np.float64)
        self.base_field = operator.apply(self.base)
        self.base_energy = float(np.vdot(self.base, self.base_field).real)

    def delta_energy(self, delta: SparseDelta) -> float:
        if delta.size == 0:
            return 0.0
        linear = 2.0 * float(
            np.dot(delta.values, self.base_field[delta.rows, delta.cols])
        )
        dr = delta.rows[:, None] - delta.rows[None, :]
        dc = delta.cols[:, None] - delta.cols[None, :]
        local_kernel = self.operator.kernel_value(dr, dc)
        quadratic = float(delta.values @ local_kernel @ delta.values)
        return linear + quadratic

    def energy(self, delta: SparseDelta) -> float:
        return self.base_energy + self.delta_energy(delta)

    def full_energy(self, delta: SparseDelta) -> float:
        return self.operator.energy(self.base + delta.dense(self.base.shape))
