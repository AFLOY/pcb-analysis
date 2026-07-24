"""CuPy/cuFFT and RawKernel implementation of sparse delta scoring."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.fft import next_fast_len

from .delta_peec import SparseDelta


_DELTA_KERNEL = r"""
extern "C" __global__
void score_delta(
    const float* base_field,
    const int ncols,
    const int* rows,
    const int* cols,
    const double* values,
    const long long* offsets,
    const double softening2,
    const double base_energy,
    double* output,
    const int candidate_count)
{
    const int candidate = blockDim.x * blockIdx.x + threadIdx.x;
    if (candidate >= candidate_count) return;
    const long long start = offsets[candidate];
    const long long stop = offsets[candidate + 1];
    double linear = 0.0;
    double quadratic = 0.0;
    for (long long i = start; i < stop; ++i) {
        const long long base_index =
            (long long)rows[i] * (long long)ncols + (long long)cols[i];
        linear += 2.0 * values[i] * (double)base_field[base_index];
        for (long long j = start; j < stop; ++j) {
            const double dr = (double)(rows[i] - rows[j]);
            const double dc = (double)(cols[i] - cols[j]);
            quadratic += values[i] * values[j] / sqrt(dr * dr + dc * dc + softening2);
        }
    }
    output[candidate] = base_energy + linear + quadratic;
}
"""


class CudaDeltaQuadraticScorer:
    """Batch exact local delta energies with one RawKernel launch."""

    def __init__(
        self,
        base: np.ndarray,
        softening: float = 0.75,
        *,
        cupy_module: Any | None = None,
    ) -> None:
        with np.errstate(over="ignore", invalid="ignore"):
            self.base = np.array(base, dtype=np.float32, copy=True)
        if self.base.ndim != 2:
            raise ValueError("base must be a two-dimensional array")
        if any(size <= 0 for size in self.base.shape):
            raise ValueError("base dimensions must be positive")
        if not np.all(np.isfinite(self.base)):
            raise ValueError("base must contain only finite float32 values")
        self.shape = self.base.shape
        self.softening = float(softening)
        if not np.isfinite(self.softening) or self.softening <= 0.0:
            raise ValueError("softening must be finite and positive")

        if cupy_module is None:
            try:
                import cupy as cp
            except (ImportError, OSError) as exc:
                raise RuntimeError("CuPy is required for CUDA delta scoring") from exc
        else:
            cp = cupy_module
        self.cp = cp
        self.fft_shape = tuple(next_fast_len(3 * n - 2) for n in self.shape)

        base_gpu = cp.asarray(self.base)
        padded = cp.zeros(self.fft_shape, dtype=cp.float32)
        padded[: self.shape[0], : self.shape[1]] = base_gpu
        kernel = cp.asarray(self._embedded_kernel(), dtype=cp.float32)
        kernel_fft = cp.fft.rfftn(kernel, self.fft_shape)
        convolved = cp.fft.irfftn(
            cp.fft.rfftn(padded) * kernel_fft, self.fft_shape
        )
        starts = (self.shape[0] - 1, self.shape[1] - 1)
        self.base_field = cp.ascontiguousarray(
            convolved[
                starts[0] : starts[0] + self.shape[0],
                starts[1] : starts[1] + self.shape[1],
            ],
            dtype=cp.float32,
        )
        self.base_energy = float(
            cp.sum(base_gpu.astype(cp.float64) * self.base_field, dtype=cp.float64)
        )
        self._kernel = cp.RawKernel(_DELTA_KERNEL, "score_delta")

    def _embedded_kernel(self) -> np.ndarray:
        nr, nc = self.shape
        row_offsets = np.arange(-(nr - 1), nr, dtype=np.float32)
        col_offsets = np.arange(-(nc - 1), nc, dtype=np.float32)
        dr, dc = np.meshgrid(row_offsets, col_offsets, indexing="ij")
        return 1.0 / np.sqrt(dr * dr + dc * dc + self.softening**2)

    @staticmethod
    def _pack(
        candidates: Iterable[SparseDelta],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        items = list(candidates)
        normalized: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        int32 = np.iinfo(np.int32)
        for item in items:
            rows = np.asarray(item.rows)
            cols = np.asarray(item.cols)
            values = np.asarray(item.values, dtype=np.float64)
            if rows.ndim != 1 or cols.ndim != 1 or values.ndim != 1:
                raise ValueError(
                    "candidate rows, cols, and values must be one-dimensional"
                )
            if not (rows.size == cols.size == values.size):
                raise ValueError(
                    "candidate rows, cols, and values must have equal lengths"
                )
            if rows.size and not np.issubdtype(rows.dtype, np.integer):
                raise TypeError("candidate rows must use an integer dtype")
            if cols.size and not np.issubdtype(cols.dtype, np.integer):
                raise TypeError("candidate cols must use an integer dtype")
            if rows.size and (
                np.any(rows < int32.min) or np.any(rows > int32.max)
            ):
                raise OverflowError("candidate row does not fit CUDA int32 indexing")
            if cols.size and (
                np.any(cols < int32.min) or np.any(cols > int32.max)
            ):
                raise OverflowError("candidate col does not fit CUDA int32 indexing")
            if not np.all(np.isfinite(values)):
                raise ValueError("candidate values must be finite")
            normalized.append(
                (
                    rows.astype(np.int32, copy=False),
                    cols.astype(np.int32, copy=False),
                    values,
                )
            )

        offsets = np.zeros(len(items) + 1, dtype=np.int64)
        if items:
            offsets[1:] = np.cumsum(
                [values.size for _, _, values in normalized], dtype=np.int64
            )
        rows = (
            np.concatenate([rows for rows, _, _ in normalized], dtype=np.int32)
            if offsets[-1]
            else np.empty(0, np.int32)
        )
        cols = (
            np.concatenate([cols for _, cols, _ in normalized], dtype=np.int32)
            if offsets[-1]
            else np.empty(0, np.int32)
        )
        values = (
            np.concatenate([values for _, _, values in normalized], dtype=np.float64)
            if offsets[-1]
            else np.empty(0, np.float64)
        )
        return rows, cols, values, offsets, len(items)

    @staticmethod
    def _validate_indices(
        rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int]
    ) -> None:
        if (
            np.any(rows < 0)
            or np.any(rows >= shape[0])
            or np.any(cols < 0)
            or np.any(cols >= shape[1])
        ):
            raise IndexError(f"candidate coordinate is outside base shape {shape}")

    def energy_many(self, candidates: Iterable[SparseDelta]) -> np.ndarray:
        cp = self.cp
        rows, cols, values, offsets, count = self._pack(candidates)
        if count == 0:
            return np.empty(0, dtype=np.float64)
        self._validate_indices(rows, cols, self.shape)
        rows_gpu = cp.asarray(rows)
        cols_gpu = cp.asarray(cols)
        values_gpu = cp.asarray(values)
        offsets_gpu = cp.asarray(offsets)
        output = cp.empty(count, dtype=cp.float64)
        threads = 128
        blocks = (count + threads - 1) // threads
        self._kernel(
            (blocks,),
            (threads,),
            (
                self.base_field,
                np.int32(self.shape[1]),
                rows_gpu,
                cols_gpu,
                values_gpu,
                offsets_gpu,
                np.float64(self.softening**2),
                np.float64(self.base_energy),
                output,
                np.int32(count),
            ),
        )
        return cp.asnumpy(output)
