"""Optional fused C++ low path for the hexahedral Q1 conduction operator.

The compiled module ``_thermal_native`` is built in place with
``python -m thermal.matrix_free_mpir_fem.native.build``.  Without it every
entry point reports unavailability and the portable NumPy path is used.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

try:  # pragma: no cover - depends on the local build
    from . import _thermal_native as _native
except ImportError:  # pragma: no cover
    _native = None


NATIVE_KERNEL_NAME = "cpp-fused-node-gather-hex-q1"


def native_available() -> bool:
    return _native is not None


def native_requested() -> bool:
    """True when ``PCB_NATIVE_THERMAL`` selects the built extension by default."""

    flag = os.environ.get("PCB_NATIVE_THERMAL", "").strip().lower()
    return flag in {"1", "true", "yes", "on"} and native_available()


def native_threads() -> int:
    """Thread count for the native path; ``PCB_NATIVE_THREADS`` overrides."""

    value = os.environ.get("PCB_NATIVE_THREADS")
    if value:
        return max(1, int(value))
    return 1


class NativeThermalHexQ1:
    """Host float32 operator and two-level inner PCG bound to one prepared mesh."""

    kernel_name = NATIVE_KERNEL_NAME

    def __init__(
        self,
        element_grid_shape: tuple[int, int, int],
        coefficients: np.ndarray,
        unit: np.ndarray,
        robin: np.ndarray,
        free_nodes: np.ndarray,
        diagonal: np.ndarray,
        *,
        coarse_block: int | None = None,
        coarse_inverse: np.ndarray | None = None,
        threads: int | None = None,
    ) -> None:
        if _native is None:
            raise ImportError(
                "the thermal native extension is not built; run "
                "python -m thermal.matrix_free_mpir_fem.native.build"
            )
        self.slabs, self.rows, self.cols = (int(axis) for axis in element_grid_shape)
        self.size = (self.slabs + 1) * (self.rows + 1) * (self.cols + 1)
        self.threads = threads if threads is not None else native_threads()
        f32 = lambda value: np.ascontiguousarray(value, dtype=np.float32).reshape(-1)
        self._coefficients = f32(coefficients)
        self._unit = f32(unit)
        if self._coefficients.size != 3 * self.slabs * self.rows * self.cols or self._unit.size != 192:
            raise ValueError("coefficients must be (3, slabs, rows, cols) and unit (3, 8, 8)")
        self._robin = f32(robin)
        self._free = np.ascontiguousarray(free_nodes, dtype=np.uint8).reshape(-1)
        self._free_mask = self._free.astype(np.float32)
        self._diagonal = f32(diagonal)
        if (coarse_block is None) != (coarse_inverse is None):
            raise ValueError("coarse_block and coarse_inverse go together")
        self.coarse_block = int(coarse_block) if coarse_block is not None else 1
        self._coarse_inverse = (
            np.ascontiguousarray(coarse_inverse, dtype=np.float32).reshape(-1)
            if coarse_inverse is not None
            else np.zeros(0, dtype=np.float32)
        )
        if coarse_inverse is not None:
            block = self.coarse_block
            coarse = (self.slabs + 1) * (-(-(self.rows + 1) // block)) * (-(-(self.cols + 1) // block))
            if self._coarse_inverse.size != coarse * coarse:
                raise ValueError("coarse_inverse does not match the patch grid")

    def apply(self, vector: Any) -> np.ndarray:
        vector = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
        if vector.size != self.size:
            raise ValueError(f"vector has size {vector.size}, expected {self.size}")
        return _native.apply_hex_q1(
            vector,
            self._coefficients,
            self._unit,
            self._robin,
            self._free,
            self._free_mask,
            self.slabs,
            self.rows,
            self.cols,
            self.threads,
        )

    def inner_pcg(
        self,
        rhs_high: np.ndarray,
        *,
        inner_relative_tolerance: float,
        max_inner_iterations: int,
    ) -> tuple[np.ndarray, int, float, int]:
        rhs_high = np.ascontiguousarray(rhs_high, dtype=np.float64).reshape(-1)
        if rhs_high.size != self.size:
            raise ValueError(f"rhs has size {rhs_high.size}, expected {self.size}")
        correction, iterations, relative_residual, applications = _native.pcg_hex_q1(
            rhs_high,
            self._diagonal,
            self._coefficients,
            self._unit,
            self._robin,
            self._free,
            self._free_mask,
            self.slabs,
            self.rows,
            self.cols,
            self.coarse_block,
            self._coarse_inverse,
            float(inner_relative_tolerance),
            int(max_inner_iterations),
            self.threads,
        )
        return (
            np.asarray(correction, dtype=np.float32),
            int(iterations),
            float(relative_residual),
            int(applications),
        )
