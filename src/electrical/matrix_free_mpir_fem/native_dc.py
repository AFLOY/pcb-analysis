"""Optional fused C++ path for the layered-PCB DC conduction operator.

The compiled module ``_layered_dc_native`` is built in place with
``python -m electrical.matrix_free_mpir_fem.native.build`` (which also builds
the scalar Maxwell kernel).  Without it every entry point reports
unavailability and the portable NumPy path is used.  ``PCB_NATIVE_Q1`` selects
the built extension by default for both Q1 kernels of this package; the
results are the same either way, only the speed differs.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .native_q1 import native_threads

try:  # pragma: no cover - depends on the local build
    from . import _layered_dc_native as _native
except ImportError:  # pragma: no cover
    _native = None


NATIVE_KERNEL_NAME = "cpp-fused-node-gather-layered-dc-q1"
NATIVE_HIGH_KERNEL_NAME = "cpp-fused-node-gather-layered-dc-q1-fp64"


def native_available() -> bool:
    return _native is not None


def native_requested() -> bool:
    """True when ``PCB_NATIVE_Q1`` selects the built extension by default."""

    flag = os.environ.get("PCB_NATIVE_Q1", "").strip().lower()
    return flag in {"1", "true", "yes", "on"} and native_available()


def via_adjacency(
    via_a: np.ndarray, via_b: np.ndarray, via_g: np.ndarray, size: int, dtype: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Node-owned CSR of the resistive links: each via under both of its ends."""

    a = np.asarray(via_a, dtype=np.int64).reshape(-1)
    b = np.asarray(via_b, dtype=np.int64).reshape(-1)
    g = np.asarray(via_g, dtype=np.float64).reshape(-1)
    if not (a.shape == b.shape == g.shape):
        raise ValueError("via endpoints and conductances must have one entry per via")
    owner = np.concatenate((a, b))
    neighbour = np.concatenate((b, a))
    conductance = np.concatenate((g, g))
    order = np.argsort(owner, kind="stable")
    counts = np.bincount(owner, minlength=size) if owner.size else np.zeros(size, dtype=np.int64)
    pointer = np.zeros(size + 1, dtype=np.int64)
    np.cumsum(counts, out=pointer[1:])
    return (
        pointer,
        np.ascontiguousarray(neighbour[order], dtype=np.int64),
        np.ascontiguousarray(conductance[order], dtype=dtype),
    )


class _Prepared:
    def __init__(
        self,
        node_shape: tuple[int, int, int],
        coefficients: np.ndarray,
        unit: np.ndarray,
        free_nodes: np.ndarray,
        via_a: np.ndarray,
        via_b: np.ndarray,
        via_g: np.ndarray,
        dtype: Any,
        threads: int | None,
    ) -> None:
        if _native is None:
            raise ImportError(
                "the layered DC native extension is not built; run "
                "python -m electrical.matrix_free_mpir_fem.native.build"
            )
        layers, node_rows, node_cols = (int(axis) for axis in node_shape)
        self.layers, self.rows, self.cols = layers, node_rows - 1, node_cols - 1
        if self.rows < 1 or self.cols < 1 or self.layers < 1:
            raise ValueError("node_shape must describe at least one element per layer")
        self.size = layers * node_rows * node_cols
        self.threads = threads if threads is not None else native_threads()
        cast = lambda value: np.ascontiguousarray(value, dtype=dtype).reshape(-1)
        self._coefficients = cast(coefficients)
        self._unit = cast(unit)
        if self._coefficients.size != 2 * self.layers * self.rows * self.cols or self._unit.size != 32:
            raise ValueError("coefficients must be (2, layers, rows, cols) and unit (2, 4, 4)")
        self._free = np.ascontiguousarray(free_nodes, dtype=np.uint8).reshape(-1)
        if self._free.size != self.size:
            raise ValueError("free_nodes must hold one flag per node")
        self._free_mask = self._free.astype(dtype)
        self._via_ptr, self._via_nbr, self._via_g = via_adjacency(via_a, via_b, via_g, self.size, dtype)


class NativeLayeredDCQ1(_Prepared):
    """Host float32 operator and two-level inner PCG bound to one prepared mesh."""

    kernel_name = NATIVE_KERNEL_NAME

    def __init__(
        self,
        node_shape: tuple[int, int, int],
        coefficients: np.ndarray,
        unit: np.ndarray,
        free_nodes: np.ndarray,
        via_a: np.ndarray,
        via_b: np.ndarray,
        via_g: np.ndarray,
        diagonal: np.ndarray,
        *,
        coarse_block: int | None = None,
        coarse_inverse: np.ndarray | None = None,
        threads: int | None = None,
    ) -> None:
        super().__init__(node_shape, coefficients, unit, free_nodes, via_a, via_b, via_g, np.float32, threads)
        self._diagonal = np.ascontiguousarray(diagonal, dtype=np.float32).reshape(-1)
        if self._diagonal.size != self.size:
            raise ValueError("diagonal must hold one value per node")
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
            coarse = self.layers * (-(-(self.rows + 1) // block)) * (-(-(self.cols + 1) // block))
            if self._coarse_inverse.size != coarse * coarse:
                raise ValueError("coarse_inverse does not match the patch grid")

    def apply(self, vector: Any) -> np.ndarray:
        vector = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
        if vector.size != self.size:
            raise ValueError(f"vector has size {vector.size}, expected {self.size}")
        return _native.apply_layered_dc_q1(
            vector, self._coefficients, self._unit, self._free, self._free_mask,
            self._via_ptr, self._via_nbr, self._via_g, self.layers, self.rows, self.cols, self.threads,
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
        correction, iterations, relative_residual, applications = _native.pcg_layered_dc_q1(
            rhs_high, self._diagonal, self._coefficients, self._unit, self._free, self._free_mask,
            self._via_ptr, self._via_nbr, self._via_g, self.layers, self.rows, self.cols,
            self.coarse_block, self._coarse_inverse, float(inner_relative_tolerance),
            int(max_inner_iterations), self.threads,
        )
        return (
            np.asarray(correction, dtype=np.float32),
            int(iterations),
            float(relative_residual),
            int(applications),
        )


class NativeLayeredDCQ1High(_Prepared):
    """Host float64 operator for the outer MPIR residual and coarse assembly.

    Built before the two-level preconditioner, whose coarse matrix is
    assembled from FP64 applications, so it holds no preconditioner data.
    """

    kernel_name = NATIVE_HIGH_KERNEL_NAME

    def __init__(
        self,
        node_shape: tuple[int, int, int],
        coefficients: np.ndarray,
        unit: np.ndarray,
        free_nodes: np.ndarray,
        via_a: np.ndarray,
        via_b: np.ndarray,
        via_g: np.ndarray,
        *,
        threads: int | None = None,
    ) -> None:
        super().__init__(node_shape, coefficients, unit, free_nodes, via_a, via_b, via_g, np.float64, threads)

    def apply(self, vector: Any) -> np.ndarray:
        vector = np.ascontiguousarray(vector, dtype=np.float64).reshape(-1)
        if vector.size != self.size:
            raise ValueError(f"vector has size {vector.size}, expected {self.size}")
        return _native.apply_layered_dc_q1_f64(
            vector, self._coefficients, self._unit, self._free, self._free_mask,
            self._via_ptr, self._via_nbr, self._via_g, self.layers, self.rows, self.cols, self.threads,
        )
