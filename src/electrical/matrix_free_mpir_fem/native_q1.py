"""Optional fused C++ low path for the scalar Maxwell Q1 operator.

The compiled module ``_scalar_maxwell_native`` is built in place with
``python -m electrical.matrix_free_mpir_fem.native.build``.  When it is
missing every entry point reports unavailability and the portable NumPy path
is used, so the experiment never changes results for users without a compiler.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

try:  # pragma: no cover - depends on the local build
    from . import _scalar_maxwell_native as _native
except ImportError:  # pragma: no cover
    _native = None


NATIVE_KERNEL_NAME = "cpp-fused-node-gather-q1"


def native_available() -> bool:
    return _native is not None


def native_requested() -> bool:
    """True when ``PCB_NATIVE_Q1`` selects the built extension by default."""

    flag = os.environ.get("PCB_NATIVE_Q1", "").strip().lower()
    return flag in {"1", "true", "yes", "on"} and native_available()


ORTHOGONALIZATIONS = ("mgs", "cgs2")


def native_orthogonalization() -> str:
    """Inner Gram-Schmidt variant; ``PCB_NATIVE_ORTHO`` selects ``mgs`` or ``cgs2``."""

    value = os.environ.get("PCB_NATIVE_ORTHO", "").strip().lower() or "mgs"
    if value not in ORTHOGONALIZATIONS:
        raise ValueError(f"PCB_NATIVE_ORTHO must be one of {ORTHOGONALIZATIONS}, got {value!r}")
    return value


def native_threads() -> int:
    """Thread count for the fused operator; ``PCB_NATIVE_THREADS`` overrides."""

    value = os.environ.get("PCB_NATIVE_THREADS")
    if value:
        return max(1, int(value))
    return 1


class NativeScalarMaxwellQ1:
    """Host complex64 operator and inner GMRES bound to one prepared mesh."""

    kernel_name = NATIVE_KERNEL_NAME

    def __init__(
        self,
        element_shape: tuple[int, int],
        inverse_mu: np.ndarray,
        reaction: np.ndarray,
        stiffness: np.ndarray,
        mass: np.ndarray,
        free_nodes: np.ndarray,
        diagonal: np.ndarray,
        *,
        threads: int | None = None,
        orthogonalization: str | None = None,
    ) -> None:
        if _native is None:
            raise ImportError(
                "the native extension is not built; run "
                "python -m electrical.matrix_free_mpir_fem.native.build"
            )
        self.element_rows = int(element_shape[0])
        self.element_columns = int(element_shape[1])
        self.size = (self.element_rows + 1) * (self.element_columns + 1)
        self.threads = threads if threads is not None else native_threads()
        self.orthogonalization = (
            orthogonalization if orthogonalization is not None else native_orthogonalization()
        )
        if self.orthogonalization not in ORTHOGONALIZATIONS:
            raise ValueError(
                f"orthogonalization must be one of {ORTHOGONALIZATIONS}, "
                f"got {self.orthogonalization!r}"
            )
        self._inverse_mu = np.ascontiguousarray(inverse_mu, dtype=np.complex64).reshape(-1)
        self._reaction = np.ascontiguousarray(reaction, dtype=np.complex64).reshape(-1)
        self._stiffness = np.ascontiguousarray(stiffness, dtype=np.complex64).reshape(-1)
        self._mass = np.ascontiguousarray(mass, dtype=np.complex64).reshape(-1)
        self._free = np.ascontiguousarray(free_nodes, dtype=np.uint8).reshape(-1)
        self._free_mask = self._free.astype(np.float32)
        self._diagonal = np.ascontiguousarray(diagonal, dtype=np.complex64).reshape(-1)

    def apply(self, vector: Any) -> np.ndarray:
        vector = np.ascontiguousarray(vector, dtype=np.complex64).reshape(-1)
        if vector.size != self.size:
            raise ValueError(f"vector has size {vector.size}, expected {self.size}")
        return _native.apply_q1(
            vector,
            self._inverse_mu,
            self._reaction,
            self._stiffness,
            self._mass,
            self._free,
            self._free_mask,
            self.element_rows,
            self.element_columns,
            self.threads,
        )

    def inner_gmres(
        self,
        rhs_high: np.ndarray,
        *,
        inner_relative_tolerance: float,
        max_inner_iterations: int,
        restart: int,
    ) -> tuple[np.ndarray, int, float, int]:
        rhs_high = np.ascontiguousarray(rhs_high, dtype=np.complex128).reshape(-1)
        if rhs_high.size != self.size:
            raise ValueError(f"rhs has size {rhs_high.size}, expected {self.size}")
        correction, iterations, relative_residual, applications = _native.gmres_q1(
            rhs_high,
            self._diagonal,
            self._inverse_mu,
            self._reaction,
            self._stiffness,
            self._mass,
            self._free,
            self._free_mask,
            self.element_rows,
            self.element_columns,
            float(inner_relative_tolerance),
            int(max_inner_iterations),
            int(restart),
            self.threads,
            self.orthogonalization == "cgs2",
        )
        return (
            np.asarray(correction, dtype=np.complex64),
            int(iterations),
            float(relative_residual),
            int(applications),
        )
