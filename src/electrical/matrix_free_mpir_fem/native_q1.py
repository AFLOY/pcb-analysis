"""Optional fused C++ low path for the scalar Maxwell Q1 operator.

The compiled module ``_scalar_maxwell_native`` is built in place with
``python -m electrical.matrix_free_mpir_fem.native.build``.  When it is
missing every entry point reports unavailability and the portable NumPy path
is used, so the experiment never changes results for users without a compiler.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..threads import thread_budget

try:  # pragma: no cover - depends on the local build
    from electrical._pcbcore import scalar_maxwell as _native
except ImportError:  # pragma: no cover
    try:  # a module built on its own by native/build.py
        from . import _scalar_maxwell_native as _native
    except ImportError:
        _native = None


NATIVE_KERNEL_NAME = "cpp-fused-node-gather-q1"


def native_available() -> bool:
    return _native is not None


def native_requested() -> bool:
    """The built extension is the default CPU path; ``native=False`` opts out."""

    return native_available()


ORTHOGONALIZATIONS = ("mgs", "cgs2")


def native_orthogonalization() -> str:
    """Inner Gram-Schmidt variant: modified Gram-Schmidt (``cgs2`` is the alternative)."""

    return "mgs"


DOT_ACCUMULATIONS = ("float64", "float32")


def native_dot_accumulation() -> str:
    """Gram-Schmidt dot-product accumulation.

    ``float64`` accumulates every term in double like the portable runtime;
    ``float32`` accumulates 1,024-element blocks in float and sums the blocks
    in double.  The default is ``float64``.
    """

    return "float64"


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
        orthogonalization: str | None = None,
        dot_accumulation: str | None = None,
    ) -> None:
        if _native is None:
            raise ImportError(
                "the native extension is not built; run "
                "python -m electrical.matrix_free_mpir_fem.native.build"
            )
        self.element_rows = int(element_shape[0])
        self.element_columns = int(element_shape[1])
        self.size = (self.element_rows + 1) * (self.element_columns + 1)
        # OpenMP team: the whole process budget, since the kernel runs alone.
        self.threads = thread_budget()
        self.orthogonalization = (
            orthogonalization if orthogonalization is not None else native_orthogonalization()
        )
        if self.orthogonalization not in ORTHOGONALIZATIONS:
            raise ValueError(
                f"orthogonalization must be one of {ORTHOGONALIZATIONS}, "
                f"got {self.orthogonalization!r}"
            )
        self.dot_accumulation = (
            dot_accumulation if dot_accumulation is not None else native_dot_accumulation()
        )
        if self.dot_accumulation not in DOT_ACCUMULATIONS:
            raise ValueError(
                f"dot_accumulation must be one of {DOT_ACCUMULATIONS}, "
                f"got {self.dot_accumulation!r}"
            )
        self._inverse_mu = np.ascontiguousarray(inverse_mu, dtype=np.complex64).reshape(-1)
        self._reaction = np.ascontiguousarray(reaction, dtype=np.complex64).reshape(-1)
        self._stiffness = np.ascontiguousarray(stiffness, dtype=np.complex64).reshape(-1)
        self._mass = np.ascontiguousarray(mass, dtype=np.complex64).reshape(-1)
        self._free = np.ascontiguousarray(free_nodes, dtype=np.uint8).reshape(-1)
        self._free_mask = self._free.astype(np.float32)
        self._diagonal = np.ascontiguousarray(diagonal, dtype=np.complex64).reshape(-1)

        self._inverse_mu_f64 = np.ascontiguousarray(inverse_mu, dtype=np.complex128).reshape(-1)
        self._reaction_f64 = np.ascontiguousarray(reaction, dtype=np.complex128).reshape(-1)
        self._stiffness_f64 = np.ascontiguousarray(stiffness, dtype=np.complex128).reshape(-1)
        self._mass_f64 = np.ascontiguousarray(mass, dtype=np.complex128).reshape(-1)
        self._free_mask_f64 = self._free.astype(np.float64)

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
            self.dot_accumulation == "float32",
        )
        return (
            np.asarray(correction, dtype=np.complex64),
            int(iterations),
            float(relative_residual),
            int(applications),
        )

    def solve_mpir(
        self,
        rhs_high: np.ndarray,
        config: Any,
        initial_guess: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool, int, int, float, int, int, list[tuple[int, float, int, float]]]:
        rhs_high = np.ascontiguousarray(rhs_high, dtype=np.complex128).reshape(-1)
        if rhs_high.size != self.size:
            raise ValueError(f"rhs has size {rhs_high.size}, expected {self.size}")
        init_guess = (
            np.ascontiguousarray(initial_guess, dtype=np.complex128).reshape(-1)
            if initial_guess is not None
            else np.zeros(0, dtype=np.complex128)
        )
        sol, conv, outer, inner, rel, n_high, n_low, history = _native.solve_mpir_scalar_maxwell_q1(
            rhs_high,
            init_guess,
            self._diagonal,
            self._inverse_mu,
            self._reaction,
            self._stiffness,
            self._mass,
            self._free,
            self._free_mask,
            self._inverse_mu_f64,
            self._reaction_f64,
            self._stiffness_f64,
            self._mass_f64,
            self._free_mask_f64,
            self.element_rows,
            self.element_columns,
            float(config.relative_tolerance),
            float(config.absolute_tolerance),
            float(config.inner_relative_tolerance),
            int(config.max_outer_iterations),
            int(config.max_inner_iterations),
            int(config.gmres_restart),
            self.threads,
            self.orthogonalization == "cgs2",
            self.dot_accumulation == "float32",
        )
        return (
            np.asarray(sol, dtype=np.complex128),
            bool(conv),
            int(outer),
            int(inner),
            float(rel),
            int(n_high),
            int(n_low),
            [(int(o), float(h), int(k), float(r)) for o, h, k, r in history],
        )

