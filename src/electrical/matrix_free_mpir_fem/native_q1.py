"""The C++ path of the scalar Maxwell Q1 operator.

With the core built, :class:`.frequency_domain.MatrixFreeScalarMaxwellOperator`
hands the prepared operator to ``electrical._pcbcore.fem.ScalarMaxwellSystem``
(GMRES variants below); without it the portable NumPy path is used.
"""

from __future__ import annotations

from .. import _backend

NATIVE_KERNEL_NAME = "cpp-fused-node-gather-q1"


def native_available() -> bool:
    return _backend.native_available()


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
