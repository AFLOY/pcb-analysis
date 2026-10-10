"""The C++ path of the layered-PCB DC conduction operator.

:class:`.pcb.MatrixFreePCBOperator` hands the whole prepared operator to
``electrical._pcbcore.fem.LayeredDCSystem`` when the core is built: element
coefficients, masks, via adjacency, the Jacobi diagonal and the two-level
coarse space, the FP64 and FP32 actions, the MPIR solve, the right-hand side
and the post-processing.  Without the build the NumPy implementation in
``pcb.py`` answers (``native=False`` keeps it); the two agree within the
solver tolerance, not bit for bit.
"""

from __future__ import annotations

from .. import _backend

NATIVE_KERNEL_NAME = "cpp-fused-node-gather-layered-dc-q1"
NATIVE_HIGH_KERNEL_NAME = "cpp-fused-node-gather-layered-dc-q1-fp64"


def native_available() -> bool:
    return _backend.native_available()


def native_requested() -> bool:
    """The built core is the default CPU path; ``native=False`` opts out."""

    return native_available()
