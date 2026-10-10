"""The C++ path of the hexahedral Q1 conduction operator.

With the core built, :class:`.operator.MatrixFreeThermalOperator` hands the
whole prepared operator to ``electrical._pcbcore.thermal.ThermalSystem`` and
the solves run in ``electrical._pcbcore.thermal`` (see :mod:`.native_system`);
``native=False``, a CUDA runtime or a missing build keep the NumPy/CuPy
implementation.  The two agree within the solver tolerance, not bit for bit.
"""

from __future__ import annotations

from electrical import _backend

NATIVE_KERNEL_NAME = "cpp-fused-node-gather-hex-q1"
NATIVE_HIGH_KERNEL_NAME = "cpp-fused-node-gather-hex-q1-fp64"


def native_available() -> bool:
    return _backend.native_available()


def native_requested() -> bool:
    """The built core is the default CPU path; ``native=False`` opts out."""

    return native_available()
