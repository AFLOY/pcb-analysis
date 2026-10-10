"""The C++ convolution operator of the uniform sheet mesh against the NumPy path.

Both multiply the same kernel spectra by the currents' spectra, layer pair by
layer pair, through pocketfft (the FFT NumPy uses), so the fluxes agree to
FFT roundoff.  The C++ operator is the same at every thread budget, and the
near-field matrix read off its tables matches the one read off the NumPy
spectra.
"""

from __future__ import annotations

import numpy as np
import pytest

from electrical import _backend
from electrical.sheet_peec.sheet_operator import SheetInductanceOperator, SheetLayer, SheetStackup
from electrical.sheet_peec.sheet_peec import SheetMesh, Terminal, ViaBranch, solve_sheet_case
from electrical.threads import thread_budget_scope
from tests.tolerance import DIRECT_RTOL, iterative_rtol

pytestmark = pytest.mark.skipif(not _backend.native_available(), reason="electrical._pcbcore is not built")

PITCH_M = 2e-4
STACKUP = SheetStackup(
    (
        SheetLayer("F", 0.0, 35e-6),
        SheetLayer("In1", -0.4e-3, 18e-6),
        SheetLayer("B", -1.6e-3, 35e-6),
    )
)
LEVELS = ((0, 1), (1, 2))


def _operators(rows: int, cols: int):
    native = SheetInductanceOperator((rows, cols), PITCH_M, STACKUP, vertical_levels=LEVELS)
    with _backend.use_reference():
        reference = SheetInductanceOperator((rows, cols), PITCH_M, STACKUP, vertical_levels=LEVELS)
    assert native._native is not None and reference._native is None
    return native, reference


def _currents(rows: int, cols: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((3, rows, cols))
    y = rng.standard_normal((3, rows, cols))
    z = rng.standard_normal((2, rows, cols))
    return x, y, z


def _close(actual: np.ndarray, expected: np.ndarray) -> None:
    scale = float(np.max(np.abs(expected)))
    np.testing.assert_allclose(actual, expected, rtol=DIRECT_RTOL, atol=DIRECT_RTOL * scale)


@pytest.mark.parametrize("shape", [(1, 1), (5, 8), (17, 11)])
def test_the_core_applies_the_same_operator(shape) -> None:
    native, reference = _operators(*shape)
    x, y, z = _currents(*shape, seed=sum(shape))
    for out, expected in zip(native.apply(x, y, z), reference.apply(x, y, z)):
        _close(out, expected)
    for out, expected in zip(native.apply(x, y), reference.apply(x, y)):
        _close(out, expected)


def test_the_core_operator_is_the_same_at_every_thread_budget() -> None:
    x, y, z = _currents(13, 9, seed=3)
    results = []
    for threads in (1, 2, 3, 8):
        with thread_budget_scope(threads):
            operator = SheetInductanceOperator((13, 9), PITCH_M, STACKUP, vertical_levels=LEVELS)
            results.append(b"".join(part.tobytes() for part in operator.apply(x, y, z)))
    assert all(result == results[0] for result in results)


def _mesh(rows: int, cols: int) -> SheetMesh:
    occupancy = np.ones((3, rows, cols), dtype=bool)
    vias = tuple(ViaBranch(r, c, 0, 1, resistance_ohm=1e-3) for r, c in ((1, 1), (rows - 2, cols - 2)))
    vias += tuple(ViaBranch(r, c, 1, 2, resistance_ohm=1e-3) for r, c in ((2, 3), (rows - 3, 1)))
    return SheetMesh((rows, cols), PITCH_M, STACKUP, occupancy, vias=vias)


def test_the_near_field_matrix_matches_the_numpy_tables() -> None:
    mesh = _mesh(9, 7)
    native, reference = _operators(9, 7)
    expected = reference.near_inductance(mesh).toarray()
    _close(native.near_inductance(mesh).toarray(), expected)


def test_a_solve_on_the_core_operator_matches_the_numpy_one() -> None:
    rows, cols = 9, 7
    mesh = _mesh(rows, cols)
    native, reference = _operators(rows, cols)
    terminals = [
        Terminal("in", 0, tuple((row, 0) for row in range(rows)), 1.0),
        Terminal("out", 2, tuple((row, cols - 1) for row in range(rows)), -1.0),
    ]
    tolerance = 1e-10
    solved = solve_sheet_case(mesh, native, terminals, frequency_hz=1e6, tolerance=tolerance)
    expected = solve_sheet_case(mesh, reference, terminals, frequency_hz=1e6, tolerance=tolerance)
    scale = float(np.max(np.abs(expected.branch_current)))
    np.testing.assert_allclose(
        solved.branch_current, expected.branch_current, rtol=0, atol=iterative_rtol(tolerance) * scale
    )
