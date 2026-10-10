"""The C++ convolution operator of the uniform sheet mesh against the NumPy path.

Both multiply kernel spectra by the currents' spectra, layer pair by layer
pair, through pocketfft (the FFT NumPy uses); the kernel tables come from the
Hoer-Love closed form on each side, so the fluxes agree to the closed form's
precision (``CLOSED_FORM_RTOL``).  The C++ operator is the same at every thread budget, and the
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
from tests.tolerance import CLOSED_FORM_RTOL, DIRECT_RTOL, iterative_rtol

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


def _close(actual: np.ndarray, expected: np.ndarray, rtol: float = CLOSED_FORM_RTOL) -> None:
    # The kernel tables come from the closed form on both sides (C++ and
    # NumPy), whose far entries keep about seven digits.
    scale = float(np.max(np.abs(expected)))
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=rtol * scale)


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


def test_the_core_kernel_tables_match_the_numpy_closed_form() -> None:
    from electrical.sheet_peec.sheet_inductance import CellGeometry, build_kernel, build_vertical_kernel

    cell = CellGeometry(PITCH_M, PITCH_M, 35e-6)
    other = CellGeometry(PITCH_M, PITCH_M, 18e-6)
    shape = (60, 46)
    for axis in ("x", "y"):
        native = build_kernel(shape, cell, 0.4e-3, other=other, axis=axis)
        with _backend.use_reference():
            expected = build_kernel(shape, cell, 0.4e-3, other=other, axis=axis)
        # The few cells around the origin keep nearly every digit of their sum.
        near = np.zeros(shape, dtype=bool)
        near[np.ix_(np.r_[0:4, -3:0], np.r_[0:4, -3:0])] = True
        np.testing.assert_allclose(native[near], expected[near], rtol=DIRECT_RTOL)
        _close(native, expected)
    native = build_vertical_kernel(shape, PITCH_M, 0.4e-3, 1.2e-3, 0.8e-3)
    with _backend.use_reference():
        expected = build_vertical_kernel(shape, PITCH_M, 0.4e-3, 1.2e-3, 0.8e-3)
    _close(native, expected)


def test_the_core_closed_form_arrays_match_numpy_to_their_retained_precision() -> None:
    from electrical.sheet_peec.sheet_inductance import closed_form_mutual_inductance_arrays
    from tests.tolerance import closed_form_rtol

    rng = np.random.default_rng(11)
    count = 500
    a = (rng.uniform(0.05e-3, 0.5e-3, count), rng.uniform(0.05e-3, 0.5e-3, count), rng.uniform(18e-6, 70e-6, count))
    b = (rng.uniform(0.05e-3, 0.5e-3, count), rng.uniform(0.05e-3, 0.5e-3, count), rng.uniform(18e-6, 70e-6, count))
    offset = (rng.uniform(-1e-3, 1e-3, count), rng.uniform(-1e-3, 1e-3, count), rng.uniform(0.0, 1.6e-3, count))
    native = closed_form_mutual_inductance_arrays(a, b, offset, check_precision=False)
    with _backend.use_reference():
        expected = closed_form_mutual_inductance_arrays(a, b, offset, check_precision=False)
    # The reference's own retained fraction, recomputed term by term.
    from electrical.sheet_peec.sheet_inductance import _AXIS_SIGNS, _primitive

    total = np.zeros(count)
    largest = np.zeros(count)
    for sx, tx in _AXIS_SIGNS:
        x = offset[0] + sx * a[0] / 2.0 + tx * b[0] / 2.0
        for sy, ty in _AXIS_SIGNS:
            y = offset[1] + sy * a[1] / 2.0 + ty * b[1] / 2.0
            for sz, tz in _AXIS_SIGNS:
                z = offset[2] + sz * a[2] / 2.0 + tz * b[2] / 2.0
                value = _primitive(x, y, z)
                total += (sx * tx) * (sy * ty) * (sz * tz) * value
                largest = np.maximum(largest, np.abs(value))
    retained = np.abs(total) / largest
    for value, reference, kept in zip(native, expected, retained):
        assert value == pytest.approx(reference, rel=closed_form_rtol(float(kept)))


def test_the_core_near_field_lists_the_numpy_entries_exactly() -> None:
    # The same tables read by both: identical entries, duplicate vias included
    # (a cell keeps the position it was last given, visited where first seen).
    rows, cols = 8, 6
    occupancy = np.ones((3, rows, cols), dtype=bool)
    vias = (
        ViaBranch(1, 1, 0, 1, resistance_ohm=1e-3),
        ViaBranch(4, 2, 1, 2, resistance_ohm=1e-3),
        ViaBranch(1, 1, 0, 1, resistance_ohm=2e-3),
        ViaBranch(5, 4, 0, 1, resistance_ohm=1e-3),
    )
    mesh = SheetMesh((rows, cols), PITCH_M, STACKUP, occupancy, vias=vias)
    operator = SheetInductanceOperator((rows, cols), PITCH_M, STACKUP, vertical_levels=LEVELS)
    for radius in (1, 2):
        native = operator.near_inductance(mesh, radius_cells=radius)
        with _backend.use_reference():
            expected = operator.near_inductance(mesh, radius_cells=radius)
        assert native.shape == expected.shape
        assert (native != expected).nnz == 0
