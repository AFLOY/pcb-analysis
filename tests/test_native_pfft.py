"""The C++ near-field pair coupling of the pFFT operator against its NumPy fallback.

``_grid_pair_coupling`` credits each near branch pair with ``P_i K P_j^T``,
the coupling the FFT path already counts, so the precorrection can subtract
it.  The native kernel and the NumPy einsum evaluate the same stencil sums in
a different order; the dense operator and a solve built on either agree to
roundoff, and the native operator is the same at every thread budget.
"""

from __future__ import annotations

import numpy as np
import pytest

from electrical.matrix_free_mpir_fem import TensorGrid, graded_edges
from electrical.sheet_peec.sheet_operator import SheetLayer, SheetStackup
from electrical.sheet_peec.sheet_peec import SheetMesh, Terminal, solve_sheet_case
from electrical.sheet_peec.sheet_pfft import PfftSheetInductanceOperator, native_available
from electrical.threads import thread_budget_scope
from tests.tolerance import DIRECT_RTOL, iterative_rtol

pytestmark = pytest.mark.skipif(
    not native_available(),
    reason="sheet pFFT native extension not built; run cmake -S . -B build/native && cmake --build build/native",
)

STACKUP = SheetStackup((SheetLayer("F", 0.0, 35e-6), SheetLayer("B", -1.6e-3, 35e-6)))


def _graded_mesh() -> SheetMesh:
    xe = graded_edges(0.0, 4e-3, coarse_pitch_m=0.5e-3, fine_pitch_m=0.1e-3, refine_m=[(1.5e-3, 2.0e-3)], margin_m=0.2e-3)
    ye = graded_edges(0.0, 2.5e-3, coarse_pitch_m=0.5e-3, fine_pitch_m=0.125e-3, refine_m=[(1.0e-3, 1.3e-3)], margin_m=0.2e-3)
    grid = TensorGrid(xe, ye)
    rows, cols = grid.shape
    return SheetMesh((rows, cols), None, STACKUP, np.ones((2, rows, cols), bool), grid=grid)


def test_the_native_dense_operator_matches_the_numpy_fallback() -> None:
    mesh = _graded_mesh()
    reference = PfftSheetInductanceOperator(mesh, order=3, near_radius_cells=4, native=False)
    native = PfftSheetInductanceOperator(mesh, order=3, near_radius_cells=4, native=True)
    assert native.use_native and not reference.use_native
    for axis in ("x", "y"):
        expected = reference.dense_matrix(axis)
        actual = native.dense_matrix(axis)
        np.testing.assert_allclose(actual, expected, rtol=DIRECT_RTOL, atol=DIRECT_RTOL * float(np.max(np.abs(expected))))


def test_a_solve_on_the_native_operator_matches_the_numpy_fallback() -> None:
    mesh = _graded_mesh()
    rows, cols = mesh.shape
    terminals = [
        Terminal("in", 0, tuple((row, 0) for row in range(rows)), 1.0),
        Terminal("out", 0, tuple((row, cols - 1) for row in range(rows)), -1.0),
    ]
    tolerance = 1e-10
    solutions = {}
    for native in (False, True):
        operator = PfftSheetInductanceOperator(mesh, order=3, near_radius_cells=4, native=native)
        solutions[native] = solve_sheet_case(mesh, operator, terminals, frequency_hz=1e6, tolerance=tolerance)
    reference, candidate = solutions[False], solutions[True]
    scale = float(np.max(np.abs(reference.branch_current)))
    np.testing.assert_allclose(
        candidate.branch_current, reference.branch_current, rtol=0, atol=iterative_rtol(tolerance) * scale
    )


def test_the_native_operator_is_the_same_at_every_thread_budget() -> None:
    mesh = _graded_mesh()
    matrices = []
    for threads in (1, 2, 3, 8):
        with thread_budget_scope(threads):
            operator = PfftSheetInductanceOperator(mesh, order=3, near_radius_cells=4, native=True)
            matrices.append(operator.dense_matrix("x").tobytes())
    assert all(matrix == matrices[0] for matrix in matrices)


def _graded_mesh_with_vias() -> SheetMesh:
    from electrical.sheet_peec.sheet_peec import ViaBranch

    xe = graded_edges(0.0, 3e-3, coarse_pitch_m=0.5e-3, fine_pitch_m=0.1e-3, refine_m=[(1.0e-3, 1.4e-3)], margin_m=0.2e-3)
    ye = graded_edges(0.0, 2e-3, coarse_pitch_m=0.4e-3, fine_pitch_m=0.1e-3, refine_m=[(0.8e-3, 1.1e-3)], margin_m=0.2e-3)
    grid = TensorGrid(xe, ye)
    rows, cols = grid.shape
    vias = (ViaBranch(1, 2, 0, 1, resistance_ohm=1e-3), ViaBranch(rows - 2, cols - 3, 0, 1, resistance_ohm=1e-3))
    return SheetMesh((rows, cols), None, STACKUP, np.ones((2, rows, cols), bool), grid=grid, vias=vias)


def test_the_core_applies_the_same_pfft_operator() -> None:
    from electrical import backend as _backend

    if not _backend.native_available():
        pytest.skip("electrical._pcbcore is not built")
    mesh = _graded_mesh_with_vias()
    rows, cols = mesh.shape
    native = PfftSheetInductanceOperator(mesh, order=3, near_radius_cells=4)
    with _backend.use_reference():
        reference = PfftSheetInductanceOperator(mesh, order=3, near_radius_cells=4)
    assert native._core is not None and reference._core is None
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, rows, cols))
    y = rng.standard_normal((2, rows, cols))
    x[:, :, -1] = 0.0
    y[:, -1, :] = 0.0
    z = rng.standard_normal((len(mesh.vertical_levels), rows, cols))
    for actual, expected in zip(native.apply(x, y, z), reference.apply(x, y, z)):
        scale = float(np.max(np.abs(expected)))
        np.testing.assert_allclose(actual, expected, rtol=DIRECT_RTOL, atol=DIRECT_RTOL * scale)
    results = []
    for threads in (1, 2, 3, 8):
        with thread_budget_scope(threads):
            results.append(b"".join(part.tobytes() for part in native.apply(x, y, z)))
    assert all(result == results[0] for result in results)


def test_the_core_builds_the_same_pfft_parts() -> None:
    from electrical import backend as _backend
    from tests.tolerance import CLOSED_FORM_RTOL

    if not _backend.native_available():
        pytest.skip("electrical._pcbcore is not built")
    mesh = _graded_mesh_with_vias()
    native = PfftSheetInductanceOperator(mesh, order=3, near_radius_cells=4, preconditioner_radius_cells=2)
    with _backend.use_reference():
        reference = PfftSheetInductanceOperator(mesh, order=3, near_radius_cells=4, preconditioner_radius_cells=2)
    families = [("x", native._projection["x"], reference._projection["x"]), ("y", native._projection["y"], reference._projection["y"]), ("z", native._projection_z, reference._projection_z)]
    for name, actual, expected in families:
        # The same stencil sums in the same order: identical.
        assert (actual != expected).nnz == 0, name
    pairs = [
        (native._correction["x"], reference._correction["x"]),
        (native._correction["y"], reference._correction["y"]),
        (native._correction_z, reference._correction_z),
        (native._near_exact["x"], reference._near_exact["x"]),
        (native._near_exact["y"], reference._near_exact["y"]),
        (native._near_exact_z, reference._near_exact_z),
    ]
    for actual, expected in pairs:
        # Closed forms on both sides (C++ and NumPy), which keep about seven digits far out.
        assert actual.shape == expected.shape and actual.nnz == expected.nnz
        scale = float(np.max(np.abs(expected.data)))
        assert np.max(np.abs((actual - expected).data), initial=0.0) <= CLOSED_FORM_RTOL * scale
    for actual, expected in ((native._self["x"], reference._self["x"]), (native._self["y"], reference._self["y"]), (native._self_z, reference._self_z)):
        np.testing.assert_allclose(actual, expected, rtol=DIRECT_RTOL)
