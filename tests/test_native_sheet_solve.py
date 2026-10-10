"""The C++ sheet solve against the NumPy reference.

The core reduces the mesh the same way (components, the undriven ones left
out, one grounded node per driven component), solves DC by SuperLU and AC by
GMRES with SciPy's algorithm and the same three preconditioners.  Answers
agree to the solver tolerance, iteration counts to within a few, and the
core's answer is bit-identical at every thread budget.
"""

from __future__ import annotations

import numpy as np
import pytest

from electrical import backend as _backend
from electrical.matrix_free_mpir_fem import TensorGrid, graded_edges
from electrical.sheet_peec.sheet_operator import SheetInductanceOperator, SheetLayer, SheetStackup
from electrical.sheet_peec.sheet_peec import SheetMesh, Terminal, ViaBranch, solve_sheet_case
from electrical.sheet_peec.sheet_pfft import PfftSheetInductanceOperator
from electrical.threads import thread_budget_scope
from tests.tolerance import DIRECT_RTOL, iterative_rtol

pytestmark = pytest.mark.skipif(not _backend.native_available(), reason="electrical._pcbcore is not built")

PITCH_M = 2e-4
STACKUP = SheetStackup((SheetLayer("F", 0.0, 35e-6), SheetLayer("B", -1.6e-3, 35e-6)))
TOLERANCE = 1e-10


def _uniform_case():
    rows, cols = 12, 15
    occupancy = np.ones((2, rows, cols), dtype=bool)
    occupancy[:, 5, 3:12] = False  # a slot the current has to go around
    occupancy[0, 0:2, 13:15] = True
    occupancy[0, 2, 13:15] = False  # an undriven island on F
    occupancy[0, 0:2, 12] = False
    vias = (
        ViaBranch(2, 2, 0, 1, resistance_ohm=2e-4),
        ViaBranch(9, 12, 0, 1, resistance_ohm=2e-4),
        ViaBranch(9, 12, 0, 1, resistance_ohm=3e-4),  # a second barrel on one cell
    )
    mesh = SheetMesh((rows, cols), PITCH_M, STACKUP, occupancy, vias=vias)
    operator = SheetInductanceOperator((rows, cols), PITCH_M, STACKUP, vertical_levels=mesh.vertical_levels)
    terminals = [
        Terminal("in", 0, tuple((r, 0) for r in range(3, 9)), 2.0),
        Terminal("out", 1, tuple((r, cols - 1) for r in range(3, 9)), -2.0),
    ]
    return mesh, operator, terminals


def _graded_case():
    xe = graded_edges(0.0, 3e-3, coarse_pitch_m=0.5e-3, fine_pitch_m=0.1e-3, refine_m=[(1.0e-3, 1.4e-3)], margin_m=0.2e-3)
    ye = graded_edges(0.0, 2e-3, coarse_pitch_m=0.4e-3, fine_pitch_m=0.1e-3, refine_m=[(0.8e-3, 1.1e-3)], margin_m=0.2e-3)
    grid = TensorGrid(xe, ye)
    rows, cols = grid.shape
    vias = (ViaBranch(1, 2, 0, 1, resistance_ohm=1e-3), ViaBranch(rows - 2, cols - 3, 0, 1, resistance_ohm=1e-3))
    mesh = SheetMesh((rows, cols), None, STACKUP, np.ones((2, rows, cols), bool), grid=grid, vias=vias)
    operator = PfftSheetInductanceOperator(mesh, order=3, near_radius_cells=4)
    terminals = [
        Terminal("in", 0, tuple((r, 0) for r in range(rows)), 1.0),
        Terminal("out", 0, tuple((r, cols - 1) for r in range(rows)), -1.0),
    ]
    return mesh, operator, terminals


def _compare(mesh, operator, terminals, **options):
    native = solve_sheet_case(mesh, operator, terminals, **options)
    with _backend.use_reference():
        expected = solve_sheet_case(mesh, operator, terminals, **options)
    rtol = DIRECT_RTOL if options.get("frequency_hz", 0.0) == 0.0 else iterative_rtol(TOLERANCE)
    for field in ("branch_current", "node_voltage"):
        actual, reference = getattr(native, field), getattr(expected, field)
        scale = float(np.max(np.abs(reference)))
        np.testing.assert_allclose(actual, reference, rtol=0, atol=rtol * scale)
    assert native.converged and expected.converged
    assert native.grounded_node == expected.grounded_node
    assert native.undriven_nodes == expected.undriven_nodes
    return native, expected


def test_the_core_dc_solve_matches_the_reference() -> None:
    mesh, operator, terminals = _uniform_case()
    native, expected = _compare(mesh, operator, terminals, frequency_hz=0.0)
    assert native.undriven_nodes > 0
    assert native.residual < 1e-12


@pytest.mark.parametrize("preconditioner", ["near", "block", "diagonal", "auto"])
def test_the_core_ac_solve_matches_the_reference(preconditioner: str) -> None:
    mesh, operator, terminals = _uniform_case()
    native, expected = _compare(
        mesh, operator, terminals, frequency_hz=2e6, tolerance=TOLERANCE, preconditioner=preconditioner
    )
    assert abs(native.iterations - expected.iterations) <= max(3, expected.iterations // 10)


@pytest.mark.parametrize("preconditioner", ["near", "block"])
def test_the_core_solves_on_the_pfft_operator(preconditioner: str) -> None:
    mesh, operator, terminals = _graded_case()
    native, expected = _compare(
        mesh, operator, terminals, frequency_hz=1e6, tolerance=TOLERANCE, preconditioner=preconditioner
    )
    assert abs(native.iterations - expected.iterations) <= max(3, expected.iterations // 10)


def test_the_core_solve_is_the_same_at_every_thread_budget() -> None:
    mesh, operator, terminals = _uniform_case()
    results = []
    for threads in (1, 2, 3, 8):
        with thread_budget_scope(threads):
            solution = solve_sheet_case(mesh, operator, terminals, frequency_hz=2e6, tolerance=TOLERANCE)
        results.append(solution.branch_current.tobytes() + solution.node_voltage.tobytes())
    assert all(result == results[0] for result in results)


def test_an_unbalanced_component_is_refused_by_the_core() -> None:
    mesh, operator, _ = _uniform_case()
    terminals = [Terminal("in", 0, ((4, 0),), 1.0), Terminal("out", 0, ((0, 14),), -1.0)]  # out sits on the island
    with pytest.raises(ValueError, match="close its own current"):
        solve_sheet_case(mesh, operator, terminals, frequency_hz=1e6)
