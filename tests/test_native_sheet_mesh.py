"""The C++ mesh topology against the NumPy mesh: the same mesh, exactly.

Node numbering, branch lists, kept vias, incidence, resistances, the spread
terminal currents and the cell current densities are integer bookkeeping or a
fixed sequence of the same floating-point operations, so they agree bit for
bit, on uniform and graded grids, with islands and vias that miss copper.
"""

from __future__ import annotations

import numpy as np
import pytest

from electrical import backend as _backend
from electrical.matrix_free_mpir_fem import TensorGrid, graded_edges
from electrical.sheet_peec.sheet_operator import SheetLayer, SheetStackup
from electrical.sheet_peec.sheet_peec import SheetMesh, SheetSolution, Terminal, ViaBranch, _source_vector
from electrical.sheet_peec.sheet_results import cell_current_density_phasor

pytestmark = pytest.mark.skipif(not _backend.native_available(), reason="electrical._pcbcore is not built")

STACKUP = SheetStackup(
    (SheetLayer("F", 0.0, 35e-6), SheetLayer("In", -0.4e-3, 18e-6), SheetLayer("B", -1.6e-3, 70e-6))
)


def _meshes(graded: bool):
    rng = np.random.default_rng(7 if graded else 3)
    if graded:
        xe = graded_edges(0.0, 3e-3, coarse_pitch_m=0.5e-3, fine_pitch_m=0.1e-3, refine_m=[(1.0e-3, 1.4e-3)], margin_m=0.2e-3)
        ye = graded_edges(0.0, 2e-3, coarse_pitch_m=0.4e-3, fine_pitch_m=0.1e-3, refine_m=[(0.8e-3, 1.1e-3)], margin_m=0.2e-3)
        grid = TensorGrid(xe, ye)
        rows, cols = grid.shape
        pitch = None
    else:
        grid = None
        rows, cols = 9, 13
        pitch = 2e-4
    occupancy = rng.random((3, rows, cols)) < 0.8
    vias = tuple(
        ViaBranch(int(rng.integers(rows)), int(rng.integers(cols)), lower, lower + 1, resistance_ohm=float(rng.uniform(1e-4, 1e-3)))
        for lower in (0, 1, 0, 1, 0)
    )
    native = SheetMesh((rows, cols), pitch, STACKUP, occupancy, vias=vias, grid=grid)
    with _backend.use_reference():
        reference = SheetMesh((rows, cols), pitch, STACKUP, occupancy, vias=vias, grid=grid)
    assert native.__dict__["_topology"] is not None and reference.__dict__["_topology"] is None
    return native, reference


@pytest.mark.parametrize("graded", [False, True])
def test_the_core_builds_the_same_mesh(graded: bool) -> None:
    native, reference = _meshes(graded)
    assert native.node_index == reference.node_index
    assert list(native.node_index) == list(reference.node_index)
    assert native.branch_x == reference.branch_x
    assert native.branch_y == reference.branch_y
    assert native.via_branches == reference.via_branches
    assert (native.incidence() != reference.incidence()).nnz == 0
    assert native.resistances().tobytes() == reference.resistances().tobytes()


@pytest.mark.parametrize("graded", [False, True])
def test_the_core_spreads_terminals_and_reads_densities_the_same_way(graded: bool) -> None:
    native, reference = _meshes(graded)
    rows, cols = native.shape
    terminals = [
        Terminal("in", 0, tuple((r, 0) for r in range(rows)) + ((rows + 5, 0),), 1.5 + 0.25j),
        Terminal("out", 2, tuple((r, cols - 1) for r in range(rows)), -1.5 - 0.25j),
    ]
    assert _source_vector(native, terminals).tobytes() == _source_vector(reference, terminals).tobytes()
    rng = np.random.default_rng(1)
    current = rng.standard_normal(native.branch_count) + 1j * rng.standard_normal(native.branch_count)
    solution = SheetSolution(np.zeros(native.node_count, complex), current, 1e6, 1, 0.0, True, 0)
    densities = cell_current_density_phasor(native, solution)
    with _backend.use_reference():
        expected = cell_current_density_phasor(reference, solution)
    assert list(densities) == list(expected)
    assert all(densities[cell] == expected[cell] for cell in expected)


def test_a_terminal_off_the_copper_is_refused() -> None:
    native, _ = _meshes(False)
    occupied = {cell for cell in native.node_index}
    rows, cols = native.shape
    empty = [(r, c) for r in range(rows) for c in range(cols) if (0, r, c) not in occupied]
    terminals = [Terminal("nowhere", 0, tuple(empty[:2]), 1.0), Terminal("out", 0, ((0, 0),), -1.0)]
    with pytest.raises(ValueError, match="terminal nowhere has no cell on the conductor"):
        _source_vector(native, terminals)
