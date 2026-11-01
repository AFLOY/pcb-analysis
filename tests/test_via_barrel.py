"""A plated barrel's own inductance in place of its cell's column."""

from __future__ import annotations

import math

import numpy as np
import pytest

from electrical.sheet_peec import (
    SheetInductanceOperator,
    SheetLayer,
    SheetMesh,
    SheetStackup,
    Terminal,
    ViaBranch,
    solve_sheet_case,
)
from electrical.sheet_peec.sheet_inductance import (
    barrel_geometric_mean_distance,
    barrel_self_inductance,
    filament_mutual_inductance,
)
from electrical.sheet_peec.sheet_peec import _with_barrels

RHO, CU, SPAN, FREQUENCY = 1.724e-8, 3.5e-5, 1.6e-3, 1e6
OUTER, BORE = 0.15e-3, 0.125e-3  # a 0.3 mm drill with its 25 um plating
OMEGA = 2 * math.pi * FREQUENCY


class _NoVerticalInductance:
    """The operator with the vertical branches' inductance removed: the copper's share alone."""

    def __init__(self, base):
        self._base = base

    def __getattr__(self, name):
        return getattr(self._base, name)

    def apply(self, x, y, z=None):
        if z is None:
            return self._base.apply(x, y)
        fx, fy, _ = self._base.apply(x, y, z)
        return fx, fy, np.zeros_like(z)

    def apply_many(self, currents):
        return [self.apply(x, y, z) if z is not None else self.apply(x, y) for x, y, z in currents]

    def near_inductance(self, mesh, *args, **kwargs):
        near = self._base.near_inductance(mesh, *args, **kwargs).tocoo()
        first = len(mesh.branch_x) + len(mesh.branch_y)
        keep = (near.row < first) & (near.col < first)
        import scipy.sparse as sp

        return sp.csr_matrix((near.data[keep], (near.row[keep], near.col[keep])), shape=near.shape)

    def vertical_self_inductance(self, mesh):
        return np.zeros(len(mesh.via_branches))


def test_the_geometric_mean_distance_of_a_tube() -> None:
    assert barrel_geometric_mean_distance(1e-4) == pytest.approx(1e-4 * math.exp(-0.25), rel=1e-15)
    assert barrel_geometric_mean_distance(1e-4, 0.999e-4) == pytest.approx(1e-4, rel=1e-3)  # a thin wall: its radius
    g = barrel_geometric_mean_distance(OUTER, BORE)
    assert BORE < g < OUTER
    with pytest.raises(ValueError):
        barrel_geometric_mean_distance(1e-4, 1e-4)


def _pair(pitch: float, spacing: float, barrel: bool, copper_only: bool = False) -> float:
    """The loop L of two vias through 1.6 mm, ``spacing`` apart, joined by a one-cell trace on F.Cu.

    ``copper_only`` removes the vias' inductance: the trace's and pads' share.
    """
    cells = round(spacing / pitch)
    stackup = SheetStackup((SheetLayer("F.Cu", 0.0, CU, RHO), SheetLayer("B.Cu", -SPAN, CU, RHO)))
    rows, cols = 5, cells + 5
    occupancy = np.zeros((2, rows, cols), bool)
    go, back = (2, 2), (2, 2 + cells)
    occupancy[0, 2, 2 : 3 + cells] = True
    occupancy[1, go[0], go[1]] = occupancy[1, back[0], back[1]] = True
    extra = {"barrel_radius_m": OUTER, "barrel_bore_radius_m": BORE} if barrel else {}
    vias = (ViaBranch(*go, 1, 0, 1e-6, **extra), ViaBranch(*back, 1, 0, 1e-6, **extra))
    mesh = SheetMesh((rows, cols), pitch, stackup, occupancy, vias=vias)
    operator = SheetInductanceOperator((rows, cols), pitch, stackup, vertical_levels=mesh.vertical_levels)
    terminals = [Terminal("go", 1, (go,), 1.0), Terminal("back", 1, (back,), -1.0)]

    def loop(op):
        solution = solve_sheet_case(mesh, op, terminals, frequency_hz=FREQUENCY, tolerance=1e-10)
        v = solution.node_voltage[mesh.node_index[(1, *go)]] - solution.node_voltage[mesh.node_index[(1, *back)]]
        return v.imag / OMEGA

    return loop(_NoVerticalInductance(operator) if copper_only else operator)


@pytest.mark.parametrize("pitch", [0.1e-3, 0.2e-3, 0.25e-3])
def test_a_go_and_return_via_pair_has_the_barrels_inductance_at_any_cell_size(pitch) -> None:
    spacing = 1e-3
    expected = 2 * (barrel_self_inductance(SPAN, OUTER, BORE) - filament_mutual_inductance(SPAN, spacing))
    with_barrels = _pair(pitch, spacing, barrel=True)
    columns = _pair(pitch, spacing, barrel=False)
    copper = _pair(pitch, spacing, barrel=False, copper_only=True)
    # the vias' share of the loop: the barrels' closed form within 1 %
    assert with_barrels - copper == pytest.approx(expected, rel=0.01)
    # the cell's column was 13 % (0.25 mm) to 70 % (0.1 mm) above it
    assert columns - copper > 1.1 * expected


def test_a_three_by_three_array_carrying_one_current() -> None:
    pitch, step, n = 0.1e-3, 5, 3
    stackup = SheetStackup((SheetLayer("F.Cu", 0.0, CU, RHO), SheetLayer("B.Cu", -SPAN, CU, RHO)))
    size = (n - 1) * step + 3
    occupancy = np.zeros((2, size, size), bool)
    where = [(1 + i * step, 1 + j * step) for i in range(n) for j in range(n)]
    for cell in where:
        occupancy[:, cell[0], cell[1]] = True
    vias = tuple(ViaBranch(*cell, 1, 0, 1e-6, barrel_radius_m=OUTER, barrel_bore_radius_m=BORE) for cell in where)
    mesh = SheetMesh((size, size), pitch, stackup, occupancy, vias=vias)
    operator = SheetInductanceOperator((size, size), pitch, stackup, vertical_levels=mesh.vertical_levels)
    # each via its own island, the same current forced through each
    solution = solve_sheet_case(mesh, operator, [Terminal("top", 0, tuple(where), 1.0),
                                                 Terminal("bottom", 1, tuple(where), -1.0)],
                                frequency_hz=FREQUENCY, tolerance=1e-10)
    v = np.mean([solution.node_voltage[mesh.node_index[(0, *c)]] - solution.node_voltage[mesh.node_index[(1, *c)]]
                 for c in where])
    xy = [(r * pitch, c * pitch) for r, c in where]
    matrix = np.array([[barrel_self_inductance(SPAN, OUTER, BORE) if i == j else
                        filament_mutual_inductance(SPAN, math.dist(xy[i], xy[j])) for j in range(n * n)]
                       for i in range(n * n)])
    assert v.imag / OMEGA == pytest.approx(matrix.sum() / (n * n) ** 2, rel=0.01)


def test_overlapping_barrels_are_refused_and_columns_are_untouched() -> None:
    pitch = 0.1e-3
    stackup = SheetStackup((SheetLayer("F.Cu", 0.0, CU, RHO), SheetLayer("B.Cu", -SPAN, CU, RHO)))
    occupancy = np.ones((2, 3, 6), bool)
    # 0.2 mm apart with 0.15 mm radii: one conductor, not two
    vias = (ViaBranch(1, 1, 1, 0, 1e-6, barrel_radius_m=OUTER), ViaBranch(1, 3, 1, 0, 1e-6, barrel_radius_m=OUTER))
    mesh = SheetMesh((3, 6), pitch, stackup, occupancy, vias=vias)
    operator = SheetInductanceOperator((3, 6), pitch, stackup, vertical_levels=mesh.vertical_levels)
    with pytest.raises(ValueError, match="overlap"):
        _with_barrels(mesh, operator)
    plain = SheetMesh((3, 6), pitch, stackup, occupancy, vias=(ViaBranch(1, 1, 1, 0, 1e-6),))
    assert _with_barrels(plain, operator) is operator
    with pytest.raises(ValueError, match="bore"):
        ViaBranch(1, 1, 1, 0, 1e-6, barrel_radius_m=1e-4, barrel_bore_radius_m=1e-4)
