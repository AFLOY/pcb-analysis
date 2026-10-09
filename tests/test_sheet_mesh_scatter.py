"""SheetMesh scatter/gather through index arrays write what the loops wrote."""

from __future__ import annotations

import numpy as np
import pytest

from electrical.sheet_peec.sheet_operator import SheetLayer, SheetStackup
from electrical.sheet_peec.sheet_peec import SheetMesh, ViaBranch

PITCH_M = 2e-4
COPPER_M = 3.5e-5
RESISTIVITY = 1.724e-8


def _stackup(layers: int) -> SheetStackup:
    return SheetStackup(
        tuple(
            SheetLayer(f"L{index}", -index * 4e-4, COPPER_M, RESISTIVITY)
            for index in range(layers)
        )
    )


def _loop_scatter(mesh, currents):
    rows, cols = mesh.shape
    grid_x = np.zeros((len(mesh.stackup), rows, cols))
    grid_y = np.zeros_like(grid_x)
    for index, (layer, row, col) in enumerate(mesh.branch_x):
        grid_x[layer, row, col] = currents[index]
    offset = len(mesh.branch_x)
    for index, (layer, row, col) in enumerate(mesh.branch_y):
        grid_y[layer, row, col] = currents[offset + index]
    return grid_x, grid_y


def _loop_scatter_vertical(mesh, currents):
    levels = mesh.vertical_levels
    index_of = {key: position for position, key in enumerate(levels)}
    rows, cols = mesh.shape
    grid = np.zeros((len(levels), rows, cols))
    offset = len(mesh.branch_x) + len(mesh.branch_y)
    for index, via in enumerate(mesh.via_branches):
        grid[index_of[(via.lower_layer, via.upper_layer)], via.row, via.col] = currents[offset + index]
    return grid


def _loop_gather(mesh, grid_x, grid_y, grid_z):
    values = np.zeros(mesh.branch_count)
    for index, (layer, row, col) in enumerate(mesh.branch_x):
        values[index] = grid_x[layer, row, col]
    offset = len(mesh.branch_x)
    for index, (layer, row, col) in enumerate(mesh.branch_y):
        values[offset + index] = grid_y[layer, row, col]
    if grid_z is not None:
        index_of = {key: position for position, key in enumerate(mesh.vertical_levels)}
        offset += len(mesh.branch_y)
        for index, via in enumerate(mesh.via_branches):
            values[offset + index] = grid_z[index_of[(via.lower_layer, via.upper_layer)], via.row, via.col]
    return values


def _random_mesh(seed: int, layers: int, duplicate_via: bool = False) -> SheetMesh:
    rng = np.random.default_rng(seed)
    rows, cols = 11, 13
    occupancy = rng.random((layers, rows, cols)) < 0.75
    occupancy[:, rows // 2, :] = True  # keep it one conductor-ish and non-empty
    vias = []
    for _ in range(9):
        row, col = int(rng.integers(rows)), int(rng.integers(cols))
        lower = int(rng.integers(layers - 1)) if layers > 1 else 0
        if layers > 1:
            occupancy[lower, row, col] = occupancy[lower + 1, row, col] = True
            vias.append(ViaBranch(row, col, lower, lower + 1, resistance_ohm=1e-3))
    if duplicate_via and vias:
        vias.append(ViaBranch(vias[0].row, vias[0].col, vias[0].lower_layer, vias[0].upper_layer, resistance_ohm=2e-3))
    return SheetMesh((rows, cols), PITCH_M, _stackup(layers), occupancy, vias=tuple(vias))


@pytest.mark.parametrize("seed,layers,duplicate", [(0, 1, False), (1, 2, False), (2, 3, False), (3, 3, True)])
def test_scatter_and_gather_match_the_loops_bit_for_bit(seed, layers, duplicate):
    mesh = _random_mesh(seed, layers, duplicate)
    rng = np.random.default_rng(100 + seed)
    currents = rng.standard_normal(mesh.branch_count)

    grid_x, grid_y = mesh.scatter(currents)
    ref_x, ref_y = _loop_scatter(mesh, currents)
    assert grid_x.tobytes() == ref_x.tobytes()
    assert grid_y.tobytes() == ref_y.tobytes()

    grid_z = None
    if mesh.via_branches:
        grid_z = mesh.scatter_vertical(currents)
        assert grid_z.tobytes() == _loop_scatter_vertical(mesh, currents).tobytes()

    field_x = rng.standard_normal(grid_x.shape)
    field_y = rng.standard_normal(grid_y.shape)
    field_z = rng.standard_normal(grid_z.shape) if grid_z is not None else None
    values = mesh.gather(field_x, field_y)
    if field_z is not None:
        mesh.gather_vertical(field_z, values)
    assert values.tobytes() == _loop_gather(mesh, field_x, field_y, field_z).tobytes()


def test_a_cell_with_two_vertical_branches_keeps_the_last_one():
    mesh = _random_mesh(3, 3, duplicate_via=True)
    assert mesh._branch_positions()["vertical_unique"] is False
    currents = np.arange(mesh.branch_count, dtype=np.float64)
    assert mesh.scatter_vertical(currents).tobytes() == _loop_scatter_vertical(mesh, currents).tobytes()
