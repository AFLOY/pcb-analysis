"""C++ construction of sheet-PEEC current elements (opt-in) against the Python loop."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from electrical.sheet_peec.sheet_operator import SheetLayer, SheetStackup
from electrical.sheet_peec.sheet_peec import SheetMesh, ViaBranch
from emc.tiled_dipole_superposition import dipoles_from_sheet_peec
from emc.tiled_dipole_superposition.native_dipole import native_available

pytestmark = pytest.mark.skipif(
    not native_available(),
    reason="emc native extension not built; run python -m emc.tiled_dipole_superposition.native.build",
)


def _mesh(rows: int, cols: int, *, holes: bool) -> SheetMesh:
    stackup = SheetStackup(
        (
            SheetLayer("F.Cu", 0.0, 35e-6, 1.724e-8),
            SheetLayer("In1.Cu", -0.4e-3, 35e-6, 1.724e-8),
            SheetLayer("B.Cu", -1.6e-3, 35e-6, 1.724e-8),
        )
    )
    occupancy = np.ones((3, rows, cols), dtype=bool)
    if holes and rows > 2 and cols > 3:
        occupancy[0, 1, 1:3] = False
        occupancy[2, :1, :] = False
    vias = tuple(
        ViaBranch(r, c, lower, upper, resistance_ohm=1e-3)
        for (r, c, lower, upper) in ((0, 0, 0, 1), (rows // 2, cols // 2, 0, 2), (rows - 1, cols - 1, 1, 2))
        if occupancy[lower, r, c] and occupancy[upper, r, c]
    )
    return SheetMesh((rows, cols), 2e-4, stackup, occupancy, vias=vias)


@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (3, 5), (9, 13)])
@pytest.mark.parametrize("holes", [False, True])
@pytest.mark.parametrize("threads", [1, 3])
def test_native_sheet_dipoles_match_the_python_loop(shape, holes, threads) -> None:
    mesh = _mesh(*shape, holes=holes)
    rng = np.random.default_rng(shape[0] * 31 + shape[1])
    current = rng.standard_normal(mesh.branch_count) + 1j * rng.standard_normal(mesh.branch_count)
    solution = SimpleNamespace(branch_current=current)
    portable = dipoles_from_sheet_peec(mesh, solution, native=False)
    native = dipoles_from_sheet_peec(mesh, solution, native=True, native_threads=threads)
    assert native.count == portable.count == mesh.branch_count
    np.testing.assert_array_equal(native.position_m, portable.position_m)
    np.testing.assert_array_equal(native.moment_a_m, portable.moment_a_m)


def test_native_sheet_dipoles_reject_a_wrong_current_count() -> None:
    mesh = _mesh(2, 3, holes=False)
    with pytest.raises(ValueError, match="one value per branch"):
        dipoles_from_sheet_peec(mesh, SimpleNamespace(branch_current=np.zeros(mesh.branch_count + 1)), native=True)
