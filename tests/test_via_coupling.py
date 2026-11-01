"""Vertical branches (vias) couple to one another with their length, on every operator."""

from __future__ import annotations

import numpy as np
import pytest

from electrical import backend
from electrical.sheet_peec.sheet_inductance import (
    NEAR_RADIUS_CELLS,
    build_vertical_kernel,
    closed_form_mutual_inductance,
    filament_mutual_inductance,
    parallel_filament_mutual_inductance,
    vertical_cell,
)

MU0_OVER_4PI = 1e-7


def test_the_filament_form_has_the_closed_forms_it_generalises() -> None:
    # aligned equal filaments: Grover's two-filament formula
    assert parallel_filament_mutual_inductance(1.6e-3, 1.6e-3, 0.0, 0.5e-3) == pytest.approx(
        filament_mutual_inductance(1.6e-3, 0.5e-3), rel=1e-14)
    # collinear and apart: (2l+g)ln(2l+g) - 2(l+g)ln(l+g) + g ln g
    l, g = 1e-3, 0.5e-3
    collinear = MU0_OVER_4PI * ((2 * l + g) * np.log(2 * l + g) - 2 * (l + g) * np.log(l + g) + g * np.log(g))
    assert parallel_filament_mutual_inductance(l, l, l + g, 0.0) == pytest.approx(collinear, rel=1e-13)
    # symmetric in the axial direction and in the order of the two
    assert parallel_filament_mutual_inductance(1.6e-3, 0.2e-3, 0.7e-3, 2e-3) == pytest.approx(
        parallel_filament_mutual_inductance(1.6e-3, 0.2e-3, -0.7e-3, 2e-3), rel=1e-13)
    assert parallel_filament_mutual_inductance(1.6e-3, 0.2e-3, 0.7e-3, 2e-3) == pytest.approx(
        parallel_filament_mutual_inductance(0.2e-3, 1.6e-3, -0.7e-3, 2e-3), rel=1e-13)
    # far away it keeps its digits and tends to the centre-to-centre limit
    far = parallel_filament_mutual_inductance(0.2e-3, 0.2e-3, 0.0, 0.1)
    assert far == pytest.approx(MU0_OVER_4PI * 0.04e-6 / 0.1, rel=1e-6)


@pytest.mark.parametrize("pitch, span", [(1e-4, 1.6e-3), (1e-4, 3.2e-3), (2e-4, 1.6e-3)])
def test_the_vertical_kernel_is_the_bars_near_and_the_filaments_far(pitch, span) -> None:
    rows = cols = 2 * (NEAR_RADIUS_CELLS + 12)
    table = build_vertical_kernel((rows, cols), pitch, span, span, 0.0)
    for cells in (NEAR_RADIUS_CELLS - 4, NEAR_RADIUS_CELLS, NEAR_RADIUS_CELLS + 1, NEAR_RADIUS_CELLS + 10):
        filaments = parallel_filament_mutual_inductance(span, span, 0.0, cells * pitch)
        # the bars' cross-section is worth w^2 / (12 d^2) at most: 1.5e-4 at the seam
        assert table[0, cells] == pytest.approx(filaments, rel=3e-4)
    # no step at the seam beyond the cross-section's own (the point form stepped 0.8 to 10.6 %)
    inside, outside = table[0, NEAR_RADIUS_CELLS], table[0, NEAR_RADIUS_CELLS + 1]
    ratio = inside / outside
    expected = parallel_filament_mutual_inductance(span, span, 0.0, NEAR_RADIUS_CELLS * pitch) / \
        parallel_filament_mutual_inductance(span, span, 0.0, (NEAR_RADIUS_CELLS + 1) * pitch)
    assert ratio == pytest.approx(expected, rel=3e-4)
    near = closed_form_mutual_inductance(vertical_cell(span, pitch), vertical_cell(span, pitch), (0.0, 3 * pitch, 0.0))
    assert table[0, 3] == pytest.approx(near, rel=1e-12)


@pytest.mark.skipif(not backend.native_available(), reason="pcbcore extension not built")
def test_the_core_tabulates_the_vertical_kernel_as_numpy_does() -> None:
    args = ((40, 60), 1e-4, 1.6e-3, 0.2e-3, 0.7e-3)
    native = build_vertical_kernel(*args, near_radius_cells=8)
    with backend.use_reference():
        reference = build_vertical_kernel(*args, near_radius_cells=8)
    np.testing.assert_allclose(native, reference, rtol=1e-12, atol=0.0)
