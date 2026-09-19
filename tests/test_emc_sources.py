from __future__ import annotations

import numpy as np
import pytest

from electrical.sheet_peec.sheet_operator import SheetInductanceOperator, SheetLayer, SheetStackup
from electrical.sheet_peec.sheet_peec import SheetMesh, Terminal, ViaBranch, solve_sheet_case
from electrical.matrix_free_mpir_fem import (
    CurrentTerminal,
    LayeredPCBMesh,
    PCBConductionProblem,
    ViaConnection,
    solve_pcb_dc,
)
from emc.tiled_dipole_superposition import (
    dipole_moments,
    dipoles_from_pcb_dc,
    dipoles_from_sheet_peec,
    far_field_pattern,
)


def test_dc_strip_dipoles_carry_the_terminal_current_over_the_strip_length() -> None:
    columns, rows = 20, 3
    pitch = 0.5e-3
    mesh = LayeredPCBMesh(
        element_active=np.ones((1, rows, columns), dtype=bool),
        layer_thickness_m=(35e-6,),
        pitch_x_m=pitch,
        pitch_y_m=pitch,
    )
    problem = PCBConductionProblem(
        mesh=mesh,
        terminals=(
            CurrentTerminal(tuple((0, r, 0) for r in range(rows + 1)), 2.0, "in"),
            CurrentTerminal(tuple((0, r, columns) for r in range(rows + 1)), -2.0, "out"),
        ),
        reference_node=(0, 0, columns),
    )
    solution = solve_pcb_dc(problem)
    dipoles = dipoles_from_pcb_dc(problem, solution, layer_height_m=(1.6e-3,))

    assert dipoles.count == rows * columns
    np.testing.assert_allclose(dipoles.position_m[:, 2], 1.6e-3)
    # Σ J V over a uniform strip is I × length.
    total = np.sum(dipoles.moment_a_m, axis=0)
    assert total[0].real == pytest.approx(2.0 * columns * pitch, rel=1e-8)
    assert abs(total[1]) < 1e-9 and abs(total[2]) == 0.0
    moments = dipole_moments(dipoles, 100e6)
    assert moments.electric_a_m[0].real == pytest.approx(2.0 * columns * pitch, rel=1e-8)

    # Closing the terminals externally removes the net moment entirely.
    closed = dipoles_from_pcb_dc(problem, solution, layer_height_m=(1.6e-3,), close_terminals=True)
    assert closed.count == dipoles.count + 2
    np.testing.assert_allclose(np.sum(closed.moment_a_m, axis=0), 0.0, atol=1e-9)
    open_pattern = far_field_pattern(dipoles, 100e6)
    closed_pattern = far_field_pattern(closed, 100e6)
    assert closed_pattern.radiated_power_w < 1e-3 * open_pattern.radiated_power_w


def test_dc_vias_become_vertical_elements_with_the_layer_separation() -> None:
    mesh = LayeredPCBMesh(
        element_active=np.ones((2, 1, 1), dtype=bool),
        layer_thickness_m=(35e-6, 35e-6),
        pitch_x_m=1e-3,
        pitch_y_m=1e-3,
    )
    problem = PCBConductionProblem(
        mesh=mesh,
        terminals=(
            CurrentTerminal(((0, 0, 0),), 1.0, "in"),
            CurrentTerminal(((1, 0, 0),), -1.0, "out"),
        ),
        reference_node=(1, 0, 0),
        vias=(ViaConnection((0, 0, 1), (1, 0, 1), 2e-3),),
    )
    solution = solve_pcb_dc(problem)
    dipoles = dipoles_from_pcb_dc(problem, solution, layer_height_m=(0.0, 1.6e-3))

    via = dipoles.moment_a_m[-1]
    assert dipoles.position_m[-1].tolist() == [1e-3, 0.0, 0.8e-3]
    assert abs(via[2]) == pytest.approx(1.0 * 1.6e-3, rel=1e-8)
    assert via[0] == 0.0 and via[1] == 0.0
    with pytest.raises(ValueError, match="one height per electrical layer"):
        dipoles_from_pcb_dc(problem, solution, layer_height_m=(0.0,))


def test_sheet_peec_branches_map_to_elements_in_the_stackup_frame() -> None:
    rows, cols, pitch = 4, 7, 2e-4
    stackup = SheetStackup(
        (
            SheetLayer("F.Cu", 0.0, 35e-6, 1.724e-8),
            SheetLayer("B.Cu", -1.6e-3, 35e-6, 1.724e-8),
        )
    )
    via = ViaBranch(rows // 2, cols // 2, 0, 1, resistance_ohm=1e-3)
    occupancy = np.ones((2, rows, cols), dtype=bool)
    mesh = SheetMesh((rows, cols), pitch, stackup, occupancy, vias=(via,))
    operator = SheetInductanceOperator((rows, cols), pitch, stackup, vertical_levels=mesh.vertical_levels)
    terminals = [
        Terminal("in", 0, ((rows // 2, 0),), 1.0),
        Terminal("out", 1, ((rows // 2, cols - 1),), -1.0),
    ]
    solution = solve_sheet_case(mesh, operator, terminals, frequency_hz=1e6)
    assert solution.converged

    dipoles = dipoles_from_sheet_peec(mesh, solution)
    assert dipoles.count == mesh.branch_count
    heights = dipoles.position_m[:, 2]
    assert set(np.round(heights[: len(mesh.branch_x) + len(mesh.branch_y)], 9)) <= {0.0, -1.6e-3}
    # The single via spans the two layers and carries the whole current downwards.
    via_moment = dipoles.moment_a_m[-1]
    assert dipoles.position_m[-1][2] == pytest.approx(-0.8e-3)
    assert via_moment[2] == pytest.approx(solution.branch_current[-1] * (-1.6e-3))
    assert abs(solution.branch_current[-1]) == pytest.approx(1.0, rel=1e-6)
    # x branches sit midway between cell centres on the layer height.
    layer, row, col = mesh.branch_x[0]
    np.testing.assert_allclose(dipoles.position_m[0], [(col + 1) * pitch, (row + 0.5) * pitch, stackup.layers[layer].z_m])
    # The whole thing radiates something finite at 100 MHz.
    pattern = far_field_pattern(dipoles, 100e6)
    assert np.isfinite(pattern.radiated_power_w) and pattern.radiated_power_w > 0.0
