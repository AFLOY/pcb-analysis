from __future__ import annotations

import numpy as np
import pytest

from electrical.matrix_free_mpir_fem import (
    CurrentTerminal,
    LayeredPCBMesh,
    PCBConductionProblem,
    ViaConnection,
    solve_pcb_dc,
)
from thermal.matrix_free_mpir_fem import (
    ConvectionBoundary,
    LayeredThermalMesh,
    ThermalConductionProblem,
    element_joule_heat_w,
    solve_thermal_conduction,
    via_joule_heat_sources,
)


def _two_layer_electrical() -> tuple[PCBConductionProblem, LayeredThermalMesh]:
    rows, cols = 2, 6
    active = np.zeros((2, rows, cols), dtype=bool)
    active[0, :, :4] = True
    active[1, :, 2:] = True
    mesh = LayeredPCBMesh(
        element_active=active,
        layer_thickness_m=(35.0e-6, 35.0e-6),
        pitch_x_m=0.5e-3,
        pitch_y_m=0.5e-3,
    )
    problem = PCBConductionProblem(
        mesh=mesh,
        terminals=(
            CurrentTerminal(tuple((0, r, 0) for r in range(rows + 1)), 3.0, "in"),
            CurrentTerminal(tuple((1, r, cols) for r in range(rows + 1)), -3.0, "out"),
        ),
        reference_node=(1, 0, cols),
        vias=(
            ViaConnection((0, 0, 3), (1, 0, 3), 2.0e-3),
            ViaConnection((0, 2, 3), (1, 2, 3), 2.0e-3),
        ),
    )
    conductivity = np.full((3, rows, cols), 0.8)
    conductivity[0] = np.where(active[0], 385.0, 0.8)
    conductivity[2] = np.where(active[1], 385.0, 0.8)
    through = conductivity.copy()
    through[1] = 0.3
    thermal = LayeredThermalMesh(
        slab_thickness_m=(35.0e-6, 1.5e-3, 35.0e-6),
        pitch_x_m=0.5e-3,
        pitch_y_m=0.5e-3,
        conductivity_w_per_m_k=conductivity,
        through_plane_conductivity_w_per_m_k=through,
    )
    return problem, thermal


def test_electrical_solution_exposes_element_and_via_losses_that_sum() -> None:
    problem, _ = _two_layer_electrical()
    solution = solve_pcb_dc(problem)

    assert solution.solve.converged
    assert solution.element_joule_loss_w.shape == problem.mesh.element_active.shape
    assert solution.via_joule_loss_w.shape == (2,)
    assert np.all(solution.element_joule_loss_w >= 0.0)
    assert np.all(solution.element_joule_loss_w[~problem.mesh.element_active] == 0.0)
    assert solution.joule_loss_w == pytest.approx(
        float(np.sum(solution.element_joule_loss_w))
        + float(np.sum(solution.via_joule_loss_w))
    )
    # Each via passes 1.5 A through 2 mΩ.
    np.testing.assert_allclose(solution.via_joule_loss_w, 1.5**2 * 2.0e-3, rtol=1e-6)


def test_joule_heat_is_mapped_onto_copper_slabs_without_loss() -> None:
    problem, thermal = _two_layer_electrical()
    solution = solve_pcb_dc(problem)
    layer_slabs = (0, 2)

    heat = element_joule_heat_w(solution, thermal, layer_slabs)
    sources = via_joule_heat_sources(problem, solution, thermal, layer_slabs)

    assert heat.shape == thermal.element_grid_shape
    np.testing.assert_allclose(heat[1], 0.0)
    np.testing.assert_allclose(heat[0], solution.element_joule_loss_w[0])
    np.testing.assert_allclose(heat[2], solution.element_joule_loss_w[1])
    assert len(sources) == 4
    assert sources[0].nodes == ((0, 0, 3), (1, 0, 3))
    assert sources[1].nodes == ((2, 0, 3), (3, 0, 3))
    assert sum(source.power_w for source in sources) == pytest.approx(
        float(np.sum(solution.via_joule_loss_w))
    )


def test_coupled_solve_removes_exactly_the_electrical_loss() -> None:
    problem, thermal = _two_layer_electrical()
    electrical = solve_pcb_dc(problem)
    layer_slabs = (0, 2)
    ambient = 298.15
    thermal_problem = ThermalConductionProblem(
        thermal,
        convection=(
            ConvectionBoundary("top", 15.0, ambient),
            ConvectionBoundary("bottom", 15.0, ambient),
        ),
        element_heat_w=element_joule_heat_w(electrical, thermal, layer_slabs),
        heat_sources=via_joule_heat_sources(problem, electrical, thermal, layer_slabs),
    )
    solution = solve_thermal_conduction(thermal_problem, initial_temperature_k=ambient)

    assert solution.solve.converged
    assert solution.total_heat_input_w == pytest.approx(electrical.joule_loss_w)
    assert float(np.sum(solution.convective_heat_w)) == pytest.approx(
        electrical.joule_loss_w, rel=1.0e-8
    )
    assert solution.max_temperature_k > ambient
    assert solution.min_temperature_k > ambient


def test_mismatched_layer_maps_are_rejected() -> None:
    problem, thermal = _two_layer_electrical()
    solution = solve_pcb_dc(problem)
    with pytest.raises(ValueError, match="layer_slabs has"):
        element_joule_heat_w(solution, thermal, (0,))
    with pytest.raises(ValueError, match="must lie in"):
        via_joule_heat_sources(problem, solution, thermal, (0, 3))
