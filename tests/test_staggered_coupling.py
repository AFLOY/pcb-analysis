from __future__ import annotations

import numpy as np
import pytest

from electrical.dice_peec.sheet_operator import SheetInductanceOperator, SheetLayer, SheetStackup
from electrical.dice_peec.sheet_peec import SheetMesh, Terminal, ViaBranch
from electrical.matrix_free_mpir_fem import (
    CurrentTerminal,
    LayeredPCBMesh,
    PCBConductionProblem,
    PCBConductionSolution,
    ViaConnection,
    solve_pcb_dc,
)
from emc.tiled_dipole_superposition import FCC_PART15_CLASS_B
from multiphysics.staggered_coupling import (
    CouplingConfig,
    ElectricalScenario,
    ElectroEmissionResult,
    ElectroEmissionScenario,
    ElectroThermalEmissionResult,
    ElectroThermalEmissionScenario,
    ElectroThermalScenario,
    EmissionScenario,
    ScanPlane,
    SheetPeecEmissionResult,
    SheetPeecEmissionScenario,
    ThermalScenario,
    conductivity_at_temperature,
    electrical_layer_temperature_k,
    run_electro_thermal,
    run_scenario,
    run_scenarios,
)
from thermal.matrix_free_mpir_fem import (
    ConvectionBoundary,
    LayeredThermalMesh,
    ThermalConductionProblem,
    ThermalConductionSolution,
)

AMBIENT = 298.15
PITCH = 0.5e-3


def _loop_board(rows: int = 6, cols: int = 24, current_a: float = 2.0) -> PCBConductionProblem:
    active = np.zeros((2, rows, cols), dtype=bool)
    active[:, 1:5, :] = True
    mesh = LayeredPCBMesh(active, (35e-6, 35e-6), PITCH, PITCH)
    node_rows = tuple(range(1, 6))
    vias = tuple(ViaConnection((0, r, c), (1, r, c), 1.5e-3) for r in node_rows[1:-1] for c in (cols - 1, cols))
    return PCBConductionProblem(
        mesh,
        (
            CurrentTerminal(tuple((1, r, 0) for r in node_rows), current_a, "source"),
            CurrentTerminal(tuple((0, r, 0) for r in node_rows), -current_a, "return"),
        ),
        reference_node=(0, 1, 0),
        vias=vias,
    )


def _thermal_mesh(problem: PCBConductionProblem) -> LayeredThermalMesh:
    active = problem.mesh.element_active
    _, rows, cols = active.shape
    k = np.full((3, rows, cols), 0.8)
    k[0] = np.where(active[0], 385.0, 0.8)
    k[2] = np.where(active[1], 385.0, 0.8)
    kz = k.copy()
    kz[1] = 0.3
    return LayeredThermalMesh((35e-6, 1.5e-3, 35e-6), PITCH, PITCH, k, through_plane_conductivity_w_per_m_k=kz)


def _scenario(current_a: float = 2.0, **overrides) -> ElectroThermalScenario:
    problem = _loop_board(current_a=current_a)
    options = dict(
        convection=(ConvectionBoundary("top", 10.0, AMBIENT), ConvectionBoundary("bottom", 10.0, AMBIENT)),
        conductivity_reference_temperature_k=293.15,
    )
    options.update(overrides)
    return ElectroThermalScenario(problem, _thermal_mesh(problem), (0, 2), **options)


def test_isothermal_copper_reproduces_the_resistivity_law_exactly() -> None:
    """With the whole copper held at T_hot the loop closes in one shot: loss = I² R(T_hot)."""

    problem = _loop_board()
    thermal = _thermal_mesh(problem)
    mask = np.ones(thermal.node_shape, dtype=bool)          # every node fixed
    hot = 353.15
    scenario = ElectroThermalScenario(
        problem, thermal, (0, 2), fixed_temperature_mask=mask, fixed_temperature_k=hot,
        conductivity_reference_temperature_k=293.15,
    )
    result = run_electro_thermal(scenario)

    cold = solve_pcb_dc(problem)
    factor = 1.0 + 3.93e-3 * (hot - 293.15)
    assert result.converged
    assert result.iterations == 3                             # hot solve, then a confirming pass
    assert result.electrical.joule_loss_w == pytest.approx(cold.joule_loss_w * factor, rel=1e-9)
    assert result.loss_increase_ratio == pytest.approx(factor, rel=1e-9)
    np.testing.assert_allclose(result.element_temperature_k, hot)
    np.testing.assert_allclose(result.thermal.temperature_k, hot)


def test_zero_temperature_coefficient_needs_one_pass() -> None:
    result = run_electro_thermal(_scenario(temperature_coefficient_per_k=0.0))
    assert result.converged and result.iterations == 1
    assert result.loss_increase_ratio == 1.0
    np.testing.assert_allclose(result.conductivity_s_per_m, _loop_board().mesh.conductivity_s_per_m)


def test_coupled_state_is_self_consistent_and_aitken_saves_iterations() -> None:
    scenario = _scenario()
    accelerated = run_electro_thermal(scenario)
    plain = run_electro_thermal(scenario, config=CouplingConfig(aitken=False))

    assert accelerated.converged and plain.converged
    assert accelerated.iterations < plain.iterations
    assert accelerated.loss_increase_ratio > 1.05
    assert accelerated.thermal.max_temperature_k > AMBIENT + 10.0
    np.testing.assert_allclose(
        accelerated.electrical.joule_loss_w, plain.electrical.joule_loss_w, rtol=1e-6
    )
    # The conductivity the final electrical solve used is the law evaluated at
    # the final temperature field.
    layer_temperature = electrical_layer_temperature_k(
        accelerated.thermal.temperature_k, scenario.thermal_mesh, scenario.layer_slabs
    )
    expected = conductivity_at_temperature(
        scenario.electrical.mesh.conductivity_s_per_m, layer_temperature,
        reference_temperature_k=293.15, coefficient_per_k=3.93e-3,
    )
    np.testing.assert_allclose(accelerated.conductivity_s_per_m, expected, rtol=1e-6)
    # Heat entering the thermal solve is the electrical loss.
    assert accelerated.thermal.total_heat_input_w == pytest.approx(accelerated.electrical.joule_loss_w, rel=1e-12)
    assert accelerated.history[-1].temperature_change_k <= 1e-3
    assert all(step.relaxation > 0.0 for step in accelerated.history)


def test_iteration_limit_reports_non_convergence_instead_of_raising() -> None:
    result = run_electro_thermal(_scenario(current_a=6.0), config=CouplingConfig(max_iterations=3))
    assert not result.converged and result.iterations == 3
    assert result.electrical.joule_loss_w > result.cold_joule_loss_w


def test_scenario_validation() -> None:
    problem = _loop_board()
    thermal = _thermal_mesh(problem)
    with pytest.raises(ValueError, match="one thermal slab per electrical layer"):
        ElectroThermalScenario(problem, thermal, (0,), convection=(ConvectionBoundary("top", 5.0, AMBIENT),))
    with pytest.raises(ValueError, match="positive film coefficient or a fixed node"):
        ElectroThermalScenario(problem, thermal, (0, 2))
    with pytest.raises(ValueError, match="relaxation"):
        CouplingConfig(relaxation=0.0)
    with pytest.raises(ValueError, match="frequencies_hz"):
        EmissionScenario(())


def test_run_scenario_dispatches_every_scenario_type() -> None:
    problem = _loop_board()
    thermal = _thermal_mesh(problem)
    emission = EmissionScenario(
        (30e6, 300e6),
        limit=FCC_PART15_CLASS_B,
        distance_m=3.0,
        scan=ScanPlane(np.linspace(0, 24 * PITCH, 6), np.linspace(0, 6 * PITCH, 3), 6.6e-3),
    )
    heights = (0.0, 1.6e-3)
    et = _scenario()

    electrical, thermal_only, coupled, emitted, chained = run_scenarios(
        [
            ElectricalScenario(problem),
            ThermalScenario(
                ThermalConductionProblem(
                    thermal, convection=(ConvectionBoundary("top", 10.0, AMBIENT),),
                    element_heat_w=np.full(thermal.element_grid_shape, 1e-4),
                )
            ),
            et,
            ElectroEmissionScenario(problem, heights, emission),
            ElectroThermalEmissionScenario(et, heights, emission),
        ]
    )

    assert isinstance(electrical, PCBConductionSolution) and electrical.solve.converged
    assert isinstance(thermal_only, ThermalConductionSolution) and thermal_only.solve.converged
    assert coupled.converged
    assert isinstance(emitted, ElectroEmissionResult)
    assert emitted.emission.frequencies_hz.tolist() == [30e6, 300e6]
    # Loop radiation grows as k⁴ in power, 40 dB per decade in field.
    field = emitted.emission.predicted_dbuv_per_m
    assert field[1] - field[0] == pytest.approx(40.0, abs=0.5)
    assert emitted.emission.limit_dbuv_per_m.tolist() == pytest.approx([40.0, 46.02], abs=0.01)
    assert emitted.emission.worst_margin_db == pytest.approx(np.min(emitted.emission.margin_db))
    assert all(np.isfinite(point.max_near_magnetic_a_per_m) for point in emitted.emission.points)
    # Terminal closure leaves no net electric moment.
    assert np.linalg.norm(emitted.emission.points[0].moments.electric_a_m) < 1e-9

    assert isinstance(chained, ElectroThermalEmissionResult)
    # Hotter copper carries the same current: the loop moment and field barely move.
    assert np.all(np.abs(chained.heating_shift_db) < 0.5)
    np.testing.assert_allclose(chained.cold_emission.predicted_dbuv_per_m, field)

    with pytest.raises(TypeError, match="unsupported scenario"):
        run_scenario(object())


def test_sheet_peec_emission_scenario_resolves_each_frequency() -> None:
    rows, cols, pitch = 4, 9, 2e-4
    stackup = SheetStackup(
        (SheetLayer("F.Cu", 0.0, 35e-6, 1.724e-8), SheetLayer("B.Cu", -1.6e-3, 35e-6, 1.724e-8))
    )
    via = ViaBranch(rows // 2, cols - 1, 0, 1, resistance_ohm=1e-3)
    mesh = SheetMesh((rows, cols), pitch, stackup, np.ones((2, rows, cols), dtype=bool), vias=(via,))
    operator = SheetInductanceOperator((rows, cols), pitch, stackup, vertical_levels=mesh.vertical_levels)
    terminals = (Terminal("in", 0, ((rows // 2, 0),), 1.0), Terminal("out", 1, ((rows // 2, 0),), -1.0))

    result = run_scenario(
        SheetPeecEmissionScenario(mesh, operator, terminals, EmissionScenario((30e6, 100e6)))
    )

    assert isinstance(result, SheetPeecEmissionResult)
    assert len(result.solutions) == 2 and all(s.converged for s in result.solutions)
    assert [s.frequency_hz for s in result.solutions] == [30e6, 100e6]
    assert len(result.emission.dipoles) == 2 and result.emission.dipoles[0] is not result.emission.dipoles[1]
    assert np.all(np.isfinite(result.emission.predicted_dbuv_per_m))
    with pytest.raises(ValueError, match="no quasi-peak limit"):
        run_scenario(SheetPeecEmissionScenario(mesh, operator, terminals, EmissionScenario((1e3,))))
