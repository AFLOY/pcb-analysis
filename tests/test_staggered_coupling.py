from __future__ import annotations

import numpy as np
import pytest

from electrical.sheet_peec.sheet_operator import SheetInductanceOperator, SheetLayer, SheetStackup
from electrical.sheet_peec.sheet_peec import SheetMesh, Terminal, ViaBranch
from electrical.matrix_free_mpir_fem import (
    CurrentTerminal,
    LayeredPCBMesh,
    PCBConductionProblem,
    PCBConductionSolution,
    ViaConnection,
    VoltageTerminal,
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


def _voltage_loop_board(voltage_v: float, rows: int = 6, cols: int = 24) -> PCBConductionProblem:
    current = _loop_board(rows, cols)
    source, sink = current.terminals
    return PCBConductionProblem(
        current.mesh,
        voltage_terminals=(
            VoltageTerminal(source.nodes, voltage_v, "source"),
            VoltageTerminal(sink.nodes, 0.0, "return"),
        ),
        vias=current.vias,
    )


def _voltage_for_cold_current(current_a: float) -> float:
    """The source voltage that drives ``current_a`` through the cold loop."""

    probe = solve_pcb_dc(_voltage_loop_board(1.0))
    return current_a / float(probe.voltage_terminal_current_a[0])


def _thermal_mesh(problem: PCBConductionProblem) -> LayeredThermalMesh:
    active = problem.mesh.element_active
    _, rows, cols = active.shape
    k = np.full((3, rows, cols), 0.8)
    k[0] = np.where(active[0], 385.0, 0.8)
    k[2] = np.where(active[1], 385.0, 0.8)
    kz = k.copy()
    kz[1] = 0.3
    return LayeredThermalMesh((35e-6, 1.5e-3, 35e-6), PITCH, PITCH, k, through_plane_conductivity_w_per_m_k=kz)


def _options(**overrides) -> dict:
    options = dict(
        convection=(ConvectionBoundary("top", 10.0, AMBIENT), ConvectionBoundary("bottom", 10.0, AMBIENT)),
        conductivity_reference_temperature_k=293.15,
    )
    options.update(overrides)
    return options


def _scenario(cold_current_a: float = 2.0, **overrides) -> ElectroThermalScenario:
    """The loop at the voltage that drives ``cold_current_a`` through cold copper."""

    problem = _voltage_loop_board(_voltage_for_cold_current(cold_current_a))
    return ElectroThermalScenario(problem, _thermal_mesh(problem), (0, 2), **_options(**overrides))


def _current_driven_scenario(current_a: float = 2.0, **overrides) -> ElectroThermalScenario:
    """The deprecated constant-current drive, kept for its runaway regression tests."""

    problem = _loop_board(current_a=current_a)
    with pytest.warns(DeprecationWarning, match="current-driven electro-thermal coupling is deprecated"):
        return ElectroThermalScenario(problem, _thermal_mesh(problem), (0, 2), **_options(**overrides))


def test_isothermal_current_drive_reproduces_the_resistivity_law_exactly() -> None:
    """With the whole copper held at T_hot the loop closes in one shot: loss = I² R(T_hot)."""

    problem = _loop_board()
    thermal = _thermal_mesh(problem)
    mask = np.ones(thermal.node_shape, dtype=bool)          # every node fixed
    hot = 353.15
    with pytest.warns(DeprecationWarning, match="current-driven"):
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


def test_isothermal_voltage_drive_loses_in_inverse_proportion_to_resistance() -> None:
    """At fixed voltage the hot copper carries I(T_hot) = V / R(T_hot): loss = V² / R(T_hot)."""

    problem = _voltage_loop_board(20e-3)
    thermal = _thermal_mesh(problem)
    hot = 353.15
    scenario = ElectroThermalScenario(
        problem, thermal, (0, 2), fixed_temperature_mask=np.ones(thermal.node_shape, dtype=bool),
        fixed_temperature_k=hot, conductivity_reference_temperature_k=293.15,
    )
    result = run_electro_thermal(scenario)

    cold = solve_pcb_dc(problem)
    factor = 1.0 + 3.93e-3 * (hot - 293.15)
    assert result.converged
    assert result.electrical.joule_loss_w == pytest.approx(cold.joule_loss_w / factor, rel=1e-9)
    np.testing.assert_allclose(
        result.electrical.voltage_terminal_current_a, cold.voltage_terminal_current_a / factor, rtol=1e-9
    )


def test_voltage_driven_coupling_settles_below_the_cold_loss() -> None:
    """Heating raises the resistance, so under a fixed voltage the current and loss fall."""

    cold_current = 2.0
    voltage = _voltage_for_cold_current(cold_current)
    result = run_electro_thermal(_scenario(cold_current))
    current_driven = run_electro_thermal(_current_driven_scenario(cold_current))

    assert result.converged
    # Same cold current; an equipotential pad spreads it slightly better than a
    # uniformly loaded one, hence the small difference in cold loss.
    assert result.cold_joule_loss_w == pytest.approx(current_driven.cold_joule_loss_w, rel=1e-2)
    assert result.loss_increase_ratio < 0.97
    assert current_driven.loss_increase_ratio > 1.05
    assert result.thermal.max_temperature_k > AMBIENT + 5.0
    assert result.thermal.total_heat_input_w == pytest.approx(result.electrical.joule_loss_w, rel=1e-12)
    supplied = float(result.electrical.voltage_terminal_current_a[0])
    assert supplied < cold_current
    assert float(np.sum(result.electrical.voltage_terminal_current_a)) == pytest.approx(0.0, abs=1e-9)
    assert result.electrical.joule_loss_w == pytest.approx(voltage * supplied, rel=1e-8)


def test_zero_temperature_coefficient_needs_one_pass() -> None:
    result = run_electro_thermal(_scenario(temperature_coefficient_per_k=0.0))
    assert result.converged and result.iterations == 1
    assert result.loss_increase_ratio == 1.0
    np.testing.assert_allclose(result.conductivity_s_per_m, _voltage_loop_board(1e-3).mesh.conductivity_s_per_m)


def test_coupled_state_is_self_consistent_and_aitken_saves_iterations() -> None:
    scenario = _scenario(3.0)
    accelerated = run_electro_thermal(scenario)
    plain = run_electro_thermal(scenario, config=CouplingConfig(aitken=False))

    assert accelerated.converged and plain.converged
    assert accelerated.iterations < plain.iterations
    assert accelerated.loss_increase_ratio < 0.9
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
    """Near runaway under the deprecated current drive: report, do not raise."""

    result = run_electro_thermal(_current_driven_scenario(6.0), config=CouplingConfig(max_iterations=3))
    assert not result.converged and result.iterations == 3
    assert result.electrical.joule_loss_w > result.cold_joule_loss_w


def test_voltage_drive_converges_where_current_drive_runs_away() -> None:
    """At a cold current of 6 A the current-driven loop is near runaway; the
    same cold operating point under voltage drive settles in a few steps."""

    result = run_electro_thermal(_scenario(6.0))
    assert result.converged and result.iterations <= 10
    assert result.loss_increase_ratio < 1.0
    assert float(result.electrical.voltage_terminal_current_a[0]) < 6.0


def test_only_current_driven_scenarios_warn() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _scenario()
    _current_driven_scenario()
    # One voltage terminal plus a current load still imposes every copper current.
    board = _voltage_loop_board(1e-3)
    source, sink = board.voltage_terminals
    pinned = PCBConductionProblem(
        board.mesh,
        terminals=(CurrentTerminal(sink.nodes, -2.0, "load"),),
        voltage_terminals=(source,),
        vias=board.vias,
    )
    with pytest.warns(DeprecationWarning, match="fewer than two voltage terminals"):
        ElectroThermalScenario(pinned, _thermal_mesh(pinned), (0, 2), **_options())


def test_scenario_validation() -> None:
    problem = _voltage_loop_board(1e-3)
    thermal = _thermal_mesh(problem)
    with pytest.raises(ValueError, match="one thermal slab per electrical layer"):
        ElectroThermalScenario(problem, thermal, (0,), convection=(ConvectionBoundary("top", 5.0, AMBIENT),))
    with pytest.raises(ValueError, match="positive film coefficient, a radiating face or a fixed node"):
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
    # Under voltage drive hotter copper carries less current; the loop field
    # follows the supplied current, 20 log10(I_hot / I_cold).
    hot_current = float(chained.electro_thermal.electrical.voltage_terminal_current_a[0])
    cold_current = float(solve_pcb_dc(et.electrical).voltage_terminal_current_a[0])
    expected_shift = 20.0 * np.log10(hot_current / cold_current)
    assert expected_shift < 0.0
    np.testing.assert_allclose(chained.heating_shift_db, expected_shift, atol=0.05)
    # The cold emission of the voltage-driven loop at 2 A matches the
    # current-driven 2 A electrical scenario to the pad equipotential effect.
    np.testing.assert_allclose(chained.cold_emission.predicted_dbuv_per_m, field, atol=0.1)

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
