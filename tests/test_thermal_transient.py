"""Backward-Euler transient conduction on the steady matrix-free operator."""

from __future__ import annotations

import numpy as np
import pytest

from thermal.matrix_free_mpir_fem import (
    ConvectionBoundary,
    ExposedFaceConvection,
    LayeredThermalMesh,
    RadiationBoundary,
    ThermalConductionProblem,
    TimeSchedule,
    TransientThermalSolution,
    solve_thermal_conduction,
    solve_thermal_transient,
)

AMBIENT = 300.0
COPPER_RHO_C = 3.45e6  # J/m³K


def test_time_schedules() -> None:
    uniform = TimeSchedule.uniform(0.5, 2.0)
    assert uniform.times_s == (0.0, 0.5, 1.0, 1.5, 2.0)
    geometric = TimeSchedule.geometric(1.0, 100.0, growth=2.0)
    assert geometric.times_s[:4] == (0.0, 1.0, 3.0, 7.0) and geometric.end_s == 100.0
    assert np.all(np.diff(geometric.times_s) > 0.0)
    capped = TimeSchedule.geometric(1.0, 20.0, growth=2.0, max_step_s=4.0)
    assert max(capped.steps_s) == 4.0
    with pytest.raises(ValueError, match="start at 0"):
        TimeSchedule((1.0, 2.0))
    with pytest.raises(ValueError, match="increasing"):
        TimeSchedule((0.0, 2.0, 1.0))


def test_lumped_block_cooling_reproduces_the_discrete_backward_euler_law() -> None:
    """A nearly isothermal block: T_{n+1} = T_a + (T_n - T_a) / (1 + Δt / τ), τ = ρ c t / h."""

    thickness, h, start = 2.0e-3, 20.0, 350.0
    mesh = LayeredThermalMesh(
        (thickness,), 1e-3, 1e-3, 4000.0, element_shape=(4, 5), volumetric_heat_capacity_j_per_m3_k=COPPER_RHO_C
    )
    problem = ThermalConductionProblem(mesh, convection=(ConvectionBoundary("top", h, AMBIENT),))
    schedule = TimeSchedule.uniform(20.0, 400.0)
    solution = solve_thermal_transient(problem, schedule, initial_temperature_k=start)
    assert isinstance(solution, TransientThermalSolution)
    assert solution.temperature_k.shape == (21, *mesh.node_shape)
    np.testing.assert_allclose(solution.temperature_k[0], start)

    tau = COPPER_RHO_C * thickness / h
    expected = start
    for step in solution.history:
        expected = AMBIENT + (expected - AMBIENT) / (1.0 + step.step_s / tau)
        assert step.max_temperature_k == pytest.approx(expected, abs=2.0e-4)  # the block is not perfectly isothermal
        assert step.converged
        # The heat leaving by convection is the heat drawn from the thermal mass.
        assert step.stored_heat_w == pytest.approx(-step.convective_heat_w, rel=1e-9)
        assert abs(step.heat_balance_error_w) < 1e-9
    analytic = AMBIENT + (start - AMBIENT) * np.exp(-schedule.end_s / tau)
    assert abs(solution.history[-1].max_temperature_k - analytic) < 0.6  # first-order time error
    finer = solve_thermal_transient(problem, TimeSchedule.uniform(5.0, 400.0), initial_temperature_k=start, store="final")
    assert finer.temperature_k.shape == (1, *mesh.node_shape)
    assert abs(finer.history[-1].max_temperature_k - analytic) < 0.16  # ~ Δt / 4


def test_heated_plate_marches_to_the_steady_solution_and_stops_when_steady() -> None:
    mesh = LayeredThermalMesh(
        (35e-6, 1.5e-3, 35e-6), 0.5e-3, 0.5e-3, (385.0, 0.8, 385.0), element_shape=(8, 12),
        through_plane_conductivity_w_per_m_k=(385.0, 0.3, 385.0),
        volumetric_heat_capacity_j_per_m3_k=(COPPER_RHO_C, 1.8e6, COPPER_RHO_C),
    )
    heat = np.zeros((3, 8, 12))
    heat[2, 3:5, 5:7] = 0.1
    problem = ThermalConductionProblem(
        mesh,
        convection=(ConvectionBoundary("top", 10.0, AMBIENT), ConvectionBoundary("bottom", 10.0, AMBIENT)),
        element_heat_w=heat,
    )
    steady = solve_thermal_conduction(problem)
    # Lumped time constant ρ c t / (2 h) ≈ 140 s; a 20 mK/τ drift is the steady criterion.
    schedule = TimeSchedule.geometric(0.01, 20000.0, growth=1.6)
    transient = solve_thermal_transient(problem, schedule, until_steady=True, steady_tolerance_k_per_s=2e-5)
    assert transient.reached_steady
    assert len(transient.history) < len(schedule.steps_s)
    rise = steady.max_temperature_k - AMBIENT
    assert rise > 5.0
    assert np.nanmax(np.abs(transient.final_temperature_k - steady.temperature_k)) < 2e-3 * rise
    # Monotone heating from the ambient start.
    peaks = transient.max_temperature_k
    assert peaks[0] > AMBIENT and np.all(np.diff(peaks) >= -1e-9)
    for step in transient.history:
        assert abs(step.heat_balance_error_w) < 1e-9
        assert step.stored_heat_w + step.convective_heat_w == pytest.approx(0.4, rel=1e-9)


def test_radiating_masked_body_relaxes_to_its_steady_state() -> None:
    active = np.zeros((3, 5, 5), dtype=bool)
    active[:, 1:4, 1:4] = True
    mesh = LayeredThermalMesh(
        (1e-3,) * 3, 1e-3, 1e-3, np.where(active, 200.0, 1.0), active=active,
        volumetric_heat_capacity_j_per_m3_k=np.where(active, 2.4e6, 1.0),
    )
    heat = np.where(active, 0.3 / 27, 0.0)
    problem = ThermalConductionProblem(
        mesh,
        convection=(ExposedFaceConvection(5.0, AMBIENT),),
        radiation=(RadiationBoundary("top", 0.9, AMBIENT),),
        element_heat_w=heat,
    )
    steady = solve_thermal_conduction(problem)
    transient = solve_thermal_transient(
        problem, TimeSchedule.geometric(0.1, 20000.0, growth=1.7), until_steady=True, steady_tolerance_k_per_s=1e-5
    )
    assert transient.reached_steady
    assert np.all(np.isnan(transient.final_temperature_k[:, 0]))
    assert np.nanmax(np.abs(transient.final_temperature_k - steady.temperature_k)) < 1e-2
    last = transient.history[-1]
    assert last.radiation_iterations >= 1 and last.radiative_heat_w > 0.0
    assert last.stored_heat_w + last.convective_heat_w + last.radiative_heat_w == pytest.approx(0.3, rel=1e-9)


def test_transient_validation() -> None:
    mesh = LayeredThermalMesh((1e-3,), 1e-3, 1e-3, 200.0, element_shape=(2, 2))
    problem = ThermalConductionProblem(mesh, convection=(ConvectionBoundary("top", 5.0, AMBIENT),))
    with pytest.raises(ValueError, match="volumetric_heat_capacity"):
        solve_thermal_transient(problem, TimeSchedule.uniform(1.0, 2.0))
    with pytest.raises(ValueError, match="positive on active"):
        LayeredThermalMesh((1e-3,), 1e-3, 1e-3, 200.0, element_shape=(2, 2), volumetric_heat_capacity_j_per_m3_k=0.0)
    with_capacity = LayeredThermalMesh(
        (1e-3,), 1e-3, 1e-3, 200.0, element_shape=(2, 2), volumetric_heat_capacity_j_per_m3_k=1e6
    )
    np.testing.assert_allclose(with_capacity.nodal_heat_capacity_j_per_k().sum(), 1e6 * 4e-9)
    with pytest.raises(ValueError, match="store"):
        solve_thermal_transient(
            ThermalConductionProblem(with_capacity, convection=(ConvectionBoundary("top", 5.0, AMBIENT),)),
            TimeSchedule.uniform(1.0, 2.0), store="none",
        )
