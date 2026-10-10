"""The C++ thermal core against the NumPy implementation it replaced.

The prepared operator (coefficients, masks, Robin lumping of every boundary
kind, loads, diagonal) agrees bit for bit or to ``DIRECT_RTOL``; the solves
(linear, radiation Newton, backward Euler) agree to the iterative tolerance
and take the same Newton and time steps.  Every output has the same bits at
every thread budget.
"""

from __future__ import annotations

import numpy as np
import pytest

from electrical.matrix_free_mpir_fem import MPIRConfig
from electrical.threads import thread_budget_scope
from thermal.matrix_free_mpir_fem import (
    ConvectionBoundary,
    ExposedFaceConvection,
    ExposedFaceRadiation,
    HeatSource,
    LayeredThermalMesh,
    MatrixFreeThermalOperator,
    RadiationBoundary,
    ThermalConductionProblem,
    TimeSchedule,
    solve_thermal_conduction,
    solve_thermal_transient,
)
from thermal.matrix_free_mpir_fem.native_hex import native_available
from tests.tolerance import DIRECT_RTOL, iterative_rtol

pytestmark = pytest.mark.skipif(not native_available(), reason="pcbcore extension not built")

AMBIENT = 298.15


def _body(*, radiation: bool = False, capacity: bool = False) -> ThermalConductionProblem:
    """A graded three-slab stack with a void pocket, every boundary kind and every load kind."""

    rows, cols = 9, 13
    active = np.ones((3, rows, cols), dtype=bool)
    active[2, 2:5, 3:7] = False
    mesh = LayeredThermalMesh(
        slab_thickness_m=(35.0e-6, 1.2e-3, 70.0e-6),
        pitch_x_m=0.3e-3 * (1.0 + 0.4 * np.linspace(0.0, 1.0, cols)),
        pitch_y_m=0.3e-3 * (1.0 + 0.2 * np.linspace(0.0, 1.0, rows)),
        conductivity_w_per_m_k=(385.0, 0.8, 385.0),
        through_plane_conductivity_w_per_m_k=(385.0, 0.3, 385.0),
        active=active,
        volumetric_heat_capacity_j_per_m3_k=(3.45e6, 1.9e6, 3.45e6) if capacity else None,
    )
    heat = np.zeros(mesh.element_grid_shape)
    heat[2, rows // 2 + 2, 2 : cols - 2] = 0.08
    nodal = np.zeros(mesh.node_shape)
    nodal[1, 1, 1] = 0.02
    fixed = np.zeros(mesh.node_shape, dtype=bool)
    fixed[0, 0, :4] = True
    coefficient = np.linspace(5.0, 25.0, rows * cols).reshape(rows, cols)
    ambient = AMBIENT + np.linspace(0.0, 4.0, rows * cols).reshape(rows, cols)
    radiators = (
        (
            RadiationBoundary("top", 0.8, AMBIENT - 5.0),
            ExposedFaceRadiation(0.3, AMBIENT, directions=("-x", "+x", "-y")),
        )
        if radiation
        else ()
    )
    return ThermalConductionProblem(
        mesh,
        convection=(
            ConvectionBoundary("top", coefficient, ambient),
            ConvectionBoundary("bottom", 15.0, AMBIENT),
            ExposedFaceConvection(4.0, AMBIENT + 1.0, directions=("+y", "+z")),
        ),
        fixed_temperature_mask=fixed,
        fixed_temperature_k=AMBIENT + 2.0,
        heat_sources=(HeatSource(((1, 3, 3), (1, 3, 4)), 0.05),),
        element_heat_w=heat,
        nodal_heat_w=nodal,
        radiation=radiators,
    )


def test_the_prepared_thermal_system_is_the_numpy_operator() -> None:
    problem = _body()
    capacity = np.linspace(0.0, 2.0, problem.mesh.size)
    native = MatrixFreeThermalOperator(problem, native=True, capacity_per_s=capacity)
    portable = MatrixFreeThermalOperator(problem, native=False, capacity_per_s=capacity)
    system = native._system
    assert np.array_equal(system.coefficients, portable._coefficients_high)
    assert np.array_equal(system.unit, portable._unit_high)
    assert np.array_equal(system.robin_total, portable._robin_total_high)
    assert np.array_equal(native.free_nodes, portable.free_nodes)
    assert np.array_equal(native.fixed_temperature_k, portable.fixed_temperature_k)
    assert np.array_equal(native._diagonal_high, portable._diagonal_high)
    assert np.array_equal(native.nodal_load(), portable.nodal_load())
    assert native.default_reference_temperature() == portable.default_reference_temperature()
    assert native.coarse_correction.block == portable.coarse_correction.block
    scale = np.abs(portable.coarse_correction.coarse_matrix).max()
    np.testing.assert_allclose(
        native.coarse_correction.coarse_matrix, portable.coarse_correction.coarse_matrix, rtol=0, atol=1e-13 * scale
    )

    rng = np.random.default_rng(4)
    temperature = AMBIENT + rng.standard_normal(native.size)
    previous = AMBIENT + rng.standard_normal(native.size)
    previous[3] = np.nan
    for name, arguments in (
        ("apply_high", (temperature,)),
        ("build_rhs", (AMBIENT + 0.5, previous)),
        ("unconstrained_residual", (temperature, previous)),
        ("stored_heat_w", (temperature, previous)),
        ("convective_heat", (temperature,)),
        ("element_heat_flux", (temperature,)),
    ):
        expected = getattr(portable, name)(*arguments)
        np.testing.assert_allclose(
            getattr(native, name)(*arguments), expected, rtol=0, atol=DIRECT_RTOL * np.abs(expected).max(), err_msg=name
        )


@pytest.mark.parametrize("radiation", [False, True])
def test_the_native_steady_solve_agrees_with_numpy(radiation: bool) -> None:
    problem = _body(radiation=radiation)
    config = MPIRConfig(max_outer_iterations=16)
    native = solve_thermal_conduction(problem, native=True, config=config)
    portable = solve_thermal_conduction(problem, native=False, config=config)
    assert native.solve.converged and portable.solve.converged
    assert native.radiation_converged and portable.radiation_converged
    assert native.radiation_iterations == portable.radiation_iterations
    active = ~np.isnan(portable.temperature_k)
    assert np.array_equal(active, ~np.isnan(native.temperature_k))
    rise = np.abs(portable.temperature_k[active] - AMBIENT).max()
    rtol = iterative_rtol(config.relative_tolerance)
    np.testing.assert_allclose(native.temperature_k[active], portable.temperature_k[active], rtol=0, atol=rtol * rise)
    for name in ("convective_heat_w", "radiative_heat_w"):
        np.testing.assert_allclose(
            getattr(native, name), getattr(portable, name), rtol=rtol, atol=1e-12, err_msg=name
        )
    assert native.total_heat_input_w == pytest.approx(portable.total_heat_input_w, rel=DIRECT_RTOL)
    assert abs(native.heat_balance_error_w) <= 1e-6 * native.total_heat_input_w
    flux = np.abs(portable.heat_flux_w_per_m2).max()
    np.testing.assert_allclose(native.heat_flux_w_per_m2, portable.heat_flux_w_per_m2, rtol=0, atol=1e-6 * flux)


def test_the_native_transient_takes_the_numpy_steps() -> None:
    problem = _body(capacity=True)
    schedule = TimeSchedule.geometric(0.05, 30.0, growth=1.6, max_step_s=4.0)
    native = solve_thermal_transient(problem, schedule, native=True, until_steady=True, steady_tolerance_k_per_s=0.05)
    portable = solve_thermal_transient(problem, schedule, native=False, until_steady=True, steady_tolerance_k_per_s=0.05)
    assert len(native.history) == len(portable.history)
    assert native.reached_steady == portable.reached_steady
    np.testing.assert_array_equal(native.times_s, portable.times_s)
    active = ~np.isnan(portable.temperature_k)
    assert np.array_equal(active, ~np.isnan(native.temperature_k))
    rise = np.abs(portable.temperature_k[active] - AMBIENT).max()
    np.testing.assert_allclose(native.temperature_k[active], portable.temperature_k[active], rtol=0, atol=1e-6 * rise)
    for a, b in zip(native.history, portable.history):
        assert (a.index, a.time_s, a.step_s, a.converged) == (b.index, b.time_s, b.step_s, b.converged)
        assert a.stored_heat_w == pytest.approx(b.stored_heat_w, rel=1e-5, abs=1e-9)


def test_the_native_thermal_solves_have_the_same_bits_at_every_budget() -> None:
    steady = _body(radiation=True)
    transient = _body(radiation=True, capacity=True)
    schedule = TimeSchedule.uniform(0.5, 2.0)
    outputs = []
    for budget in (1, 2, 3, 8):
        with thread_budget_scope(budget):
            solution = solve_thermal_conduction(steady, native=True)
            run = solve_thermal_transient(transient, schedule, native=True)
        outputs.append(
            (
                solution.temperature_k.tobytes(),
                solution.heat_flux_w_per_m2.tobytes(),
                solution.convective_heat_w.tobytes(),
                solution.radiative_heat_w.tobytes(),
                solution.heat_balance_error_w,
                solution.solve.history,
                run.temperature_k.tobytes(),
                run.history,
            )
        )
    assert all(output == outputs[0] for output in outputs[1:])


def test_the_native_solve_re_references_like_numpy() -> None:
    # A first solve that stops short of the tolerance (here: two outer steps
    # for 1e-12) is re-referenced to its mean rise and continued from the
    # iterate, on both paths.
    problem = _body()
    config = MPIRConfig(relative_tolerance=1.0e-12, max_outer_iterations=2)
    native = solve_thermal_conduction(problem, native=True, config=config)
    portable = solve_thermal_conduction(problem, native=False, config=config)
    assert len(portable.solve.history) > config.max_outer_iterations  # the restart ran
    assert len(native.solve.history) == len(portable.solve.history)
    assert native.solve.outer_iterations == portable.solve.outer_iterations
    active = ~np.isnan(portable.temperature_k)
    rise = np.abs(portable.temperature_k[active] - AMBIENT).max()
    np.testing.assert_allclose(native.temperature_k[active], portable.temperature_k[active], rtol=0, atol=1e-6 * rise)
