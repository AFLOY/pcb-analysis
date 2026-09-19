"""Surface-to-ambient radiation as a Newton-linearised Robin boundary."""

from __future__ import annotations

import numpy as np
import pytest

from thermal.matrix_free_mpir_fem import (
    STEFAN_BOLTZMANN_W_PER_M2_K4 as SIGMA,
    ConvectionBoundary,
    ExposedFaceConvection,
    ExposedFaceRadiation,
    LayeredThermalMesh,
    RadiationBoundary,
    ThermalConductionProblem,
    newton_linearisation,
    solve_thermal_conduction,
)

ROWS, COLS, PITCH, THICKNESS, K = 8, 10, 1.0e-3, 1.0e-3, 200.0
AREA = ROWS * COLS * PITCH * PITCH
AMBIENT = 300.0


def _plate(power_w: float) -> tuple[LayeredThermalMesh, np.ndarray]:
    mesh = LayeredThermalMesh((THICKNESS,), PITCH, PITCH, K, element_shape=(ROWS, COLS))
    heat = np.full((1, ROWS, COLS), power_w / (ROWS * COLS))
    return mesh, heat


def _surface_temperature(power_w: float, emissivity: float) -> float:
    return (AMBIENT**4 + power_w / (emissivity * SIGMA * AREA)) ** 0.25


@pytest.mark.parametrize("power_w, emissivity", [(0.05, 0.9), (0.6, 0.8), (3.0, 0.8)])
@pytest.mark.parametrize("kind", ["top", "exposed"])
def test_uniformly_heated_plate_radiating_from_its_top_matches_the_analytic_surface_temperature(
    kind, power_w, emissivity
) -> None:
    mesh, heat = _plate(power_w)
    if kind == "top":
        radiation = RadiationBoundary("top", emissivity, AMBIENT)
    else:
        radiation = ExposedFaceRadiation(emissivity, AMBIENT, directions=("+z",))
    solution = solve_thermal_conduction(ThermalConductionProblem(mesh, radiation=(radiation,), element_heat_w=heat))

    surface = _surface_temperature(power_w, emissivity)
    # Uniform volumetric heat conducted to the top face raises the bottom by q t / (2 k).
    conduction_drop = power_w / AREA * THICKNESS / (2.0 * K)
    assert solution.radiation_converged and solution.solve.converged
    assert solution.radiation_iterations <= 15
    np.testing.assert_allclose(solution.temperature_k[1], surface, rtol=1.0e-7)
    np.testing.assert_allclose(solution.temperature_k[0], surface + conduction_drop, rtol=1.0e-7)
    assert solution.radiative_heat_w.shape == (1,)
    assert solution.radiative_heat_w[0] == pytest.approx(power_w, rel=1.0e-9)
    assert solution.convective_heat_w.shape == (0,)
    assert abs(solution.heat_balance_error_w) < 1.0e-9 * max(power_w, 1.0)


def test_newton_linearisation_is_exact_at_the_linearisation_point_and_converges_fast() -> None:
    emissivity, surface = 0.7, np.array([350.0, 420.0])
    coefficient, effective = newton_linearisation(emissivity, surface, AMBIENT)
    exact = emissivity * SIGMA * (surface**4 - AMBIENT**4)
    np.testing.assert_allclose(coefficient * (surface - effective), exact, rtol=1.0e-12)
    np.testing.assert_allclose(coefficient, 4.0 * emissivity * SIGMA * surface**3)

    # An 80 K rise from the ambient start converges in a handful of Newton steps.
    mesh, heat = _plate(0.05)
    solution = solve_thermal_conduction(
        ThermalConductionProblem(mesh, radiation=(RadiationBoundary("top", 0.8, AMBIENT),), element_heat_w=heat)
    )
    assert 40.0 < solution.max_temperature_k - AMBIENT < 120.0
    assert solution.radiation_iterations <= 6
    # Warm-started from the answer, one confirming step suffices.
    again = solve_thermal_conduction(
        ThermalConductionProblem(mesh, radiation=(RadiationBoundary("top", 0.8, AMBIENT),), element_heat_w=heat),
        initial_temperature_k=solution.temperature_k,
    )
    assert again.radiation_iterations == 1


def test_convection_and_radiation_share_the_budget_and_radiation_alone_fixes_the_level() -> None:
    mesh, heat = _plate(0.6)
    h, emissivity = 8.0, 0.85
    solution = solve_thermal_conduction(
        ThermalConductionProblem(
            mesh,
            convection=(ConvectionBoundary("top", h, AMBIENT),),
            radiation=(RadiationBoundary("top", emissivity, AMBIENT),),
            element_heat_w=heat,
        )
    )
    surface = float(np.mean(solution.temperature_k[1]))
    convective = h * AREA * (surface - AMBIENT)
    radiative = emissivity * SIGMA * AREA * (surface**4 - AMBIENT**4)
    assert solution.convective_heat_w[0] == pytest.approx(convective, rel=1.0e-6)
    assert solution.radiative_heat_w[0] == pytest.approx(radiative, rel=1.0e-6)
    assert convective + radiative == pytest.approx(0.6, rel=1.0e-6)
    # Cooler than radiation alone, hotter than convection alone.
    only_convection = solve_thermal_conduction(
        ThermalConductionProblem(mesh, convection=(ConvectionBoundary("top", h, AMBIENT),), element_heat_w=heat)
    )
    assert AMBIENT < solution.max_temperature_k < only_convection.max_temperature_k

    with pytest.raises(ValueError, match="radiating face"):
        ThermalConductionProblem(mesh, element_heat_w=heat)
    with pytest.raises(ValueError, match="emissivity"):
        RadiationBoundary("top", 1.2, AMBIENT)
    with pytest.raises(ValueError, match="kelvin"):
        ExposedFaceRadiation(0.5, 0.0)
    with pytest.raises(ValueError, match="rows, cols"):
        ThermalConductionProblem(mesh, radiation=(RadiationBoundary("top", np.ones((3, 3)), AMBIENT),))


def test_exposed_face_convection_accepts_per_element_coefficient_and_ambient() -> None:
    mesh, heat = _plate(0.6)
    coefficient = np.zeros(mesh.element_grid_shape)
    coefficient[0, :4] = 10.0  # cool half of the plate from every exposed face
    ambient = np.full(mesh.element_grid_shape, AMBIENT)
    ambient[0, :4] = 290.0
    per_element = ExposedFaceConvection(coefficient, ambient)
    uniform = ExposedFaceConvection(10.0, 290.0)
    weights, rhs = per_element.lumped_nodal_weights(mesh)
    all_weights, _ = uniform.lumped_nodal_weights(mesh)
    assert per_element.cools and per_element.mean_ambient_k() == pytest.approx(295.0)
    assert 0.0 < float(np.sum(weights)) < float(np.sum(all_weights))
    np.testing.assert_allclose(rhs, weights * 290.0)
    with pytest.raises(ValueError, match="slabs, rows, cols"):
        ThermalConductionProblem(mesh, convection=(ExposedFaceConvection(np.ones((2, 2, 2)), AMBIENT),))
    solution = solve_thermal_conduction(ThermalConductionProblem(mesh, convection=(per_element,), element_heat_w=heat))
    assert solution.solve.converged and abs(solution.heat_balance_error_w) < 1.0e-9


def test_masked_body_radiates_only_from_exposed_faces() -> None:
    """A cube in a void radiates from its six faces; the void's nodes stay silent."""

    active = np.zeros((4, 6, 6), dtype=bool)
    active[1:3, 2:4, 2:4] = True
    mesh = LayeredThermalMesh((1e-3,) * 4, 1e-3, 1e-3, np.where(active, 200.0, 1.0), active=active)
    heat = np.where(active, 0.5 / 8, 0.0)
    radiation = ExposedFaceRadiation(0.9, AMBIENT)
    solution = solve_thermal_conduction(ThermalConductionProblem(mesh, radiation=(radiation,), element_heat_w=heat))
    area = 6 * (2e-3) ** 2
    surface = (AMBIENT**4 + 0.5 / (0.9 * SIGMA * area)) ** 0.25
    assert solution.radiation_converged
    assert solution.radiative_heat_w[0] == pytest.approx(0.5, rel=1.0e-9)
    assert np.nanmin(solution.temperature_k) > surface - 0.5
    assert np.nanmax(solution.temperature_k) < surface + 0.5
    assert np.all(np.isnan(solution.temperature_k[0])) and np.all(np.isnan(solution.temperature_k[:, 0]))
