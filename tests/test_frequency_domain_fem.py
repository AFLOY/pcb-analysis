from __future__ import annotations

import numpy as np
import pytest

from electrical.matrix_free_mpir_fem import (
    EPSILON_0_F_PER_M,
    MU_0_H_PER_M,
    MPIRConfig,
    MatrixFreeScalarMaxwellOperator,
    ScalarMaxwellMesh2D,
    ScalarMaxwellProblem,
    make_complex64_runtime,
    propagation_constant_per_m,
    skin_depth_m,
    solve_scalar_maxwell,
)


def _end_driven_problem(
    mesh: ScalarMaxwellMesh2D,
    frequency_hz: float,
    *,
    right_value: complex = 0.0,
) -> ScalarMaxwellProblem:
    mask = np.zeros(mesh.node_shape, dtype=bool)
    mask[:, 0] = True
    mask[:, -1] = True
    values = np.zeros(mesh.node_shape, dtype=np.complex128)
    values[:, 0] = 1.0
    values[:, -1] = right_value
    return ScalarMaxwellProblem(mesh, frequency_hz, mask, values)


def _config(tolerance: float = 1.0e-11) -> MPIRConfig:
    return MPIRConfig(
        relative_tolerance=tolerance,
        inner_relative_tolerance=2.0e-3,
        max_outer_iterations=12,
        max_inner_iterations=300,
        gmres_restart=32,
    )


def test_backend_selector_preserves_cpu_default_and_rejects_ambiguity() -> None:
    mesh = ScalarMaxwellMesh2D((1, 2), 0.5e-3, 1.0e-3)
    problem = _end_driven_problem(mesh, 1.0e6)
    operator = MatrixFreeScalarMaxwellOperator(problem, backend="cpu")

    assert operator.runtime.name == "numpy-complex64"
    assert operator.low_operator_backend == "portable-array-q1"
    assert make_complex64_runtime("auto").name in {
        "numpy-complex64",
        "cupy-complex64",
    }
    with pytest.raises(ValueError, match="either runtime or backend"):
        MatrixFreeScalarMaxwellOperator(
            problem,
            runtime=make_complex64_runtime("cpu"),
            backend="cpu",
        )
    with pytest.raises(ValueError, match="backend must"):
        make_complex64_runtime("invalid")  # type: ignore[arg-type]


def test_complex_matrix_free_action_is_complex_symmetric() -> None:
    mesh = ScalarMaxwellMesh2D(
        (2, 3),
        0.4e-3,
        0.5e-3,
        relative_permittivity=np.array([[3.8, 4.0, 4.2], [3.7, 3.9, 4.1]]),
        conductivity_s_per_m=np.array([[0.0, 2.0, 0.0], [1.0, 0.0, 3.0]]),
        dielectric_loss_tangent=0.015,
    )
    problem = _end_driven_problem(mesh, 2.0e9)
    operator = MatrixFreeScalarMaxwellOperator(problem)
    identity = np.eye(operator.size, dtype=np.complex128)
    dense = np.column_stack(
        [operator.apply_high(identity[:, column]) for column in range(operator.size)]
    )

    np.testing.assert_allclose(dense, dense.T, rtol=1.0e-13, atol=1.0e-9)
    vector = np.linspace(0.1, 1.2, operator.size) * (1.0 + 0.3j)
    np.testing.assert_allclose(operator.apply_high(vector), dense @ vector)


@pytest.mark.parametrize("elements", [8, 16, 32])
def test_lossless_dielectric_wave_converges_to_closed_form(elements: int) -> None:
    frequency_hz = 1.0e9
    relative_permittivity = 4.0
    length_m = 20.0e-3
    mesh = ScalarMaxwellMesh2D(
        (1, elements),
        length_m / elements,
        1.0e-3,
        relative_permittivity=relative_permittivity,
    )
    solution = solve_scalar_maxwell(
        _end_driven_problem(mesh, frequency_hz), config=_config()
    )
    x = np.linspace(0.0, length_m, elements + 1)
    wave_number = 2.0 * np.pi * frequency_hz * np.sqrt(
        MU_0_H_PER_M * EPSILON_0_F_PER_M * relative_permittivity
    )
    exact = np.sin(wave_number * (length_m - x)) / np.sin(
        wave_number * length_m
    )
    relative_error = np.linalg.norm(
        solution.electric_field_z_v_per_m[0] - exact
    ) / np.linalg.norm(exact)

    assert solution.solve.converged
    # Q1 field error falls quadratically; this bound is tight enough to catch
    # omission of the displacement-current mass term.
    assert relative_error < 3.7e-3 / (elements * elements)
    assert solution.dielectric_loss_w_per_m == pytest.approx(0.0, abs=1.0e-20)


def test_lossy_dielectric_matches_complex_propagation_constant() -> None:
    frequency_hz = 2.0e9
    relative_permittivity = 4.2
    loss_tangent = 0.02
    length_m = 15.0e-3
    elements = 48
    mesh = ScalarMaxwellMesh2D(
        (1, elements),
        length_m / elements,
        1.0e-3,
        relative_permittivity=relative_permittivity,
        dielectric_loss_tangent=loss_tangent,
    )
    solution = solve_scalar_maxwell(
        _end_driven_problem(mesh, frequency_hz), config=_config()
    )
    gamma = propagation_constant_per_m(
        frequency_hz,
        relative_permittivity=relative_permittivity,
        dielectric_loss_tangent=loss_tangent,
    )
    x = np.linspace(0.0, length_m, elements + 1)
    exact = np.sinh(gamma * (length_m - x)) / np.sinh(gamma * length_m)
    relative_error = np.linalg.norm(
        solution.electric_field_z_v_per_m[0] - exact
    ) / np.linalg.norm(exact)

    assert relative_error < 1.2e-5
    assert solution.dielectric_loss_w_per_m > 0.0
    assert solution.conduction_loss_w_per_m == pytest.approx(0.0, abs=1.0e-20)


def test_copper_slab_resolves_eddy_current_skin_effect_and_impedance() -> None:
    conductivity = 1.0 / 1.724e-8
    frequency_hz = 1.0e6
    thickness_m = 0.5e-3
    elements = 64
    mesh = ScalarMaxwellMesh2D(
        (1, elements),
        thickness_m / elements,
        1.0e-3,
        conductivity_s_per_m=conductivity,
    )
    solution = solve_scalar_maxwell(
        _end_driven_problem(mesh, frequency_hz, right_value=1.0),
        config=_config(),
    )
    gamma = propagation_constant_per_m(
        frequency_hz, conductivity_s_per_m=conductivity
    )
    x = np.linspace(-thickness_m / 2.0, thickness_m / 2.0, elements + 1)
    exact_field = np.cosh(gamma * x) / np.cosh(gamma * thickness_m / 2.0)
    relative_field_error = np.linalg.norm(
        solution.electric_field_z_v_per_m[0] - exact_field
    ) / np.linalg.norm(exact_field)

    nodal_field = solution.electric_field_z_v_per_m[0]
    sheet_current = conductivity * np.sum(
        0.5
        * (nodal_field[:-1] + nodal_field[1:])
        * (thickness_m / elements)
    )
    numerical_impedance = 1.0 / sheet_current
    exact_impedance = gamma / (
        2.0 * conductivity * np.tanh(gamma * thickness_m / 2.0)
    )
    relative_impedance_error = abs(numerical_impedance - exact_impedance) / abs(
        exact_impedance
    )

    assert solution.solve.converged
    assert skin_depth_m(frequency_hz, conductivity) == pytest.approx(
        66.0828496e-6, rel=2.0e-8
    )
    assert relative_field_error < 1.2e-3
    assert relative_impedance_error < 1.3e-3
    assert abs(solution.eddy_current_density_z_a_per_m2[0, 0]) > abs(
        solution.eddy_current_density_z_a_per_m2[0, elements // 2]
    )
    assert solution.conduction_loss_w_per_m > 0.0
    assert solution.solve.low_operator_applications > (
        20 * solution.solve.high_operator_applications
    )
