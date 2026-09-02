from __future__ import annotations

import numpy as np
import pytest

from electrical.matrix_free_mpir_fem import MPIRConfig, solve_mpir
from thermal.matrix_free_mpir_fem import (
    ConvectionBoundary,
    HeatSource,
    LayeredThermalMesh,
    MatrixFreeThermalOperator,
    ThermalConductionProblem,
    solve_thermal_conduction,
)


def _dense_from_actions(operator: MatrixFreeThermalOperator) -> np.ndarray:
    identity = np.eye(operator.size)
    return np.column_stack(
        [operator.apply_high(identity[:, column]) for column in range(operator.size)]
    )


def _uniform_mesh(
    slabs: int, thickness_m: float, shape: tuple[int, int], conductivity: float
) -> LayeredThermalMesh:
    return LayeredThermalMesh(
        slab_thickness_m=(thickness_m,) * slabs,
        pitch_x_m=1.0e-3,
        pitch_y_m=1.0e-3,
        conductivity_w_per_m_k=conductivity,
        element_shape=shape,
    )


def test_operator_matches_its_dense_action_and_is_spd() -> None:
    mesh = LayeredThermalMesh(
        slab_thickness_m=(35.0e-6, 0.4e-3),
        pitch_x_m=0.2e-3,
        pitch_y_m=0.3e-3,
        conductivity_w_per_m_k=np.array(
            [[[385.0, 385.0], [0.8, 385.0]], [[0.8, 0.8], [0.8, 385.0]]]
        ),
        through_plane_conductivity_w_per_m_k=(385.0, 0.3),
    )
    mask = np.zeros(mesh.node_shape, dtype=bool)
    mask[0, 0, 0] = True
    problem = ThermalConductionProblem(
        mesh,
        convection=(ConvectionBoundary("top", 12.0, 300.0),),
        fixed_temperature_mask=mask,
        fixed_temperature_k=310.0,
    )
    operator = MatrixFreeThermalOperator(problem)
    dense = _dense_from_actions(operator)

    np.testing.assert_allclose(dense, dense.T, atol=1.0e-12)
    assert np.linalg.eigvalsh(dense)[0] > 0.0
    vector = np.linspace(-0.4, 0.8, operator.size)
    np.testing.assert_allclose(operator.apply_high(vector), dense @ vector)
    # Constant temperature is in the kernel of pure conduction: without fixed
    # nodes only the lumped Robin conductance survives.
    convective_only = MatrixFreeThermalOperator(
        ThermalConductionProblem(
            mesh, convection=(ConvectionBoundary("top", 12.0, 300.0),)
        )
    )
    action = convective_only.apply_high(np.ones(operator.size)).reshape(mesh.node_shape)
    np.testing.assert_allclose(action[:-1], 0.0, atol=1.0e-12)
    face_area = 0.2e-3 * 0.3e-3
    np.testing.assert_allclose(action[-1, 1, 1], 12.0 * face_area)
    np.testing.assert_allclose(action[-1, 0, 0], 12.0 * face_area / 4.0)
    np.testing.assert_allclose(action[-1, 0, 1], 12.0 * face_area / 2.0)
    assert float(np.sum(action)) == pytest.approx(12.0 * face_area * 4)


def test_mpir_matches_dense_solve_with_fp32_inner_work() -> None:
    mesh = _uniform_mesh(3, 0.3e-3, (3, 4), 2.0)
    heat = np.zeros(mesh.element_grid_shape)
    heat[1, 1, 2] = 0.1
    mask = np.zeros(mesh.node_shape, dtype=bool)
    mask[0] = True
    problem = ThermalConductionProblem(
        mesh,
        convection=(ConvectionBoundary("top", 50.0, 295.0),),
        fixed_temperature_mask=mask,
        fixed_temperature_k=305.0,
        heat_sources=(HeatSource(((3, 0, 0), (3, 0, 1)), 0.05),),
        element_heat_w=heat,
    )
    operator = MatrixFreeThermalOperator(problem)
    rhs = operator.build_rhs()
    expected = np.linalg.solve(_dense_from_actions(operator), rhs)

    result = solve_mpir(
        operator, rhs, config=MPIRConfig(relative_tolerance=1.0e-11)
    )

    assert result.converged
    assert result.low_runtime == "numpy-fp32"
    assert result.low_operator_applications > result.high_operator_applications
    np.testing.assert_allclose(result.solution, expected, rtol=1.0e-9, atol=1.0e-9)
    np.testing.assert_allclose(
        result.solution.reshape(mesh.node_shape)[0], 305.0, atol=1.0e-12
    )


def test_fixed_faces_give_a_linear_profile_and_the_exact_flux() -> None:
    mesh = _uniform_mesh(4, 0.5e-3, (2, 3), 1.0)
    mask = np.zeros(mesh.node_shape, dtype=bool)
    mask[0] = mask[-1] = True
    values = np.zeros(mesh.node_shape)
    values[0] = 300.0
    values[-1] = 350.0
    problem = ThermalConductionProblem(
        mesh, fixed_temperature_mask=mask, fixed_temperature_k=values
    )
    solution = solve_thermal_conduction(problem)

    assert solution.solve.converged
    expected = 300.0 + 12.5 * np.arange(5)
    for face, temperature in zip(expected, solution.temperature_k):
        np.testing.assert_allclose(temperature, face, rtol=1.0e-9)
    np.testing.assert_allclose(
        solution.heat_flux_w_per_m2[..., 2], -50.0 / 2.0e-3, rtol=1.0e-8
    )
    np.testing.assert_allclose(solution.heat_flux_w_per_m2[..., :2], 0.0, atol=1e-4)
    # The two fixed faces exchange equal and opposite heat.
    assert solution.fixed_temperature_heat_w == pytest.approx(0.0, abs=1.0e-9)
    assert solution.heat_balance_error_w == pytest.approx(0.0, abs=1.0e-9)


def test_uniform_heating_with_one_cooled_face_matches_the_parabola() -> None:
    conductivity = 0.3
    film = 100.0
    ambient = 300.0
    total_power = 1.0
    mesh = _uniform_mesh(8, 0.25e-3, (4, 5), conductivity)
    element_count = np.prod(mesh.element_grid_shape)
    problem = ThermalConductionProblem(
        mesh,
        convection=(ConvectionBoundary("top", film, ambient),),
        element_heat_w=np.full(mesh.element_grid_shape, total_power / element_count),
    )
    solution = solve_thermal_conduction(problem, initial_temperature_k=ambient)

    area = 4.0e-3 * 5.0e-3
    thickness = 2.0e-3
    flux = total_power / area
    top = ambient + flux / film
    bottom = top + flux * thickness / (2.0 * conductivity)
    assert solution.solve.converged
    assert solution.temperature_k[-1, 2, 2] == pytest.approx(top, rel=1.0e-9)
    assert solution.temperature_k[0, 2, 2] == pytest.approx(bottom, rel=1.0e-9)
    assert solution.max_temperature_k == pytest.approx(bottom, rel=1.0e-9)
    assert solution.total_heat_input_w == pytest.approx(total_power)
    assert solution.convective_heat_w[0] == pytest.approx(total_power, rel=1.0e-9)
    assert solution.heat_balance_error_w == pytest.approx(0.0, abs=1.0e-9)


def test_nodal_source_heat_is_conserved_and_leaves_through_both_sinks() -> None:
    mesh = _uniform_mesh(2, 0.4e-3, (3, 3), 5.0)
    mask = np.zeros(mesh.node_shape, dtype=bool)
    mask[0, 0, :] = True
    problem = ThermalConductionProblem(
        mesh,
        convection=(ConvectionBoundary("bottom", 20.0, 290.0),),
        fixed_temperature_mask=mask,
        fixed_temperature_k=290.0,
        heat_sources=(HeatSource(((2, 1, 1), (2, 1, 2)), 0.3, "chip"),),
    )
    solution = solve_thermal_conduction(problem)

    assert solution.solve.converged
    assert solution.total_heat_input_w == pytest.approx(0.3)
    assert solution.convective_heat_w[0] > 0.0
    assert solution.fixed_temperature_heat_w > 0.0
    assert solution.convective_heat_w[0] + solution.fixed_temperature_heat_w == (
        pytest.approx(0.3, rel=1.0e-8)
    )
    assert solution.min_temperature_k == pytest.approx(290.0)
    assert np.argmax(solution.temperature_k) in {
        np.ravel_multi_index((2, 1, 1), mesh.node_shape),
        np.ravel_multi_index((2, 1, 2), mesh.node_shape),
    }


def test_mesh_broadcasts_per_slab_conductivity_and_reports_volumes() -> None:
    mesh = LayeredThermalMesh(
        slab_thickness_m=(35.0e-6, 1.5e-3),
        pitch_x_m=0.2e-3,
        pitch_y_m=0.1e-3,
        conductivity_w_per_m_k=(385.0, 0.8),
        element_shape=(2, 3),
        through_plane_conductivity_w_per_m_k=(385.0, 0.3),
    )
    assert mesh.element_grid_shape == (2, 2, 3)
    assert mesh.node_shape == (3, 3, 4)
    np.testing.assert_allclose(mesh.conductivity_w_per_m_k[1], 0.8)
    np.testing.assert_allclose(mesh.through_plane_conductivity_w_per_m_k[1], 0.3)
    np.testing.assert_allclose(mesh.element_volume_m3[0], 35.0e-6 * 0.2e-3 * 0.1e-3)


def test_floating_problem_and_bad_inputs_are_rejected() -> None:
    mesh = _uniform_mesh(1, 1.0e-3, (1, 1), 1.0)
    with pytest.raises(ValueError, match="positive film coefficient or a fixed node"):
        ThermalConductionProblem(mesh)
    with pytest.raises(ValueError, match="positive film coefficient or a fixed node"):
        ThermalConductionProblem(mesh, convection=(ConvectionBoundary("top", 0.0, 300.0),))
    with pytest.raises(ValueError, match="element_shape is required"):
        LayeredThermalMesh((1.0e-3,), 1.0e-3, 1.0e-3, 1.0)
    with pytest.raises(ValueError, match="finite and positive"):
        LayeredThermalMesh((1.0e-3,), 1.0e-3, 1.0e-3, 0.0, element_shape=(1, 1))
    with pytest.raises(ValueError, match="side"):
        ConvectionBoundary("left", 1.0, 300.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="outside mesh shape"):
        ThermalConductionProblem(
            mesh,
            convection=(ConvectionBoundary("top", 1.0, 300.0),),
            heat_sources=(HeatSource(((5, 0, 0),), 1.0),),
        )
    with pytest.raises(ValueError, match="element_heat_w"):
        ThermalConductionProblem(
            mesh,
            convection=(ConvectionBoundary("top", 1.0, 300.0),),
            element_heat_w=np.zeros((2, 1, 1)),
        )


def _copper_stack(rows: int, cols: int) -> ThermalConductionProblem:
    mesh = LayeredThermalMesh(
        slab_thickness_m=(35.0e-6, 1.5e-3, 35.0e-6),
        pitch_x_m=0.2e-3,
        pitch_y_m=0.2e-3,
        conductivity_w_per_m_k=(385.0, 0.8, 385.0),
        through_plane_conductivity_w_per_m_k=(385.0, 0.3, 385.0),
        element_shape=(rows, cols),
    )
    heat = np.zeros(mesh.element_grid_shape)
    heat[2, rows // 2, cols // 4 : 3 * cols // 4] = 0.5 / (cols // 2)
    return ThermalConductionProblem(
        mesh,
        convection=(
            ConvectionBoundary("top", 10.0, 298.15),
            ConvectionBoundary("bottom", 10.0, 298.15),
        ),
        element_heat_w=heat,
    )


def test_coarse_matrix_is_the_exact_galerkin_product() -> None:
    problem = _copper_stack(7, 9)
    mask = np.zeros(problem.mesh.node_shape, dtype=bool)
    mask[0, :2, :2] = True  # one patch made only of fixed nodes
    problem = ThermalConductionProblem(
        problem.mesh,
        convection=problem.convection,
        fixed_temperature_mask=mask,
        fixed_temperature_k=298.15,
        element_heat_w=problem.element_heat_w,
    )
    operator = MatrixFreeThermalOperator(problem, coarse_block_nodes=2)
    correction = operator.coarse_correction
    assert correction is not None
    assert correction.coarse_shape == (4, 4, 5)

    dense = _dense_from_actions(operator)
    layers, rows, cols = problem.mesh.node_shape
    prolongation = np.zeros((operator.size, correction.coarse_size))
    for node in range(operator.size):
        layer, row, col = np.unravel_index(node, problem.mesh.node_shape)
        if not operator.free_nodes[layer, row, col]:
            continue
        coarse = np.ravel_multi_index(
            (layer, row // 2, col // 2), correction.coarse_shape
        )
        prolongation[node, coarse] = 1.0
    expected = prolongation.T @ dense @ prolongation
    empty = np.sum(prolongation, axis=0) == 0.0
    assert empty.sum() == 1
    expected[empty, empty] = 1.0
    np.testing.assert_allclose(correction.coarse_matrix, expected, rtol=1e-12, atol=1e-15)

    # The preconditioner is SPD and its FP32 action tracks the FP64 action.
    preconditioner = np.column_stack(
        [correction.apply_high(np.eye(operator.size)[:, i]) for i in range(operator.size)]
    )
    np.testing.assert_allclose(preconditioner, preconditioner.T, atol=1e-12)
    assert np.linalg.eigvalsh(preconditioner)[0] > 0.0
    vector = np.random.default_rng(1).standard_normal(operator.size)
    low = operator.runtime.to_host(
        operator.precondition_low(operator.runtime.from_host(vector))
    )
    high = correction.apply_high(vector)
    np.testing.assert_allclose(low, high, rtol=0.0, atol=1e-5 * np.max(np.abs(high)))


def test_two_level_preconditioner_cuts_inner_iterations_on_a_copper_stack() -> None:
    problem = _copper_stack(24, 24)
    config = MPIRConfig(max_inner_iterations=3000)
    two_level = solve_thermal_conduction(
        problem, config=config, initial_temperature_k=298.15
    )
    jacobi = solve_thermal_conduction(
        problem, config=config, initial_temperature_k=298.15, preconditioner="jacobi"
    )

    assert two_level.solve.converged and jacobi.solve.converged
    assert two_level.solve.inner_iterations * 5 < jacobi.solve.inner_iterations
    np.testing.assert_allclose(
        two_level.temperature_k, jacobi.temperature_k, rtol=1e-9, atol=1e-7
    )
    assert two_level.heat_balance_error_w == pytest.approx(0.0, abs=1e-9)


def test_default_config_converges_on_a_copper_stack() -> None:
    solution = solve_thermal_conduction(_copper_stack(40, 40), initial_temperature_k=298.15)
    assert solution.solve.converged
    assert solution.solve.inner_iterations < 600


def test_block_size_selection_respects_the_coarse_size_cap() -> None:
    from thermal.matrix_free_mpir_fem import choose_block_size

    assert choose_block_size((4, 41, 41)) == 4
    assert choose_block_size((4, 201, 201), max_coarse_size=2048) == 10
    assert choose_block_size((2, 3, 3)) == 4
    with pytest.raises(ValueError, match="preconditioner must be"):
        MatrixFreeThermalOperator(_copper_stack(2, 2), preconditioner="ilu")  # type: ignore[arg-type]
