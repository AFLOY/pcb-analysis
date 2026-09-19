from __future__ import annotations

import numpy as np
import pytest

from electrical.matrix_free_mpir_fem import (
    CurrentTerminal,
    LayeredPCBMesh,
    MPIRConfig,
    MatrixFreePCBOperator,
    PCBConductionProblem,
    ViaConnection,
    solve_mpir,
    solve_pcb_dc,
)


def _strip_problem(columns: int = 4) -> PCBConductionProblem:
    mesh = LayeredPCBMesh(
        element_active=np.ones((1, 1, columns), dtype=bool),
        layer_thickness_m=(35.0e-6,),
        pitch_x_m=1.0e-3,
        pitch_y_m=1.0e-3,
    )
    return PCBConductionProblem(
        mesh=mesh,
        terminals=(
            CurrentTerminal(((0, 0, 0), (0, 1, 0)), 1.0, "source"),
            CurrentTerminal(
                ((0, 0, columns), (0, 1, columns)), -1.0, "sink"
            ),
        ),
        reference_node=(0, 0, columns),
    )


def _dense_from_actions(operator: MatrixFreePCBOperator) -> np.ndarray:
    identity = np.eye(operator.size)
    return np.column_stack(
        [operator.apply_high(identity[:, column]) for column in range(operator.size)]
    )


def test_matrix_free_operator_matches_its_dense_action_and_is_spd() -> None:
    problem = _strip_problem(columns=2)
    operator = MatrixFreePCBOperator(
        problem.mesh, reference_node=problem.reference_node
    )
    dense = _dense_from_actions(operator)
    np.testing.assert_allclose(dense, dense.T, atol=1.0e-12)
    assert np.linalg.eigvalsh(dense)[0] > 0.0

    vector = np.linspace(-0.4, 0.8, operator.size)
    np.testing.assert_allclose(operator.apply_high(vector), dense @ vector)


def test_mpir_reaches_fp64_residual_with_most_actions_in_fp32() -> None:
    problem = _strip_problem(columns=8)
    operator = MatrixFreePCBOperator(
        problem.mesh, reference_node=problem.reference_node
    )
    rhs = operator.build_rhs(problem.terminals)
    dense = _dense_from_actions(operator)
    expected = np.linalg.solve(dense, rhs)

    result = solve_mpir(
        operator,
        rhs,
        config=MPIRConfig(
            relative_tolerance=1.0e-11,
            inner_relative_tolerance=2.0e-3,
            max_outer_iterations=8,
        ),
    )

    assert result.converged
    assert result.relative_residual <= 1.0e-11
    assert result.low_operator_applications > result.high_operator_applications
    assert result.low_runtime == "numpy-fp32"
    np.testing.assert_allclose(result.solution, expected, rtol=2.0e-10, atol=1.0e-13)


def test_uniform_strip_matches_closed_form_resistance_and_current_density() -> None:
    problem = _strip_problem(columns=4)
    solution = solve_pcb_dc(problem)
    conductivity = float(problem.mesh.conductivity_s_per_m[0, 0, 0])
    length = 4.0e-3
    width = 1.0e-3
    thickness = problem.mesh.layer_thickness_m[0]
    expected_resistance = length / (conductivity * width * thickness)

    assert solution.solve.converged
    assert solution.joule_loss_w == pytest.approx(expected_resistance, rel=2.0e-10)
    assert solution.max_current_density_a_per_m2 == pytest.approx(
        1.0 / (width * thickness), rel=2.0e-10
    )
    # Transverse current at the FP64 residual floor relative to the axial density.
    np.testing.assert_allclose(
        solution.current_density_a_per_m2[..., 1], 0.0, atol=1.0e-9 / (width * thickness)
    )


def test_layered_problem_reports_the_only_via_current() -> None:
    mesh = LayeredPCBMesh(
        element_active=np.ones((2, 1, 1), dtype=bool),
        layer_thickness_m=(35.0e-6, 35.0e-6),
        pitch_x_m=1.0e-3,
        pitch_y_m=1.0e-3,
    )
    problem = PCBConductionProblem(
        mesh=mesh,
        terminals=(
            CurrentTerminal(((0, 0, 0),), 1.0, "source"),
            CurrentTerminal(((1, 0, 0),), -1.0, "sink"),
        ),
        reference_node=(1, 0, 0),
        vias=(ViaConnection((0, 0, 1), (1, 0, 1), 2.0e-3),),
    )
    solution = solve_pcb_dc(problem)

    assert solution.solve.converged
    assert abs(solution.via_current_a[0]) == pytest.approx(1.0, rel=2.0e-9)


def test_unbalanced_terminal_currents_are_rejected() -> None:
    mesh = _strip_problem(columns=1).mesh
    with pytest.raises(ValueError, match="sum to zero"):
        PCBConductionProblem(
            mesh=mesh,
            terminals=(CurrentTerminal(((0, 0, 0),), 1.0),),
            reference_node=(0, 0, 0),
        )

