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
    VoltageTerminal,
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



def _voltage_strip_problem(columns: int = 4, voltage_v: float = 1.0e-3) -> PCBConductionProblem:
    mesh = _strip_problem(columns).mesh
    return PCBConductionProblem(
        mesh=mesh,
        voltage_terminals=(
            VoltageTerminal(((0, 0, 0), (0, 1, 0)), voltage_v, "source"),
            VoltageTerminal(((0, 0, columns), (0, 1, columns)), 0.0, "sink"),
        ),
    )


def test_voltage_driven_strip_matches_closed_form_current() -> None:
    voltage = 1.0e-3
    problem = _voltage_strip_problem(columns=4, voltage_v=voltage)
    solution = solve_pcb_dc(problem)
    conductivity = float(problem.mesh.conductivity_s_per_m[0, 0, 0])
    thickness = problem.mesh.layer_thickness_m[0]
    resistance = 4.0e-3 / (conductivity * 1.0e-3 * thickness)
    current = voltage / resistance

    assert solution.solve.converged
    np.testing.assert_allclose(
        solution.voltage_terminal_current_a, (current, -current), rtol=2.0e-10
    )
    assert solution.joule_loss_w == pytest.approx(voltage * current, rel=2.0e-10)
    np.testing.assert_allclose(solution.potential_v[0, :, 0], voltage)
    np.testing.assert_allclose(solution.potential_v[0, :, 4], 0.0)
    np.testing.assert_allclose(
        solution.potential_v[0, 0], voltage * np.linspace(1.0, 0.0, 5), atol=1.0e-12
    )


def test_voltage_driven_solve_matches_the_dense_lifted_system() -> None:
    mesh = LayeredPCBMesh(
        element_active=np.ones((2, 3, 4), dtype=bool),
        layer_thickness_m=(35.0e-6, 18.0e-6),
        pitch_x_m=(0.4e-3, 0.5e-3, 0.6e-3, 0.4e-3),
        pitch_y_m=0.5e-3,
    )
    problem = PCBConductionProblem(
        mesh=mesh,
        terminals=(CurrentTerminal(((0, 3, 2),), -0.25, "load"),),
        voltage_terminals=(
            VoltageTerminal(((0, 0, 0), (0, 1, 0)), 1.2, "vrm"),
            VoltageTerminal(((1, 3, 4), (1, 2, 4)), 0.0, "return"),
        ),
        vias=(ViaConnection((0, 1, 2), (1, 1, 2), 1.0e-3),),
    )
    solution = solve_pcb_dc(problem, config=MPIRConfig(relative_tolerance=1.0e-12))
    operator = MatrixFreePCBOperator(
        mesh,
        dirichlet_nodes=[node for terminal in problem.voltage_terminals for node in terminal.nodes],
        vias=problem.vias,
    )
    identity = np.eye(operator.size)
    full = np.column_stack([operator.apply_full_high(column) for column in identity.T])
    fixed = operator.dirichlet_potential(problem.voltage_terminals)
    known = ~operator.free_nodes.reshape(-1)
    free = ~known
    injection = np.zeros(operator.size)
    injection[np.ravel_multi_index((0, 3, 2), mesh.node_shape)] = -0.25
    expected = fixed.copy()
    expected[free] = np.linalg.solve(
        full[np.ix_(free, free)], injection[free] - full[np.ix_(free, known)] @ fixed[known]
    )

    assert solution.solve.converged
    np.testing.assert_allclose(solution.potential_v.reshape(-1), expected, rtol=1.0e-10, atol=1.0e-13)
    # Kirchhoff: the two sources supply what the load draws.
    assert float(np.sum(solution.voltage_terminal_current_a)) == pytest.approx(0.25, rel=1.0e-9)
    supplied = float(np.dot(solution.voltage_terminal_current_a, (1.2, 0.0)))
    load_voltage = float(solution.potential_v[0, 3, 2])
    assert solution.joule_loss_w == pytest.approx(supplied - 0.25 * load_voltage, rel=1.0e-9)


def test_voltage_driven_solve_warm_starts_from_its_own_potential() -> None:
    problem = _voltage_strip_problem(columns=8)
    cold = solve_pcb_dc(problem)
    warm = solve_pcb_dc(problem, initial_potential_v=cold.potential_v)

    assert warm.solve.converged
    assert warm.solve.inner_iterations < cold.solve.inner_iterations
    np.testing.assert_allclose(warm.potential_v, cold.potential_v, rtol=1.0e-10, atol=1.0e-15)


def test_inconsistent_terminal_drives_are_rejected() -> None:
    mesh = _strip_problem(columns=1).mesh
    source = VoltageTerminal(((0, 0, 0),), 1.0, "source")
    with pytest.raises(ValueError, match="reference_node must be None"):
        PCBConductionProblem(mesh=mesh, voltage_terminals=(source,), reference_node=(0, 0, 1))
    with pytest.raises(ValueError, match="belongs to terminals"):
        PCBConductionProblem(
            mesh=mesh,
            terminals=(CurrentTerminal(((0, 0, 0),), 1.0, "load"),),
            voltage_terminals=(source,),
        )
    with pytest.raises(ValueError, match="needs a reference_node"):
        PCBConductionProblem(
            mesh=mesh,
            terminals=(
                CurrentTerminal(((0, 0, 0),), 1.0),
                CurrentTerminal(((0, 0, 1),), -1.0),
            ),
        )
    with pytest.raises(ValueError, match="current or voltage terminals"):
        PCBConductionProblem(mesh=mesh, reference_node=(0, 0, 0))


class _DenseFloat32System:
    """An SPD matrix on the NumPy FP32 runtime, with no native inner solve."""

    inner_solver = "pcg"
    high_dtype = np.float64

    def __init__(self, matrix: np.ndarray) -> None:
        from electrical.matrix_free_mpir_fem.runtime import NumpyFloat32Runtime

        self.matrix = matrix
        self.size = matrix.shape[0]
        self.runtime = NumpyFloat32Runtime()

    def apply_high(self, vector: np.ndarray) -> np.ndarray:
        return self.matrix @ vector

    def apply_low(self, vector: np.ndarray) -> np.ndarray:
        return (self.matrix.astype(np.float32) @ vector).astype(np.float32)

    def diagonal_low(self) -> np.ndarray:
        return np.diag(self.matrix).astype(np.float32)


def test_the_fp32_inner_pcg_solves_a_correction_far_below_one() -> None:
    """A correction of ~1e-21 squared underflows FP32; the PCG scales it first,
    by a power of two, so the answer is the unit one scaled to the bit."""
    from electrical.matrix_free_mpir_fem.solver import _inner_pcg

    rng = np.random.default_rng(0)
    coupling = rng.standard_normal((12, 12))
    matrix = coupling @ coupling.T + 12.0 * np.eye(12)
    system = _DenseFloat32System(matrix)
    config = MPIRConfig(inner_relative_tolerance=1e-4)
    unit = rng.standard_normal(12)
    tiny = unit * 2.0**-70  # about 8.5e-22
    big, *_ = _inner_pcg(system, unit, config)
    small, iterations, residual, _ = _inner_pcg(system, tiny, config)
    assert residual <= 1e-4 and iterations > 0
    np.testing.assert_array_equal(small * 2.0**70, big)
    np.testing.assert_allclose(big, np.linalg.solve(matrix, unit), rtol=1e-3)
