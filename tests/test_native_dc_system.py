"""The C++ layered DC system against the NumPy operator it replaced.

Everything the system computes without iterating (coefficients, masks,
diagonal, right-hand side, fields, losses) is a direct result and is held to
``DIRECT_RTOL`` (most of it agrees bit for bit); the solve is held to the
iterative tolerance.  Every output of a solve is the same at every thread
budget.
"""

from __future__ import annotations

import numpy as np
import pytest

from electrical.matrix_free_mpir_fem import (
    CurrentTerminal,
    MatrixFreePCBOperator,
    MPIRConfig,
    PCBConductionProblem,
    VoltageTerminal,
    solve_pcb_dc,
)
from electrical.matrix_free_mpir_fem.native_dc import native_available
from electrical.threads import thread_budget_scope
from tests.test_native_dc import _board
from tests.tolerance import DIRECT_RTOL, iterative_rtol

pytestmark = pytest.mark.skipif(not native_available(), reason="pcbcore extension not built")


def _voltage_driven(rows: int, cols: int) -> PCBConductionProblem:
    current = _board(rows, cols, graded=True)
    source, sink = current.terminals
    return PCBConductionProblem(
        mesh=current.mesh,
        voltage_terminals=(VoltageTerminal(source.nodes, 3.0e-3, "source"), VoltageTerminal(sink.nodes, 0.0, "sink")),
        vias=current.vias,
    )


def _operators(problem: PCBConductionProblem, **options) -> tuple[MatrixFreePCBOperator, MatrixFreePCBOperator]:
    arguments = dict(
        reference_node=problem.reference_node,
        dirichlet_nodes=tuple(node for terminal in problem.voltage_terminals for node in terminal.nodes),
        vias=problem.vias,
        **options,
    )
    return (
        MatrixFreePCBOperator(problem.mesh, native=True, **arguments),
        MatrixFreePCBOperator(problem.mesh, native=False, **arguments),
    )


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("preconditioner", ["two-level", "jacobi"])
def test_the_prepared_system_is_the_numpy_operator(graded: bool, preconditioner: str) -> None:
    problem = _board(9, 14, graded=graded)
    native, portable = _operators(problem, preconditioner=preconditioner)
    system = native._system
    assert np.array_equal(system.coefficients, portable._coefficients_high)
    assert np.array_equal(system.unit, portable._unit_high)
    assert np.array_equal(native.active_nodes, portable.active_nodes)
    assert np.array_equal(native.free_nodes, portable.free_nodes)
    assert np.array_equal(native._diagonal_high, portable._diagonal_high)
    if preconditioner == "two-level":
        assert native.coarse_correction.block == portable.coarse_correction.block
        scale = np.abs(portable.coarse_correction.coarse_matrix).max()
        np.testing.assert_allclose(
            native.coarse_correction.coarse_matrix, portable.coarse_correction.coarse_matrix, rtol=0, atol=1e-14 * scale
        )
    else:
        assert native.coarse_correction is None and not system.two_level

    vector = np.random.default_rng(7).standard_normal(native.size)
    for name in ("apply_high", "apply_full_high"):
        expected = getattr(portable, name)(vector)
        np.testing.assert_allclose(
            getattr(native, name)(vector), expected, rtol=0, atol=DIRECT_RTOL * np.abs(expected).max()
        )


def test_the_right_hand_side_and_the_lifting_are_the_numpy_ones() -> None:
    current = _board(9, 14, graded=True)
    native, portable = _operators(current)
    assert np.array_equal(native.build_rhs(current.terminals), portable.build_rhs(current.terminals))

    problem = _voltage_driven(9, 14)
    native, portable = _operators(problem)
    assert np.array_equal(
        native.dirichlet_potential(problem.voltage_terminals), portable.dirichlet_potential(problem.voltage_terminals)
    )
    expected = portable.build_rhs((), problem.voltage_terminals)
    np.testing.assert_allclose(
        native.build_rhs((), problem.voltage_terminals), expected, rtol=0, atol=DIRECT_RTOL * np.abs(expected).max()
    )
    potential = np.random.default_rng(8).standard_normal(native.size)
    potential[~native.active_nodes.reshape(-1)] = np.nan
    expected = portable.terminal_currents(potential, problem.voltage_terminals)
    np.testing.assert_allclose(
        native.terminal_currents(potential, problem.voltage_terminals),
        expected,
        rtol=0,
        atol=DIRECT_RTOL * np.abs(expected).max(),
    )


def test_the_post_processing_is_the_numpy_one() -> None:
    problem = _board(9, 14, graded=True)
    native, portable = _operators(problem)
    potential = np.random.default_rng(9).standard_normal(native.size)
    assert np.array_equal(native.element_electric_field(potential), portable.element_electric_field(potential))
    assert np.array_equal(native.via_currents(potential), portable.via_currents(potential))
    assert np.array_equal(native.via_joule_loss(potential), portable.via_joule_loss(potential))
    expected = portable.element_joule_loss(potential)
    np.testing.assert_allclose(
        native.element_joule_loss(potential), expected, rtol=0, atol=DIRECT_RTOL * np.abs(expected).max()
    )
    assert native.joule_loss(potential) == pytest.approx(portable.joule_loss(potential), rel=DIRECT_RTOL)


@pytest.mark.parametrize("voltage", [False, True])
def test_the_native_dc_solution_agrees_with_numpy(voltage: bool) -> None:
    problem = _voltage_driven(10, 16) if voltage else _board(10, 16, graded=True)
    config = MPIRConfig()
    native = solve_pcb_dc(problem, native=True, config=config)
    portable = solve_pcb_dc(problem, native=False, config=config)
    assert native.solve.converged and portable.solve.converged
    rtol = iterative_rtol(config.relative_tolerance)
    active = ~np.isnan(portable.potential_v)
    assert np.array_equal(active, ~np.isnan(native.potential_v))
    scale = np.abs(portable.potential_v[active]).max()
    np.testing.assert_allclose(native.potential_v[active], portable.potential_v[active], rtol=0, atol=rtol * scale)
    for name in ("current_density_a_per_m2", "element_joule_loss_w", "via_current_a", "via_joule_loss_w"):
        expected = getattr(portable, name)
        np.testing.assert_allclose(
            getattr(native, name), expected, rtol=0, atol=rtol * max(np.abs(expected).max(), 1e-300)
        )
    assert native.joule_loss_w == pytest.approx(portable.joule_loss_w, rel=rtol)
    assert native.max_current_density_a_per_m2 == pytest.approx(portable.max_current_density_a_per_m2, rel=rtol)
    if voltage:
        np.testing.assert_allclose(
            native.voltage_terminal_current_a, portable.voltage_terminal_current_a, rtol=rtol
        )


@pytest.mark.parametrize("voltage", [False, True])
def test_the_native_dc_solution_has_the_same_bits_at_every_budget(voltage: bool) -> None:
    problem = _voltage_driven(13, 21) if voltage else _board(13, 21, graded=True)
    fields = (
        "potential_v",
        "current_density_a_per_m2",
        "element_joule_loss_w",
        "via_current_a",
        "via_joule_loss_w",
        "voltage_terminal_current_a",
    )
    outputs = []
    for budget in (1, 2, 3, 8):
        with thread_budget_scope(budget):
            solution = solve_pcb_dc(problem, native=True)
        outputs.append(
            (
                tuple(np.asarray(getattr(solution, name)).tobytes() for name in fields),
                solution.joule_loss_w,
                solution.max_current_density_a_per_m2,
                solution.solve.history,
            )
        )
    assert all(output == outputs[0] for output in outputs[1:])


def test_the_native_system_rejects_what_the_numpy_operator_rejects() -> None:
    problem = _board(6, 8)
    mesh = problem.mesh
    inactive = tuple(int(i) for i in np.argwhere(~MatrixFreePCBOperator(mesh, reference_node=problem.reference_node, native=False).active_nodes)[0])
    with pytest.raises(ValueError, match="reference_node must lie on active copper"):
        MatrixFreePCBOperator(mesh, reference_node=inactive, native=True)
    with pytest.raises(ValueError, match="dirichlet_nodes must lie on active copper"):
        MatrixFreePCBOperator(mesh, dirichlet_nodes=(inactive,), native=True)
    with pytest.raises(ValueError, match="reference_node or dirichlet_nodes"):
        MatrixFreePCBOperator(mesh, native=True)
    native = MatrixFreePCBOperator(mesh, reference_node=problem.reference_node, vias=problem.vias, native=True)
    with pytest.raises(ValueError, match=r"terminal 'bad' contains inactive node"):
        native.build_rhs((CurrentTerminal((inactive,), 1.0, "bad"),))
    free = tuple(int(i) for i in np.argwhere(native.free_nodes)[0])
    with pytest.raises(ValueError, match=r"voltage terminal 'v' node .* is not a Dirichlet node"):
        native.dirichlet_potential((VoltageTerminal((free,), 1.0, "v"),))
