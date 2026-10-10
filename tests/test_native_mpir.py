"""The end-to-end native MPIR solves and coarse assembly against the NumPy path.

The native path runs the whole mixed-precision refinement in C++: the FP64
outer residual, the FP32 inner PCG and the two-level coarse space assembled
from FP64 applications.  Its answers agree with the NumPy path to the solver
tolerance, its coarse matrix is the same Galerkin product, it records the same
per-step history, and it gives the same bits at every thread budget.
"""

from __future__ import annotations

import numpy as np
import pytest

from electrical.matrix_free_mpir_fem import MatrixFreePCBOperator, MPIRConfig, solve_mpir
from electrical.matrix_free_mpir_fem.native_dc import native_available as dc_available
from electrical.threads import thread_budget_scope
from tests.test_native_dc import _board
from tests.test_native_thermal import _stack
from tests.tolerance import iterative_rtol
from thermal.matrix_free_mpir_fem import MatrixFreeThermalOperator
from thermal.matrix_free_mpir_fem.native_hex import native_available as thermal_available

CONFIG = MPIRConfig(relative_tolerance=1.0e-10)


def _dc_operator(native: bool) -> tuple[MatrixFreePCBOperator, np.ndarray]:
    problem = _board(12, 20, graded=True)
    operator = MatrixFreePCBOperator(
        problem.mesh, reference_node=problem.reference_node, vias=problem.vias, native=native
    )
    return operator, operator.build_rhs(problem.terminals)


def _thermal_operator(native: bool) -> tuple[MatrixFreeThermalOperator, np.ndarray]:
    problem = _stack(12, 20, fixed=True)
    operator = MatrixFreeThermalOperator(problem, native=native)
    return operator, operator.build_rhs()


SYSTEMS = [
    pytest.param(_dc_operator, marks=pytest.mark.skipif(not dc_available(), reason="DC native not built"), id="dc"),
    pytest.param(
        _thermal_operator,
        marks=pytest.mark.skipif(not thermal_available(), reason="thermal native not built"),
        id="thermal",
    ),
]


@pytest.mark.parametrize("build", SYSTEMS)
def test_the_native_coarse_matrix_is_the_numpy_galerkin_product(build) -> None:
    portable, _ = build(False)
    native, _ = build(True)
    assert native.coarse_correction is not None and portable.coarse_correction is not None
    np.testing.assert_allclose(
        native.coarse_correction.coarse_matrix,
        portable.coarse_correction.coarse_matrix,
        rtol=1e-12,
        atol=1e-15 * float(np.max(np.abs(portable.coarse_correction.coarse_matrix))),
    )
    np.testing.assert_allclose(
        native.coarse_correction._coarse_inverse_high,
        portable.coarse_correction._coarse_inverse_high,
        rtol=1e-9,
        atol=1e-12 * float(np.max(np.abs(portable.coarse_correction._coarse_inverse_high))),
    )


@pytest.mark.parametrize("build", SYSTEMS)
def test_the_native_solve_agrees_with_numpy_and_records_its_steps(build) -> None:
    portable_system, rhs = build(False)
    native_system, _ = build(True)
    portable = solve_mpir(portable_system, rhs, config=CONFIG)
    native = solve_mpir(native_system, rhs, config=CONFIG)

    assert native.converged and portable.converged
    scale = float(np.max(np.abs(portable.solution)))
    np.testing.assert_allclose(
        native.solution, portable.solution, rtol=0, atol=iterative_rtol(CONFIG.relative_tolerance) * scale
    )
    assert native.relative_residual <= CONFIG.relative_tolerance
    # One record per correction, numbered from 1, starting from the residual of
    # the initial guess and falling.
    assert len(native.history) == native.outer_iterations
    assert [step.outer_iteration for step in native.history] == list(range(1, native.outer_iterations + 1))
    assert native.history[0].high_relative_residual == pytest.approx(1.0)
    residuals = [step.high_relative_residual for step in native.history]
    assert all(later < earlier for earlier, later in zip(residuals, residuals[1:]))
    assert sum(step.inner_iterations for step in native.history) == native.inner_iterations
    assert abs(native.outer_iterations - portable.outer_iterations) <= 1


@pytest.mark.parametrize("build", SYSTEMS)
def test_the_native_solve_gives_the_same_bits_at_every_thread_budget(build) -> None:
    solutions = []
    for threads in (1, 2, 3, 8):
        with thread_budget_scope(threads):
            system, rhs = build(True)
            solutions.append(solve_mpir(system, rhs, config=CONFIG).solution.tobytes())
    assert all(solution == solutions[0] for solution in solutions)


def _scramble_the_heap() -> list:
    # Allocations of odd sizes move where the next vectors land, and with them
    # the alignment a vectorised reduction would split its terms by.
    rng = np.random.default_rng()
    return [np.empty(int(rng.integers(1, 5000))) for _ in range(7)]


@pytest.mark.parametrize("build", SYSTEMS)
@pytest.mark.parametrize("shape", [(7, 11), (19, 13)])
def test_the_native_solve_does_not_depend_on_where_its_vectors_land(build, shape) -> None:
    import tests.test_native_dc as dc_fixtures
    import tests.test_native_thermal as thermal_fixtures

    def solve(threads: int) -> bytes:
        with thread_budget_scope(threads):
            if build is _dc_operator:
                problem = dc_fixtures._board(*shape, graded=True)
                system = MatrixFreePCBOperator(
                    problem.mesh, reference_node=problem.reference_node, vias=problem.vias, native=True
                )
                rhs = system.build_rhs(problem.terminals)
            else:
                system = MatrixFreeThermalOperator(thermal_fixtures._stack(*shape, fixed=True), native=True)
                rhs = system.build_rhs()
            heap = _scramble_the_heap()
            solution = solve_mpir(system, rhs, config=CONFIG).solution.tobytes()
            del heap
            return solution

    assert len({solve(threads) for threads in (1, 1, 1, 2, 3, 8, 8)}) == 1


@pytest.mark.skipif(not dc_available(), reason="Maxwell native not built")
@pytest.mark.parametrize("orthogonalization, dot_accumulation", [("mgs", "float64"), ("cgs2", "float64"), ("mgs", "float32")])
def test_the_native_maxwell_solve_does_not_depend_on_the_team_or_the_heap(orthogonalization, dot_accumulation) -> None:
    from electrical.matrix_free_mpir_fem.frequency_domain import MatrixFreeScalarMaxwellOperator
    from electrical.matrix_free_mpir_fem.runtime import NumpyComplex64Runtime
    from tests.test_native_q1 import _problem

    config = MPIRConfig(
        relative_tolerance=1e-10, inner_relative_tolerance=2e-3, max_outer_iterations=12,
        max_inner_iterations=300, gmres_restart=32,
    )

    def solve(threads: int) -> bytes:
        with thread_budget_scope(threads):
            system = MatrixFreeScalarMaxwellOperator(
                _problem(12, 48, conductive=True), runtime=NumpyComplex64Runtime(), native=True,
                native_orthogonalization=orthogonalization, native_dot_accumulation=dot_accumulation,
            )
            heap = _scramble_the_heap()
            solution = solve_mpir(system, system.build_rhs(), config=config).solution.tobytes()
            del heap
            return solution

    assert len({solve(threads) for threads in (1, 1, 2, 3, 8, 8)}) == 1
