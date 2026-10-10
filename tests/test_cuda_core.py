"""The CUDA inner solves of electrical._pcbcore_cuda against the CPU core.

Skipped unless the module is built (-DPCB_NATIVE_CUDA=ON) and a device is
present.  The device solve must converge where the CPU one does, give the
same solution to the solver tolerance, and give the same bits on every run.
"""

from __future__ import annotations

import numpy as np
import pytest

from electrical import backend
from electrical.matrix_free_mpir_fem import MatrixFreePCBOperator, MPIRConfig, solve_mpir
from electrical.matrix_free_mpir_fem.solver import solve_mpir_device
from tests.test_native_dc import _board
from tests.test_native_thermal_system import _body
from tests.tolerance import iterative_rtol
from thermal.matrix_free_mpir_fem import MatrixFreeThermalOperator

pytestmark = pytest.mark.skipif(backend.cuda_core() is None, reason="electrical._pcbcore_cuda or a CUDA device missing")

CONFIG = MPIRConfig(max_outer_iterations=16)


def _compare(operator, rhs) -> None:
    device = operator.device_system()
    expected = solve_mpir(operator, rhs, config=CONFIG)
    first = solve_mpir_device(device, rhs, config=CONFIG)
    second = solve_mpir_device(device, rhs, config=CONFIG)
    assert expected.converged and first.converged
    assert first.solution.tobytes() == second.solution.tobytes()
    assert first.history == second.history
    scale = np.abs(expected.solution).max()
    np.testing.assert_allclose(
        first.solution, expected.solution, rtol=0, atol=iterative_rtol(CONFIG.relative_tolerance) * scale
    )


@pytest.mark.parametrize("preconditioner", ["two-level", "jacobi"])
def test_the_layered_dc_inner_solve_on_the_device(preconditioner: str) -> None:
    problem = _board(16, 24, graded=True)
    operator = MatrixFreePCBOperator(
        problem.mesh, reference_node=problem.reference_node, vias=problem.vias, preconditioner=preconditioner
    )
    _compare(operator, operator.build_rhs(problem.terminals))


def test_the_thermal_inner_solve_on_the_device() -> None:
    problem = _body()
    operator = MatrixFreeThermalOperator(problem)
    _compare(operator, operator.build_rhs(operator.default_reference_temperature()))


def test_the_cuda_module_reports_its_devices() -> None:
    assert backend.cuda_core().device_count() >= 1
