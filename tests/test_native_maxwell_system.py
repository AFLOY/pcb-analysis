"""The C++ scalar Maxwell system against the NumPy operator it replaced."""

from __future__ import annotations

import numpy as np
import pytest

from electrical.matrix_free_mpir_fem import (
    MatrixFreeScalarMaxwellOperator,
    MPIRConfig,
    NumpyComplex64Runtime,
    solve_scalar_maxwell,
)
from electrical.matrix_free_mpir_fem.native_q1 import native_available
from electrical.threads import thread_budget_scope
from tests.test_native_q1 import _problem
from tests.tolerance import DIRECT_RTOL, iterative_rtol

pytestmark = pytest.mark.skipif(not native_available(), reason="pcbcore extension not built")

CONFIG = MPIRConfig(relative_tolerance=1.0e-10, max_outer_iterations=12, max_inner_iterations=600, gmres_restart=32)


def test_the_prepared_maxwell_system_is_the_numpy_operator() -> None:
    problem = _problem(9, 14, conductive=True)
    native = MatrixFreeScalarMaxwellOperator(problem, runtime=NumpyComplex64Runtime(), native=True)
    portable = MatrixFreeScalarMaxwellOperator(problem, runtime=NumpyComplex64Runtime(), native=False)
    np.testing.assert_allclose(native._system.diagonal, portable._diagonal_low.astype(np.complex128), rtol=1e-6)
    expected = portable.build_rhs()
    np.testing.assert_allclose(native.build_rhs(), expected, rtol=0, atol=DIRECT_RTOL * np.abs(expected).max())
    rng = np.random.default_rng(6)
    field = rng.standard_normal(native.size) + 1j * rng.standard_normal(native.size)
    expected = portable.apply_high(field)
    np.testing.assert_allclose(native.apply_high(field), expected, rtol=0, atol=DIRECT_RTOL * np.abs(expected).max())
    for a, b in zip(native.element_fields(field), portable.element_fields(field)):
        np.testing.assert_allclose(a, b, rtol=0, atol=DIRECT_RTOL * np.abs(b).max())
    for a, b in zip(native.losses(field), portable.losses(field)):
        assert a == pytest.approx(b, rel=DIRECT_RTOL)


def test_the_native_maxwell_solution_agrees_with_numpy_and_is_the_same_at_every_budget() -> None:
    problem = _problem(12, 30, conductive=True)
    portable = solve_scalar_maxwell(problem, config=CONFIG, native=False)
    rtol = iterative_rtol(CONFIG.relative_tolerance)
    outputs = []
    for budget in (1, 2, 3, 8):
        with thread_budget_scope(budget):
            native = solve_scalar_maxwell(problem, config=CONFIG)
        assert native.solve.converged
        outputs.append(
            (
                native.electric_field_z_v_per_m.tobytes(),
                native.magnetic_field_xy_a_per_m.tobytes(),
                native.conduction_loss_w_per_m,
                native.dielectric_loss_w_per_m,
                native.solve.history,
            )
        )
    assert all(output == outputs[0] for output in outputs[1:])
    scale = np.abs(portable.electric_field_z_v_per_m).max()
    np.testing.assert_allclose(
        native.electric_field_z_v_per_m, portable.electric_field_z_v_per_m, rtol=0, atol=rtol * scale
    )
    assert native.conduction_loss_w_per_m == pytest.approx(portable.conduction_loss_w_per_m, rel=rtol)
    assert native.dielectric_loss_w_per_m == pytest.approx(portable.dielectric_loss_w_per_m, rel=rtol)
