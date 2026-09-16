"""Fused C++ low path for the scalar Maxwell operator (opt-in, exp branch)."""

from __future__ import annotations

import numpy as np
import pytest

from electrical.matrix_free_mpir_fem import (
    MPIRConfig,
    MatrixFreeScalarMaxwellOperator,
    NumpyComplex64Runtime,
    ScalarMaxwellMesh2D,
    ScalarMaxwellProblem,
    solve_mpir,
)
from electrical.matrix_free_mpir_fem.native_q1 import native_available

pytestmark = pytest.mark.skipif(
    not native_available(),
    reason="native extension not built; run python -m electrical.matrix_free_mpir_fem.native.build",
)


def _problem(rows: int, columns: int, *, conductive: bool) -> ScalarMaxwellProblem:
    conductivity = np.zeros((rows, columns))
    if conductive:
        conductivity[rows // 3 : 2 * rows // 3, columns // 4 : columns // 2] = 5.8e7
    mesh = ScalarMaxwellMesh2D(
        (rows, columns),
        20.0e-3 / columns,
        8.0e-3 / rows,
        relative_permittivity=4.0,
        dielectric_loss_tangent=0.01,
        conductivity_s_per_m=conductivity,
    )
    mask = np.zeros(mesh.node_shape, dtype=bool)
    mask[:, 0] = True
    mask[:, -1] = True
    mask[0, ::3] = True  # irregular Dirichlet pattern exercises the mask path
    values = np.zeros(mesh.node_shape, dtype=np.complex128)
    values[:, 0] = 1.0
    return ScalarMaxwellProblem(mesh, 1.0e9, mask, values)


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (7, 3), (16, 40)])
@pytest.mark.parametrize("conductive", [False, True])
def test_native_apply_matches_portable_complex64(shape, conductive) -> None:
    problem = _problem(*shape, conductive=conductive)
    portable = MatrixFreeScalarMaxwellOperator(problem, runtime=NumpyComplex64Runtime())
    native = MatrixFreeScalarMaxwellOperator(
        problem, runtime=NumpyComplex64Runtime(), native=True
    )
    assert portable.low_operator_backend == "portable-array-q1"
    assert native.low_operator_backend == "cpp-fused-node-gather-q1"

    rng = np.random.default_rng(7)
    vector = (
        rng.standard_normal(portable.size) + 1j * rng.standard_normal(portable.size)
    ).astype(np.complex64)
    expected = portable.apply_low(vector)
    actual = native.apply_low(vector)
    assert actual.dtype == np.complex64
    scale = np.linalg.norm(expected)
    assert np.linalg.norm(actual - expected) <= 4.0 * np.finfo(np.float32).eps * scale
    # Dirichlet rows copy the input in both paths.
    dirichlet = problem.dirichlet_mask.reshape(-1)
    np.testing.assert_array_equal(actual[dirichlet], vector[dirichlet])


@pytest.mark.parametrize("orthogonalization", ["mgs", "cgs2"])
def test_native_inner_gmres_reaches_the_same_fp64_solution(orthogonalization) -> None:
    problem = _problem(12, 48, conductive=True)
    portable = MatrixFreeScalarMaxwellOperator(problem, runtime=NumpyComplex64Runtime())
    native = MatrixFreeScalarMaxwellOperator(
        problem,
        runtime=NumpyComplex64Runtime(),
        native=True,
        native_orthogonalization=orthogonalization,
    )
    rhs = portable.build_rhs()
    config = MPIRConfig(
        relative_tolerance=1.0e-10,
        inner_relative_tolerance=2.0e-3,
        max_outer_iterations=12,
        max_inner_iterations=300,
        gmres_restart=32,
    )
    reference = solve_mpir(portable, rhs, config=config)
    result = solve_mpir(native, rhs, config=config)

    assert reference.converged and result.converged
    assert result.relative_residual <= config.relative_tolerance
    assert result.low_operator_applications > 10 * result.high_operator_applications
    assert result.inner_iterations > 0
    residual = rhs - native.apply_high(result.solution)
    assert np.linalg.norm(residual) <= config.relative_tolerance * np.linalg.norm(rhs)
    difference = np.linalg.norm(result.solution - reference.solution)
    assert difference <= 1.0e-7 * np.linalg.norm(reference.solution)


def test_native_rejects_wrong_sizes_and_cuda_runtime() -> None:
    problem = _problem(3, 4, conductive=False)
    native = MatrixFreeScalarMaxwellOperator(
        problem, runtime=NumpyComplex64Runtime(), native=True
    )
    with pytest.raises(ValueError, match="size"):
        native.apply_low(np.zeros(native.size + 1, dtype=np.complex64))
    with pytest.raises(ValueError, match="size"):
        native.native_inner_gmres(
            np.zeros(native.size - 1, dtype=np.complex128), MPIRConfig()
        )

    class FakeCudaRuntime(NumpyComplex64Runtime):
        is_cuda = True

    with pytest.raises(ValueError, match="CPU runtime"):
        MatrixFreeScalarMaxwellOperator(problem, runtime=FakeCudaRuntime(), native=True)
