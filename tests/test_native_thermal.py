"""Fused C++ low path for the hexahedral Q1 conduction operator (opt-in)."""

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
from thermal.matrix_free_mpir_fem.native_hex import native_available

pytestmark = pytest.mark.skipif(
    not native_available(),
    reason="thermal native extension not built; run python -m thermal.matrix_free_mpir_fem.native.build",
)


def _stack(rows: int, cols: int, *, fixed: bool = False) -> ThermalConductionProblem:
    mesh = LayeredThermalMesh(
        slab_thickness_m=(35.0e-6, 1.5e-3, 35.0e-6),
        pitch_x_m=0.2e-3,
        pitch_y_m=0.2e-3,
        conductivity_w_per_m_k=(385.0, 0.8, 385.0),
        through_plane_conductivity_w_per_m_k=(385.0, 0.3, 385.0),
        element_shape=(rows, cols),
    )
    heat = np.zeros(mesh.element_grid_shape)
    heat[2, rows // 2, cols // 4 : 3 * cols // 4] = 0.5 / max(1, cols // 2)
    fixed_mask = np.zeros(mesh.node_shape, dtype=bool)
    fixed_k = np.full(mesh.node_shape, 298.15)
    if fixed:
        fixed_mask[0, ::3, 1::2] = True
        fixed_k[0] = 310.0
    return ThermalConductionProblem(
        mesh,
        convection=(
            ConvectionBoundary("top", 10.0, 298.15),
            ConvectionBoundary("bottom", 10.0, 298.15),
        ),
        element_heat_w=heat,
        fixed_temperature_mask=fixed_mask if fixed else None,
        fixed_temperature_k=fixed_k if fixed else None,
        heat_sources=(HeatSource(((1, rows // 3, cols // 3),), 0.05),),
    )


@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (3, 2), (12, 20)])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("threads", [1, 3])
def test_native_apply_matches_portable_float32(shape, fixed, threads) -> None:
    problem = _stack(*shape, fixed=fixed)
    portable = MatrixFreeThermalOperator(problem, preconditioner="jacobi")
    native = MatrixFreeThermalOperator(
        problem, preconditioner="jacobi", native=True, native_threads=threads
    )
    assert portable.low_operator_backend == "array-corner-products"
    assert native.low_operator_backend == "cpp-fused-node-gather-hex-q1"
    rng = np.random.default_rng(3)
    vector = rng.standard_normal(portable.size).astype(np.float32)
    expected = portable.apply_low(vector)
    actual = native.apply_low(vector)
    assert actual.dtype == np.float32
    assert np.linalg.norm(actual - expected) <= 8.0 * np.finfo(np.float32).eps * np.linalg.norm(expected)
    fixed_nodes = problem.fixed_temperature_mask.reshape(-1)
    np.testing.assert_array_equal(actual[fixed_nodes], vector[fixed_nodes])


@pytest.mark.parametrize("preconditioner", ["two-level", "jacobi"])
@pytest.mark.parametrize("threads", [1, 4])
def test_native_inner_pcg_reaches_the_same_fp64_solution(preconditioner, threads) -> None:
    problem = _stack(16, 24, fixed=True)
    config = MPIRConfig(max_outer_iterations=16, max_inner_iterations=3000)
    portable = MatrixFreeThermalOperator(problem, preconditioner=preconditioner)
    native = MatrixFreeThermalOperator(
        problem, preconditioner=preconditioner, native=True, native_threads=threads
    )
    rhs = portable.build_rhs(portable.default_reference_temperature())
    reference = solve_mpir(portable, rhs, config=config)
    result = solve_mpir(native, rhs, config=config)
    assert reference.converged and result.converged
    assert result.low_operator_applications > 5 * result.high_operator_applications
    residual = rhs - native.apply_high(result.solution)
    assert np.linalg.norm(residual) <= config.relative_tolerance * np.linalg.norm(rhs)
    assert np.linalg.norm(result.solution - reference.solution) <= 1.0e-8 * np.linalg.norm(reference.solution)


def test_native_solution_and_heat_budget_match_the_front_end() -> None:
    problem = _stack(20, 20)
    portable = solve_thermal_conduction(problem, initial_temperature_k=298.15)
    native = solve_thermal_conduction(problem, initial_temperature_k=298.15, native=True, native_threads=2)
    assert native.solve.converged
    np.testing.assert_allclose(native.temperature_k, portable.temperature_k, rtol=1e-9, atol=1e-7)
    assert native.heat_balance_error_w == pytest.approx(0.0, abs=1e-9)


def test_native_rejects_wrong_sizes_and_cuda_runtime() -> None:
    from electrical.matrix_free_mpir_fem import NumpyFloat32Runtime

    problem = _stack(2, 3)
    native = MatrixFreeThermalOperator(problem, native=True)
    with pytest.raises(ValueError, match="size"):
        native.apply_low(np.zeros(native.size + 1, dtype=np.float32))
    with pytest.raises(ValueError, match="size"):
        native.native_inner_pcg(np.zeros(native.size - 1), MPIRConfig())

    class FakeCudaRuntime(NumpyFloat32Runtime):
        is_cuda = True

    with pytest.raises(ValueError, match="CPU runtime"):
        MatrixFreeThermalOperator(problem, runtime=FakeCudaRuntime(), native=True)
