from __future__ import annotations

import numpy as np
import pytest

from electrical.matrix_free_mpir_fem import CupyFloat32Runtime, MPIRConfig
from thermal.matrix_free_mpir_fem import (
    ConvectionBoundary,
    HeatSource,
    LayeredThermalMesh,
    MatrixFreeThermalOperator,
    ThermalConductionProblem,
    solve_thermal_conduction,
)


def _cuda_runtime() -> CupyFloat32Runtime:
    pytest.importorskip("cupy")
    try:
        return CupyFloat32Runtime()
    except RuntimeError as exc:
        pytest.skip(str(exc))


def _problem() -> ThermalConductionProblem:
    rows = cols = 12
    conductivity = np.full((3, rows, cols), 0.8)
    conductivity[0] = conductivity[2] = 385.0
    conductivity[1, 4:8, 4:8] = 385.0
    through = conductivity.copy()
    through[1, :4] = 0.3
    mesh = LayeredThermalMesh(
        slab_thickness_m=(35.0e-6, 0.4e-3, 35.0e-6),
        pitch_x_m=0.25e-3,
        pitch_y_m=0.25e-3,
        conductivity_w_per_m_k=conductivity,
        through_plane_conductivity_w_per_m_k=through,
    )
    heat = np.zeros(mesh.element_grid_shape)
    heat[2, 5:7, 2:10] = 0.02
    mask = np.zeros(mesh.node_shape, dtype=bool)
    mask[0, :, 0] = True
    return ThermalConductionProblem(
        mesh,
        convection=(ConvectionBoundary("top", 10.0, 300.0),),
        fixed_temperature_mask=mask,
        fixed_temperature_k=300.0,
        heat_sources=(HeatSource(((3, 6, 6),), 0.05),),
        element_heat_w=heat,
    )


def test_cuda_low_action_matches_the_host_operator() -> None:
    runtime = _cuda_runtime()
    problem = _problem()
    cpu = MatrixFreeThermalOperator(problem)
    gpu = MatrixFreeThermalOperator(problem, runtime=runtime)
    vector = np.random.default_rng(3).standard_normal(cpu.size)

    assert gpu.low_operator_backend == "cuda-fused-node-gather-hex-q1"
    assert cpu.low_operator_backend == "array-corner-products"
    expected = cpu.apply_high(vector)
    actual = runtime.to_host(gpu.apply_low(runtime.from_host(vector)))
    scale = np.max(np.abs(expected))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-5 * scale)
    # The fused kernel and the generic CuPy corner-product path agree.
    generic = runtime.to_host(
        gpu._apply_impl(
            runtime.from_host(vector),
            runtime.namespace,
            gpu._in_plane_low,
            gpu._through_low,
            gpu._local_in_plane_low,
            gpu._local_through_low,
            gpu._robin_total_low,
            gpu._free_low,
        )
    )
    np.testing.assert_allclose(actual, generic, rtol=0.0, atol=2.0e-5 * scale)
    # Node-owned gather is deterministic.
    again = runtime.to_host(gpu.apply_low(runtime.from_host(vector)))
    assert np.array_equal(actual, again)
    np.testing.assert_allclose(
        runtime.to_host(gpu.diagonal_low()),
        runtime.to_host(cpu.diagonal_low()),
        rtol=1.0e-6,
    )


def test_cuda_backend_reaches_the_same_fp64_temperature_field() -> None:
    _cuda_runtime()
    problem = _problem()
    config = MPIRConfig(max_inner_iterations=1000)
    cpu = solve_thermal_conduction(problem, config=config, initial_temperature_k=300.0)
    gpu = solve_thermal_conduction(
        problem, config=config, backend="cuda", initial_temperature_k=300.0
    )

    assert cpu.solve.converged and gpu.solve.converged
    assert gpu.solve.low_runtime == "cupy-fp32"
    np.testing.assert_allclose(
        gpu.temperature_k, cpu.temperature_k, rtol=1.0e-8, atol=1.0e-7
    )
    assert gpu.heat_balance_error_w == pytest.approx(0.0, abs=1.0e-8)
