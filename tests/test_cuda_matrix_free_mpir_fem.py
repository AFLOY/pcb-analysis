from __future__ import annotations

import numpy as np
import pytest

from electrical.matrix_free_mpir_fem import (
    CupyComplex64Runtime,
    CurrentTerminal,
    LayeredPCBMesh,
    MPIRConfig,
    MatrixFreeScalarMaxwellOperator,
    NumpyComplex64Runtime,
    PCBConductionProblem,
    ScalarMaxwellMesh2D,
    ScalarMaxwellProblem,
    solve_pcb_dc,
    solve_scalar_maxwell,
)
from electrical.matrix_free_mpir_fem.cuda import scalar_maxwell_cuda_source


def _cuda_runtime() -> CupyComplex64Runtime:
    pytest.importorskip("cupy")
    try:
        return CupyComplex64Runtime()
    except RuntimeError as exc:
        pytest.skip(str(exc))


def _problem(elements: int = 24) -> ScalarMaxwellProblem:
    mesh = ScalarMaxwellMesh2D(
        (2, elements),
        15.0e-3 / elements,
        0.5e-3,
        relative_permittivity=np.vstack(
            (
                np.full(elements, 4.0),
                np.linspace(3.7, 4.3, elements),
            )
        ),
        conductivity_s_per_m=np.vstack(
            (
                np.zeros(elements),
                np.linspace(0.0, 2.0, elements),
            )
        ),
        dielectric_loss_tangent=0.015,
    )
    mask = np.zeros(mesh.node_shape, dtype=bool)
    mask[:, 0] = True
    mask[:, -1] = True
    values = np.zeros(mesh.node_shape, dtype=np.complex128)
    values[:, 0] = 1.0 + 0.2j
    return ScalarMaxwellProblem(mesh, 1.5e9, mask, values)


def test_cuda_source_uses_node_owned_gather_without_atomics() -> None:
    source = scalar_maxwell_cuda_source()
    assert "scalar_maxwell_q1_apply" in source
    assert "atomicAdd" not in source


def test_cuda_fused_action_matches_portable_complex64_action() -> None:
    problem = _problem()
    cpu = MatrixFreeScalarMaxwellOperator(
        problem, runtime=NumpyComplex64Runtime()
    )
    runtime = _cuda_runtime()
    cuda = MatrixFreeScalarMaxwellOperator(problem, runtime=runtime)
    rng = np.random.default_rng(20260902)
    vector = (
        rng.standard_normal(cuda.size) + 1j * rng.standard_normal(cuda.size)
    ).astype(np.complex64)

    expected = cpu.apply_low(vector)
    actual = runtime.to_host(cuda.apply_low(runtime.from_host(vector)))

    assert cuda.low_operator_backend == "cuda-fused-node-gather-q1"
    np.testing.assert_allclose(actual, expected, rtol=3.0e-6, atol=2.0e-2)


def test_cuda_mpir_solution_matches_cpu_reliable_solution() -> None:
    problem = _problem(elements=32)
    config = MPIRConfig(
        relative_tolerance=1.0e-10,
        inner_relative_tolerance=2.0e-3,
        max_outer_iterations=12,
        max_inner_iterations=300,
        gmres_restart=24,
    )
    cpu = solve_scalar_maxwell(problem, config=config, backend="cpu")
    _cuda_runtime()
    cuda = solve_scalar_maxwell(problem, config=config, backend="cuda")

    assert cpu.solve.converged
    assert cuda.solve.converged
    assert cuda.solve.low_runtime == "cupy-complex64"
    assert cuda.solve.relative_residual <= config.relative_tolerance
    np.testing.assert_allclose(
        cuda.electric_field_z_v_per_m,
        cpu.electric_field_z_v_per_m,
        rtol=2.0e-8,
        atol=2.0e-10,
    )


def test_cuda_dc_pcg_matches_cpu_solution() -> None:
    _cuda_runtime()
    columns = 16
    mesh = LayeredPCBMesh(
        element_active=np.ones((1, 1, columns), dtype=bool),
        layer_thickness_m=(35.0e-6,),
        pitch_x_m=0.2e-3,
        pitch_y_m=0.2e-3,
    )
    problem = PCBConductionProblem(
        mesh=mesh,
        terminals=(
            CurrentTerminal(((0, 0, 0), (0, 1, 0)), 1.0, "source"),
            CurrentTerminal(
                ((0, 0, columns), (0, 1, columns)), -1.0, "sink"
            ),
        ),
        reference_node=(0, 0, columns),
    )
    cpu = solve_pcb_dc(problem, backend="cpu")
    cuda = solve_pcb_dc(problem, backend="cuda")

    assert cpu.solve.converged
    assert cuda.solve.converged
    assert cuda.solve.low_runtime == "cupy-fp32"
    np.testing.assert_allclose(
        cuda.potential_v,
        cpu.potential_v,
        rtol=2.0e-8,
        atol=2.0e-10,
    )
