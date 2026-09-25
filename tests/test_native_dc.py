"""Fused C++ path for the layered-PCB DC conduction operator (opt-in)."""

from __future__ import annotations

import numpy as np
import pytest

from electrical.matrix_free_mpir_fem import (
    CurrentTerminal,
    LayeredPCBMesh,
    MatrixFreePCBOperator,
    MPIRConfig,
    PCBConductionProblem,
    ViaConnection,
    solve_mpir,
    solve_pcb_dc,
)
from electrical.matrix_free_mpir_fem.native_dc import native_available, via_adjacency

pytestmark = pytest.mark.skipif(
    not native_available(),
    reason="layered DC native extension not built; run python -m electrical.matrix_free_mpir_fem.native.build",
)


def _board(rows: int, cols: int, *, layers: int = 2, vias: bool = True, graded: bool = False) -> PCBConductionProblem:
    """Two copper layers joined by vias, with holes in the copper and a graded grid option."""

    active = np.ones((layers, rows, cols), dtype=bool)
    if rows >= 3 and cols >= 4:
        active[0, rows // 3, cols // 4 : cols // 2] = False  # a slot on the top layer
    if layers > 1 and rows >= 2:
        active[1, : max(1, rows // 4), : max(1, cols // 3)] = False  # a missing corner below
    pitch_x = 0.4e-3 * (1.0 + 0.5 * np.linspace(0.0, 1.0, cols)) if graded else 0.4e-3
    pitch_y = 0.4e-3 * (1.0 + 0.3 * np.linspace(0.0, 1.0, rows)) if graded else 0.4e-3
    mesh = LayeredPCBMesh(
        element_active=active,
        layer_thickness_m=tuple(35.0e-6 for _ in range(layers)),
        pitch_x_m=pitch_x,
        pitch_y_m=pitch_y,
        conductivity_s_per_m=tuple(5.8e7 / (1.0 + 0.1 * index) for index in range(layers)),
    )
    source = tuple((0, r, 0) for r in range(rows + 1) if active[0, min(r, rows - 1), 0] or active[0, max(r - 1, 0), 0])
    sink_layer = layers - 1
    sink = tuple((sink_layer, r, cols) for r in range(rows + 1))
    links = ()
    if vias and layers > 1:
        links = tuple(
            ViaConnection((0, r, c), (sink_layer, r, c), 1.5e-3 * (1 + (r + c) % 3))
            for r in range(1, rows, max(1, rows // 3))
            for c in range(max(1, cols // 2), cols, max(1, cols // 4))
        )
    return PCBConductionProblem(
        mesh=mesh,
        terminals=(CurrentTerminal(source, 2.0, "source"), CurrentTerminal(sink, -2.0, "sink")),
        reference_node=sink[0],
        vias=links,
    )


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (3, 2), (7, 11), (12, 20)])
@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("threads", [1, 3])
def test_native_apply_matches_portable_float32(shape, graded, threads) -> None:
    problem = _board(*shape, graded=graded)
    portable = MatrixFreePCBOperator(problem.mesh, reference_node=problem.reference_node, vias=problem.vias, preconditioner="jacobi")
    native = MatrixFreePCBOperator(
        problem.mesh, reference_node=problem.reference_node, vias=problem.vias, preconditioner="jacobi",
        native=True, native_threads=threads,
    )
    assert portable.low_operator_backend == "array-element-loops"
    assert native.low_operator_backend == "cpp-fused-node-gather-layered-dc-q1"
    rng = np.random.default_rng(3)
    vector = rng.standard_normal(portable.size).astype(np.float32)
    expected = portable.apply_low(vector)
    actual = native.apply_low(vector)
    assert actual.dtype == np.float32
    assert np.linalg.norm(actual - expected) <= 8.0 * np.finfo(np.float32).eps * np.linalg.norm(expected)
    fixed = ~portable.free_nodes.reshape(-1)
    np.testing.assert_array_equal(actual[fixed], vector[fixed])


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (3, 2), (7, 11), (12, 20)])
@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("threads", [1, 3])
def test_native_apply_high_matches_portable_float64(shape, graded, threads) -> None:
    problem = _board(*shape, graded=graded)
    portable = MatrixFreePCBOperator(problem.mesh, reference_node=problem.reference_node, vias=problem.vias, preconditioner="jacobi")
    native = MatrixFreePCBOperator(
        problem.mesh, reference_node=problem.reference_node, vias=problem.vias, preconditioner="jacobi",
        native=True, native_threads=threads,
    )
    assert portable.high_operator_backend == "array-element-loops-fp64"
    assert native.high_operator_backend == "cpp-fused-node-gather-layered-dc-q1-fp64"
    rng = np.random.default_rng(5)
    vector = rng.standard_normal(portable.size)
    expected = portable.apply_high(vector)
    actual = native.apply_high(vector)
    assert actual.dtype == np.float64
    assert np.linalg.norm(actual - expected) <= 64.0 * np.finfo(np.float64).eps * np.linalg.norm(expected)
    fixed = ~portable.free_nodes.reshape(-1)
    np.testing.assert_array_equal(actual[fixed], vector[fixed])


def test_native_coarse_space_matches_portable() -> None:
    problem = _board(9, 14)
    portable = MatrixFreePCBOperator(problem.mesh, reference_node=problem.reference_node, vias=problem.vias)
    native = MatrixFreePCBOperator(problem.mesh, reference_node=problem.reference_node, vias=problem.vias, native=True, native_threads=2)
    np.testing.assert_allclose(
        native.coarse_correction._coarse_inverse_high,
        portable.coarse_correction._coarse_inverse_high,
        rtol=1e-9,
        atol=1e-12 * np.abs(portable.coarse_correction._coarse_inverse_high).max(),
    )


def test_via_adjacency_lists_every_link_under_both_ends() -> None:
    pointer, neighbour, conductance = via_adjacency([0, 2, 2], [5, 6, 7], [1.0, 2.0, 3.0], 8, np.float32)
    assert pointer.tolist() == [0, 1, 1, 3, 3, 3, 4, 5, 6]
    assert neighbour.tolist() == [5, 6, 7, 0, 2, 2]
    assert conductance.tolist() == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    empty = via_adjacency([], [], [], 4, np.float64)
    assert empty[0].tolist() == [0, 0, 0, 0, 0] and empty[1].size == 0 and empty[2].size == 0


@pytest.mark.parametrize("preconditioner", ["two-level", "jacobi"])
@pytest.mark.parametrize("threads", [1, 4])
def test_native_inner_pcg_reaches_the_same_fp64_solution(preconditioner, threads) -> None:
    problem = _board(16, 24, graded=True)
    config = MPIRConfig(max_outer_iterations=16, max_inner_iterations=3000)
    portable = MatrixFreePCBOperator(problem.mesh, reference_node=problem.reference_node, vias=problem.vias, preconditioner=preconditioner)
    native = MatrixFreePCBOperator(
        problem.mesh, reference_node=problem.reference_node, vias=problem.vias, preconditioner=preconditioner,
        native=True, native_threads=threads,
    )
    rhs = portable.build_rhs(problem.terminals)
    reference = solve_mpir(portable, rhs, config=config)
    result = solve_mpir(native, rhs, config=config)
    assert reference.converged and result.converged
    assert result.low_operator_applications > 5 * result.high_operator_applications
    residual = rhs - native.apply_high(result.solution)
    assert np.linalg.norm(residual) <= config.relative_tolerance * np.linalg.norm(rhs)
    assert np.linalg.norm(result.solution - reference.solution) <= 1.0e-8 * np.linalg.norm(reference.solution)


def test_native_solve_pcb_dc_matches_the_portable_currents_and_losses() -> None:
    problem = _board(10, 16)
    portable = solve_pcb_dc(problem)
    native = solve_pcb_dc(problem, native=True, native_threads=2)
    assert native.solve.converged
    np.testing.assert_allclose(native.via_current_a, portable.via_current_a, rtol=1e-8, atol=1e-12)
    assert native.joule_loss_w == pytest.approx(portable.joule_loss_w, rel=1e-8)
    np.testing.assert_allclose(
        np.nan_to_num(native.potential_v), np.nan_to_num(portable.potential_v), rtol=1e-8, atol=1e-12
    )
    assert abs(float(np.sum(native.via_current_a))) == pytest.approx(2.0, rel=1e-8)


def test_native_rejects_wrong_sizes_and_cuda_runtime() -> None:
    from electrical.matrix_free_mpir_fem import NumpyFloat32Runtime

    problem = _board(2, 3)
    native = MatrixFreePCBOperator(problem.mesh, reference_node=problem.reference_node, vias=problem.vias, native=True)
    with pytest.raises(ValueError, match="size"):
        native.apply_low(np.zeros(native.size + 1, dtype=np.float32))
    with pytest.raises(ValueError, match="size"):
        native.apply_high(np.zeros(native.size + 1))
    with pytest.raises(ValueError, match="size"):
        native.native_inner_pcg(np.zeros(native.size - 1), MPIRConfig())

    class FakeCudaRuntime(NumpyFloat32Runtime):
        is_cuda = True

    with pytest.raises(ValueError, match="CPU runtime"):
        MatrixFreePCBOperator(problem.mesh, reference_node=problem.reference_node, runtime=FakeCudaRuntime(), native=True)
