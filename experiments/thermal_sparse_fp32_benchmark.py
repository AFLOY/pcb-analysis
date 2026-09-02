"""Benchmark sparse FP32 alternatives for the thermal MPIR inner solve.

The production solver keeps host FP64 reliable residuals and a CUDA FP32
inner PCG.  This experiment compares three possible bandwidth reductions:

* an assembled FP32 CSR fine operator versus the fused matrix-free operator;
* an exact FP32 sparse-LU coarse solve versus the dense coarse inverse;
* an SPD, compactly supported sparse approximation of the coarse inverse.

The sparse coarse inverse is formed by a Schur product with a separable
Bartlett taper.  The taper is positive semidefinite, so the approximation and
the resulting two-level preconditioner retain the symmetry/positivity needed
by PCG.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from electrical.matrix_free_mpir_fem import MPIRConfig, solve_mpir
from thermal.matrix_free_mpir_fem import (
    ConvectionBoundary,
    HeatSource,
    LayeredThermalMesh,
    MatrixFreeThermalOperator,
    ThermalConductionProblem,
)


def _problem(side: int) -> ThermalConductionProblem:
    pitch_m = 50.0e-3 / side
    mesh = LayeredThermalMesh(
        slab_thickness_m=(35.0e-6, 0.7e-3, 0.7e-3, 35.0e-6),
        pitch_x_m=pitch_m,
        pitch_y_m=pitch_m,
        conductivity_w_per_m_k=(385.0, 0.8, 0.8, 385.0),
        through_plane_conductivity_w_per_m_k=(385.0, 0.3, 0.3, 385.0),
        element_shape=(side, side),
    )
    source_lo = int(round(0.45 * side))
    source_hi = int(round(0.55 * side))
    source_nodes = tuple(
        (4, row, column)
        for row in range(source_lo, source_hi + 1)
        for column in range(source_lo, source_hi + 1)
    )
    return ThermalConductionProblem(
        mesh=mesh,
        convection=(
            ConvectionBoundary("top", 10.0, 298.15),
            ConvectionBoundary("bottom", 10.0, 298.15),
        ),
        heat_sources=(HeatSource(source_nodes, 1.0),),
    )


def _synchronize(cp: Any) -> None:
    cp.cuda.Stream.null.synchronize()


def _timed_batch(
    function: Callable[[], Any],
    *,
    batch: int,
    samples: int,
    synchronize: Callable[[], None],
) -> dict[str, Any]:
    for _ in range(5):
        function()
    synchronize()
    timings: list[float] = []
    for _ in range(samples):
        synchronize()
        started = time.perf_counter_ns()
        for _ in range(batch):
            function()
        synchronize()
        timings.append((time.perf_counter_ns() - started) / (1.0e6 * batch))
    return {
        "median_ms": statistics.median(timings),
        "minimum_ms": min(timings),
        "maximum_ms": max(timings),
        "samples_ms": timings,
        "batch": batch,
    }


def _timed_solve(
    function: Callable[[], Any],
    *,
    repeats: int,
    synchronize: Callable[[], None],
) -> tuple[Any, dict[str, Any]]:
    function()
    synchronize()
    result: Any = None
    timings: list[float] = []
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter_ns()
        result = function()
        synchronize()
        timings.append((time.perf_counter_ns() - started) / 1.0e6)
    return result, {
        "median_ms": statistics.median(timings),
        "minimum_ms": min(timings),
        "maximum_ms": max(timings),
        "samples_ms": timings,
        "repeats": repeats,
    }


def _csr_bytes(matrix: sp.spmatrix) -> int:
    csr = matrix.tocsr()
    return int(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes)


def _assemble_fine_csr(operator: MatrixFreeThermalOperator) -> sp.csr_matrix:
    """Assemble the unconstrained benchmark fixture's Q1 matrix in FP32."""

    if np.any(operator.fixed_nodes):
        raise ValueError("the benchmark assembler expects convection-only fixtures")
    slabs, rows, columns = operator.mesh.element_grid_shape
    layer, row, column = np.indices((slabs, rows, columns))
    base = (
        layer * (rows + 1) * (columns + 1)
        + row * (columns + 1)
        + column
    ).reshape(-1)
    offsets = np.asarray(
        [
            dz * (rows + 1) * (columns + 1)
            + dy * (columns + 1)
            + dx
            for dz in (0, 1)
            for dy in (0, 1)
            for dx in (0, 1)
        ]
    )
    nodes = base[None, :] + offsets[:, None]
    in_plane = operator._in_plane_high.reshape(-1)
    through = operator._through_high.reshape(-1)
    values: list[np.ndarray] = []
    row_indices: list[np.ndarray] = []
    column_indices: list[np.ndarray] = []
    for local_row in range(8):
        for local_column in range(8):
            coefficient = (
                in_plane
                * operator._local_in_plane_high[:, local_row, local_column].repeat(
                    rows * columns
                )
                + through
                * operator._local_through_high[:, local_row, local_column].repeat(
                    rows * columns
                )
            )
            row_indices.append(nodes[local_row])
            column_indices.append(nodes[local_column])
            values.append(coefficient.astype(np.float32))
    matrix = sp.coo_matrix(
        (
            np.concatenate(values),
            (np.concatenate(row_indices), np.concatenate(column_indices)),
        ),
        shape=(operator.size, operator.size),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix += sp.diags(operator._robin_total_high.astype(np.float32))
    matrix.eliminate_zeros()
    return matrix


def _tapered_inverse(
    inverse: np.ndarray,
    coarse_shape: tuple[int, int, int],
    radius: int,
) -> sp.csr_matrix:
    coordinates = np.indices(coarse_shape).reshape(3, -1).T
    row_distance = np.abs(
        coordinates[:, None, 1] - coordinates[None, :, 1]
    )
    column_distance = np.abs(
        coordinates[:, None, 2] - coordinates[None, :, 2]
    )
    row_taper = np.maximum(1.0 - row_distance / (radius + 1.0), 0.0)
    column_taper = np.maximum(
        1.0 - column_distance / (radius + 1.0), 0.0
    )
    tapered = inverse * row_taper * column_taper
    tapered = 0.5 * (tapered + tapered.T)
    # This also catches an accidental change to a non-SPD taper.
    np.linalg.cholesky(tapered)
    result = sp.csr_matrix(tapered.astype(np.float32))
    result.eliminate_zeros()
    return result


def _sparse_preconditioner(operator: Any, coarse_gpu: Any) -> Callable[[Any], Any]:
    correction = operator.coarse_correction

    def apply(vector: Any) -> Any:
        xp = correction.runtime.namespace
        flat = vector.reshape(-1)
        smoothed = flat / correction._diagonal_low
        grid = xp.where(
            correction._free_low,
            flat.reshape(correction.node_shape),
            0,
        )
        restricted = correction._restrict(grid, xp).reshape(-1)
        coarse = coarse_gpu @ restricted
        prolonged = xp.where(
            correction._free_low,
            correction._prolong(coarse, xp),
            0,
        )
        return xp.asarray(
            smoothed + prolonged.reshape(-1), dtype=vector.dtype
        )

    return apply


def _case(
    side: int,
    *,
    taper_radius: int,
    solve_repeats: int,
    action_batch: int,
    device_id: int,
) -> dict[str, Any]:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from cupyx import cusparse

    problem = _problem(side)
    setup_started = time.perf_counter_ns()
    operator = MatrixFreeThermalOperator(
        problem, backend="cuda", device_id=device_id
    )
    _synchronize(cp)
    setup_ms = (time.perf_counter_ns() - setup_started) / 1.0e6
    correction = operator.coarse_correction
    assert correction is not None

    rng = np.random.default_rng(20260902 + side)
    fine_host = rng.standard_normal(operator.size).astype(np.float32)
    fine_gpu = cp.asarray(fine_host)
    matrix_free_action = operator.apply_low(fine_gpu)
    fine_csr_started = time.perf_counter_ns()
    fine_csr = _assemble_fine_csr(operator)
    fine_csr_gpu = csp.csr_matrix(fine_csr)
    _synchronize(cp)
    fine_csr_setup_ms = (time.perf_counter_ns() - fine_csr_started) / 1.0e6
    fine_csr_action = fine_csr_gpu @ fine_gpu
    high_action = operator.apply_high(fine_host)
    matrix_free_error = np.linalg.norm(cp.asnumpy(matrix_free_action) - high_action)
    matrix_free_error /= np.linalg.norm(high_action)
    csr_error = np.linalg.norm(cp.asnumpy(fine_csr_action) - high_action)
    csr_error /= np.linalg.norm(high_action)
    matrix_free_bytes = sum(
        int(array.nbytes)
        for array in (
            operator._in_plane_low,
            operator._through_low,
            operator._local_in_plane_low,
            operator._local_through_low,
            operator._robin_total_low,
            operator._free_low_u8,
        )
    )
    matrix_free_timing = _timed_batch(
        lambda: operator.apply_low(fine_gpu),
        batch=action_batch,
        samples=7,
        synchronize=lambda: _synchronize(cp),
    )
    csr_timing = _timed_batch(
        lambda: fine_csr_gpu @ fine_gpu,
        batch=action_batch,
        samples=7,
        synchronize=lambda: _synchronize(cp),
    )

    coarse_host = rng.standard_normal(correction.coarse_size).astype(np.float32)
    coarse_gpu = cp.asarray(coarse_host)
    dense_timing = _timed_batch(
        lambda: correction._coarse_inverse_low @ coarse_gpu,
        batch=action_batch,
        samples=7,
        synchronize=lambda: _synchronize(cp),
    )
    tapered = _tapered_inverse(
        correction._coarse_inverse_high,
        correction.coarse_shape,
        taper_radius,
    )
    tapered_gpu = csp.csr_matrix(tapered)
    tapered_timing = _timed_batch(
        lambda: tapered_gpu @ coarse_gpu,
        batch=action_batch,
        samples=7,
        synchronize=lambda: _synchronize(cp),
    )

    coarse_matrix = sp.csr_matrix(correction.coarse_matrix)
    coarse_matrix.eliminate_zeros()
    factor_started = time.perf_counter_ns()
    lu = spla.splu(
        coarse_matrix.tocsc(),
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True},
    )
    factor_ms = (time.perf_counter_ns() - factor_started) / 1.0e6
    lower = lu.L.astype(np.float32).tocsr()
    upper = lu.U.astype(np.float32).tocsr()
    lower_gpu = csp.csr_matrix(lower)
    upper_gpu = csp.csr_matrix(upper)
    row_permutation = cp.asarray(np.argsort(lu.perm_r))
    column_permutation = cp.asarray(lu.perm_c)

    def sparse_lu_apply() -> Any:
        intermediate = cusparse.spsm(
            lower_gpu,
            coarse_gpu[row_permutation],
            lower=True,
            unit_diag=True,
        )
        solved = cusparse.spsm(
            upper_gpu,
            intermediate,
            lower=False,
            unit_diag=False,
        )
        return solved[column_permutation]

    sparse_lu_timing = _timed_batch(
        sparse_lu_apply,
        batch=max(1, min(action_batch, 20)),
        samples=5,
        synchronize=lambda: _synchronize(cp),
    )
    sparse_lu_host = cp.asnumpy(sparse_lu_apply())
    sparse_lu_reference = lu.solve(coarse_host.astype(np.float64))
    sparse_lu_error = np.linalg.norm(sparse_lu_host - sparse_lu_reference)
    sparse_lu_error /= np.linalg.norm(sparse_lu_reference)

    rhs = operator.build_rhs(298.15)
    config = MPIRConfig(max_inner_iterations=400)
    original_preconditioner = operator.precondition_low
    dense_result, dense_solve_timing = _timed_solve(
        lambda: solve_mpir(operator, rhs, config=config),
        repeats=solve_repeats,
        synchronize=lambda: _synchronize(cp),
    )
    operator.precondition_low = _sparse_preconditioner(operator, tapered_gpu)
    sparse_result, sparse_solve_timing = _timed_solve(
        lambda: solve_mpir(operator, rhs, config=config),
        repeats=solve_repeats,
        synchronize=lambda: _synchronize(cp),
    )
    operator.precondition_low = original_preconditioner
    solution_delta = np.linalg.norm(
        sparse_result.solution - dense_result.solution
    ) / np.linalg.norm(dense_result.solution)

    return {
        "element_shape": [side, side],
        "fine_nodes": operator.size,
        "operator_setup_ms": setup_ms,
        "fine_operator": {
            "matrix_free": {
                "storage_bytes": matrix_free_bytes,
                "timing": matrix_free_timing,
                "relative_action_error_vs_fp64": float(matrix_free_error),
            },
            "csr": {
                "nnz": int(fine_csr.nnz),
                "storage_bytes": _csr_bytes(fine_csr),
                "setup_ms": fine_csr_setup_ms,
                "timing": csr_timing,
                "relative_action_error_vs_fp64": float(csr_error),
            },
            "csr_speedup_over_matrix_free": (
                matrix_free_timing["median_ms"] / csr_timing["median_ms"]
            ),
        },
        "coarse_operator": {
            "shape": list(correction.coarse_shape),
            "size": correction.coarse_size,
            "block_nodes": correction.block,
            "dense_inverse": {
                "storage_bytes": int(correction._coarse_inverse_low.nbytes),
                "timing": dense_timing,
            },
            "sparse_lu": {
                "matrix_nnz": int(coarse_matrix.nnz),
                "factor_nnz": int(lower.nnz + upper.nnz),
                "factor_storage_bytes": _csr_bytes(lower) + _csr_bytes(upper),
                "factorization_ms": factor_ms,
                "timing": sparse_lu_timing,
                "relative_action_error_vs_fp64_lu": float(sparse_lu_error),
            },
            "tapered_inverse": {
                "radius": taper_radius,
                "nnz": int(tapered.nnz),
                "density": float(tapered.nnz / tapered.shape[0] ** 2),
                "storage_bytes": _csr_bytes(tapered),
                "timing": tapered_timing,
            },
        },
        "solver": {
            "precision": {"outer": "float64", "inner": "float32"},
            "dense_coarse": {
                "timing": dense_solve_timing,
                "converged": dense_result.converged,
                "outer_iterations": dense_result.outer_iterations,
                "inner_iterations": dense_result.inner_iterations,
                "relative_residual": dense_result.relative_residual,
            },
            "tapered_sparse_coarse": {
                "timing": sparse_solve_timing,
                "converged": sparse_result.converged,
                "outer_iterations": sparse_result.outer_iterations,
                "inner_iterations": sparse_result.inner_iterations,
                "relative_residual": sparse_result.relative_residual,
            },
            "sparse_speedup_over_dense": (
                dense_solve_timing["median_ms"]
                / sparse_solve_timing["median_ms"]
            ),
            "relative_solution_delta": float(solution_delta),
        },
    }


def run_benchmark(
    *,
    sides: tuple[int, ...],
    taper_radius: int,
    solve_repeats: int,
    action_batch: int,
    device_id: int,
) -> dict[str, Any]:
    import cupy as cp

    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    cases = [
        _case(
            side,
            taper_radius=taper_radius,
            solve_repeats=solve_repeats,
            action_batch=action_batch,
            device_id=device_id,
        )
        for side in sides
    ]
    return {
        "schema": "thermal-sparse-fp32-benchmark/v1",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "cupy": cp.__version__,
            "device": device_name,
            "compute_capability": (
                f"{properties['major']}.{properties['minor']}"
            ),
            "global_memory_bytes": int(properties["totalGlobalMem"]),
        },
        "definitions": {
            "outer_precision": "host float64 reliable residual/refinement",
            "inner_precision": "CUDA float32 PCG",
            "timing": "CUDA-synchronized wall time; setup excluded from solves",
            "fixture": "50 mm four-slab PCB, 1 W centered source, top/bottom 10 W/m2/K convection",
            "taper": "separable in-plane Bartlett taper; all slab pairs retained",
        },
        "cases": cases,
        "decision": {
            "adopt_fine_csr": all(
                case["fine_operator"]["csr_speedup_over_matrix_free"] > 1.0
                for case in cases
            ),
            "adopt_sparse_coarse": all(
                case["solver"]["sparse_speedup_over_dense"] > 1.0
                for case in cases
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sides", type=int, nargs="+", default=(50, 100, 200))
    parser.add_argument("--taper-radius", type=int, default=4)
    parser.add_argument("--solve-repeats", type=int, default=3)
    parser.add_argument("--action-batch", type=int, default=200)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/thermal-sparse-fp32.json"),
    )
    args = parser.parse_args()
    if any(side < 1 for side in args.sides):
        parser.error("sides must be positive")
    if args.taper_radius < 0:
        parser.error("taper-radius must be non-negative")
    if args.solve_repeats < 1 or args.action_batch < 1:
        parser.error("repeat counts must be positive")
    result = run_benchmark(
        sides=tuple(args.sides),
        taper_radius=args.taper_radius,
        solve_repeats=args.solve_repeats,
        action_batch=args.action_batch,
        device_id=args.device_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
