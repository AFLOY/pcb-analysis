"""GTX/CUDA audit for the fused matrix-free scalar-Maxwell backend.

This benchmark separates low-precision operator throughput from the complete
mixed-precision solve.  The latter still includes host complex128 reliable
residuals by design.  CUDA timings always synchronize the active stream.
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

from electrical.matrix_free_mpir_fem import (
    CupyComplex64Runtime,
    MPIRConfig,
    MatrixFreeScalarMaxwellOperator,
    NumpyComplex64Runtime,
    ScalarMaxwellMesh2D,
    ScalarMaxwellProblem,
    solve_mpir,
)


def _problem(rows: int, columns: int) -> ScalarMaxwellProblem:
    length_x_m = 20.0e-3
    length_y_m = 8.0e-3
    mesh = ScalarMaxwellMesh2D(
        (rows, columns),
        length_x_m / columns,
        length_y_m / rows,
        relative_permittivity=4.0,
        dielectric_loss_tangent=0.01,
    )
    mask = np.zeros(mesh.node_shape, dtype=bool)
    mask[:, 0] = True
    mask[:, -1] = True
    values = np.zeros(mesh.node_shape, dtype=np.complex128)
    values[:, 0] = 1.0
    return ScalarMaxwellProblem(mesh, 1.0e9, mask, values)


def _timed(
    function: Callable[[], Any],
    *,
    repeats: int,
    warmups: int,
    synchronize: Callable[[], None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    result: Any = None
    for _ in range(warmups):
        result = function()
        if synchronize is not None:
            synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        if synchronize is not None:
            synchronize()
        started = time.perf_counter_ns()
        result = function()
        if synchronize is not None:
            synchronize()
        samples.append((time.perf_counter_ns() - started) / 1.0e6)
    return result, {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "samples_ms": samples,
        "repeats": repeats,
    }


def _operator_case(
    side: int,
    repeats: int,
    cuda_runtime: CupyComplex64Runtime,
) -> dict[str, Any]:
    problem = _problem(side, side)
    cpu = MatrixFreeScalarMaxwellOperator(
        problem, runtime=NumpyComplex64Runtime()
    )
    cuda = MatrixFreeScalarMaxwellOperator(problem, runtime=cuda_runtime)
    rng = np.random.default_rng(20260902 + side)
    vector = (
        rng.standard_normal(cpu.size) + 1j * rng.standard_normal(cpu.size)
    ).astype(np.complex64)
    cuda_vector = cuda_runtime.from_host(vector)

    cpu_result, cpu_timing = _timed(
        lambda: cpu.apply_low(vector),
        repeats=repeats,
        warmups=3,
    )
    cuda_result, cuda_timing = _timed(
        lambda: cuda.apply_low(cuda_vector),
        repeats=repeats,
        warmups=5,
        synchronize=cuda_runtime.synchronize,
    )
    cuda_host = cuda_runtime.to_host(cuda_result)
    relative_error = np.linalg.norm(cuda_host - cpu_result) / np.linalg.norm(
        cpu_result
    )
    return {
        "element_shape": [side, side],
        "elements": side * side,
        "unknowns": cpu.size,
        "kernel": cuda.low_operator_backend,
        "cpu_complex64": cpu_timing,
        "cuda_complex64": cuda_timing,
        "speedup_cpu_over_cuda": (
            cpu_timing["median_ms"] / cuda_timing["median_ms"]
        ),
        "relative_action_error": float(relative_error),
    }


def _solver_case(
    rows: int,
    columns: int,
    repeats: int,
    cuda_runtime: CupyComplex64Runtime,
) -> dict[str, Any]:
    problem = _problem(rows, columns)
    cpu = MatrixFreeScalarMaxwellOperator(
        problem, runtime=NumpyComplex64Runtime()
    )
    cuda = MatrixFreeScalarMaxwellOperator(problem, runtime=cuda_runtime)
    rhs = cpu.build_rhs()
    config = MPIRConfig(
        relative_tolerance=1.0e-10,
        inner_relative_tolerance=2.0e-3,
        max_outer_iterations=12,
        max_inner_iterations=400,
        gmres_restart=32,
    )

    cpu_result, cpu_timing = _timed(
        lambda: solve_mpir(cpu, rhs, config=config),
        repeats=repeats,
        warmups=1,
    )
    cuda_result, cuda_timing = _timed(
        lambda: solve_mpir(cuda, rhs, config=config),
        repeats=repeats,
        warmups=1,
        synchronize=cuda_runtime.synchronize,
    )
    relative_error = np.linalg.norm(cuda_result.solution - cpu_result.solution)
    relative_error /= np.linalg.norm(cpu_result.solution)
    return {
        "element_shape": [rows, columns],
        "elements": rows * columns,
        "unknowns": cpu.size,
        "cpu_mpir": {
            "timing": cpu_timing,
            "converged": cpu_result.converged,
            "relative_residual": cpu_result.relative_residual,
            "outer_iterations": cpu_result.outer_iterations,
            "inner_iterations": cpu_result.inner_iterations,
        },
        "cuda_mpir": {
            "timing": cuda_timing,
            "converged": cuda_result.converged,
            "relative_residual": cuda_result.relative_residual,
            "outer_iterations": cuda_result.outer_iterations,
            "inner_iterations": cuda_result.inner_iterations,
            "runtime": cuda_result.low_runtime,
            "kernel": cuda.low_operator_backend,
        },
        "speedup_cpu_over_cuda": (
            cpu_timing["median_ms"] / cuda_timing["median_ms"]
        ),
        "relative_solution_error": float(relative_error),
    }


def _device_properties(runtime: CupyComplex64Runtime) -> dict[str, Any]:
    cp = runtime.namespace
    properties = cp.cuda.runtime.getDeviceProperties(runtime.device_id)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode()
    return {
        "id": runtime.device_id,
        "name": name,
        "compute_capability": (
            f"{properties['major']}.{properties['minor']}"
        ),
        "global_memory_bytes": int(properties["totalGlobalMem"]),
        "cupy": cp.__version__,
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "driver": int(cp.cuda.runtime.driverGetVersion()),
    }


def run_benchmark(
    *,
    operator_sides: tuple[int, ...],
    solver_shape: tuple[int, int],
    repeats: int,
    device_id: int,
) -> dict[str, Any]:
    cuda_runtime = CupyComplex64Runtime(device_id=device_id)
    return {
        "schema": "electrical-maxwell-cuda-benchmark/v1",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "device": _device_properties(cuda_runtime),
        },
        "definitions": {
            "operator_timing": "synchronised apply_low wall time; allocations included",
            "solver_timing": "synchronised solve_mpir wall time; operator setup excluded",
            "speedup": "CPU median divided by CUDA median",
            "outer_precision": "host complex128",
            "inner_precision": "CPU/CUDA complex64",
        },
        "operator": [
            _operator_case(side, repeats, cuda_runtime)
            for side in operator_sides
        ],
        "solver": _solver_case(
            solver_shape[0],
            solver_shape[1],
            max(1, min(repeats, 5)),
            cuda_runtime,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operator-sides", type=int, nargs="+", default=(64, 128, 256)
    )
    parser.add_argument(
        "--solver-shape", type=int, nargs=2, default=(16, 256)
    )
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/maxwell-cuda.json"),
    )
    args = parser.parse_args()
    if args.repeats < 1 or any(side < 1 for side in args.operator_sides):
        parser.error("repeats and operator sides must be positive")
    if any(axis < 1 for axis in args.solver_shape):
        parser.error("solver shape axes must be positive")
    result = run_benchmark(
        operator_sides=tuple(args.operator_sides),
        solver_shape=tuple(args.solver_shape),
        repeats=args.repeats,
        device_id=args.device_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
