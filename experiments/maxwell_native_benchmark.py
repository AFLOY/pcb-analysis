"""Portable NumPy versus fused C++ inner Krylov for the scalar Maxwell MPIR solve.

Same fixture, tolerance, and MPIRConfig as ``maxwell_cuda_benchmark.py``.  Both
paths run on the host; the low path differs only in who executes the complex64
operator and the restarted GMRES cycle.  Writes a JSON with ``environment`` and
``decision`` to ``benchmark-results/``.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/maxwell_native_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from electrical.matrix_free_mpir_fem import (  # noqa: E402
    MPIRConfig,
    MatrixFreeScalarMaxwellOperator,
    NumpyComplex64Runtime,
    solve_mpir,
)
from electrical.matrix_free_mpir_fem.native_q1 import (  # noqa: E402
    native_available,
    native_dot_accumulation,
    native_orthogonalization,
    native_threads,
)
from maxwell_cuda_benchmark import _problem  # noqa: E402

CONFIG = MPIRConfig(
    relative_tolerance=1.0e-10,
    inner_relative_tolerance=2.0e-3,
    max_outer_iterations=12,
    max_inner_iterations=400,
    gmres_restart=32,
)


def _timed(fn: Callable[[], Any], repeats: int, warmups: int) -> tuple[Any, dict[str, Any]]:
    for _ in range(warmups):
        result = fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - start) * 1.0e3)
    return result, {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "samples_ms": samples,
        "repeats": repeats,
    }


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _compiler() -> str:
    try:
        return subprocess.run(
            ["g++", "--version"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _case(
    rows: int, columns: int, repeats: int, orthogonalization: str, dot_accumulation: str
) -> dict[str, Any]:
    problem = _problem(rows, columns)
    portable = MatrixFreeScalarMaxwellOperator(problem, runtime=NumpyComplex64Runtime())
    native = MatrixFreeScalarMaxwellOperator(
        problem,
        runtime=NumpyComplex64Runtime(),
        native=True,
        native_orthogonalization=orthogonalization,
        native_dot_accumulation=dot_accumulation,
    )
    rhs = portable.build_rhs()
    rng = np.random.default_rng(0)
    probe = (
        rng.standard_normal(portable.size) + 1j * rng.standard_normal(portable.size)
    ).astype(np.complex64)

    portable_apply, portable_apply_timing = _timed(
        lambda: portable.apply_low(probe), repeats=max(repeats, 9) * 20, warmups=5
    )
    native_apply, native_apply_timing = _timed(
        lambda: native.apply_low(probe), repeats=max(repeats, 9) * 20, warmups=5
    )
    action_error = float(
        np.linalg.norm(native_apply - portable_apply) / np.linalg.norm(portable_apply)
    )

    portable_result, portable_timing = _timed(
        lambda: solve_mpir(portable, rhs, config=CONFIG), repeats=repeats, warmups=1
    )
    native_result, native_timing = _timed(
        lambda: solve_mpir(native, rhs, config=CONFIG), repeats=repeats, warmups=1
    )
    solution_error = float(
        np.linalg.norm(native_result.solution - portable_result.solution)
        / np.linalg.norm(portable_result.solution)
    )

    def summary(result: Any, timing: dict[str, Any], backend: str) -> dict[str, Any]:
        return {
            "kernel": backend,
            "timing": timing,
            "converged": result.converged,
            "relative_residual": result.relative_residual,
            "outer_iterations": result.outer_iterations,
            "inner_iterations": result.inner_iterations,
            "low_operator_applications": result.low_operator_applications,
            "high_operator_applications": result.high_operator_applications,
        }

    return {
        "element_shape": [rows, columns],
        "elements": rows * columns,
        "unknowns": portable.size,
        "operator": {
            "portable_complex64_ms": portable_apply_timing,
            "native_complex64_ms": native_apply_timing,
            "speedup_portable_over_native": portable_apply_timing["median_ms"]
            / native_apply_timing["median_ms"],
            "relative_action_error": action_error,
        },
        "portable_mpir": summary(portable_result, portable_timing, portable.low_operator_backend),
        "native_mpir": summary(native_result, native_timing, native.low_operator_backend),
        "speedup_portable_over_native": portable_timing["median_ms"]
        / native_timing["median_ms"],
        "relative_solution_error": solution_error,
    }


def run(
    shapes: list[tuple[int, int]], repeats: int, orthogonalization: str, dot_accumulation: str
) -> dict[str, Any]:
    if not native_available():
        raise SystemExit(
            "native extension not built; run "
            "python -m electrical.matrix_free_mpir_fem.native.build"
        )
    cases = [
        _case(rows, columns, repeats, orthogonalization, dot_accumulation)
        for rows, columns in shapes
    ]
    speedups = [case["speedup_portable_over_native"] for case in cases]
    all_converged = all(
        case["portable_mpir"]["converged"] == case["native_mpir"]["converged"]
        for case in cases
    )
    same_work = all(
        case["portable_mpir"]["inner_iterations"] == case["native_mpir"]["inner_iterations"]
        for case in cases
    )
    # The solution difference is only meaningful where both paths reached the
    # requested FP64 residual; a stalled case is compared by reached residual.
    converged_errors = [
        case["relative_solution_error"]
        for case in cases
        if case["portable_mpir"]["converged"] and case["native_mpir"]["converged"]
    ]
    max_solution_error = max(converged_errors) if converged_errors else None
    adopted = (
        min(speedups) >= 2.0
        and all_converged
        and bool(converged_errors)
        and max_solution_error <= 1.0e-6
    )
    return {
        "schema": "electrical-maxwell-native-benchmark/v1",
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpu": _cpu_model(),
            "compiler": _compiler(),
            "native_threads": native_threads(),
            "native_orthogonalization": orthogonalization,
            "native_dot_accumulation": dot_accumulation,
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "device": "cpu",
        },
        "definitions": {
            "operator_timing": "apply_low wall time on a fixed complex64 probe; allocations included",
            "solver_timing": "solve_mpir wall time; operator setup excluded",
            "speedup": "portable NumPy median divided by native C++ median",
            "outer_precision": "host complex128",
            "inner_precision": "complex64 on both paths",
            "native_path": "fused node-gather Q1 operator plus restarted right-Jacobi GMRES "
            "(Givens) in C++, one SPMD OpenMP region per solve, FTZ/DAZ enabled during "
            "the native call",
            "native_orthogonalization": {
                "mgs": "modified Gram-Schmidt, one reduction per basis vector, "
                "complex64-rounded coefficients as in the portable runtime",
                "cgs2": "classical Gram-Schmidt with one reorthogonalisation, blocked so w "
                "stays in L1, two reductions per column; coefficients rounded to complex64 "
                "per pass",
            }[orthogonalization],
            "native_dot_accumulation": {
                "float64": "every dot-product term accumulated in double, as in the "
                "portable runtime",
                "float32": "1,024-element blocks accumulated in float, blocks summed in "
                "double; the norms stay in double",
            }[dot_accumulation],
        },
        "config": {
            "relative_tolerance": CONFIG.relative_tolerance,
            "inner_relative_tolerance": CONFIG.inner_relative_tolerance,
            "max_outer_iterations": CONFIG.max_outer_iterations,
            "max_inner_iterations": CONFIG.max_inner_iterations,
            "gmres_restart": CONFIG.gmres_restart,
        },
        "cases": cases,
        "decision": {
            "adopted": adopted,
            "criteria": "every case at least 2x faster end to end, identical convergence "
            "outcome per case, and relative solution difference at most 1e-6 on the "
            "cases where both paths converged",
            "minimum_speedup": min(speedups),
            "all_converged_alike": all_converged,
            "same_inner_iterations": same_work,
            "maximum_relative_solution_error_converged_cases": max_solution_error,
            "converged_case_count": len(converged_errors),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results") / "maxwell_native_benchmark.json",
    )
    parser.add_argument(
        "--shapes",
        default="16x256,64x256,256x256",
        help="comma separated rows x columns element shapes",
    )
    parser.add_argument(
        "--orthogonalization",
        choices=("mgs", "cgs2"),
        default=None,
        help="native Gram-Schmidt variant; default follows PCB_NATIVE_ORTHO or mgs",
    )
    parser.add_argument(
        "--dot-accumulation",
        choices=("float64", "float32"),
        default=None,
        help="native dot-product accumulation; default follows PCB_NATIVE_DOT or float64",
    )
    args = parser.parse_args()
    shapes = [tuple(int(v) for v in item.split("x")) for item in args.shapes.split(",")]
    orthogonalization = args.orthogonalization or native_orthogonalization()
    dot_accumulation = args.dot_accumulation or native_dot_accumulation()
    report = run(shapes, args.repeats, orthogonalization, dot_accumulation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for case in report["cases"]:
        print(
            f"unknowns={case['unknowns']:6d} portable={case['portable_mpir']['timing']['median_ms']:9.1f} ms "
            f"native={case['native_mpir']['timing']['median_ms']:9.1f} ms "
            f"speedup={case['speedup_portable_over_native']:.2f}x "
            f"operator={case['operator']['speedup_portable_over_native']:.2f}x "
            f"solution_diff={case['relative_solution_error']:.2e}"
        )
    print("decision:", json.dumps(report["decision"]))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
