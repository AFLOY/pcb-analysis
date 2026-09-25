"""Portable NumPy versus fused C++ inner PCG for the thermal MPIR solve.

Fixture: the 4-slab stack of docs/THERMAL_MPIR_FEM.md (35 um Cu / 0.7 mm FR-4 /
0.7 mm FR-4 / 35 um Cu, 50 mm x 50 mm, convection on both faces, a heated trace
on top), default MPIRConfig(max_outer_iterations=16), default two-level
preconditioner.  Solve timings exclude operator construction, as in the docs.
Writes a JSON with ``environment`` and ``decision`` to ``benchmark-results/``.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/thermal_native_benchmark.py --threads 1,4,16
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

from electrical.matrix_free_mpir_fem import MPIRConfig, solve_mpir  # noqa: E402
from thermal.matrix_free_mpir_fem import (  # noqa: E402
    ConvectionBoundary,
    LayeredThermalMesh,
    MatrixFreeThermalOperator,
    ThermalConductionProblem,
)
from thermal.matrix_free_mpir_fem.native_hex import native_available  # noqa: E402

CONFIG = MPIRConfig(max_outer_iterations=16)


def stack(elements: int) -> ThermalConductionProblem:
    mesh = LayeredThermalMesh(
        slab_thickness_m=(35.0e-6, 0.7e-3, 0.7e-3, 35.0e-6),
        pitch_x_m=50.0e-3 / elements,
        pitch_y_m=50.0e-3 / elements,
        conductivity_w_per_m_k=(385.0, 0.8, 0.8, 385.0),
        through_plane_conductivity_w_per_m_k=(385.0, 0.3, 0.3, 385.0),
        element_shape=(elements, elements),
    )
    heat = np.zeros(mesh.element_grid_shape)
    heat[3, elements // 2, elements // 4 : 3 * elements // 4] = 1.0 / (elements // 2)
    return ThermalConductionProblem(
        mesh,
        convection=(
            ConvectionBoundary("top", 10.0, 298.15),
            ConvectionBoundary("bottom", 10.0, 298.15),
        ),
        element_heat_w=heat,
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
        return subprocess.run(["g++", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _summary(result: Any, timing: dict[str, Any], backend: str) -> dict[str, Any]:
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


def _case(elements: int, repeats: int, threads: list[int]) -> dict[str, Any]:
    problem = stack(elements)
    start = time.perf_counter()
    portable = MatrixFreeThermalOperator(problem)
    construction_ms = (time.perf_counter() - start) * 1e3
    rhs = portable.build_rhs(portable.default_reference_temperature())
    rng = np.random.default_rng(0)
    probe = rng.standard_normal(portable.size).astype(np.float32)
    portable_apply, portable_apply_timing = _timed(lambda: portable.apply_low(probe), repeats=40, warmups=3)
    probe_high = rng.standard_normal(portable.size)
    portable_high, portable_high_timing = _timed(lambda: portable.apply_high(probe_high), repeats=20, warmups=2)
    portable_result, portable_timing = _timed(lambda: solve_mpir(portable, rhs, config=CONFIG), repeats=repeats, warmups=1)

    by_threads: dict[str, Any] = {}
    for count in threads:
        start = time.perf_counter()
        native = MatrixFreeThermalOperator(problem, native=True, native_threads=count)
        native_construction_ms = (time.perf_counter() - start) * 1e3
        native_high, high_timing = _timed(lambda: native.apply_high(probe_high), repeats=20, warmups=2)
        native_apply, apply_timing = _timed(lambda: native.apply_low(probe), repeats=40, warmups=3)
        native_result, native_timing = _timed(lambda: solve_mpir(native, rhs, config=CONFIG), repeats=repeats, warmups=1)
        by_threads[str(count)] = {
            "threads": count,
            "kernel": native.low_operator_backend,
            "operator_ms": apply_timing,
            "construction_ms": native_construction_ms,
            "high_operator_ms": high_timing,
            "high_operator_speedup_vs_portable": portable_high_timing["median_ms"] / high_timing["median_ms"],
            "relative_high_action_error": float(np.linalg.norm(native_high - portable_high) / np.linalg.norm(portable_high)),
            "relative_action_error": float(np.linalg.norm(native_apply - portable_apply) / np.linalg.norm(portable_apply)),
            "mpir": _summary(native_result, native_timing, native.low_operator_backend),
            "speedup_vs_portable": portable_timing["median_ms"] / native_timing["median_ms"],
            "operator_speedup_vs_portable": portable_apply_timing["median_ms"] / apply_timing["median_ms"],
            "relative_solution_error_vs_portable": float(
                np.linalg.norm(native_result.solution - portable_result.solution) / np.linalg.norm(portable_result.solution)
            ),
        }
    return {
        "elements": [elements, elements],
        "nodes": portable.size,
        "coarse_size": portable.coarse_correction.coarse_size,
        "coarse_block": portable.coarse_correction.block,
        "construction_ms_portable": construction_ms,
        "portable_operator_ms": portable_apply_timing,
        "portable_high_operator_ms": portable_high_timing,
        "portable_mpir": _summary(portable_result, portable_timing, portable.low_operator_backend),
        "native_by_threads": by_threads,
    }


def run(sizes: list[int], repeats: int, threads: list[int]) -> dict[str, Any]:
    if not native_available():
        raise SystemExit("thermal native extension not built; run python -m thermal.matrix_free_mpir_fem.native.build")
    cases = [_case(size, repeats, threads) for size in sizes]
    first = str(threads[0])
    speedups = [case["native_by_threads"][first]["speedup_vs_portable"] for case in cases]
    alike = all(case["native_by_threads"][first]["mpir"]["converged"] == case["portable_mpir"]["converged"] for case in cases)
    errors = [case["native_by_threads"][first]["relative_solution_error_vs_portable"] for case in cases
              if case["portable_mpir"]["converged"] and case["native_by_threads"][first]["mpir"]["converged"]]
    return {
        "schema": "thermal-native-benchmark/v1",
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpu": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "compiler": _compiler(),
            "thread_sweep": threads,
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_PROC_BIND": os.environ.get("OMP_PROC_BIND"),
            "device": "cpu",
        },
        "definitions": {
            "operator_timing": "apply_low wall time on a fixed float32 probe",
            "high_operator_timing": "apply_high wall time on a fixed float64 probe",
            "construction_ms": "native MatrixFreeThermalOperator construction wall time (FP64 coarse assembly and dense inverse included)",
            "solver_timing": "solve_mpir wall time; operator construction (27 FP64 applications and the dense coarse inverse) excluded and reported separately",
            "speedup": "portable NumPy median divided by native median",
            "native_path": "fused node-gather hex Q1 operator plus the whole inner PCG with the two-level preconditioner in one SPMD OpenMP region; FTZ/DAZ per thread",
            "decision_threads": "the first thread count in the sweep",
        },
        "config": {"relative_tolerance": CONFIG.relative_tolerance, "inner_relative_tolerance": CONFIG.inner_relative_tolerance,
                   "max_outer_iterations": CONFIG.max_outer_iterations, "max_inner_iterations": CONFIG.max_inner_iterations},
        "cases": cases,
        "decision": {
            "adopted": min(speedups) >= 2.0 and alike and bool(errors) and max(errors) <= 1.0e-6,
            "criteria": "every case at least 2x faster end to end at the decision thread count, identical convergence outcome, converged solutions within 1e-6",
            "minimum_speedup": min(speedups),
            "all_converged_alike": alike,
            "maximum_relative_solution_error_converged_cases": max(errors) if errors else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sizes", default="50,100,200", help="comma separated in-plane element counts")
    parser.add_argument("--threads", default="1", help="comma separated OpenMP thread counts; the first drives the decision")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results") / "thermal_native_benchmark.json")
    args = parser.parse_args()
    report = run([int(v) for v in args.sizes.split(",")], args.repeats, [int(v) for v in args.threads.split(",")])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for case in report["cases"]:
        p = case["portable_mpir"]
        print(f"nodes={case['nodes']:7d} coarse={case['coarse_size']} portable={p['timing']['median_ms']:9.1f} ms inner={p['inner_iterations']} construction={case['construction_ms_portable']:.0f} ms")
        for item in case["native_by_threads"].values():
            m = item["mpir"]
            print(f"    threads={item['threads']:2d} solve={m['timing']['median_ms']:8.1f} ms x{item['speedup_vs_portable']:6.2f} operator={item['operator_ms']['median_ms']:.4f} ms x{item['operator_speedup_vs_portable']:5.1f} high={item['high_operator_ms']['median_ms']:.3f} ms x{item['high_operator_speedup_vs_portable']:5.1f} construction={item['construction_ms']:.0f} ms inner={m['inner_iterations']} conv={m['converged']} soldiff={item['relative_solution_error_vs_portable']:.1e}")
    print("decision:", json.dumps(report["decision"]))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
