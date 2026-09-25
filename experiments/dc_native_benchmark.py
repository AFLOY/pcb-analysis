"""Portable NumPy path versus the fused C++ path of the layered-PCB DC solver.

Two copper layers of a 60 mm square board joined by a bank of vias, a slot in
the top layer, current injected along the left edge of the top layer and
removed along the right edge of the bottom layer.  Times the float32 action,
the float64 action, the operator construction (27 FP64 coarse probes and the
dense coarse inverse) and the whole MPIR solve, portable against native at
each thread count, and writes a JSON with ``environment`` and ``decision``.

    OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \\
        .venv/bin/python experiments/dc_native_benchmark.py --sizes 100,200,320 --threads 1,4,16
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
    CurrentTerminal,
    LayeredPCBMesh,
    MatrixFreePCBOperator,
    MPIRConfig,
    PCBConductionProblem,
    ViaConnection,
    solve_mpir,
)
from electrical.matrix_free_mpir_fem.native_dc import native_available  # noqa: E402

CONFIG = MPIRConfig()


def board(elements: int) -> PCBConductionProblem:
    active = np.ones((2, elements, elements), dtype=bool)
    active[0, elements // 3 : elements // 3 + max(1, elements // 20), elements // 4 : 3 * elements // 4] = False
    mesh = LayeredPCBMesh(
        element_active=active,
        layer_thickness_m=(35.0e-6, 35.0e-6),
        pitch_x_m=60.0e-3 / elements,
        pitch_y_m=60.0e-3 / elements,
    )
    step = max(1, elements // 16)
    vias = tuple(
        ViaConnection((0, r, c), (1, r, c), 1.0e-3)
        for r in range(step, elements, step)
        for c in range(elements // 2, elements, step)
    )
    source = tuple((0, r, 0) for r in range(elements + 1))
    sink = tuple((1, r, elements) for r in range(elements + 1))
    return PCBConductionProblem(
        mesh=mesh,
        terminals=(CurrentTerminal(source, 10.0, "source"), CurrentTerminal(sink, -10.0, "sink")),
        reference_node=sink[0],
        vias=vias,
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


def _operator(problem: PCBConductionProblem, **options: Any) -> MatrixFreePCBOperator:
    return MatrixFreePCBOperator(problem.mesh, reference_node=problem.reference_node, vias=problem.vias, **options)


def _case(elements: int, repeats: int, threads: list[int]) -> dict[str, Any]:
    problem = board(elements)
    start = time.perf_counter()
    portable = _operator(problem)
    construction_ms = (time.perf_counter() - start) * 1e3
    rhs = portable.build_rhs(problem.terminals)
    rng = np.random.default_rng(0)
    probe = rng.standard_normal(portable.size).astype(np.float32)
    portable_apply, portable_apply_timing = _timed(lambda: portable.apply_low(probe), repeats=40, warmups=3)
    probe_high = rng.standard_normal(portable.size)
    portable_high, portable_high_timing = _timed(lambda: portable.apply_high(probe_high), repeats=20, warmups=2)
    portable_result, portable_timing = _timed(lambda: solve_mpir(portable, rhs, config=CONFIG), repeats=repeats, warmups=1)

    by_threads: dict[str, Any] = {}
    for count in threads:
        start = time.perf_counter()
        native = _operator(problem, native=True, native_threads=count)
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
            "construction_speedup_vs_portable": construction_ms / native_construction_ms,
            "relative_solution_error_vs_portable": float(
                np.linalg.norm(native_result.solution - portable_result.solution) / np.linalg.norm(portable_result.solution)
            ),
        }
    return {
        "elements": [2, elements, elements],
        "nodes": portable.size,
        "vias": len(problem.vias),
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
        raise SystemExit("layered DC native extension not built; run python -m electrical.matrix_free_mpir_fem.native.build")
    cases = [_case(size, repeats, threads) for size in sizes]
    first = str(threads[0])
    speedups = [case["native_by_threads"][first]["speedup_vs_portable"] for case in cases]
    alike = all(case["native_by_threads"][first]["mpir"]["converged"] == case["portable_mpir"]["converged"] for case in cases)
    errors = [case["native_by_threads"][first]["relative_solution_error_vs_portable"] for case in cases
              if case["portable_mpir"]["converged"] and case["native_by_threads"][first]["mpir"]["converged"]]
    return {
        "schema": "dc-native-benchmark/v1",
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
            "construction_ms": "MatrixFreePCBOperator construction wall time (27 FP64 coarse probes and the dense coarse inverse included)",
            "solver_timing": "solve_mpir wall time; operator construction excluded and reported separately",
            "speedup": "portable NumPy median divided by native median",
            "native_path": "fused node-gather layered Q1 sheet operator with node-owned via links, plus the whole inner PCG with the two-level preconditioner in one SPMD OpenMP region; FTZ/DAZ per thread on the float32 path",
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
    parser.add_argument("--sizes", default="100,200,320", help="comma separated in-plane element counts (two layers each)")
    parser.add_argument("--threads", default="1", help="comma separated OpenMP thread counts; the first drives the decision")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results") / "dc_native_benchmark.json")
    args = parser.parse_args()
    report = run([int(v) for v in args.sizes.split(",")], args.repeats, [int(v) for v in args.threads.split(",")])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for case in report["cases"]:
        p = case["portable_mpir"]
        print(f"nodes={case['nodes']:7d} vias={case['vias']} coarse={case['coarse_size']} portable={p['timing']['median_ms']:9.1f} ms inner={p['inner_iterations']} construction={case['construction_ms_portable']:.0f} ms")
        for item in case["native_by_threads"].values():
            m = item["mpir"]
            print(f"    threads={item['threads']:2d} solve={m['timing']['median_ms']:8.1f} ms x{item['speedup_vs_portable']:6.2f} operator={item['operator_ms']['median_ms']:.4f} ms x{item['operator_speedup_vs_portable']:5.1f} high={item['high_operator_ms']['median_ms']:.3f} ms x{item['high_operator_speedup_vs_portable']:5.1f} construction={item['construction_ms']:.0f} ms x{item['construction_speedup_vs_portable']:4.2f} inner={m['inner_iterations']} conv={m['converged']} soldiff={item['relative_solution_error_vs_portable']:.1e}")
    print("decision:", json.dumps(report["decision"]))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
