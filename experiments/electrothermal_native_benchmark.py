"""Portable NumPy path versus the fused C++ paths of the staggered electro-thermal coupling.

The same two-layer board as ``dc_native_benchmark.py`` (60 mm square, a slot
in the top layer, a via bank, 10 A across), on a three-slab thermal stack with
10 W/m^2/K convection on both faces and rho(T) copper.  Times one whole
``run_electro_thermal`` (every electrical and thermal solve, every operator
construction, the Joule-heat mapping and the fixed point) with both solvers
portable and with both native at each thread count, and writes a JSON with
``environment`` and ``decision``.  This is the pcb-analysis side of the
Kicad_PowerOpt thermal-coupling path.

    OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \\
        .venv/bin/python experiments/electrothermal_native_benchmark.py --sizes 100,200,320 --threads 1,4,16
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dc_native_benchmark import _compiler, _cpu_model, board  # noqa: E402
from electrical.matrix_free_mpir_fem import MPIRConfig  # noqa: E402
from electrical.matrix_free_mpir_fem.native_dc import native_available as dc_native_available  # noqa: E402
from multiphysics.staggered_coupling import (  # noqa: E402
    CouplingConfig,
    ElectroThermalScenario,
    run_electro_thermal,
)
from thermal.matrix_free_mpir_fem import ConvectionBoundary, LayeredThermalMesh  # noqa: E402
from thermal.matrix_free_mpir_fem.native_hex import native_available as thermal_native_available  # noqa: E402

AMBIENT_K = 298.15
COUPLING = CouplingConfig(
    max_iterations=20,
    temperature_tolerance_k=1.0e-3,
    relative_loss_tolerance=1.0e-6,
    aitken=True,
    electrical=MPIRConfig(max_inner_iterations=400, max_outer_iterations=16, relative_tolerance=1.0e-8),
    thermal=MPIRConfig(max_inner_iterations=400, max_outer_iterations=16, relative_tolerance=1.0e-8),
)


def scenario(elements: int) -> ElectroThermalScenario:
    problem = board(elements)
    active = problem.mesh.element_active
    conductivity = np.full((3, elements, elements), 0.8)
    conductivity[0] = np.where(active[0], 385.0, 0.8)
    conductivity[2] = np.where(active[1], 385.0, 0.8)
    through = conductivity.copy()
    through[1] = 0.3
    thermal = LayeredThermalMesh(
        slab_thickness_m=(35.0e-6, 1.5e-3, 35.0e-6),
        pitch_x_m=60.0e-3 / elements,
        pitch_y_m=60.0e-3 / elements,
        conductivity_w_per_m_k=conductivity,
        through_plane_conductivity_w_per_m_k=through,
    )
    return ElectroThermalScenario(
        electrical=problem,
        thermal_mesh=thermal,
        layer_slabs=(0, 2),
        convection=(ConvectionBoundary("top", 10.0, AMBIENT_K), ConvectionBoundary("bottom", 10.0, AMBIENT_K)),
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


def _summary(result: Any, timing: dict[str, Any]) -> dict[str, Any]:
    return {
        "timing": timing,
        "converged": bool(result.converged),
        "coupling_iterations": int(result.iterations),
        "electrical_inner_iterations": [int(step.electrical_inner_iterations) for step in result.history],
        "thermal_inner_iterations": [int(step.thermal_inner_iterations) for step in result.history],
        "joule_loss_w": float(result.electrical.joule_loss_w),
        "cold_joule_loss_w": float(result.cold_joule_loss_w),
        "max_temperature_rise_k": float(np.nanmax(result.thermal.temperature_k) - AMBIENT_K),
    }


def _case(elements: int, repeats: int, threads: list[int]) -> dict[str, Any]:
    case = scenario(elements)
    portable_result, portable_timing = _timed(
        lambda: run_electro_thermal(case, config=COUPLING, native=False), repeats=repeats, warmups=1
    )
    portable = _summary(portable_result, portable_timing)
    by_threads: dict[str, Any] = {}
    for count in threads:
        native_result, native_timing = _timed(
            lambda: run_electro_thermal(case, config=COUPLING, native=True, native_threads=count),
            repeats=repeats,
            warmups=1,
        )
        summary = _summary(native_result, native_timing)
        by_threads[str(count)] = {
            "threads": count,
            **summary,
            "speedup_vs_portable": portable_timing["median_ms"] / native_timing["median_ms"],
            "relative_temperature_error_vs_portable": float(
                np.nanmax(np.abs(native_result.thermal.temperature_k - portable_result.thermal.temperature_k))
                / max(1e-300, float(np.nanmax(portable_result.thermal.temperature_k) - AMBIENT_K))
            ),
            "relative_loss_error_vs_portable": float(
                abs(native_result.electrical.joule_loss_w - portable_result.electrical.joule_loss_w)
                / portable_result.electrical.joule_loss_w
            ),
        }
    return {
        "elements": [2, elements, elements],
        "electrical_nodes": int(case.electrical.mesh.size),
        "thermal_nodes": int(np.prod(case.thermal_mesh.node_shape)),
        "vias": len(case.electrical.vias),
        "portable": portable,
        "native_by_threads": by_threads,
    }


def run(sizes: list[int], repeats: int, threads: list[int]) -> dict[str, Any]:
    if not (dc_native_available() and thermal_native_available()):
        raise SystemExit("build both native extensions first (electrical and thermal native.build)")
    cases = [_case(size, repeats, threads) for size in sizes]
    first = str(threads[0])
    speedups = [case["native_by_threads"][first]["speedup_vs_portable"] for case in cases]
    alike = all(case["native_by_threads"][first]["converged"] == case["portable"]["converged"] for case in cases)
    errors = [
        max(case["native_by_threads"][first]["relative_temperature_error_vs_portable"],
            case["native_by_threads"][first]["relative_loss_error_vs_portable"])
        for case in cases
        if case["portable"]["converged"] and case["native_by_threads"][first]["converged"]
    ]
    return {
        "schema": "electrothermal-native-benchmark/v1",
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
            "timing": "run_electro_thermal wall time: every DC and thermal solve including operator construction, the Joule-heat mapping and the Aitken fixed point",
            "portable": "native=False for both solvers (NumPy actions, NumPy inner PCG, NumPy FP64 residual)",
            "native": "native=True for both solvers: fused C++ float32 action and inner PCG, C++ float64 action for the outer residual and the coarse assembly",
            "speedup": "portable median divided by native median",
            "relative_temperature_error": "max |T_native - T_portable| over the max rise above ambient",
            "decision_threads": "the first thread count in the sweep",
        },
        "config": {
            "coupling_max_iterations": COUPLING.max_iterations,
            "temperature_tolerance_k": COUPLING.temperature_tolerance_k,
            "relative_loss_tolerance": COUPLING.relative_loss_tolerance,
            "solver": {"max_inner_iterations": 400, "max_outer_iterations": 16, "relative_tolerance": 1.0e-8},
        },
        "cases": cases,
        "decision": {
            "adopted": min(speedups) >= 2.0 and alike and bool(errors) and max(errors) <= 1.0e-6,
            "criteria": "every case at least 2x faster end to end at the decision thread count, identical convergence outcome, temperatures and losses within 1e-6",
            "minimum_speedup": min(speedups),
            "all_converged_alike": alike,
            "maximum_relative_error_converged_cases": max(errors) if errors else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sizes", default="100,200,320", help="comma separated in-plane element counts (two copper layers each)")
    parser.add_argument("--threads", default="1", help="comma separated OpenMP thread counts; the first drives the decision")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results") / "electrothermal_native_benchmark.json")
    args = parser.parse_args()
    report = run([int(v) for v in args.sizes.split(",")], args.repeats, [int(v) for v in args.threads.split(",")])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for case in report["cases"]:
        p = case["portable"]
        print(f"electrical nodes={case['electrical_nodes']:7d} thermal nodes={case['thermal_nodes']:7d} portable={p['timing']['median_ms']:9.0f} ms coupling={p['coupling_iterations']} rise={p['max_temperature_rise_k']:.3f} K")
        for item in case["native_by_threads"].values():
            print(f"    threads={item['threads']:2d} native={item['timing']['median_ms']:8.0f} ms x{item['speedup_vs_portable']:6.2f} coupling={item['coupling_iterations']} conv={item['converged']} dT={item['relative_temperature_error_vs_portable']:.1e} dP={item['relative_loss_error_vs_portable']:.1e}")
    print("decision:", json.dumps(report["decision"]))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
