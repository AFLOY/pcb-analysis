"""Acceptance of graded tensor grids in the thermal and DC electrical solvers.

1. Thermal hot spot: a 1 W, 2 x 2 mm patch on a 30 x 20 mm three-slab board,
   solved on uniform 0.5 / 0.25 / 0.125 mm grids and on graded grids (fine
   0.125 mm over the patch plus a margin, growth 1.4 / 1.2). Records cells,
   wall time, inner iterations and the peak error against the finest uniform
   grid, on the array and C++ paths.
2. DC conduction: a 0.2 mm x 20 mm copper trace on a grid graded along its
   length against the closed-form resistance, and the inner PCG iterations of
   the two-level against the Jacobi preconditioner on the graded strip and on
   a uniform 40 x 60 mm plane.

Writes a JSON with ``environment`` and ``decision``; the adopted copy lives at
``docs/TENSOR_GRID_RESULTS.json``.

    OPENBLAS_NUM_THREADS=1 PCB_NATIVE_THREADS=4 .venv/bin/python experiments/tensor_grid_acceptance.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from electrical.matrix_free_mpir_fem import (  # noqa: E402
    CurrentTerminal,
    LayeredPCBMesh,
    MPIRConfig,
    PCBConductionProblem,
    TensorGrid,
    graded_edges,
    refined_grid,
    solve_pcb_dc,
)
from thermal.matrix_free_mpir_fem import (  # noqa: E402
    ConvectionBoundary,
    LayeredThermalMesh,
    ThermalConductionProblem,
    solve_thermal_conduction,
)
from thermal.matrix_free_mpir_fem.native_hex import native_available  # noqa: E402

AMBIENT = 300.0
PATCH = (14e-3, 16e-3, 9e-3, 11e-3)


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
        return "unavailable"


def _environment() -> dict[str, Any]:
    try:
        import cupy

        cupy_version = cupy.__version__
    except Exception:  # noqa: BLE001
        cupy_version = None
    return {
        "platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
        "cupy": cupy_version, "cpu": _cpu_model(), "cpu_count": os.cpu_count(), "compiler": _compiler(),
        "native_extension_built": native_available(),
        "PCB_NATIVE_THREADS": os.environ.get("PCB_NATIVE_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def _board(grid: TensorGrid) -> ThermalConductionProblem:
    mesh = LayeredThermalMesh(
        (35e-6, 1.5e-3, 35e-6), grid.pitch_x_m, grid.pitch_y_m, (385.0, 0.8, 385.0),
        through_plane_conductivity_w_per_m_k=(385.0, 0.3, 385.0),
    )
    heat = np.zeros(mesh.element_grid_shape)
    overlap = grid.cell_overlap_fraction(*PATCH) * grid.cell_area_m2
    heat[2] = overlap / np.sum(overlap)
    return ThermalConductionProblem(
        mesh, convection=(ConvectionBoundary("top", 10.0, AMBIENT), ConvectionBoundary("bottom", 10.0, AMBIENT)),
        element_heat_w=heat,
    )


def thermal_hot_spot() -> dict[str, Any]:
    grids = [
        ("uniform 0.5 mm", TensorGrid.uniform(0.5e-3, (40, 60))),
        ("uniform 0.25 mm", TensorGrid.uniform(0.25e-3, (80, 120))),
        ("uniform 0.125 mm", TensorGrid.uniform(0.125e-3, (160, 240))),
        ("graded 0.125 mm, margin 1 mm, growth 1.4",
         refined_grid((0.0, 30e-3), (0.0, 20e-3), coarse_pitch_m=0.5e-3, fine_pitch_m=0.125e-3, refine_boxes_m=[PATCH], margin_m=1e-3, growth=1.4)),
        ("graded 0.125 mm, margin 2 mm, growth 1.2",
         refined_grid((0.0, 30e-3), (0.0, 20e-3), coarse_pitch_m=0.5e-3, fine_pitch_m=0.125e-3, refine_boxes_m=[PATCH], margin_m=2e-3, growth=1.2)),
    ]
    runs = []
    reference = None
    for label, grid in grids:
        problem = _board(grid)
        for path, native in (("array", False), ("native", True)):
            if native and not native_available():
                continue
            start = time.perf_counter()
            solution = solve_thermal_conduction(problem, native=native)
            wall = time.perf_counter() - start
            if label == "uniform 0.125 mm" and path == "array":
                reference = solution.max_temperature_k
            runs.append({
                "grid": label, "path": path, "cells": int(grid.size), "nodes": int(problem.mesh.size),
                "min_pitch_mm": float(min(grid.pitch_x_m.min(), grid.pitch_y_m.min())) * 1e3,
                "max_pitch_mm": float(max(grid.pitch_x_m.max(), grid.pitch_y_m.max())) * 1e3,
                "wall_ms": wall * 1e3, "inner_iterations": solution.solve.inner_iterations,
                "converged": solution.solve.converged, "max_temperature_k": solution.max_temperature_k,
                "heat_balance_error_w": solution.heat_balance_error_w,
            })
    assert reference is not None
    for run in runs:
        run["peak_error_vs_uniform_0125_k"] = run["max_temperature_k"] - reference
    return {"rise_k_reference": reference - AMBIENT, "runs": runs}


def _strip(pitch_x, pitch_y, rows, cols, current=1.0) -> PCBConductionProblem:
    active = np.ones((1, rows, cols), dtype=bool)
    mesh = LayeredPCBMesh(active, (35e-6,), pitch_x, pitch_y)
    return PCBConductionProblem(
        mesh,
        (CurrentTerminal(tuple((0, r, 0) for r in range(rows + 1)), current, "in"),
         CurrentTerminal(tuple((0, r, cols) for r in range(rows + 1)), -current, "out")),
        reference_node=(0, 0, 0),
    )


def dc_conduction() -> dict[str, Any]:
    edges = graded_edges(0.0, 20e-3, coarse_pitch_m=0.5e-3, fine_pitch_m=0.1e-3, refine_m=[(8e-3, 9e-3)], margin_m=0.5e-3)
    cases = []
    for label, problem in (
        ("0.2 mm trace, graded length (one row)", _strip(np.diff(edges), np.array([0.2e-3]), 1, edges.size - 1)),
        ("0.2 mm trace, graded length and rows 0.05/0.1/0.05", _strip(np.diff(edges), np.array([0.05e-3, 0.1e-3, 0.05e-3]), 3, edges.size - 1)),
        ("40 x 60 mm plane, uniform 0.5 mm", _strip(0.5e-3, 0.5e-3, 80, 120)),
    ):
        sigma = float(problem.mesh.conductivity_s_per_m[0, 0, 0])
        width = float(np.sum(problem.mesh.pitch_y_m))
        exact = float(np.sum(problem.mesh.pitch_x_m)) / (sigma * 35e-6 * width)
        row: dict[str, Any] = {"case": label, "elements": int(problem.mesh.element_active.size), "exact_resistance_ohm": exact}
        for pre in ("two-level", "jacobi"):
            start = time.perf_counter()
            solution = solve_pcb_dc(problem, preconditioner=pre, config=MPIRConfig(max_inner_iterations=4000))
            wall = time.perf_counter() - start
            row[pre] = {
                "wall_ms": wall * 1e3, "inner_iterations": solution.solve.inner_iterations,
                "outer_iterations": solution.solve.outer_iterations, "converged": solution.solve.converged,
                "resistance_ohm": solution.joule_loss_w, "resistance_rel_error": solution.joule_loss_w / exact - 1.0,
            }
        cases.append(row)
    return {"cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/tensor_grid_acceptance.json"))
    args = parser.parse_args()
    thermal = thermal_hot_spot()
    dc = dc_conduction()
    rise = thermal["rise_k_reference"]
    graded = [r for r in thermal["runs"] if r["grid"].startswith("graded") and r["path"] == "array"]
    coarse = next(r for r in thermal["runs"] if r["grid"] == "uniform 0.5 mm" and r["path"] == "array")
    thermal_ok = all(
        r["converged"] and abs(r["peak_error_vs_uniform_0125_k"]) < abs(coarse["peak_error_vs_uniform_0125_k"])
        and abs(r["peak_error_vs_uniform_0125_k"]) < 1e-3 * rise and r["cells"] < 0.25 * 160 * 240 for r in graded
    )
    paths_agree = all(
        abs(r["max_temperature_k"] - next(a["max_temperature_k"] for a in thermal["runs"] if a["grid"] == r["grid"] and a["path"] == "array")) < 1e-6
        for r in thermal["runs"] if r["path"] == "native"
    )
    dc_ok = (
        abs(dc["cases"][0]["two-level"]["resistance_rel_error"]) < 1e-9
        and all(c["two-level"]["converged"] for c in dc["cases"])
        and all(c["two-level"]["inner_iterations"] < c["jacobi"]["inner_iterations"] for c in dc["cases"])
    )
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": _environment(),
        "thermal_hot_spot": thermal,
        "dc_conduction": dc,
        "decision": {
            "graded_tensor_grid": "adopted" if (thermal_ok and paths_agree) else "not adopted",
            "dc_two_level_preconditioner": "adopted" if dc_ok else "not adopted",
            "criteria": {
                "thermal": "graded peak error below the uniform 0.5 mm error and below 1e-3 of the rise with under a quarter of the fine cells; native path within 1e-6 K of the array path",
                "dc": "graded one-row trace resistance exact to 1e-9; two-level converges with fewer inner iterations than Jacobi on every case",
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"thermal rise {rise:.2f} K")
    for r in thermal["runs"]:
        print(f"  {r['grid']:42s} {r['path']:6s} cells={r['cells']:6d} wall={r['wall_ms']:7.0f} ms inner={r['inner_iterations']:4d} err={r['peak_error_vs_uniform_0125_k']:+.4f} K")
    for c in dc["cases"]:
        print(f"  {c['case']:52s} two-level inner={c['two-level']['inner_iterations']:5d} ({c['two-level']['wall_ms']:.0f} ms) jacobi inner={c['jacobi']['inner_iterations']:5d} ({c['jacobi']['wall_ms']:.0f} ms) R err={c['two-level']['resistance_rel_error']:.1e}")
    print("decision:", json.dumps(report["decision"]["graded_tensor_grid"]), json.dumps(report["decision"]["dc_two_level_preconditioner"]))


if __name__ == "__main__":
    main()
