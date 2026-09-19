"""Acceptance of component-driven graded grids on a KiCad board.

Exports ``power_module`` with fused copper and component models, builds a graded
grid (fine pitch under every component solid plus a margin, coarse pitch
elsewhere) and compares it with uniform grids at the coarse and the fine
pitch: cells, raster time, copper area per layer, and the thermal solve of
1 W dissipated in one component's footprint (peak temperature, solve time,
inner iterations).

Needs ``kicad-cli``, the KiCad 3D library models of the board's parts, and the
``plane_opt_refactor`` checkout next to this repository.  Writes a JSON with
``environment`` and ``decision``; the adopted copy lives at
``docs/KICAD_REFINEMENT_RESULTS.json``.

    OPENBLAS_NUM_THREADS=1 PCB_NATIVE_THREADS=4 .venv/bin/python experiments/kicad_refinement_acceptance.py
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

from electrical.matrix_free_mpir_fem import TensorGrid  # noqa: E402
from geometry.cad_import import (  # noqa: E402
    board_refined_grid,
    board_thermal_mesh,
    component_boxes,
    default_kicad_model_dir,
    export_kicad_step,
    kicad_cli_version,
    kicad_component_solids,
    kicad_step_body_map,
    layers_from_kicad_stackup,
    load_step,
    rasterize_board,
    read_kicad_stackup,
    refinement_summary,
    resolve_bodies,
    sample_plane_fill,
)
from thermal.matrix_free_mpir_fem import ConvectionBoundary, ThermalConductionProblem, solve_thermal_conduction  # noqa: E402
from thermal.matrix_free_mpir_fem.native_hex import native_available  # noqa: E402

AMBIENT = 298.15
H = 10.0


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
    return {
        "platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
        "cpu": _cpu_model(), "cpu_count": os.cpu_count(), "compiler": _compiler(),
        "native_extension_built": native_available(), "kicad_cli": kicad_cli_version(),
        "PCB_NATIVE_THREADS": os.environ.get("PCB_NATIVE_THREADS"), "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def run(board: Path, out_dir: Path, *, coarse_mm: float, fine_mm: float, margin_mm: float, growth: float, heated: str) -> dict[str, Any]:
    step = export_kicad_step(
        board, out_dir / f"{board.stem}_components_fused.step", model_dir=default_kicad_model_dir(), fuse_shapes=True, extra_args=("--no-dnp",)
    )
    model = load_step(step)
    components = kicad_component_solids(model)
    layers, top = layers_from_kicad_stackup(read_kicad_stackup(board))
    body_map = kicad_step_body_map(layers, board_top_z_mm=top, ignore=tuple(f".*/{ref}/.*" for ref in components))
    resolved = resolve_bodies(model, body_map)
    boxes = component_boxes(components, min_size_m=1.0e-3)
    lo, hi = (np.asarray(b) for b in resolved.board.bounds_m)
    extent = (hi[:2] - lo[:2])

    grids = {
        f"uniform {coarse_mm} mm": TensorGrid.uniform(coarse_mm * 1e-3, tuple(int(np.ceil(e / (coarse_mm * 1e-3) - 1e-6)) for e in extent[::-1]), tuple(lo[:2])),
        f"uniform {fine_mm} mm": TensorGrid.uniform(fine_mm * 1e-3, tuple(int(np.ceil(e / (fine_mm * 1e-3) - 1e-6)) for e in extent[::-1]), tuple(lo[:2])),
        f"graded {fine_mm} mm under components, margin {margin_mm} mm, growth {growth}":
            board_refined_grid(resolved.board, coarse_pitch_mm=coarse_mm, fine_pitch_mm=fine_mm, boxes=boxes, margin_mm=margin_mm, growth=growth),
        f"graded {fine_mm} mm under components, margin {2 * margin_mm} mm, growth {1.0 + (growth - 1.0) / 2:.2f}":
            board_refined_grid(resolved.board, coarse_pitch_mm=coarse_mm, fine_pitch_mm=fine_mm, boxes=boxes, margin_mm=2 * margin_mm, growth=1.0 + (growth - 1.0) / 2),
    }
    heated_box = next(box for box in boxes if box.name == heated)
    copper = [solid for _, solids in resolved.copper for solid in solids] + [s for _, solids in resolved.vias for s in solids]
    rows = []
    for label, grid in grids.items():
        start = time.perf_counter()
        raster = rasterize_board(resolved, body_map.board, grid=grid, method="section")
        raster_wall = time.perf_counter() - start
        thermal = board_thermal_mesh(raster)
        mesh = thermal.mesh
        heat = np.zeros(mesh.element_grid_shape)
        # Component power enters through its pads: weight the box overlap by the copper fill.
        overlap = grid.cell_overlap_fraction(*heated_box.bounds_m) * grid.cell_area_m2 * raster.fill[-1]
        top_slab = thermal.layer_slabs[-1]
        heat[top_slab] = np.where(mesh.active[top_slab], overlap, 0.0)
        heat[top_slab] *= 1.0 / np.sum(heat[top_slab])
        problem = ThermalConductionProblem(
            mesh, convection=(ConvectionBoundary("top", H, AMBIENT), ConvectionBoundary("bottom", H, AMBIENT)), element_heat_w=heat
        )
        start = time.perf_counter()
        solution = solve_thermal_conduction(problem, native=native_available() or None)
        solve_wall = time.perf_counter() - start
        rows.append({
            "grid": label, "cells": int(grid.size), "rows": grid.shape[0], "cols": grid.shape[1],
            "thermal_nodes": int(mesh.size), "raster_wall_ms": raster_wall * 1e3, "solve_wall_ms": solve_wall * 1e3,
            "inner_iterations": solution.solve.inner_iterations, "converged": solution.solve.converged,
            # Exact section coverage over the whole grid (the raster's own area also drops edge cells below the outline threshold).
            "copper_area_mm2": [
                float(np.sum(sample_plane_fill(copper, z_m=layer.center_z_mm * 1e-3, grid=grid, method="section") * grid.cell_area_m2)) * 1e6
                for layer in body_map.board.layers
            ],
            "raster_copper_area_mm2": [raster.copper_area_m2(i) * 1e6 for i in range(len(body_map.board.layers))],
            "peak_temperature_k": solution.max_temperature_k, "heat_balance_error_w": solution.heat_balance_error_w,
        })
    reference = next(r for r in rows if r["grid"].startswith(f"uniform {fine_mm}"))
    for r in rows:
        r["peak_error_vs_uniform_fine_k"] = r["peak_temperature_k"] - reference["peak_temperature_k"]
        r["copper_area_rel_diff_vs_uniform_fine"] = [a / b - 1.0 for a, b in zip(r["copper_area_mm2"], reference["copper_area_mm2"])]
    graded_rows = [r for r in rows if r["grid"].startswith("graded")]
    graded = graded_rows[0]
    return {
        "board": board.stem, "components_with_models": sorted(components), "heated_component": heated,
        "refinement": [refinement_summary(grid, boxes) for label, grid in grids.items() if label.startswith("graded")],
        "rise_k_reference": reference["peak_temperature_k"] - AMBIENT,
        "runs": rows,
        "graded_ok": all(
            r["converged"] and abs(r["peak_error_vs_uniform_fine_k"]) < abs(rows[0]["peak_error_vs_uniform_fine_k"])
            and r["cells"] < 0.5 * reference["cells"] and max(abs(d) for d in r["copper_area_rel_diff_vs_uniform_fine"]) < 1e-9
            for r in graded_rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plane-opt", type=Path, default=Path(__file__).resolve().parents[2] / "plane_opt_refactor")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/kicad_refinement_acceptance.json"))
    parser.add_argument("--coarse-mm", type=float, default=0.5)
    parser.add_argument("--fine-mm", type=float, default=0.1)
    parser.add_argument("--margin-mm", type=float, default=1.0)
    parser.add_argument("--growth", type=float, default=1.4)
    parser.add_argument("--heated", default="C1")
    args = parser.parse_args()
    out_dir = Path("benchmark-results/kicad-step")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run(args.plane_opt / "board" / "power_module" / "power_module.kicad_pcb", out_dir, coarse_mm=args.coarse_mm,
                 fine_mm=args.fine_mm, margin_mm=args.margin_mm, growth=args.growth, heated=args.heated)
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": _environment(),
        "power_module": result,
        "decision": {
            "component_driven_graded_grid": "adopted" if result["graded_ok"] else "not adopted",
            "criteria": "graded peak error below the coarse uniform error, under half the fine cells, copper area within 1e-9 of the fine grid (fused KiCad shapes, so the exact section coverage is grid independent)",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for summary in result["refinement"]:
        print("refinement:", json.dumps(summary))
    print(f"rise {result['rise_k_reference']:.2f} K on the fine grid, heated {result['heated_component']}")
    for r in result["runs"]:
        print(f"  {r['grid']:62s} cells={r['cells']:6d} raster={r['raster_wall_ms']:6.0f} ms solve={r['solve_wall_ms']:6.0f} ms inner={r['inner_iterations']:4d} peak_err={r['peak_error_vs_uniform_fine_k']:+.4f} K cu_diff={[f'{d:+.1e}' for d in r['copper_area_rel_diff_vs_uniform_fine']]}")
    print("decision:", report["decision"]["component_driven_graded_grid"])


if __name__ == "__main__":
    main()
