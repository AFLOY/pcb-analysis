"""Acceptance of the precorrected-FFT sheet inductance operator on graded grids.

1. Operator accuracy: the pFFT operator's dense matrix against the exact
   convolution operator on a uniform 8 x 10 mesh, and against the direct
   closed-form pair sum on a graded mesh, for several interpolation orders and
   near radii (relative Frobenius error, worst entry, build time).
2. AC strip line at 1 MHz (F.Cu go, B.Cu return, 12 x 1 mm, filaments): Joule
   loss, unknowns, operator build and apply time and GMRES iterations on a
   uniform 0.1 mm grid (convolution), a uniform 0.5 mm grid, and a graded
   grid fine at both ends (pFFT), CPU path.
3. current-field schema v2: ``power_module`` from KiCad with component-driven
   grading solved through ``solve_current_field_problem`` at DC and 1 MHz (one
   filament per layer on both grids) against the uniform 0.25 mm v1 problem
   (branches, kernel bytes, wall time, current density).

Writes a JSON with ``environment`` and ``decision``; the adopted copy lives at
``docs/SHEET_PFFT_RESULTS.json``.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/sheet_pfft_acceptance.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from electrical.matrix_free_mpir_fem import TensorGrid, graded_edges  # noqa: E402
from electrical.sheet_peec.current_field_contract import solve_current_field_problem  # noqa: E402
from electrical.sheet_peec.sheet_operator import SheetInductanceOperator, SheetLayer, SheetStackup  # noqa: E402
from electrical.sheet_peec.sheet_peec import SheetMesh, Terminal, ViaBranch, solve_sheet_case  # noqa: E402
from electrical.sheet_peec.sheet_pfft import PfftSheetInductanceOperator  # noqa: E402

STACKUP = SheetStackup((SheetLayer("F", 0.0, 35e-6), SheetLayer("B", -1.6e-3, 35e-6)))


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _environment() -> dict[str, Any]:
    return {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
            "cpu": _cpu_model(), "cpu_count": os.cpu_count(), "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")}


def operator_accuracy() -> dict[str, Any]:
    from test_sheet_pfft import _dense_reference, _real_branch_mask

    rows, cols, pitch = 8, 10, 0.2e-3
    uniform = SheetMesh((rows, cols), pitch, STACKUP, np.ones((2, rows, cols), bool))
    reference_operator = SheetInductanceOperator((rows, cols), pitch, STACKUP)
    xe = graded_edges(0.0, 4e-3, coarse_pitch_m=0.5e-3, fine_pitch_m=0.1e-3, refine_m=[(1.5e-3, 2.0e-3)], margin_m=0.2e-3)
    ye = graded_edges(0.0, 2.5e-3, coarse_pitch_m=0.5e-3, fine_pitch_m=0.125e-3, refine_m=[(1.0e-3, 1.3e-3)], margin_m=0.2e-3)
    graded_grid = TensorGrid(xe, ye)
    graded = SheetMesh(graded_grid.shape, None, STACKUP, np.ones((2,) + graded_grid.shape, bool), grid=graded_grid)
    cases = []
    for order, radius in ((1, 2), (2, 3), (3, 3), (3, 4), (3, 5), (4, 5)):
        row: dict[str, Any] = {"order": order, "near_radius_cells": radius}
        for label, mesh, reference in (("uniform 0.2 mm", uniform, None), ("graded 0.1-0.5 mm", graded, None)):
            start = time.perf_counter()
            operator = PfftSheetInductanceOperator(mesh, order=order, near_radius_cells=radius)
            build = time.perf_counter() - start
            errors = {}
            for axis in ("x", "y"):
                mask = _real_branch_mask(mesh, axis)
                if label.startswith("uniform"):
                    expected = reference_operator.dense_matrix(axis)[np.ix_(mask, mask)]
                else:
                    expected = _dense_reference(mesh, axis)
                actual = operator.dense_matrix(axis)[np.ix_(mask, mask)]
                errors[axis] = {
                    "relative_frobenius": float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)),
                    "worst_entry_over_max": float(np.max(np.abs(actual - expected)) / np.max(np.abs(expected))),
                    "diagonal_relative": float(np.max(np.abs(np.diag(actual) - np.diag(expected)) / np.diag(expected))),
                }
            row[label] = {"build_ms": build * 1e3, "cells": int(mesh.grid.size), "errors": errors}
        cases.append(row)
    return {"uniform_reference": "SheetInductanceOperator (closed form to 24 cells, centre limit beyond)",
            "graded_reference": "closed form for every pair", "cases": cases}


def strip_line() -> dict[str, Any]:
    frequency, length, width = 1.0e6, 12e-3, 1.0e-3

    def run(label: str, grid: TensorGrid, kind: str) -> dict[str, Any]:
        rows, cols = grid.shape
        occupancy = np.ones((2, rows, cols), bool)
        vias = tuple(ViaBranch(r, cols - 1, 1, 0, 1e-4) for r in range(rows))
        mesh = SheetMesh((rows, cols), float(grid.pitch_x_m[0]) if grid.is_uniform else None, STACKUP, occupancy, vias=vias, grid=grid)
        start = time.perf_counter()
        if kind == "fft":
            operator: Any = SheetInductanceOperator((rows, cols), float(grid.pitch_x_m[0]), STACKUP, vertical_levels=mesh.vertical_levels)
        else:
            operator = PfftSheetInductanceOperator(mesh)
        build = time.perf_counter() - start
        gx = np.zeros((2, rows, cols)); gy = np.zeros_like(gx); gz = np.zeros((1, rows, cols))
        gx[:, :, :-1] = 1.0
        start = time.perf_counter()
        for _ in range(5):
            operator.apply(gx, gy, gz)
        apply = (time.perf_counter() - start) / 5
        terminals = [Terminal("in", 0, tuple((r, 0) for r in range(rows)), 1.0), Terminal("out", 1, tuple((r, 0) for r in range(rows)), -1.0)]
        solves = {}
        for preconditioner in ("diagonal", "near"):
            start = time.perf_counter()
            solution = solve_sheet_case(mesh, operator, terminals, frequency_hz=frequency, tolerance=1e-9, preconditioner=preconditioner)
            solves[preconditioner] = {"solve_ms": (time.perf_counter() - start) * 1e3, "gmres_iterations": solution.iterations, "converged": solution.converged,
                                      "joule_loss_w": float(np.sum(mesh.resistances() * np.abs(solution.branch_current) ** 2))}
        loss = solves["near"]["joule_loss_w"]
        return {"grid": label, "operator": type(operator).__name__, "cells": int(grid.size), "branches": mesh.branch_count,
                "kernel_bytes": int(operator.kernel_bytes), "build_ms": build * 1e3, "apply_ms": apply * 1e3,
                "solve_ms": solves["near"]["solve_ms"], "gmres_iterations": solves["near"]["gmres_iterations"], "converged": solves["near"]["converged"],
                "joule_loss_w": loss, "preconditioners": solves,
                "loss_rel_diff_near_vs_diagonal": loss / solves["diagonal"]["joule_loss_w"] - 1.0}

    runs = [
        run("uniform 0.1 mm", TensorGrid.uniform(0.1e-3, (10, 120)), "fft"),
        run("uniform 0.5 mm", TensorGrid.uniform(0.5e-3, (2, 24)), "fft"),
        run("uniform 0.1 mm, pFFT", TensorGrid.uniform(0.1e-3, (10, 120)), "pfft"),
    ]
    xe = graded_edges(0.0, length, coarse_pitch_m=0.5e-3, fine_pitch_m=0.1e-3, refine_m=[(0.0, 1.0e-3), (11e-3, 12e-3)], margin_m=0.5e-3)
    runs.append(run("graded 0.1 mm at the ends, 0.5 mm between", TensorGrid(xe, np.linspace(0.0, width, 11)), "pfft"))
    reference = runs[0]["joule_loss_w"]
    for r in runs:
        r["loss_rel_diff_vs_uniform_fine"] = r["joule_loss_w"] / reference - 1.0
    return {"frequency_hz": frequency, "runs": runs}


def power_module(plane_opt: Path) -> dict[str, Any] | None:
    from geometry.cad_import import (
        board_refined_grid, board_vias, component_boxes, default_kicad_model_dir, export_kicad_step, kicad_cli_available,
        kicad_component_solids, kicad_step_body_map, layers_from_kicad_stackup, load_step, current_field_problem_mapping,
        rasterize_board, read_kicad_stackup, resolve_bodies,
    )

    board = plane_opt / "board" / "power_module" / "power_module.kicad_pcb"
    model_dir = default_kicad_model_dir()
    if not (kicad_cli_available() and board.is_file() and model_dir is not None):
        return None
    out = Path("benchmark-results/kicad-step"); out.mkdir(parents=True, exist_ok=True)
    step = export_kicad_step(board, out / "power_module_components_fused.step", model_dir=model_dir, fuse_shapes=True, extra_args=("--no-dnp",))
    model = load_step(step)
    components = kicad_component_solids(model)
    layers, top = layers_from_kicad_stackup(read_kicad_stackup(board))
    body_map = kicad_step_body_map(layers, board_top_z_mm=top, ignore=tuple(f".*/{ref}/.*" for ref in components))
    resolved = resolve_bodies(model, body_map)
    boxes = component_boxes(components, min_size_m=1.0e-3)
    grid = board_refined_grid(resolved.board, coarse_pitch_mm=0.5, fine_pitch_mm=0.1, boxes=boxes, margin_mm=1.0, growth=1.4)
    rasters = {
        "uniform 0.25 mm (v1)": rasterize_board(resolved, body_map.board, pitch_mm=0.25, method="section", y_down=True),
        "graded 0.1 mm under components (v2)": rasterize_board(resolved, body_map.board, grid=grid, method="section", y_down=True),
    }
    runs = []
    for label, raster in rasters.items():
        from scipy import ndimage

        occupancy = raster.occupancy[0] > 0.0  # B.Cu (order 0)
        # Terminals must sit on one piece of copper: take the largest connected component.
        labels, count = ndimage.label(occupancy)
        largest = 1 + int(np.argmax(np.bincount(labels.reshape(-1))[1:]))
        rows_idx, cols_idx = np.nonzero(labels == largest)
        # Drive the bottom layer from its left-most to its right-most copper cells.
        left = cols_idx.min(); right = cols_idx.max()
        cells_in = [{"layer": raster.layers[0].name, "x": int(c), "y": int(r)} for r, c in zip(rows_idx, cols_idx) if c == left]
        cells_out = [{"layer": raster.layers[0].name, "x": int(c), "y": int(r)} for r, c in zip(rows_idx, cols_idx) if c == right]
        terminals = [{"name": "in", "pad": "L", "current_a": 1.0, "cells": cells_in}, {"name": "out", "pad": "R", "current_a": -1.0, "cells": cells_out}]
        for frequency in (0.0, 1.0e6):
            mapping = current_field_problem_mapping(raster, terminals=terminals, frequency_hz=frequency, vias=board_vias(resolved, raster))
            # A barrel whose axis lands on a cell the outline threshold left bare
            # cannot be a branch; drop it as the KiCad acceptance does.
            copper = {name: {(c["x"], c["y"]) for c in cells} for name, cells in mapping["copper_by_layer"].items()}
            mapping["vertical_connections"] = [
                v for v in mapping["vertical_connections"]
                if all((v["cell"]["x"], v["cell"]["y"]) in copper[s[key]] for s in v["segments"] for key in ("upper_layer", "lower_layer"))
            ]
            start = time.perf_counter()
            # One filament per layer on both grids: the comparison is between the
            # two inductance operators, and six filament layers would give the
            # pFFT precorrection 36 layer pairs of near terms on this board.
            result = solve_current_field_problem(mapping, {"maximum_iterations": 60, "relative_tolerance": 1e-8, "maximum_filaments": 1})
            wall = time.perf_counter() - start
            diagonal_wall = None
            if frequency > 0.0:
                start = time.perf_counter()
                solve_current_field_problem(mapping, {"maximum_iterations": 60, "relative_tolerance": 1e-8, "maximum_filaments": 1, "preconditioner": "diagonal"})
                diagonal_wall = (time.perf_counter() - start) * 1e3
            runs.append({
                "preconditioner": "near", "wall_ms_diagonal_preconditioner": diagonal_wall,
                "grid": label, "frequency_hz": frequency, "schema": mapping["schema"], "cells": int(raster.grid.size),
                "branches": result.metrics["branch_count"], "operator": result.metrics["inductance_operator"],
                "kernel_bytes": result.metrics["operator_kernel_bytes"], "wall_ms": wall * 1e3,
                "iterations": result.metrics.get("iterations"), "converged": result.metrics.get("converged"),
                "max_current_density_a_per_mm2": result.metrics["max_current_density_a_per_mm2"],
                "bulk_p99_current_density_a_per_mm2": result.metrics["bulk_p99_current_density_a_per_mm2"],
            })
    return {"board": "power_module", "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plane-opt", type=Path, default=Path(__file__).resolve().parents[2] / "plane_opt_refactor")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/sheet_pfft_acceptance.json"))
    parser.add_argument("--skip-kicad", action="store_true")
    args = parser.parse_args()
    accuracy = operator_accuracy()
    strip = strip_line()
    kicad = None if args.skip_kicad else power_module(args.plane_opt)
    default = next(c for c in accuracy["cases"] if c["order"] == 3 and c["near_radius_cells"] == 4)
    accuracy_ok = all(default[g]["errors"][a]["relative_frobenius"] < 1e-3 and default[g]["errors"][a]["diagonal_relative"] < 1e-9
                      for g in ("uniform 0.2 mm", "graded 0.1-0.5 mm") for a in ("x", "y"))
    fine, coarse, graded = strip["runs"][0], strip["runs"][1], strip["runs"][-1]
    strip_ok = graded["converged"] and abs(graded["loss_rel_diff_vs_uniform_fine"]) < abs(coarse["loss_rel_diff_vs_uniform_fine"]) and graded["branches"] < 0.5 * fine["branches"]
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "environment": _environment(),
        "operator_accuracy": accuracy, "strip_line": strip, "power_module": kicad,
        "decision": {"pfft_sheet_inductance": "adopted" if (accuracy_ok and strip_ok) else "not adopted",
                     "criteria": "order 3, near radius 4: relative Frobenius error < 1e-3 against both references with exact diagonals; graded strip-line loss closer to the uniform 0.1 mm result than the uniform 0.5 mm grid with under half its branches"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for c in accuracy["cases"]:
        u, g = c["uniform 0.2 mm"]["errors"], c["graded 0.1-0.5 mm"]["errors"]
        print(f"order={c['order']} R={c['near_radius_cells']} uniform x/y {u['x']['relative_frobenius']:.1e}/{u['y']['relative_frobenius']:.1e} graded x/y {g['x']['relative_frobenius']:.1e}/{g['y']['relative_frobenius']:.1e} build {c['graded 0.1-0.5 mm']['build_ms']:.0f} ms")
    for r in strip["runs"]:
        d = r["preconditioners"]["diagonal"]
        print(f"strip {r['grid']:42s} {r['operator']:30s} branches={r['branches']:5d} build={r['build_ms']:6.0f} apply={r['apply_ms']:6.1f} solve near={r['solve_ms']:6.0f} ms it={r['gmres_iterations']:3d} (diagonal {d['solve_ms']:6.0f} ms it={d['gmres_iterations']:3d}) loss_diff={r['loss_rel_diff_vs_uniform_fine']:+.3e}")
    if kicad:
        for r in kicad["runs"]:
            print(f"power_module {r['grid']:36s} f={r['frequency_hz']:.0e} {r['operator']:30s} cells={r['cells']:6d} branches={r['branches']:6d} wall near={r['wall_ms']:7.0f} ms diagonal={r['wall_ms_diagonal_preconditioner']} Jmax={r['max_current_density_a_per_mm2']:.3f} p99={r['bulk_p99_current_density_a_per_mm2']:.3f}")
    print("decision:", report["decision"]["pfft_sheet_inductance"])


if __name__ == "__main__":
    main()
