"""Measured, not adopted: refinement boxes along narrow copper on a trace-dense board.

``power_module`` at 1 MHz, one filament per layer, near-field preconditioner:
the uniform 0.1 mm grid against a tensor grid refined to 0.1 mm around every
copper feature narrower than 0.3 mm plus the component footprints (0.5 mm
elsewhere). Records cells, branches, wall time and the current-density
metrics. Writes a JSON with ``environment`` and ``decision``; the copy that
the documents cite is ``docs/TRACE_REFINEMENT_RESULTS.json``.

    OPENBLAS_NUM_THREADS=1 PCB_NATIVE_THREADS=6 .venv/bin/python experiments/trace_refinement_probe.py
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
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from electrical.sheet_peec.plane_opt_contract import solve_plane_opt_problem  # noqa: E402
from geometry.cad_import import (  # noqa: E402
    board_refined_grid,
    board_vias,
    component_boxes,
    default_kicad_model_dir,
    export_kicad_step,
    kicad_cli_version,
    kicad_component_solids,
    kicad_step_body_map,
    layers_from_kicad_stackup,
    load_step,
    narrow_copper_boxes,
    plane_opt_problem_mapping,
    rasterize_board,
    read_kicad_stackup,
    resolve_bodies,
)


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plane-opt", type=Path, default=Path(__file__).resolve().parents[2] / "plane_opt_refactor")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/trace_refinement_probe.json"))
    args = parser.parse_args()
    board = args.plane_opt / "board" / "power_module" / "power_module.kicad_pcb"
    out = Path("benchmark-results/kicad-step"); out.mkdir(parents=True, exist_ok=True)
    step = export_kicad_step(board, out / "power_module_components_fused.step", model_dir=default_kicad_model_dir(), fuse_shapes=True, extra_args=("--no-dnp",))
    model = load_step(step)
    components = kicad_component_solids(model)
    layers, top = layers_from_kicad_stackup(read_kicad_stackup(board))
    body_map = kicad_step_body_map(layers, board_top_z_mm=top, ignore=tuple(f".*/{ref}/.*" for ref in components))
    resolved = resolve_bodies(model, body_map)
    zs = [layer.center_z_mm for layer in body_map.board.layers]
    pads = component_boxes(components, min_size_m=1.0e-3)
    thresholds = []
    for threshold in (0.3, 0.2, 0.16):
        start = time.perf_counter()
        traces = narrow_copper_boxes(resolved, zs, width_threshold_mm=threshold)
        detect = time.perf_counter() - start
        grid = board_refined_grid(resolved.board, coarse_pitch_mm=0.5, fine_pitch_mm=0.1, boxes=traces + pads, margin_mm=0.2, growth=1.4)
        thresholds.append({"width_threshold_mm": threshold, "trace_boxes": len(traces), "cells": int(grid.size), "rows": grid.shape[0], "cols": grid.shape[1], "detect_ms": detect * 1e3})
    grids = {
        "uniform 0.1 mm (v1)": rasterize_board(resolved, body_map.board, pitch_mm=0.1, method="section", y_down=True),
        "uniform 0.25 mm (v1)": rasterize_board(resolved, body_map.board, pitch_mm=0.25, method="section", y_down=True),
        "components 0.1 mm, margin 1 mm (v2)": rasterize_board(
            resolved, body_map.board,
            grid=board_refined_grid(resolved.board, coarse_pitch_mm=0.5, fine_pitch_mm=0.1, boxes=pads, margin_mm=1.0, growth=1.4),
            method="section", y_down=True),
        "traces < 0.3 mm + components 0.1 mm, margin 0.2 mm (v2)": rasterize_board(
            resolved, body_map.board,
            grid=board_refined_grid(resolved.board, coarse_pitch_mm=0.5, fine_pitch_mm=0.1,
                                    boxes=narrow_copper_boxes(resolved, zs, width_threshold_mm=0.3) + pads, margin_mm=0.2, growth=1.4),
            method="section", y_down=True),
    }
    runs = []
    for label, raster in grids.items():
        occupancy = raster.occupancy[0] > 0.0
        labels, _ = ndimage.label(occupancy)
        largest = 1 + int(np.argmax(np.bincount(labels.reshape(-1))[1:]))
        rows_idx, cols_idx = np.nonzero(labels == largest)
        name = raster.layers[0].name
        terminals = [
            {"name": "in", "pad": "L", "current_a": 1.0, "cells": [{"layer": name, "x": int(c), "y": int(r)} for r, c in zip(rows_idx, cols_idx) if c == cols_idx.min()]},
            {"name": "out", "pad": "R", "current_a": -1.0, "cells": [{"layer": name, "x": int(c), "y": int(r)} for r, c in zip(rows_idx, cols_idx) if c == cols_idx.max()]},
        ]
        mapping = plane_opt_problem_mapping(raster, terminals=terminals, frequency_hz=1.0e6, vias=board_vias(resolved, raster))
        copper = {k: {(c["x"], c["y"]) for c in v} for k, v in mapping["copper_by_layer"].items()}
        mapping["vertical_connections"] = [
            v for v in mapping["vertical_connections"]
            if all((v["cell"]["x"], v["cell"]["y"]) in copper[s[key]] for s in v["segments"] for key in ("upper_layer", "lower_layer"))
        ]
        start = time.perf_counter()
        result = solve_plane_opt_problem(mapping, {"maximum_iterations": 60, "relative_tolerance": 1e-8, "maximum_filaments": 1})
        wall = time.perf_counter() - start
        runs.append({
            "grid": label, "schema": mapping["schema"], "operator": result.metrics["inductance_operator"], "cells": int(raster.grid.size),
            "branches": result.metrics["branch_count"], "wall_ms": wall * 1e3,
            "max_current_density_a_per_mm2": result.metrics["max_current_density_a_per_mm2"],
            "bulk_p99_current_density_a_per_mm2": result.metrics["bulk_p99_current_density_a_per_mm2"],
        })
    uniform_fine, trace_aware = runs[0], runs[-1]
    adopted = trace_aware["branches"] < 0.5 * uniform_fine["branches"] and trace_aware["wall_ms"] < uniform_fine["wall_ms"]
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__, "cpu": _cpu_model(),
                        "cpu_count": os.cpu_count(), "kicad_cli": kicad_cli_version(), "PCB_NATIVE_THREADS": os.environ.get("PCB_NATIVE_THREADS")},
        "board": "power_module", "frequency_hz": 1.0e6, "thresholds": thresholds, "runs": runs,
        "decision": {"trace_aware_refinement": "adopted" if adopted else "not adopted",
                     "criteria": "under half the branches of the uniform 0.1 mm grid and a shorter 1 MHz solve; a tensor grid refines whole row and column strips, so scattered narrow traces refine the board"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for t in thresholds:
        print(f"threshold {t['width_threshold_mm']} mm: boxes={t['trace_boxes']} cells={t['cells']}")
    for r in runs:
        print(f"{r['grid']:58s} cells={r['cells']:6d} branches={r['branches']:6d} wall={r['wall_ms']/1e3:6.1f}s Jmax={r['max_current_density_a_per_mm2']:.3f} p99={r['bulk_p99_current_density_a_per_mm2']:.3f}")
    print("decision:", report["decision"]["trace_aware_refinement"])


if __name__ == "__main__":
    main()
