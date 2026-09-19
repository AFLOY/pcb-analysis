"""Acceptance 5: KiCad STEP exports against plane_opt's rasterisation of the same boards.

For each board in a plane_opt_refactor checkout: export STEP with copper
through kicad-cli, load it, sort the solids by z into board, copper layers
and via barrels, rasterise every layer onto plane_opt's grid, and compare
with plane_opt's CopperGrid built from the .kicad_pcb with every net in one
role.  Cells are split into interior (no neighbour differs in either mask),
drill holes (plane_opt fills them; the export leaves them open) and boundary
cells (plane_opt's inclusive centre-within-radius rule against sampling).
Writes a JSON with ``environment`` and ``decision``; the adopted copy lives at
``docs/KICAD_STEP_RESULTS.json``.

    PCB_NATIVE_THREADS=6 .venv/bin/python experiments/kicad_step_acceptance.py --boards power_module,bldc_driver,drone
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometry.step_voxelize import (  # noqa: E402
    board_vias,
    default_plane_method,
    export_kicad_step,
    kicad_cli_version,
    kicad_grid_origin_mm,
    kicad_step_body_map,
    layers_from_kicad_stackup,
    load_step,
    native_classify_available,
    rasterize_board,
    read_kicad_stackup,
    resolve_bodies,
)


class _AllRoles(dict):
    def get(self, key, default=None):
        return "all"

    def __getitem__(self, key):
        return "all"

    def __contains__(self, key):
        return True


def _boundary(mask: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(mask)
    edge[1:] |= mask[1:] != mask[:-1]
    edge[:-1] |= mask[:-1] != mask[1:]
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return edge


def reference(plane_opt: Path, board_path: Path, pitch_mm: float) -> dict[str, Any]:
    sys.path.insert(0, str(plane_opt / "src"))
    from plane_opt.geometry.metrics import CopperGrid
    from plane_opt.inputs import kicad as K

    start = time.perf_counter()
    board = K.load(board_path)
    roles = _AllRoles()
    _, pads = K.extract_components_and_pads(board, roles)
    tracks = K.extract_tracks(board, roles)
    vias = K.extract_vias(board, roles)
    zones, _ = K.extract_zones(board, roles)
    _, bounds = K.extract_outline(board)
    stackup = K.extract_stackup(board)
    data = {"pads": pads, "tracks": tracks, "vias": vias, "zones": zones, "stackup": stackup, "target_region": bounds, "target_connections": [], "roles": ["all"]}
    grid = CopperGrid(data, pitch_mm)
    rows, cols = grid.spec.height, grid.spec.width
    masks = {layer: np.zeros((rows, cols), dtype=bool) for layer in grid.layers}
    for (_, layer), cells in grid.masks.items():
        for x, y in cells:
            masks[layer][y, x] = True
    holes = np.zeros((rows, cols), dtype=bool)

    def mark_hole(x: float, y: float, radius: float) -> None:
        reach = radius + 0.5 * pitch_mm * math.sqrt(2.0)
        for cy in range(max(0, int((y - reach - bounds["min_y_mm"]) / pitch_mm)), min(rows, int((y + reach - bounds["min_y_mm"]) / pitch_mm) + 2)):
            for cx in range(max(0, int((x - reach - bounds["min_x_mm"]) / pitch_mm)), min(cols, int((x + reach - bounds["min_x_mm"]) / pitch_mm) + 2)):
                if math.hypot(bounds["min_x_mm"] + (cx + 0.5) * pitch_mm - x, bounds["min_y_mm"] + (cy + 0.5) * pitch_mm - y) <= reach:
                    holes[cy, cx] = True

    for via in vias:
        mark_hole(via["position"]["x_mm"], via["position"]["y_mm"], via["drill_mm"] / 2.0)
    thru = [pad for pad in pads if pad["type"] == "thru_hole"]
    for pad in thru:
        mark_hole(pad["position"]["x_mm"], pad["position"]["y_mm"], max(pad["drill"]["size_mm"]) / 2.0)
    # Non-plated holes carry no copper at all; plane_opt paints the pad size.
    npth = [pad for pad in pads if pad["type"] == "np_thru_hole"]
    for pad in npth:
        mark_hole(pad["position"]["x_mm"], pad["position"]["y_mm"], max(pad["size_mm"]) / 2.0)
    return {
        "masks": masks, "holes": holes, "bounds": bounds, "shape": (rows, cols), "seconds": time.perf_counter() - start,
        "counts": {"pads": len(pads), "thru_pads": len(thru), "np_thru_holes": len(npth), "tracks": len(tracks), "vias": len(vias), "zones": len(zones)},
        "copper_layers": list(stackup["copper_layers"]),
    }


def run_board(plane_opt: Path, name: str, pitch_mm: float, supersamples: list[int], step_dir: Path, threads: int, max_points: int) -> dict[str, Any]:
    board_path = plane_opt / "board" / name / f"{name}.kicad_pcb"
    ref = reference(plane_opt, board_path, pitch_mm)
    rows, cols = ref["shape"]
    start = time.perf_counter()
    step = export_kicad_step(board_path, step_dir / f"{name}.step")
    export_s = time.perf_counter() - start
    start = time.perf_counter()
    model = load_step(step)
    load_s = time.perf_counter() - start
    layers, top = layers_from_kicad_stackup(read_kicad_stackup(board_path))
    body_map = kicad_step_body_map(layers, board_top_z_mm=top)
    resolved = resolve_bodies(model, body_map)
    start = time.perf_counter()
    for _, solids in resolved.copper + resolved.vias:
        for solid in solids:
            solid.tessellate()
    tessellate_s = time.perf_counter() - start
    origin = kicad_grid_origin_mm(ref["bounds"]["min_x_mm"], ref["bounds"]["min_y_mm"], rows, pitch_mm)
    result: dict[str, Any] = {
        "board": name,
        "kicad_pcb": str(board_path.relative_to(plane_opt.parent)),
        "step_bytes": step.stat().st_size,
        "solids": len(model.solids),
        "triangles": int(sum(s.tessellate().size for _, ss in resolved.copper + resolved.vias for s in ss)),
        "copper_solids_by_layer": {spec.layer: len(solids) for spec, solids in resolved.copper},
        "via_barrels": sum(len(solids) for _, solids in resolved.vias),
        "kicad_vias_plus_thru_pads": ref["counts"]["vias"] + ref["counts"]["thru_pads"],
        "kicad_entities": ref["counts"],
        "layers": [{"name": layer.name, "bottom_z_mm": layer.bottom_z_mm, "top_z_mm": layer.top_z_mm} for layer in layers],
        "grid": {"rows": rows, "cols": cols, "pitch_mm": pitch_mm},
        "timing_s": {"kicad_cli_export": export_s, "load_step": load_s, "tessellate": tessellate_s, "plane_opt_reference": ref["seconds"]},
        "rasters": [],
    }
    for supersample in supersamples:
        points = rows * cols * supersample * supersample * len(layers)
        if points > max_points:
            result["rasters"].append({"supersample": supersample, "sample_points": points, "skipped": f"above --max-points {max_points}"})
            continue
        start = time.perf_counter()
        raster = rasterize_board(resolved, body_map.board, pitch_mm=pitch_mm, supersample=supersample, origin_mm=origin, shape=(rows, cols), y_down=True)
        raster_s = time.perf_counter() - start
        per_layer = []
        for index, layer in enumerate(raster.layers):
            step_mask = raster.occupancy[index] > 0.0
            ref_mask = ref["masks"][layer.name]
            edge = _boundary(step_mask) | _boundary(ref_mask)
            holes = ref["holes"]
            interior = ~edge & ~holes
            disagree = step_mask != ref_mask
            per_layer.append({
                "layer": layer.name,
                "step_cells": int(step_mask.sum()),
                "plane_opt_cells": int(ref_mask.sum()),
                "iou": float(np.sum(step_mask & ref_mask) / max(1, np.sum(step_mask | ref_mask))),
                "interior_cells": int(interior.sum()),
                "interior_disagreement": int(np.sum(disagree & interior)),
                "interior_step_only": int(np.sum(step_mask & ~ref_mask & interior)),
                "interior_plane_opt_only": int(np.sum(~step_mask & ref_mask & interior)),
                "interior_plane_opt_only_fraction_of_copper": float(np.sum(~step_mask & ref_mask & interior) / max(1, np.sum(ref_mask))),
                "hole_cells_step_only": int(np.sum(step_mask & ~ref_mask & holes)),
                "hole_cells_plane_opt_only": int(np.sum(~step_mask & ref_mask & holes)),
                "hole_cells_disagreeing": int(np.sum(disagree & holes)),
                "hole_disagreements_are_plane_opt_only": bool(np.all(ref_mask[disagree & holes]) and not np.any(step_mask[disagree & holes])),
                "boundary_cells": int(np.sum(edge & ~holes)),
                "boundary_disagreement": int(np.sum(disagree & edge & ~holes)),
                "step_only_outside_holes": int(np.sum(step_mask & ~ref_mask & ~holes)),
            })
        vias = board_vias(resolved, raster)
        result["rasters"].append({
            "method": default_plane_method(),
            "supersample": supersample,
            "sample_points": rows * cols * supersample * supersample * len(raster.layers),
            "seconds": raster_s,
            "threads": threads,
            "via_specs": len(vias.vias),
            "layers": per_layer,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plane-opt", type=Path, default=Path(os.environ.get("PCB_ANALYSIS_PLANE_OPT", ROOT.parent / "plane_opt_refactor")))
    parser.add_argument("--boards", default="power_module,bldc_driver,drone")
    parser.add_argument("--pitch-mm", type=float, default=0.25)
    parser.add_argument("--supersamples", default="1,3")
    parser.add_argument("--step-dir", type=Path, default=Path("benchmark-results") / "kicad-step")
    parser.add_argument("--max-points", type=int, default=2_000_000, help="skip a sampling whose point count exceeds this")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results") / "kicad_step_acceptance.json")
    args = parser.parse_args()
    threads = int(os.environ.get("PCB_NATIVE_THREADS", "1"))
    args.step_dir.mkdir(parents=True, exist_ok=True)
    boards = [run_board(args.plane_opt, name, args.pitch_mm, [int(v) for v in args.supersamples.split(",")], args.step_dir, threads, args.max_points) for name in args.boards.split(",")]
    layers_all = [layer for b in boards for r in b["rasters"] for layer in r.get("layers", [])]
    step_only_ok = all(layer["interior_step_only"] == 0 for layer in layers_all)
    plane_opt_only = max(layer["interior_plane_opt_only_fraction_of_copper"] for layer in layers_all)
    boundary = max(layer["boundary_disagreement"] / max(1, layer["boundary_cells"]) for b in boards for r in b["rasters"] for layer in r.get("layers", []))
    vias_ok = all(b["via_barrels"] == b["kicad_vias_plus_thru_pads"] for b in boards)
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
            "ocp": __import__("importlib.metadata").metadata.version("cadquery-ocp"),
            "kicad_cli": kicad_cli_version(), "cpu_count": os.cpu_count(), "native_extension_built": native_classify_available(),
            "PCB_NATIVE_THREADS": threads, "plane_opt_checkout": str(args.plane_opt), "device": "cpu",
        },
        "definitions": {
            "interior": "cells where no 4-neighbour differs in either mask and no drill hole is within half a diagonal",
            "hole": "cells within a via or through-hole drill radius, or within a non-plated hole's pad size, plus half a cell diagonal; plane_opt paints these as copper, the export leaves them open",
            "boundary": "cells with a differing 4-neighbour in either mask; plane_opt marks a cell whose centre is within the radius inclusive, the sampler tests the centre point against the solid",
            "reference": "plane_opt CopperGrid from the .kicad_pcb with every net in one role, cell-centre rasterisation",
        },
        "boards": boards,
        "decision": {
            "adopted": bool(step_only_ok and plane_opt_only <= 0.01 and vias_ok and boundary < 0.2),
            "criteria": (
                "away from edges and holes the export never has copper plane_opt lacks (interior step-only = 0 on every layer); "
                "interior copper plane_opt has and the export lacks is at most 1 % of plane_opt's copper per layer "
                "(plane_opt paints non-plated holes, drills and zone clearance cut-outs, and keeps off-board stubs); "
                "via barrels equal KiCad vias plus through-hole pads; boundary disagreement below 20 % of boundary cells"
            ),
            "interior_step_only_zero": step_only_ok,
            "max_interior_plane_opt_only_fraction_of_copper": plane_opt_only,
            "vias_match": vias_ok,
            "max_boundary_disagreement_fraction": boundary,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for b in boards:
        t = b["timing_s"]
        print(f"{b['board']}: {b['solids']} solids, {b['triangles']} triangles, grid {b['grid']['rows']}x{b['grid']['cols']}; export {t['kicad_cli_export']:.1f} s load {t['load_step']:.1f} s tessellate {t['tessellate']:.1f} s reference {t['plane_opt_reference']:.1f} s")
        for r in b["rasters"]:
            if "skipped" in r:
                print(f"  supersample {r['supersample']}: skipped ({r['skipped']}, {r['sample_points']} points)")
                continue
            print(f"  {r['method']} supersample {r['supersample']}: {r['seconds']:.1f} s, {r['sample_points']} points, {r['via_specs']} vias")
            for layer in r["layers"]:
                print(f"    {layer['layer']:6s} IoU={layer['iou']:.3f} interior step-only={layer['interior_step_only']} plane_opt-only={layer['interior_plane_opt_only']} ({100*layer['interior_plane_opt_only_fraction_of_copper']:.2f} %) holes step/plane_opt={layer['hole_cells_step_only']}/{layer['hole_cells_plane_opt_only']} boundary diff={layer['boundary_disagreement']}/{layer['boundary_cells']}")
    print("decision:", json.dumps(report["decision"]))


if __name__ == "__main__":
    main()
