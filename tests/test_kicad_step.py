"""Acceptance 5: a KiCad STEP export against plane_opt's rasterisation of the same board.

Needs ``kicad-cli``, the ``cad`` extra, the ``plane_opt_refactor`` checkout
next to this repository (its boards and its KiCad extractor) and the geometry
native extension; each is skipped when missing.  ``PCB_ANALYSIS_PLANE_OPT``
points at another checkout.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from geometry.cad_import import (
    board_thermal_mesh,
    board_vias,
    export_kicad_step,
    kicad_cli_available,
    kicad_grid_origin_mm,
    kicad_step_body_map,
    layers_from_kicad_stackup,
    load_step,
    ocp_available,
    rasterize_board,
    read_kicad_stackup,
    resolve_bodies,
)

PLANE_OPT = Path(os.environ.get("PCB_ANALYSIS_PLANE_OPT", Path(__file__).resolve().parents[2] / "plane_opt_refactor"))
BOARD = PLANE_OPT / "board" / "power_module" / "power_module.kicad_pcb"
PITCH_MM = 0.25

pytestmark = pytest.mark.skipif(
    not (ocp_available() and kicad_cli_available() and BOARD.is_file() and (PLANE_OPT / "src").is_dir()),
    reason="needs OCP, kicad-cli and the plane_opt_refactor checkout with its boards",
)


class _AllRoles(dict):
    """plane_opt assigns copper to roles by net; give every net the same role."""

    def get(self, key, default=None):
        return "all"

    def __getitem__(self, key):
        return "all"

    def __contains__(self, key):
        return True


@pytest.fixture(scope="module")
def reference():
    sys.path.insert(0, str(PLANE_OPT / "src"))
    from plane_opt.geometry.metrics import CopperGrid
    from plane_opt.inputs import kicad as K

    board = K.load(BOARD)
    roles = _AllRoles()
    _, pads = K.extract_components_and_pads(board, roles)
    tracks = K.extract_tracks(board, roles)
    vias = K.extract_vias(board, roles)
    zones, _ = K.extract_zones(board, roles)
    _, bounds = K.extract_outline(board)
    stackup = K.extract_stackup(board)
    data = {
        "pads": pads, "tracks": tracks, "vias": vias, "zones": zones, "stackup": stackup,
        "target_region": bounds, "target_connections": [], "roles": ["all"],
    }
    grid = CopperGrid(data, PITCH_MM)
    rows, cols = grid.spec.height, grid.spec.width
    masks = {layer: np.zeros((rows, cols), dtype=bool) for layer in grid.layers}
    for (_, layer), cells in grid.masks.items():
        for x, y in cells:
            masks[layer][y, x] = True
    holes = np.zeros((rows, cols), dtype=bool)

    def mark_hole(x: float, y: float, radius: float) -> None:
        reach = radius + 0.5 * PITCH_MM * math.sqrt(2.0)
        for cy in range(max(0, int((y - reach - bounds["min_y_mm"]) / PITCH_MM)), min(rows, int((y + reach - bounds["min_y_mm"]) / PITCH_MM) + 2)):
            for cx in range(max(0, int((x - reach - bounds["min_x_mm"]) / PITCH_MM)), min(cols, int((x + reach - bounds["min_x_mm"]) / PITCH_MM) + 2)):
                if math.hypot(bounds["min_x_mm"] + (cx + 0.5) * PITCH_MM - x, bounds["min_y_mm"] + (cy + 0.5) * PITCH_MM - y) <= reach:
                    holes[cy, cx] = True

    for via in vias:
        mark_hole(via["position"]["x_mm"], via["position"]["y_mm"], via["drill_mm"] / 2.0)
    thru = [pad for pad in pads if pad["type"] == "thru_hole"]
    for pad in thru:
        mark_hole(pad["position"]["x_mm"], pad["position"]["y_mm"], max(pad["drill"]["size_mm"]) / 2.0)
    return {"masks": masks, "holes": holes, "bounds": bounds, "shape": (rows, cols), "vias": len(vias), "thru_pads": len(thru), "copper_layers": stackup["copper_layers"]}


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    step = export_kicad_step(BOARD, tmp_path_factory.mktemp("kicad") / "power_module.step")
    model = load_step(step)
    layers, top = layers_from_kicad_stackup(read_kicad_stackup(BOARD))
    body_map = kicad_step_body_map(layers, board_top_z_mm=top)
    return model, body_map, resolve_bodies(model, body_map)


def _boundary(mask: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(mask)
    edge[1:] |= mask[1:] != mask[:-1]
    edge[:-1] |= mask[:-1] != mask[1:]
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return edge


def test_kicad_export_is_sorted_into_board_copper_and_vias(exported, reference) -> None:
    model, body_map, resolved = exported
    assert body_map.board.layer_names == ("B.Cu", "F.Cu")
    assert resolved.board.volume_m3 == pytest.approx(34.7 * 34.7 * 1.51 * 1e-9, rel=0.2)
    counts = {spec.layer: len(solids) for spec, solids in resolved.copper}
    assert counts["F.Cu"] > counts["B.Cu"] > 0
    barrels = sum(len(solids) for _, solids in resolved.vias)
    assert barrels == reference["vias"] + reference["thru_pads"]
    assert not resolved.ignored


def test_step_copper_agrees_with_plane_opt_away_from_edges_and_drills(exported, reference) -> None:
    """Interior cells agree exactly; plane_opt fills drill holes and rounds edges outwards."""

    _, body_map, resolved = exported
    rows, cols = reference["shape"]
    bounds = reference["bounds"]
    raster = rasterize_board(
        resolved,
        body_map.board,
        pitch_mm=PITCH_MM,
        supersample=1,
        origin_mm=kicad_grid_origin_mm(bounds["min_x_mm"], bounds["min_y_mm"], rows, PITCH_MM),
        shape=(rows, cols),
        y_down=True,
    )
    assert raster.shape == (rows, cols)
    holes = reference["holes"]
    for index, layer in enumerate(raster.layers):
        step = raster.occupancy[index] > 0.0
        ref = reference["masks"][layer.name]
        edge = _boundary(step) | _boundary(ref)
        interior = ~edge & ~holes
        disagree = step != ref
        assert int(np.sum(disagree & interior)) == 0, layer.name
        # plane_opt marks the drill as copper; the export leaves the hole open.
        assert np.all(ref[disagree & holes]) and not np.any(step[disagree & holes])
        boundary_fraction = float(np.sum(disagree & edge & ~holes) / np.sum(edge & ~holes))
        assert boundary_fraction < 0.1, (layer.name, boundary_fraction)
        # plane_opt's inclusive edge rule adds copper; the export never has more.
        assert int(np.sum(step & ~ref & ~holes)) <= 0.01 * np.sum(ref)
        print(f"{layer.name}: step={int(step.sum())} ref={int(ref.sum())} boundary disagreement {boundary_fraction:.3f}")

    vias = board_vias(resolved, raster)
    assert len(vias.vias) == reference["vias"] + reference["thru_pads"]
    assert all(via.layer_from == 0 and via.layer_to == 1 for via in vias.vias)
    thermal = board_thermal_mesh(raster)
    assert thermal.mesh.element_grid_shape == (3, rows, cols)
    assert thermal.mesh.active.mean() > 0.99


def _library_model_present() -> bool:
    from geometry.cad_import import default_kicad_model_dir

    directory = default_kicad_model_dir()
    return directory is not None and (directory / "Capacitor_SMD.3dshapes" / "C_0603_1608Metric.step").is_file()


@pytest.mark.skipif(not _library_model_present(), reason="needs the KiCad 3D model library (C_0603_1608Metric.step)")
def test_component_solids_bind_to_footprints_by_reference(tmp_path) -> None:
    """KiCad names each footprint model by its reference designator and places it at the footprint origin."""

    from geometry.cad_import import default_kicad_model_dir, kicad_component_solids, read_kicad_footprints

    path = export_kicad_step(
        BOARD, tmp_path / "power_module_components.step", model_dir=default_kicad_model_dir(),
        extra_args=("--no-dnp",),
    )
    model = load_step(path)
    footprints = read_kicad_footprints(BOARD)
    components = kicad_component_solids(model)

    # The library holds the 0603 chip models; parts without a resolvable model are skipped by kicad-cli.
    with_library_model = {
        reference for reference, footprint in footprints.items()
        if footprint.footprint in ("Capacitor_SMD:C_0603_1608Metric", "Resistor_SMD:R_0603_1608Metric")
    }
    assert with_library_model and with_library_model <= set(components)
    assert set(components) <= set(footprints)

    laminate = max((s for s in model.solids if s.name.split("/")[1].startswith("=>")), key=lambda s: s.volume_m3)
    board_top_mm = laminate.bounds_m[1][2] / 1e-3
    for reference in sorted(with_library_model):
        lo, hi = model.bounds_m(components[reference])
        centre_mm = (lo + hi) / 2 / 1e-3
        x_mm, y_mm = footprints[reference].step_xy_mm
        assert math.hypot(centre_mm[0] - x_mm, centre_mm[1] - y_mm) < 0.05, reference
        # F.Cu parts stand on the board: above the laminate, within the copper and mask thickness.
        assert footprints[reference].layer == "F.Cu"
        assert board_top_mm <= lo[2] / 1e-3 <= board_top_mm + 0.1, reference


@pytest.mark.skipif(not _library_model_present(), reason="needs the KiCad 3D model library (C_0603_1608Metric.step)")
def test_component_footprints_drive_a_graded_grid_that_keeps_the_copper(tmp_path) -> None:
    from geometry.cad_import import (
        board_refined_grid,
        board_thermal_mesh,
        component_boxes,
        default_kicad_model_dir,
        kicad_component_solids,
        refinement_summary,
    )

    path = export_kicad_step(
        BOARD, tmp_path / "power_module_components.step", model_dir=default_kicad_model_dir(), fuse_shapes=True, extra_args=("--no-dnp",)
    )
    model = load_step(path)
    components = kicad_component_solids(model)
    layers, top = layers_from_kicad_stackup(read_kicad_stackup(BOARD))
    # Component solids are not copper: the body map skips them by reference designator.
    body_map = kicad_step_body_map(layers, board_top_z_mm=top, ignore=tuple(f".*/{ref}/.*" for ref in components))
    resolved = resolve_bodies(model, body_map)
    boxes = component_boxes(components, min_size_m=1.0e-3)
    assert len(boxes) == 11
    grid = board_refined_grid(resolved.board, coarse_pitch_mm=0.5, fine_pitch_mm=0.1, boxes=boxes, margin_mm=1.0)
    summary = refinement_summary(grid, boxes)
    assert summary["cells"] < 0.5 * summary["uniform_fine_cells"]
    graded = rasterize_board(resolved, body_map.board, grid=grid, method="section")
    fine = rasterize_board(resolved, body_map.board, pitch_mm=0.1, method="section")
    from geometry.cad_import import sample_plane_fill

    # With fused shapes no solids overlap, so the exact section coverage carries
    # the same copper area on any grid (unfused pads and tracks overlap, and the
    # per-cell sum-and-clamp then depends slightly on the cell size).
    copper = [solid for _, solids in resolved.copper for solid in solids] + [s for _, solids in resolved.vias for s in solids]
    for layer in body_map.board.layers:
        z = layer.center_z_mm * 1e-3
        on_graded = sample_plane_fill(copper, z_m=z, grid=grid, method="section")
        on_fine = sample_plane_fill(copper, z_m=z, grid=fine.grid, method="section")
        assert float(np.sum(on_graded * grid.cell_area_m2)) == pytest.approx(float(np.sum(on_fine * fine.grid.cell_area_m2)), rel=1.0e-9)
    # The rasters differ only by the edge cells the outline threshold drops.
    for layer in range(len(body_map.board.layers)):
        assert graded.copper_area_m2(layer) == pytest.approx(fine.copper_area_m2(layer), rel=2.0e-2)
    thermal = board_thermal_mesh(graded)
    assert thermal.mesh.element_grid_shape[1:] == grid.shape
