"""STEP front end: named solids, board raster, voxel bodies, contact, end to end."""

from __future__ import annotations

import numpy as np
import pytest

from electrical.dice_peec.plane_opt_contract import PlaneOptProblem
from geometry.step_voxelize import (
    BoardSpec,
    BodyMap,
    BodySpec,
    ContactSpec,
    CopperSpec,
    LayerSpec,
    ViaSpec,
    board_body_contact,
    board_occupancy,
    board_stackup,
    board_thermal_mesh,
    board_vias,
    body_heat_sources,
    body_thermal_mesh,
    box_solid,
    cylinder_solid,
    load_step,
    ocp_available,
    plane_opt_problem_mapping,
    rasterize_board,
    resolve_bodies,
    sample_plane_fill,
    synthetic_model,
    voxelize_bodies,
    write_step,
)
from multiphysics.staggered_coupling import (
    BoardEnclosureThermalScenario,
    BodyContact,
    run_board_enclosure_thermal,
)
from thermal.matrix_free_mpir_fem import (
    ExposedFaceConvection,
    ThermalConductionProblem,
    VoxelMaterial,
)

pytestmark = pytest.mark.skipif(not ocp_available(), reason="OCP not installed; pip install 'pcb-analysis[cad]'")

MM = 1.0e-3
CU = 35.0e-6
BOARD = (20.0 * MM, 12.0 * MM, 1.6 * MM)


def _solids():
    """A 20 x 12 x 1.6 mm board, copper on both faces, one via, a sink, a cap."""

    return [
        box_solid("PCB", (0.0, 0.0, 0.0), BOARD),
        # Bottom copper: full plane minus nothing (a 18 x 10 mm rectangle).
        box_solid("Cu_B1_plane", (1.0 * MM, 1.0 * MM, -CU), (18.0 * MM, 10.0 * MM, CU)),
        # Top copper: a 2 mm wide trace and a 6 x 6 mm pad under the sink.
        box_solid("Cu_F1_trace", (1.0 * MM, 5.0 * MM, BOARD[2]), (6.0 * MM, 2.0 * MM, CU)),
        box_solid("Cu_F1_pad", (7.0 * MM, 3.0 * MM, BOARD[2]), (6.0 * MM, 6.0 * MM, CU)),
        cylinder_solid("VIA1", (2.0 * MM, 6.0 * MM), -CU, BOARD[2] + 2 * CU, 0.3 * MM),
        box_solid("HeatSink", (7.0 * MM, 3.0 * MM, BOARD[2] + CU), (6.0 * MM, 6.0 * MM, 4.0 * MM)),
        box_solid("C1", (15.0 * MM, 2.0 * MM, BOARD[2] + CU), (2.0 * MM, 1.2 * MM, 1.0 * MM)),
    ]


def _body_map() -> BodyMap:
    return BodyMap(
        board=BoardSpec(
            "PCB",
            (
                LayerSpec("B1", center_z_mm=-0.0175, thickness_mm=0.035),
                LayerSpec("F1", center_z_mm=1.6175, thickness_mm=0.035),
            ),
        ),
        copper=(CopperSpec(r"Cu_B1_.*", "B1"), CopperSpec(r"Cu_F1_.*", "F1")),
        vias=(ViaSpec(r"VIA\d+"),),
        bodies=(
            BodySpec("HeatSink", VoxelMaterial("aluminium", 200.0, role="sink"), contact=ContactSpec("top", 1.0e4)),
            BodySpec("C1", VoxelMaterial("package", 5.0, role="component"), power_w=0.5, contact=ContactSpec("top", 2.0e3)),
        ),
    )


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = write_step(tmp_path_factory.mktemp("step") / "board.step", _solids())
    return load_step(path)


def test_step_round_trip_keeps_names_bounds_and_volumes(model) -> None:
    assert set(model.names) == {"PCB", "Cu_B1_plane", "Cu_F1_trace", "Cu_F1_pad", "VIA1", "HeatSink", "C1"}
    board = model.solid("PCB")
    np.testing.assert_allclose(board.size_m, BOARD, rtol=1.0e-9, atol=1.0e-9)
    assert board.volume_m3 == pytest.approx(20.0 * 12.0 * 1.6 * MM**3, rel=1.0e-9)
    via = model.solid("VIA1")
    assert via.volume_m3 == pytest.approx(np.pi * (0.3 * MM) ** 2 * (BOARD[2] + 2 * CU), rel=1.0e-6)
    inside = board.contains(np.array([[1.0 * MM, 1.0 * MM, 0.8 * MM], [1.0 * MM, 1.0 * MM, 2.0 * MM]]))
    assert inside.tolist() == [True, False]
    assert model.matching(r"Cu_F1_.*") == (model.solid("Cu_F1_trace"), model.solid("Cu_F1_pad"))


def test_body_map_resolution_is_exhaustive(model) -> None:
    resolved = resolve_bodies(model, _body_map())
    assert resolved.board.name == "PCB"
    assert {spec.layer: tuple(s.name for s in solids) for spec, solids in resolved.copper} == {
        "B1": ("Cu_B1_plane",),
        "F1": ("Cu_F1_trace", "Cu_F1_pad"),
    }
    incomplete = BodyMap(board=_body_map().board, copper=_body_map().copper)
    with pytest.raises(ValueError, match="not covered by the body map"):
        resolve_bodies(model, incomplete)
    double = BodyMap(
        board=_body_map().board,
        copper=_body_map().copper,
        vias=_body_map().vias,
        bodies=_body_map().bodies + (BodySpec("C1", VoxelMaterial("again", 1.0)),),
    )
    with pytest.raises(ValueError, match="claimed by both"):
        resolve_bodies(model, double)


@pytest.mark.parametrize("pitch_mm,supersample,tolerance", [(1.0, 1, 0.0), (0.5, 3, 0.0), (0.4, 3, 0.06)])
def test_rasterised_copper_area_converges(model, pitch_mm, supersample, tolerance) -> None:
    """Acceptance 1: copper area of rectangles against the analytic value."""

    resolved = resolve_bodies(model, _body_map())
    raster = rasterize_board(resolved, _body_map().board, pitch_mm=pitch_mm, supersample=supersample)
    plane = 18.0 * 10.0 * MM**2
    top = (6.0 * 2.0 + 6.0 * 6.0) * MM**2
    # Rectangles aligned with a 1 mm or 0.5 mm grid are exact.  At 0.4 mm the
    # edges fall mid-cell and a boundary sample counts as inside, so three
    # samples per axis quantise a half-covered cell to 2/3: a bias of one
    # sixth of a cell per edge, about 5 % on these small rectangles.
    assert raster.copper_area_m2(0) == pytest.approx(plane, rel=max(tolerance, 1.0e-9))
    assert raster.copper_area_m2(1) == pytest.approx(top, rel=max(tolerance, 1.0e-9))
    assert raster.outline.all()
    occupancy = board_occupancy(raster)
    assert occupancy.shape == (2,) + raster.shape
    if pitch_mm == 1.0:
        assert occupancy[1, 5:7, 1:7].all() and occupancy[1, 3:9, 7:13].all()
        assert occupancy[1].sum() == 12 + 36


def test_disc_area_error_falls_with_supersampling(model) -> None:
    via = model.solid("VIA1")
    radius = 0.3 * MM
    errors = []
    for supersample in (1, 2, 4):
        fill = sample_plane_fill(
            [via], z_m=0.8 * MM, origin_m=(1.5 * MM, 5.5 * MM), pitch_m=0.1 * MM, shape=(10, 10), supersample=supersample
        )
        errors.append(abs(float(fill.sum()) * (0.1 * MM) ** 2 - np.pi * radius**2) / (np.pi * radius**2))
    assert errors[-1] < 0.02
    assert errors[-1] <= errors[0]
    print(f"disc area error by supersample 1/2/4: {errors}")


def test_stackup_vias_thermal_mesh_and_plane_opt_mapping(model) -> None:
    body_map = _body_map()
    resolved = resolve_bodies(model, body_map)
    raster = rasterize_board(resolved, body_map.board, pitch_mm=0.5)
    stackup = board_stackup(raster)
    assert stackup.layer_names == ("B1", "F1") and stackup.z_mm == (-0.0175, 1.6175)

    vias = board_vias(resolved, raster)
    assert len(vias.vias) == 1
    via = vias.vias[0]
    assert (via.row, via.col, via.layer_from, via.layer_to) == (12, 4, 0, 1)
    barrel = np.pi * (0.3 * MM) ** 2
    assert via.resistance_ohm == pytest.approx(1.68e-8 * 1.635 * MM / barrel, rel=1.0e-6)

    thermal = board_thermal_mesh(raster)
    assert thermal.slab_names == ("B1", "laminate:B1-F1", "F1")
    assert thermal.layer_slabs == (0, 2)
    mesh = thermal.mesh
    assert mesh.element_grid_shape == (3, 24, 40)
    np.testing.assert_allclose(mesh.slab_thickness_m, (35.0e-6, 1.6e-3, 35.0e-6))
    assert mesh.conductivity_w_per_m_k[2, 10, 20] == pytest.approx(385.0)  # under the pad
    assert mesh.conductivity_w_per_m_k[2, 0, 0] == pytest.approx(0.8)  # bare laminate
    assert mesh.is_full

    mapping = plane_opt_problem_mapping(
        raster,
        terminals=[
            {"name": "src", "pad": "P1", "current_a": 1.0, "cells": [{"layer": "F1", "x": 2, "y": 11}]},
            {"name": "ret", "pad": "P2", "current_a": -1.0, "cells": [{"layer": "B1", "x": 30, "y": 11}]},
        ],
        vias=vias,
    )
    problem = PlaneOptProblem.from_mapping(mapping)
    assert problem.rows == 24 and problem.columns == 40 and problem.pitch_mm == 0.5
    assert len(problem.copper_by_layer["F1"]) == int(raster.occupancy[1].sum())
    assert len(problem.vertical_segments) == 1


def test_voxelised_sink_matches_its_volume_and_couples_to_the_board(model) -> None:
    """Acceptance 5 (synthetic): STEP → raster + voxels → contact → coupled solve."""

    body_map = _body_map()
    resolved = resolve_bodies(model, body_map)
    raster = rasterize_board(resolved, body_map.board, pitch_mm=0.5)
    board_model = board_thermal_mesh(raster)

    sink_spec = body_map.bodies[0]
    sink_model, specs = voxelize_bodies(resolved, pitch_m=(0.5 * MM, 0.5 * MM, 1.0 * MM), bodies=[sink_spec])
    assert sink_model.shape == (4, 12, 12)
    assert sink_model.volume_m3() == pytest.approx(6.0 * 6.0 * 4.0 * MM**3, rel=1.0e-9)
    assert sink_model.origin_m == pytest.approx((7.0 * MM, 3.0 * MM, BOARD[2] + CU))
    sink_mesh = body_thermal_mesh(sink_model)
    assert sink_mesh.is_full
    assert body_heat_sources(sink_model, specs, sink_mesh) == ()

    contact = board_body_contact(raster, board_model.mesh, sink_model, sink_mesh, sink_spec.contact)
    assert contact.size == 144
    assert float(np.sum(contact.area_m2)) == pytest.approx(36.0 * MM**2)
    # The sink stands on rows 6..17, cols 14..25 of the 0.5 mm board grid.
    assert contact.board_cells[:, 0].min() == 6 and contact.board_cells[:, 0].max() == 17
    assert contact.board_cells[:, 1].min() == 14 and contact.board_cells[:, 1].max() == 25

    heat = np.zeros(board_model.mesh.element_grid_shape)
    heat[2][raster.occupancy[1] > 0.0] = 1.0 / float(raster.occupancy[1].sum())  # 1 W in the top copper
    board = ThermalConductionProblem(
        board_model.mesh,
        convection=(ExposedFaceConvection(10.0, 300.0, directions=("-z", "+z", "-x", "+x", "-y", "+y")),),
        element_heat_w=heat,
    )
    sink = ThermalConductionProblem(sink_mesh, convection=(ExposedFaceConvection(25.0, 300.0),))
    result = run_board_enclosure_thermal(BoardEnclosureThermalScenario(board, (BodyContact(sink, contact, "sink"),)))
    assert result.converged
    assert 0.0 < result.interface_heat_w < 1.0
    assert result.board.max_temperature_k > result.bodies[0].max_temperature_k > 300.0
    print(
        f"coupled STEP case: {result.iterations} interface iterations, "
        f"board max {result.board.max_temperature_k:.2f} K, sink max {result.bodies[0].max_temperature_k:.2f} K, "
        f"interface {result.interface_heat_w:.3f} W"
    )

    component = body_map.bodies[1]
    cap_model, cap_specs = voxelize_bodies(resolved, pitch_m=(0.2 * MM, 0.2 * MM, 0.25 * MM), bodies=[component])
    cap_mesh = body_thermal_mesh(cap_model)
    sources = body_heat_sources(cap_model, cap_specs, cap_mesh)
    assert len(sources) == 1 and sources[0].power_w == 0.5
    assert len(sources[0].nodes) == cap_mesh.active_nodes.sum()


def test_synthetic_model_needs_no_file() -> None:
    model = synthetic_model(_solids()[:2])
    assert model.names == ("PCB", "Cu_B1_plane")
    with pytest.raises(KeyError):
        model.solid("nope")
