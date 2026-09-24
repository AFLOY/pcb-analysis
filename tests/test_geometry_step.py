"""STEP front end: named solids, board raster, voxel bodies, contact, end to end."""

from __future__ import annotations

import numpy as np
import pytest

from electrical.sheet_peec.current_field_contract import CurrentFieldProblem
from geometry.cad_import import (
    native_classify_available,
    BoardRaster,
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
    board_barrels,
    board_vias,
    body_heat_sources,
    body_thermal_mesh,
    box_solid,
    cylinder_solid,
    drilled_solid,
    load_step,
    ocp_available,
    current_field_problem_mapping,
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


def test_stackup_vias_thermal_mesh_and_current_field_mapping(model) -> None:
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

    mapping = current_field_problem_mapping(
        raster,
        terminals=[
            {"name": "src", "pad": "P1", "current_a": 1.0, "cells": [{"layer": "F1", "x": 2, "y": 11}]},
            {"name": "ret", "pad": "P2", "current_a": -1.0, "cells": [{"layer": "B1", "x": 30, "y": 11}]},
        ],
        vias=vias,
    )
    problem = CurrentFieldProblem.from_mapping(mapping)
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


def test_measured_thickness_and_skin_screening(model) -> None:
    import warnings

    from geometry.cad_import import skin_report

    body_map = _body_map()
    resolved = resolve_bodies(model, body_map)
    raster = rasterize_board(resolved, body_map.board, pitch_mm=0.5)
    # The fixture's copper is 35 um thick, as the stackup says.
    assert raster.measured_thickness_mm == pytest.approx((0.035, 0.035), rel=1.0e-6)
    assert raster.thickness_mismatches() == ()
    assert raster.layer_thickness_mm(1, source="measured") == pytest.approx(0.035)

    # A stackup that claims 70 um gets a warning and the measured value on request.
    thick = BoardSpec(
        "PCB",
        (LayerSpec("B1", center_z_mm=-0.0175, thickness_mm=0.035), LayerSpec("F1", center_z_mm=1.6175, thickness_mm=0.070)),
    )
    wrong_map = BodyMap(board=thick, copper=body_map.copper, vias=body_map.vias, bodies=body_map.bodies)
    with pytest.warns(UserWarning, match="F1: the copper solids are 0.0350 mm thick"):
        wrong = rasterize_board(resolve_bodies(model, wrong_map), thick, pitch_mm=0.5)
    assert wrong.thickness_mismatches() == (("F1", 0.070, pytest.approx(0.035)),)
    mapping = current_field_problem_mapping(
        wrong,
        terminals=[
            {"name": "src", "pad": "P1", "current_a": 1.0, "cells": [{"layer": "F1", "x": 2, "y": 11}]},
            {"name": "ret", "pad": "P2", "current_a": -1.0, "cells": [{"layer": "B1", "x": 30, "y": 11}]},
        ],
        thickness_source="measured",
    )
    assert mapping["layers"][1]["thickness_mm"] == pytest.approx(0.035)
    thermal = board_thermal_mesh(wrong, thickness_source="measured")
    assert thermal.mesh.slab_thickness_m[2] == pytest.approx(35.0e-6)
    assert board_thermal_mesh(wrong).mesh.slab_thickness_m[2] == pytest.approx(70.0e-6)

    # Skin screening: 35 um copper is uniform at 1 MHz (delta = 65 um), filaments at 100 MHz.
    low = skin_report(raster, 1.0e6)
    assert [item.classification for item in low.layers] == ["uniform", "uniform"]
    assert low.layers[0].skin_depth_mm == pytest.approx(0.0652, rel=2.0e-2)
    high = skin_report(raster, 100.0e6)
    assert high.needs_filaments == ("B1", "F1") and high.needs_3d == ()
    dc = skin_report(raster, 0.0)
    assert all(item.classification == "uniform" for item in dc.layers)
    # A 2 mm busbar layer is a 3D body, and the mapping says so.
    busbar = BoardSpec("PCB", (LayerSpec("B1", center_z_mm=-0.0175, thickness_mm=0.035), LayerSpec("F1", center_z_mm=2.6, thickness_mm=2.0)))
    bus_raster = BoardRaster(busbar, raster.pitch_mm, raster.origin_mm, raster.outline, raster.fill)
    assert skin_report(bus_raster, 1.0e6).needs_3d == ("F1",)
    with pytest.warns(UserWarning, match="a 2.5D sheet cannot represent it"):
        current_field_problem_mapping(
            bus_raster,
            terminals=[
                {"name": "src", "pad": "P1", "current_a": 1.0, "cells": [{"layer": "F1", "x": 2, "y": 11}]},
                {"name": "ret", "pad": "P2", "current_a": -1.0, "cells": [{"layer": "B1", "x": 30, "y": 11}]},
            ],
            frequency_hz=1.0e6,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        current_field_problem_mapping(
            raster,
            terminals=[
                {"name": "src", "pad": "P1", "current_a": 1.0, "cells": [{"layer": "F1", "x": 2, "y": 11}]},
                {"name": "ret", "pad": "P2", "current_a": -1.0, "cells": [{"layer": "B1", "x": 30, "y": 11}]},
            ],
            frequency_hz=100.0e6,
        )


def test_thick_conductor_goes_to_the_3d_voxel_peec() -> None:
    pypeec = pytest.importorskip("pypeec")
    from electrical.voxel_peec import solve_voxel_peec
    from geometry.cad_import import TerminalRegion, conductor_heat_w, conductor_problem_from_solids
    from thermal.matrix_free_mpir_fem import ExposedFaceConvection, ThermalConductionProblem, solve_thermal_conduction

    del pypeec
    bar = box_solid("busbar", (0.0, 0.0, 0.0), (20.0 * MM, 4.0 * MM, 2.0 * MM))
    pitch = (0.5 * MM, 0.5 * MM, 0.5 * MM)
    problem, model = conductor_problem_from_solids(
        [bar],
        [
            TerminalRegion("src", box_m=((-1e-9, -1e-9, -1e-9), (1.0 * MM, 5.0 * MM, 3.0 * MM)), current_a=20.0),
            TerminalRegion("ref", box_m=((19.0 * MM, -1e-9, -1e-9), (21.0 * MM, 5.0 * MM, 3.0 * MM))),
        ],
        pitch_m=pitch,
        name="busbar",
    )
    assert problem.shape == (4, 8, 40) and problem.conductor.all()
    assert problem.terminals[0].voxels.sum() == 2 * 8 * 4
    solution = solve_voxel_peec(problem)
    assert solution.converged
    analytic = 1.68e-8 * 19.0 * MM / (4.0 * MM * 2.0 * MM)  # between the terminal midplanes
    assert solution.impedance_ohm["src"].real == pytest.approx(analytic, rel=2.0e-2)

    # The same grid carries the thermal solve with the Joule heat as source.
    heat = conductor_heat_w(solution, model)
    assert heat.sum() == pytest.approx(solution.joule_loss_w)
    mesh = body_thermal_mesh(model)
    thermal = solve_thermal_conduction(
        ThermalConductionProblem(mesh, convection=(ExposedFaceConvection(15.0, 300.0),), element_heat_w=heat)
    )
    assert thermal.solve.converged
    assert thermal.total_heat_input_w == pytest.approx(solution.joule_loss_w)
    assert thermal.max_temperature_k > 300.0
    # Voxels coarser than the skin depth cannot resolve the AC profile: warn.
    with pytest.warns(UserWarning, match="exceeds the skin depth"):
        conductor_problem_from_solids(
            [bar],
            [TerminalRegion("src", box_m=((-1e-9, -1e-9, -1e-9), (1.0 * MM, 5.0 * MM, 3.0 * MM)), current_a=1.0),
             TerminalRegion("ref", box_m=((19.0 * MM, -1e-9, -1e-9), (21.0 * MM, 5.0 * MM, 3.0 * MM)))],
            pitch_m=pitch,
            frequency_hz=1.0e6,
        )


def test_nested_assembly_components_keep_their_placement(tmp_path) -> None:
    """A solid two assembly levels down lands where the component puts it.

    KiCad places every footprint model as ``board/<refdes>/<model>``; the
    component label carries the placement and the solid sits below a
    prototype assembly, so the flattener must accumulate the transforms.
    """

    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Trsf, gp_Vec
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopLoc import TopLoc_Location
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    def name(label, text):
        TDataStd_Name.Set_s(label, TCollection_ExtendedString(text))

    def shift(x, y, z):
        trsf = gp_Trsf()
        trsf.SetTranslation(gp_Vec(x, y, z))
        return TopLoc_Location(trsf)

    document = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    body = tool.NewShape()
    name(body, "model")
    inner = tool.AddComponent(body, tool.AddShape(BRepPrimAPI_MakeBox(1.6, 0.8, 0.8).Shape(), False), shift(0.0, 0.0, 0.5))
    name(inner, "body")
    root = tool.NewShape()
    name(root, "board")
    for reference, (x, y) in {"C1": (10.0, -20.0), "C2": (30.0, -5.0)}.items():
        name(tool.AddComponent(root, body, shift(x, y, 1.6)), reference)
    tool.UpdateAssemblies()
    writer = STEPCAFControl_Writer()
    writer.Transfer(document, STEPControl_AsIs)
    path = tmp_path / "nested.step"
    assert writer.Write(str(path)) == IFSelect_RetDone

    model = load_step(path)
    assert sorted(model.names) == ["board/C1/body", "board/C2/body"]
    for reference, (x, y) in {"C1": (10.0, -20.0), "C2": (30.0, -5.0)}.items():
        lo, hi = np.asarray(model.solid(f"board/{reference}/body").bounds_m) / 1e-3
        assert np.allclose(lo, (x, y, 2.1), atol=1e-6)
        assert np.allclose(hi, (x + 1.6, y + 0.8, 2.9), atol=1e-6)


def test_graded_raster_conserves_copper_area_and_builds_a_graded_thermal_mesh(model) -> None:
    if not native_classify_available():
        pytest.skip("the exact section rasteriser needs the geometry native extension")
    """The same board on a graded grid: exact areas per layer, per-cell pitches in the thermal mesh."""

    from electrical.matrix_free_mpir_fem import TensorGrid
    from geometry.cad_import import RefinementBox, board_refined_grid, board_thermal_mesh, refinement_summary

    body_map = _body_map()
    resolved = resolve_bodies(model, body_map)
    uniform = rasterize_board(resolved, body_map.board, pitch_mm=0.25, method="section")
    box = RefinementBox("hot", 6.0 * MM, 9.0 * MM, 2.0 * MM, 5.0 * MM)
    grid = board_refined_grid(resolved.board, coarse_pitch_mm=0.5, fine_pitch_mm=0.1, boxes=[box], margin_mm=0.5)
    graded = rasterize_board(resolved, body_map.board, grid=grid, method="section")
    assert isinstance(graded.grid, TensorGrid) and not graded.is_uniform and graded.pitch_mm is None
    summary = refinement_summary(grid, [box])
    assert summary["cells"] < summary["uniform_fine_cells"] and summary["cells"] > summary["uniform_coarse_cells"]
    # The exact section coverage carries the same copper area on both grids;
    # the raster's own area differs only by edge cells the outline threshold drops.
    copper = [solid for _, solids in resolved.copper for solid in solids]
    for layer in body_map.board.layers:
        z = layer.center_z_mm * MM
        on_graded = sample_plane_fill(copper, z_m=z, grid=grid, method="section")
        on_uniform = sample_plane_fill(copper, z_m=z, grid=uniform.grid, method="section")
        assert float(np.sum(on_graded * grid.cell_area_m2)) == pytest.approx(
            float(np.sum(on_uniform * uniform.grid.cell_area_m2)), rel=1.0e-9
        )
    assert float(np.sum(graded.outline * graded.cell_area_m2)) == pytest.approx(
        float(np.sum(uniform.outline * uniform.cell_area_m2)), rel=2.0e-2
    )
    with pytest.raises(ValueError, match="graded"):
        graded.pitch_m
    thermal = board_thermal_mesh(graded)
    assert thermal.mesh.element_grid_shape[1:] == grid.shape
    np.testing.assert_array_equal(thermal.mesh.pitch_x_m, grid.pitch_x_m)
    np.testing.assert_array_equal(thermal.mesh.pitch_y_m, grid.pitch_y_m)
    mapping = current_field_problem_mapping(graded, terminals=())
    assert mapping["schema"].endswith("/v2") and "pitch_mm" not in mapping["grid"]
    assert len(mapping["grid"]["x_edges_mm"]) == grid.shape[1] + 1
    # y-down storage reverses the row heights with the rows.
    flipped = rasterize_board(resolved, body_map.board, grid=grid, method="section", y_down=True)
    np.testing.assert_array_equal(flipped.pitch_y_m, grid.pitch_y_m[::-1])
    np.testing.assert_array_equal(flipped.fill, graded.fill[:, ::-1])
    assert flipped.row_y_m(0) == pytest.approx(float(grid.y_centres_m[-1]))


# --------------------------------------------------- plated holes are conductor
#
# Two boards, both 10 x 10 mm and two layers, both joined only through one
# plated hole, and both drilled: the laminate really has the hole cut out of
# it, and the plating really is a tube, which is what a mechanical export
# gives.  One is a 0.3 mm via with a 25 um wall; the other is a through-hole
# pad, a 1.0 mm drill with a pad annulus on each face.  At 0.1 mm the wall of
# either is a quarter of a cell and the bore falls between the samples, so
# neither can be found by sampling its material.
CENTRE = (5.0 * MM, 5.0 * MM)
PLATING = 0.025 * MM
BOARD_MM = 10.0


def _plated_board(drill_radius_m: float, *, pad_radius_m: float | None = None):
    """A drilled two-layer board whose only vertical path is the barrel."""

    outer = drill_radius_m
    slab = box_solid("slab", (0.0, 0.0, 0.0), (BOARD_MM * MM, BOARD_MM * MM, 1.6 * MM))
    drill = cylinder_solid("drill", CENTRE, -CU, 1.6 * MM + 2 * CU, outer)
    bore = cylinder_solid("bore", CENTRE, -CU, 1.6 * MM + 2 * CU, outer - PLATING)
    barrel = drilled_solid(
        "VIA1", cylinder_solid("outer", CENTRE, -CU, 1.6 * MM + 2 * CU, outer), [bore]
    )
    solids = [
        drilled_solid("PCB", slab, [drill]),
        # A run on each layer, reaching the hole from opposite sides.
        box_solid("Cu_B1_run", (0.5 * MM, 4.5 * MM, -CU), (4.5 * MM, 1.0 * MM, CU)),
        box_solid("Cu_F1_run", (5.0 * MM, 4.5 * MM, 1.6 * MM), (4.5 * MM, 1.0 * MM, CU)),
        barrel,
    ]
    if pad_radius_m is not None:
        # A through-hole pad: an annulus on each face, drilled like the board.
        for name, z in (("Cu_B1_pad", -CU), ("Cu_F1_pad", 1.6 * MM)):
            solids.append(
                drilled_solid(
                    name,
                    cylinder_solid(f"{name}_disc", CENTRE, z, CU, pad_radius_m),
                    [cylinder_solid(f"{name}_hole", CENTRE, z - CU, 3 * CU, outer)],
                )
            )
    return solids


def _plated_body_map() -> BodyMap:
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
    )


def _solve_through(raster, vias, barrels, row):
    """One vertical connection, its geometry recorded, and a closing solve."""

    from electrical.sheet_peec.current_field_contract import solve_current_field_problem

    mapping = current_field_problem_mapping(
        raster,
        terminals=[
            {"name": "src", "pad": "P1", "current_a": 1.0, "cells": [{"layer": "F1", "x": 92, "y": row}]},
            {"name": "ret", "pad": "P2", "current_a": -1.0, "cells": [{"layer": "B1", "x": 7, "y": row}]},
        ],
        vias=vias,
        barrels=barrels,
    )
    problem = CurrentFieldProblem.from_mapping(mapping)
    assert len(problem.vertical_segments) == 1
    result = solve_current_field_problem(problem)
    assert result.metrics["converged"]
    assert result.metrics["undriven_node_count"] == 0
    assert result.metrics["current_closure_error_a"] < 1.0e-9
    assert result.metrics["voltage_span_v"] > 0.0
    return mapping["vertical_connections"][0]


@pytest.mark.parametrize(
    "drill_mm,pad_mm,name",
    [(0.3, None, "via"), (1.0, 1.6, "through-hole pad")],
)
def test_a_plated_barrel_is_a_vertical_conductor_with_its_geometry(drill_mm, pad_mm, name) -> None:
    """A tube 25 um thick joins the layers, and the problem carries the tube.

    Sampling the plating alone finds nothing at the barrel's own cell -- the
    cell the connection attaches to -- and the drilled laminate would mask off
    whatever it did find.  The barrel is read from its own geometry instead,
    which covers a through-hole pad's barrel exactly as it covers a via's.
    """

    body_map = _plated_body_map()
    drill_radius = drill_mm / 2.0 * MM
    solids = _plated_board(drill_radius, pad_radius_m=None if pad_mm is None else pad_mm / 2.0 * MM)
    resolved = resolve_bodies(synthetic_model(solids), body_map)
    raster = rasterize_board(resolved, body_map.board, pitch_mm=0.1, supersample=3)

    vias = board_vias(resolved, raster)
    assert len(vias.vias) == 1
    via = vias.vias[0]
    assert (via.layer_from, via.layer_to) == (0, 1)

    # The plating, sampled as material, is not there at the barrel's cell.
    sampled = sample_plane_fill(
        [solid for solid in solids if solid.name == "VIA1"],
        z_m=-0.0175 * MM,
        origin_m=raster.origin_m,
        pitch_m=0.1 * MM,
        shape=raster.shape,
    )
    assert sampled[via.row, via.col] < 0.5, f"{name}: the wall should be sub-cell"

    # The barrel is, on both layers, and the outline did not mask it away.
    assert raster.outline[via.row, via.col]
    occupancy = raster.occupancy
    assert occupancy[0, via.row, via.col] == 1.0
    assert occupancy[1, via.row, via.col] == 1.0
    # And only the barrel: a cell away from copper and hole is still outside.
    assert occupancy[0, 5, 5] == 0.0 and occupancy[1, 5, 5] == 0.0

    barrels = board_barrels(resolved, raster)
    assert set(barrels) == {(via.row, via.col)}
    connection = _solve_through(raster, vias, barrels, via.row)

    recorded = connection["barrel"]
    assert recorded["outer_diameter_mm"] == pytest.approx(drill_mm, rel=1.0e-3)
    assert recorded["drill_diameter_mm"] == pytest.approx(drill_mm - 2 * 0.025, rel=1.0e-2)
    assert recorded["plating_thickness_mm"] == pytest.approx(0.025, rel=1.0e-2)
    assert recorded["z_range_mm"] == pytest.approx([-0.035, 1.635], abs=1.0e-6)
    ring = np.pi * ((drill_mm / 2.0) ** 2 - (drill_mm / 2.0 - 0.025) ** 2)
    assert recorded["wall_area_mm2"] == pytest.approx(ring, rel=1.0e-2)
    assert not recorded["solid_pin"]


def test_a_solid_that_is_not_a_barrel_is_left_to_the_sampler() -> None:
    """Recognition is narrow: only a tube or a pin around its own axis."""

    from geometry.cad_import import barrel_of

    slab = box_solid("PCB", (0.0, 0.0, 0.0), (BOARD_MM * MM, BOARD_MM * MM, 1.6 * MM))
    assert barrel_of(slab) is None
    # A square prism whose footprint is square but whose section is not a disc.
    prism = box_solid("pin", (4.5 * MM, 4.5 * MM, 0.0), (1.0 * MM, 1.0 * MM, 1.6 * MM))
    assert barrel_of(prism) is None
    # Two copper sheets joined off-axis: a square bounding box, a small area,
    # and nothing around the axis -- a fused net, not a barrel.
    fused = drilled_solid(
        "fused",
        box_solid("outer", (0.0, 0.0, 0.0), (4.0 * MM, 4.0 * MM, 1.6 * MM)),
        [box_solid("cut", (0.0, 0.0, 0.1 * MM), (4.0 * MM, 3.0 * MM, 1.4 * MM))],
    )
    assert barrel_of(fused) is None
    # A solid pin is a barrel, and says so.
    pin = cylinder_solid("PIN", CENTRE, 0.0, 1.6 * MM, 0.3 * MM)
    found = barrel_of(pin)
    assert found is not None and found.solid_pin
    assert found.drill_radius_m == pytest.approx(0.0, abs=1.0e-9)
