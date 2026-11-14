"""Face coverage: sheet branches sized by the copper on the face they cross.

On a 45-degree edge the grid's staircase leaves the cells that touch only at a
corner unjoined, and a mesh of whole cells overstated a 1.2 mm strip's DC
resistance by 6 % at every pitch from 0.1 mm down.  ``rasterize_board(faces=True)``
records how much of every grid face the copper covers; the contract mapping
lists the partly covered ones and the sheet mesh narrows those branches.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse.csgraph as csgraph

geometry = pytest.importorskip("geometry.cad_import")
from geometry.cad_import import mesh as mesh_module  # noqa: E402

pytestmark = pytest.mark.skipif(not mesh_module.native_available(), reason="needs the geometry native extension")

T_MM, Z_F, RHO = 0.07112, -0.43306, 1.724e-8
S2 = math.sqrt(2.0)


def _prism(polygon: np.ndarray, z0: float, z1: float) -> np.ndarray:
    """Outward triangles of a convex prism, polygon counter-clockwise."""
    p, triangles = np.asarray(polygon, float), []
    for i in range(1, len(p) - 1):
        triangles.append([[*p[0], z1], [*p[i], z1], [*p[i + 1], z1]])
        triangles.append([[*p[0], z0], [*p[i + 1], z0], [*p[i], z0]])
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        triangles += [[[*a, z0], [*b, z0], [*b, z1]], [[*a, z0], [*b, z1], [*a, z1]]]
    return np.asarray(triangles)


def test_face_coverage_is_the_exact_covered_length_and_centroid() -> None:
    shapely = pytest.importorskip("shapely.geometry")
    rng = np.random.default_rng(0)
    pitch, rows, cols = 0.1, 24, 24
    for _ in range(8):
        centre, angle = rng.uniform(0.8, 1.6, 2), rng.uniform(0, np.pi)
        width, length = rng.uniform(0.2, 0.8), rng.uniform(0.4, 1.4)
        u = np.array([np.cos(angle), np.sin(angle)])
        v = np.array([-u[1], u[0]])
        polygon = [centre - u * length / 2 - v * width / 2, centre + u * length / 2 - v * width / 2,
                   centre + u * length / 2 + v * width / 2, centre - u * length / 2 + v * width / 2]
        triangles = _prism(polygon, -1e-4, 1e-4)
        vertical, vertical_offset, horizontal, horizontal_offset, _, _ = (
            np.asarray(a) for a in mesh_module._native.plane_face_coverage([triangles], 0.0, 0.0, 0.0, pitch, rows, cols)
        )
        shape = shapely.Polygon(polygon)
        for r in range(rows):
            for c in range(1, cols):
                piece = shapely.LineString([(c * pitch, r * pitch), (c * pitch, (r + 1) * pitch)]).intersection(shape)
                assert vertical[r, c] == pytest.approx(piece.length / pitch, abs=1e-12)
                if piece.length > 1e-9:
                    assert vertical_offset[r, c] == pytest.approx((piece.centroid.y - (r + 0.5) * pitch) / pitch, abs=1e-9)
        for r in range(1, rows):
            for c in range(cols):
                piece = shapely.LineString([(c * pitch, r * pitch), ((c + 1) * pitch, r * pitch)]).intersection(shape)
                assert horizontal[r, c] == pytest.approx(piece.length / pitch, abs=1e-12)
                if piece.length > 1e-9:
                    assert horizontal_offset[r, c] == pytest.approx((piece.centroid.x - (c + 0.5) * pitch) / pitch, abs=1e-9)


def _resolved(strips: list[tuple[float, float, float, float]], margin: float, reach: float):
    """``strips`` as (x0, y0, length, width) boxes along x, turned by ``angle`` about the origin."""
    reader = pytest.importorskip("geometry.cad_import.reader")
    if not reader.ocp_available():
        pytest.skip("needs OpenCASCADE")
    from geometry.cad_import.bodymap import BoardSpec, CopperSpec, LayerSpec, ResolvedBodies

    spec = BoardSpec(solid="board", layers=(LayerSpec("F.Cu", Z_F, T_MM, RHO),))
    board = reader.box_solid("board", (-margin * 1e-3, -margin * 1e-3, -1.0e-3), ((reach + 2 * margin) * 1e-3,) * 2 + (0.9e-3,))
    return reader, spec, board


def _strip_solid(reader, name: str, x0: float, y0: float, length: float, width: float, angle: float, index: int):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf

    box = reader.box_solid(name, (x0 * 1e-3, (y0 - width / 2) * 1e-3, (Z_F - T_MM / 2) * 1e-3),
                           (length * 1e-3, width * 1e-3, T_MM * 1e-3), index=index)
    if angle == 0.0:
        return box
    turn = gp_Trsf()
    turn.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), angle)
    return reader._solid_from_shape(name, index, BRepBuilderAPI_Transform(box._shape, turn, True).Shape())


def _raster(solids, spec, board, margin: float, reach: float, pitch: float, faces: bool):
    from geometry.cad_import import rasterize_board
    from geometry.cad_import.bodymap import CopperSpec, ResolvedBodies

    resolved = ResolvedBodies(board=board, copper=((CopperSpec("F", "F.Cu"), tuple(solids)),), vias=(), bodies=())
    size = int(math.ceil((reach + 2 * margin) / pitch))
    return rasterize_board(resolved, spec, pitch_mm=pitch, origin_mm=(-margin, -margin), shape=(size, size),
                           method="section", faces=faces)


def _dc_resistance_per_mm(pitch: float, faces: bool) -> float:
    """A 1.2 mm strip at 45 degrees, two lengths, end bands driven: R per mm along the strip."""
    from electrical.sheet_peec.current_field_contract import solve_current_field_problem
    from geometry.cad_import import current_field_problem_mapping

    values = []
    # Whole numbers of diagonal cell steps, so the staircase ends of the two
    # lengths are alike and cancel in the difference.
    lengths = (50 * pitch * S2, 100 * pitch * S2)
    for length in lengths:
        margin, reach = 3 * pitch + 1.2, length / S2 + 1.2
        reader, spec, board = _resolved([], margin, reach)
        strip = _strip_solid(reader, "F", 0.0, 0.0, length, 1.2, math.pi / 4, 1)
        raster = _raster([strip], spec, board, margin, reach, pitch, faces)
        value = current_field_problem_mapping(raster, terminals=(), name="diag", role="LOOP")
        cells = [(c["x"], c["y"]) for c in value["copper_by_layer"]["F.Cu"]]
        along = {cell: ((cell[0] + 0.5) * pitch - margin + (cell[1] + 0.5) * pitch - margin) / S2 for cell in cells}
        start = [cell for cell, a in along.items() if a < pitch * S2]
        end = [cell for cell, a in along.items() if a > length - pitch * S2]
        value["terminals"] = [
            {"name": "in", "pad": "IN.1", "current_a": {"real": 1.0, "imag": 0.0},
             "cells": [{"layer": "F.Cu", "x": x, "y": y} for x, y in start]},
            {"name": "out", "pad": "OUT.1", "current_a": {"real": -1.0, "imag": 0.0},
             "cells": [{"layer": "F.Cu", "x": x, "y": y} for x, y in end]},
        ]
        result = solve_current_field_problem(value, {})
        v = result.voltage_phasor
        drop = np.mean([v[("F.Cu", x, y)] for x, y in start]) - np.mean([v[("F.Cu", x, y)] for x, y in end])
        values.append(float(np.real(drop)))
    return (values[1] - values[0]) / (lengths[1] - lengths[0])


def test_a_45_degree_strip_has_its_dc_resistance_with_face_coverage() -> None:
    exact = RHO / (1.2e-3 * T_MM * 1e-3) * 1e-3  # ohm per mm
    whole_cells = _dc_resistance_per_mm(0.1, faces=False)
    covered = _dc_resistance_per_mm(0.1, faces=True)
    assert whole_cells > exact * 1.03  # the staircase's dead teeth
    assert covered == pytest.approx(exact, rel=0.005)  # -0.3 %: the coverage is quantised to 1/64 of a cell


def test_a_clearance_narrower_than_a_cell_is_not_bridged() -> None:
    """Two strips of one raster 0.05 mm apart on a 0.1 mm grid stay two conductors with face coverage.

    Whole cells at the 0.5 area threshold join them: the cell row the gap
    runs through is half covered (0.3 by one strip, 0.2 by the other).  With
    face coverage that cell's copper is found to be two pieces and is left
    out.
    """
    from electrical.sheet_peec import current_field_contract as cfc
    from geometry.cad_import import current_field_problem_mapping

    pitch, margin, reach = 0.1, 0.5, 3.0
    reader, spec, board = _resolved([], margin, reach)
    # The gap 0.53 .. 0.58 mm lies inside the cell row 0.5 .. 0.6 mm.
    lower = _strip_solid(reader, "A", 0.0, 0.25, 3.0, 0.56, 0.0, 1)
    upper = _strip_solid(reader, "B", 0.0, 0.88, 3.0, 0.60, 0.0, 2)
    for faces in (True, False):
        raster = _raster([lower, upper], spec, board, margin, reach, pitch, faces)
        if faces:
            with pytest.warns(UserWarning, match="two separate pieces"):
                value = current_field_problem_mapping(raster, terminals=(), name="pair", role="LOOP")
        else:
            value = current_field_problem_mapping(raster, terminals=(), name="pair", role="LOOP")
        first, last = value["copper_by_layer"]["F.Cu"][0], value["copper_by_layer"]["F.Cu"][-1]
        value["terminals"] = [{"name": f"t{k}", "pad": f"T.{k}", "current_a": {"real": 1.0 - 2.0 * k, "imag": 0.0},
                               "cells": [{"layer": "F.Cu", "x": cell["x"], "y": cell["y"]}]} for k, cell in enumerate((first, last))]
        problem = cfc.CurrentFieldProblem.from_mapping(value)
        mesh, _, _ = cfc._sheet_geometry(problem, {})
        count, _ = csgraph.connected_components(mesh.incidence().T @ mesh.incidence(), directed=False)
        assert count == (2 if faces else 1)


def test_the_faces_are_recorded_by_default_on_a_uniform_grid() -> None:
    """``rasterize_board`` records the faces unless asked not to (``faces="auto"`` on a uniform grid)."""
    from geometry.cad_import import rasterize_board
    from geometry.cad_import.bodymap import CopperSpec, ResolvedBodies

    pitch, margin, reach = 0.1, 1.5, 2.0
    reader, spec, board = _resolved([], margin, reach)
    strip = _strip_solid(reader, "F", 0.0, 0.0, 1.0, 1.2, math.pi / 4, 1)
    resolved = ResolvedBodies(board=board, copper=((CopperSpec("F", "F.Cu"), (strip,)),), vias=(), bodies=())
    size = int(math.ceil((reach + 2 * margin) / pitch))
    raster = rasterize_board(resolved, spec, pitch_mm=pitch, origin_mm=(-margin, -margin), shape=(size, size),
                             method="section")
    assert raster.faces is not None
    off = rasterize_board(resolved, spec, pitch_mm=pitch, origin_mm=(-margin, -margin), shape=(size, size),
                          method="section", faces=False)
    assert off.faces is None
