"""Edge strands (``edge_strands`` setting): the edge branches cut across their width."""

from __future__ import annotations

import numpy as np
import pytest

from electrical.sheet_peec import current_field_contract as cfc
from electrical.sheet_peec.edge_strands import graded_widths, lagrange_across
from electrical.sheet_peec.near_factor import mumps_available

T_MM, Z_F, Z_B, RHO = 0.07112, -0.43306, -1.91642, 1.724e-8


def _strip_pair(width_mm: float, length_mm: float, pitch_mm: float, frequency_hz: float) -> dict:
    """The go/return strip pair of work/skin_ref: F.Cu out, B.Cu back, joined across the far end."""
    cols, rows = int(round(length_mm / pitch_mm)), int(round(width_mm / pitch_mm))
    cells = [{"x": x, "y": y} for x in range(cols) for y in range(rows)]
    layers = [
        {"name": name, "order": k, "center_z_mm": z, "thickness_mm": T_MM, "resistivity_ohm_m": RHO,
         "thickness_source": "board_stackup"}
        for k, (name, z) in enumerate((("F.Cu", Z_F), ("B.Cu", Z_B)))
    ]
    vias = [
        {"name": f"end_{y}", "cell": {"x": cols - 1, "y": y}, "layers": ["F.Cu", "B.Cu"],
         "segments": [{"upper_layer": "F.Cu", "lower_layer": "B.Cu", "resistance_ohm": 1e-7, "length_mm": Z_F - Z_B}]}
        for y in range(rows)
    ]
    terminals = [
        {"name": "src", "pad": "SRC.1", "current_a": {"real": 1.0, "imag": 0.0},
         "cells": [{"layer": "F.Cu", "x": 0, "y": y} for y in range(rows)]},
        {"name": "snk", "pad": "SNK.1", "current_a": {"real": -1.0, "imag": 0.0},
         "cells": [{"layer": "B.Cu", "x": 0, "y": y} for y in range(rows)]},
    ]
    return {"schema": cfc.CURRENT_FIELD_PROBLEM_SCHEMA, "name": "strip_pair", "role": "LOOP", "state": "steady",
            "weight": 1.0, "frequency_hz": frequency_hz,
            "grid": {"min_x_mm": 0.0, "min_y_mm": 0.0, "columns": cols, "rows": rows, "pitch_mm": pitch_mm},
            "layers": layers, "copper_by_layer": {"F.Cu": cells, "B.Cu": cells}, "vertical_connections": vias,
            "terminals": terminals}


def _loop_impedance(value: dict, settings: dict) -> complex:
    result = cfc.solve_current_field_problem(value, settings)
    v = result.voltage_phasor

    def mean(term: dict) -> complex:
        return complex(np.mean([v[(c["layer"], c["x"], c["y"])] for c in term["cells"]]))

    return mean(value["terminals"][0]) - mean(value["terminals"][1])


def test_graded_widths_cover_the_branch_and_grade_toward_the_open_sides() -> None:
    pitch = 100e-6
    for open_minus, open_plus in ((True, False), (False, True), (True, True)):
        strands = graded_widths(pitch, open_minus, open_plus, 6e-6, 2.0)
        widths = np.array([w for _, w in strands])
        centres = np.array([c for c, _ in strands])
        assert widths.sum() == pytest.approx(pitch)
        # Contiguous and inside the branch.
        edges = np.sort(np.concatenate([centres - widths / 2, centres + widths / 2]))
        assert edges[0] == pytest.approx(-pitch / 2) and edges[-1] == pytest.approx(pitch / 2)
        assert widths.min() == pytest.approx(6e-6)
        if open_plus and not open_minus:
            assert centres[np.argmin(widths)] > 0
    assert graded_widths(pitch, False, False, 6e-6, 2.0) == [(0.0, pitch)]
    assert graded_widths(pitch, True, True, 60e-6, 2.0) == [(0.0, pitch)]


def test_the_projection_keeps_a_strand_s_first_and_second_moments() -> None:
    offsets = np.array([-0.47, -0.2, 0.0, 0.31, 0.49])
    for position, size in ((5, 10), (0, 10), (9, 10)):
        nodes, weights = lagrange_across(offsets, np.full(offsets.size, position), size)
        assert np.all(position + nodes >= 0) and np.all(position + nodes < size)
        assert np.allclose(weights.sum(axis=1), 1.0)
        assert np.allclose((weights * nodes).sum(axis=1), offsets)
        assert np.allclose((weights * nodes**2).sum(axis=1), offsets**2)


def _stranded(frequency_hz: float = 3e7):
    problem = cfc.CurrentFieldProblem.from_mapping(_strip_pair(1.2, 4.0, 0.2, frequency_hz))
    mesh, operator, _ = cfc._sheet_geometry(problem, {"maximum_filaments": 64, "edge_strands": True})
    return mesh, operator


def test_a_stranded_mesh_conducts_like_its_mesh_at_dc_and_its_grid_maps_are_adjoint() -> None:
    mesh, _ = _stranded()
    strands = mesh.strands
    assert strands.extra_count > 0
    base = strands.base_count
    resistance = mesh.resistances()
    # The strands of a branch in parallel are the branch: one square of its layer.
    conductance = np.zeros(base)
    np.add.at(conductance, strands.branch, 1.0 / resistance[strands.strand_index])
    cut = strands.cut_branches
    first = np.searchsorted(strands.branch, cut)
    sheet = np.array([mesh.stackup.layers[layer].sheet_resistance_ohm for layer in strands.cell[first, 0]])
    assert np.allclose(1.0 / conductance[cut], sheet, rtol=1e-12)
    rows = mesh.incidence()
    assert rows.shape == (mesh.branch_count, mesh.node_count)
    extra = np.arange(base, mesh.branch_count)
    assert (rows[extra] != rows[strands.extra_parent]).nnz == 0
    rng = np.random.default_rng(1)
    currents = rng.standard_normal(mesh.branch_count)
    grid_x, grid_y = mesh.scatter(currents)
    probe_x, probe_y = rng.standard_normal(grid_x.shape), rng.standard_normal(grid_y.shape)
    assert np.sum(probe_x * grid_x) + np.sum(probe_y * grid_y) == pytest.approx(
        float(mesh.gather(probe_x, probe_y) @ currents), rel=1e-12
    )
    # What a cell carries is what its strands carry.
    folded = mesh.fold(currents)
    assert folded.shape == (base,)
    assert np.sum(folded) == pytest.approx(np.sum(currents))


def test_the_correction_and_the_exact_near_field_are_symmetric() -> None:
    mesh, _ = _stranded()
    for matrix in (mesh.correction_matrix(), mesh.near_exact_matrix()):
        difference = abs(matrix - matrix.T).max()
        assert difference <= 1e-12 * abs(matrix).max()
    # Every strand's own exact inductance is on the diagonal and positive.
    diagonal = mesh.near_exact_matrix().diagonal()[mesh.strands.strand_index]
    assert np.all(diagonal > 0)


@pytest.mark.skipif(not mumps_available(), reason="the strip solves take the MUMPS near factor")
def test_edge_strands_bring_a_narrow_strip_s_resistance_to_the_fem() -> None:
    """1.2 mm strip pair at 30 MHz, 0.4 mm cells: the 2D FEM gives 1.802 mOhm/mm (work/skin_ref).

    Without strands this mesh gives 1.248 (-31 %); with them 1.810.
    """
    settings = {"maximum_filaments": 64, "relative_tolerance": 1e-9, "preconditioner": "near",
                "near_factorization": "mumps_quasi_definite"}
    per_length = {}
    for strands in (False, True):
        z = [
            _loop_impedance(_strip_pair(1.2, length, 0.4, 3e7), dict(settings, edge_strands=strands))
            for length in (10.0, 20.0)
        ]
        per_length[strands] = (z[1] - z[0]) / 10.0
    plain, cut = per_length[False].real * 1e3, per_length[True].real * 1e3
    assert plain == pytest.approx(1.248, rel=0.01)
    assert cut == pytest.approx(1.802, rel=0.03)
    # And the loop inductance: 0.665 nH/mm without, 0.630 with, the FEM's 0.630.
    inductance = per_length[True].imag / (2 * np.pi * 3e7) * 1e9
    assert inductance == pytest.approx(0.630, rel=0.01)


def test_edge_strands_are_refused_where_they_are_not_built() -> None:
    mesh, operator = _stranded()
    from electrical.sheet_peec.sheet_peec import solve_sheet_cases

    with pytest.raises(ValueError, match="near preconditioner"):
        solve_sheet_cases(mesh, operator, [], frequency_hz=3e7, preconditioner="block")


@pytest.mark.skipif(not __import__("electrical.backend", fromlist=["core"]).core(), reason="needs the C++ core")
def test_the_compact_correction_is_the_explicit_one() -> None:
    """Kept as blocks by layout, layer pair and offset, applied by the core: the same C, a fraction of the memory."""
    from electrical.sheet_peec import edge_strands as es

    problem = cfc.CurrentFieldProblem.from_mapping(_strip_pair(1.2, 4.0, 0.2, 3e7))
    base, operator, _ = cfc._sheet_geometry(problem, {"maximum_filaments": 64})
    mesh = es.StrandedSheetMesh.from_mesh(base, es.plan_edge_strands(base, 6e-6))
    explicit, exact = es.strand_correction(mesh, operator)
    compact = es.compact_strand_correction(mesh, operator)
    x = np.random.default_rng(2).standard_normal(mesh.branch_count)
    assert np.allclose(compact @ x, explicit @ x, rtol=0, atol=1e-10 * np.abs(explicit @ x).max())
    for built, reference in ((compact.matrix(False), explicit), (compact.matrix(True), exact)):
        assert built.nnz == reference.nnz
        assert abs(built - reference).max() <= 1e-10 * abs(reference).max()
    stored = sum(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes for m in (explicit, exact))
    assert compact.stored_bytes < stored / 5  # 8x on this short strip, 16x at 1.2 x 20 mm, 0.4 mm


@pytest.mark.skipif(not __import__("electrical.backend", fromlist=["core"]).core(), reason="needs the C++ core")
def test_the_compact_correction_gives_the_same_bits_at_any_thread_count() -> None:
    """docs/DESIGN.md: the same bits whatever the thread budget (the mirrored adds run in a fixed order)."""
    from electrical.sheet_peec import edge_strands as es

    problem = cfc.CurrentFieldProblem.from_mapping(_strip_pair(1.2, 4.0, 0.2, 3e7))
    base, operator, _ = cfc._sheet_geometry(problem, {"maximum_filaments": 64})
    mesh = es.StrandedSheetMesh.from_mesh(base, es.plan_edge_strands(base, 6e-6))
    compact = es.compact_strand_correction(mesh, operator)
    x = np.random.default_rng(5).standard_normal(mesh.branch_count)
    results = [compact._native.apply(x, threads) for threads in (1, 2, 3, 8)]
    for other in results[1:]:
        assert other.tobytes() == results[0].tobytes()


def test_edge_strands_are_cut_by_default_where_they_are_built() -> None:
    """``edge_strands`` "auto": cut above zero frequency, not at DC nor without the near preconditioner."""
    def applied(frequency: float, **settings) -> bool:
        problem = cfc.CurrentFieldProblem.from_mapping(_strip_pair(1.2, 4.0, 0.2, frequency))
        _mesh, _operator, geometry = cfc._sheet_geometry(problem, {"maximum_filaments": 1, **settings})
        return geometry.edge_strands

    assert applied(3e7)
    assert not applied(0.0)
    assert not applied(3e7, preconditioner="block")
    assert not applied(3e7, preconditioner="diagonal")
    assert applied(3e7, execution_backend="cuda")  # the device cuts them as the CPU does
    assert not applied(3e7, edge_strands=False)
    result = cfc.solve_current_field_problem(_strip_pair(1.2, 4.0, 0.2, 3e7), {"maximum_filaments": 1})
    assert result.metrics["edge_strands"] is True and result.metrics["face_coverage"] is False
    with pytest.raises(ValueError, match="edge_strands"):
        applied(3e7, edge_strands="yes")


def test_a_neighbour_of_another_region_is_an_edge_for_the_strands() -> None:
    """Two nets a clearance narrower than a cell apart: their touching rows are edges."""
    def strands_on_row(row: int, regions: bool) -> int:
        cells = [{"x": x, "y": y} for x in range(10) for y in range(4)]
        value = {
            "schema": cfc.CURRENT_FIELD_PROBLEM_SCHEMA, "name": "two_nets", "role": "A", "state": "steady",
            "weight": 1.0, "frequency_hz": 3e7,
            "grid": {"min_x_mm": 0.0, "min_y_mm": 0.0, "columns": 10, "rows": 4, "pitch_mm": 0.2},
            "layers": [{"name": "F.Cu", "order": 0, "center_z_mm": 0.0, "thickness_mm": 0.035,
                        "resistivity_ohm_m": 1.724e-8}],
            "copper_by_layer": {"F.Cu": cells}, "vertical_connections": [],
            "terminals": [
                {"name": "in", "pad": "IN.1", "current_a": {"real": 1.0, "imag": 0.0},
                 "cells": [{"layer": "F.Cu", "x": 0, "y": y} for y in (0, 1)]},
                {"name": "out", "pad": "OUT.1", "current_a": {"real": -1.0, "imag": 0.0},
                 "cells": [{"layer": "F.Cu", "x": 9, "y": y} for y in (0, 1)]},
            ],
        }
        if regions:
            value["cell_regions"] = {"F.Cu": [{"x": c["x"], "y": c["y"], "region": 1 if c["y"] < 2 else 2} for c in cells]}
        problem = cfc.CurrentFieldProblem.from_mapping(value)
        mesh, _operator, _geometry = cfc._sheet_geometry(problem, {"maximum_filaments": 1, "edge_strands": True})
        rows = {position: r for position, (_layer, r, _col) in enumerate(mesh.branch_x)}
        cut = mesh.strands.branch
        return int(np.count_nonzero([rows.get(int(p)) == row for p in cut]))

    assert strands_on_row(1, regions=False) == 0  # inside one net: no edge
    assert strands_on_row(1, regions=True) > 0  # beside the other net: an edge


def test_the_device_flux_of_a_stranded_mesh_is_the_host_s() -> None:
    """The CUDA path's product on a stranded mesh, NumPy standing in for CuPy: the host's flux.

    The device spreads the strands over the grid and reads them back with two
    sparse matrices and adds the precorrection as a third; the host does the
    same through the mesh's scatter and gather and the core's blocks.
    """
    import scipy.sparse as sp

    from electrical.sheet_peec.sheet_cuda import (
        CudaSheetInductanceOperator, _branch_indices, _device_flux, _strand_projection, strand_projection_matrices,
    )

    mesh, operator = _stranded()
    assert mesh.via_branches and mesh.strands.extra_count > 0
    rng = np.random.default_rng(5)
    # the maps as matrices are the mesh's scatter and gather
    scatter_x, scatter_y = strand_projection_matrices(mesh)
    currents = rng.standard_normal(mesh.branch_count)
    grid_x, grid_y = mesh.scatter(currents)
    assert np.allclose(scatter_x @ currents, grid_x.reshape(-1), rtol=0, atol=1e-13)
    assert np.allclose(scatter_y @ currents, grid_y.reshape(-1), rtol=0, atol=1e-13)
    probe_x, probe_y = rng.standard_normal(grid_x.shape), rng.standard_normal(grid_y.shape)
    gathered = scatter_x.T @ probe_x.reshape(-1) + scatter_y.T @ probe_y.reshape(-1)
    assert np.allclose(gathered, mesh.gather(probe_x, probe_y), rtol=0, atol=1e-13)

    def host_flux(full: np.ndarray) -> np.ndarray:
        gx, gy = mesh.scatter(full)
        fx, fy, fz = operator.apply(gx, gy, mesh.scatter_vertical(full))
        flux = mesh.gather(fx, fy)
        mesh.gather_vertical(fz, flux)
        return flux + mesh.inductance_correction @ full

    device = CudaSheetInductanceOperator(operator, np)
    projection = _strand_projection(mesh, np, sp, True)
    assert projection is not None and projection.correction is not None
    active = np.sort(rng.choice(mesh.branch_count, size=mesh.branch_count - 7, replace=False))
    batch = rng.standard_normal((2, active.size))
    got = _device_flux(np, mesh, device, _branch_indices(mesh, np), projection, active, batch)
    for b in range(2):
        full = np.zeros(mesh.branch_count)
        full[active] = batch[b]
        want = host_flux(full)[active]
        assert np.allclose(got[b], want, rtol=0, atol=1e-10 * np.abs(want).max())
