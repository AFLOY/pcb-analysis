"""Sheet PEEC inductance on graded grids by precorrected FFT."""

from __future__ import annotations

import numpy as np
import pytest

from electrical.matrix_free_mpir_fem import TensorGrid, graded_edges
from electrical.sheet_peec.sheet_inductance import (
    closed_form_mutual_inductance_arrays,
    far_field_mutual_inductance,
    CellGeometry,
    closed_form_mutual_inductance,
)
from electrical.sheet_peec.sheet_operator import SheetInductanceOperator, SheetLayer, SheetStackup
from electrical.sheet_peec.sheet_peec import SheetMesh, Terminal, ViaBranch, solve_sheet_case
from electrical.sheet_peec.sheet_pfft import PfftSheetInductanceOperator, lagrange_stencil

STACKUP = SheetStackup((SheetLayer("F", 0.0, 35e-6), SheetLayer("B", -1.6e-3, 35e-6)))


def _dense_reference(mesh: SheetMesh, axis: str) -> np.ndarray:
    """Every in-plane pair of one axis by the exact closed form (near) or the centre limit (far)."""

    cx, cy, length, width = mesh.branch_geometry(axis)
    n = cx.size
    layers = mesh.stackup.layers
    count = len(layers) * n
    matrix = np.zeros((count, count))
    fx, fy, fl, fw = (a.reshape(-1) for a in (cx, cy, length, width))
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    ii, jj = ii.reshape(-1), jj.reshape(-1)
    for a, la in enumerate(layers):
        for b, lb in enumerate(layers):
            dz = lb.z_m - la.z_m
            du = fx[jj] - fx[ii] if axis == "x" else fy[jj] - fy[ii]
            dv = fy[jj] - fy[ii] if axis == "x" else fx[jj] - fx[ii]
            block = closed_form_mutual_inductance_arrays(
                (fl[ii], fw[ii], np.full(ii.size, la.thickness_m)),
                (fl[jj], fw[jj], np.full(jj.size, lb.thickness_m)),
                (du, dv, np.full(ii.size, dz)),
                check_precision=False,
            )
            matrix[a * n : (a + 1) * n, b * n : (b + 1) * n] = block.reshape(n, n)
    return matrix


def _real_branch_mask(mesh: SheetMesh, axis: str) -> np.ndarray:
    rows, cols = mesh.shape
    mask = np.ones((len(mesh.stackup), rows, cols), dtype=bool)
    if axis == "x":
        mask[:, :, -1] = False
    else:
        mask[:, -1, :] = False
    return mask.reshape(-1)


def test_lagrange_stencil_reproduces_polynomials() -> None:
    x = np.array([0.37, 2.9, 7.61])
    for order in (1, 2, 3):
        nodes, weights = lagrange_stencil(x, 0.0, 1.0, 12, order)
        for power in range(order + 1):
            np.testing.assert_allclose(np.sum(weights * nodes**power, axis=1), x**power, atol=1e-12)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)


def test_array_closed_form_matches_the_scalar_one() -> None:
    cell = CellGeometry(0.2e-3, 0.15e-3, 35e-6)
    other = CellGeometry(0.35e-3, 0.1e-3, 70e-6)
    offsets = (np.array([0.0, 0.4e-3, 1.1e-3]), np.array([0.0, 0.2e-3, -0.7e-3]), np.array([0.0, 0.0, 1.6e-3]))
    expected = [closed_form_mutual_inductance(cell, other, (du, dv, dw)) for du, dv, dw in zip(*offsets)]
    actual = closed_form_mutual_inductance_arrays(
        (np.full(3, cell.length_m), np.full(3, cell.width_m), np.full(3, cell.thickness_m)),
        (np.full(3, other.length_m), np.full(3, other.width_m), np.full(3, other.thickness_m)),
        offsets,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_pfft_matches_the_convolution_operator_on_a_uniform_mesh() -> None:
    rows, cols, pitch = 8, 10, 0.2e-3
    mesh = SheetMesh((rows, cols), pitch, STACKUP, np.ones((2, rows, cols), bool))
    reference = SheetInductanceOperator((rows, cols), pitch, STACKUP)
    operator = PfftSheetInductanceOperator(mesh, order=3, near_radius_cells=5)
    for axis in ("x", "y"):
        mask = _real_branch_mask(mesh, axis)
        expected = reference.dense_matrix(axis)[np.ix_(mask, mask)]
        actual = operator.dense_matrix(axis)[np.ix_(mask, mask)]
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 5e-4
        np.testing.assert_allclose(np.diag(actual), np.diag(expected), rtol=1e-12)
        np.testing.assert_allclose(actual, actual.T, rtol=1e-9, atol=1e-16)


def test_pfft_matches_the_direct_pair_sum_on_a_graded_mesh() -> None:
    xe = graded_edges(0.0, 4e-3, coarse_pitch_m=0.5e-3, fine_pitch_m=0.1e-3, refine_m=[(1.5e-3, 2.0e-3)], margin_m=0.2e-3)
    ye = graded_edges(0.0, 2.5e-3, coarse_pitch_m=0.5e-3, fine_pitch_m=0.125e-3, refine_m=[(1.0e-3, 1.3e-3)], margin_m=0.2e-3)
    grid = TensorGrid(xe, ye)
    rows, cols = grid.shape
    mesh = SheetMesh((rows, cols), None, STACKUP, np.ones((2, rows, cols), bool), grid=grid)
    assert not mesh.is_uniform and mesh.pitch_m is None
    operator = PfftSheetInductanceOperator(mesh, order=3, near_radius_cells=4)
    for axis in ("x", "y"):
        mask = _real_branch_mask(mesh, axis)
        expected = _dense_reference(mesh, axis)
        actual = operator.dense_matrix(axis)[np.ix_(mask, mask)]
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 1e-3
        # Near pairs (including the self term) are exact.
        np.testing.assert_allclose(np.diag(actual), np.diag(expected), rtol=1e-12)
    # Resistances follow the bar's aspect ratio.
    _cx, _cy, length, width = mesh.branch_geometry("x")
    sheet = STACKUP.layers[0].sheet_resistance_ohm
    resistances = mesh.resistances()
    np.testing.assert_allclose(resistances[: len(mesh.branch_x)], [sheet * length[r, c] / width[r, c] for _, r, c in mesh.branch_x])


def _strip(grid: TensorGrid, current: float = 1.0):
    rows, cols = grid.shape
    stackup = SheetStackup((SheetLayer("F", 0.0, 35e-6),))
    mesh = SheetMesh((rows, cols), None, stackup, np.ones((1, rows, cols), bool), grid=grid)
    terminals = [
        Terminal("in", 0, tuple((r, 0) for r in range(rows)), current),
        Terminal("out", 0, tuple((r, cols - 1) for r in range(rows)), -current),
    ]
    return mesh, terminals


def test_dc_strip_on_a_graded_grid_has_the_closed_form_resistance() -> None:
    xe = graded_edges(0.0, 20e-3, coarse_pitch_m=0.5e-3, fine_pitch_m=0.1e-3, refine_m=[(8e-3, 9e-3)], margin_m=0.5e-3)
    grid = TensorGrid(xe, np.array([0.0, 0.2e-3]))
    mesh, terminals = _strip(grid)
    operator = PfftSheetInductanceOperator(mesh)
    solution = solve_sheet_case(mesh, operator, terminals, frequency_hz=0.0)
    sheet = mesh.stackup.layers[0].sheet_resistance_ohm
    # Branch centres sit on the cell edges: the first and last half cells carry no branch.
    span = float(xe[-1] - xe[0]) - 0.5 * float(xe[1] - xe[0]) - 0.5 * float(xe[-1] - xe[-2])
    expected = sheet * span / 0.2e-3
    assert solution.converged
    assert solution.voltage_span_v() == pytest.approx(expected, rel=1e-9)


def test_ac_loop_impedance_on_a_graded_grid_agrees_with_the_uniform_fine_grid() -> None:
    """A two-layer strip line (go on F, return on B) at 1 MHz: graded along the length."""

    frequency = 1.0e6
    length, width = 12e-3, 1.0e-3

    def loop(grid: TensorGrid, operator_kind: str):
        rows, cols = grid.shape
        occupancy = np.ones((2, rows, cols), bool)
        vias = tuple(ViaBranch(r, cols - 1, 1, 0, 1e-4) for r in range(rows))
        mesh = SheetMesh((rows, cols), None if not grid.is_uniform else float(grid.pitch_x_m[0]), STACKUP, occupancy, vias=vias, grid=grid)
        if operator_kind == "fft":
            operator = SheetInductanceOperator((rows, cols), float(grid.pitch_x_m[0]), STACKUP, vertical_levels=mesh.vertical_levels)
        else:
            operator = PfftSheetInductanceOperator(mesh, order=3, near_radius_cells=4)
        terminals = [
            Terminal("in", 0, tuple((r, 0) for r in range(rows)), 1.0),
            Terminal("out", 1, tuple((r, 0) for r in range(rows)), -1.0),
        ]
        solution = solve_sheet_case(mesh, operator, terminals, frequency_hz=frequency, tolerance=1e-9)
        assert solution.converged
        loss = float(np.sum(mesh.resistances() * np.abs(solution.branch_current) ** 2))
        return loss, solution

    fine, _ = loop(TensorGrid.uniform(0.1e-3, (10, 120)), "fft")
    coarse, _ = loop(TensorGrid.uniform(0.5e-3, (2, 24)), "fft")
    xe = graded_edges(0.0, length, coarse_pitch_m=0.5e-3, fine_pitch_m=0.1e-3, refine_m=[(0.0, 1.0e-3), (11e-3, 12e-3)], margin_m=0.5e-3)
    graded, _ = loop(TensorGrid(xe, np.linspace(0.0, width, 11)), "pfft")
    assert abs(graded - fine) < abs(coarse - fine)
    assert graded == pytest.approx(fine, rel=5e-2)


def _graded_problem(frequency_hz: float, *, uniform: bool = False) -> dict:
    """A plane_opt v2 problem: a 0.2 mm wide F.Cu trace returning on B.Cu, graded along x."""

    from electrical.sheet_peec.plane_opt_contract import PLANE_OPT_PROBLEM_SCHEMA_V2

    if uniform:
        x_edges = np.linspace(0.0, 6.0, 31)
    else:
        x_edges = graded_edges(0.0, 6e-3, coarse_pitch_m=0.4e-3, fine_pitch_m=0.1e-3, refine_m=[(2.5e-3, 3.5e-3)], margin_m=0.3e-3) * 1e3
    y_edges = np.array([0.0, 0.1, 0.2])
    cols, rows = x_edges.size - 1, y_edges.size - 1
    layers = [
        {"name": "F.Cu", "order": 0, "center_z_mm": 0.0, "thickness_mm": 0.035, "resistivity_ohm_m": 1.724e-8},
        {"name": "B.Cu", "order": 1, "center_z_mm": -1.6, "thickness_mm": 0.035, "resistivity_ohm_m": 1.724e-8},
    ]
    copper = {name: [{"x": x, "y": y} for x in range(cols) for y in range(rows)] for name in ("F.Cu", "B.Cu")}
    connections = [
        {
            "name": f"via_{y}", "cell": {"x": cols - 1, "y": y},
            "segments": [{"upper_layer": "F.Cu", "lower_layer": "B.Cu", "resistance_ohm": 1e-4, "length_mm": 1.6}],
        }
        for y in range(rows)
    ]
    terminals = [
        {"name": "in", "pad": "P1", "current_a": 1.0, "cells": [{"layer": "F.Cu", "x": 0, "y": y} for y in range(rows)]},
        {"name": "out", "pad": "P2", "current_a": -1.0, "cells": [{"layer": "B.Cu", "x": 0, "y": y} for y in range(rows)]},
    ]
    return {
        "schema": PLANE_OPT_PROBLEM_SCHEMA_V2, "name": "graded-trace", "role": "test", "frequency_hz": frequency_hz,
        "grid": {"rows": rows, "columns": cols, "x_edges_mm": x_edges.tolist(), "y_edges_mm": y_edges.tolist()},
        "layers": layers, "copper_by_layer": copper, "vertical_connections": connections, "terminals": terminals,
    }


def test_plane_opt_schema_v2_solves_graded_grids_with_the_pfft_operator() -> None:
    from electrical.sheet_peec.plane_opt_contract import PlaneOptProblem, build_plane_opt_sheet_inputs, solve_plane_opt_problem

    problem = PlaneOptProblem.from_mapping(_graded_problem(1.0e6))
    assert not problem.is_uniform and problem.pitch_mm is None
    mesh, operator, _terminals, _context = build_plane_opt_sheet_inputs(problem)
    assert isinstance(operator, PfftSheetInductanceOperator) and not mesh.is_uniform
    result = solve_plane_opt_problem(problem)
    assert result.metrics["inductance_operator"] == "PfftSheetInductanceOperator"
    assert result.metrics["problem_schema"].endswith("/v2") and result.metrics["grid_uniform"] is False
    assert result.metrics["max_current_density_a_per_mm2"] > 0.0
    # A uniform v2 grid keeps the convolution operator and matches the v1 form.
    uniform = PlaneOptProblem.from_mapping(_graded_problem(1.0e6, uniform=True))
    assert uniform.is_uniform and uniform.pitch_mm == pytest.approx(0.2)
    _mesh, operator_u, _t, _c = build_plane_opt_sheet_inputs(uniform)
    assert isinstance(operator_u, SheetInductanceOperator)
    forced = build_plane_opt_sheet_inputs(uniform, {"operator": "pfft"})[1]
    assert isinstance(forced, PfftSheetInductanceOperator)
    with pytest.raises(ValueError, match="uniform grid"):
        build_plane_opt_sheet_inputs(problem, {"operator": "fft"})


def test_cuda_pfft_solve_matches_the_cpu_solve_on_a_graded_grid() -> None:
    pytest.importorskip("cupy")
    from electrical.matrix_free_mpir_fem import cuda_available
    from electrical.sheet_peec.plane_opt_contract import build_plane_opt_sheet_inputs
    from electrical.sheet_peec.sheet_cuda import CudaPfftSheetInductanceOperator, solve_sheet_case_cuda

    if not cuda_available():
        pytest.skip("no CUDA device")
    mesh, operator, terminals, context = build_plane_opt_sheet_inputs(_graded_problem(1.0e6))
    cpu = solve_sheet_case(mesh, operator, terminals, frequency_hz=1.0e6, tolerance=1e-9)
    # The CUDA path aims its Krylov residual three decades below the public
    # tolerance; 1e-6 keeps that target above the float64 floor.
    gpu, telemetry = solve_sheet_case_cuda(
        mesh, operator, terminals, frequency_hz=1.0e6, tolerance=1e-6, max_iterations=6, restart=60
    )
    assert cpu.converged and gpu.residual < 1e-6
    scale = np.max(np.abs(cpu.branch_current))
    np.testing.assert_allclose(gpu.branch_current, cpu.branch_current, atol=1e-5 * scale)
    import cupy as cp

    device = CudaPfftSheetInductanceOperator(operator, cp)
    rows, cols = mesh.shape
    rng = np.random.default_rng(0)
    layers = len(mesh.stackup)
    gx, gy = rng.standard_normal((layers, rows, cols)), rng.standard_normal((layers, rows, cols))
    gx[:, :, -1] = 0.0
    gy[:, -1, :] = 0.0
    gz = rng.standard_normal((len(mesh.vertical_levels), rows, cols))
    fx, fy, fz = operator.apply(gx, gy, gz)
    dx, dy, dz = device.apply(cp.asarray(gx), cp.asarray(gy), cp.asarray(gz))
    np.testing.assert_allclose(cp.asnumpy(dx), fx, rtol=1e-10, atol=1e-18)
    np.testing.assert_allclose(cp.asnumpy(dy), fy, rtol=1e-10, atol=1e-18)
    np.testing.assert_allclose(cp.asnumpy(dz), fz, rtol=1e-10, atol=1e-18)
