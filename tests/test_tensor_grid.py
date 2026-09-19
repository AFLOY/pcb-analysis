"""Graded tensor grids in the thermal and electrical DC solvers."""

from __future__ import annotations

import numpy as np
import pytest

from electrical.matrix_free_mpir_fem import (
    CurrentTerminal,
    LayeredPCBMesh,
    MPIRConfig,
    PCBConductionProblem,
    TensorGrid,
    cuda_available,
    graded_edges,
    refined_grid,
    solve_pcb_dc,
)
from electrical.matrix_free_mpir_fem.pcb import MatrixFreePCBOperator, _local_stiffness
from thermal.matrix_free_mpir_fem import (
    ConvectionBoundary,
    ExposedFaceConvection,
    LayeredThermalMesh,
    MatrixFreeThermalOperator,
    ThermalConductionProblem,
    planar_contact_map,
    solve_thermal_conduction,
)
from thermal.matrix_free_mpir_fem.mesh import _local_hexahedron_matrices
from thermal.matrix_free_mpir_fem.native_hex import native_available

AMBIENT = 300.0


def test_graded_edges_are_fine_where_asked_and_grow_gently() -> None:
    edges = graded_edges(0.0, 60e-3, coarse_pitch_m=0.5e-3, fine_pitch_m=0.1e-3, refine_m=[(20e-3, 24e-3)], margin_m=1e-3, growth=1.5)
    widths = np.diff(edges)
    assert edges[0] == 0.0 and edges[-1] == 60e-3
    assert np.all(widths > 0.0)
    inside = widths[(edges[:-1] >= 19e-3) & (edges[1:] <= 25e-3)]
    assert inside.size >= 55 and np.allclose(inside, 0.1e-3, rtol=2e-3)
    assert widths.max() <= 0.5e-3 * 1.01
    ratio = np.maximum(widths[1:] / widths[:-1], widths[:-1] / widths[1:])
    assert ratio.max() < 1.75
    assert widths.size < 600 / 3  # far fewer cells than the uniform fine grid
    plain = graded_edges(0.0, 10e-3, coarse_pitch_m=0.5e-3, fine_pitch_m=0.1e-3)
    np.testing.assert_allclose(plain, np.linspace(0.0, 10e-3, 21))

    grid = refined_grid((0.0, 60e-3), (0.0, 40e-3), coarse_pitch_m=0.5e-3, fine_pitch_m=0.1e-3, refine_boxes_m=[(20e-3, 24e-3, 10e-3, 14e-3)], margin_m=1e-3)
    assert grid.size < 0.15 * 600 * 400 and not grid.is_uniform
    uniform = TensorGrid.uniform(0.5e-3, (4, 6), origin_m=(1.0, 2.0))
    assert uniform.is_uniform and uniform.shape == (4, 6) and uniform.origin_m == (1.0, 2.0)
    np.testing.assert_allclose(uniform.cell_area_m2, 0.25e-6)
    row, col = grid.cell_of(np.array([21e-3]), np.array([11e-3]))
    assert grid.x_edges_m[col[0]] <= 21e-3 < grid.x_edges_m[col[0] + 1]
    assert grid.cell_mask(20e-3, 24e-3, 10e-3, 14e-3).sum() == pytest.approx(40 * 40, rel=0.1)
    with pytest.raises(ValueError, match="increasing"):
        TensorGrid([0.0, 1.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="fine_pitch_m"):
        graded_edges(0.0, 1.0, coarse_pitch_m=0.1, fine_pitch_m=0.2)


def _thermal_mesh(pitch_x, pitch_y, shape=None) -> LayeredThermalMesh:
    return LayeredThermalMesh(
        (35e-6, 1.5e-3, 35e-6), pitch_x, pitch_y, (385.0, 0.8, 385.0), element_shape=shape,
        through_plane_conductivity_w_per_m_k=(385.0, 0.3, 385.0),
    )


def test_uniform_arrays_reproduce_the_scalar_pitch_operator_and_unit_matrices() -> None:
    rows, cols, pitch = 6, 9, 0.4e-3
    scalar = _thermal_mesh(pitch, pitch, (rows, cols))
    arrays = _thermal_mesh(np.full(cols, pitch), np.full(rows, pitch))
    assert scalar.uniform_pitch and arrays.element_grid_shape == scalar.element_grid_shape
    problem_a = ThermalConductionProblem(scalar, convection=(ConvectionBoundary("top", 10.0, AMBIENT),))
    problem_b = ThermalConductionProblem(arrays, convection=(ConvectionBoundary("top", 10.0, AMBIENT),))
    op_a, op_b = MatrixFreeThermalOperator(problem_a), MatrixFreeThermalOperator(problem_b)
    vector = np.random.default_rng(1).standard_normal(op_a.size)
    np.testing.assert_array_equal(op_a.apply_high(vector), op_b.apply_high(vector))
    # The three unit matrices with the element coefficients equal the old per-slab matrices.
    in_plane, through = _local_hexahedron_matrices(pitch, pitch, scalar.slab_thickness_m)
    ax, ay, az = scalar.element_coefficients()
    ux, uy, uz = op_a._unit_high
    for slab in range(3):
        rebuilt = (ax[slab, 0, 0] * ux + ay[slab, 0, 0] * uy) / 385.0 * (1.0 if slab != 1 else 385.0 / 0.8)
        expected = in_plane[slab]
        np.testing.assert_allclose(rebuilt, expected, rtol=1e-12)
        np.testing.assert_allclose(az[slab, 0, 0] * uz, through[slab] * (385.0 if slab != 1 else 0.3), rtol=1e-12)


def test_graded_thermal_grid_matches_the_uniform_fine_grid_at_a_hot_spot() -> None:
    """A 2 x 2 mm heated patch on a 30 x 20 mm board: graded grid against uniform 0.1 mm."""

    coarse, fine = 0.5e-3, 0.125e-3
    patch = (14e-3, 16e-3, 9e-3, 11e-3)

    def solve(grid: TensorGrid):
        mesh = _thermal_mesh(grid.pitch_x_m, grid.pitch_y_m)
        heat = np.zeros(mesh.element_grid_shape)
        # 1 W over exactly the patch area on every grid, by cell overlap.
        overlap = grid.cell_overlap_fraction(*patch) * grid.cell_area_m2
        assert np.sum(overlap) == pytest.approx(4e-6, rel=1e-9)
        heat[2] = overlap / np.sum(overlap)
        problem = ThermalConductionProblem(
            mesh, convection=(ConvectionBoundary("top", 10.0, AMBIENT), ConvectionBoundary("bottom", 10.0, AMBIENT)),
            element_heat_w=heat,
        )
        solution = solve_thermal_conduction(problem)
        assert solution.solve.converged and abs(solution.heat_balance_error_w) < 1e-9
        return grid, solution

    fine_grid, fine_solution = solve(TensorGrid.uniform(fine, (160, 240)))
    coarse_grid, coarse_solution = solve(TensorGrid.uniform(coarse, (40, 60)))
    graded_grid, graded_solution = solve(
        refined_grid((0.0, 30e-3), (0.0, 20e-3), coarse_pitch_m=coarse, fine_pitch_m=fine, refine_boxes_m=[patch], margin_m=1e-3, growth=1.4)
    )
    rise = fine_solution.max_temperature_k - AMBIENT
    coarse_error = abs(coarse_solution.max_temperature_k - fine_solution.max_temperature_k)
    graded_error = abs(graded_solution.max_temperature_k - fine_solution.max_temperature_k)
    assert rise > 5.0
    assert graded_grid.size < 0.2 * fine_grid.size
    assert graded_error < coarse_error
    assert graded_error < 1e-3 * rise
    print(f"rise={rise:.2f} K coarse_err={coarse_error:.3f} K graded_err={graded_error:.4f} K cells fine={fine_grid.size} graded={graded_grid.size} coarse={coarse_grid.size}")


def test_graded_grid_low_paths_match_the_host_operator() -> None:
    grid = refined_grid((0.0, 12e-3), (0.0, 8e-3), coarse_pitch_m=1e-3, fine_pitch_m=0.25e-3, refine_boxes_m=[(5e-3, 7e-3, 3e-3, 5e-3)], margin_m=0.5e-3)
    mesh = _thermal_mesh(grid.pitch_x_m, grid.pitch_y_m)
    heat = np.zeros(mesh.element_grid_shape)
    heat[2][grid.cell_mask(5e-3, 7e-3, 3e-3, 5e-3)] = 0.01
    problem = ThermalConductionProblem(mesh, convection=(ExposedFaceConvection(10.0, AMBIENT),), element_heat_w=heat)
    host = MatrixFreeThermalOperator(problem)
    vector = np.random.default_rng(2).standard_normal(host.size)
    expected = host.apply_high(vector)
    generic = host.runtime.to_host(host.apply_low(host.runtime.from_host(vector)))
    scale = np.max(np.abs(expected))
    np.testing.assert_allclose(generic, expected, atol=2e-5 * scale)
    if native_available():
        native = MatrixFreeThermalOperator(problem, native=True)
        assert native.low_operator_backend.startswith("cpp")
        np.testing.assert_allclose(native.apply_low(vector.astype(np.float32)), expected, atol=2e-5 * scale)
        solved = solve_thermal_conduction(problem, native=True)
        plain = solve_thermal_conduction(problem)
        np.testing.assert_allclose(solved.temperature_k, plain.temperature_k, atol=1e-6)
    if cuda_available():
        gpu = MatrixFreeThermalOperator(problem, backend="cuda")
        assert gpu.low_operator_backend.startswith("cuda")
        actual = gpu.runtime.to_host(gpu.apply_low(gpu.runtime.from_host(vector)))
        np.testing.assert_allclose(actual, expected, atol=2e-5 * scale)


def test_contact_map_intersects_graded_cells_by_area() -> None:
    board = _thermal_mesh(graded_edges(0.0, 6e-3, coarse_pitch_m=1e-3, fine_pitch_m=0.25e-3, refine_m=[(2e-3, 3e-3)])[1:] - graded_edges(0.0, 6e-3, coarse_pitch_m=1e-3, fine_pitch_m=0.25e-3, refine_m=[(2e-3, 3e-3)])[:-1], np.full(4, 1e-3))
    body = LayeredThermalMesh((1e-3,), 0.5e-3, 0.5e-3, 200.0, element_shape=(4, 6))
    contact = planar_contact_map(board, body, board_side="top", body_origin_m=(1.5e-3, 1e-3), conductance_per_area_w_per_m2_k=1e4)
    assert float(np.sum(contact.area_m2)) == pytest.approx(3e-3 * 2e-3, rel=1e-9)
    coefficient, ambient = contact.board_robin(board, np.full(contact.size, 310.0))
    covered = coefficient > 0.0
    # Conductance per area is the joint value scaled by the covered fraction of each board cell.
    assert np.all(coefficient[covered] <= 1e4 * (1.0 + 1e-9))
    np.testing.assert_allclose(np.sum(coefficient * board.cell_area_m2), 1e4 * 3e-3 * 2e-3, rtol=1e-9)
    np.testing.assert_allclose(ambient[covered], 310.0)


def _strip(pitch_x, pitch_y, rows: int, cols: int) -> PCBConductionProblem:
    active = np.ones((1, rows, cols), dtype=bool)
    mesh = LayeredPCBMesh(active, (35e-6,), pitch_x, pitch_y)
    return PCBConductionProblem(
        mesh,
        (
            CurrentTerminal(tuple((0, r, 0) for r in range(rows + 1)), 1.0, "in"),
            CurrentTerminal(tuple((0, r, cols) for r in range(rows + 1)), -1.0, "out"),
        ),
        reference_node=(0, 0, 0),
    )


def test_graded_dc_strip_has_the_analytic_resistance_and_uniform_arrays_match_the_scalar_mesh() -> None:
    edges = graded_edges(0.0, 20e-3, coarse_pitch_m=0.5e-3, fine_pitch_m=0.1e-3, refine_m=[(8e-3, 9e-3)], margin_m=0.5e-3)
    # A 0.2 mm trace as one row: the two end nodes' equal shares are the uniform injection.
    graded = _strip(np.diff(edges), np.array([0.2e-3]), 1, edges.size - 1)
    solution = solve_pcb_dc(graded)
    sigma = graded.mesh.conductivity_s_per_m[0, 0, 0]
    exact = 20e-3 / (sigma * 35e-6 * 0.2e-3)
    assert solution.solve.converged
    # Uniform current in a straight strip is exact for Q1 on any rectangle grid.
    assert solution.joule_loss_w == pytest.approx(exact, rel=1e-9)
    assert np.allclose(solution.current_density_a_per_m2[..., 0], 1.0 / (35e-6 * 0.2e-3), rtol=1e-9)
    np.testing.assert_allclose(solution.current_density_a_per_m2[..., 1], 0.0, atol=1e-9 / (35e-6 * 0.2e-3))
    # Graded rows too (0.05 / 0.1 / 0.05 mm): the equal nodal shares no longer
    # match the uniform injection exactly, so only the resistance is checked,
    # and the two-level preconditioner has to carry the elongated cells.
    rows = _strip(np.diff(edges), np.array([0.05e-3, 0.1e-3, 0.05e-3]), 3, edges.size - 1)
    rows_solution = solve_pcb_dc(rows)
    assert rows_solution.solve.converged and rows_solution.solve.inner_iterations < 400
    assert rows_solution.joule_loss_w == pytest.approx(exact, rel=1e-3)
    jacobi = solve_pcb_dc(rows, preconditioner="jacobi", config=MPIRConfig(max_inner_iterations=2000))
    assert jacobi.solve.inner_iterations > 2 * rows_solution.solve.inner_iterations

    scalar = _strip(0.25e-3, 0.25e-3, 4, 12)
    arrays = _strip(np.full(12, 0.25e-3), np.full(4, 0.25e-3), 4, 12)
    op_a = MatrixFreePCBOperator(scalar.mesh, reference_node=scalar.reference_node)
    op_b = MatrixFreePCBOperator(arrays.mesh, reference_node=arrays.reference_node)
    vector = np.random.default_rng(3).standard_normal(op_a.size)
    np.testing.assert_array_equal(op_a.apply_high(vector), op_b.apply_high(vector))
    cx, cy = scalar.mesh.element_coefficients()
    ux, uy = op_a._unit_high
    np.testing.assert_allclose(cx[0, 0, 0] * ux + cy[0, 0, 0] * uy, sigma * 35e-6 * _local_stiffness(0.25e-3, 0.25e-3), rtol=1e-12)
