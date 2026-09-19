"""Active-element masks, exposed-face convection, and voxelised bodies."""

from __future__ import annotations

import numpy as np
import pytest

from thermal.matrix_free_mpir_fem import (
    ConvectionBoundary,
    ExposedFaceConvection,
    HeatSource,
    LayeredThermalMesh,
    MatrixFreeThermalOperator,
    ThermalConductionProblem,
    VoxelMaterial,
    VoxelSolidModel,
    VoxelThermalMesh,
    exposed_element_faces,
    solve_thermal_conduction,
)
from thermal.matrix_free_mpir_fem.native_hex import native_available


def _dense(operator: MatrixFreeThermalOperator) -> np.ndarray:
    identity = np.eye(operator.size)
    return np.column_stack([operator.apply_high(identity[:, i]) for i in range(operator.size)])


def _block_in_void(
    grid: tuple[int, int, int], lo: tuple[int, int, int], hi: tuple[int, int, int]
) -> np.ndarray:
    active = np.zeros(grid, dtype=bool)
    active[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = True
    return active


def test_exposed_faces_of_a_block_are_its_six_surfaces() -> None:
    active = _block_in_void((5, 6, 7), (1, 2, 1), (4, 5, 6))
    faces = exposed_element_faces(active)
    counts = {direction: int(np.sum(mask)) for direction, mask in faces.items()}
    # Block is 3 x 3 x 5 elements (z, y, x).
    assert counts == {"-x": 9, "+x": 9, "-y": 15, "+y": 15, "-z": 15, "+z": 15}
    assert faces["+z"][3, 2:5, 1:6].all() and not faces["+z"][1:3].any()
    # A full grid exposes the grid boundary only.
    full = exposed_element_faces(np.ones((2, 3, 4), dtype=bool))
    assert int(np.sum(full["-z"])) == 12 and int(np.sum(full["+x"])) == 6


def test_exposed_face_convection_totals_h_times_area_and_keeps_spd() -> None:
    active = _block_in_void((4, 5, 6), (1, 1, 2), (3, 4, 5))
    mesh = LayeredThermalMesh(
        slab_thickness_m=(0.4e-3, 0.6e-3, 0.5e-3, 0.3e-3),
        pitch_x_m=1.0e-3,
        pitch_y_m=0.8e-3,
        conductivity_w_per_m_k=200.0,
        element_shape=(5, 6),
        active=active,
    )
    convection = ExposedFaceConvection(25.0, 300.0)
    problem = ThermalConductionProblem(mesh, convection=(convection,))
    operator = MatrixFreeThermalOperator(problem, preconditioner="jacobi")
    dense = _dense(operator)
    np.testing.assert_allclose(dense, dense.T, atol=1.0e-12)
    assert np.linalg.eigvalsh(dense)[0] > 0.0

    # Constant temperature: pure conduction vanishes, only the lumped Robin
    # conductance survives, and inactive rows are the identity.
    ones = np.ones(operator.size)
    action = operator.apply_high(ones)
    active_nodes = mesh.active_nodes.reshape(-1)
    np.testing.assert_allclose(action[~active_nodes], 1.0)
    total = float(np.sum(action[active_nodes]))
    assert total == pytest.approx(25.0 * convection.exposed_area_m2(mesh))
    # 2 x 3 x 3 block: two z faces of 9 cells, two y faces (3 x 2 slabs), two x faces.
    area = 2 * 9 * (1.0e-3 * 0.8e-3) + 2 * 3 * 1.0e-3 * (0.6e-3 + 0.5e-3) + 2 * 3 * 0.8e-3 * (0.6e-3 + 0.5e-3)
    assert convection.exposed_area_m2(mesh) == pytest.approx(area)


def test_top_boundary_equals_plus_z_exposed_faces_on_a_full_mesh() -> None:
    mesh = LayeredThermalMesh(
        slab_thickness_m=(0.2e-3, 0.5e-3),
        pitch_x_m=0.4e-3,
        pitch_y_m=0.4e-3,
        conductivity_w_per_m_k=(385.0, 0.8),
        element_shape=(3, 4),
    )
    heat = np.zeros(mesh.element_grid_shape)
    heat[0, 1, 2] = 0.2
    a = solve_thermal_conduction(
        ThermalConductionProblem(mesh, convection=(ConvectionBoundary("top", 15.0, 300.0),), element_heat_w=heat)
    )
    b = solve_thermal_conduction(
        ThermalConductionProblem(
            mesh, convection=(ExposedFaceConvection(15.0, 300.0, directions=("+z",)),), element_heat_w=heat
        )
    )
    np.testing.assert_allclose(a.temperature_k, b.temperature_k, rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(a.convective_heat_w, b.convective_heat_w, rtol=1.0e-8)


def test_block_in_void_matches_the_plain_layered_block() -> None:
    """Acceptance 2: the masked mesh reproduces the layered solve of the same body."""

    block_shape = (3, 4, 5)
    heat_block = np.zeros(block_shape)
    heat_block[1, 1, 2] = 0.5
    heat_block[2, 3, 0] = 0.2
    plain = LayeredThermalMesh(
        slab_thickness_m=(0.5e-3, 0.7e-3, 0.5e-3),
        pitch_x_m=1.0e-3,
        pitch_y_m=1.2e-3,
        conductivity_w_per_m_k=150.0,
        element_shape=block_shape[1:],
    )
    plain_solution = solve_thermal_conduction(
        ThermalConductionProblem(
            plain,
            convection=(ExposedFaceConvection(40.0, 300.0),),
            element_heat_w=heat_block,
        )
    )

    lo, hi = (1, 2, 3), (4, 6, 8)
    active = _block_in_void((6, 9, 10), lo, hi)
    heat = np.zeros(active.shape)
    heat[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = heat_block
    masked = LayeredThermalMesh(
        slab_thickness_m=(0.9e-3, 0.5e-3, 0.7e-3, 0.5e-3, 0.3e-3, 0.3e-3),
        pitch_x_m=1.0e-3,
        pitch_y_m=1.2e-3,
        conductivity_w_per_m_k=150.0,
        element_shape=active.shape[1:],
        active=active,
    )
    masked_solution = solve_thermal_conduction(
        ThermalConductionProblem(
            masked,
            convection=(ExposedFaceConvection(40.0, 300.0),),
            element_heat_w=heat,
        )
    )
    inside = masked_solution.temperature_k[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1]
    np.testing.assert_allclose(inside, plain_solution.temperature_k, rtol=1.0e-7, atol=1.0e-6)
    outside = masked_solution.temperature_k[~masked.active_nodes]
    assert np.all(np.isnan(outside))
    assert masked_solution.max_temperature_k == pytest.approx(plain_solution.max_temperature_k)
    assert masked_solution.min_temperature_k == pytest.approx(plain_solution.min_temperature_k)
    assert abs(masked_solution.heat_balance_error_w) < 1.0e-9
    assert masked_solution.total_heat_input_w == pytest.approx(0.7)
    assert np.all(masked_solution.heat_flux_w_per_m2[~active] == 0.0)
    assert masked_solution.solve.converged


def test_fin_in_a_void_grid_matches_the_one_dimensional_fin() -> None:
    """Acceptance 3: a rectangular fin against theta = theta_b cosh(m(L-x))/cosh(mL)."""

    k, h, thickness, pitch = 200.0, 10.0, 1.0e-3, 0.5e-3
    cols, rows = 40, 8  # 20 mm long, 4 mm wide
    grid = (3, rows + 4, cols + 4)
    active = np.zeros(grid, dtype=bool)
    active[1, 2 : 2 + rows, 0:cols] = True
    mesh = LayeredThermalMesh(
        slab_thickness_m=(thickness, thickness, thickness),
        pitch_x_m=pitch,
        pitch_y_m=pitch,
        conductivity_w_per_m_k=k,
        element_shape=grid[1:],
        active=active,
    )
    fixed = np.zeros(mesh.node_shape, dtype=bool)
    fixed[1:3, 2 : 3 + rows, 0] = True
    base, ambient = 350.0, 300.0
    solution = solve_thermal_conduction(
        ThermalConductionProblem(
            mesh,
            convection=(ExposedFaceConvection(h, ambient, directions=("-z", "+z")),),
            fixed_temperature_mask=fixed,
            fixed_temperature_k=base,
        )
    )
    assert solution.solve.converged
    m = np.sqrt(2.0 * h / (k * thickness))
    length = cols * pitch
    x = pitch * np.arange(cols + 1)
    analytic = ambient + (base - ambient) * np.cosh(m * (length - x)) / np.cosh(m * length)
    centre = solution.temperature_k[1, 2 + rows // 2, : cols + 1]
    np.testing.assert_allclose(centre, analytic, rtol=2.0e-3)
    # Heat entering at the base, k A m theta_b tanh(mL), leaves as convection.
    area = rows * pitch * thickness
    q_base = k * area * m * (base - ambient) * np.tanh(m * length)
    assert solution.fixed_temperature_heat_w == pytest.approx(-q_base, rel=2.0e-2)
    assert float(np.sum(solution.convective_heat_w)) == pytest.approx(q_base, rel=2.0e-2)


def test_per_face_ambient_on_a_convection_boundary() -> None:
    mesh = LayeredThermalMesh(
        slab_thickness_m=(0.3e-3, 0.3e-3),
        pitch_x_m=0.5e-3,
        pitch_y_m=0.5e-3,
        conductivity_w_per_m_k=100.0,
        element_shape=(3, 5),
    )
    ambient = np.full((3, 5), 300.0)
    ambient[:, 3:] = 320.0
    problem = ThermalConductionProblem(
        mesh,
        convection=(
            ConvectionBoundary("top", 30.0, ambient),
            ConvectionBoundary("bottom", 5.0, 300.0),
        ),
    )
    operator = MatrixFreeThermalOperator(problem, preconditioner="jacobi")
    dense = _dense(operator)
    np.testing.assert_allclose(dense, dense.T, atol=1.0e-12)
    solution = solve_thermal_conduction(problem)
    assert solution.solve.converged
    top = solution.temperature_k[-1]
    # Warm ambient on the right pulls the plate above the cold ambient there.
    assert top[:, -1].mean() > top[:, 0].mean() > 300.0
    assert abs(solution.heat_balance_error_w) < 1.0e-9
    # Energy conservation with no heat input: the two faces exchange the same heat.
    assert float(np.sum(solution.convective_heat_w)) == pytest.approx(0.0, abs=1.0e-9)
    with pytest.raises(ValueError, match="ambient array must match"):
        ThermalConductionProblem(mesh, convection=(ConvectionBoundary("top", 1.0, np.zeros((2, 2))),))


def test_voxel_solid_model_builds_a_masked_mesh() -> None:
    ids = np.zeros((4, 5, 6), dtype=np.int64)
    ids[0, 1:4, 1:5] = 1  # base plate
    ids[1:4, 1:4:2, 1:5] = 2  # two fins
    fill = np.ones(ids.shape)
    fill[3] = 0.5  # fin tips half filled
    model = VoxelSolidModel(
        ids,
        {1: VoxelMaterial("copper", 385.0), 2: VoxelMaterial("aluminium", 200.0, role="fin")},
        pitch_m=(1.0e-3, 1.0e-3, 2.0e-3),
        origin_m=(0.01, 0.02, 0.0),
        fill=fill,
    )
    assert model.shape == (4, 5, 6)
    assert model.volume_m3() == pytest.approx((12 + 2 * 4 * 2 + 2 * 4 * 0.5) * 2.0e-9)
    z, y, x = model.voxel_centres_m()
    assert z[0] == pytest.approx(1.0e-3) and y[0] == pytest.approx(0.0205) and x[-1] == pytest.approx(0.0155)
    mesh = VoxelThermalMesh.from_solid_model(model)
    assert isinstance(mesh, LayeredThermalMesh)
    assert mesh.element_grid_shape == (4, 5, 6)
    np.testing.assert_array_equal(mesh.active, ids > 0)
    assert mesh.conductivity_w_per_m_k[0, 1, 1] == 385.0
    assert mesh.conductivity_w_per_m_k[3, 1, 1] == pytest.approx(100.0)
    assert mesh.conductivity_w_per_m_k[2, 2, 1] == 0.0
    dropped = VoxelThermalMesh.from_solid_model(model, min_fill=0.6)
    assert not dropped.active[3].any() and dropped.active[2].any()

    solution = solve_thermal_conduction(
        ThermalConductionProblem(
            mesh,
            convection=(ExposedFaceConvection(12.0, 298.0),),
            heat_sources=(HeatSource(((0, 2, 3),), 2.0),),
        )
    )
    assert solution.solve.converged
    assert solution.max_temperature_k > 298.0
    assert abs(solution.heat_balance_error_w) < 1.0e-9

    with pytest.raises(ValueError, match="reserved for void"):
        VoxelSolidModel(ids, {0: VoxelMaterial("void", 1.0), 1: VoxelMaterial("a", 1.0), 2: VoxelMaterial("b", 1.0)}, (1e-3,) * 3)
    with pytest.raises(ValueError, match="without a material"):
        VoxelSolidModel(ids, {1: VoxelMaterial("a", 1.0)}, (1e-3,) * 3)


def test_loads_on_inactive_elements_are_rejected() -> None:
    active = _block_in_void((2, 3, 3), (0, 0, 0), (2, 2, 3))
    mesh = LayeredThermalMesh((1e-3, 1e-3), 1e-3, 1e-3, 10.0, element_shape=(3, 3), active=active)
    heat = np.zeros(mesh.element_grid_shape)
    heat[1, 2, 1] = 1.0
    with pytest.raises(ValueError, match="inactive elements"):
        ThermalConductionProblem(mesh, convection=(ExposedFaceConvection(1.0, 300.0),), element_heat_w=heat)
    with pytest.raises(ValueError, match="inactive node"):
        ThermalConductionProblem(
            mesh, convection=(ExposedFaceConvection(1.0, 300.0),), heat_sources=(HeatSource(((2, 3, 0),), 1.0),)
        )
    with pytest.raises(ValueError, match="at least one element"):
        LayeredThermalMesh((1e-3,), 1e-3, 1e-3, 10.0, element_shape=(2, 2), active=np.zeros((1, 2, 2), bool))
    # A fixed node in the void does not anchor the temperature level.
    fixed = np.zeros(mesh.node_shape, dtype=bool)
    fixed[2, 3, 1] = True
    with pytest.raises(ValueError, match="needs a positive film coefficient"):
        ThermalConductionProblem(mesh, fixed_temperature_mask=fixed, fixed_temperature_k=300.0)


@pytest.mark.skipif(not native_available(), reason="thermal native extension not built")
@pytest.mark.parametrize("threads", [1, 3])
def test_native_path_solves_the_masked_mesh_like_the_array_path(threads: int) -> None:
    active = _block_in_void((5, 7, 9), (1, 1, 2), (4, 6, 7))
    active[2, 3, 4] = False  # an internal void cell
    mesh = LayeredThermalMesh(
        slab_thickness_m=(0.5e-3,) * 5,
        pitch_x_m=0.5e-3,
        pitch_y_m=0.5e-3,
        conductivity_w_per_m_k=200.0,
        element_shape=(7, 9),
        active=active,
    )
    heat = np.zeros(mesh.element_grid_shape)
    heat[1, 2, 3] = 0.3
    problem = ThermalConductionProblem(mesh, convection=(ExposedFaceConvection(20.0, 300.0),), element_heat_w=heat)
    portable = MatrixFreeThermalOperator(problem)
    native = MatrixFreeThermalOperator(problem, native=True, native_threads=threads)
    vector = np.random.default_rng(3).standard_normal(portable.size).astype(np.float32)
    np.testing.assert_allclose(native.apply_low(vector), portable.apply_low(vector), rtol=2.0e-5, atol=1.0e-6)
    a = solve_thermal_conduction(problem)
    b = solve_thermal_conduction(problem, native=True, native_threads=threads)
    assert a.solve.converged and b.solve.converged
    np.testing.assert_allclose(b.temperature_k, a.temperature_k, rtol=1.0e-7, atol=1.0e-6, equal_nan=True)


def test_cuda_backend_solves_the_masked_mesh_like_the_host() -> None:
    cupy = pytest.importorskip("cupy")
    from electrical.matrix_free_mpir_fem import CupyFloat32Runtime

    try:
        runtime = CupyFloat32Runtime()
    except RuntimeError as exc:  # pragma: no cover - no device
        pytest.skip(str(exc))
    del cupy
    active = _block_in_void((5, 7, 9), (1, 1, 2), (4, 6, 7))
    active[2, 3, 4] = False
    mesh = LayeredThermalMesh(
        slab_thickness_m=(0.5e-3,) * 5,
        pitch_x_m=0.5e-3,
        pitch_y_m=0.5e-3,
        conductivity_w_per_m_k=200.0,
        element_shape=(7, 9),
        active=active,
    )
    heat = np.zeros(mesh.element_grid_shape)
    heat[1, 2, 3] = 0.3
    problem = ThermalConductionProblem(mesh, convection=(ExposedFaceConvection(20.0, 300.0),), element_heat_w=heat)
    cpu = MatrixFreeThermalOperator(problem)
    gpu = MatrixFreeThermalOperator(problem, runtime=runtime)
    vector = np.random.default_rng(5).standard_normal(cpu.size)
    expected = cpu.apply_high(vector)
    actual = runtime.to_host(gpu.apply_low(runtime.from_host(vector)))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-5 * np.max(np.abs(expected)))
    a = solve_thermal_conduction(problem)
    b = solve_thermal_conduction(problem, runtime=runtime)
    assert a.solve.converged and b.solve.converged
    np.testing.assert_allclose(b.temperature_k, a.temperature_k, rtol=1.0e-7, atol=1.0e-6, equal_nan=True)
