"""Board and a separately meshed heat sink, coupled through a contact map."""

from __future__ import annotations

import numpy as np
import pytest

from multiphysics.staggered_coupling import (
    BoardEnclosureThermalResult,
    BoardEnclosureThermalScenario,
    BodyContact,
    InterfaceCouplingConfig,
    run_board_enclosure_thermal,
    run_scenario,
)
from thermal.matrix_free_mpir_fem import (
    ContactMap,
    ConvectionBoundary,
    ExposedFaceConvection,
    LayeredThermalMesh,
    ThermalConductionProblem,
    planar_contact_map,
    solve_thermal_conduction,
)

PITCH = 1.0e-3
ROWS, COLS = 12, 16
BOARD_SLABS = (35.0e-6, 1.5e-3, 35.0e-6)
BOARD_K = (385.0, 0.8, 385.0)
BOARD_KZ = (385.0, 0.3, 385.0)
SINK_ROWS, SINK_COLS = slice(3, 9), slice(5, 11)
SINK_SLABS = (1.0e-3,) * 4
SINK_K = 200.0
TIM_K, TIM_T = 1.0e-2, 1.0e-6  # G / A = 1e4 W/m²K
H_BOARD, H_SINK, AMBIENT = 10.0, 25.0, 300.0


def _board_heat() -> np.ndarray:
    heat = np.zeros((3, ROWS, COLS))
    heat[0, 5:7, 6:10] = 0.05  # 0.4 W in the bottom copper under the sink
    heat[2, 1, 13] = 0.1  # 0.1 W in the top copper, away from the sink
    return heat


def _covered() -> np.ndarray:
    covered = np.zeros((ROWS, COLS), dtype=bool)
    covered[SINK_ROWS, SINK_COLS] = True
    return covered


def _monolithic() -> ThermalConductionProblem:
    """Board, TIM slab and sink as one masked layered mesh (the reference)."""

    slabs = BOARD_SLABS + (TIM_T,) + SINK_SLABS
    active = np.ones((len(slabs), ROWS, COLS), dtype=bool)
    active[3:] = False
    active[3:, SINK_ROWS, SINK_COLS] = True
    k = np.zeros(active.shape)
    kz = np.zeros(active.shape)
    for index, (value, through) in enumerate(zip(BOARD_K, BOARD_KZ)):
        k[index] = value
        kz[index] = through
    k[3] = kz[3] = TIM_K
    k[4:] = kz[4:] = SINK_K
    mesh = LayeredThermalMesh(
        slab_thickness_m=slabs,
        pitch_x_m=PITCH,
        pitch_y_m=PITCH,
        conductivity_w_per_m_k=np.where(active, k, 1.0),
        through_plane_conductivity_w_per_m_k=np.where(active, kz, 1.0),
        active=active,
    )
    heat = np.zeros(active.shape)
    heat[:3] = _board_heat()
    # The board (and the TIM sliver) cool with h_board, the sink with h_sink.
    board_faces = ExposedFaceConvection(H_BOARD, AMBIENT)
    problem = ThermalConductionProblem(mesh, convection=(board_faces,), element_heat_w=heat)
    return problem


def _partitioned() -> BoardEnclosureThermalScenario:
    board_mesh = LayeredThermalMesh(
        slab_thickness_m=BOARD_SLABS,
        pitch_x_m=PITCH,
        pitch_y_m=PITCH,
        conductivity_w_per_m_k=BOARD_K,
        through_plane_conductivity_w_per_m_k=BOARD_KZ,
        element_shape=(ROWS, COLS),
    )
    covered = _covered()
    board = ThermalConductionProblem(
        board_mesh,
        convection=(
            ExposedFaceConvection(H_BOARD, AMBIENT, directions=("-z", "-x", "+x", "-y", "+y")),
            ConvectionBoundary("top", np.where(covered, 0.0, H_BOARD), AMBIENT),
        ),
        element_heat_w=_board_heat(),
    )
    sink_mesh = LayeredThermalMesh(
        slab_thickness_m=SINK_SLABS,
        pitch_x_m=PITCH,
        pitch_y_m=PITCH,
        conductivity_w_per_m_k=SINK_K,
        element_shape=(6, 6),
    )
    sink = ThermalConductionProblem(
        sink_mesh,
        convection=(ExposedFaceConvection(H_BOARD, AMBIENT, directions=("+z", "-x", "+x", "-y", "+y")),),
    )
    contact = planar_contact_map(
        board_mesh,
        sink_mesh,
        board_side="top",
        board_origin_m=(0.0, 0.0),
        body_origin_m=(SINK_COLS.start * PITCH, SINK_ROWS.start * PITCH),
        conductance_per_area_w_per_m2_k=TIM_K / TIM_T,
    )
    return BoardEnclosureThermalScenario(board, (BodyContact(sink, contact, name="sink"),))


def test_contact_map_geometry_and_validation() -> None:
    scenario = _partitioned()
    contact = scenario.bodies[0].contact
    assert contact.size == 36
    assert contact.body_face == "-z" and contact.board_face == "+z"
    np.testing.assert_allclose(contact.area_m2, PITCH * PITCH)
    np.testing.assert_allclose(contact.conductance_w_per_k, TIM_K / TIM_T * PITCH * PITCH)
    assert set(map(tuple, contact.board_cells)) == {(r, c) for r in range(3, 9) for c in range(5, 11)}
    assert set(map(tuple, contact.body_cells)) == {(0, r, c) for r in range(6) for c in range(6)}
    coefficient, ambient = contact.board_robin(scenario.board.mesh, np.full(36, 310.0))
    assert coefficient[4, 6] == pytest.approx(TIM_K / TIM_T) and coefficient[0, 0] == 0.0
    assert ambient[4, 6] == 310.0 and ambient[0, 0] == 0.0

    # Half-pitch offset: every sink cell straddles four board cells.
    shifted = planar_contact_map(
        scenario.board.mesh,
        scenario.bodies[0].body.mesh,
        board_side="top",
        body_origin_m=(5.5 * PITCH, 3.5 * PITCH),
        conductance_per_area_w_per_m2_k=1.0e4,
    )
    assert shifted.size == 36 * 4
    assert float(np.sum(shifted.area_m2)) == pytest.approx(36 * PITCH * PITCH)
    np.testing.assert_allclose(shifted.area_m2, PITCH * PITCH / 4.0)

    with pytest.raises(ValueError, match="do not overlap"):
        planar_contact_map(
            scenario.board.mesh,
            scenario.bodies[0].body.mesh,
            board_side="top",
            body_origin_m=(1.0, 1.0),
            conductance_per_area_w_per_m2_k=1.0e4,
        )
    with pytest.raises(ValueError, match="inactive body"):
        ContactMap("top", "-z", [[3, 5]], [[0, 0, 0]], [1.0e-6], [1.0]).check(
            scenario.board.mesh,
            LayeredThermalMesh((1e-3,), 1e-3, 1e-3, 1.0, element_shape=(2, 2), active=np.array([[[False, True], [True, True]]])),
        )


def test_partitioned_iteration_matches_the_monolithic_mesh() -> None:
    """Acceptance 4: two meshes through a contact map versus one masked mesh."""

    reference = solve_thermal_conduction(_monolithic())
    assert reference.solve.converged
    scenario = _partitioned()
    result = run_board_enclosure_thermal(scenario, config=InterfaceCouplingConfig(temperature_tolerance_k=1.0e-5))
    assert isinstance(result, BoardEnclosureThermalResult)
    assert result.converged
    assert result.iterations < 30

    board_ref = reference.temperature_k[:4]  # board node layers; layer 3 is the TIM bottom
    rise = reference.max_temperature_k - AMBIENT
    board_error = np.nanmax(np.abs(result.board.temperature_k - board_ref))
    sink_ref = reference.temperature_k[4:, 3:10, 5:12]
    sink_error = np.nanmax(np.abs(result.bodies[0].temperature_k - sink_ref))
    assert rise > 1.0
    assert board_error < 2.0e-3 * rise
    assert sink_error < 2.0e-3 * rise

    # Heat crossing the interface equals the board's convective heat on that
    # boundary and is what the monolithic mesh carries through the TIM.
    interface = result.interface_heat_w
    assert interface == pytest.approx(float(result.board.convective_heat_w[-1]), rel=1.0e-9)
    assert interface == pytest.approx(float(np.sum(result.bodies[0].convective_heat_w)), rel=1.0e-6)
    assert 0.0 < interface < 0.5
    assert abs(result.board.heat_balance_error_w) < 1.0e-9
    assert abs(result.bodies[0].heat_balance_error_w) < 1.0e-9
    # The history is monotone in the sense that the last step is the smallest.
    changes = [step.max_temperature_change_k for step in result.history]
    assert changes[-1] == min(changes)
    print(
        f"interface iterations={result.iterations} board_err={board_error:.2e} "
        f"sink_err={sink_error:.2e} rise={rise:.3f} heat={interface:.4f} W"
    )


def test_run_scenario_dispatches_and_fixed_relaxation_converges_slower() -> None:
    scenario = _partitioned()
    dispatched = run_scenario(scenario)
    assert isinstance(dispatched, BoardEnclosureThermalResult)
    assert dispatched.converged
    fixed = run_board_enclosure_thermal(
        scenario, config=InterfaceCouplingConfig(aitken=False, relaxation=0.2, max_iterations=400)
    )
    assert fixed.converged
    np.testing.assert_allclose(
        fixed.board.temperature_k, dispatched.board.temperature_k, atol=5.0e-4, equal_nan=True
    )
    assert dispatched.iterations < fixed.iterations
    # The hard contact on a stiff sink has a fixed-point gain near one: the
    # unrelaxed exchange diverges, and the run stops instead of overflowing.
    unrelaxed = run_board_enclosure_thermal(
        scenario, config=InterfaceCouplingConfig(aitken=False, relaxation=1.0, max_iterations=400)
    )
    assert not unrelaxed.converged
    assert unrelaxed.iterations < 400
    print(f"aitken={dispatched.iterations} fixed0.2={fixed.iterations} unrelaxed stopped at {unrelaxed.iterations}")
