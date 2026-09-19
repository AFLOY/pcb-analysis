"""σ(T) around the board/body interface iteration, against one monolithic masked mesh."""

from __future__ import annotations

import numpy as np
import pytest

from electrical.matrix_free_mpir_fem import CurrentTerminal, LayeredPCBMesh, PCBConductionProblem, ViaConnection
from multiphysics.staggered_coupling import (
    BodyContact,
    CouplingConfig,
    ElectroThermalEnclosureResult,
    ElectroThermalEnclosureScenario,
    ElectroThermalScenario,
    InterfaceCouplingConfig,
    run_board_enclosure_thermal,
    run_electro_thermal,
    run_electro_thermal_enclosure,
    run_scenario,
    BoardEnclosureThermalScenario,
    thermal_problem_with_joule_heat,
)
from electrical.matrix_free_mpir_fem import solve_pcb_dc
from thermal.matrix_free_mpir_fem import (
    ConvectionBoundary,
    ExposedFaceConvection,
    ExposedFaceRadiation,
    LayeredThermalMesh,
    RadiationBoundary,
    ThermalConductionProblem,
    planar_contact_map,
)

AMBIENT = 298.15
PITCH = 0.5e-3
ROWS, COLS = 6, 24
BOARD_SLABS = (35e-6, 1.5e-3, 35e-6)
SINK_ROWS, SINK_COLS = slice(1, 5), slice(6, 18)
SINK_SLABS = (1.0e-3,) * 3
SINK_K = 200.0
TIM_K, TIM_T = 1.0e-2, 1.0e-6
H = 12.0
CURRENT = 6.0


def _loop_board() -> PCBConductionProblem:
    active = np.zeros((2, ROWS, COLS), dtype=bool)
    active[:, 1:5, :] = True
    mesh = LayeredPCBMesh(active, (35e-6, 35e-6), PITCH, PITCH)
    node_rows = tuple(range(1, 6))
    vias = tuple(ViaConnection((0, r, c), (1, r, c), 1.5e-3) for r in node_rows[1:-1] for c in (COLS - 1, COLS))
    return PCBConductionProblem(
        mesh,
        (
            CurrentTerminal(tuple((1, r, 0) for r in node_rows), CURRENT, "source"),
            CurrentTerminal(tuple((0, r, 0) for r in node_rows), -CURRENT, "return"),
        ),
        reference_node=(0, 1, 0),
        vias=vias,
    )


def _board_conductivity(problem: PCBConductionProblem) -> tuple[np.ndarray, np.ndarray]:
    active = problem.mesh.element_active
    k = np.full((3, ROWS, COLS), 0.8)
    k[0] = np.where(active[0], 385.0, 0.8)
    k[2] = np.where(active[1], 385.0, 0.8)
    kz = k.copy()
    kz[1] = 0.3
    return k, kz


def _covered() -> np.ndarray:
    covered = np.zeros((ROWS, COLS), dtype=bool)
    covered[SINK_ROWS, SINK_COLS] = True
    return covered


def _partitioned(radiating: bool = False, alpha: float | None = None) -> ElectroThermalEnclosureScenario:
    problem = _loop_board()
    k, kz = _board_conductivity(problem)
    board_mesh = LayeredThermalMesh(BOARD_SLABS, PITCH, PITCH, k, through_plane_conductivity_w_per_m_k=kz)
    covered = _covered()
    options = dict(
        convection=(
            ExposedFaceConvection(H, AMBIENT, directions=("-z", "-x", "+x", "-y", "+y")),
            ConvectionBoundary("top", np.where(covered, 0.0, H), AMBIENT),
        ),
        conductivity_reference_temperature_k=293.15,
    )
    if radiating:
        # Board top outside the sink and the board edges radiate; the covered
        # top-slab elements are interior, so zeroing their emissivity only
        # switches off the +z faces under the sink.
        emissivity = np.full((3, ROWS, COLS), 0.9)
        emissivity[2, covered] = 0.0
        options["radiation"] = (ExposedFaceRadiation(emissivity, AMBIENT, directions=("+z", "-x", "+x", "-y", "+y")),)
    if alpha is not None:
        options["temperature_coefficient_per_k"] = alpha
    electro_thermal = ElectroThermalScenario(problem, board_mesh, (0, 2), **options)

    sink_mesh = LayeredThermalMesh(SINK_SLABS, PITCH, PITCH, SINK_K, element_shape=(4, 12))
    sink_boundaries = dict(convection=(ExposedFaceConvection(H, AMBIENT, directions=("+z", "-x", "+x", "-y", "+y")),))
    if radiating:
        sink_boundaries["radiation"] = (ExposedFaceRadiation(0.9, AMBIENT, directions=("+z", "-x", "+x", "-y", "+y")),)
    sink = ThermalConductionProblem(sink_mesh, **sink_boundaries)
    contact = planar_contact_map(
        board_mesh, sink_mesh, board_side="top", board_origin_m=(0.0, 0.0),
        body_origin_m=(SINK_COLS.start * PITCH, SINK_ROWS.start * PITCH),
        conductance_per_area_w_per_m2_k=TIM_K / TIM_T,
    )
    return ElectroThermalEnclosureScenario(electro_thermal, (BodyContact(sink, contact, name="sink"),))


def _monolithic(radiating: bool = False) -> ElectroThermalScenario:
    """Board, TIM sliver and sink as one masked stack with the same electrical drive."""

    problem = _loop_board()
    k_board, kz_board = _board_conductivity(problem)
    slabs = BOARD_SLABS + (TIM_T,) + SINK_SLABS
    active = np.ones((len(slabs), ROWS, COLS), dtype=bool)
    active[3:] = False
    active[3:, SINK_ROWS, SINK_COLS] = True
    k = np.ones(active.shape)
    kz = np.ones(active.shape)
    k[:3], kz[:3] = k_board, kz_board
    k[3] = kz[3] = TIM_K
    k[4:] = kz[4:] = SINK_K
    mesh = LayeredThermalMesh(slabs, PITCH, PITCH, k, through_plane_conductivity_w_per_m_k=kz, active=active)
    options = dict(convection=(ExposedFaceConvection(H, AMBIENT),), conductivity_reference_temperature_k=293.15)
    if radiating:
        options["radiation"] = (ExposedFaceRadiation(0.9, AMBIENT, directions=("+z", "-x", "+x", "-y", "+y")),)
    return ElectroThermalScenario(problem, mesh, (0, 2), **options)


def test_zero_coefficient_reduces_to_the_thermal_only_interface_iteration() -> None:
    scenario = _partitioned(alpha=0.0)
    result = run_electro_thermal_enclosure(scenario)
    assert result.converged and result.iterations == 1 and result.loss_increase_ratio == 1.0
    cold = solve_pcb_dc(scenario.electro_thermal.electrical)
    thermal_only = run_board_enclosure_thermal(
        BoardEnclosureThermalScenario(thermal_problem_with_joule_heat(scenario.electro_thermal, cold), scenario.bodies)
    )
    np.testing.assert_allclose(result.thermal.board.temperature_k, thermal_only.board.temperature_k, atol=1e-9)
    assert result.electrical.joule_loss_w == pytest.approx(cold.joule_loss_w)


@pytest.mark.parametrize("radiating", [False, True])
def test_partitioned_sigma_t_matches_the_monolithic_electro_thermal_solution(radiating) -> None:
    """Acceptance: σ(T) through the interface iteration versus σ(T) on one masked mesh."""

    reference = run_electro_thermal(_monolithic(radiating), config=CouplingConfig(temperature_tolerance_k=1e-4))
    result = run_electro_thermal_enclosure(
        _partitioned(radiating),
        config=CouplingConfig(temperature_tolerance_k=1e-4),
        interface=InterfaceCouplingConfig(temperature_tolerance_k=1e-5),
    )
    assert isinstance(result, ElectroThermalEnclosureResult)
    assert reference.converged and result.converged
    assert result.iterations <= 12

    rise = reference.thermal.max_temperature_k - AMBIENT
    assert rise > 10.0
    board_error = np.nanmax(np.abs(result.thermal.board.temperature_k - reference.thermal.temperature_k[:4]))
    sink_reference = reference.thermal.temperature_k[4:, SINK_ROWS.start : SINK_ROWS.stop + 1, SINK_COLS.start : SINK_COLS.stop + 1]
    sink_error = np.nanmax(np.abs(result.thermal.bodies[0].temperature_k - sink_reference))
    assert board_error < 3.0e-3 * rise
    assert sink_error < 3.0e-3 * rise
    # The resistivity feedback is real and both routes agree on it.
    assert result.loss_increase_ratio > 1.02
    assert result.loss_increase_ratio == pytest.approx(reference.loss_increase_ratio, rel=2.0e-3)
    np.testing.assert_allclose(result.element_temperature_k, reference.element_temperature_k, atol=3.0e-3 * rise)
    if radiating:
        assert float(np.sum(result.thermal.board.radiative_heat_w)) > 0.0
        assert float(np.sum(result.thermal.bodies[0].radiative_heat_w)) > 0.0
    # Warm starts: later interface iterations are short.
    assert result.history[-1].interface_iterations <= result.history[0].interface_iterations
    print(
        f"radiating={radiating} outer={result.iterations} rise={rise:.2f} K board_err={board_error:.2e} "
        f"sink_err={sink_error:.2e} loss_ratio={result.loss_increase_ratio:.4f} vs {reference.loss_increase_ratio:.4f}"
    )


def test_run_scenario_dispatches_and_validates() -> None:
    scenario = _partitioned(alpha=0.0)
    assert isinstance(run_scenario(scenario), ElectroThermalEnclosureResult)
    with pytest.raises(ValueError, match="at least one body"):
        ElectroThermalEnclosureScenario(scenario.electro_thermal, ())
