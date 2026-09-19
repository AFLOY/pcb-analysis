"""σ(T) around a board that is coupled to separately meshed bodies.

The copper resistivity loop of :mod:`.electro_thermal` wrapped around the
board/body interface iteration of :mod:`.board_enclosure`: an electrical
solve gives the Joule heat, the board *and* its bodies are solved to a common
interface temperature, the copper conductivity follows the board temperature,
repeat.  Radiating faces on the board or the bodies are part of each thermal
solve (Newton linearisation inside ``solve_thermal_conduction``), so one
outer loop closes σ(T), the contact exchange and radiation together.

Each interface iteration is warm-started from the previous outer step, so
after the first pass it costs one or two board/body solves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from electrical.matrix_free_mpir_fem import PCBConductionSolution, RuntimeBackend, solve_pcb_dc
from thermal.matrix_free_mpir_fem import ThermalConductionProblem

from .board_enclosure import (
    BoardEnclosureThermalResult,
    BoardEnclosureThermalScenario,
    BodyContact,
    InterfaceCouplingConfig,
    run_board_enclosure_thermal,
)
from .electro_thermal import (
    CouplingConfig,
    CouplingStep,
    ElectroThermalScenario,
    TemperatureFixedPoint,
    electrical_layer_temperature_k,
    heated_electrical_problem,
    thermal_problem_with_joule_heat,
)


@dataclass(frozen=True)
class ElectroThermalEnclosureScenario:
    """A ρ(T) board scenario plus the bodies in contact with its thermal mesh."""

    electro_thermal: ElectroThermalScenario
    bodies: tuple[BodyContact, ...]

    def __post_init__(self) -> None:
        bodies = tuple(self.bodies)
        if not bodies:
            raise ValueError("at least one body is required; use ElectroThermalScenario without bodies")
        object.__setattr__(self, "bodies", bodies)
        # Validate the contacts against the board mesh once, without heat.
        BoardEnclosureThermalScenario(
            _unloaded_board(self.electro_thermal),
            bodies,
        )


def _unloaded_board(scenario: ElectroThermalScenario) -> ThermalConductionProblem:
    return ThermalConductionProblem(
        scenario.thermal_mesh,
        convection=scenario.convection,
        fixed_temperature_mask=scenario.fixed_temperature_mask,
        fixed_temperature_k=scenario.fixed_temperature_k,
        heat_sources=scenario.extra_heat_sources,
        element_heat_w=scenario.extra_element_heat_w,
        radiation=scenario.radiation,
    )


@dataclass(frozen=True)
class ElectroThermalEnclosureStep(CouplingStep):
    interface_iterations: int = 0
    interface_heat_w: float = 0.0


@dataclass(frozen=True)
class ElectroThermalEnclosureResult:
    electrical: PCBConductionSolution
    thermal: BoardEnclosureThermalResult
    conductivity_s_per_m: np.ndarray
    via_resistance_ohm: np.ndarray
    element_temperature_k: np.ndarray
    converged: bool
    iterations: int
    history: tuple[ElectroThermalEnclosureStep, ...]
    cold_joule_loss_w: float

    @property
    def loss_increase_ratio(self) -> float:
        return self.electrical.joule_loss_w / self.cold_joule_loss_w if self.cold_joule_loss_w else float("nan")


def run_electro_thermal_enclosure(
    scenario: ElectroThermalEnclosureScenario,
    *,
    config: CouplingConfig | None = None,
    interface: InterfaceCouplingConfig | None = None,
    backend: RuntimeBackend | None = None,
    device_id: int = 0,
) -> ElectroThermalEnclosureResult:
    """Iterate electrical, board and body solves to a self-consistent ρ(T) state."""

    config = config or CouplingConfig()
    interface = interface or InterfaceCouplingConfig()
    board = scenario.electro_thermal
    problem = board.electrical
    fixed_point = TemperatureFixedPoint(board, config)
    history: list[ElectroThermalEnclosureStep] = []
    potential: np.ndarray | None = None
    cold_loss = float("nan")
    electrical_solution: PCBConductionSolution | None = None
    thermal_result: BoardEnclosureThermalResult | None = None
    converged = False
    previous_loss = float("nan")

    for iteration in range(1, config.max_iterations + 1):
        electrical_solution = solve_pcb_dc(
            problem, config=config.electrical, backend=backend, device_id=device_id, initial_potential_v=potential
        )
        potential = electrical_solution.potential_v
        loss = electrical_solution.joule_loss_w
        if iteration == 1:
            cold_loss = loss

        thermal_result = run_board_enclosure_thermal(
            BoardEnclosureThermalScenario(
                thermal_problem_with_joule_heat(board, electrical_solution), scenario.bodies
            ),
            config=interface,
            backend=backend,
            device_id=device_id,
            initial=thermal_result,
        )
        proposed = np.where(
            np.isfinite(thermal_result.board.temperature_k),
            thermal_result.board.temperature_k,
            thermal_result.board.min_temperature_k,
        )
        change = fixed_point.update(proposed)
        temperature = fixed_point.temperature
        assert temperature is not None

        relative_loss_change = (
            abs(loss - previous_loss) / abs(loss) if np.isfinite(previous_loss) and loss else float("inf")
        )
        previous_loss = loss
        history.append(
            ElectroThermalEnclosureStep(
                iteration=iteration,
                joule_loss_w=loss,
                max_temperature_k=float(np.max(temperature)),
                temperature_change_k=change,
                relative_loss_change=relative_loss_change,
                relaxation=fixed_point.relaxation,
                electrical_inner_iterations=electrical_solution.solve.inner_iterations,
                thermal_inner_iterations=thermal_result.board.solve.inner_iterations
                + sum(s.solve.inner_iterations for s in thermal_result.bodies),
                interface_iterations=thermal_result.iterations,
                interface_heat_w=thermal_result.interface_heat_w,
            )
        )
        if board.temperature_coefficient_per_k == 0.0:
            converged = electrical_solution.solve.converged and thermal_result.converged
            break
        if (
            change <= config.temperature_tolerance_k
            and relative_loss_change <= config.relative_loss_tolerance
            and electrical_solution.solve.converged
            and thermal_result.converged
        ):
            converged = True
            break
        problem = heated_electrical_problem(board, temperature)

    assert electrical_solution is not None and thermal_result is not None and temperature is not None
    return ElectroThermalEnclosureResult(
        electrical=electrical_solution,
        thermal=thermal_result,
        conductivity_s_per_m=np.asarray(problem.mesh.conductivity_s_per_m, dtype=np.float64),
        via_resistance_ohm=np.asarray([via.resistance_ohm for via in problem.vias], dtype=np.float64),
        element_temperature_k=electrical_layer_temperature_k(temperature, board.thermal_mesh, board.layer_slabs),
        converged=converged,
        iterations=len(history),
        history=tuple(history),
        cold_joule_loss_w=cold_loss,
    )


__all__ = [
    "ElectroThermalEnclosureResult",
    "ElectroThermalEnclosureScenario",
    "ElectroThermalEnclosureStep",
    "run_electro_thermal_enclosure",
]
