"""Board and separately meshed bodies (heat sinks, enclosures), coupled at contacts.

The board keeps its layered mesh on the routing grid; every body keeps its own
``LayeredThermalMesh`` (a ``VoxelThermalMesh`` from CAD).  A ``ContactMap`` per
body names the touching faces and the joint conductance ``G``.  The partitioned
iteration is Robin on the board and Neumann on the body:

1. the board is solved with an extra convection boundary on the contact faces,
   film coefficient ``G / A`` and ambient equal to the body's current face
   temperature;
2. the heat that crossed each pair, ``G (T_board - T_body)``, is lumped onto the
   body's face nodes and the body is solved with its own boundaries;
3. the body face temperatures are relaxed (Aitken) and step 1 repeats until
   they stop moving.

Neither solver knows about the other; the exchange is two arrays per body.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from electrical.matrix_free_mpir_fem import MPIRConfig
from electrical.matrix_free_mpir_fem.runtime import RuntimeBackend
from thermal.matrix_free_mpir_fem import (
    ContactMap,
    ConvectionBoundary,
    ThermalConductionProblem,
    ThermalConductionSolution,
    solve_thermal_conduction,
)


@dataclass(frozen=True)
class BodyContact:
    """One body with its own thermal problem and its contact to the board."""

    body: ThermalConductionProblem
    contact: ContactMap
    name: str = "body"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("body name must not be empty")


@dataclass(frozen=True)
class BoardEnclosureThermalScenario:
    """A board and the bodies in contact with it, each on its own mesh."""

    board: ThermalConductionProblem
    bodies: tuple[BodyContact, ...]

    def __post_init__(self) -> None:
        bodies = tuple(self.bodies)
        if not bodies:
            raise ValueError("at least one body is required")
        names = [body.name for body in bodies]
        if len(set(names)) != len(names):
            raise ValueError("body names must be unique")
        for body in bodies:
            body.contact.check(self.board.mesh, body.body.mesh)
        object.__setattr__(self, "bodies", bodies)


@dataclass(frozen=True)
class InterfaceCouplingConfig:
    """Fixed-point limits for the board/body interface iteration."""

    max_iterations: int = 50
    temperature_tolerance_k: float = 1.0e-4
    relative_heat_tolerance: float = 1.0e-6
    relaxation: float = 1.0
    aitken: bool = True
    max_relaxation: float = 4.0
    divergence_temperature_k: float = 1.0e6
    board: MPIRConfig | None = None
    body: MPIRConfig | None = None

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.temperature_tolerance_k <= 0.0 or self.relative_heat_tolerance <= 0.0:
            raise ValueError("tolerances must be positive")
        if self.divergence_temperature_k <= 0.0:
            raise ValueError("divergence_temperature_k must be positive")
        if not 0.0 < self.relaxation <= self.max_relaxation:
            raise ValueError("relaxation must lie in (0, max_relaxation]")
        if self.max_relaxation < 1.0:
            raise ValueError("max_relaxation must be at least one")


@dataclass(frozen=True)
class InterfaceStep:
    iteration: int
    interface_heat_w: float
    max_temperature_change_k: float
    relative_heat_change: float
    relaxation: float
    board_inner_iterations: int
    body_inner_iterations: int


@dataclass(frozen=True)
class BoardEnclosureThermalResult:
    board: ThermalConductionSolution
    bodies: tuple[ThermalConductionSolution, ...]
    contact_heat_w: tuple[np.ndarray, ...]
    contact_temperature_k: tuple[np.ndarray, ...]
    converged: bool
    iterations: int
    history: tuple[InterfaceStep, ...]

    @property
    def interface_heat_w(self) -> float:
        return float(sum(np.sum(heat) for heat in self.contact_heat_w))


def _board_problem(
    scenario: BoardEnclosureThermalScenario, body_temperatures: list[np.ndarray]
) -> ThermalConductionProblem:
    extra = []
    for body, temperature in zip(scenario.bodies, body_temperatures):
        coefficient, ambient = body.contact.board_robin(scenario.board.mesh, temperature)
        extra.append(ConvectionBoundary(body.contact.board_side, coefficient, ambient))
    return dataclasses.replace(
        scenario.board, convection=tuple(scenario.board.convection) + tuple(extra)
    )


def run_board_enclosure_thermal(
    scenario: BoardEnclosureThermalScenario,
    *,
    config: InterfaceCouplingConfig | None = None,
    backend: RuntimeBackend | None = None,
    device_id: int = 0,
) -> BoardEnclosureThermalResult:
    """Iterate the board and its bodies to a common interface temperature."""

    config = config or InterfaceCouplingConfig()
    board_mesh = scenario.board.mesh
    bodies = scenario.bodies

    # Start every contact at the body's own reference: its first convective
    # ambient, else its mean fixed temperature.
    body_temperatures: list[np.ndarray] = []
    for body in bodies:
        problem = body.body
        if problem.convection:
            start = problem.convection[0].mean_ambient_k()
        else:
            mask = problem.fixed_temperature_mask
            start = float(np.mean(problem.fixed_temperature_k[mask]))
        body_temperatures.append(np.full(body.contact.size, start))

    board_solution: ThermalConductionSolution | None = None
    body_solutions: list[ThermalConductionSolution] = []
    pair_heat: list[np.ndarray] = []
    board_guess: np.ndarray | None = None
    body_guesses: list[np.ndarray | None] = [None] * len(bodies)
    previous_increment: np.ndarray | None = None
    relaxation = config.relaxation
    previous_heat = float("nan")
    history: list[InterfaceStep] = []
    converged = False

    for iteration in range(1, config.max_iterations + 1):
        board_solution = solve_thermal_conduction(
            _board_problem(scenario, body_temperatures),
            config=config.board,
            backend=backend,
            device_id=device_id,
            initial_temperature_k=board_guess,
        )
        board_guess = np.nan_to_num(board_solution.temperature_k, nan=board_solution.min_temperature_k)

        body_solutions = []
        pair_heat = []
        proposed: list[np.ndarray] = []
        body_inner = 0
        for index, (body, current) in enumerate(zip(bodies, body_temperatures)):
            contact = body.contact
            board_face = contact.board_face_temperature_k(board_mesh, board_solution.temperature_k)
            heat = contact.pair_heat_w(board_face, current)
            problem = dataclasses.replace(
                body.body,
                nodal_heat_w=body.body.nodal_heat_w + contact.body_nodal_heat_w(body.body.mesh, heat),
            )
            solution = solve_thermal_conduction(
                problem,
                config=config.body,
                backend=backend,
                device_id=device_id,
                initial_temperature_k=body_guesses[index],
            )
            body_guesses[index] = np.nan_to_num(solution.temperature_k, nan=solution.min_temperature_k)
            body_solutions.append(solution)
            pair_heat.append(heat)
            proposed.append(contact.body_face_temperature_k(body.body.mesh, solution.temperature_k))

        # Relaxed fixed point on the stacked contact temperatures, with
        # Aitken's Δ² rescaling from two successive increments.
        current_all = np.concatenate(body_temperatures)
        proposed_all = np.concatenate(proposed)
        increment = proposed_all - current_all
        if config.aitken and previous_increment is not None:
            difference = increment - previous_increment
            denominator = float(np.dot(difference, difference))
            if denominator > 0.0:
                relaxation = -relaxation * float(np.dot(previous_increment, difference)) / denominator
                relaxation = float(np.clip(relaxation, 0.05, config.max_relaxation))
        updated = current_all + relaxation * increment
        change = float(np.max(np.abs(updated - current_all)))
        previous_increment = increment
        offsets = np.cumsum([0] + [body.contact.size for body in bodies])
        body_temperatures = [updated[offsets[i] : offsets[i + 1]] for i in range(len(bodies))]

        total_heat = float(sum(np.sum(heat) for heat in pair_heat))
        relative_heat_change = (
            abs(total_heat - previous_heat) / abs(total_heat)
            if np.isfinite(previous_heat) and total_heat
            else float("inf")
        )
        previous_heat = total_heat
        history.append(
            InterfaceStep(
                iteration=iteration,
                interface_heat_w=total_heat,
                max_temperature_change_k=change,
                relative_heat_change=relative_heat_change,
                relaxation=relaxation,
                board_inner_iterations=board_solution.solve.inner_iterations,
                body_inner_iterations=sum(s.solve.inner_iterations for s in body_solutions),
            )
        )
        if change <= config.temperature_tolerance_k and relative_heat_change <= config.relative_heat_tolerance:
            converged = board_solution.solve.converged and all(s.solve.converged for s in body_solutions)
            break
        if not np.isfinite(change) or change > config.divergence_temperature_k:
            # A hard contact on a stiff body has a fixed-point gain near one;
            # without relaxation the Dirichlet-Neumann exchange oscillates
            # with growing amplitude.  Stop before the solves see infinities.
            break

    assert board_solution is not None
    return BoardEnclosureThermalResult(
        board=board_solution,
        bodies=tuple(body_solutions),
        contact_heat_w=tuple(pair_heat),
        contact_temperature_k=tuple(body_temperatures),
        converged=converged,
        iterations=len(history),
        history=tuple(history),
    )


__all__ = [
    "BoardEnclosureThermalResult",
    "BoardEnclosureThermalScenario",
    "BodyContact",
    "InterfaceCouplingConfig",
    "InterfaceStep",
    "run_board_enclosure_thermal",
]
