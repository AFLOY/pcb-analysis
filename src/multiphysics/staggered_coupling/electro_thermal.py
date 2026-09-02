"""Staggered electro-thermal coupling with temperature-dependent copper.

Joule loss heats the board; hotter copper has higher resistivity,

```text
σ(T) = σ_ref / (1 + α (T - T_ref)),
```

and the higher resistance changes both the loss and, where the heating is
uneven, the current distribution.  The two solves are iterated in a
partitioned (staggered) fixed point: electrical solve with the current
conductivity field, thermal solve with the resulting Joule heat, update the
conductivity from the element temperatures, repeat.  Each solve is warm-
started from the previous iterate, so late iterations cost a fraction of a
cold solve, and Aitken relaxation is available for the strongly coupled case
where the plain iteration stalls or diverges (thermal runaway under constant
current).

Both meshes have to share the in-plane element grid; ``layer_slabs`` names the
thermal slab that holds each electrical copper layer.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

from electrical.matrix_free_mpir_fem import (
    MPIRConfig,
    PCBConductionProblem,
    PCBConductionSolution,
    RuntimeBackend,
    solve_pcb_dc,
)
from thermal.matrix_free_mpir_fem import (
    ConvectionBoundary,
    HeatSource,
    LayeredThermalMesh,
    ThermalConductionProblem,
    ThermalConductionSolution,
    element_joule_heat_w,
    solve_thermal_conduction,
    via_joule_heat_sources,
)


COPPER_TEMPERATURE_COEFFICIENT_PER_K = 3.93e-3


@dataclass(frozen=True)
class ElectroThermalScenario:
    """One board, its electrical drive, its thermal environment, and ρ(T)."""

    electrical: PCBConductionProblem
    thermal_mesh: LayeredThermalMesh
    layer_slabs: tuple[int, ...]
    convection: tuple[ConvectionBoundary, ...] = ()
    fixed_temperature_mask: np.ndarray | None = None
    fixed_temperature_k: float | np.ndarray | None = None
    extra_heat_sources: tuple[HeatSource, ...] = ()
    extra_element_heat_w: np.ndarray | None = None
    conductivity_reference_temperature_k: float = 293.15
    temperature_coefficient_per_k: float = COPPER_TEMPERATURE_COEFFICIENT_PER_K

    def __post_init__(self) -> None:
        layers, rows, cols = self.electrical.mesh.element_active.shape
        slabs, thermal_rows, thermal_cols = self.thermal_mesh.element_grid_shape
        if (rows, cols) != (thermal_rows, thermal_cols):
            raise ValueError("electrical and thermal meshes must share (rows, cols)")
        mapping = tuple(int(index) for index in self.layer_slabs)
        if len(mapping) != layers or any(not 0 <= index < slabs for index in mapping):
            raise ValueError("layer_slabs must name one thermal slab per electrical layer")
        if not np.isfinite(self.temperature_coefficient_per_k) or self.temperature_coefficient_per_k < 0.0:
            raise ValueError("temperature_coefficient_per_k must be finite and non-negative")
        object.__setattr__(self, "layer_slabs", mapping)
        object.__setattr__(self, "convection", tuple(self.convection))
        object.__setattr__(self, "extra_heat_sources", tuple(self.extra_heat_sources))
        # Validate the thermal boundary conditions once, without heat.
        ThermalConductionProblem(
            self.thermal_mesh,
            convection=self.convection,
            fixed_temperature_mask=self.fixed_temperature_mask,
            fixed_temperature_k=self.fixed_temperature_k,
            heat_sources=self.extra_heat_sources,
            element_heat_w=self.extra_element_heat_w,
        )


@dataclass(frozen=True)
class CouplingConfig:
    """Fixed-point limits for a staggered coupling."""

    max_iterations: int = 20
    temperature_tolerance_k: float = 1.0e-3
    relative_loss_tolerance: float = 1.0e-6
    relaxation: float = 1.0
    aitken: bool = True
    max_relaxation: float = 4.0
    electrical: MPIRConfig | None = None
    thermal: MPIRConfig | None = None

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.temperature_tolerance_k <= 0.0 or self.relative_loss_tolerance <= 0.0:
            raise ValueError("tolerances must be positive")
        if not 0.0 < self.relaxation <= self.max_relaxation:
            raise ValueError("relaxation must lie in (0, max_relaxation]")
        if self.max_relaxation < 1.0:
            raise ValueError("max_relaxation must be at least one")


@dataclass(frozen=True)
class CouplingStep:
    iteration: int
    joule_loss_w: float
    max_temperature_k: float
    temperature_change_k: float
    relative_loss_change: float
    relaxation: float
    electrical_inner_iterations: int
    thermal_inner_iterations: int


@dataclass(frozen=True)
class ElectroThermalResult:
    electrical: PCBConductionSolution
    thermal: ThermalConductionSolution
    conductivity_s_per_m: np.ndarray
    via_resistance_ohm: np.ndarray
    element_temperature_k: np.ndarray
    converged: bool
    iterations: int
    history: tuple[CouplingStep, ...]
    cold_joule_loss_w: float

    @property
    def loss_increase_ratio(self) -> float:
        return self.electrical.joule_loss_w / self.cold_joule_loss_w if self.cold_joule_loss_w else float("nan")


def slab_element_temperature_k(
    temperature_k: np.ndarray, thermal_mesh: LayeredThermalMesh, slab: int
) -> np.ndarray:
    """Mean corner temperature of every element in one thermal slab."""

    grid = np.asarray(temperature_k, dtype=np.float64).reshape(thermal_mesh.node_shape)
    lower, upper = grid[slab], grid[slab + 1]
    return 0.125 * (
        lower[:-1, :-1] + lower[:-1, 1:] + lower[1:, :-1] + lower[1:, 1:]
        + upper[:-1, :-1] + upper[:-1, 1:] + upper[1:, :-1] + upper[1:, 1:]
    )


def electrical_layer_temperature_k(
    temperature_k: np.ndarray, thermal_mesh: LayeredThermalMesh, layer_slabs: tuple[int, ...]
) -> np.ndarray:
    """Element temperature of each electrical layer, shape ``(layers, rows, cols)``."""

    return np.stack(
        [slab_element_temperature_k(temperature_k, thermal_mesh, slab) for slab in layer_slabs]
    )


def via_node_temperature_k(
    temperature_k: np.ndarray,
    thermal_mesh: LayeredThermalMesh,
    layer_slabs: tuple[int, ...],
    problem: PCBConductionProblem,
) -> np.ndarray:
    """Mean temperature over the two endpoints of every via, each endpoint being
    the average of the thermal node faces bounding its copper slab."""

    grid = np.asarray(temperature_k, dtype=np.float64).reshape(thermal_mesh.node_shape)
    values = []
    for via in problem.vias:
        endpoint_temperatures = []
        for layer, row, col in (via.lower, via.upper):
            slab = layer_slabs[layer]
            endpoint_temperatures.append(0.5 * (grid[slab, row, col] + grid[slab + 1, row, col]))
        values.append(0.5 * sum(endpoint_temperatures))
    return np.asarray(values, dtype=np.float64)


def conductivity_at_temperature(
    reference_conductivity: np.ndarray,
    temperature_k: np.ndarray,
    *,
    reference_temperature_k: float,
    coefficient_per_k: float,
) -> np.ndarray:
    factor = 1.0 + coefficient_per_k * (np.asarray(temperature_k) - reference_temperature_k)
    if np.any(factor <= 0.0):
        raise ValueError("the linear resistivity model left the valid range (σ ≤ 0)")
    return np.asarray(reference_conductivity) / factor


def _thermal_problem(
    scenario: ElectroThermalScenario, electrical: PCBConductionSolution
) -> ThermalConductionProblem:
    heat = element_joule_heat_w(electrical, scenario.thermal_mesh, scenario.layer_slabs)
    if scenario.extra_element_heat_w is not None:
        heat = heat + np.asarray(scenario.extra_element_heat_w, dtype=np.float64)
    sources = via_joule_heat_sources(
        scenario.electrical, electrical, scenario.thermal_mesh, scenario.layer_slabs
    )
    return ThermalConductionProblem(
        scenario.thermal_mesh,
        convection=scenario.convection,
        fixed_temperature_mask=scenario.fixed_temperature_mask,
        fixed_temperature_k=scenario.fixed_temperature_k,
        heat_sources=sources + scenario.extra_heat_sources,
        element_heat_w=heat,
    )


def run_electro_thermal(
    scenario: ElectroThermalScenario,
    *,
    config: CouplingConfig | None = None,
    backend: RuntimeBackend | None = None,
    device_id: int = 0,
) -> ElectroThermalResult:
    """Iterate electrical and thermal solves to a self-consistent ρ(T) state."""

    config = config or CouplingConfig()
    base_mesh = scenario.electrical.mesh
    reference_conductivity = np.asarray(base_mesh.conductivity_s_per_m, dtype=np.float64)
    conductivity = reference_conductivity.copy()
    reference_vias = scenario.electrical.vias
    vias = reference_vias

    history: list[CouplingStep] = []
    potential: np.ndarray | None = None
    temperature: np.ndarray | None = None
    previous_increment: np.ndarray | None = None
    minimum_factor = 1.0e-3  # keep 1 + α (T - T_ref) safely positive
    relaxation = config.relaxation
    cold_loss = float("nan")
    electrical_solution: PCBConductionSolution | None = None
    thermal_solution: ThermalConductionSolution | None = None
    converged = False
    previous_loss = float("nan")

    for iteration in range(1, config.max_iterations + 1):
        problem = dataclasses.replace(
            scenario.electrical,
            mesh=dataclasses.replace(base_mesh, conductivity_s_per_m=conductivity),
            vias=vias,
        )
        electrical_solution = solve_pcb_dc(
            problem,
            config=config.electrical,
            backend=backend,
            device_id=device_id,
            initial_potential_v=potential,
        )
        potential = electrical_solution.potential_v
        loss = electrical_solution.joule_loss_w
        if iteration == 1:
            cold_loss = loss

        thermal_solution = solve_thermal_conduction(
            _thermal_problem(scenario, electrical_solution),
            config=config.thermal,
            backend=backend,
            device_id=device_id,
            initial_temperature_k=temperature,
        )
        proposed = thermal_solution.temperature_k

        # Relaxed fixed-point update on the temperature field.  Aitken's Δ²
        # estimate rescales the relaxation from two successive increments: a
        # slowly contracting iteration (loss rising with temperature under
        # constant current) gets ω > 1, an oscillating one gets ω < 1.
        if temperature is None:
            increment = None
            temperature = proposed
            change = float("inf")
        else:
            increment = proposed - temperature
            if config.aitken and previous_increment is not None:
                difference = increment - previous_increment
                denominator = float(np.dot(difference.reshape(-1), difference.reshape(-1)))
                if denominator > 0.0:
                    relaxation = -relaxation * float(
                        np.dot(previous_increment.reshape(-1), difference.reshape(-1))
                    ) / denominator
                    relaxation = float(np.clip(relaxation, 0.05, config.max_relaxation))
            candidate = temperature + relaxation * increment
            if scenario.temperature_coefficient_per_k > 0.0:
                # The linear model 1 + α (T - T_ref) turns non-positive only far
                # below the reference; an over-relaxed undershoot must not get
                # there.  Fall back to the plain step when it would.
                floor = (
                    scenario.conductivity_reference_temperature_k
                    - (1.0 - minimum_factor) / scenario.temperature_coefficient_per_k
                )
                if float(np.min(candidate)) < floor <= float(np.min(proposed)):
                    relaxation = 1.0
                    candidate = proposed
            change = float(np.max(np.abs(candidate - temperature)))
            temperature = candidate
        previous_increment = increment

        relative_loss_change = (
            abs(loss - previous_loss) / abs(loss) if np.isfinite(previous_loss) and loss else float("inf")
        )
        previous_loss = loss
        history.append(
            CouplingStep(
                iteration=iteration,
                joule_loss_w=loss,
                max_temperature_k=float(np.max(temperature)),
                temperature_change_k=change,
                relative_loss_change=relative_loss_change,
                relaxation=relaxation,
                electrical_inner_iterations=electrical_solution.solve.inner_iterations,
                thermal_inner_iterations=thermal_solution.solve.inner_iterations,
            )
        )
        if scenario.temperature_coefficient_per_k == 0.0:
            converged = electrical_solution.solve.converged and thermal_solution.solve.converged
            break
        if (
            change <= config.temperature_tolerance_k
            and relative_loss_change <= config.relative_loss_tolerance
            and electrical_solution.solve.converged
            and thermal_solution.solve.converged
        ):
            converged = True
            break

        layer_temperature = electrical_layer_temperature_k(
            temperature, scenario.thermal_mesh, scenario.layer_slabs
        )
        conductivity = conductivity_at_temperature(
            reference_conductivity,
            layer_temperature,
            reference_temperature_k=scenario.conductivity_reference_temperature_k,
            coefficient_per_k=scenario.temperature_coefficient_per_k,
        )
        # Via barrels are copper too: their resistance follows the same law at
        # the temperature of their endpoints.
        via_temperature = via_node_temperature_k(
            temperature, scenario.thermal_mesh, scenario.layer_slabs, scenario.electrical
        )
        via_factor = 1.0 + scenario.temperature_coefficient_per_k * (
            via_temperature - scenario.conductivity_reference_temperature_k
        )
        vias = tuple(
            dataclasses.replace(via, resistance_ohm=via.resistance_ohm * float(factor))
            for via, factor in zip(reference_vias, via_factor)
        )

    assert electrical_solution is not None and thermal_solution is not None and temperature is not None
    layer_temperature = electrical_layer_temperature_k(
        temperature, scenario.thermal_mesh, scenario.layer_slabs
    )
    return ElectroThermalResult(
        electrical=electrical_solution,
        thermal=thermal_solution,
        conductivity_s_per_m=conductivity,
        via_resistance_ohm=np.asarray([via.resistance_ohm for via in vias], dtype=np.float64),
        element_temperature_k=layer_temperature,
        converged=converged,
        iterations=len(history),
        history=tuple(history),
        cold_joule_loss_w=cold_loss,
    )
