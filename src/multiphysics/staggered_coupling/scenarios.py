"""One entry point for every analysis scenario the repository supports.

A scenario is a frozen dataclass naming what is coupled to what; ``run_scenario``
dispatches on its type and returns the matching result.  The chains are:

| Scenario | Solves | Couples |
|---|---|---|
| ``ElectricalScenario`` | DC conduction | — |
| ``ThermalScenario`` | heat conduction | — |
| ``ThermalTransientScenario`` | backward-Euler heat conduction over a time schedule | — |
| ``ElectroThermalScenario`` | both, iterated | Joule heat → T → σ(T) |
| ``ElectroEmissionScenario`` | DC conduction, dipole fields | J → radiated field |
| ``ElectroThermalEmissionScenario`` | all three | σ(T)-converged J → radiated field |
| ``SheetPeecEmissionScenario`` | sheet PEEC per frequency, dipole fields | J(f) → radiated field |
| ``BoardEnclosureThermalScenario`` | board and body conduction, iterated | contact heat ↔ contact temperature |
| ``ElectroThermalEnclosureScenario`` | DC conduction, board and bodies, iterated | Joule heat → T (board ↔ bodies) → σ(T) |

The EMC step never feeds back: radiation at these levels does not change the
currents.  The thermal step feeds back through the copper resistivity only.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Sequence

from electrical.matrix_free_mpir_fem import (
    MPIRConfig,
    PCBConductionProblem,
    PCBConductionSolution,
    solve_pcb_dc,
)
from thermal.matrix_free_mpir_fem import (
    ThermalConductionProblem,
    ThermalConductionSolution,
    TimeSchedule,
    TransientThermalSolution,
    solve_thermal_conduction,
    solve_thermal_transient,
)

from .board_enclosure import (
    BoardEnclosureThermalResult,
    BoardEnclosureThermalScenario,
    run_board_enclosure_thermal,
)
from .electro_thermal import (
    CouplingConfig,
    ElectroThermalResult,
    ElectroThermalScenario,
    run_electro_thermal,
)
from .electro_thermal_enclosure import (
    ElectroThermalEnclosureResult,
    ElectroThermalEnclosureScenario,
    run_electro_thermal_enclosure,
)
from .emission import (
    EmissionResult,
    EmissionScenario,
    run_pcb_dc_emission,
    run_sheet_peec_emission,
)


@dataclass(frozen=True)
class ElectricalScenario:
    problem: PCBConductionProblem
    config: MPIRConfig | None = None


@dataclass(frozen=True)
class ThermalScenario:
    problem: ThermalConductionProblem
    config: MPIRConfig | None = None


@dataclass(frozen=True)
class ThermalTransientScenario:
    """Heat conduction marched from ``initial_temperature_k`` through ``schedule``."""

    problem: ThermalConductionProblem
    schedule: TimeSchedule
    initial_temperature_k: Any = None
    until_steady: bool = False
    store: str = "all"
    config: MPIRConfig | None = None


@dataclass(frozen=True)
class ElectroEmissionScenario:
    """DC conduction, then a quasi-static emission sweep of its current."""

    electrical: PCBConductionProblem
    layer_height_m: tuple[float, ...]
    emission: EmissionScenario
    config: MPIRConfig | None = None


@dataclass(frozen=True)
class ElectroThermalEmissionScenario:
    """Electro-thermal iteration, then emission of the ρ(T)-converged current."""

    electro_thermal: ElectroThermalScenario
    layer_height_m: tuple[float, ...]
    emission: EmissionScenario
    coupling: CouplingConfig | None = None


@dataclass(frozen=True)
class SheetPeecEmissionScenario:
    """Sheet-PEEC solves at each frequency, then their emission."""

    mesh: Any
    operator: Any
    terminals: tuple[Any, ...]
    emission: EmissionScenario
    solve_options: dict[str, Any] | None = None


@dataclass(frozen=True)
class ElectroEmissionResult:
    electrical: PCBConductionSolution
    emission: EmissionResult


@dataclass(frozen=True)
class ElectroThermalEmissionResult:
    electro_thermal: ElectroThermalResult
    emission: EmissionResult
    cold_emission: EmissionResult

    @property
    def heating_shift_db(self) -> Any:
        """Change of the predicted field from the cold to the ρ(T)-converged current."""

        return self.emission.predicted_dbuv_per_m - self.cold_emission.predicted_dbuv_per_m


@dataclass(frozen=True)
class SheetPeecEmissionResult:
    solutions: tuple[Any, ...]
    emission: EmissionResult


@singledispatch
def run_scenario(scenario: Any, *, backend: str | None = None, device_id: int = 0) -> Any:
    """Run any scenario dataclass and return its result."""

    raise TypeError(f"unsupported scenario type {type(scenario).__name__}")


@run_scenario.register
def _(scenario: ElectricalScenario, *, backend: str | None = None, device_id: int = 0) -> PCBConductionSolution:
    return solve_pcb_dc(scenario.problem, config=scenario.config, backend=backend, device_id=device_id)


@run_scenario.register
def _(scenario: ThermalScenario, *, backend: str | None = None, device_id: int = 0) -> ThermalConductionSolution:
    return solve_thermal_conduction(
        scenario.problem, config=scenario.config, backend=backend, device_id=device_id
    )


@run_scenario.register
def _(scenario: ThermalTransientScenario, *, backend: str | None = None, device_id: int = 0) -> TransientThermalSolution:
    return solve_thermal_transient(
        scenario.problem,
        scenario.schedule,
        initial_temperature_k=scenario.initial_temperature_k,
        until_steady=scenario.until_steady,
        store=scenario.store,
        config=scenario.config,
        backend=backend,
        device_id=device_id,
    )


@run_scenario.register
def _(scenario: ElectroThermalScenario, *, backend: str | None = None, device_id: int = 0) -> ElectroThermalResult:
    return run_electro_thermal(scenario, backend=backend, device_id=device_id)


@run_scenario.register
def _(
    scenario: BoardEnclosureThermalScenario, *, backend: str | None = None, device_id: int = 0
) -> BoardEnclosureThermalResult:
    return run_board_enclosure_thermal(scenario, backend=backend, device_id=device_id)


@run_scenario.register
def _(
    scenario: ElectroThermalEnclosureScenario, *, backend: str | None = None, device_id: int = 0
) -> ElectroThermalEnclosureResult:
    return run_electro_thermal_enclosure(scenario, backend=backend, device_id=device_id)


def _emission_backend(backend: str | None) -> str:
    return backend or "cpu"


@run_scenario.register
def _(scenario: ElectroEmissionScenario, *, backend: str | None = None, device_id: int = 0) -> ElectroEmissionResult:
    electrical = solve_pcb_dc(
        scenario.electrical, config=scenario.config, backend=backend, device_id=device_id
    )
    emission = run_pcb_dc_emission(
        scenario.electrical,
        electrical,
        scenario.layer_height_m,
        scenario.emission,
        backend=_emission_backend(backend),
    )
    return ElectroEmissionResult(electrical=electrical, emission=emission)


@run_scenario.register
def _(
    scenario: ElectroThermalEmissionScenario, *, backend: str | None = None, device_id: int = 0
) -> ElectroThermalEmissionResult:
    coupled = run_electro_thermal(
        scenario.electro_thermal, config=scenario.coupling, backend=backend, device_id=device_id
    )
    cold = solve_pcb_dc(
        scenario.electro_thermal.electrical,
        config=(scenario.coupling or CouplingConfig()).electrical,
        backend=backend,
        device_id=device_id,
    )
    emission_backend = _emission_backend(backend)
    hot_emission = run_pcb_dc_emission(
        scenario.electro_thermal.electrical,
        coupled.electrical,
        scenario.layer_height_m,
        scenario.emission,
        backend=emission_backend,
    )
    cold_emission = run_pcb_dc_emission(
        scenario.electro_thermal.electrical,
        cold,
        scenario.layer_height_m,
        scenario.emission,
        backend=emission_backend,
    )
    return ElectroThermalEmissionResult(
        electro_thermal=coupled, emission=hot_emission, cold_emission=cold_emission
    )


@run_scenario.register
def _(scenario: SheetPeecEmissionScenario, *, backend: str | None = None, device_id: int = 0) -> SheetPeecEmissionResult:
    emission, solutions = run_sheet_peec_emission(
        scenario.mesh,
        scenario.operator,
        scenario.terminals,
        scenario.emission,
        backend=_emission_backend(backend),
        **(scenario.solve_options or {}),
    )
    return SheetPeecEmissionResult(solutions=solutions, emission=emission)


def run_scenarios(
    scenarios: Sequence[Any], *, backend: str | None = None, device_id: int = 0
) -> tuple[Any, ...]:
    """Run several scenarios in order and return their results as a tuple."""

    return tuple(run_scenario(scenario, backend=backend, device_id=device_id) for scenario in scenarios)
