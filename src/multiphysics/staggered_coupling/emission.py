"""Electrical solve to radiated-emission spectrum.

Two source paths feed the same evaluation:

- a DC conduction solution used as a phasor at every frequency (quasi-static,
  one electrical solve for the whole sweep), which is what the electro-thermal
  chain hands over after the ρ(T) iteration has converged;
- a sheet-PEEC case solved at every frequency, which resolves the frequency
  dependence of the current distribution and needs no quasi-static assumption.

For each frequency the evaluation reports the far-field pattern at the limit's
distance, the margin to the chosen limit line, the dipole moments, and, when a
scan plane is given, the near-field maxima.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from electrical.matrix_free_mpir_fem import PCBConductionProblem, PCBConductionSolution
from emc.tiled_dipole_superposition import (
    CISPR32_CLASS_B,
    CurrentDipoles,
    DipoleMoments,
    EmissionLimit,
    EmissionMargin,
    FarFieldPattern,
    FieldSamples,
    dipole_moments,
    dipoles_from_pcb_dc,
    dipoles_from_sheet_peec,
    emission_margin,
    evaluate_fields,
    far_field_pattern,
    scan_plane,
)
from emc.tiled_dipole_superposition.fields import Backend


@dataclass(frozen=True)
class ScanPlane:
    """A horizontal near-field scan: probe positions in m."""

    x_m: np.ndarray
    y_m: np.ndarray
    z_m: float

    def points(self) -> np.ndarray:
        return scan_plane(self.x_m, self.y_m, self.z_m)


@dataclass(frozen=True)
class EmissionScenario:
    """What to evaluate and against which limit."""

    frequencies_hz: tuple[float, ...]
    limit: EmissionLimit = CISPR32_CLASS_B
    distance_m: float = 10.0
    close_terminals: bool = True
    ground_plane_z_m: float | None = None
    scan: ScanPlane | None = None

    def __post_init__(self) -> None:
        frequencies = tuple(float(value) for value in self.frequencies_hz)
        if not frequencies or any(not np.isfinite(f) or f <= 0.0 for f in frequencies):
            raise ValueError("frequencies_hz must be positive and finite")
        if self.distance_m <= 0.0:
            raise ValueError("distance_m must be positive")
        object.__setattr__(self, "frequencies_hz", frequencies)


@dataclass(frozen=True)
class EmissionPoint:
    frequency_hz: float
    pattern: FarFieldPattern
    margin: EmissionMargin
    moments: DipoleMoments
    near_field: FieldSamples | None

    @property
    def max_near_magnetic_a_per_m(self) -> float:
        return float(np.max(self.near_field.magnetic_magnitude_a_per_m)) if self.near_field is not None else float("nan")


@dataclass(frozen=True)
class EmissionResult:
    scenario: EmissionScenario
    dipoles: tuple[CurrentDipoles, ...]        # one per frequency (shared object for quasi-static)
    points: tuple[EmissionPoint, ...]

    @property
    def frequencies_hz(self) -> np.ndarray:
        return np.asarray([point.frequency_hz for point in self.points])

    @property
    def predicted_dbuv_per_m(self) -> np.ndarray:
        return np.asarray([point.margin.predicted_dbuv_per_m for point in self.points])

    @property
    def limit_dbuv_per_m(self) -> np.ndarray:
        return np.asarray([point.margin.limit_dbuv_per_m for point in self.points])

    @property
    def margin_db(self) -> np.ndarray:
        return np.asarray([point.margin.margin_db for point in self.points])

    @property
    def worst_margin_db(self) -> float:
        return float(np.min(self.margin_db))

    @property
    def compliant(self) -> bool:
        return bool(np.all(self.margin_db >= 0.0))


def evaluate_emission(
    dipoles_per_frequency: Sequence[CurrentDipoles],
    scenario: EmissionScenario,
    *,
    backend: Backend = "cpu",
) -> EmissionResult:
    """Evaluate one current distribution per frequency against the scenario."""

    if len(dipoles_per_frequency) != len(scenario.frequencies_hz):
        raise ValueError("one current distribution per frequency is required")
    points: list[EmissionPoint] = []
    probe = scenario.scan.points() if scenario.scan is not None else None
    for frequency, dipoles in zip(scenario.frequencies_hz, dipoles_per_frequency):
        if scenario.ground_plane_z_m is not None:
            dipoles = dipoles.with_ground_plane_images(scenario.ground_plane_z_m)
        pattern = far_field_pattern(dipoles, frequency, distance_m=scenario.distance_m, backend=backend)
        margin = emission_margin(
            pattern.max_polarised_field_v_per_m, frequency, scenario.limit, distance_m=scenario.distance_m
        )
        near = evaluate_fields(dipoles, probe, frequency, backend=backend) if probe is not None else None
        points.append(
            EmissionPoint(
                frequency_hz=frequency,
                pattern=pattern,
                margin=margin,
                moments=dipole_moments(dipoles, frequency),
                near_field=near,
            )
        )
    return EmissionResult(scenario=scenario, dipoles=tuple(dipoles_per_frequency), points=tuple(points))


def run_pcb_dc_emission(
    problem: PCBConductionProblem,
    solution: PCBConductionSolution,
    layer_height_m: Sequence[float],
    scenario: EmissionScenario,
    *,
    backend: Backend = "cpu",
) -> EmissionResult:
    """Quasi-static sweep: one DC current pattern read as a phasor at every frequency."""

    dipoles = dipoles_from_pcb_dc(
        problem, solution, layer_height_m, close_terminals=scenario.close_terminals
    )
    return evaluate_emission([dipoles] * len(scenario.frequencies_hz), scenario, backend=backend)


def run_sheet_peec_emission(
    mesh: Any,
    operator: Any,
    terminals: Sequence[Any],
    scenario: EmissionScenario,
    *,
    backend: Backend = "cpu",
    **solve_options: Any,
) -> tuple[EmissionResult, tuple[Any, ...]]:
    """Frequency-resolved sweep: solve the sheet-PEEC case at every frequency.

    Returns the emission result and the sheet solutions in frequency order.
    Terminal closure is not applied: the sheet solver's terminals are lumped
    current sources whose external return is likewise absent, so ``|P|`` in the
    result's moments reports what that omission is worth.
    """

    from electrical.sheet_peec.sheet_peec import solve_sheet_case

    solutions = []
    dipoles = []
    for frequency in scenario.frequencies_hz:
        solution = solve_sheet_case(mesh, operator, terminals, frequency_hz=frequency, **solve_options)
        solutions.append(solution)
        dipoles.append(dipoles_from_sheet_peec(mesh, solution))
    return evaluate_emission(dipoles, scenario, backend=backend), tuple(solutions)
