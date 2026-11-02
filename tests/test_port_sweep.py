"""The copper's Z(f) over a band from a few solves, checked at the midpoints it interpolates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pytest

from electrical.port_network import (
    BoardNetwork,
    CurrentSource,
    Part,
    PortImpedanceSweep,
    capacitor,
    refine_frequencies,
)
from electrical.sheet_peec import solve_port_impedance_problem
from test_port_impedance_contract import _problem


@dataclass(frozen=True)
class Copper:
    ports: tuple
    references: dict
    z_ohm: np.ndarray
    frequency_hz: float


PORTS, REFERENCES = ("A", "B"), {"A": "G", "B": "G"}


def _synthetic(f: float) -> Copper:
    """Skin-effect R (DC value, then sqrt f) and an L falling from 12 to 9 nH around 1 MHz."""
    omega = 2 * math.pi * f
    r = np.array([[2e-3, 4e-4], [4e-4, 3e-3]]) * math.sqrt(1 + f / 2e5)
    l = np.array([[9e-9, 5e-9], [5e-9, 8e-9]]) + np.array([[3e-9, 1e-9], [1e-9, 2e-9]]) / (1 + (f / 1e6) ** 2) ** 0.25
    return Copper(PORTS, REFERENCES, r + 1j * omega * l, f)


def _errors(sweep: PortImpedanceSweep, f: float) -> tuple[float, float]:
    exact, guess = _synthetic(f).z_ohm, sweep.at(f).z_ohm
    return (float(np.abs(guess.real - exact.real).max() / np.abs(exact.real).max()),
            float(np.abs(guess.imag - exact.imag).max() / np.abs(exact.imag).max()))


def test_refinement_checks_each_interval_and_meets_its_tolerance_between_the_checks() -> None:
    calls = []

    def solve(f):
        calls.append(f)
        return _synthetic(f)

    sweep = refine_frequencies(solve, [1e4, 1e8], tolerance=2e-3)
    assert sweep.converged
    assert len(calls) == len(sweep.points) and len(calls) < 24
    assert all(c.accepted or c.resistance_error > 2e-3 or c.inductance_error > 2e-3 for c in sweep.checks)
    # anywhere in the band, not only at the checks
    worst = max(max(_errors(sweep, f)) for f in np.logspace(4.01, 7.99, 60))
    assert worst < 2e-3
    # at a solved frequency the solve itself, flagged as such
    solved = sweep.at(sweep.solved_frequencies_hz[1])
    assert not solved.interpolated
    assert sweep.at(3.3e5).interpolated
    with pytest.raises(ValueError, match="outside the solved band"):
        sweep.at(2e8)


def test_the_forms_are_exact_for_what_they_assume() -> None:
    # R = a + b sqrt(f) and a constant L: two points describe the band
    def solve(f):
        omega = 2 * math.pi * f
        return Copper(("A",), {"A": "G"}, np.array([[1e-3 + 2e-6 * math.sqrt(f) + 1j * omega * 5e-9]]), f)

    sweep = refine_frequencies(solve, [1e3, 1e9], tolerance=1e-9)
    assert sweep.converged and len(sweep.points) == 3  # one check, accepted
    f = 4.4e6
    np.testing.assert_allclose(sweep.at(f).z_ohm, solve(f).z_ohm, rtol=1e-12)


def test_a_capped_refinement_says_it_did_not_converge() -> None:
    sweep = refine_frequencies(_synthetic, [1e4, 1e8], tolerance=1e-6, maximum_solves=5)
    assert len(sweep.points) == 5 and sweep.converged is False


def test_a_sweep_round_trips_through_json() -> None:
    sweep = refine_frequencies(_synthetic, [1e4, 1e8], tolerance=1e-2)
    again = PortImpedanceSweep.from_mapping(json.loads(json.dumps(sweep.to_mapping())))
    assert again.solved_frequencies_hz == sweep.solved_frequencies_hz
    assert again.checks == sweep.checks and again.converged == sweep.converged
    np.testing.assert_array_equal(again.at(7.7e5).z_ohm, sweep.at(7.7e5).z_ohm)


def test_a_network_on_a_sweep_solves_harmonics_it_did_not_solve_and_says_so() -> None:
    sweep = refine_frequencies(_synthetic, [1e5, 1e7], tolerance=1e-3)
    parts = [Part("C1", "A", "G", capacitor(1e-6, 2e-3, 0.5e-9)), Part("C2", "B", "G", capacitor(10e-6, 5e-3, 1e-9))]
    sources = [CurrentSource("Q", "B", "A", {})]
    network = BoardNetwork(sweep, parts, ground="G", current_sources=sources)
    harmonics = {f"h{k}": 3.3e5 * k for k in range(1, 20)}
    spectra = network.solve_cases({"run": {"Q": {f: 1.0 / k for k, f in enumerate(harmonics.values(), start=1)}}})
    run = spectra["run"]
    assert all(s.copper_interpolated for s in run.solutions)
    # each harmonic against the network on the exact copper there
    for f, solution in zip(run.frequency_hz, run.solutions):
        exact = BoardNetwork([_synthetic(f)], parts, ground="G",
                             current_sources=[CurrentSource("Q", "B", "A", {f: solution_drive(f, harmonics)})]).solve(f)
        assert solution.part_current_a["C1"] == pytest.approx(exact.part_current_a["C1"], rel=2e-3, abs=1e-9)
    solved_here = network.solve(sweep.solved_frequencies_hz[0])
    assert not solved_here.copper_interpolated
    with pytest.raises(ValueError, match="outside the solved band"):
        network.solve(2e7)
    # copper given as matrices still refuses what it was not solved at
    with pytest.raises(ValueError, match="no copper matrix"):
        BoardNetwork(list(sweep.points), parts, ground="G", current_sources=sources).solve(3.3e5)


def solution_drive(f, harmonics):
    k = list(harmonics.values()).index(f) + 1
    return 1.0 / k


def test_the_contract_refines_the_band_of_two_strips_and_the_unsolved_points_hold() -> None:
    # skin filaments on: R and L move with frequency over the band
    settings = {"maximum_filaments": 6, "relative_tolerance": 1e-10,
                "frequency_refinement": {"relative_tolerance": 0.01, "maximum_solves": 24}}
    result = solve_port_impedance_problem(_problem(frequencies=(1e5, 1e8)), settings)
    refinement = result.refinement
    assert refinement["converged"] and refinement["requested_hz"] == [1e5, 1e8]
    assert len(result.impedances) == len(refinement["checks"]) + 2
    sweep = result.sweep()
    # unsolved points, solved directly now, against the interpolation
    check = solve_port_impedance_problem(_problem(frequencies=(2.7e5, 4.1e6, 3.3e7)), {"maximum_filaments": 6,
                                                                                      "relative_tolerance": 1e-10})
    for direct in check.impedances:
        guess = sweep.at(direct.frequency_hz)
        assert guess.interpolated
        z = direct.z_ohm
        assert np.abs(guess.z_ohm.real - z.real).max() <= 0.02 * np.abs(z.real).max()
        assert np.abs(guess.z_ohm.imag - z.imag).max() <= 0.02 * np.abs(z.imag).max()
    again = type(result).from_mapping(json.loads(json.dumps(result.to_mapping())))
    assert again.refinement == refinement
    with pytest.raises(ValueError, match="frequency_refinement takes"):
        solve_port_impedance_problem(_problem(frequencies=(1e5, 1e8)), {"frequency_refinement": {"tol": 0.1}})


def test_frequencies_match_within_their_rounding_not_to_the_bit() -> None:
    from electrical.port_network import FREQUENCY_RTOL, distinct_frequencies, match_frequency

    f = 1e6
    assert match_frequency(f * (1 + 1e-13), [1e5, f, 1e7]) == f
    assert match_frequency(f * (1 + 10 * FREQUENCY_RTOL), [f]) is None
    assert distinct_frequencies([3e6, 1e6 * (1 + 1e-14), 1e6, 2e6]) == [1e6 * (1 + 1e-14), 2e6, 3e6]
    sweep = refine_frequencies(_synthetic, [1e4, 1e8], tolerance=1e-2)
    middle = sweep.solved_frequencies_hz[1]  # a geometric midpoint, sqrt(low * high)
    near = sweep.at(middle * (1 + 1e-13))
    assert not near.interpolated and near.frequency_hz == middle


def test_an_interpolated_matrix_says_whether_its_solves_converged() -> None:
    @dataclass(frozen=True)
    class Flagged(Copper):
        converged: tuple = (True, True)

    def solve(f):
        c = _synthetic(f)
        return Flagged(c.ports, c.references, c.z_ohm, f, (True, f < 5e7))

    sweep = refine_frequencies(solve, [1e4, 1e8], tolerance=1e-2)
    assert sweep.at(1e4).converged is True and sweep.at(1e8).converged is False
    assert sweep.at(3.3e5).interpolated and sweep.at(3.3e5).converged is False
    assert refine_frequencies(_synthetic, [1e4, 1e8], tolerance=1e-2).at(3.3e5).converged is None
