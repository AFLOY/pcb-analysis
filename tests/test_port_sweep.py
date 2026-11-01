"""The copper's Z(f) over a band from a few solves, checked at the midpoints it interpolates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pytest

from electrical.port_network import PortImpedanceSweep, refine_frequencies


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
