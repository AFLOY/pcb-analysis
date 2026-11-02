"""A pad-port impedance over a band, from a few solved frequencies.

The copper's ``Z(f)`` is smooth between resonances it does not have: its
resistance grows from the DC value towards the skin-effect ``sqrt(f)`` law
and its inductance falls slowly and monotonically as the current crowds to
the surfaces and edges.  So a handful of field solves describe a band, if
each part is interpolated in the form it follows:

- ``R = Re Z`` monotone cubic (PCHIP) in ``sqrt(f)``: the skin-effect law is
  a straight line there, and the cubic follows the bend from the DC value
  into it, which a straight line between two points does not;
- ``L = Im Z / omega`` monotone cubic (PCHIP) in ``log f``.

PCHIP neither overshoots nor invents a turning point between the points.
These forms were chosen by measurement (``docs/PORT_SWEEP_RESULTS.json``):
on two strips with skin filaments over 0.1-100 MHz, ``a + b sqrt(f)``
between neighbours (linear in ``sqrt f``) passed its 1 % checks yet missed R
by 1.9 % between them, where these kept within 0.6 % with fewer solves
(15 against 17).  ``forms`` selects another pair for such comparisons.

Every entry of the matrix, couplings across conductors included, is
interpolated alike.  Outside the solved band nothing is returned: an
extrapolated copper is a guess.

:func:`refine_frequencies` decides where to solve.  Each interval between
solved points is checked at its geometric midpoint: the copper is solved
there and compared with what the interpolation over the points solved so far
predicted.  An interval whose prediction misses by more than the tolerance
(in R or in L, each against the largest entry of its own matrix) is split and
its halves checked in turn.  The checks are kept: they are the evidence that
an interpolated value is good to the tolerance, measured before the checked
point was added (after it is, the interpolation there is closer still).

No field solve lives here: the caller passes the solve.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator

from .frequency import distinct_frequencies, match_frequency

PORT_IMPEDANCE_SWEEP_SCHEMA = "port-impedance-sweep/v1"


@dataclass(frozen=True)
class SweepPoint:
    """The copper at one frequency, solved or interpolated."""

    ports: tuple[str, ...]
    references: dict[str, str]
    z_ohm: np.ndarray
    frequency_hz: float
    interpolated: bool


@dataclass(frozen=True)
class FrequencyCheck:
    """One interval's midpoint: solved, and compared with the interpolation that preceded it."""

    low_hz: float
    high_hz: float
    frequency_hz: float
    resistance_error: float
    """max |R_interpolated - R_solved| over max |R_solved|."""
    inductance_error: float
    """max |L_interpolated - L_solved| over max |L_solved|."""
    accepted: bool


def _matrices(points: Sequence[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequency = np.array([float(p.frequency_hz) for p in points])
    z = np.stack([np.asarray(p.z_ohm, dtype=complex) for p in points])
    return frequency, z.real, z.imag / (2.0 * math.pi * frequency)[:, None, None]


DEFAULT_FORMS = ("pchip-sqrt", "pchip-log")
"""How R and L are interpolated: ``<pchip|linear>-<sqrt|log>`` of the frequency."""

_AXES = {"sqrt": math.sqrt, "log": math.log}


def _along(form: str, frequency: np.ndarray, values: np.ndarray, f: float) -> np.ndarray:
    kind, axis = form.split("-")
    if kind not in ("pchip", "linear") or axis not in _AXES:
        raise ValueError(f"unknown interpolation form {form!r}")
    x = np.array([_AXES[axis](v) for v in frequency])
    xq = _AXES[axis](f)
    if kind == "linear" or len(frequency) == 2:
        return np.array([np.interp(xq, x, values[:, k]) for k in range(values.shape[1])])
    return np.asarray(PchipInterpolator(x, values, axis=0)(xq))


def _interpolate(points: Sequence[Any], frequency_hz: float, forms: tuple[str, str] = DEFAULT_FORMS) -> np.ndarray:
    frequency, resistance, inductance = _matrices(points)
    f = float(frequency_hz)
    count = len(points)
    r = _along(forms[0], frequency, resistance.reshape(count, -1), f)
    l = _along(forms[1], frequency, inductance.reshape(count, -1), f)
    shape = resistance.shape[1:]
    return r.reshape(shape) + 1j * 2.0 * math.pi * f * l.reshape(shape)


def _relative_errors(predicted: np.ndarray, solved: np.ndarray, frequency_hz: float) -> tuple[float, float]:
    omega = 2.0 * math.pi * frequency_hz
    r_scale = float(np.abs(solved.real).max()) or 1.0
    l_scale = float(np.abs(solved.imag).max()) or 1.0
    return (
        float(np.abs(predicted.real - solved.real).max()) / r_scale,
        float(np.abs(predicted.imag - solved.imag).max() / omega) / (l_scale / omega),
    )


class PortImpedanceSweep:
    """Solved pad-port impedance matrices and the interpolation between them.

    ``points`` are matrices of one set of ports (anything with ``ports``,
    ``references``, ``z_ohm`` and ``frequency_hz``, such as
    ``electrical.sheet_peec.SheetPortImpedance``).  ``checks`` are the
    midpoint comparisons :func:`refine_frequencies` made; a sweep built from
    points alone has none, and its interpolation is unchecked.
    """

    def __init__(
        self,
        points: Iterable[Any],
        checks: Sequence[FrequencyCheck] = (),
        *,
        tolerance: float | None = None,
        converged: bool | None = None,
        forms: tuple[str, str] = DEFAULT_FORMS,
    ) -> None:
        ordered = sorted(points, key=lambda p: float(p.frequency_hz))
        if len(ordered) < 2:
            raise ValueError("a sweep needs at least two solved frequencies")
        frequencies = [float(p.frequency_hz) for p in ordered]
        if frequencies[0] <= 0.0 or len(distinct_frequencies(frequencies)) != len(frequencies):
            raise ValueError("solved frequencies must be positive and distinct")
        self.ports = tuple(ordered[0].ports)
        self.references = dict(ordered[0].references)
        for point in ordered:
            if tuple(point.ports) != self.ports or dict(point.references) != self.references:
                raise ValueError(f"the matrix at {float(point.frequency_hz):g} Hz has other ports")
        self.points = tuple(ordered)
        self.checks = tuple(checks)
        self.tolerance = tolerance
        self.converged = converged
        self.forms = tuple(forms)
        _along(self.forms[0], np.array([1.0, 2.0]), np.zeros((2, 1)), 1.5)  # refuse an unknown form now
        _along(self.forms[1], np.array([1.0, 2.0]), np.zeros((2, 1)), 1.5)

    @property
    def solved_frequencies_hz(self) -> tuple[float, ...]:
        return tuple(float(p.frequency_hz) for p in self.points)

    @property
    def band_hz(self) -> tuple[float, float]:
        return self.solved_frequencies_hz[0], self.solved_frequencies_hz[-1]

    def at(self, frequency_hz: float) -> SweepPoint:
        """The matrix at ``frequency_hz``: the solved one (within ``FREQUENCY_RTOL``), or interpolated inside the band."""
        f = float(frequency_hz)
        solved = match_frequency(f, self.solved_frequencies_hz)
        if solved is not None:
            point = self.points[self.solved_frequencies_hz.index(solved)]
            return SweepPoint(self.ports, dict(self.references), np.asarray(point.z_ohm, dtype=complex), solved, False)
        low, high = self.band_hz
        if not (low <= f <= high):
            raise ValueError(f"{f:g} Hz is outside the solved band [{low:g}, {high:g}] Hz")
        return SweepPoint(self.ports, dict(self.references), _interpolate(self.points, f, self.forms), f, True)

    def worst_check(self) -> tuple[float, float]:
        """The largest R and L errors among the accepted checks (0 without checks)."""
        accepted = [c for c in self.checks if c.accepted]
        return (max((c.resistance_error for c in accepted), default=0.0),
                max((c.inductance_error for c in accepted), default=0.0))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": PORT_IMPEDANCE_SWEEP_SCHEMA,
            "ports": list(self.ports),
            "references": dict(self.references),
            "tolerance": self.tolerance,
            "converged": self.converged,
            "forms": list(self.forms),
            "points": [
                {"frequency_hz": float(p.frequency_hz),
                 "z_ohm": {"real": np.real(p.z_ohm).tolist(), "imag": np.imag(p.z_ohm).tolist()}}
                for p in self.points
            ],
            "checks": [asdict(c) for c in self.checks],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PortImpedanceSweep":
        if str(value.get("schema") or "") != PORT_IMPEDANCE_SWEEP_SCHEMA:
            raise ValueError(f"unsupported port-impedance sweep schema: {value.get('schema')}")
        ports = tuple(str(p) for p in value["ports"])
        references = {str(k): str(v) for k, v in value["references"].items()}
        points = [
            SweepPoint(ports, dict(references),
                       np.asarray(item["z_ohm"]["real"], dtype=float) + 1j * np.asarray(item["z_ohm"]["imag"], dtype=float),
                       float(item["frequency_hz"]), False)
            for item in value["points"]
        ]
        checks = [FrequencyCheck(**{k: (bool(v) if k == "accepted" else float(v)) for k, v in item.items()})
                  for item in value.get("checks") or []]
        return cls(points, checks, tolerance=value.get("tolerance"), converged=value.get("converged"),
                   forms=tuple(value.get("forms") or DEFAULT_FORMS))


def refine_frequencies(
    solve: Callable[[float], Any],
    frequencies_hz: Sequence[float],
    *,
    tolerance: float = 0.01,
    maximum_solves: int = 24,
    minimum_ratio: float = 1.05,
    forms: tuple[str, str] = DEFAULT_FORMS,
) -> PortImpedanceSweep:
    """Solve at ``frequencies_hz`` and at the midpoints the interpolation needs.

    ``solve(f)`` returns the copper's matrix at ``f``.  Every interval is
    checked at its geometric midpoint until its R and L predictions are within
    ``tolerance`` of the solve there, an interval is narrower than
    ``minimum_ratio`` (then accepted as it stands, and the sweep not
    converged), or ``maximum_solves`` solves have been spent (the rest
    unchecked, the sweep not converged).
    """
    initial = sorted({float(f) for f in frequencies_hz})
    if len(initial) < 2:
        raise ValueError("refinement needs at least two frequencies")
    if not tolerance > 0.0 or not minimum_ratio > 1.0:
        raise ValueError("tolerance must be positive and minimum_ratio above 1")
    solved = {f: solve(f) for f in initial}
    checks: list[FrequencyCheck] = []
    pending = list(zip(initial[:-1], initial[1:]))
    converged = True
    while pending:
        low, high = pending.pop(0)
        if high / low < minimum_ratio:
            converged = False
            continue
        if len(solved) >= maximum_solves:
            converged = False
            break
        middle = math.sqrt(low * high)
        points = [solved[f] for f in sorted(solved)]
        predicted = _interpolate(points, middle, forms)
        actual = solve(middle)
        solved[middle] = actual
        r_error, l_error = _relative_errors(predicted, np.asarray(actual.z_ohm, dtype=complex), middle)
        accepted = r_error <= tolerance and l_error <= tolerance
        checks.append(FrequencyCheck(low, high, middle, r_error, l_error, accepted))
        if not accepted:
            pending.extend([(low, middle), (middle, high)])
    return PortImpedanceSweep(solved.values(), checks, tolerance=tolerance, converged=converged, forms=forms)


__all__ = [
    "DEFAULT_FORMS",
    "PORT_IMPEDANCE_SWEEP_SCHEMA",
    "FrequencyCheck",
    "PortImpedanceSweep",
    "SweepPoint",
    "refine_frequencies",
]
