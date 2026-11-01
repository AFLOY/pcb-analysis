"""Second moments and harmonics of port currents over a simulation window.

A circuit simulator hands back waveforms; the copper's loss needs their
time averages of second order.  The copper is linear, so for any current
waveform ``I_k(t)`` on the ports of one conductor the time-averaged loss of
every copper element follows from ``<I_k I_l>`` alone (``PortExcitation``
of ``multiphysics.staggered_coupling`` takes that matrix).  Averaging the
currents first and squaring afterwards drops the ripple and every cross
term between ports that share copper.

Everything here is arithmetic on sampled waveforms.  The samples are the
simulator's own time points, which are not uniform, so every average is a
trapezoid integral over the window divided by its length.  A harmonic
phasor is the projection ``(2 / T) ∫ I(t) e^{-j ω t} dt`` with the
convention ``I(t) = Re[h e^{j ω t}]``; it is a Fourier coefficient only when
the window spans an integer number of periods, which the caller checks.

KCL closure spreads a group's residual over its active ports in proportion
to their mean current, as one linear projection built from the means and
applied to the means, the second moments (``P C Pᵀ``) and the harmonics
alike, so every quantity closes with the same correction and the
second-moment matrix stays positive semi-definite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class WindowMoments:
    """Means, second moments and harmonic phasors of several ports over one window.

    ``second_moment_a2[k, l] = <I_k I_l>``; ``harmonics[h]`` holds the phasor of
    every port at ``(h + 1) × fundamental_hz``.
    """

    ports: tuple[str, ...]
    start_s: float
    stop_s: float
    samples: int
    mean_a: np.ndarray
    second_moment_a2: np.ndarray
    fundamental_hz: float = 0.0
    harmonics: tuple[np.ndarray, ...] = ()
    closure_by_group: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    @property
    def rms_a(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.second_moment_a2), 0.0))

    @property
    def ripple_variance_a2(self) -> np.ndarray:
        return np.maximum(np.diag(self.second_moment_a2) - self.mean_a**2, 0.0)

    def index(self, port: str) -> int:
        return self.ports.index(port)

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable record; complex phasors become ``{real, imag}``."""

        return {
            "ports": list(self.ports),
            "window_s": {"from": self.start_s, "to": self.stop_s, "samples": self.samples},
            "mean_a": {p: float(v) for p, v in zip(self.ports, self.mean_a)},
            "second_moment_a2": {
                p: {q: float(self.second_moment_a2[i, j]) for j, q in enumerate(self.ports)}
                for i, p in enumerate(self.ports)
            },
            "fundamental_hz": self.fundamental_hz,
            "harmonics": [
                {
                    "order": order + 1,
                    "frequency_hz": (order + 1) * self.fundamental_hz,
                    "phasor_a": {p: {"real": float(z.real), "imag": float(z.imag)} for p, z in zip(self.ports, h)},
                }
                for order, h in enumerate(self.harmonics)
            ],
            "closure_by_group": {r: dict(v) for r, v in self.closure_by_group.items()},
        }


def _trapezoid_weights(time_s: np.ndarray) -> np.ndarray:
    """Weights ``w`` with ``Σ w_i f_i = ∫ f dt`` for piecewise-linear ``f``."""

    gaps = np.diff(time_s)
    weights = np.zeros_like(time_s)
    weights[:-1] += 0.5 * gaps
    weights[1:] += 0.5 * gaps
    return weights


def _window_samples(time_s: np.ndarray, currents_a: np.ndarray, start_s: float, stop_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Samples inside the window, with the two ends interpolated exactly onto it."""

    if stop_s <= start_s:
        raise ValueError("the window must have positive length")
    if start_s < time_s[0] or stop_s > time_s[-1]:
        raise ValueError(
            f"window [{start_s:g}, {stop_s:g}] s lies outside the simulated [{time_s[0]:g}, {time_s[-1]:g}] s"
        )
    inside = (time_s > start_s) & (time_s < stop_s)
    ends = np.array([start_s, stop_s])
    end_values = np.stack([np.interp(ends, time_s, column) for column in currents_a.T], axis=1)
    t = np.concatenate(([start_s], time_s[inside], [stop_s]))
    i = np.vstack((end_values[:1], currents_a[inside], end_values[1:]))
    return t, i


def window_moments(
    ports: Sequence[str],
    time_s: np.ndarray,
    currents_a: np.ndarray,
    start_s: float,
    stop_s: float,
    *,
    fundamental_hz: float = 0.0,
    harmonics: int = 0,
    period_tolerance: float = 1.0e-2,
) -> WindowMoments:
    """Trapezoid time averages of the port currents over ``[start_s, stop_s]``.

    ``currents_a`` is ``(samples, ports)`` on the simulator's time points.
    Harmonics are evaluated only when the window spans an integer number of
    periods of ``fundamental_hz`` within ``period_tolerance``; otherwise the
    record carries none, because a partial window has no Fourier series.
    """

    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    currents_a = np.asarray(currents_a, dtype=np.float64)
    if currents_a.shape != (time_s.size, len(ports)):
        raise ValueError("currents_a must be (samples, ports) on the time axis")
    if np.any(np.diff(time_s) < 0.0):
        raise ValueError("time samples must be non-decreasing")
    t, i = _window_samples(time_s, currents_a, float(start_s), float(stop_s))
    weights = _trapezoid_weights(t)
    length = float(stop_s - start_s)
    mean = (weights @ i) / length
    second = (i.T * weights) @ i / length
    second = 0.5 * (second + second.T)
    phasors: list[np.ndarray] = []
    if harmonics > 0 and fundamental_hz > 0.0:
        periods = length * fundamental_hz
        if abs(periods - round(periods)) <= period_tolerance * max(1.0, round(periods)) and round(periods) >= 1:
            for order in range(1, harmonics + 1):
                omega = 2.0 * math.pi * order * fundamental_hz
                kernel = np.exp(-1j * omega * (t - start_s)) * weights
                phasors.append(2.0 / length * (kernel @ i))
    return WindowMoments(
        ports=tuple(ports),
        start_s=float(start_s),
        stop_s=float(stop_s),
        samples=int(t.size),
        mean_a=mean,
        second_moment_a2=second,
        fundamental_hz=float(fundamental_hz),
        harmonics=tuple(phasors),
    )


def close_kcl(
    moments: WindowMoments,
    groups_by_port: Mapping[str, str],
    *,
    tolerance_a: float = 1.0e-9,
    maximum_relative_residual: float = 1.0e-3,
) -> WindowMoments:
    """Remove each group's closure residual with one linear projection.

    A group is the ports of one conductor, whose currents must sum to zero.
    For a group whose active ports (``|mean| > tolerance_a``) carry a residual
    ``r = Σ mean``, port ``k`` gives up ``r · |mean_k| / Σ|mean|``; written as
    the projection ``P = I − w 1ᵀ`` on the active ports it is applied to the
    means, to the second moments as ``P C Pᵀ`` and to every harmonic.  A
    residual above ``maximum_relative_residual`` of the carried current is a
    model error and is refused.  Ports the mapping does not name are left
    untouched.
    """

    n = len(moments.ports)
    projection = np.eye(n)
    report: dict[str, dict[str, float]] = {}
    for group in dict.fromkeys(groups_by_port.values()):
        members = [k for k, port in enumerate(moments.ports) if groups_by_port.get(port) == group]
        active = [k for k in members if abs(moments.mean_a[k]) > tolerance_a]
        if len(active) < 2:
            continue
        means = moments.mean_a[active]
        residual = float(math.fsum(means))
        carried = float(math.fsum(np.abs(means)))
        if carried <= 0.0:
            continue
        relative = abs(residual) / carried
        if relative > maximum_relative_residual:
            raise ValueError(
                f"group {group!r} current does not close to within {maximum_relative_residual:g} of what it "
                f"carries: residual {residual:g} A over {carried:g} A. This is a circuit error, not solver noise."
            )
        weights = np.abs(means) / carried
        block = np.eye(len(active)) - np.outer(weights, np.ones(len(active)))
        projection[np.ix_(active, active)] = block
        report[group] = {"raw_residual_a": residual, "carried_a": carried, "relative_residual": relative}
    mean = projection @ moments.mean_a
    second = projection @ moments.second_moment_a2 @ projection.T
    harmonics = tuple(projection @ h for h in moments.harmonics)
    return WindowMoments(
        ports=moments.ports,
        start_s=moments.start_s,
        stop_s=moments.stop_s,
        samples=moments.samples,
        mean_a=mean,
        second_moment_a2=0.5 * (second + second.T),
        fundamental_hz=moments.fundamental_hz,
        harmonics=harmonics,
        closure_by_group=report,
    )


def port_block(moments: WindowMoments, ports: Sequence[str]) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """Mean vector, second-moment matrix and harmonics restricted to ``ports``, in that order."""

    index = [moments.index(port) for port in ports]
    return (
        moments.mean_a[index],
        moments.second_moment_a2[np.ix_(index, index)],
        tuple(h[index] for h in moments.harmonics),
    )

__all__ = ["WindowMoments", "close_kcl", "port_block", "window_moments"]
