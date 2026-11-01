"""Parts that join nets: each one an element between its own pads.

A part keeps its own identity in the network, so the current through every
capacitor, inductor or resistor is a result at every frequency: the share of
the ripple a decoupling capacitor shunts is what shows its filtering.

A two-terminal part's model is anything with ``impedance(omega)``:
``SeriesRLC`` for a datasheet's C-ESR-ESL, ``TabulatedImpedance`` for a
measured or simulated curve (a capacitor at its DC bias, a power inductor),
and ``Tie`` for an ideal connection whose current is still reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class ImpedanceModel(Protocol):
    def impedance(self, omega: float) -> complex: ...


@dataclass(frozen=True)
class TabulatedImpedance:
    """An impedance known on a frequency grid.

    Between grid points ``log|Z|`` and the unwrapped phase are interpolated
    linearly in ``log f``, which follows a resonance without the ringing a
    real/imaginary interpolation shows.  Outside the grid it refuses: an
    extrapolated part is a guess, not a model.
    """

    frequency_hz: tuple[float, ...]
    z_ohm: tuple[complex, ...]
    _log_f: np.ndarray = field(init=False, repr=False, compare=False)
    _log_mag: np.ndarray = field(init=False, repr=False, compare=False)
    _phase: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        f = np.asarray(self.frequency_hz, dtype=float)
        z = np.asarray(self.z_ohm, dtype=complex)
        if f.ndim != 1 or f.size < 2 or z.shape != f.shape:
            raise ValueError("a tabulated impedance needs at least two (frequency, impedance) points")
        if np.any(f <= 0.0) or np.any(np.diff(f) <= 0.0):
            raise ValueError("frequencies must be positive and increasing")
        if np.any(z == 0):
            raise ValueError("a tabulated impedance cannot be zero at a grid point")
        object.__setattr__(self, "frequency_hz", tuple(float(v) for v in f))
        object.__setattr__(self, "z_ohm", tuple(complex(v) for v in z))
        object.__setattr__(self, "_log_f", np.log(f))
        object.__setattr__(self, "_log_mag", np.log(np.abs(z)))
        object.__setattr__(self, "_phase", np.unwrap(np.angle(z)))

    def impedance(self, omega: float) -> complex:
        f = omega / (2.0 * math.pi)
        lo, hi = self.frequency_hz[0], self.frequency_hz[-1]
        if not (lo * (1 - 1e-12) <= f <= hi * (1 + 1e-12)):
            raise ValueError(f"{f:g} Hz is outside the table [{lo:g}, {hi:g}] Hz")
        x = math.log(min(max(f, lo), hi))
        magnitude = math.exp(float(np.interp(x, self._log_f, self._log_mag)))
        phase = float(np.interp(x, self._log_f, self._phase))
        return complex(magnitude * math.cos(phase), magnitude * math.sin(phase))


@dataclass(frozen=True)
class Tie:
    """An ideal connection between two pads (a jumper, a shorted switch, pins of one part
    on one internal node).  It is solved as a 0 V source, so its current is a result."""

    def impedance(self, omega: float) -> complex:
        return 0j


@dataclass(frozen=True)
class Part:
    """A two-terminal part between pads ``a`` and ``b``; positive current flows from ``a`` to ``b``
    through the part."""

    name: str
    a: str
    b: str
    model: ImpedanceModel

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise ValueError(f"part {self.name} joins a pad to itself")


@dataclass(frozen=True)
class CurrentSource:
    """A current source that carries its current from pad ``a`` to pad ``b`` through itself:
    it draws the current from the network at ``a`` and delivers it at ``b``."""

    name: str
    a: str
    b: str
    spectrum: dict[float, complex]
    """Phasor per frequency (Hz).  A frequency it does not list carries no current."""

    def at(self, frequency_hz: float) -> complex:
        return complex(self.spectrum.get(float(frequency_hz), 0.0))


@dataclass(frozen=True)
class VoltageSource:
    """``V(a) - V(b) = voltage(f)``; its current (from ``a`` through the source to ``b``) is a result."""

    name: str
    a: str
    b: str
    spectrum: dict[float, complex]

    def at(self, frequency_hz: float) -> complex:
        return complex(self.spectrum.get(float(frequency_hz), 0.0))


__all__ = ["CurrentSource", "ImpedanceModel", "Part", "TabulatedImpedance", "Tie", "VoltageSource"]
