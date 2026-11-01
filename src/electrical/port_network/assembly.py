"""Copper and parts solved together, one frequency at a time.

The nodes are pads.  The copper enters as its pad-port impedance ``Z(f)``:
port ``j`` is the voltage of its pad over its conductor's reference pad, so
with ``B`` the port-to-node incidence (``+1`` at the port pad, ``-1`` at its
reference) the copper adds ``Bᵀ Z⁻¹ B`` to the nodal admittance.  The
copper may be in several conductors (go and return), which then couple
through ``Z``'s cross terms.  Every part is stamped between its own two
pads, so every part's current is a result.  A ``Tie`` or a ``VoltageSource``
adds a branch current as an unknown (modified nodal analysis), and a
``CurrentSource`` adds to the right-hand side.  One pad is the ground.

The copper's ``Z(f)`` comes from the caller at the frequencies asked for.
This module runs no field solve, so ``electrical.sheet_peec`` stays the
solver.  A frequency the copper has no matrix for is refused, not
interpolated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np

from .parts import CurrentSource, Part, Tie, VoltageSource


class CopperPorts(Protocol):
    """What a pad-port impedance offers (``electrical.sheet_peec.SheetPortImpedance``)."""

    ports: tuple[str, ...]
    references: Mapping[str, str]
    z_ohm: np.ndarray
    frequency_hz: float


@dataclass(frozen=True)
class NetworkSolution:
    """Everything at one frequency."""

    frequency_hz: float
    pad_voltage_v: dict[str, complex]
    """Against the ground pad."""
    part_current_a: dict[str, complex]
    """Through each part from its pad ``a`` to its pad ``b``; ties included."""
    source_current_a: dict[str, complex]
    """Through each voltage source from ``a`` to ``b`` (a current source carries its own)."""
    copper_port_current_a: dict[str, complex]
    """Into each copper port pad from the network (out of its reference)."""
    solve_residual: float = 0.0
    """Largest entry of the linear system's residual, for checking."""


@dataclass(frozen=True)
class NetworkSpectrum:
    """Solutions over frequency, with each part's current as an array."""

    frequency_hz: tuple[float, ...]
    solutions: tuple[NetworkSolution, ...]

    def part_current_a(self, name: str) -> np.ndarray:
        return np.array([s.part_current_a[name] for s in self.solutions])

    def pad_voltage_v(self, pad: str) -> np.ndarray:
        return np.array([s.pad_voltage_v[pad] for s in self.solutions])


class BoardNetwork:
    """Copper (``Z(f)`` per frequency) and the parts that join its nets."""

    def __init__(
        self,
        copper: Mapping[float, CopperPorts] | Sequence[CopperPorts],
        parts: Sequence[Part],
        ground: str,
        *,
        current_sources: Sequence[CurrentSource] = (),
        voltage_sources: Sequence[VoltageSource] = (),
    ) -> None:
        if not isinstance(copper, Mapping):
            copper = {float(c.frequency_hz): c for c in copper}
        self.copper = {float(f): c for f, c in copper.items()}
        if not self.copper:
            raise ValueError("no copper matrix given")
        first = next(iter(self.copper.values()))
        self.ports = tuple(first.ports)
        self.references = dict(first.references)
        for f, c in self.copper.items():
            if tuple(c.ports) != self.ports or dict(c.references) != self.references:
                raise ValueError(f"the copper at {f:g} Hz has other ports than at the first frequency")
        self.parts = tuple(parts)
        self.current_sources = tuple(current_sources)
        self.voltage_sources = tuple(voltage_sources)
        names = [p.name for p in self.parts] + [s.name for s in self.current_sources] + [s.name for s in self.voltage_sources]
        duplicate = sorted({n for n in names if names.count(n) > 1})
        if duplicate:
            raise ValueError(f"names used twice: {duplicate}")
        pads = list(dict.fromkeys([*self.ports, *(self.references[p] for p in self.ports)]))
        for item in (*self.parts, *self.current_sources, *self.voltage_sources):
            for pad in (item.a, item.b):
                if pad not in pads:
                    pads.append(pad)
        if ground not in pads:
            raise ValueError(f"ground pad {ground} is not a node of the network")
        self.ground = ground
        self.pads = tuple(pads)
        self._index = {pad: i for i, pad in enumerate(p for p in self.pads if p != ground)}
        self._branches = [p for p in self.parts if isinstance(p.model, Tie)] + list(self.voltage_sources)

    def solve(self, frequency_hz: float) -> NetworkSolution:
        frequency_hz = float(frequency_hz)
        if frequency_hz <= 0.0:
            raise ValueError("solve at a positive frequency")
        copper = self.copper.get(frequency_hz)
        if copper is None:
            raise ValueError(f"no copper matrix at {frequency_hz:g} Hz (have {sorted(self.copper)})")
        omega = 2.0 * math.pi * frequency_hz
        n, m = len(self._index), len(self._branches)
        a = np.zeros((n + m, n + m), dtype=complex)
        rhs = np.zeros(n + m, dtype=complex)

        def node(pad: str) -> int | None:
            return self._index.get(pad)

        # copper: B^T Z^-1 B over the port pads and their references
        y_cu = np.linalg.inv(np.asarray(copper.z_ohm, dtype=complex))
        incidence = np.zeros((len(self.ports), n), dtype=complex)
        for j, port in enumerate(self.ports):
            for pad, sign in ((port, 1.0), (self.references[port], -1.0)):
                k = node(pad)
                if k is not None:
                    incidence[j, k] += sign
        a[:n, :n] += incidence.T @ y_cu @ incidence

        admittance: dict[str, complex] = {}
        for part in self.parts:
            if isinstance(part.model, Tie):
                continue
            z = complex(part.model.impedance(omega))
            if z == 0:
                raise ValueError(f"part {part.name} has zero impedance at {frequency_hz:g} Hz; make it a Tie")
            y = 1.0 / z
            admittance[part.name] = y
            ia, ib = node(part.a), node(part.b)
            for i, j, sign in ((ia, ia, 1), (ib, ib, 1), (ia, ib, -1), (ib, ia, -1)):
                if i is not None and j is not None:
                    a[i, j] += sign * y
        for k, branch in enumerate(self._branches):
            row = n + k
            ia, ib = node(branch.a), node(branch.b)
            if ia is not None:
                a[ia, row] += 1.0  # branch current leaves node a
                a[row, ia] += 1.0
            if ib is not None:
                a[ib, row] -= 1.0
                a[row, ib] -= 1.0
            rhs[row] = branch.at(frequency_hz) if isinstance(branch, VoltageSource) else 0.0
        for source in self.current_sources:
            current = source.at(frequency_hz)
            ia, ib = node(source.a), node(source.b)
            if ia is not None:
                rhs[ia] -= current
            if ib is not None:
                rhs[ib] += current
        try:
            x = np.linalg.solve(a, rhs)
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                f"the network is singular at {frequency_hz:g} Hz: a pad or a conductor has no path to {self.ground}"
            ) from error

        def voltage(pad: str) -> complex:
            k = node(pad)
            return complex(x[k]) if k is not None else 0j

        pad_voltage = {pad: voltage(pad) for pad in self.pads}
        part_current = {name: y * (pad_voltage[p.a] - pad_voltage[p.b])
                        for p in self.parts if (name := p.name) in admittance
                        for y in (admittance[name],)}
        source_current: dict[str, complex] = {}
        for k, branch in enumerate(self._branches):
            if isinstance(branch, VoltageSource):
                source_current[branch.name] = complex(x[n + k])
            else:
                part_current[branch.name] = complex(x[n + k])
        port_voltage = incidence @ x[:n]
        port_current = y_cu @ port_voltage
        solution = NetworkSolution(
            frequency_hz=frequency_hz,
            pad_voltage_v=pad_voltage,
            part_current_a={p.name: part_current[p.name] for p in self.parts},
            source_current_a=source_current,
            copper_port_current_a={port: complex(i) for port, i in zip(self.ports, port_current)},
            solve_residual=float(np.abs(a @ x - rhs).max()) if x.size else 0.0,
        )
        return solution

    def solve_spectrum(self, frequencies_hz: Sequence[float] | None = None) -> NetworkSpectrum:
        """Every frequency asked for (default: every frequency the copper has)."""
        frequencies = tuple(sorted(float(f) for f in (frequencies_hz if frequencies_hz is not None else self.copper)))
        return NetworkSpectrum(frequencies, tuple(self.solve(f) for f in frequencies))


__all__ = ["BoardNetwork", "CopperPorts", "NetworkSolution", "NetworkSpectrum"]
