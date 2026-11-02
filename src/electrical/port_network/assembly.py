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

Each frequency is factorised once, for every source at unit strength
(``transfer``): the network is linear, so the responses to any excitation
-- several operating cases, each with its harmonics (``solve_cases``) --
are those columns weighted by the source phasors.

The copper's ``Z(f)`` comes from the caller at the frequencies asked for.
This module runs no field solve, so ``electrical.sheet_peec`` stays the
solver.  A frequency the copper has no matrix for is refused, not
interpolated -- unless the copper is a ``PortImpedanceSweep``, whose
interpolation answers inside its solved band; a solution then says whether
its copper was solved or interpolated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np

from .frequency import distinct_frequencies, match_frequency
from .parts import CurrentSource, Part, Tie, VoltageSource, _phasor
from .sweep import PortImpedanceSweep


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
    """Through each source from ``a`` to ``b``: a voltage source's solved current, a current source's own."""
    copper_port_current_a: dict[str, complex]
    """Into each copper port pad from the network (out of its reference)."""
    solve_residual: float = 0.0
    """Largest entry of the linear system's residual, for checking."""
    copper_interpolated: bool = False
    """The copper's matrix at this frequency came from a sweep's interpolation, not a solve."""


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
        copper: Mapping[float, CopperPorts] | Sequence[CopperPorts] | PortImpedanceSweep,
        parts: Sequence[Part],
        ground: str,
        *,
        current_sources: Sequence[CurrentSource] = (),
        voltage_sources: Sequence[VoltageSource] = (),
    ) -> None:
        self.sweep = copper if isinstance(copper, PortImpedanceSweep) else None
        if self.sweep is not None:
            copper = {float(p.frequency_hz): p for p in self.sweep.points}
        elif not isinstance(copper, Mapping):
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
        on_copper = list(dict.fromkeys([*self.ports, *(self.references[p] for p in self.ports)]))
        pads = list(on_copper)
        touching: dict[str, list[str]] = {}
        for item in (*self.parts, *self.current_sources, *self.voltage_sources):
            for pad in (item.a, item.b):
                touching.setdefault(pad, []).append(item.name)
                if pad not in pads:
                    pads.append(pad)
        if ground not in pads:
            raise ValueError(f"ground pad {ground} is not a node of the network")
        # A pad off the copper is a node of the parts alone (between a source
        # and its series part, say).  Joined by one element only, it is a dead
        # end: that element carries no current, which is never what was meant
        # -- usually the pad was named otherwise than the copper's ports
        # (``C2.1`` where merged pins make the port ``C2:VIN``).
        dead = sorted(pad for pad in pads if pad not in on_copper and len(touching.get(pad, ())) < 2)
        if dead:
            hints = []
            for pad in dead:
                part = pad.replace(":", ".").split(".")[0]
                similar = [name for name in on_copper if name.replace(":", ".").split(".")[0] == part]
                hint = f" (the copper has {similar})" if similar else ""
                hints.append(f"{pad} [{', '.join(touching.get(pad, ()))}]{hint}")
            raise ValueError(
                "pads that are neither a copper port nor a reference and join one element only, "
                f"which would carry no current: {'; '.join(hints)}"
            )
        self.ground = ground
        self.pads = tuple(pads)
        self._index = {pad: i for i, pad in enumerate(p for p in self.pads if p != ground)}
        self._branches = [p for p in self.parts if isinstance(p.model, Tie)] + list(self.voltage_sources)
        self._transfers: dict[float, NetworkTransfer] = {}

    @property
    def sources(self) -> tuple[str, ...]:
        """The sources in the order a transfer's columns take: current sources, then voltage sources."""
        return tuple(s.name for s in (*self.current_sources, *self.voltage_sources))

    def transfer(self, frequency_hz: float) -> "NetworkTransfer":
        """Every response at one frequency to each source alone, from one factorisation.

        Column ``k`` is the network driven by 1 A through current source ``k``
        (or 1 V across voltage source ``k``) and nothing else.  The network is
        linear, so any excitation is a weighted sum of the columns
        (:meth:`NetworkTransfer.apply`).  Kept per frequency.
        """
        frequency_hz = float(frequency_hz)
        if frequency_hz <= 0.0:
            raise ValueError("solve at a positive frequency")
        # the copper's own frequency when it has one within FREQUENCY_RTOL
        frequency_hz = match_frequency(frequency_hz, self.copper) or frequency_hz
        cached = self._transfers.get(frequency_hz)
        if cached is not None:
            return cached
        copper = self.copper.get(frequency_hz)
        interpolated = False
        if copper is None and self.sweep is not None:
            copper = self.sweep.at(frequency_hz)  # refuses outside the solved band
            interpolated = copper.interpolated
        if copper is None:
            raise ValueError(f"no copper matrix at {frequency_hz:g} Hz (have {sorted(self.copper)})")
        omega = 2.0 * math.pi * frequency_hz
        n, m = len(self._index), len(self._branches)
        a = np.zeros((n + m, n + m), dtype=complex)

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

        # one right-hand side per source at unit strength (a zero column keeps the singular check)
        rhs = np.zeros((n + m, max(1, len(self.sources))), dtype=complex)
        for column, source in enumerate(self.current_sources):
            ia, ib = node(source.a), node(source.b)
            if ia is not None:
                rhs[ia, column] -= 1.0
            if ib is not None:
                rhs[ib, column] += 1.0
        for k, branch in enumerate(self._branches):
            if isinstance(branch, VoltageSource):
                rhs[n + k, len(self.current_sources) + self.voltage_sources.index(branch)] = 1.0
        try:
            x = np.linalg.solve(a, rhs)
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                f"the network is singular at {frequency_hz:g} Hz: a pad or a conductor has no path to {self.ground}"
            ) from error
        residual = float(np.abs(a @ x - rhs).max()) if x.size else 0.0
        x = x[:, : len(self.sources)]

        voltage = np.zeros((len(self.pads), x.shape[1]), dtype=complex)
        for i, pad in enumerate(self.pads):
            k = node(pad)
            if k is not None:
                voltage[i] = x[k]
        row_of = {pad: i for i, pad in enumerate(self.pads)}
        branch_row = {branch.name: n + k for k, branch in enumerate(self._branches)}
        part_current = np.zeros((len(self.parts), x.shape[1]), dtype=complex)
        for i, part in enumerate(self.parts):
            if part.name in admittance:
                part_current[i] = admittance[part.name] * (voltage[row_of[part.a]] - voltage[row_of[part.b]])
            else:
                part_current[i] = x[branch_row[part.name]]
        source_current = np.array([x[branch_row[s.name]] for s in self.voltage_sources],
                                  dtype=complex).reshape(len(self.voltage_sources), x.shape[1])
        port_current = y_cu @ (incidence @ x[:n])
        transfer = NetworkTransfer(
            frequency_hz=frequency_hz,
            sources=self.sources,
            pads=self.pads,
            parts=tuple(p.name for p in self.parts),
            voltage_sources=tuple(s.name for s in self.voltage_sources),
            ports=self.ports,
            pad_voltage_v=voltage,
            part_current_a=part_current,
            source_current_a=source_current,
            copper_port_current_a=port_current,
            solve_residual=residual,
            copper_interpolated=interpolated,
        )
        self._transfers[frequency_hz] = transfer
        return transfer

    def solve(self, frequency_hz: float) -> NetworkSolution:
        """The network driven by its sources' own spectra at one frequency."""
        transfer = self.transfer(frequency_hz)
        return transfer.apply({s.name: s.at(transfer.frequency_hz)
                               for s in (*self.current_sources, *self.voltage_sources)})

    def solve_spectrum(self, frequencies_hz: Sequence[float] | None = None) -> NetworkSpectrum:
        """Every frequency asked for (default: every frequency the copper has)."""
        asked = distinct_frequencies(frequencies_hz if frequencies_hz is not None else self.copper)
        solutions = tuple(self.solve(f) for f in asked)
        return NetworkSpectrum(tuple(s.frequency_hz for s in solutions), solutions)

    def solve_cases(
        self,
        cases: Mapping[str, Mapping[str, Mapping[float, complex]]],
        frequencies_hz: Sequence[float] | None = None,
    ) -> dict[str, NetworkSpectrum]:
        """Several excitations at once: ``cases[name][source][f]`` is that source's phasor at ``f``.

        A case is one operating point of the circuit (a load, an input
        voltage), its spectra the harmonics of each source.  A source a case
        does not name carries nothing in it, and neither does a frequency its
        spectrum does not list.  Each frequency is factorised once for every
        case (:meth:`transfer`).  The frequencies default to every one any
        case lists; each needs a copper matrix.
        """
        known = set(self.sources)
        for name, spectra in cases.items():
            unknown = sorted(set(spectra) - known)
            if unknown:
                raise ValueError(f"case {name} drives sources the network does not have: {unknown}")
        if frequencies_hz is None:
            frequencies = distinct_frequencies(f for spectra in cases.values() for spectrum in spectra.values() for f in spectrum)
        else:
            frequencies = distinct_frequencies(frequencies_hz)
        names = list(cases)
        by_case: dict[str, list[NetworkSolution]] = {name: [] for name in names}
        for frequency in frequencies:
            transfer = self.transfer(frequency)
            drive = np.zeros((len(self.sources), len(names)), dtype=complex)
            for column, name in enumerate(names):
                drive[:, column] = transfer.excitation_vector(
                    {source: _phasor(spectrum, frequency) for source, spectrum in cases[name].items()}
                )
            for column, solution in enumerate(transfer.apply_matrix(drive)):
                by_case[names[column]].append(solution)
        return {name: NetworkSpectrum(tuple(s.frequency_hz for s in by_case[name]), tuple(by_case[name]))
                for name in names}


@dataclass(frozen=True)
class NetworkTransfer:
    """The network at one frequency driven by each source alone at unit strength.

    Arrays have one column per source (``sources``: current sources at 1 A,
    then voltage sources at 1 V) and one row per pad, part, voltage source
    or copper port.  An excitation is a vector of source phasors in the same
    order; the responses are the arrays times it.
    """

    frequency_hz: float
    sources: tuple[str, ...]
    pads: tuple[str, ...]
    parts: tuple[str, ...]
    voltage_sources: tuple[str, ...]
    ports: tuple[str, ...]
    pad_voltage_v: np.ndarray
    part_current_a: np.ndarray
    source_current_a: np.ndarray
    copper_port_current_a: np.ndarray
    solve_residual: float
    """Largest entry of the factorised system's residual over the unit columns."""
    copper_interpolated: bool = False
    """The copper's matrix came from a sweep's interpolation, not a solve."""

    def excitation_vector(self, phasors: Mapping[str, complex]) -> np.ndarray:
        unknown = sorted(set(phasors) - set(self.sources))
        if unknown:
            raise ValueError(f"not sources of this network: {unknown}")
        return np.array([complex(phasors.get(name, 0.0)) for name in self.sources], dtype=complex)

    def apply(self, phasors: Mapping[str, complex]) -> NetworkSolution:
        """The solution for one excitation, ``phasors[source]`` (sources not named carry nothing)."""
        return self.apply_matrix(self.excitation_vector(phasors)[:, None])[0]

    def apply_matrix(self, drive: np.ndarray) -> list[NetworkSolution]:
        """One solution per column of ``drive`` (sources x excitations)."""
        drive = np.asarray(drive, dtype=complex)
        if drive.ndim != 2 or drive.shape[0] != len(self.sources):
            raise ValueError("drive must hold one row per source")
        voltage = self.pad_voltage_v @ drive
        part = self.part_current_a @ drive
        source = self.source_current_a @ drive
        port = self.copper_port_current_a @ drive
        return [
            NetworkSolution(
                frequency_hz=self.frequency_hz,
                pad_voltage_v={pad: complex(voltage[i, c]) for i, pad in enumerate(self.pads)},
                part_current_a={name: complex(part[i, c]) for i, name in enumerate(self.parts)},
                source_current_a={
                    **{name: complex(drive[i, c]) for i, name in enumerate(self.sources)
                       if name not in self.voltage_sources},
                    **{name: complex(source[i, c]) for i, name in enumerate(self.voltage_sources)},
                },
                copper_port_current_a={name: complex(port[i, c]) for i, name in enumerate(self.ports)},
                solve_residual=self.solve_residual,
                copper_interpolated=self.copper_interpolated,
            )
            for c in range(drive.shape[1])
        ]


__all__ = ["BoardNetwork", "CopperPorts", "NetworkSolution", "NetworkSpectrum", "NetworkTransfer"]
