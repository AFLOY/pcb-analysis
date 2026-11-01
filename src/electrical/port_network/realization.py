"""A pad-port impedance matrix realized as resistors and coupled inductors.

Copper seen from ``n + 1`` pads, one of them the reference, is an n-port
with impedance ``Z = R + jωL`` (``z[j, k]`` = V(pad j) − V(reference) for 1 A
into pad k and out of the reference).  The realization keeps resistance and
inductance apart: every non-reference pad enters through its own inductor
to an inner node, the inductors carry the full inductance matrix as mutual
couplings, and a resistor network between the inner nodes and the
reference carries ``(Re Z)^-1``.  It reproduces ``Z`` at the extraction
frequency and is a positive-definite resistor network at DC, which an
operating-point solve needs; a series R-L branch network matches only at
one frequency and is singular at DC.

A resistor can come out negative when two pads share most of their copper
path.  It is flagged, not clipped: the network as a whole is what was
measured.

Copper in several conductors (a loop's go and return) has a reference pad
per conductor, and each port names its own.  The resistor network is then
built per conductor from its block of ``Re Z``: a resistor between two
conductors would join them at DC, which they are not.  The resistance a
matrix carries between conductors (eddy and proximity loss, small) is
dropped, and its largest magnitude recorded.  The inductors couple across
conductors, which is what such a matrix is for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


def conductance_branches(
    conductance_s: np.ndarray, *, reduced: bool
) -> list[tuple[int, int | None, float]]:
    """The branch conductances of the resistor network a conductance matrix is.

    The branch between ports ``j`` and ``k`` has conductance ``−G[j, k]``.
    ``reduced``: the matrix has the reference row and column removed, and
    each port also has a branch to the reference (``k`` of ``None``) with
    the row sum.  Otherwise the rows sum to zero and there is none.  Order:
    for each ``j``, the pairs ``(j, k > j)``, then ``(j, None)``.
    """
    g = np.asarray(conductance_s, dtype=np.float64)
    n = g.shape[0]
    out: list[tuple[int, int | None, float]] = []
    for j in range(n):
        for k in range(j + 1, n):
            out.append((j, k, float(-g[j, k])))
        if reduced:
            out.append((j, None, float(np.sum(g[j, :]))))
    return out


@dataclass(frozen=True)
class Resistor:
    """One resistor between two pads (the reference pad or a pad's inner node)."""

    pad_a: str
    pad_b: str
    resistance_ohm: float
    flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PortNetwork:
    """Resistors and coupled inductors that present ``Z`` between ``pads`` and the reference."""

    references: dict[str, str]
    """Each non-reference pad's reference pad (one per conductor)."""
    pads: tuple[str, ...]
    """Non-reference pads, in matrix order."""
    frequency_hz: float
    resistors: tuple[Resistor, ...]
    self_inductance_h: dict[str, float]
    """Per non-reference pad; empty at DC."""
    couplings: tuple[tuple[str, str, float], ...]
    """(pad_a, pad_b, k) for the mutual inductance between their inductors."""
    flags: tuple[str, ...] = field(default_factory=tuple)
    dropped_cross_resistance_ohm: float = 0.0
    """Largest ``|Re Z|`` between pads of different conductors, left out of the resistors."""

    @property
    def reference_pads(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.references[p] for p in self.pads))

    @property
    def reference_pad(self) -> str:
        """The reference pad of a network of one conductor."""
        refs = self.reference_pads
        if len(refs) != 1:
            raise ValueError(f"the network has {len(refs)} reference pads: {list(refs)}")
        return refs[0]

    def impedance_matrix(self, frequency_hz: float | None = None) -> np.ndarray:
        """The reduced impedance this network presents, for checking."""
        omega = 2.0 * math.pi * (self.frequency_hz if frequency_hz is None else frequency_hz)
        index = {pad: i for i, pad in enumerate(self.pads)}
        size = len(self.pads)
        conductance = np.zeros((size, size))
        for branch in self.resistors:
            g = 1.0 / branch.resistance_ohm
            a = index.get(branch.pad_a)
            b = index.get(branch.pad_b)
            for node in (a, b):
                if node is not None:
                    conductance[node, node] += g
            if a is not None and b is not None:
                conductance[a, b] -= g
                conductance[b, a] -= g
        resistance = np.linalg.inv(conductance)
        inductance = np.zeros((size, size))
        for pad, value in self.self_inductance_h.items():
            inductance[index[pad], index[pad]] = value
        for pad_a, pad_b, k in self.couplings:
            a, b = index[pad_a], index[pad_b]
            mutual = k * math.sqrt(inductance[a, a] * inductance[b, b])
            inductance[a, b] = inductance[b, a] = mutual
        return resistance + 1j * omega * inductance

    def as_dict(self) -> dict[str, Any]:
        return {
            "references": dict(self.references),
            "pads": list(self.pads),
            "frequency_hz": self.frequency_hz,
            "resistors": [
                {"pad_a": r.pad_a, "pad_b": r.pad_b, "resistance_ohm": r.resistance_ohm, "flags": list(r.flags)}
                for r in self.resistors
            ],
            "self_inductance_h": dict(self.self_inductance_h),
            "couplings": [list(item) for item in self.couplings],
            "flags": list(self.flags),
            "dropped_cross_resistance_ohm": self.dropped_cross_resistance_ohm,
        }


def resistor_network(
    pads: Sequence[str], reference_pad: str, resistance_ohm: np.ndarray
) -> tuple[list[Resistor], list[str]]:
    """Resistors that present the reduced resistance matrix; branches below 1e-9 S are left out."""
    flags: list[str] = []
    eigenvalues = np.linalg.eigvalsh(0.5 * (resistance_ohm + resistance_ohm.T))
    if eigenvalues.min() <= 0.0:
        flags.append("resistance_matrix_not_positive_definite")
    conductance = np.linalg.inv(resistance_ohm)
    resistors: list[Resistor] = []
    for j, k, g in conductance_branches(conductance, reduced=True):
        if abs(g) < 1e-9:
            continue
        resistors.append(Resistor(pads[j], reference_pad if k is None else pads[k], 1.0 / g,
                                  ("negative_resistance",) if g < 0.0 else ()))
    return resistors, flags


def realize_port_network(
    z_ohm: np.ndarray, pads: Sequence[str], references: str | Mapping[str, str], frequency_hz: float
) -> PortNetwork:
    """Turn a reduced impedance matrix into resistors and coupled inductors.

    ``references`` is the one reference pad of a single conductor, or each
    pad's reference pad (``SheetPortImpedance.references``) for several.

    The inductance matrix ``Im Z / ω`` must be positive definite for coupled
    inductors; a quasi-static extraction can miss that slightly, and then
    the self terms are kept and the couplings dropped
    (``inductance_matrix_not_positive_definite``) rather than emit a network
    that stores negative energy.  A coupling of magnitude 1 or more is
    clipped to 0.999 (``coupling_clipped``); couplings below 1e-6 are left out.
    """
    z = np.asarray(z_ohm)
    pads = tuple(pads)
    reference_of = {p: references for p in pads} if isinstance(references, str) else {p: str(references[p]) for p in pads}
    omega = 2.0 * math.pi * frequency_hz
    resistors: list[Resistor] = []
    flags: list[str] = []
    groups = list(dict.fromkeys(reference_of[p] for p in pads))
    for reference in groups:
        members = [j for j, p in enumerate(pads) if reference_of[p] == reference]
        block, block_flags = resistor_network([pads[j] for j in members], reference, z.real[np.ix_(members, members)])
        resistors += block
        flags += block_flags
    cross = 0.0
    if len(groups) > 1:
        label = np.array([groups.index(reference_of[p]) for p in pads])
        across = label[:, None] != label[None, :]
        cross = float(np.max(np.abs(z.real[across]))) if across.any() else 0.0
    self_inductance: dict[str, float] = {}
    couplings: list[tuple[str, str, float]] = []
    if omega > 0.0:
        inductance = z.imag / omega
        eigenvalues = np.linalg.eigvalsh(0.5 * (inductance + inductance.T))
        if eigenvalues.min() <= 0.0:
            flags.append("inductance_matrix_not_positive_definite")
            couplings_allowed = False
        else:
            couplings_allowed = True
        for j, pad in enumerate(pads):
            self_inductance[pad] = float(max(inductance[j, j], 0.0))
        if couplings_allowed:
            for j, pad_a in enumerate(pads):
                for k in range(j + 1, len(pads)):
                    denominator = math.sqrt(inductance[j, j] * inductance[k, k])
                    if denominator <= 0.0:
                        continue
                    coefficient = float(inductance[j, k] / denominator)
                    if abs(coefficient) >= 1.0:
                        flags.append("coupling_clipped")
                        coefficient = math.copysign(0.999, coefficient)
                    if abs(coefficient) >= 1e-6:
                        couplings.append((pad_a, pads[k], coefficient))
    return PortNetwork(
        references=reference_of,
        pads=pads,
        frequency_hz=frequency_hz,
        resistors=tuple(resistors),
        self_inductance_h=self_inductance,
        couplings=tuple(couplings),
        flags=tuple(dict.fromkeys(flags)),
        dropped_cross_resistance_ohm=cross,
    )


__all__ = ["PortNetwork", "Resistor", "conductance_branches", "realize_port_network", "resistor_network"]
