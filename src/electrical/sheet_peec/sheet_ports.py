"""The pad-port impedance of copper in several conductors, from one lockstep solve.

Copper seen from its pads is a multi-port.  Each pad that is a port names
the reference pad of its own conductor; a reference pad is the port's
return.  Port ``k`` is driven with 1 A into its pad and out of its
reference, every other pad is a terminal of 0 A, and the mean potential
over every port pad less its reference's is column ``k`` of the impedance:
``z[j, k]`` = V(port j) − V(reference of j) for 1 A through port ``k``.

The conductors that carry no current in an excitation are kept
(``keep_terminal_components``), so the voltage the driven copper induces
on them is read: the matrix couples ports on different conductors through
their mutual inductance, which a matrix per conductor drops.  Every
excitation lists the same pads, so all of them solve in one lockstep group.

Nothing here reads a board.  The caller states the pads as mesh cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .sheet_operator import SheetInductanceOperator
from .sheet_peec import Cell, SheetMesh, Terminal, solve_sheet_cases


@dataclass(frozen=True)
class PortPad:
    """One pad: its cells on one layer of the mesh."""

    name: str
    layer: int
    cells: tuple[Cell, ...]


@dataclass(frozen=True)
class SheetPortImpedance:
    """``z_ohm[j, k]`` = V(ports[j]) − V(reference of ports[j]) for 1 A through ``ports[k]``."""

    ports: tuple[str, ...]
    references: dict[str, str]
    """Each port's reference pad."""
    z_ohm: np.ndarray
    frequency_hz: float
    converged: tuple[bool, ...]
    iterations: tuple[int, ...]

    @property
    def reference_pads(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.references[p] for p in self.ports))


def _mean_voltage(mesh: SheetMesh, voltage: np.ndarray, pad: PortPad) -> complex:
    nodes = [mesh.node_index[(pad.layer, row, col)] for row, col in pad.cells
             if (pad.layer, row, col) in mesh.node_index]
    if not nodes:
        raise ValueError(f"pad {pad.name} has no cell on the conductor")
    return complex(np.mean(voltage[nodes]))


def solve_port_impedance(
    mesh: SheetMesh,
    operator: SheetInductanceOperator,
    pads: Sequence[PortPad],
    references: Mapping[str, str],
    *,
    frequency_hz: float,
    tolerance: float = 1e-10,
    max_iterations: int = 400,
    restart: int = 60,
    preconditioner: str = "auto",
    near_tiles: int | None = None,
) -> SheetPortImpedance:
    """The impedance matrix between the ports ``references`` names (port -> its reference pad).

    ``pads`` lists every port pad and every reference pad.  A port and its
    reference have to be on one conductor (the solve refuses a component that
    does not close its current); ports of different conductors couple.
    """
    by_name = {pad.name: pad for pad in pads}
    if len(by_name) != len(pads):
        raise ValueError("pad names must be unique")
    ports = tuple(references)
    missing = sorted({p for p in ports if p not in by_name} | {r for r in references.values() if r not in by_name})
    if missing:
        raise ValueError(f"pads named but not given: {missing}")
    if any(references[p] == p for p in ports):
        raise ValueError("a port cannot be its own reference")
    if set(ports) & set(references.values()):
        raise ValueError("a reference pad cannot also be a port")
    if not ports:
        raise ValueError("no ports")

    def excitation(port: str) -> list[Terminal]:
        current = {name: 0.0 for name in by_name}
        current[port] = 1.0
        current[references[port]] = -1.0
        return [Terminal(pad.name, pad.layer, pad.cells, current[pad.name]) for pad in pads]

    solutions = solve_sheet_cases(
        mesh, operator, [excitation(port) for port in ports], frequency_hz=frequency_hz, tolerance=tolerance,
        max_iterations=max_iterations, restart=restart, preconditioner=preconditioner, near_tiles=near_tiles,
        keep_terminal_components=True,
    )
    z = np.zeros((len(ports), len(ports)), dtype=complex)
    for k, solution in enumerate(solutions):
        for j, port in enumerate(ports):
            z[j, k] = (_mean_voltage(mesh, solution.node_voltage, by_name[port])
                       - _mean_voltage(mesh, solution.node_voltage, by_name[references[port]]))
    return SheetPortImpedance(
        ports=ports,
        references=dict(references),
        z_ohm=z,
        frequency_hz=float(frequency_hz),
        converged=tuple(bool(s.converged) for s in solutions),
        iterations=tuple(int(s.iterations) for s in solutions),
    )


__all__ = ["PortPad", "SheetPortImpedance", "solve_port_impedance"]
