"""The pad-port impedance of copper in several conductors, from one lockstep solve.

Copper seen from its pads is a multi-port.  Each pad that is a port names
the reference pad of its own conductor; a reference pad is the port's
return.  Port ``k`` is driven with 1 A into its pad and out of its
reference, every other pad is a terminal of 0 A, and the mean potential
over every port pad less its reference's is column ``k`` of the impedance:
``z[j, k]`` = V(port j) − V(reference of j) for 1 A through port ``k``.

A pad may cover cells on several layers (a plated pin, or several pins
taken as one port with :func:`merge_pads`).  Its current is spread evenly
over all its cells on the conductor and its voltage is the mean over the
same cells, the convention a one-layer ``Terminal`` already follows, so the
matrix stays reciprocal.  Merging the pins of one part on one net keeps the
part's current and loses only how it divides between those pins.

The conductors that carry no current in an excitation are kept
(``keep_terminal_components``), so the voltage the driven copper induces
on them is read: the matrix couples ports on different conductors through
their mutual inductance, which a matrix per conductor drops.  Every
excitation lists the same pads, so all of them solve in one lockstep group.

With ``keep_solutions`` each port's solve is kept.  The copper is linear,
so the field of any set of port currents is their sum
(:meth:`SheetPortImpedance.superpose`): the copper's current distribution
under a circuit's solution, without another field solve.

Nothing here reads a board.  The caller states the pads as mesh cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from .sheet_operator import SheetInductanceOperator
from .sheet_peec import Cell, SheetMesh, SheetSolution, Terminal, solve_sheet_cases


@dataclass(frozen=True)
class PortPad:
    """One pad: its cells on one layer of the mesh, and on others if it spans them.

    ``other_layers`` lists ``(layer, cells)`` for every further layer the pad
    covers, such as the annular rings of a plated pin.
    """

    name: str
    layer: int
    cells: tuple[Cell, ...]
    other_layers: tuple[tuple[int, tuple[Cell, ...]], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "cells", tuple((int(r), int(c)) for r, c in self.cells))
        object.__setattr__(
            self,
            "other_layers",
            tuple((int(layer), tuple((int(r), int(c)) for r, c in cells)) for layer, cells in self.other_layers),
        )
        layers = [self.layer, *(layer for layer, _ in self.other_layers)]
        if len(set(layers)) != len(layers):
            raise ValueError(f"pad {self.name} lists a layer twice")
        if not any(cells for _, cells in self.footprint):
            raise ValueError(f"pad {self.name} covers no cells")

    @property
    def footprint(self) -> tuple[tuple[int, tuple[Cell, ...]], ...]:
        """``(layer, cells)`` for every layer the pad covers, its first layer first."""
        return ((self.layer, self.cells), *self.other_layers)


def merge_pads(name: str, pads: Iterable[PortPad]) -> PortPad:
    """One pad covering the union of ``pads``' cells (the pins of one part on one net, say)."""
    by_layer: dict[int, dict[Cell, None]] = {}
    for pad in pads:
        for layer, cells in pad.footprint:
            by_layer.setdefault(layer, {}).update(dict.fromkeys(cells))
    if not by_layer:
        raise ValueError(f"no pads to merge into {name}")
    layers = list(by_layer)
    return PortPad(
        name,
        layers[0],
        tuple(by_layer[layers[0]]),
        tuple((layer, tuple(by_layer[layer])) for layer in layers[1:]),
    )


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
    solutions: tuple[SheetSolution, ...] | None = field(default=None, repr=False, compare=False)
    """Port ``k``'s solve (1 A into it, out of its reference), with ``keep_solutions``."""

    @property
    def reference_pads(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.references[p] for p in self.ports))

    def superpose(self, port_current_a: Mapping[str, complex]) -> SheetSolution:
        """The copper's solution when each port carries the current given (others 0 A).

        ``port_current_a[p]`` flows into port ``p``'s pad and out of its
        reference, for instance ``NetworkSolution.copper_port_current_a`` of
        ``electrical.port_network.BoardNetwork``.  Node voltages and branch
        currents are the sums of the kept port solutions; every one of those
        shares its grounded nodes, so the voltages add on one gauge.
        """
        if self.solutions is None:
            raise ValueError("the port solutions were not kept; solve with keep_solutions=True")
        unknown = sorted(set(port_current_a) - set(self.ports))
        if unknown:
            raise ValueError(f"not ports of this matrix: {unknown}")
        weights = [complex(port_current_a.get(port, 0.0)) for port in self.ports]
        first = self.solutions[0]
        voltage = np.zeros_like(first.node_voltage, dtype=np.complex128)
        current = np.zeros_like(first.branch_current, dtype=np.complex128)
        for weight, solution in zip(weights, self.solutions):
            if weight != 0:
                voltage += weight * solution.node_voltage
                current += weight * solution.branch_current
        return SheetSolution(
            node_voltage=voltage,
            branch_current=current,
            frequency_hz=first.frequency_hz,
            iterations=max(s.iterations for s in self.solutions),
            residual=max(s.residual for s in self.solutions),
            converged=all(s.converged for s in self.solutions),
            grounded_node=first.grounded_node,
            undriven_nodes=first.undriven_nodes,
            lockstep_columns=first.lockstep_columns,
        )


def _pad_nodes(mesh: SheetMesh, pad: PortPad) -> list[int]:
    nodes = [mesh.node_index[(layer, row, col)] for layer, cells in pad.footprint for row, col in cells
             if (layer, row, col) in mesh.node_index]
    if not nodes:
        raise ValueError(f"pad {pad.name} has no cell on the conductor")
    return nodes


def _pad_terminals(mesh: SheetMesh, pad: PortPad, current: float) -> list[Terminal]:
    """One terminal per layer, carrying that layer's share of the pad's cells on the conductor."""
    usable = {layer: sum((layer, r, c) in mesh.node_index for r, c in cells) for layer, cells in pad.footprint}
    total = sum(usable.values())
    if total == 0:
        raise ValueError(f"pad {pad.name} has no cell on the conductor")
    return [Terminal(pad.name, layer, cells, current * usable[layer] / total)
            for layer, cells in pad.footprint if usable[layer]]


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
    keep_solutions: bool = False,
    initial: SheetPortImpedance | None = None,
    backend: str = "cpu",
    device_id: int = 0,
) -> SheetPortImpedance:
    """The impedance matrix between the ports ``references`` names (port -> its reference pad).

    ``pads`` lists every port pad and every reference pad.  A port and its
    reference have to be on one conductor (the solve refuses a component that
    does not close its current); ports of different conductors couple.
    ``keep_solutions`` keeps each port's solve for :meth:`SheetPortImpedance.superpose`.
    ``initial``: the same ports on the same mesh solved at another frequency
    with their solves kept; each port's solve starts from its solve there
    (``solve_sheet_cases``' ``initial``).
    ``backend="cuda"`` solves the ports on the device
    (:func:`~electrical.sheet_peec.sheet_cuda.solve_sheet_cases_cuda`, one
    after another, sharing the operator and the factored preconditioner);
    ``near_tiles`` is the CPU's block preconditioner's and does not apply.
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
    nodes = {pad.name: _pad_nodes(mesh, pad) for pad in pads}
    starts = None
    if initial is not None:
        if initial.solutions is None:
            raise ValueError("initial was solved without keep_solutions")
        if tuple(initial.ports) != ports or dict(initial.references) != dict(references):
            raise ValueError("initial is not of the same ports")
        starts = list(initial.solutions)

    def excitation(port: str) -> list[Terminal]:
        current = {name: 0.0 for name in by_name}
        current[port] = 1.0
        current[references[port]] = -1.0
        return [terminal for pad in pads for terminal in _pad_terminals(mesh, pad, current[pad.name])]

    if backend == "cpu":
        solutions = solve_sheet_cases(
            mesh, operator, [excitation(port) for port in ports], frequency_hz=frequency_hz, tolerance=tolerance,
            max_iterations=max_iterations, restart=restart, preconditioner=preconditioner, near_tiles=near_tiles,
            keep_terminal_components=True, initial=starts,
        )
    elif backend == "cuda":
        from .sheet_cuda import solve_sheet_cases_cuda

        solutions = [solution for solution, _ in solve_sheet_cases_cuda(
            mesh, operator, [excitation(port) for port in ports], frequency_hz=frequency_hz, tolerance=tolerance,
            max_iterations=max_iterations, restart=restart, preconditioner=preconditioner, device_id=device_id,
            keep_terminal_components=True, initial=starts,
        )]
    else:
        raise ValueError("backend must be 'cpu' or 'cuda'")
    z = np.zeros((len(ports), len(ports)), dtype=complex)
    for k, solution in enumerate(solutions):
        for j, port in enumerate(ports):
            z[j, k] = (complex(np.mean(solution.node_voltage[nodes[port]]))
                       - complex(np.mean(solution.node_voltage[nodes[references[port]]])))
    return SheetPortImpedance(
        ports=ports,
        references=dict(references),
        z_ohm=z,
        frequency_hz=float(frequency_hz),
        converged=tuple(bool(s.converged) for s in solutions),
        iterations=tuple(int(s.iterations) for s in solutions),
        solutions=tuple(solutions) if keep_solutions else None,
    )


__all__ = ["PortPad", "SheetPortImpedance", "merge_pads", "solve_port_impedance"]
