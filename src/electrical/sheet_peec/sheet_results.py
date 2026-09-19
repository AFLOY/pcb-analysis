"""Turn a sheet-PEEC solve into the quantities a plane optimizer's gates read.

The solve returns node potentials and branch currents, which is what a circuit
has.  A gate reads a current density per cell, a potential spread, the current a
layer change carries, and the loss.  This is the map between the two, and it is
where the definitions in `docs/REQUIREMENTS.md` are honoured -- the ones that
have caught errors before are here:

* the current density at a cell is a vector, so the two axes are combined as a
  magnitude rather than added
* the current crossing a layer is the current on the branches that cross it, not
  the length of a vector that also has in-plane components
* the bulk percentile excludes the terminal cells, because a lead's own
  singularity is not a property of the shape being judged
* the potential spread has two definitions and says which one it used
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .sheet_peec import SheetMesh, SheetSolution, Terminal

# A cell of one layer, as the mesh indexes it.
Cell = tuple[int, int, int]


@dataclass(frozen=True)
class SheetFields:
    """The per-cell fields a gate reads, and the metrics taken from them."""

    current_density: dict[Cell, float]
    voltage: dict[Cell, float]
    metrics: dict[str, Any]
    terminal_cells: frozenset[Cell] = field(default_factory=frozenset)


def _terminal_cells(mesh: SheetMesh, terminals: Sequence[Terminal]) -> set[Cell]:
    return {
        (terminal.layer, row, col)
        for terminal in terminals
        for row, col in terminal.cells
        if (terminal.layer, row, col) in mesh.node_index
    }


def cell_current_density_phasor(
    mesh: SheetMesh, solution: SheetSolution, *, thickness_m: Sequence[float] | None = None
) -> dict[Cell, tuple[complex, complex]]:
    """Give each cell's two in-plane density phasors, in A/mm^2.

    A branch carries the current between two cells; a cell's own density is the
    current through it, which is the mean of the branches on either side.  A cell
    at the conductor's edge has a branch on one side only, and the missing one
    carries no current, so the mean over both is the honest average of what
    crosses the cell -- not the value of the one branch that exists.

    The two axes are combined as a magnitude.  Adding them would report a cell
    carrying equal current along x and y as carrying twice what it does, and a
    cell carrying opposite currents as carrying none.
    """
    rows, cols = mesh.shape
    layers = len(mesh.stackup)
    if thickness_m is None:
        thickness_m = [layer.thickness_m for layer in mesh.stackup.layers]
    if len(thickness_m) != layers:
        raise ValueError("one thickness per layer is needed")

    along_x = np.zeros((layers, rows, cols), dtype=np.complex128)
    along_y = np.zeros_like(along_x)
    for index, (layer, row, col) in enumerate(mesh.branch_x):
        value = solution.branch_current[index]
        along_x[layer, row, col] += value
        along_x[layer, row, col + 1] += value
    offset = len(mesh.branch_x)
    for index, (layer, row, col) in enumerate(mesh.branch_y):
        value = solution.branch_current[offset + index]
        along_y[layer, row, col] += value
        along_y[layer, row + 1, col] += value

    # Both branches of an axis, whether or not both exist: a missing branch
    # carries nothing, and dividing by the two that could have been there is
    # what makes this the current through the cell rather than along one face.
    with np.errstate(invalid="ignore"):
        along_x = along_x / 2.0
        along_y = along_y / 2.0

    # Current along x crosses the cell's y extent, current along y its x extent.
    hx = mesh.grid.pitch_x_m * 1e3  # type: ignore[union-attr]
    hy = mesh.grid.pitch_y_m * 1e3  # type: ignore[union-attr]
    density: dict[Cell, tuple[complex, complex]] = {}
    for (layer, row, col) in mesh.node_index:
        thickness_mm = thickness_m[layer] * 1e3
        density[(layer, row, col)] = (
            complex(along_x[layer, row, col] / (hy[row] * thickness_mm)),
            complex(along_y[layer, row, col] / (hx[col] * thickness_mm)),
        )
    return density


def cell_current_density(
    mesh: SheetMesh, solution: SheetSolution, *, thickness_m: Sequence[float] | None = None
) -> dict[Cell, float]:
    """Give the magnitude of each cell's in-plane density, in A/mm^2."""
    phasor = cell_current_density_phasor(
        mesh, solution, thickness_m=thickness_m
    )
    return {
        cell: math.hypot(abs(value[0]), abs(value[1]))
        for cell, value in phasor.items()
    }


def vertical_currents(mesh: SheetMesh, solution: SheetSolution) -> dict[Cell, float]:
    """Give the current each vertical branch carries, in amperes.

    Keyed by the cell of the upper layer it leaves.  This is the current on the
    branches that cross a layer, not a component of an in-plane vector: a branch
    that crosses is the only thing that carries current across.
    """
    offset = len(mesh.branch_x) + len(mesh.branch_y)
    return {
        (via.upper_layer, via.row, via.col): float(
            abs(solution.branch_current[offset + index])
        )
        for index, via in enumerate(mesh.via_branches)
    }


def sheet_fields(
    mesh: SheetMesh,
    solution: SheetSolution,
    terminals: Sequence[Terminal],
    *,
    percentile: float = 99.0,
) -> SheetFields:
    """Map a solve onto the fields and metrics a gate reads."""
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    density = cell_current_density(mesh, solution)
    terminal = _terminal_cells(mesh, terminals)

    voltage: dict[Cell, float] = {}
    for cell, index in mesh.node_index.items():
        value = solution.node_voltage[index]
        voltage[cell] = (
            float(value.real) if solution.frequency_hz == 0.0 else float(abs(value))
        )

    values = np.fromiter(density.values(), dtype=float, count=len(density))
    bulk = np.fromiter(
        (value for cell, value in density.items() if cell not in terminal),
        dtype=float,
    )
    vertical = vertical_currents(mesh, solution)

    resistance = mesh.resistances()
    loss = float(np.sum(resistance * np.abs(solution.branch_current) ** 2))

    incidence = mesh.incidence()
    node_balance = incidence.T @ solution.branch_current
    injected = np.zeros(mesh.node_count, dtype=np.complex128)
    for item in terminals:
        usable = [
            mesh.node_index[(item.layer, row, col)]
            for row, col in item.cells
            if (item.layer, row, col) in mesh.node_index
        ]
        for position in usable:
            injected[position] += item.current_a / len(usable)
    closure = float(np.abs(node_balance - injected).max()) if mesh.node_count else 0.0

    metrics: dict[str, Any] = {
        "bulk_p99_current_density_a_per_mm2": (
            float(np.percentile(bulk, percentile)) if bulk.size else 0.0
        ),
        "max_current_density_a_per_mm2": float(values.max()) if values.size else 0.0,
        "voltage_span_v": solution.voltage_span_v(),
        "voltage_span_definition": solution.voltage_span_definition,
        "max_vertical_connection_current_a": max(vertical.values(), default=0.0),
        "current_closure_error_a": closure,
        "i2r_loss_w": loss,
        "conductive_node_count": mesh.node_count,
        "undriven_node_count": solution.undriven_nodes,
        "frequency_hz": solution.frequency_hz,
        "solver_iterations": solution.iterations,
        "solver_relative_residual": solution.residual,
        "converged": solution.converged,
        "terminal_cell_count": len(terminal),
        "percentile": float(percentile),
        "backend": "sheet_peec",
    }
    return SheetFields(
        current_density=density,
        voltage=voltage,
        metrics=metrics,
        terminal_cells=frozenset(terminal),
    )
