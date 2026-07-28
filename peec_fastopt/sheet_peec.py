"""A physical PEEC solve on a layered sheet mesh.

This is the circuit half of the sheet formulation: :mod:`sheet_operator`
supplies the inductance, and this supplies the conductor, the sources, and the
two laws that tie them together.

The mesh has one node per occupied cell per layer and one branch between each
pair of adjacent occupied cells.  A via adds a branch between the two layers it
joins.  With ``A`` the branch-node incidence, ``Z = R + j omega L`` the branch
impedance and ``V`` the node potentials, the unknowns satisfy

.. math::

    Z I - A V = 0 \\qquad A^{T} I = I_\\text{source}

which is Kirchhoff's voltage law along each branch and his current law at each
node.  Potential is defined up to a constant, so one node is grounded.

``L`` is dense -- every branch couples to every other -- and is never formed.
It is applied through the two-dimensional transforms of the operator, which is
what makes the system affordable.  ``R`` is diagonal.

At zero frequency the inductance drops out and what remains is exactly a
resistor network, which is the check this formulation is held to: the same
copper, the same terminals, solved as a resistor network by a direct method,
has to give the same node potentials.  Any error in the incidence, the
resistances, the source handling or the grounding shows up there, separated
from any question about the inductance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla

from .sheet_operator import (
    COPPER_RESISTIVITY_OHM_M,
    SheetInductanceOperator,
    SheetStackup,
)

Cell = tuple[int, int]
Node = tuple[int, int, int]


@dataclass(frozen=True)
class ViaBranch:
    """A vertical conductor joining one cell of two layers.

    ``resistance_ohm`` is the barrel's own resistance.  ``inductance_h`` is its
    partial self inductance.

    A vertical branch couples to no in-plane branch: the two are perpendicular
    and the defining integral carries ``dl_i . dl_j``.  It does couple to every
    other vertical branch, because those are parallel to it, and that coupling
    is not modelled here -- only the scalar self term is, and it defaults to
    zero.

    What that costs depends on where the vertical current is.  In a developed
    region there is none, so a developed in-plane profile and the resistance
    read off it do not depend on this at all.  Where vertical current does flow
    -- the entry region of a conductor, the neighbourhood of a barrel -- it is
    not negligible: for the filament links of a 3.5mm inlay on a 0.2mm grid at
    300kHz the reactance of the self term alone reaches twice the resistance.
    Adding the self term without the mutual terms would be worse than adding
    neither, since redistribution currents in neighbouring columns often oppose
    one another and the mutual terms cancel much of the loop inductance the
    self terms would claim.
    """

    row: int
    col: int
    lower_layer: int
    upper_layer: int
    resistance_ohm: float
    inductance_h: float = 0.0

    def __post_init__(self) -> None:
        if self.lower_layer == self.upper_layer:
            raise ValueError("a via has to join two different layers")
        for name in ("resistance_ohm", "inductance_h"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class Terminal:
    """Where a case drives current into the conductor.

    ``cells`` are the mesh cells of one pad on one layer.  The current is
    injected across them in proportion to nothing at all -- evenly -- because a
    pad is a short compared with the copper it feeds and the mesh has no finer
    statement of where within it the lead lands.
    """

    name: str
    layer: int
    cells: tuple[Cell, ...]
    current_a: float

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError(f"terminal {self.name} covers no cells")
        object.__setattr__(self, "cells", tuple(self.cells))
        object.__setattr__(self, "current_a", float(self.current_a))


@dataclass
class SheetMesh:
    """The conductor: which cells are copper, and how they join."""

    shape: tuple[int, int]
    pitch_m: float
    stackup: SheetStackup
    occupancy: np.ndarray
    vias: tuple[ViaBranch, ...] = ()

    node_index: dict[Node, int] = field(init=False, repr=False)
    branch_x: list[tuple[int, int, int]] = field(init=False, repr=False)
    branch_y: list[tuple[int, int, int]] = field(init=False, repr=False)
    via_branches: tuple[ViaBranch, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rows, cols = (int(value) for value in self.shape)
        expected = (len(self.stackup), rows, cols)
        occupancy = np.asarray(self.occupancy, dtype=bool)
        if occupancy.shape != expected:
            raise ValueError(f"occupancy must have shape {expected}")
        self.shape = (rows, cols)
        self.occupancy = occupancy

        self.node_index = {}
        for layer in range(len(self.stackup)):
            for row in range(rows):
                for col in range(cols):
                    if occupancy[layer, row, col]:
                        self.node_index[(layer, row, col)] = len(self.node_index)
        if not self.node_index:
            raise ValueError("the conductor is empty")

        # A branch exists where both of the cells it would join are copper.
        self.branch_x = [
            (layer, row, col)
            for layer in range(len(self.stackup))
            for row in range(rows)
            for col in range(cols - 1)
            if occupancy[layer, row, col] and occupancy[layer, row, col + 1]
        ]
        self.branch_y = [
            (layer, row, col)
            for layer in range(len(self.stackup))
            for row in range(rows - 1)
            for col in range(cols)
            if occupancy[layer, row, col] and occupancy[layer, row + 1, col]
        ]
        self.via_branches = tuple(
            via
            for via in self.vias
            if occupancy[via.lower_layer, via.row, via.col]
            and occupancy[via.upper_layer, via.row, via.col]
        )

    @property
    def vertical_levels(self) -> tuple[tuple[int, int], ...]:
        """Name the layer pairs this mesh joins with vertical branches.

        Every branch joining the same two layers spans the same height, so a
        level is the unit the vertical inductance operator is built per.  The
        order is the one branches are indexed in.
        """
        seen: list[tuple[int, int]] = []
        for via in self.via_branches:
            key = (via.lower_layer, via.upper_layer)
            if key not in seen:
                seen.append(key)
        return tuple(seen)

    @property
    def node_count(self) -> int:
        return len(self.node_index)

    @property
    def branch_count(self) -> int:
        return len(self.branch_x) + len(self.branch_y) + len(self.via_branches)

    def incidence(self) -> sp.csr_matrix:
        """Build the branch-node incidence, one row per branch.

        Row ``b`` holds ``+1`` at the node the branch leaves and ``-1`` at the
        node it enters, so that ``A V`` is the drop along each branch and
        ``A^T I`` is the current leaving each node.
        """
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for branch, (layer, row, col) in enumerate(self.branch_x):
            rows.extend((branch, branch))
            cols.extend(
                (self.node_index[(layer, row, col)], self.node_index[(layer, row, col + 1)])
            )
            data.extend((1.0, -1.0))
        offset = len(self.branch_x)
        for index, (layer, row, col) in enumerate(self.branch_y):
            branch = offset + index
            rows.extend((branch, branch))
            cols.extend(
                (self.node_index[(layer, row, col)], self.node_index[(layer, row + 1, col)])
            )
            data.extend((1.0, -1.0))
        offset += len(self.branch_y)
        for index, via in enumerate(self.via_branches):
            branch = offset + index
            rows.extend((branch, branch))
            cols.extend(
                (
                    self.node_index[(via.lower_layer, via.row, via.col)],
                    self.node_index[(via.upper_layer, via.row, via.col)],
                )
            )
            data.extend((1.0, -1.0))
        return sp.csr_matrix(
            (data, (rows, cols)), shape=(self.branch_count, self.node_count)
        )

    def resistances(self) -> np.ndarray:
        """Give each branch's resistance.

        An in-plane branch of a uniform grid is one pitch long and one pitch
        wide, so its resistance is the layer's sheet resistance whatever the
        pitch is.
        """
        values = np.empty(self.branch_count, dtype=np.float64)
        position = 0
        for group in (self.branch_x, self.branch_y):
            for layer, _row, _col in group:
                values[position] = self.stackup.layers[layer].sheet_resistance_ohm
                position += 1
        for via in self.via_branches:
            values[position] = via.resistance_ohm
            position += 1
        return values

    def scatter(self, currents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lay the in-plane branch currents out on the grid the operator wants."""
        rows, cols = self.shape
        grid_x = np.zeros((len(self.stackup), rows, cols), dtype=np.float64)
        grid_y = np.zeros_like(grid_x)
        for index, (layer, row, col) in enumerate(self.branch_x):
            grid_x[layer, row, col] = currents[index]
        offset = len(self.branch_x)
        for index, (layer, row, col) in enumerate(self.branch_y):
            grid_y[layer, row, col] = currents[offset + index]
        return grid_x, grid_y

    def scatter_vertical(self, currents: np.ndarray) -> np.ndarray:
        """Lay the vertical branch currents out by level, row and column."""
        levels = self.vertical_levels
        index_of = {key: position for position, key in enumerate(levels)}
        rows, cols = self.shape
        grid = np.zeros((len(levels), rows, cols), dtype=np.float64)
        offset = len(self.branch_x) + len(self.branch_y)
        for index, via in enumerate(self.via_branches):
            level = index_of[(via.lower_layer, via.upper_layer)]
            grid[level, via.row, via.col] = currents[offset + index]
        return grid

    def gather_vertical(self, grid: np.ndarray, into: np.ndarray) -> np.ndarray:
        """Read the vertical branch values back off the grid, in place."""
        levels = self.vertical_levels
        index_of = {key: position for position, key in enumerate(levels)}
        offset = len(self.branch_x) + len(self.branch_y)
        for index, via in enumerate(self.via_branches):
            level = index_of[(via.lower_layer, via.upper_layer)]
            into[offset + index] = grid[level, via.row, via.col]
        return into

    def gather(self, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        """Read the in-plane branch values back off the grid."""
        values = np.zeros(self.branch_count, dtype=np.float64)
        for index, (layer, row, col) in enumerate(self.branch_x):
            values[index] = grid_x[layer, row, col]
        offset = len(self.branch_x)
        for index, (layer, row, col) in enumerate(self.branch_y):
            values[offset + index] = grid_y[layer, row, col]
        return values


def via_resistance(
    barrel_length_m: float,
    drill_diameter_m: float,
    plating_thickness_m: float,
    resistivity_ohm_m: float = COPPER_RESISTIVITY_OHM_M,
) -> float:
    """Give a plated barrel's resistance from the hole it is plated in."""
    for value in (barrel_length_m, drill_diameter_m, plating_thickness_m):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("barrel dimensions must be finite and positive")
    inner = 0.5 * drill_diameter_m
    outer = inner + plating_thickness_m
    area = math.pi * (outer * outer - inner * inner)
    return resistivity_ohm_m * barrel_length_m / area


@dataclass(frozen=True)
class SheetSolution:
    """What one case leaves behind."""

    node_voltage: np.ndarray
    branch_current: np.ndarray
    frequency_hz: float
    iterations: int
    residual: float
    converged: bool
    grounded_node: int
    # Nodes on components no terminal drives.  Their potential is not solved for
    # and is left at zero, which is a placeholder and not an answer.
    undriven_nodes: int = 0

    def voltage_span_v(self) -> float:
        """Give the potential spread the gate reads, in volts.

        Two definitions, because two regimes.  At zero frequency the potentials
        are real and signed and the spread is the difference between the
        extremes.  Above it they are phasors with no common sign to take a
        difference of, and the quantity that means anything is the largest
        departure from the reference node.  This follows the convention the
        optimizer's other backends already report under
        ``voltage_span_definition``, so the numbers stay comparable.
        """
        if self.node_voltage.size == 0:
            return 0.0
        if self.frequency_hz == 0.0:
            real = self.node_voltage.real
            return float(real.max() - real.min())
        return float(np.abs(self.node_voltage).max())

    @property
    def voltage_span_definition(self) -> str:
        return (
            "signed_max_minus_min"
            if self.frequency_hz == 0.0
            else "maximum_phasor_magnitude_from_reference"
        )


def _components(incidence: sp.csr_matrix, node_count: int) -> tuple[int, np.ndarray]:
    """Label each node with the connected component of copper it belongs to."""
    adjacency = (incidence.T @ incidence).tocsr()
    return csgraph.connected_components(adjacency, directed=False)


def _components_with_terminals(
    labels: np.ndarray, components: int, injected: np.ndarray
) -> set[int]:
    """Name the components a case actually drives.

    A component with no terminal carries no current in this case.  It is
    reported and left out rather than solved: including it would leave its
    potential undetermined, and refusing the whole case would refuse a plane
    that shaping merely left an island on.
    """
    return {
        int(component)
        for component in range(components)
        if np.abs(injected[labels == component]).sum() > 0.0
    }


def _source_vector(mesh: SheetMesh, terminals: Sequence[Terminal]) -> np.ndarray:
    """Spread each terminal's current over the mesh cells its pad covers."""
    injected = np.zeros(mesh.node_count, dtype=np.complex128)
    total = 0.0
    for terminal in terminals:
        usable = [
            mesh.node_index[(terminal.layer, row, col)]
            for row, col in terminal.cells
            if (terminal.layer, row, col) in mesh.node_index
        ]
        if not usable:
            raise ValueError(
                f"terminal {terminal.name} has no cell on the conductor"
            )
        share = terminal.current_a / len(usable)
        for index in usable:
            injected[index] += share
        total += terminal.current_a
    if abs(total) > 1e-9 * max(
        1.0, sum(abs(terminal.current_a) for terminal in terminals)
    ):
        raise ValueError(
            f"the case injects a net {total:.6g} A into an isolated conductor; "
            "terminal currents have to sum to zero"
        )
    return injected


def solve_sheet_case(
    mesh: SheetMesh,
    operator: SheetInductanceOperator,
    terminals: Sequence[Terminal],
    *,
    frequency_hz: float = 0.0,
    tolerance: float = 1e-10,
    max_iterations: int = 400,
    restart: int = 60,
) -> SheetSolution:
    """Solve one excitation of the mesh.

    The saddle-point system is solved for the node potentials by eliminating
    the branch currents: with ``Y = Z^{-1}`` the system reduces to
    ``A^T Y A V = I_source``, the familiar nodal admittance form.  At zero
    frequency ``Z`` is diagonal and that reduction is exact and direct.  Above
    it, ``Z`` couples every branch to every other and is applied through the
    operator's transforms, so the reduction is carried out by a Krylov method
    with the diagonal of ``Z`` as the preconditioner.
    """
    if operator.shape != mesh.shape:
        raise ValueError("the operator and the mesh must share a shape")
    if len(operator.stackup) != len(mesh.stackup):
        raise ValueError("the operator and the mesh must share a stackup")
    frequency_hz = float(frequency_hz)
    if frequency_hz < 0.0:
        raise ValueError("frequency must be non-negative")

    incidence = mesh.incidence()
    resistance = mesh.resistances()
    injected = _source_vector(mesh, terminals)

    # Potential is defined up to a constant per connected component, so each
    # component needs its own gauge.  Grounding one node and hoping the
    # conductor is one piece made the system exactly singular the moment it was
    # not: shaping cuts islands off a plane, and a mesh with two components
    # returned NaN with a warning rather than saying so.
    components, labels = _components(incidence, mesh.node_count)
    carried = _components_with_terminals(labels, components, injected)
    if not carried:
        raise ValueError("no connected component of the conductor carries a terminal")
    kept_nodes = np.isin(labels, list(carried))
    dropped = int(mesh.node_count - kept_nodes.sum())

    # Every component that carries current must close its own current: a
    # component is isolated from the others, so a net injection into one of them
    # has nowhere to go however well the whole model balances.
    for component in sorted(carried):
        total = complex(injected[labels == component].sum())
        scale = float(np.abs(injected[labels == component]).sum())
        if abs(total) > 1e-9 * max(1.0, scale):
            raise ValueError(
                f"connected component {component} is given a net "
                f"{total:.6g} A; each isolated component has to close its own "
                "current"
            )

    keep = kept_nodes.copy()
    grounded_nodes = []
    for component in sorted(carried):
        first = int(np.flatnonzero(labels == component)[0])
        keep[first] = False
        grounded_nodes.append(first)
    grounded = grounded_nodes[0]
    # Branches inside a dropped component keep their rows, which are then all
    # zero: KVL leaves their current at zero, which is what a component nothing
    # drives should carry.  Removing them would only save iterations.
    reduced = incidence[:, keep]

    if frequency_hz == 0.0:
        admittance = sp.diags(1.0 / resistance)
        system = (reduced.T @ admittance @ reduced).tocsc()
        solution = spla.spsolve(system, injected[keep].real)
        voltage = np.zeros(mesh.node_count, dtype=np.complex128)
        voltage[keep] = solution
        current = admittance @ (reduced @ solution)
        residual = float(
            np.linalg.norm(reduced.T @ current - injected[keep].real)
            / max(np.linalg.norm(injected[keep].real), 1e-30)
        )
        return SheetSolution(
            node_voltage=voltage,
            branch_current=current.astype(np.complex128),
            frequency_hz=0.0,
            iterations=1,
            residual=residual,
            converged=residual < 1e-8,
            grounded_node=grounded,
            undriven_nodes=dropped,
        )

    omega = 2.0 * math.pi * frequency_hz
    inline_count = len(mesh.branch_x) + len(mesh.branch_y)

    # The vertical branches' own inductance is an operator, not a per-branch
    # scalar.  They are parallel to one another so they couple, and their mutual
    # terms cancel much of the loop inductance their self terms claim: adding the
    # self term alone would be worse than adding neither.  The operator has to
    # know which layer pairs the mesh joins, and a mesh carrying a level it was
    # not told about is refused rather than solved with that coupling silently
    # dropped.
    levels = mesh.vertical_levels
    if levels and tuple(levels) != tuple(operator.vertical_levels):
        raise ValueError(
            f"the mesh joins layer pairs {list(levels)} with vertical branches "
            f"but the operator was built for {list(operator.vertical_levels)}; "
            "build it with vertical_levels=mesh.vertical_levels"
        )
    has_vertical = bool(levels)

    def _flux(currents: np.ndarray) -> np.ndarray:
        """Apply the partial inductance to a real vector of branch currents."""
        grid_x, grid_y = mesh.scatter(currents)
        if has_vertical:
            grid_z = mesh.scatter_vertical(currents)
            flux_x, flux_y, flux_z = operator.apply(grid_x, grid_y, grid_z)
        else:
            flux_x, flux_y = operator.apply(grid_x, grid_y)
            flux_z = None
        flux = mesh.gather(flux_x, flux_y)
        if flux_z is not None:
            mesh.gather_vertical(flux_z, flux)
        return flux

    def impedance(currents: np.ndarray) -> np.ndarray:
        """Apply ``Z = R + j omega L`` to a vector of branch currents."""
        flux = _flux(currents.real).astype(np.complex128)
        if np.iscomplexobj(currents):
            flux = flux + 1j * _flux(currents.imag)
        return resistance * currents + 1j * omega * flux

    # The whole system is solved at once rather than by eliminating the branch
    # currents first.  Eliminating them needs Z^{-1}, and with Z dense that is
    # an inner Krylov solve inside every outer product -- the cost multiplies,
    # and on a stack of filaments it stops being affordable.  The saddle-point
    # form has one Krylov iteration and one application of Z per step.
    branches = mesh.branch_count
    unknowns = int(keep.sum())
    size = branches + unknowns

    def saddle(vector: np.ndarray) -> np.ndarray:
        currents = vector[:branches]
        voltages = vector[branches:]
        return np.concatenate(
            [impedance(currents) - reduced @ voltages, reduced.T @ currents]
        )

    # Preconditioner: the same system with Z replaced by its diagonal, which is
    # exactly solvable and, because the self term dominates the coupling,
    # close.  Its Schur complement is the sparse nodal admittance matrix, so it
    # is factored once and reused for every iteration.
    diagonal = resistance + 1j * omega * np.concatenate(
        [
            np.full(inline_count, _self_inductance(operator)),
            _vertical_self_inductance(mesh, operator),
        ]
    )
    schur = (reduced.T @ sp.diags(1.0 / diagonal) @ reduced).tocsc()
    factored = spla.splu(schur)

    def precondition(vector: np.ndarray) -> np.ndarray:
        rhs_current = vector[:branches]
        rhs_node = vector[branches:]
        # Block factorisation of [[D, -A], [A^T, 0]].
        node = factored.solve(rhs_node + reduced.T @ (rhs_current / diagonal))
        current = (rhs_current + reduced @ node) / diagonal
        return np.concatenate([current, node])

    right_hand_side = np.concatenate(
        [np.zeros(branches, dtype=np.complex128), injected[keep]]
    )
    system = spla.LinearOperator((size, size), matvec=saddle, dtype=np.complex128)
    preconditioner = spla.LinearOperator(
        (size, size), matvec=precondition, dtype=np.complex128
    )
    counter = {"iterations": 0}

    def count(_value: np.ndarray) -> None:
        counter["iterations"] += 1

    result, info = spla.gmres(
        system,
        right_hand_side,
        M=preconditioner,
        rtol=tolerance,
        restart=restart,
        maxiter=max_iterations,
        callback=count,
        callback_type="pr_norm",
    )
    current = result[:branches]
    voltage = np.zeros(mesh.node_count, dtype=np.complex128)
    voltage[keep] = result[branches:]
    residual = float(
        np.linalg.norm(saddle(result) - right_hand_side)
        / max(np.linalg.norm(right_hand_side), 1e-30)
    )
    return SheetSolution(
        node_voltage=voltage,
        branch_current=current,
        frequency_hz=frequency_hz,
        iterations=counter["iterations"],
        residual=residual,
        converged=info == 0 and residual < 1e-6,
        grounded_node=grounded,
        undriven_nodes=dropped,
    )


def _vertical_self_inductance(
    mesh: SheetMesh, operator: SheetInductanceOperator
) -> np.ndarray:
    """Read each vertical branch's own partial inductance off the operator."""
    if not mesh.via_branches:
        return np.zeros(0, dtype=np.float64)
    levels = mesh.vertical_levels
    if not levels or tuple(levels) != tuple(operator.vertical_levels):
        # No operator to read from; fall back to whatever the caller stated.
        return np.asarray(
            [via.inductance_h for via in mesh.via_branches], dtype=np.float64
        )
    rows, cols = operator.shape
    probe = np.zeros((len(levels), rows, cols))
    values = []
    for level in range(len(levels)):
        probe[:] = 0.0
        probe[level, rows // 2, cols // 2] = 1.0
        _x, _y, flux = operator.apply(
            np.zeros((len(operator.stackup), rows, cols)),
            np.zeros((len(operator.stackup), rows, cols)),
            probe,
        )
        values.append(float(flux[level, rows // 2, cols // 2]))
    index_of = {key: position for position, key in enumerate(levels)}
    return np.asarray(
        [
            values[index_of[(via.lower_layer, via.upper_layer)]]
            for via in mesh.via_branches
        ],
        dtype=np.float64,
    )


def _self_inductance(operator: SheetInductanceOperator) -> float:
    """Read one in-plane branch's own partial inductance off the operator."""
    rows, cols = operator.shape
    probe_x = np.zeros((len(operator.stackup), rows, cols))
    probe_y = np.zeros_like(probe_x)
    probe_x[0, rows // 2, cols // 2] = 1.0
    flux_x, _flux_y = operator.apply(probe_x, probe_y)
    return float(flux_x[0, rows // 2, cols // 2])
