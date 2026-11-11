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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from electrical.matrix_free_mpir_fem.grid import TensorGrid
from electrical import backend as _backend
from electrical.threads import thread_budget
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
from scipy.linalg import get_lapack_funcs

from .near_factor import (
    NEAR_REGULARIZATION,
    MumpsQuasiDefiniteNear,
    mumps_available,
    resolve_near_factor_threads,
    resolve_near_factorization,
)
from .sheet_inductance import barrel_self_inductance
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

    ``resistance_ohm`` is the barrel's own resistance.

    A vertical branch couples to no in-plane branch: the two are perpendicular
    and the defining integral carries ``dl_i . dl_j``.  It couples to every
    other vertical branch, which is parallel to it, through the operator
    (``SheetInductanceOperator(vertical_levels=mesh.vertical_levels)``: a
    kernel per pair of via levels, self terms included).  The AC solve refuses
    a mesh whose levels the operator was not built for, so no vertical
    coupling is dropped silently.  Self terms without the mutual ones would be
    worse than neither: redistribution currents in neighbouring columns often
    oppose one another and the mutual terms cancel much of the loop inductance
    the self terms would claim.

    The branch is a column of one cell's cross section, so its self
    inductance follows the cell size, not the drill of a barrel.

    ``barrel_radius_m`` (with ``barrel_bore_radius_m``, 0 for a solid pin)
    states the plated barrel the branch is: its own inductance is then the
    barrel's (:func:`~electrical.sheet_peec.sheet_inductance.barrel_self_inductance`
    over the level's span) in place of the column's, while its couplings to
    other vertical branches stay the operator's -- beyond a cell or two they
    do not depend on the cross-section.  Left out, the branch is the column,
    which is what a skin-filament link is.

    ``inductance_h`` does not enter a solved result.  The preconditioner
    falls back to it as the self term only when the operator has no vertical
    kernels for the mesh's levels, which the AC solve refuses.
    """

    row: int
    col: int
    lower_layer: int
    upper_layer: int
    resistance_ohm: float
    inductance_h: float = 0.0
    barrel_radius_m: float | None = None
    barrel_bore_radius_m: float = 0.0

    def __post_init__(self) -> None:
        if self.lower_layer == self.upper_layer:
            raise ValueError("a via has to join two different layers")
        for name in ("resistance_ohm", "inductance_h", "barrel_bore_radius_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.barrel_radius_m is not None:
            radius = float(self.barrel_radius_m)
            if not math.isfinite(radius) or radius <= 0.0 or self.barrel_bore_radius_m >= radius:
                raise ValueError("a barrel needs 0 <= bore radius < radius")
            object.__setattr__(self, "barrel_radius_m", radius)


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
    current_a: complex

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError(f"terminal {self.name} covers no cells")
        object.__setattr__(self, "cells", tuple(self.cells))
        current = complex(self.current_a)
        if not math.isfinite(current.real) or not math.isfinite(current.imag):
            raise ValueError("terminal current must be finite")
        object.__setattr__(self, "current_a", current)


@dataclass
class SheetMesh:
    """The conductor: which cells are copper, and how they join.

    The cells are those of a tensor grid: ``pitch_m`` for a uniform grid, or
    ``grid`` (a ``TensorGrid`` in metres) for graded column widths and row
    heights, in which case ``pitch_m`` may be ``None``.  A branch along x from
    cell ``(r, c)`` to ``(r, c + 1)`` is a bar centred on the shared edge, as
    long as the two half-cells and as wide as row ``r``; the y branch likewise.

    ``region`` (optional, integers shaped like ``occupancy``) keeps conductors
    apart that the grid sets side by side: two adjacent copper cells of
    different non-zero regions are not joined, so two nets a clearance
    narrower than a cell separates do not short.  A region of 0 joins any
    neighbour (a bridge over a part that joins two nets).
    """

    shape: tuple[int, int]
    pitch_m: float | None
    stackup: SheetStackup
    occupancy: np.ndarray
    vias: tuple[ViaBranch, ...] = ()
    grid: TensorGrid | None = None
    region: np.ndarray | None = None

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
        if self.grid is None:
            if self.pitch_m is None or float(self.pitch_m) <= 0.0:
                raise ValueError("pitch_m must be positive unless a grid is given")
            self.grid = TensorGrid.uniform(float(self.pitch_m), (rows, cols))
        else:
            if self.grid.shape != (rows, cols):
                raise ValueError(f"grid shape {self.grid.shape} does not match {(rows, cols)}")
            if self.pitch_m is None:
                self.pitch_m = float(self.grid.pitch_x_m[0]) if self.grid.is_uniform else None
            elif not (self.grid.is_uniform and np.isclose(float(self.pitch_m), float(self.grid.pitch_x_m[0]))):
                raise ValueError("pitch_m names one square cell; this grid's cells are not all that square (pass pitch_m=None)")
        self.shape = (rows, cols)
        self.occupancy = occupancy

        # The C++ topology joins every pair of neighbours; cell regions are
        # built here, in Python.
        core = _backend.core()
        if core is not None and self.region is None:
            via_cells = np.asarray(
                [(via.lower_layer, via.upper_layer, via.row, via.col) for via in self.vias], dtype=np.int64
            ).reshape(-1, 4)
            topology = core.sheet.MeshTopology(np.ascontiguousarray(occupancy, dtype=np.uint8), via_cells)
            self.__dict__["_topology"] = topology
            coordinates = np.argwhere(topology.node_of >= 0)
            self.node_index = dict(zip(map(tuple, coordinates.tolist()), range(topology.node_count)))
            self.branch_x = list(map(tuple, topology.branch_x.tolist()))
            self.branch_y = list(map(tuple, topology.branch_y.tolist()))
            self.via_branches = tuple(self.vias[index] for index in topology.via_kept.tolist())
            return
        self.__dict__["_topology"] = None

        self.node_index = {}
        for layer in range(len(self.stackup)):
            for row in range(rows):
                for col in range(cols):
                    if occupancy[layer, row, col]:
                        self.node_index[(layer, row, col)] = len(self.node_index)
        if not self.node_index:
            raise ValueError("the conductor is empty")

        if self.region is not None:
            region = np.asarray(self.region)
            if region.shape != expected:
                raise ValueError(f"region must have shape {expected}")
            if not np.issubdtype(region.dtype, np.integer):
                raise ValueError("region must hold integers")
            self.region = region

        def joined(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
            if not (occupancy[a] and occupancy[b]):
                return False
            if self.region is None:
                return True
            ra, rb = int(self.region[a]), int(self.region[b])
            return ra == rb or ra == 0 or rb == 0

        # A branch exists where both of the cells it would join are copper (of
        # one region, when regions are given).
        self.branch_x = [
            (layer, row, col)
            for layer in range(len(self.stackup))
            for row in range(rows)
            for col in range(cols - 1)
            if joined((layer, row, col), (layer, row, col + 1))
        ]
        self.branch_y = [
            (layer, row, col)
            for layer in range(len(self.stackup))
            for row in range(rows - 1)
            for col in range(cols)
            if joined((layer, row, col), (layer, row + 1, col))
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
    def is_uniform(self) -> bool:
        return self.grid.is_uniform  # type: ignore[union-attr]

    def branch_geometry(self, axis: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Centre x, centre y, length along the current and width of every possible branch of one axis.

        Arrays are ``(rows, cols - 1)`` for ``"x"`` and ``(rows - 1, cols)`` for
        ``"y"``, indexed like the operator's current grids, whether or not the
        cells are copper.
        """

        grid = self.grid
        assert grid is not None
        hx, hy = grid.pitch_x_m, grid.pitch_y_m
        xe, ye = grid.x_edges_m - grid.x_edges_m[0], grid.y_edges_m - grid.y_edges_m[0]
        xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
        if axis == "x":
            centre_x = np.broadcast_to(xe[1:-1][None, :], (hy.size, hx.size - 1))
            centre_y = np.broadcast_to(yc[:, None], (hy.size, hx.size - 1))
            length = np.broadcast_to((0.5 * (hx[:-1] + hx[1:]))[None, :], (hy.size, hx.size - 1))
            width = np.broadcast_to(hy[:, None], (hy.size, hx.size - 1))
        elif axis == "y":
            centre_x = np.broadcast_to(xc[None, :], (hy.size - 1, hx.size))
            centre_y = np.broadcast_to(ye[1:-1][:, None], (hy.size - 1, hx.size))
            length = np.broadcast_to((0.5 * (hy[:-1] + hy[1:]))[:, None], (hy.size - 1, hx.size))
            width = np.broadcast_to(hx[None, :], (hy.size - 1, hx.size))
        else:
            raise ValueError("axis must be 'x' or 'y'")
        return (np.ascontiguousarray(centre_x), np.ascontiguousarray(centre_y), np.ascontiguousarray(length), np.ascontiguousarray(width))

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
        topology = self.__dict__.get("_topology")
        if topology is not None:
            count = self.branch_count
            rows_ = np.repeat(np.arange(count, dtype=np.int64), 2)
            cols_ = np.empty(2 * count, dtype=np.int64)
            cols_[0::2] = topology.left
            cols_[1::2] = topology.right
            data_ = np.tile(np.array([1.0, -1.0]), count)
            return sp.csr_matrix((data_, (rows_, cols_)), shape=(count, self.node_count))
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

        An in-plane branch is a bar ``length / width`` squares long, so its
        resistance is the layer's sheet resistance times that ratio; on a
        uniform grid the ratio is one whatever the pitch is.
        """
        topology = self.__dict__.get("_topology")
        if topology is not None:
            grid = self.grid
            assert grid is not None
            return topology.resistances(
                np.asarray([layer.sheet_resistance_ohm for layer in self.stackup.layers], dtype=np.float64),
                np.ascontiguousarray(grid.pitch_x_m, dtype=np.float64),
                np.ascontiguousarray(grid.pitch_y_m, dtype=np.float64),
                np.asarray([via.resistance_ohm for via in self.vias], dtype=np.float64),
            )
        values = np.empty(self.branch_count, dtype=np.float64)
        position = 0
        for axis, group in (("x", self.branch_x), ("y", self.branch_y)):
            _cx, _cy, length, width = self.branch_geometry(axis)
            squares = length / width
            for layer, row, col in group:
                values[position] = self.stackup.layers[layer].sheet_resistance_ohm * squares[row, col]
                position += 1
        for via in self.via_branches:
            values[position] = via.resistance_ohm
            position += 1
        return values

    def _branch_positions(self) -> dict[str, Any]:
        """Index arrays of every branch's grid position, built once per mesh.

        ``scatter``/``gather`` and their vertical variants run once per
        operator application, hundreds of times a solve; walking the branch
        lists in Python each time was a fifth of a sheet solve.  An in-plane
        position is never repeated, so assigning through these arrays writes
        exactly what the loop wrote.  Two vertical branches on one cell of one
        level would make the last one win in the loop, which fancy assignment
        does not promise; such a mesh keeps the loop (``vertical_unique``).
        """
        cached = self.__dict__.get("_positions")
        if cached is not None:
            return cached
        def columns(group: list[tuple[int, int, int]]) -> tuple[np.ndarray, ...]:
            if not group:
                empty = np.zeros(0, dtype=np.intp)
                return empty, empty, empty
            array = np.asarray(group, dtype=np.intp)
            return array[:, 0], array[:, 1], array[:, 2]
        index_of = {key: position for position, key in enumerate(self.vertical_levels)}
        vertical = columns(
            [
                (index_of[(via.lower_layer, via.upper_layer)], via.row, via.col)
                for via in self.via_branches
            ]
        )
        keys = set(zip(*(axis.tolist() for axis in vertical)))
        positions = {
            "x": columns(self.branch_x),
            "y": columns(self.branch_y),
            "vertical": vertical,
            "vertical_unique": len(keys) == len(self.via_branches),
        }
        self.__dict__["_positions"] = positions
        return positions

    def scatter(self, currents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lay the in-plane branch currents out on the grid the operator wants."""
        rows, cols = self.shape
        grid_x = np.zeros((len(self.stackup), rows, cols), dtype=np.float64)
        grid_y = np.zeros_like(grid_x)
        positions = self._branch_positions()
        count_x = len(self.branch_x)
        count_y = len(self.branch_y)
        grid_x[positions["x"]] = currents[:count_x]
        grid_y[positions["y"]] = currents[count_x : count_x + count_y]
        return grid_x, grid_y

    def scatter_vertical(self, currents: np.ndarray) -> np.ndarray:
        """Lay the vertical branch currents out by level, row and column."""
        levels = self.vertical_levels
        rows, cols = self.shape
        grid = np.zeros((len(levels), rows, cols), dtype=np.float64)
        offset = len(self.branch_x) + len(self.branch_y)
        positions = self._branch_positions()
        if positions["vertical_unique"]:
            grid[positions["vertical"]] = currents[offset : offset + len(self.via_branches)]
            return grid
        index_of = {key: position for position, key in enumerate(levels)}
        for index, via in enumerate(self.via_branches):
            level = index_of[(via.lower_layer, via.upper_layer)]
            grid[level, via.row, via.col] = currents[offset + index]
        return grid

    def gather_vertical(self, grid: np.ndarray, into: np.ndarray) -> np.ndarray:
        """Read the vertical branch values back off the grid, in place."""
        offset = len(self.branch_x) + len(self.branch_y)
        positions = self._branch_positions()
        into[offset : offset + len(self.via_branches)] = grid[positions["vertical"]]
        return into

    def gather(self, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        """Read the in-plane branch values back off the grid."""
        values = np.zeros(self.branch_count, dtype=np.float64)
        positions = self._branch_positions()
        count_x = len(self.branch_x)
        count_y = len(self.branch_y)
        values[:count_x] = grid_x[positions["x"]]
        values[count_x : count_x + count_y] = grid_y[positions["y"]]
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
    # How many excitations shared this one's Krylov steps (1: solved alone).
    lockstep_columns: int = 1

    def voltage_span_v(self) -> float:
        """Give the potential spread the gate reads, in volts.

        The largest difference between the potentials of two nodes.  At zero
        frequency the potentials are real and that is the difference between
        the extremes.  Above it they are phasors, and it is the diameter of
        their cloud in the complex plane, ``max |V_i - V_j|``, which is the
        same number at zero frequency and does not depend on which node the
        solve grounded.  (It used to be the largest departure from the
        grounded node, the first node in mesh order: a change that added a
        sliver cell above the copper's top row moved that node across the
        board and the EVM-2L's VIN and SW spans at 214 kHz by +56 % and
        -35 %, the copper's conduction unchanged.)
        """
        if self.node_voltage.size == 0:
            return 0.0
        if self.frequency_hz == 0.0:
            real = self.node_voltage.real
            return float(real.max() - real.min())
        return phasor_diameter(self.node_voltage)

    @property
    def voltage_span_definition(self) -> str:
        return "signed_max_minus_min" if self.frequency_hz == 0.0 else "maximum_phasor_difference"


def phasor_diameter(values: np.ndarray) -> float:
    """``max |v_i - v_j|`` over a set of phasors: the diameter of their cloud in the complex plane.

    The farthest pair lies on the convex hull, so the pairs are searched among
    its vertices only; points on one line (a hull of no area) are measured
    along that line.
    """
    from scipy.spatial import ConvexHull, QhullError

    points = np.column_stack([np.real(values).ravel(), np.imag(values).ravel()]).astype(np.float64)
    if points.shape[0] < 2:
        return 0.0
    centred = points - points.mean(axis=0)
    scale = float(np.abs(centred).max())
    if scale == 0.0:
        return 0.0
    try:
        candidates = points[ConvexHull(centred / scale).vertices]
    except QhullError:
        # Collinear (or nearly): along the line's direction the diameter is max - min.
        _, _, axes = np.linalg.svd(centred, full_matrices=False)
        along = centred @ axes[0]
        return float(along.max() - along.min())
    best = 0.0
    for start in range(0, candidates.shape[0], 1024):
        block = candidates[start:start + 1024]
        distance = np.sqrt(((block[:, None, :] - candidates[None, :, :]) ** 2).sum(axis=2))
        best = max(best, float(distance.max()))
    return best


def _components(incidence: sp.csr_matrix, node_count: int) -> tuple[int, np.ndarray]:
    """Label each node with the connected component of copper it belongs to."""
    adjacency = (incidence.T @ incidence).tocsr()
    return csgraph.connected_components(adjacency, directed=False)


def _components_with_terminals(
    labels: np.ndarray, components: int, injected: np.ndarray, touched: np.ndarray | None = None
) -> set[int]:
    """Name the components a case actually drives.

    A component with no terminal carries no current in this case.  It is
    reported and left out rather than solved: including it would leave its
    potential undetermined, and refusing the whole case would refuse a plane
    that shaping merely left an island on.  ``touched`` (a node mask) adds the
    components a terminal touches whatever its current: those are solved too,
    each on its own gauge with no net current, which is what reading the
    voltage induced on an open port needs.
    """
    driven = {
        int(component)
        for component in range(components)
        if np.abs(injected[labels == component]).sum() > 0.0
    }
    if touched is not None:
        driven |= {int(component) for component in np.unique(labels[touched])}
    return driven


def _terminal_nodes(mesh: SheetMesh, terminals: Sequence[Terminal]) -> np.ndarray:
    """Mask of the mesh nodes the terminals' pads cover."""
    mask = np.zeros(mesh.node_count, dtype=bool)
    for terminal in terminals:
        for row, col in terminal.cells:
            index = mesh.node_index.get((terminal.layer, row, col))
            if index is not None:
                mask[index] = True
    return mask


def _source_vector(mesh: SheetMesh, terminals: Sequence[Terminal]) -> np.ndarray:
    """Spread each terminal's current over the mesh cells its pad covers."""
    topology = mesh.__dict__.get("_topology")
    if topology is not None:
        terminals = list(terminals)
        counts = [len(terminal.cells) for terminal in terminals]
        cell_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        cells = np.asarray(
            [cell for terminal in terminals for cell in terminal.cells], dtype=np.int64
        ).reshape(-1, 2)
        injected, usable = topology.source_vector(
            np.asarray([terminal.layer for terminal in terminals], dtype=np.int64),
            cell_start,
            cells,
            np.asarray([terminal.current_a for terminal in terminals], dtype=np.complex128),
        )
        for terminal, count in zip(terminals, usable.tolist()):
            if count == 0:
                raise ValueError(f"terminal {terminal.name} has no cell on the conductor")
        total = sum(terminal.current_a for terminal in terminals)
        if abs(total) > 1e-9 * max(1.0, sum(abs(terminal.current_a) for terminal in terminals)):
            raise ValueError(
                f"the case injects a net {total:.6g} A into an isolated conductor; "
                "terminal currents have to sum to zero"
            )
        return np.asarray(injected, dtype=np.complex128)
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


class _BarrelSelf:
    """An inductance operator whose barrel vias carry the barrel's own inductance.

    Each vertical branch with a barrel adds ``L_barrel - L_column`` on its own
    diagonal: in every product, in the near-field matrix and in the self terms
    the preconditioners read.  Everything else is the wrapped operator's.
    """

    def __init__(self, base: Any, mesh: "SheetMesh", delta: np.ndarray) -> None:
        self._base = base
        self._delta_branch = delta
        rows, cols = mesh.shape
        index_of = {key: position for position, key in enumerate(mesh.vertical_levels)}
        grid = np.zeros((len(mesh.vertical_levels), rows, cols), dtype=np.float64)
        for via, value in zip(mesh.via_branches, delta):
            grid[index_of[(via.lower_layer, via.upper_layer)], via.row, via.col] = value
        self._delta_grid = grid
        self._via_offset = len(mesh.branch_x) + len(mesh.branch_y)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def apply(self, currents_x: np.ndarray, currents_y: np.ndarray, currents_z: np.ndarray | None = None):
        if currents_z is None:
            return self._base.apply(currents_x, currents_y)
        flux_x, flux_y, flux_z = self._base.apply(currents_x, currents_y, currents_z)
        return flux_x, flux_y, flux_z + self._delta_grid * currents_z

    def apply_many(self, currents: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray | None]]) -> list:
        if hasattr(self._base, "apply_many"):
            applied = self._base.apply_many(currents)
        else:
            applied = [self._base.apply(x, y, z) if z is not None else self._base.apply(x, y) for x, y, z in currents]
        out = []
        for result, (_, _, currents_z) in zip(applied, currents):
            if currents_z is not None and len(result) == 3:
                result = (result[0], result[1], result[2] + self._delta_grid * currents_z)
            out.append(result)
        return out

    def near_inductance(self, mesh: "SheetMesh", *args: Any, **kwargs: Any) -> sp.csr_matrix:
        near = self._base.near_inductance(mesh, *args, **kwargs).tocsr()
        index = self._via_offset + np.arange(len(self._delta_branch))
        return (near + sp.csr_matrix((self._delta_branch, (index, index)), shape=near.shape)).tocsr()

    def vertical_self_inductance(self, mesh: "SheetMesh") -> np.ndarray:
        return _vertical_self_inductance(mesh, self._base) + self._delta_branch


def _barrel_span(mesh: "SheetMesh", via: ViaBranch) -> float:
    return abs(mesh.stackup.layers[via.upper_layer].z_m - mesh.stackup.layers[via.lower_layer].z_m)


def _check_barrels_apart(mesh: "SheetMesh", barrels: list[int]) -> None:
    """Refuse two barrels whose circles overlap over a common height: one conductor, not two."""
    grid = mesh.grid
    assert grid is not None
    xc = 0.5 * (grid.x_edges_m[:-1] + grid.x_edges_m[1:])
    yc = 0.5 * (grid.y_edges_m[:-1] + grid.y_edges_m[1:])
    vias = [mesh.via_branches[k] for k in barrels]
    points = np.array([(xc[v.col], yc[v.row]) for v in vias])
    radii = np.array([v.barrel_radius_m for v in vias])
    z = np.array([sorted((mesh.stackup.layers[v.lower_layer].z_m, mesh.stackup.layers[v.upper_layer].z_m)) for v in vias])
    for i, j in sorted(cKDTree(points).query_pairs(2.0 * float(radii.max()))):
        apart = float(np.hypot(*(points[i] - points[j])))
        common = min(z[i, 1], z[j, 1]) - max(z[i, 0], z[j, 0])
        if apart < radii[i] + radii[j] and common > 1e-12:
            a, b = vias[i], vias[j]
            raise ValueError(
                f"the barrels of the vias at cells {(a.row, a.col)} and {(b.row, b.col)} overlap "
                f"({apart * 1e3:.3f} mm apart, radii {radii[i] * 1e3:.3f} and {radii[j] * 1e3:.3f} mm)"
            )


def _with_barrels(mesh: "SheetMesh", operator: Any) -> Any:
    """The operator itself, or wrapped so that barrel vias carry their own inductance."""
    barrels = [k for k, via in enumerate(mesh.via_branches) if via.barrel_radius_m is not None]
    if not barrels:
        return operator
    _check_barrels_apart(mesh, barrels)
    column = _vertical_self_inductance(mesh, operator)
    delta = np.zeros(len(mesh.via_branches), dtype=np.float64)
    for k in barrels:
        via = mesh.via_branches[k]
        delta[k] = barrel_self_inductance(_barrel_span(mesh, via), via.barrel_radius_m, via.barrel_bore_radius_m) - column[k]
    return _BarrelSelf(operator, mesh, delta)


def solve_sheet_case(
    mesh: SheetMesh,
    operator: SheetInductanceOperator,
    terminals: Sequence[Terminal],
    *,
    frequency_hz: float = 0.0,
    tolerance: float = 1e-10,
    max_iterations: int = 400,
    restart: int = 60,
    preconditioner: str = "auto",
    near_tiles: int | None = None,
    keep_terminal_components: bool = False,
    near_factorization: str = "auto",
    near_regularization: float | None = None,
    near_factor_threads: int | str | None = None,
) -> SheetSolution:
    """Solve one excitation of the mesh.

    The saddle-point system is solved for the node potentials by eliminating
    the branch currents: with ``Y = Z^{-1}`` the system reduces to
    ``A^T Y A V = I_source``, the familiar nodal admittance form.  At zero
    frequency ``Z`` is diagonal and that reduction is exact and direct.  Above
    it, ``Z`` couples every branch to every other and is applied through the
    operator's transforms, so the reduction is carried out by a Krylov method.

    ``preconditioner="auto"`` (default) picks ``near``, or ``block`` from
    :data:`AUTO_BLOCK_FROM_UNKNOWNS` saddle-point unknowns when the near
    matrix would be factored by SuperLU (no MUMPS in the core).  ``preconditioner="block"`` keeps the near-field impedance but factors it
    on its own and approximates the Schur complement by the diagonal
    variant's nodal admittance: block elimination at a fraction of the
    saddle LU's memory.  ``preconditioner="near"`` factors the saddle-point matrix with
    ``Z`` replaced by ``R + j omega L_near``, the exact partial inductance of
    each branch with itself and its neighbours on its own layer within the
    operator's preconditioner radius (``operator.near_inductance(mesh)`` less
    its couplings across layers), by sparse LU;
    ``"diagonal"`` keeps only the self terms, which is exactly solvable
    through a Schur complement but leaves every mutual term to the Krylov
    iterations.

    ``near_tiles`` (``block`` only) factors ``R + j omega L_near`` in that many
    strips of the node order, dropping the couplings between strips: a weaker
    preconditioner of the same system, whose strips factor with less fill and
    are factored and solved side by side on the thread budget.  ``None`` takes
    :data:`DEFAULT_NEAR_TILES`; ``1`` is the whole near block.

    ``near_factorization`` (``near`` only): ``"auto"`` (default) is MUMPS
    where the core links it and SuperLU otherwise; ``"superlu"`` factors the
    saddle-point matrix by SciPy's sparse LU; ``"mumps_quasi_definite"``
    factors its quasi-definite form (node rows negated, ``-delta I`` in the
    zero block, ``delta`` = ``near_regularization`` over the median
    ``|Z_ii|``, ``None`` for :data:`.near_factor.NEAR_REGULARIZATION`) as a
    symmetric LDL^T by MUMPS, in a core built with ``-DPCB_NATIVE_MUMPS=ON``
    (see :mod:`.near_factor`).  MUMPS runs on ``near_factor_threads``
    threads: ``None`` (default) is one, so that every thread budget gives the
    same bits; an integer, or ``"budget"`` for the thread budget, is faster
    and gives the same bits only at the same count.
    """
    return solve_sheet_cases(
        mesh, operator, [terminals], frequency_hz=frequency_hz, tolerance=tolerance,
        max_iterations=max_iterations, restart=restart, preconditioner=preconditioner, near_tiles=near_tiles,
        keep_terminal_components=keep_terminal_components, near_factorization=near_factorization,
        near_regularization=near_regularization, near_factor_threads=near_factor_threads,
    )[0]


def solve_sheet_cases(
    mesh: SheetMesh,
    operator: SheetInductanceOperator,
    terminal_sets: Sequence[Sequence[Terminal]],
    *,
    frequency_hz: float = 0.0,
    tolerance: float = 1e-10,
    max_iterations: int = 400,
    restart: int = 60,
    preconditioner: str = "auto",
    near_tiles: int | None = None,
    keep_terminal_components: bool = False,
    initial: Sequence[SheetSolution | None] | None = None,
    near_factorization: str = "auto",
    near_regularization: float | None = None,
    near_factor_threads: int | str | None = None,
) -> list[SheetSolution]:
    """Solve several excitations of one mesh, in the order given.

    Each set of terminals is the excitation :func:`solve_sheet_case` takes,
    with the same options.  Excitations that drive the same components of the
    conductor share the reduced saddle-point system and its preconditioner,
    and their Krylov solves run in lockstep: every step applies the operator
    and the preconditioner to all of them at once, so the factors and the
    transforms are read once per step for the whole group instead of once
    per excitation.  Each lockstep solve is the restarted GMRES that
    :func:`solve_sheet_case` runs, step for step; an excitation alone in its
    group is solved exactly as :func:`solve_sheet_case` solves it.

    ``keep_terminal_components``: a component that a terminal touches is
    solved even when the excitation puts no current into it (a terminal of
    ``0`` A), on its own gauge and closing its own (zero) current.  The
    voltages on it are then those induced by the driven copper: the
    open-port voltages a multi-port impedance is read from.  Off by default,
    when such a component is left out like any undriven island.

    ``initial`` (one entry per excitation, ``None`` for none): a solution of
    the same mesh and terminals at another frequency, from which the Krylov
    solve starts.  Its branch currents and node potentials are split into
    their real and imaginary parts and the start is the combination of those
    four vectors with the least residual here: between frequencies the
    current's distribution moves slowly and the potentials scale with the
    impedance, so the start is already close.  It changes the iterations,
    not the solution's tolerance.

    ``near_factorization``, ``near_regularization`` and
    ``near_factor_threads``: as in
    :func:`solve_sheet_case`.
    """
    near_factorization = resolve_near_factorization(near_factorization)
    if near_factorization == "mumps_quasi_definite" and not mumps_available():
        raise ValueError(
            "near_factorization='mumps_quasi_definite' needs the C++ core built with -DPCB_NATIVE_MUMPS=ON"
        )
    if near_regularization is None:
        near_regularization = NEAR_REGULARIZATION[near_factorization]
    factor_threads = resolve_near_factor_threads(near_factor_threads)
    if preconditioner not in ("auto", "near", "block", "diagonal"):
        raise ValueError("preconditioner must be 'auto', 'near', 'block' or 'diagonal'")
    if getattr(mesh, "strands", None) is not None:
        # The block and diagonal variants read one self inductance per grid
        # branch; only the near factor knows the strands.
        if preconditioner not in ("auto", "near"):
            raise ValueError("a mesh with edge strands is solved with the near preconditioner")
        preconditioner = "near"
    near_tiles = DEFAULT_NEAR_TILES if near_tiles is None else int(near_tiles)
    if near_tiles < 1:
        raise ValueError("near_tiles must be at least 1")
    if operator.shape != mesh.shape:
        raise ValueError("the operator and the mesh must share a shape")
    if len(operator.stackup) != len(mesh.stackup):
        raise ValueError("the operator and the mesh must share a stackup")
    frequency_hz = float(frequency_hz)
    if frequency_hz < 0.0:
        raise ValueError("frequency must be non-negative")
    if initial is not None and len(initial) != len(terminal_sets):
        raise ValueError("initial needs one entry (or None) per excitation")

    incidence = mesh.incidence()
    resistance = mesh.resistances()
    cases = []
    for terminals in terminal_sets:
        injected = _source_vector(mesh, terminals)
        if frequency_hz == 0.0 and float(np.max(np.abs(injected.imag))) > 1e-15:
            raise ValueError("DC terminal currents must be real")
        touched = _terminal_nodes(mesh, terminals) if keep_terminal_components else None
        cases.append((injected, _reduce(mesh, incidence, injected, touched)))
    if frequency_hz == 0.0:
        return [_solve_dc(mesh, resistance, reduction, injected) for injected, reduction in cases]

    levels = mesh.vertical_levels
    if not levels or tuple(levels) == tuple(operator.vertical_levels):  # else _solve_ac refuses the mesh
        operator = _with_barrels(mesh, operator)
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, (_, reduction) in enumerate(cases):
        groups.setdefault(reduction.carried, []).append(index)
    solutions: list[SheetSolution | None] = [None] * len(cases)
    for members in groups.values():
        solved = _solve_ac(
            mesh, operator, resistance, cases[members[0]][1], [cases[k][0] for k in members],
            frequency_hz, tolerance, max_iterations, restart, preconditioner, near_tiles,
            None if initial is None else [initial[k] for k in members],
            near_factorization=near_factorization, near_regularization=near_regularization,
            near_factor_threads=factor_threads,
        )
        for index, solution in zip(members, solved):
            solutions[index] = solution
    return [solution for solution in solutions if solution is not None]


@dataclass(frozen=True)
class _Reduction:
    """The driven part of the conductor for one excitation: what the solve keeps."""

    carried: tuple[int, ...]
    keep: np.ndarray
    grounded: int
    dropped: int
    active_branches: np.ndarray
    active_incidence: sp.csr_matrix
    reduced: sp.csr_matrix


def _reduce(
    mesh: SheetMesh, incidence: sp.csr_matrix, injected: np.ndarray, touched: np.ndarray | None = None
) -> _Reduction:
    # Potential is defined up to a constant per connected component, so each
    # component needs its own gauge.  Grounding one node and hoping the
    # conductor is one piece made the system exactly singular the moment it was
    # not: shaping cuts islands off a plane, and a mesh with two components
    # returned NaN with a warning rather than saying so.
    components, labels = _components(incidence, mesh.node_count)
    carried = _components_with_terminals(labels, components, injected, touched)
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
    # Remove every branch that is not wholly inside a driven component.
    # Keeping an undriven island's KVL row without its KCL columns does *not*
    # leave its current at zero once the dense inductance operator couples it
    # to driven copper; it creates a current with nowhere to close.  The stated
    # policy is to omit undriven islands, so their branches must leave the
    # unknown vector as well as their nodes.
    driven_endpoint_count = np.asarray(
        abs(incidence) @ kept_nodes.astype(np.int8)
    ).reshape(-1)
    active_branches = driven_endpoint_count == 2
    active_incidence = incidence[active_branches]
    reduced = active_incidence[:, keep]

    return _Reduction(
        carried=tuple(sorted(carried)),
        keep=keep,
        grounded=grounded,
        dropped=dropped,
        active_branches=active_branches,
        active_incidence=active_incidence,
        reduced=reduced,
    )


def _solve_dc(mesh: SheetMesh, resistance: np.ndarray, reduction: _Reduction, injected: np.ndarray) -> SheetSolution:
    keep, grounded, dropped = reduction.keep, reduction.grounded, reduction.dropped
    active_branches, reduced = reduction.active_branches, reduction.reduced
    active_resistance = resistance[active_branches]
    admittance = sp.diags(1.0 / active_resistance)
    system = (reduced.T @ admittance @ reduced).tocsc()
    solution = spla.spsolve(system, injected[keep].real)
    voltage = np.zeros(mesh.node_count, dtype=np.complex128)
    voltage[keep] = solution
    active_current = admittance @ (reduced @ solution)
    current = np.zeros(mesh.branch_count, dtype=np.complex128)
    current[active_branches] = active_current
    residual = float(
        np.linalg.norm(
            reduced.T @ active_current - injected[keep].real
        )
        / max(np.linalg.norm(injected[keep].real), 1e-30)
    )
    return SheetSolution(
        node_voltage=voltage,
        branch_current=current,
        frequency_hz=0.0,
        iterations=1,
        residual=residual,
        converged=residual < 1e-8,
        grounded_node=grounded,
        undriven_nodes=dropped,
    )


def _solve_ac(
    mesh: SheetMesh,
    operator: Any,
    resistance: np.ndarray,
    reduction: _Reduction,
    injected_list: list[np.ndarray],
    frequency_hz: float,
    tolerance: float,
    max_iterations: int,
    restart: int,
    preconditioner: str,
    near_tiles: int,
    initial_list: list[SheetSolution | None] | None = None,
    *,
    near_factorization: str = "auto",
    near_regularization: float | None = None,
    near_factor_threads: int = 1,
) -> list[SheetSolution]:
    keep, grounded, dropped = reduction.keep, reduction.grounded, reduction.dropped
    active_branches, active_incidence, reduced = (
        reduction.active_branches, reduction.active_incidence, reduction.reduced,
    )
    active_resistance = resistance[active_branches]
    omega = 2.0 * math.pi * frequency_hz
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
    # Edge strands (:mod:`.edge_strands`): what the grid gets wrong about the
    # strands, added to the grid's flux at branch level.
    correction = getattr(mesh, "inductance_correction", None)

    def _flux(currents: np.ndarray) -> np.ndarray:
        """Apply the partial inductance to a real vector of branch currents."""
        full_currents = np.zeros(mesh.branch_count, dtype=np.float64)
        full_currents[active_branches] = currents
        grid_x, grid_y = mesh.scatter(full_currents)
        if has_vertical:
            grid_z = mesh.scatter_vertical(full_currents)
            flux_x, flux_y, flux_z = operator.apply(grid_x, grid_y, grid_z)
        else:
            flux_x, flux_y = operator.apply(grid_x, grid_y)
            flux_z = None
        flux = mesh.gather(flux_x, flux_y)
        if flux_z is not None:
            mesh.gather_vertical(flux_z, flux)
        if correction is not None:
            flux += correction @ full_currents
        return flux[active_branches]

    def impedance(currents: np.ndarray) -> np.ndarray:
        """Apply ``Z = R + j omega L`` to a vector of branch currents."""
        flux = _flux(currents.real).astype(np.complex128)
        if np.iscomplexobj(currents):
            flux = flux + 1j * _flux(currents.imag)
        return active_resistance * currents + 1j * omega * flux

    def impedance_block(currents: np.ndarray) -> np.ndarray:
        """:func:`impedance` of every column, the real and imaginary parts of all in one operator call."""
        parts = [currents.real[:, k] for k in range(currents.shape[1])] + [
            currents.imag[:, k] for k in range(currents.shape[1])
        ]
        grids = []
        for part in parts:
            full_currents = np.zeros(mesh.branch_count, dtype=np.float64)
            full_currents[active_branches] = part
            grid_x, grid_y = mesh.scatter(full_currents)
            grids.append((grid_x, grid_y, mesh.scatter_vertical(full_currents) if has_vertical else None))
        if hasattr(operator, "apply_many"):
            applied = operator.apply_many(grids)
        else:
            applied = [operator.apply(*grid) if grid[2] is not None else operator.apply(grid[0], grid[1]) for grid in grids]
        fluxes = []
        for part, result in zip(parts, applied):
            flux = mesh.gather(result[0], result[1])
            if len(result) == 3:
                mesh.gather_vertical(result[2], flux)
            if correction is not None:
                full_currents = np.zeros(mesh.branch_count, dtype=np.float64)
                full_currents[active_branches] = part
                flux += correction @ full_currents
            fluxes.append(flux[active_branches])
        count = currents.shape[1]
        flux = np.stack(fluxes[:count], axis=1).astype(np.complex128) + 1j * np.stack(fluxes[count:], axis=1)
        return active_resistance[:, None] * currents + 1j * omega * flux

    # The whole system is solved at once rather than by eliminating the branch
    # currents first.  Eliminating them needs Z^{-1}, and with Z dense that is
    # an inner Krylov solve inside every outer product -- the cost multiplies,
    # and on a stack of filaments it stops being affordable.  The saddle-point
    # form has one Krylov iteration and one application of Z per step.
    branches = int(active_branches.sum())
    unknowns = int(keep.sum())
    size = branches + unknowns

    def saddle(vector: np.ndarray) -> np.ndarray:
        currents = vector[:branches]
        voltages = vector[branches:]
        return np.concatenate(
            [impedance(currents) - reduced @ voltages, reduced.T @ currents]
        )

    def saddle_block(block: np.ndarray) -> np.ndarray:
        currents = block[:branches]
        voltages = block[branches:]
        return np.concatenate(
            [impedance_block(currents) - reduced @ voltages, reduced.T @ currents]
        )

    def starts(given: list[SheetSolution | None], right_hand_sides: np.ndarray) -> np.ndarray | None:
        """Each column's start from its solution at another frequency: the least-residual mix of its parts."""
        columns = [k for k, solution in enumerate(given) if solution is not None]
        if not columns:
            return None
        parts = []
        for k in columns:
            solution = given[k]
            if solution.branch_current.shape != (mesh.branch_count,) or solution.node_voltage.shape != (mesh.node_count,):
                raise ValueError("an initial solution is not of this mesh")
            current = solution.branch_current[active_branches]
            voltage = solution.node_voltage[keep]
            no_current, no_voltage = np.zeros(branches), np.zeros(unknowns)
            parts += [np.concatenate([current.real, no_voltage]), np.concatenate([current.imag, no_voltage]),
                      np.concatenate([no_current, voltage.real]), np.concatenate([no_current, voltage.imag])]
        basis = np.stack(parts, axis=1).astype(np.complex128)
        images = saddle_block(basis)
        x0 = np.zeros(right_hand_sides.shape, dtype=np.complex128)
        for n, k in enumerate(columns):
            coefficients, *_ = np.linalg.lstsq(images[:, 4 * n:4 * n + 4], right_hand_sides[:, k], rcond=None)
            x0[:, k] = basis[:, 4 * n:4 * n + 4] @ coefficients
        return x0

    if preconditioner == "auto":
        preconditioner = choose_preconditioner(size, near_factorization)
    # The near block's strips are factored and solved on this pool; it lives
    # for the solve and its width is the budget, capped by the strips.  The
    # near LU solves a lockstep group's columns on it, a pair at a time.
    if preconditioner == "block":
        width = min(thread_budget(), near_tiles)
    elif preconditioner == "near" and len(injected_list) > 1 and near_factorization == "superlu":
        width = min(thread_budget(), -(-len(injected_list) // NEAR_SOLVE_COLUMNS))
    else:
        width = 1
    pool = ThreadPoolExecutor(max_workers=width) if width > 1 else None
    closers: list[Callable[[], None]] = []
    try:
        precondition = _build_preconditioner(
            mesh, operator, preconditioner, omega, resistance, active_branches, reduced, branches,
            near_strips=_near_strips(active_incidence, near_tiles) if preconditioner == "block" else None,
            pool=pool, near_factorization=near_factorization, near_regularization=near_regularization,
            closers=closers, near_factor_threads=near_factor_threads,
        )

        def solve_alone(injected: np.ndarray, given: SheetSolution | None = None) -> SheetSolution:
            right_hand_side = np.concatenate(
                [np.zeros(branches, dtype=np.complex128), injected[keep]]
            )
            x0 = None if given is None else starts([given], right_hand_side[:, None])[:, 0]
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
                x0=x0,
                M=preconditioner,
                rtol=tolerance,
                restart=restart,
                maxiter=max_iterations,
                callback=count,
                callback_type="pr_norm",
            )
            current = np.zeros(mesh.branch_count, dtype=np.complex128)
            current[active_branches] = result[:branches]
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

        def solve_together(chunk: list[np.ndarray], given: list[SheetSolution | None]) -> list[SheetSolution]:
            right_hand_sides = np.stack(
                [np.concatenate([np.zeros(branches, dtype=np.complex128), injected[keep]]) for injected in chunk],
                axis=1,
            )
            results, infos, iterations = _gmres_lockstep(
                saddle_block, precondition, right_hand_sides, tolerance, restart, max_iterations,
                x0=starts(given, right_hand_sides),
            )
            residuals = np.linalg.norm(saddle_block(results) - right_hand_sides, axis=0) / np.maximum(
                np.linalg.norm(right_hand_sides, axis=0), 1e-30
            )
            solutions = []
            for column in range(results.shape[1]):
                current = np.zeros(mesh.branch_count, dtype=np.complex128)
                current[active_branches] = results[:branches, column]
                voltage = np.zeros(mesh.node_count, dtype=np.complex128)
                voltage[keep] = results[branches:, column]
                solutions.append(
                    SheetSolution(
                        node_voltage=voltage,
                        branch_current=current,
                        frequency_hz=frequency_hz,
                        iterations=iterations[column],
                        residual=float(residuals[column]),
                        converged=bool(infos[column] == 0 and residuals[column] < 1e-6),
                        grounded_node=grounded,
                        undriven_nodes=dropped,
                        lockstep_columns=len(chunk),
                    )
                )
            return solutions

        given_list = initial_list if initial_list is not None else [None] * len(injected_list)
        if len(injected_list) == 1:
            return [solve_alone(injected_list[0], given_list[0])]
        # Measured once the factors exist, so the memory they took is not counted twice.
        width = lockstep_width(len(injected_list), lockstep_column_bytes(size, restart, operator))
        solutions: list[SheetSolution] = []
        for first in range(0, len(injected_list), width):
            chunk = injected_list[first : first + width]
            given = given_list[first : first + width]
            solutions.extend([solve_alone(chunk[0], given[0])] if len(chunk) == 1 else solve_together(chunk, given))
        return solutions
    finally:
        for close in closers:
            close()
        if pool is not None:
            pool.shutdown()




def _gmres_lockstep(
    matvec: Callable[[np.ndarray], np.ndarray],
    psolve: Callable[[np.ndarray], np.ndarray],
    right_hand_sides: np.ndarray,
    rtol: float,
    restart: int,
    maxiter: int,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Restarted, left-preconditioned GMRES on each column of ``right_hand_sides``, in lockstep.

    Every column runs the algorithm of :func:`scipy.sparse.linalg.gmres`
    (SciPy 1.17: modified Gram-Schmidt, Givens rotations, the inner tolerance
    of gh-8400) on its own Krylov space; only ``matvec`` and ``psolve`` are
    called on the block of columns still iterating, which is what lets a
    group of excitations read the preconditioner's factors once per step.
    A column whose inner loop has stopped waits for the others before the
    restart.  ``x0`` starts the columns as ``x0`` starts SciPy's.  Returns
    the solutions, each column's ``info`` (0: converged, else ``maxiter``)
    and its inner iteration count.
    """

    size, count = right_hand_sides.shape
    dtype = right_hand_sides.dtype
    eps = np.finfo(dtype).eps
    restart = min(restart, size)
    lartg = get_lapack_funcs("lartg", dtype=dtype)
    x = np.zeros((size, count), dtype=dtype) if x0 is None else np.array(x0, dtype=dtype)
    bnrm2 = [float(np.linalg.norm(right_hand_sides[:, j])) for j in range(count)]
    atol = [max(0.0, float(rtol) * value) for value in bnrm2]
    done = [value == 0.0 for value in bnrm2]
    info = [0] * count
    iterations = [0] * count
    preconditioned = psolve(right_hand_sides)
    ptol_max_factor = [1.0] * count
    ptol = [
        float(np.linalg.norm(preconditioned[:, j])) * min(1.0, atol[j] / bnrm2[j]) if bnrm2[j] else 0.0
        for j in range(count)
    ]
    presid = [0.0] * count
    rnorm = [0.0] * count
    r = right_hand_sides.copy()
    started = [j for j in range(count) if not done[j] and x[:, j].any()]
    if started:
        r[:, started] = right_hand_sides[:, started] - matvec(x[:, started])
        for j in started:
            rnorm[j] = float(np.linalg.norm(r[:, j]))
            done[j] = rnorm[j] < atol[j]
    width = min(thread_budget(), count)
    pool = ThreadPoolExecutor(max_workers=width) if width > 1 else None
    try:
        return _lockstep_cycles(
            matvec, psolve, right_hand_sides, x, r, atol, bnrm2, done, info, iterations, ptol, ptol_max_factor,
            presid, rnorm, restart, maxiter, eps, lartg, pool,
        )
    finally:
        if pool is not None:
            pool.shutdown()


def _lockstep_cycles(
    matvec: Callable[[np.ndarray], np.ndarray],
    psolve: Callable[[np.ndarray], np.ndarray],
    right_hand_sides: np.ndarray,
    x: np.ndarray,
    r: np.ndarray,
    atol: list[float],
    bnrm2: list[float],
    done: list[bool],
    info: list[int],
    iterations: list[int],
    ptol: list[float],
    ptol_max_factor: list[float],
    presid: list[float],
    rnorm: list[float],
    restart: int,
    maxiter: int,
    eps: float,
    lartg: Any,
    pool: ThreadPoolExecutor | None,
) -> tuple[np.ndarray, list[int], list[int]]:
    """The restart cycles of :func:`_gmres_lockstep` (its state passed in)."""

    size, count = right_hand_sides.shape
    dtype = right_hand_sides.dtype
    for _ in range(maxiter):
        cycle = [j for j in range(count) if not done[j]]
        if not cycle:
            break
        v = {j: np.empty((restart + 1, size), dtype=dtype) for j in cycle}
        h = {j: np.zeros((restart, restart + 1), dtype=dtype) for j in cycle}
        givens = {j: np.zeros((restart, 2), dtype=dtype) for j in cycle}
        rhs = {j: np.zeros(restart + 1, dtype=dtype) for j in cycle}
        last = {j: 0 for j in cycle}
        breakdown = {j: False for j in cycle}
        start = psolve(r[:, cycle])
        for position, j in enumerate(cycle):
            v[j][0, :] = start[:, position]
            norm = np.linalg.norm(v[j][0, :])
            v[j][0, :] *= 1 / norm
            rhs[j][0] = norm
        running = list(cycle)
        for col in range(restart):
            if not running:
                break
            block = np.stack([v[j][col, :] for j in running], axis=1)
            w_block = psolve(matvec(block))

            def orthogonalize(position: int, j: int, col: int = col, w_block: np.ndarray = w_block) -> bool:
                """One Arnoldi step of column ``j``; True while its inner loop goes on."""
                w = np.array(w_block[:, position])
                h0 = np.linalg.norm(w)
                for k in range(col + 1):
                    tmp = np.vdot(v[j][k, :], w)
                    h[j][col, k] = tmp
                    w -= tmp * v[j][k, :]
                h1 = np.linalg.norm(w)
                h[j][col, col + 1] = h1
                v[j][col + 1, :] = w[:]
                if h1 <= eps * h0:
                    h[j][col, col + 1] = 0
                    breakdown[j] = True
                else:
                    v[j][col + 1, :] *= 1 / h1
                for k in range(col):
                    c, s_ = givens[j][k, 0], givens[j][k, 1]
                    n0, n1 = h[j][col, [k, k + 1]]
                    h[j][col, [k, k + 1]] = [c * n0 + s_ * n1, -s_.conj() * n0 + c * n1]
                c, s_, mag = lartg(h[j][col, col], h[j][col, col + 1])
                givens[j][col, :] = [c, s_]
                h[j][col, [col, col + 1]] = mag, 0
                tmp = -np.conjugate(s_) * rhs[j][col]
                rhs[j][[col, col + 1]] = [c * rhs[j][col], tmp]
                presid[j] = np.abs(tmp)
                iterations[j] += 1
                last[j] = col
                return not (presid[j] <= ptol[j] or breakdown[j])

            # The columns' Krylov bases are independent and NumPy's vector
            # kernels release the GIL, so their Gram-Schmidt steps run side by
            # side; each is the same arithmetic as alone.
            if pool is not None and len(running) > 1:
                going = list(pool.map(orthogonalize, range(len(running)), running))
            else:
                going = [orthogonalize(position, j) for position, j in enumerate(running)]
            running = [j for j, keep_going in zip(running, going) if keep_going]
        for j in cycle:
            col = last[j]
            if h[j][col, col] == 0:
                rhs[j][col] = 0
            y = np.zeros([col + 1], dtype=dtype)
            y[:] = rhs[j][: col + 1]
            for k in range(col, 0, -1):
                if y[k] != 0:
                    y[k] /= h[j][k, k]
                    tmp = y[k]
                    y[:k] -= tmp * h[j][k, :k]
            if y[0] != 0:
                y[0] /= h[j][0, 0]
            x[:, j] += y @ v[j][: col + 1, :]
        residual_block = right_hand_sides[:, cycle] - matvec(x[:, cycle])
        for position, j in enumerate(cycle):
            r[:, j] = residual_block[:, position]
            rnorm[j] = float(np.linalg.norm(r[:, j]))
            if rnorm[j] <= atol[j] or breakdown[j]:
                done[j] = True
                continue
            if presid[j] <= ptol[j]:
                ptol_max_factor[j] = max(eps, 0.25 * ptol_max_factor[j])
            else:
                ptol_max_factor[j] = min(1.0, 1.5 * ptol_max_factor[j])
            ptol[j] = presid[j] * min(ptol_max_factor[j], atol[j] / rnorm[j])
    for j in range(count):
        info[j] = 0 if (bnrm2[j] == 0.0 or rnorm[j] <= atol[j]) else maxiter
    return x, info, iterations


def _branch_layers(mesh: SheetMesh) -> np.ndarray:
    """The layer of every branch in mesh order; -1 for the vertical branches."""

    planar = [layer for layer, _row, _col in mesh.branch_x] + [layer for layer, _row, _col in mesh.branch_y]
    layers = np.concatenate(
        [np.asarray(planar, dtype=np.int64), np.full(len(mesh.via_branches), -1, dtype=np.int64)]
    )
    strands = getattr(mesh, "strands", None)
    if strands is not None:
        layers = np.concatenate([layers, layers[strands.extra_parent]])
    return layers


def _conductor_of_layer(name: str) -> str:
    """The copper layer a stackup layer belongs to: ``F.Cu#3``, a skin filament, is ``F.Cu``."""
    return str(name).split("#", 1)[0]


def _near_impedance(
    mesh: SheetMesh, operator: Any, omega: float, resistance: np.ndarray, active_branches: np.ndarray
) -> sp.csr_matrix:
    """``R + j omega L_near`` over the active branches: the sparse near-field impedance.

    ``L_near`` keeps a branch's couplings to the branches near it on its own
    layer -- all the filaments of one cut layer counting as that layer -- (and
    the vertical branches' among themselves), not those to the
    layers above and below.  Two layers a tenth of a millimetre apart couple
    almost as strongly a few cells away as next door, so cutting that coupling
    at the near radius gives a worse approximation than leaving it out: on a
    four-layer hot loop (183k saddle unknowns, seven excitations, tolerance
    1e-6) dropping it cut GMRES from 211 to 135 iterations and the time in
    the preconditioner from 195 s to 43 s (the solve from 264 s to 73 s),
    the loop inductance unchanged to 1e-6 of itself.
    """

    near = operator.near_inductance(mesh)
    strands = getattr(mesh, "strands", None)
    if strands is not None:
        # The grid's near field between uncut branches, and the exact partial
        # inductances of every pair with a strand in it (``near_exact``).
        base = strands.base_count
        count = mesh.branch_count
        keep = np.ones(count)
        keep[strands.cut_branches] = 0.0
        keep[base:] = 0.0
        mask = sp.diags(keep)
        grid = sp.csr_matrix(near)[:base, :base]
        grid = sp.bmat([[grid, None], [None, sp.csr_matrix((count - base, count - base))]], format="csr")
        near = (mask @ grid @ mask + mesh.near_exact).tocsr()
    near = near.tocoo()
    # A layer cut into skin filaments is one conductor: its filaments sit
    # microns apart and couple almost as strongly as a filament to itself, so
    # their mutual terms stay (dropped, a 71 um plane cut seven ways took the
    # GMRES of a 30 MHz hot loop from 17 iterations to its cap).  Filaments
    # are the stackup layers named ``<layer>#<k>``.
    parent = np.asarray(
        [_conductor_of_layer(layer.name) for layer in mesh.stackup.layers] + [-1], dtype=object
    )
    sheet_ids = {name: index for index, name in enumerate(dict.fromkeys(parent[:-1]))}
    sheet = np.asarray([sheet_ids[name] for name in parent[:-1]] + [-1], dtype=np.int64)
    layer = sheet[_branch_layers(mesh)]
    same = layer[near.row] == layer[near.col]
    near = sp.csr_matrix((near.data[same], (near.row[same], near.col[same])), shape=near.shape)
    active_index = np.flatnonzero(active_branches)
    near = near[active_index][:, active_index]
    return (sp.diags(resistance[active_branches]) + 1j * omega * near).tocsr()


# Columns of a lockstep group the near LU solves together on one thread.
# Measured on a hot-loop mesh (183k unknowns, 104M LU entries, 7 columns):
# 0.91 s as one solve, 0.56 s on 2 threads, 0.37 s on 4 (pairs), 0.42 s on 7.
NEAR_SOLVE_COLUMNS = 2


# From this many saddle-point unknowns (branches + nodes) ``auto`` takes block
# when the near matrix would be factored by SuperLU (a core without MUMPS, or
# ``near_factorization="superlu"``), for memory alone: where SuperLU's saddle
# LU, at its largest per-unknown peak (8 to 10 KB), reaches 4 GB
# (docs/SHEET_PRECONDITIONER_AUTO_RESULTS.json).  Factored by MUMPS the near
# matrix fits far beyond, and ``auto`` is near at every size.  A fixed size
# rather than the memory free keeps the choice, and the bits, the same on
# every machine.
AUTO_BLOCK_FROM_UNKNOWNS = 400_000

# Strips the ``block`` preconditioner factors the near-field impedance in;
# 1 factors it whole.  Four is the fastest on the preconditioner benchmark's
# 0.2 mm plane at 8 threads and no slower than the whole block anywhere
# (docs/SHEET_NEAR_TILES_RESULTS.json).
DEFAULT_NEAR_TILES = 4

# Excitations solved in lockstep each hold a Krylov basis of restart + 1
# vectors of the saddle-point system and their own share of the operator's
# transforms, so a group's memory grows with its width: on a four-layer board
# at 0.1 mm six excitations of one mesh took 3.5 GB of bases alone.  A group
# is therefore cut into lockstep columns that fit in this fraction of the
# memory available when its Krylov solve starts (the factors are built by
# then).  The cut changes nothing but the time: every column gets the bits
# it gets alone.
LOCKSTEP_MEMORY_FRACTION = 0.5
# Work vectors of the saddle-point size a lockstep column holds besides its
# basis: the solution, residual, matvec and preconditioner blocks.
LOCKSTEP_WORK_VECTORS = 8
# Bytes per padded spectral plane element, per grid cell, live at the peak
# of one transform pass: the zero-padded input, the spectra, the accumulated
# products and the inverse's output (8 + 16 + 16 + 8 over a 2 x 2 padding,
# which the operator's padding to twice the grid makes 4 cells per cell).
LOCKSTEP_FFT_BYTES_PER_CELL = 4 * (8 + 16 + 16 + 8)
# The count above is within 5% of the peak RSS a column adds on the
# benchmark's two- and four-layer meshes, on either side of it; a column is
# reckoned this much larger so that a cut sized by it fits
# (docs/SHEET_LOCKSTEP_MEMORY_RESULTS.json).
LOCKSTEP_ESTIMATE_MARGIN = 1.1


def lockstep_column_bytes(size: int, restart: int, operator: Any) -> int:
    """Memory one excitation adds to a lockstep group of saddle-point ``size``.

    Its Krylov basis and work vectors in complex128, and its part of the
    batched transforms: every column is applied as a real and an imaginary
    set of currents, each a plane per layer and axis plus one per vertical
    level.  An operator without a uniform transform (the pFFT operator) is
    counted by its basis and work vectors only.  The count is taken
    :data:`LOCKSTEP_ESTIMATE_MARGIN` times.
    """
    counted = 16 * int(size) * (int(restart) + 1 + LOCKSTEP_WORK_VECTORS)
    shape = getattr(operator, "shape", None)
    stackup = getattr(operator, "stackup", None)
    if shape is not None and stackup is not None and hasattr(operator, "apply_many"):
        planes = 2 * (2 * len(stackup) + len(getattr(operator, "vertical_levels", ())))
        counted += planes * LOCKSTEP_FFT_BYTES_PER_CELL * int(shape[0]) * int(shape[1])
    return int(LOCKSTEP_ESTIMATE_MARGIN * counted)


def _available_memory_bytes() -> int | None:
    """``MemAvailable`` of ``/proc/meminfo``: what can be allocated without swapping. None off Linux."""
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def lockstep_width(count: int, column_bytes: int) -> int:
    """How many of ``count`` excitations to solve in one lockstep group, at least one.

    All of them when the memory available is unknown.
    """
    available = _available_memory_bytes()
    if available is None or column_bytes <= 0:
        return count
    return max(1, min(count, int(LOCKSTEP_MEMORY_FRACTION * available) // int(column_bytes)))


def choose_preconditioner(unknowns: int, near_factorization: str = "mumps_quasi_definite") -> str:
    """What ``auto`` takes, given the resolved near factorization.

    With MUMPS, ``near`` at every size (docs/SHEET_PEEC.md, "near at every
    size"): block was slower everywhere it was measured, 2234 s against 85 s
    on a 1.5M-unknown board conductor, and not finished in 90 minutes on two
    others near solved in 97 s and 1849 s.  With SuperLU, ``block`` from
    :data:`AUTO_BLOCK_FROM_UNKNOWNS`, where SuperLU's saddle LU would need
    more than 4 GB: slower, but it fits.
    """

    if near_factorization == "superlu" and unknowns >= AUTO_BLOCK_FROM_UNKNOWNS:
        return "block"
    return "near"


def _near_strips(active_incidence: sp.csr_matrix, tiles: int) -> list[np.ndarray] | None:
    """The active branches in ``tiles`` strips of equal size, in the order of their left node.

    The mesh numbers its nodes in raster order (layer, row, column), so a strip
    is a band of rows.  ``None`` for one strip: the whole near block.
    """

    branches = active_incidence.shape[0]
    if tiles <= 1 or branches == 0:
        return None
    entries = active_incidence.tocoo()
    left = np.empty(branches, dtype=np.int64)
    leaves = entries.data > 0
    left[entries.row[leaves]] = entries.col[leaves]
    order = np.argsort(left, kind="stable")
    tiles = min(tiles, branches)
    bounds = (np.arange(tiles + 1) * branches) // tiles
    return [order[bounds[k] : bounds[k + 1]] for k in range(tiles)]


def _factor_near_strips(
    impedance: sp.csr_matrix, strips: list[np.ndarray], pool: ThreadPoolExecutor | None
) -> Callable[[np.ndarray], np.ndarray]:
    """``Z_near`` with the couplings between strips dropped, factored strip by strip.

    A weaker preconditioner than the whole near block, but the strips factor
    with far less fill than the whole and are independent, so they are solved
    side by side on ``pool``; the result does not depend on the pool's width.

    The strips are factored on the calling thread.  A SciPy ``splu`` run on a
    worker thread left its memory behind when its factors were freed: the
    preconditioner benchmark's strips grew a process by 0.46 GB per two solves
    at a budget of 8 and not at all when factored here, and the factorisation
    is a small part of a solve.
    """

    factors = [spla.splu(impedance[m][:, m].tocsc(), permc_spec="MMD_AT_PLUS_A") for m in strips]

    def solve(vector: np.ndarray) -> np.ndarray:
        out = np.empty_like(vector)

        def strip(index: int) -> None:
            members = strips[index]
            out[members] = factors[index].solve(vector[members])

        if pool is not None:
            list(pool.map(strip, range(len(strips))))
        else:
            for index in range(len(strips)):
                strip(index)
        return out

    return solve


def _build_preconditioner(
    mesh: SheetMesh,
    operator: Any,
    kind: str,
    omega: float,
    resistance: np.ndarray,
    active_branches: np.ndarray,
    reduced: sp.csr_matrix,
    branches: int,
    near_strips: list[np.ndarray] | None = None,
    pool: ThreadPoolExecutor | None = None,
    near_factorization: str = "superlu",
    near_regularization: float = 1e-8,
    closers: list[Callable[[], None]] | None = None,
    near_factor_threads: int = 1,
) -> Callable[[np.ndarray], np.ndarray]:
    """The approximate inverse of the saddle-point matrix used by the inner GMRES.

    ``closers`` collects what the caller calls when the solve is done (the
    MUMPS factor hands its analysis back for the mesh's next frequency).
    """

    if kind == "near" and hasattr(operator, "near_inductance") and near_factorization == "mumps_quasi_definite":
        impedance = _near_impedance(mesh, operator, omega, resistance, active_branches)
        factor = MumpsQuasiDefiniteNear(impedance, reduced, near_regularization, threads=near_factor_threads)
        if closers is not None:
            closers.append(factor.close)
        # Every column of a lockstep block in one MUMPS solve: its OpenMP team
        # and the multi-column triangular solves are the parallelism.
        return factor.solve
    if kind == "near" and hasattr(operator, "near_inductance"):
        impedance = _near_impedance(mesh, operator, omega, resistance, active_branches)
        saddle = sp.bmat([[impedance, -reduced], [reduced.T, None]], format="csc")
        factored = spla.splu(saddle)
        if pool is None:
            return factored.solve

        def precondition_columns(vector: np.ndarray) -> np.ndarray:
            # SuperLU's solve releases the GIL, and one solve of a few columns
            # does not keep a core busy: pairs of columns on the pool.  A
            # column's bits do not depend on the columns solved beside it.
            if vector.ndim == 1 or vector.shape[1] <= NEAR_SOLVE_COLUMNS:
                return factored.solve(vector)
            starts = range(0, vector.shape[1], NEAR_SOLVE_COLUMNS)
            parts = pool.map(
                lambda start: factored.solve(np.ascontiguousarray(vector[:, start:start + NEAR_SOLVE_COLUMNS])),
                starts,
            )
            return np.concatenate(list(parts), axis=1)

        return precondition_columns
    if kind == "block" and hasattr(operator, "near_inductance"):
        # The same near-field impedance, applied by block elimination instead
        # of one LU of the whole saddle-point matrix.  Z_near is factored on
        # its own (a local stencil, so the fill stays small), and the Schur
        # complement A^T Z_near^-1 A, which is dense, is approximated by
        # A^T diag(Z)^-1 A, the nodal admittance of the diagonal variant.
        # The saddle LU needs about 13x the memory of these two factors and
        # is what runs a board at the acceptance grid out of memory.
        impedance = _near_impedance(mesh, operator, omega, resistance, active_branches)
        if near_strips is None:
            solve_z = spla.splu(impedance.tocsc(), permc_spec="MMD_AT_PLUS_A").solve
        else:
            solve_z = _factor_near_strips(impedance, near_strips, pool)
        full_diagonal = resistance + 1j * omega * np.concatenate(
            [_inline_self_inductance(mesh, operator), _vertical_self_inductance(mesh, operator)]
        )
        diagonal = full_diagonal[active_branches]
        schur = (reduced.T @ sp.diags(1.0 / diagonal) @ reduced).tocsc()
        factored_s = spla.splu(schur, permc_spec="MMD_AT_PLUS_A")

        def precondition_block(vector: np.ndarray) -> np.ndarray:
            rhs_current = vector[:branches]
            rhs_node = vector[branches:]
            # [[Z, -A], [A^T, 0]] [x; y] = [r1; r2]:
            # x = Z^-1 (r1 + A y),  A^T Z^-1 A y = r2 - A^T Z^-1 r1.
            z = solve_z(rhs_current)
            node = factored_s.solve(rhs_node - reduced.T @ z)
            current = solve_z(rhs_current + reduced @ node)
            return np.concatenate([current, node])

        return precondition_block
    # The same system with Z replaced by its diagonal, which is exactly
    # solvable: its Schur complement is the sparse nodal admittance matrix,
    # factored once and reused for every iteration.
    full_diagonal = resistance + 1j * omega * np.concatenate(
        [_inline_self_inductance(mesh, operator), _vertical_self_inductance(mesh, operator)]
    )
    diagonal = full_diagonal[active_branches]
    schur = (reduced.T @ sp.diags(1.0 / diagonal) @ reduced).tocsc()
    factored = spla.splu(schur)

    def precondition(vector: np.ndarray) -> np.ndarray:
        rhs_current = vector[:branches]
        rhs_node = vector[branches:]
        # Block factorisation of [[D, -A], [A^T, 0]].
        scale = diagonal if vector.ndim == 1 else diagonal[:, None]
        node = factored.solve(rhs_node + reduced.T @ (rhs_current / scale))
        current = (rhs_current + reduced @ node) / scale
        return np.concatenate([current, node])

    return precondition


def _vertical_self_inductance(
    mesh: SheetMesh, operator: SheetInductanceOperator
) -> np.ndarray:
    """Read each vertical branch's own partial inductance off the operator."""
    if not mesh.via_branches:
        return np.zeros(0, dtype=np.float64)
    reporter = getattr(operator, "vertical_self_inductance", None)
    if reporter is not None and tuple(mesh.vertical_levels) == tuple(operator.vertical_levels):
        return np.asarray(reporter(mesh), dtype=np.float64)
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


def _inline_self_inductance(mesh: SheetMesh, operator: Any) -> np.ndarray:
    """Own partial inductance of every in-plane branch, in branch order.

    An operator that knows the branch geometry (the graded-grid pFFT operator)
    reports it per branch; the uniform convolution operator has one value.
    """
    reporter = getattr(operator, "branch_self_inductance", None)
    if reporter is not None:
        return np.asarray(reporter(mesh), dtype=np.float64)
    inline_count = len(mesh.branch_x) + len(mesh.branch_y)
    return np.full(inline_count, _self_inductance(operator))


def _self_inductance(operator: SheetInductanceOperator) -> float:
    """Read one in-plane branch's own partial inductance off the operator."""
    rows, cols = operator.shape
    probe_x = np.zeros((len(operator.stackup), rows, cols))
    probe_y = np.zeros_like(probe_x)
    probe_x[0, rows // 2, cols // 2] = 1.0
    flux_x, _flux_y = operator.apply(probe_x, probe_y)
    return float(flux_x[0, rows // 2, cols // 2])
