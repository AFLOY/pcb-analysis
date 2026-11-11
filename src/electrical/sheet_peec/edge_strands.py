"""Edge strands: the in-plane branches along a conductor's edge cut across their width.

At high frequency the current of a strip crowds toward its edges, on a scale of
the skin depth and the thickness rather than of the cell.  A uniform grid puts
one bar across each cell, so the edge cell's current is spread evenly over the
cell and its loss is underestimated; the error falls only slowly with the pitch
(the strip pair of ``work/skin_ref``: width 1.2 mm, 30 MHz, seven filaments a
layer, -31/-27/-22 % in R at 0.4/0.2/0.1 mm).  Grading the rows toward the
edges on a tensor grid closed it (-0.05 % with 10 um edge rows, L within
0.1 %), but a tensor grid refines whole rows, which on a board means
everywhere.

Here only the edge branches are cut, into strands graded toward the open side
(the first ``first_width_m`` wide, each next ``growth`` times wider, the
innermost taking the rest).  The strands of a branch join the same two nodes;
the first strand keeps the branch's index and the others are appended after
the vertical branches.

The uniform operator is used as the projection grid of a precorrected FFT,
across the branch only: a strand's current is spread over three grid
positions across its branch by quadratic Lagrange weights of its offset, and
it reads its flux back with the same weights, so the grid carries its first
and second moments and the far field is right to third order in
(pitch / distance).  (Lumped at its branch's centre instead, the strip's R
kept rising as the correction radius grew: 1.573, 1.640, 1.691 mOhm/mm at
radius 1, 2, 3 against the FEM's 1.802.)  For every strand ``s`` and every
like-directed branch or strand ``j`` of the same conductor within
``radius_cells``, the grid's value is replaced by the exact one:

    C[s, j] = M(s, j) - sum_mn w_sm w_jn M_grid(pos_s + m, pos_j + n).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from .sheet_inductance import closed_form_mutual_inductance_arrays
from .sheet_peec import SheetMesh, _conductor_of_layer

EDGE_STRAND_GROWTH = 2.0
EDGE_STRAND_RADIUS_CELLS = 2


def graded_widths(pitch_m: float, open_minus: bool, open_plus: bool, first_m: float, growth: float) -> list[tuple[float, float]]:
    """``(centre offset, width)`` of the strands across one branch, innermost first.

    Offsets run from ``-pitch/2`` (the minus side) to ``+pitch/2``.  A branch
    open on neither side, or too narrow to cut, is one strand.
    """
    sides = int(open_minus) + int(open_plus)
    budget = pitch_m / sides if sides else 0.0
    graded: list[float] = []
    size = first_m
    while sides and budget - sum(graded) - size >= size:
        graded.append(size)
        size *= growth
    if not graded:
        return [(0.0, pitch_m)]
    inner = pitch_m - sides * sum(graded)
    strands: list[tuple[float, float]] = []
    low = -pitch_m / 2.0 + (sum(graded) if open_minus else 0.0)
    strands.append((low + inner / 2.0, inner))
    if open_minus:
        edge = -pitch_m / 2.0
        for width in graded:
            strands.append((edge + width / 2.0, width))
            edge += width
    if open_plus:
        edge = pitch_m / 2.0
        for width in graded:
            strands.append((edge - width / 2.0, width))
            edge -= width
    return strands


def lagrange_across(offset_cells: np.ndarray, position: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Three grid offsets and their quadratic Lagrange weights for points ``offset_cells`` from their cell.

    The stencil is centred on the cell, shifted to stay on the grid
    (``position`` the cell's index across, ``size`` the grid's extent).
    """
    first = np.clip(position - 1, 0, max(size - 3, 0)) - position
    nodes = first[:, None] + np.arange(3)[None, :]
    if size < 3:
        nodes = np.zeros((offset_cells.size, 3), dtype=np.int64)
        weights = np.zeros((offset_cells.size, 3))
        weights[:, 0] = 1.0
        return nodes, weights
    x = offset_cells[:, None]
    weights = np.ones((offset_cells.size, 3))
    for k in range(3):
        for m in range(3):
            if m != k:
                weights[:, k] *= (x[:, 0] - nodes[:, m]) / (nodes[:, k] - nodes[:, m])
    return nodes.astype(np.int64), weights


@dataclass(frozen=True)
class EdgeStrands:
    """Which branches are cut and how.

    Per strand: ``branch`` (the mesh branch it belongs to), ``strand_index``
    (its place in the stranded branch vector: the branch's own for the first
    strand, from the mesh's branch count up for the rest), ``offset_m`` and
    ``width_m`` across the branch, and its projection: ``axis`` (0 x, 1 y),
    ``node`` (three grid offsets across) and ``weight``.
    """

    base_count: int
    branch: np.ndarray
    strand_index: np.ndarray
    offset_m: np.ndarray
    width_m: np.ndarray
    axis: np.ndarray
    cell: np.ndarray       # (S, 3): layer, row, col of the branch's grid position
    node: np.ndarray       # (S, 3)
    weight: np.ndarray     # (S, 3)

    @property
    def extra_count(self) -> int:
        return int(np.count_nonzero(self.strand_index >= self.base_count))

    @property
    def extra_parent(self) -> np.ndarray:
        """The branch each appended strand belongs to, in appended order."""
        extra = self.strand_index >= self.base_count
        order = np.argsort(self.strand_index[extra])
        return self.branch[extra][order]

    @property
    def cut_branches(self) -> np.ndarray:
        return np.unique(self.branch)


def plan_edge_strands(mesh: SheetMesh, first_width_m: float, growth: float = EDGE_STRAND_GROWTH) -> EdgeStrands:
    """Cut every in-plane branch with an open side across its width."""
    if mesh.pitch_m is None or not mesh.is_uniform:
        raise ValueError("edge strands need a uniform grid")
    if not first_width_m > 0.0 or not growth > 1.0:
        raise ValueError("first_width_m must be positive and growth above 1")
    pitch = float(mesh.pitch_m)
    occupancy = mesh.occupancy
    layers, rows, cols = occupancy.shape

    def filled(layer: int, row: int, col: int) -> bool:
        return 0 <= row < rows and 0 <= col < cols and bool(occupancy[layer, row, col])

    branch: list[int] = []
    strand_index: list[int] = []
    offset: list[float] = []
    width: list[float] = []
    axis_of: list[int] = []
    cell: list[tuple[int, int, int]] = []
    next_index = mesh.branch_count
    position = 0
    for axis, group in ((0, mesh.branch_x), (1, mesh.branch_y)):
        for layer, row, col in group:
            if axis == 0:
                open_plus = not (filled(layer, row + 1, col) and filled(layer, row + 1, col + 1))
                open_minus = not (filled(layer, row - 1, col) and filled(layer, row - 1, col + 1))
            else:
                open_plus = not (filled(layer, row, col + 1) and filled(layer, row + 1, col + 1))
                open_minus = not (filled(layer, row, col - 1) and filled(layer, row + 1, col - 1))
            strands = graded_widths(pitch, open_minus, open_plus, first_width_m, growth)
            if len(strands) > 1:
                for k, (centre, size) in enumerate(strands):
                    branch.append(position)
                    strand_index.append(position if k == 0 else next_index)
                    if k:
                        next_index += 1
                    offset.append(centre)
                    width.append(size)
                    axis_of.append(axis)
                    cell.append((layer, row, col))
            position += 1
    axis_array = np.asarray(axis_of, dtype=np.int64)
    cell_array = np.asarray(cell, dtype=np.int64).reshape(-1, 3)
    offset_array = np.asarray(offset, dtype=np.float64)
    node = np.zeros((offset_array.size, 3), dtype=np.int64)
    weight = np.zeros((offset_array.size, 3), dtype=np.float64)
    for axis, across, size in ((0, 1, rows), (1, 2, cols)):
        mine = axis_array == axis
        if np.any(mine):
            node[mine], weight[mine] = lagrange_across(offset_array[mine] / pitch, cell_array[mine, across], size)
    return EdgeStrands(
        base_count=mesh.branch_count,
        branch=np.asarray(branch, dtype=np.int64),
        strand_index=np.asarray(strand_index, dtype=np.int64),
        offset_m=offset_array,
        width_m=np.asarray(width, dtype=np.float64),
        axis=axis_array,
        cell=cell_array,
        node=node,
        weight=weight,
    )


class StrandedSheetMesh(SheetMesh):
    """A :class:`SheetMesh` whose edge branches are cut into strands.

    The branch vector is the mesh's own followed by the appended strands.
    ``scatter`` spreads every strand over its three grid positions and
    ``gather`` reads it back with the same weights; ``fold`` sums strand
    currents onto their branches (what a cell carries).
    ``inductance_correction`` is the precorrection ``C`` the solver adds to the
    grid's flux; ``near_exact`` the exact partial inductances of the pairs it
    covers, for the near preconditioner.
    """

    strands: EdgeStrands
    inductance_correction: sp.csr_matrix
    near_exact: sp.csr_matrix

    @classmethod
    def from_mesh(cls, mesh: SheetMesh, strands: EdgeStrands) -> "StrandedSheetMesh":
        if strands.base_count != mesh.branch_count:
            raise ValueError("the strands were planned for another mesh")
        stranded = object.__new__(cls)
        stranded.__dict__.update(mesh.__dict__)
        stranded.__dict__.pop("_positions", None)
        stranded.strands = strands
        stranded.inductance_correction = sp.csr_matrix((0, 0))
        stranded.near_exact = sp.csr_matrix((0, 0))
        rows, cols = mesh.shape
        layer, row, col = strands.cell[:, 0], strands.cell[:, 1], strands.cell[:, 2]
        across_row = np.where(strands.axis[:, None] == 0, strands.node, 0)
        across_col = np.where(strands.axis[:, None] == 1, strands.node, 0)
        stranded.__dict__["_strand_flat"] = (
            (layer[:, None] * rows + row[:, None] + across_row) * cols + col[:, None] + across_col
        )
        return stranded

    @property
    def base_branch_count(self) -> int:
        return self.strands.base_count

    @property
    def branch_count(self) -> int:  # type: ignore[override]
        return self.strands.base_count + self.strands.extra_count

    def incidence(self) -> sp.csr_matrix:
        base = SheetMesh.incidence(_BaseView(self))
        return sp.vstack([base, base[self.strands.extra_parent]], format="csr")

    def resistances(self) -> np.ndarray:
        base = SheetMesh.resistances(_BaseView(self))
        values = np.concatenate([base, base[self.strands.extra_parent]])
        pitch = float(self.pitch_m)  # type: ignore[arg-type]
        # A strand is the branch's bar narrowed to its own width.
        values[self.strands.strand_index] = base[self.strands.branch] * (pitch / self.strands.width_m)
        return values

    def fold(self, values: np.ndarray) -> np.ndarray:
        """Sum every strand onto its branch: a vector in the mesh's own branch order."""
        base = self.strands.base_count
        folded = np.array(values[:base], copy=True)
        np.add.at(folded, self.strands.extra_parent, values[base:])
        return folded

    def scatter(self, currents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        strands = self.strands
        base = np.array(currents[: strands.base_count], copy=True)
        base[strands.cut_branches] = 0.0
        grid_x, grid_y = SheetMesh.scatter(_BaseView(self), base)
        flat = self.__dict__["_strand_flat"]
        spread = strands.weight * currents[strands.strand_index][:, None]
        for axis, grid in ((0, grid_x), (1, grid_y)):
            mine = strands.axis == axis
            np.add.at(grid.reshape(-1), flat[mine].reshape(-1), spread[mine].reshape(-1))
        return grid_x, grid_y

    def gather(self, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        strands = self.strands
        base = SheetMesh.gather(_BaseView(self), grid_x, grid_y)
        values = np.concatenate([base, np.zeros(strands.extra_count)])
        flat = self.__dict__["_strand_flat"]
        read = np.empty(strands.branch.size)
        for axis, grid in ((0, grid_x), (1, grid_y)):
            mine = strands.axis == axis
            read[mine] = np.sum(strands.weight[mine] * grid.reshape(-1)[flat[mine]], axis=1)
        values[strands.strand_index] = read
        return values


class _BaseView:
    """The stranded mesh seen as the mesh it was cut from, for the base class's methods."""

    def __init__(self, mesh: StrandedSheetMesh) -> None:
        self._mesh = mesh

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mesh, name)

    @property
    def branch_count(self) -> int:
        return self._mesh.strands.base_count

    def _branch_positions(self) -> dict[str, Any]:
        return SheetMesh._branch_positions(self)  # type: ignore[arg-type]

    @property
    def __dict__(self) -> dict[str, Any]:  # type: ignore[override]
        return self._mesh.__dict__


def strand_correction(
    mesh: StrandedSheetMesh, operator: Any, radius_cells: int = EDGE_STRAND_RADIUS_CELLS
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """``(C, E)``: the precorrection and the exact partial inductances of the pairs that involve a strand.

    ``operator`` is the uniform convolution operator; its tables give the
    grid's values.  Like-directed branches only (in-plane branches of the two
    axes do not couple), and only within one conductor: the filaments of a cut
    layer are one conductor, other conductors see the strands through the grid.
    """
    strands = mesh.strands
    tables = operator._tables  # same package: the grid's partial inductances by layer pair and offset
    padded_rows, padded_cols = operator.padded
    pitch = float(mesh.pitch_m)  # type: ignore[arg-type]
    layers, rows, cols = mesh.occupancy.shape
    conductor = [_conductor_of_layer(layer.name) for layer in mesh.stackup.layers]
    same_conductor = {a: [b for b in range(layers) if conductor[b] == conductor[a]] for a in range(layers)}
    z = np.asarray([layer.z_m for layer in mesh.stackup.layers], dtype=np.float64)
    thickness = np.asarray([layer.thickness_m for layer in mesh.stackup.layers], dtype=np.float64)

    entries_i: list[np.ndarray] = []
    entries_j: list[np.ndarray] = []
    corrections: list[np.ndarray] = []
    exacts: list[np.ndarray] = []
    position = 0
    for axis, group in ((0, mesh.branch_x), (1, mesh.branch_y)):
        count = len(group)
        if count == 0:
            continue
        name = "x" if axis == 0 else "y"
        cells = np.asarray(group, dtype=np.int64).reshape(-1, 3)
        # Every branch of this axis as padded strand lists, (count, width):
        # an uncut branch is one strand of its own at the centre.
        local = strands.branch - position
        mine = (local >= 0) & (local < count)
        per_branch = np.bincount(local[mine], minlength=count)
        width = max(1, int(per_branch.max(initial=0)))
        index = np.full((count, width), -1, dtype=np.int64)
        offset = np.zeros((count, width))
        size = np.zeros((count, width))
        node = np.zeros((count, width, 3), dtype=np.int64)
        weight = np.zeros((count, width, 3))
        index[:, 0] = position + np.arange(count)
        size[:, 0] = pitch
        weight[:, 0, 0] = 1.0
        cut = per_branch > 0
        if np.any(cut):
            order = np.argsort(local[mine], kind="stable")
            rows_ = local[mine][order]
            slot = np.arange(order.size) - np.repeat(np.cumsum(per_branch) - per_branch, per_branch)[: order.size]
            picked = np.flatnonzero(mine)[order]
            index[rows_, slot] = strands.strand_index[picked]
            offset[rows_, slot] = strands.offset_m[picked]
            size[rows_, slot] = strands.width_m[picked]
            node[rows_, slot] = strands.node[picked]
            weight[rows_, slot] = strands.weight[picked]
        lookup = np.full((layers, rows, cols), -1, dtype=np.int64)
        lookup[cells[:, 0], cells[:, 1], cells[:, 2]] = np.arange(count)
        sources = np.flatnonzero(cut)
        span = range(-radius_cells, radius_cells + 1)
        for d_along in span:
            for d_across in span:
                dr, dc = (d_across, d_along) if axis == 0 else (d_along, d_across)
                for layer_a in np.unique(cells[sources, 0]):
                    at_a = sources[cells[sources, 0] == layer_a]
                    for layer_b in same_conductor[int(layer_a)]:
                        row = cells[at_a, 1] + dr
                        col = cells[at_a, 2] + dc
                        inside = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
                        src = at_a[inside]
                        target = lookup[layer_b, row[inside], col[inside]]
                        hit = target >= 0
                        src, target = src[hit], target[hit]
                        if src.size == 0:
                            continue
                        table = tables[(name, min(int(layer_a), layer_b), max(int(layer_a), layer_b))]
                        shape = (src.size, width, width)
                        valid = (size[src][:, :, None] > 0) & (size[target][:, None, :] > 0)
                        sel = np.nonzero(valid)
                        a, k, l = sel
                        s_rows, t_rows = src[a], target[a]
                        across = d_across * pitch + offset[t_rows, l] - offset[s_rows, k]
                        n = a.size
                        exact = closed_form_mutual_inductance_arrays(
                            (np.full(n, pitch), size[s_rows, k], np.full(n, thickness[int(layer_a)])),
                            (np.full(n, pitch), size[t_rows, l], np.full(n, thickness[layer_b])),
                            (np.full(n, d_along * pitch), across, np.full(n, z[layer_b] - z[int(layer_a)])),
                        )
                        # The grid's value of the two projected strands.
                        node_s, node_t = node[s_rows, k], node[t_rows, l]
                        weight_s, weight_t = weight[s_rows, k], weight[t_rows, l]
                        shift = d_across + node_t[:, None, :] - node_s[:, :, None]
                        if axis == 0:
                            values = table[shift % padded_rows, d_along % padded_cols]
                        else:
                            values = table[d_along % padded_rows, shift % padded_cols]
                        grid = np.einsum("pm,pmn,pn->p", weight_s, values, weight_t)
                        i, j = index[s_rows, k], index[t_rows, l]
                        entries_i.append(i)
                        entries_j.append(j)
                        corrections.append(exact - grid)
                        exacts.append(exact)
                        # An uncut target writes its row here; a cut one in its own pass.
                        mirror = ~cut[t_rows]
                        if np.any(mirror):
                            entries_i.append(j[mirror])
                            entries_j.append(i[mirror])
                            corrections.append((exact - grid)[mirror])
                            exacts.append(exact[mirror])
                        del shape
        position += count
    n = mesh.branch_count
    if not corrections:
        empty = sp.csr_matrix((n, n))
        return empty, empty
    i = np.concatenate(entries_i)
    j = np.concatenate(entries_j)
    correction = sp.csr_matrix((np.concatenate(corrections), (i, j)), shape=(n, n))
    exact = sp.csr_matrix((np.concatenate(exacts), (i, j)), shape=(n, n))
    # M(s, j) and M(j, s) come from the closed form with the bars swapped and
    # differ in the last digits the cancellation leaves (2e-9 of the largest
    # entry on the strip pair); the matrices are symmetric by construction.
    return ((correction + correction.T) * 0.5).tocsr(), ((exact + exact.T) * 0.5).tocsr()


def stranded_mesh(mesh: SheetMesh, operator: Any, first_width_m: float, growth: float = EDGE_STRAND_GROWTH,
                  radius_cells: int = EDGE_STRAND_RADIUS_CELLS) -> StrandedSheetMesh:
    """The mesh with its edge branches cut and the correction built for ``operator``."""
    stranded = StrandedSheetMesh.from_mesh(mesh, plan_edge_strands(mesh, first_width_m, growth))
    stranded.inductance_correction, stranded.near_exact = strand_correction(stranded, operator, radius_cells)
    return stranded
