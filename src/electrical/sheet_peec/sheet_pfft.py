"""Sheet PEEC inductance on a graded tensor grid by precorrected FFT (pFFT).

The uniform-grid operator (:mod:`.sheet_operator`) is a circular convolution:
the coupling of two like-directed branches depends on their offset alone, so
one table per layer pair serves every branch.  On a graded grid the branches
are bars of different lengths and widths at irregular positions and no
convolution exists.  The precorrected FFT keeps the transform anyway:

1. every branch's current-length product ``I_b l_b`` is projected onto a
   uniform *projection grid* of pitch ``h`` (the coarse pitch) through a
   tensor-product Lagrange stencil of ``order + 1`` points per axis;
2. the grid sources are convolved with the centre-to-centre kernel
   ``mu0 / (4 pi r)`` between grid points of the two layers by FFT, one table
   per layer separation;
3. the vector potential is interpolated back to the branch centres with the
   same stencil and multiplied by ``l_b``: that is the far-field flux;
4. for pairs whose centres are within ``near_radius`` grid cells the grid
   approximation is subtracted and the exact partial mutual inductance of
   the two bars (Hoer and Love closed form, per-pair dimensions) is added:
   a sparse *precorrection* built once.

The result is exact for near pairs and carries the interpolation error of a
smooth ``1/r`` on far pairs, which falls as ``(h / r)^(order + 1)``.  Cost per
application: two sparse products and one FFT pair per layer pair and axis on
the coarse grid, independent of how fine the graded cells are.  Vertical
branches (vias, filament links) get the same treatment on their level pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp

from electrical.matrix_free_mpir_fem.grid import TensorGrid

from .sheet_inductance import _MU0_OVER_4PI, closed_form_mutual_inductance_arrays
from .sheet_operator import SheetStackup
from .sheet_peec import SheetMesh


def lagrange_stencil(coordinate: np.ndarray, origin: float, pitch: float, count: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Grid indices ``(n, order + 1)`` and Lagrange weights for interpolating at ``coordinate``.

    The stencil is the ``order + 1`` grid points around the coordinate,
    shifted inward at the grid ends; the weights reproduce polynomials up to
    ``order`` exactly, so the projected sources match the branch's moments to
    that order.
    """

    x = (np.asarray(coordinate, dtype=np.float64) - origin) / pitch
    first = np.floor(x).astype(np.int64) - (order - 1) // 2
    first = np.clip(first, 0, count - 1 - order)
    nodes = first[:, None] + np.arange(order + 1)[None, :]
    weights = np.ones(nodes.shape, dtype=np.float64)
    for j in range(order + 1):
        for k in range(order + 1):
            if k != j:
                weights[:, j] *= (x[:, None][:, 0] - nodes[:, k]) / (nodes[:, j] - nodes[:, k])
    return nodes, weights


@dataclass(frozen=True)
class ProjectionGrid:
    """The uniform grid the sources are projected onto, covering the board with a stencil margin."""

    origin_x_m: float
    origin_y_m: float
    pitch_m: float
    nodes_x: int
    nodes_y: int

    @property
    def padded(self) -> tuple[int, int]:
        return 2 * self.nodes_y, 2 * self.nodes_x

    def kernel_table(self, separation_m: float) -> np.ndarray:
        """``mu0 / (4 pi r)`` between grid points at every wrapped offset, zero at ``r = 0``."""

        rows, cols = self.padded
        dy = np.fft.fftfreq(rows, d=1.0 / rows) * self.pitch_m
        dx = np.fft.fftfreq(cols, d=1.0 / cols) * self.pitch_m
        r = np.sqrt(dx[None, :] ** 2 + dy[:, None] ** 2 + float(separation_m) ** 2)
        with np.errstate(divide="ignore"):
            table = np.where(r > 0.0, _MU0_OVER_4PI / np.where(r > 0.0, r, 1.0), 0.0)
        return table


_GAUSS_3 = (np.array([-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)]) / 2.0, np.array([5.0, 8.0, 5.0]) / 18.0)
_GAUSS_2 = (np.array([-1.0, 1.0]) / (2.0 * math.sqrt(3.0)), np.array([0.5, 0.5]))


def _projection_matrix(
    centre_x: np.ndarray,
    centre_y: np.ndarray,
    extent_x: np.ndarray,
    extent_y: np.ndarray,
    grid: ProjectionGrid,
    order: int,
) -> sp.csr_matrix:
    """Sparse ``(branches, nodes)`` projection of each bar onto the grid.

    A bar is sampled at Gauss points over its in-plane extent (three along
    the longer side, two across the shorter), each sample interpolated with
    the Lagrange stencil; the same matrix interpolates the potential back and
    averages it over the bar.  Carrying the extent keeps the far field of a
    bar longer than a grid cell right, so the near radius only has to cover
    the stencil's own interpolation range.
    """

    cx, cy = centre_x.reshape(-1), centre_y.reshape(-1)
    ex, ey = np.broadcast_to(extent_x, centre_x.shape).reshape(-1), np.broadcast_to(extent_y, centre_y.shape).reshape(-1)
    n = cx.size
    along_x = np.mean(ex) >= np.mean(ey)
    (px, wx_q), (py, wy_q) = (_GAUSS_3, _GAUSS_2) if along_x else (_GAUSS_2, _GAUSS_3)
    blocks = []
    for ox, qx in zip(px, wx_q):
        for oy, qy in zip(py, wy_q):
            ix, wx = lagrange_stencil(cx + ox * ex, grid.origin_x_m, grid.pitch_m, grid.nodes_x, order)
            iy, wy = lagrange_stencil(cy + oy * ey, grid.origin_y_m, grid.pitch_m, grid.nodes_y, order)
            rows = np.repeat(np.arange(n), (order + 1) ** 2)
            cols = (iy[:, :, None] * grid.nodes_x + ix[:, None, :]).reshape(-1)
            data = (qx * qy) * (wy[:, :, None] * wx[:, None, :]).reshape(-1)
            blocks.append(sp.csr_matrix((data, (rows, cols)), shape=(n, grid.nodes_x * grid.nodes_y)))
    total = blocks[0]
    for block in blocks[1:]:
        total = total + block
    total.sum_duplicates()
    return total.tocsr()


class PfftSheetInductanceOperator:
    """Partial inductance of a graded sheet mesh, applied by precorrected FFT.

    Same interface as :class:`~.sheet_operator.SheetInductanceOperator`:
    ``apply(currents_x, currents_y[, currents_z])`` on ``(layers, rows, cols)``
    grids indexed like the branches (``[l, r, c]`` of the x array is the
    branch from cell ``(r, c)`` to ``(r, c + 1)``), plus
    ``branch_self_inductance(mesh)`` for the solver's preconditioner.
    """

    def __init__(
        self,
        mesh: SheetMesh,
        *,
        grid_pitch_m: float | None = None,
        order: int = 3,
        near_radius_cells: int = 3,
    ) -> None:
        if order < 1 or order > 5:
            raise ValueError("order must lie in [1, 5]")
        if near_radius_cells < 1:
            raise ValueError("near_radius_cells must be at least one")
        tensor = mesh.grid
        assert tensor is not None
        self.mesh_grid = tensor
        self.shape = mesh.shape
        self.stackup: SheetStackup = mesh.stackup
        self.order = int(order)
        self.near_radius_cells = int(near_radius_cells)
        # The projection grid defaults to twice the finest cell: fine enough that
        # a bar's Gauss samples resolve its extent, coarse enough that the FFT
        # stays small; the near radius is measured in its cells.
        pitch = float(grid_pitch_m) if grid_pitch_m is not None else 2.0 * float(min(tensor.pitch_x_m.min(), tensor.pitch_y_m.min()))
        if pitch <= 0.0:
            raise ValueError("grid_pitch_m must be positive")
        width = float(tensor.x_edges_m[-1] - tensor.x_edges_m[0])
        height = float(tensor.y_edges_m[-1] - tensor.y_edges_m[0])
        margin = order + 1
        self.grid = ProjectionGrid(
            origin_x_m=-margin * pitch,
            origin_y_m=-margin * pitch,
            pitch_m=pitch,
            nodes_x=int(math.ceil(width / pitch)) + 2 * margin + 1,
            nodes_y=int(math.ceil(height / pitch)) + 2 * margin + 1,
        )
        self.vertical_levels = tuple((int(a), int(b)) for a, b in mesh.vertical_levels)

        # In-plane branches: geometry, projection and precorrection per axis.
        self._geometry: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._projection: dict[str, sp.csr_matrix] = {}
        self._correction: dict[str, sp.csr_matrix] = {}
        self._self: dict[str, np.ndarray] = {}
        separations = {}
        layers = self.stackup.layers
        for first in range(len(layers)):
            for second in range(len(layers)):
                separations[(first, second)] = abs(layers[second].z_m - layers[first].z_m)
        distinct = sorted(set(round(v, 15) for v in separations.values()))
        self._spectra = {sep: np.fft.rfft2(self.grid.kernel_table(sep)) for sep in distinct}
        self._separation = {key: round(v, 15) for key, v in separations.items()}
        for axis in ("x", "y"):
            cx, cy, length, width_ = mesh.branch_geometry(axis)
            self._geometry[axis] = (cx, cy, length, width_)
            extent_x, extent_y = (length, width_) if axis == "x" else (width_, length)
            self._projection[axis] = _projection_matrix(cx, cy, extent_x, extent_y, self.grid, self.order)
            self._correction[axis], self._self[axis] = self._build_correction(axis)

        # Vertical branches: one "layer" per level, centred at the level's mid-plane.
        self._vertical_geometry: list[tuple[float, float]] = [
            (abs(layers[b].z_m - layers[a].z_m), 0.5 * (layers[a].z_m + layers[b].z_m)) for a, b in self.vertical_levels
        ]
        self._projection_z: sp.csr_matrix | None = None
        self._correction_z: sp.csr_matrix | None = None
        self._self_z: np.ndarray | None = None
        self._spectra_z: dict[float, np.ndarray] = {}
        if self.vertical_levels:
            xc = 0.5 * (tensor.x_edges_m[:-1] + tensor.x_edges_m[1:]) - tensor.x_edges_m[0]
            yc = 0.5 * (tensor.y_edges_m[:-1] + tensor.y_edges_m[1:]) - tensor.y_edges_m[0]
            centre_x = np.broadcast_to(xc[None, :], self.shape)
            centre_y = np.broadcast_to(yc[:, None], self.shape)
            self._projection_z = _projection_matrix(
                centre_x, centre_y, tensor.pitch_x_m[None, :], tensor.pitch_y_m[:, None], self.grid, self.order
            )
            for a in range(len(self.vertical_levels)):
                for b in range(len(self.vertical_levels)):
                    sep = round(abs(self._vertical_geometry[b][1] - self._vertical_geometry[a][1]), 15)
                    if sep not in self._spectra_z:
                        self._spectra_z[sep] = np.fft.rfft2(self.grid.kernel_table(sep))
            self._correction_z, self._self_z = self._build_vertical_correction(centre_x, centre_y)

    # ------------------------------------------------------------- construction
    def _near_pairs(self, coords: np.ndarray, half_extent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Index pairs of a sorted 1D coordinate array within the near radius plus both half extents."""

        radius = self.near_radius_cells * self.grid.pitch_m * (1.0 + 1.0e-9)
        reach = radius + half_extent + float(np.max(half_extent))
        lo = np.searchsorted(coords, coords - reach, side="left")
        hi = np.searchsorted(coords, coords + reach, side="right")
        counts = hi - lo
        i = np.repeat(np.arange(coords.size), counts)
        j = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)]) if coords.size else np.zeros(0, dtype=np.int64)
        keep = np.abs(coords[j] - coords[i]) <= radius + half_extent[i] + half_extent[j]
        return i[keep], j[keep]

    @staticmethod
    def _dense_stencils(projection: sp.csr_matrix, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Node indices and weights of the listed rows as padded ``(n, width)`` arrays."""

        starts = projection.indptr[idx]
        counts = projection.indptr[idx + 1] - starts
        width = int(counts.max()) if counts.size else 0
        offsets = np.arange(width)[None, :]
        mask = offsets < counts[:, None]
        positions = (starts[:, None] + offsets)[mask]
        nodes = np.zeros((idx.size, width), dtype=np.int64)
        weights = np.zeros((idx.size, width))
        nodes[mask] = projection.indices[positions]
        weights[mask] = projection.data[positions]
        return nodes, weights

    def _grid_pair_coupling(self, projection: sp.csr_matrix, i: np.ndarray, j: np.ndarray, separation: float) -> np.ndarray:
        """``P_i K P_j^T`` for the listed branch pairs: what the FFT path credits them with."""

        table = self.grid.kernel_table(separation)
        rows, cols = self.grid.padded
        p = projection.tocsr()
        out = np.zeros(i.size)
        width = int(np.max(np.diff(p.indptr))) if p.indptr.size > 1 else 0
        chunk = max(1, int(4.0e6 // max(1, width * width)))
        for start in range(0, i.size, chunk):
            si, sj = i[start : start + chunk], j[start : start + chunk]
            ni, wi = self._dense_stencils(p, si)
            nj, wj = self._dense_stencils(p, sj)
            ny_i, nx_i = np.divmod(ni, self.grid.nodes_x)
            ny_j, nx_j = np.divmod(nj, self.grid.nodes_x)
            dy = (ny_j[:, None, :] - ny_i[:, :, None]) % rows
            dx = (nx_j[:, None, :] - nx_i[:, :, None]) % cols
            out[start : start + chunk] = np.einsum("pa,pab,pb->p", wi, table[dy, dx], wj, optimize=True)
        return out

    @staticmethod
    def _unique_rows(columns: Sequence[np.ndarray], quantum: float) -> tuple[np.ndarray, np.ndarray]:
        """Indices of one representative per distinct quantised row, and each row's representative."""

        keys = np.stack([np.round(np.asarray(c, dtype=np.float64) / quantum).astype(np.int64) for c in columns], axis=1)
        _unique, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        return first, inverse.reshape(-1)

    def _pair_structure(self, cx: np.ndarray, cy: np.ndarray, half_x: np.ndarray, half_y: np.ndarray) -> dict[str, np.ndarray]:
        """Near pairs of one branch family and the representatives of their repeated configurations."""

        n_rows, n_cols = cx.shape
        ci, cj = self._near_pairs(cx[0], half_x[0])
        ri, rj = self._near_pairs(cy[:, 0], half_y[:, 0])
        bi = (ri[:, None] * n_cols + ci[None, :]).reshape(-1)
        bj = (rj[:, None] * n_cols + cj[None, :]).reshape(-1)
        fx, fy = cx.reshape(-1), cy.reshape(-1)
        hx, hy = half_x.reshape(-1), half_y.reshape(-1)
        dx, dy = fx[bj] - fx[bi], fy[bj] - fy[bi]
        quantum = 1.0e-12
        # The exact term depends on the two bars and their offset; the grid term
        # also on where the first bar sits relative to the grid lines.
        exact_first, exact_inverse = self._unique_rows((hx[bi], hy[bi], hx[bj], hy[bj], dx, dy), quantum)
        frac_x = np.mod(fx - self.grid.origin_x_m, self.grid.pitch_m)
        frac_y = np.mod(fy - self.grid.origin_y_m, self.grid.pitch_m)
        grid_first, grid_inverse = self._unique_rows((frac_x[bi], frac_y[bi], hx[bi], hy[bi], hx[bj], hy[bj], dx, dy), quantum)
        return {"bi": bi, "bj": bj, "exact_first": exact_first, "exact_inverse": exact_inverse, "grid_first": grid_first, "grid_inverse": grid_inverse}

    def _build_correction(self, axis: str) -> tuple[sp.csr_matrix, np.ndarray]:
        cx, cy, length, width = self._geometry[axis]
        n_rows, n_cols = cx.shape
        layers = self.stackup.layers
        count = n_rows * n_cols
        if count == 0:
            empty = sp.csr_matrix((len(layers) * count, len(layers) * count))
            return empty, np.zeros((len(layers), 0))
        half_x, half_y = (length / 2.0, width / 2.0) if axis == "x" else (width / 2.0, length / 2.0)
        pairs = self._pair_structure(cx, cy, half_x, half_y)
        bi, bj = pairs["bi"], pairs["bj"]
        flat_cx, flat_cy, flat_l, flat_w = (a.reshape(-1) for a in (cx, cy, length, width))
        ef, ei = pairs["exact_first"], pairs["exact_inverse"]
        gf, gi = pairs["grid_first"], pairs["grid_inverse"]
        blocks = []
        self_values = np.zeros((len(layers), count))
        diagonal_pairs = bi == bj
        # Both terms depend on the layer pair only through the separation (and
        # the exact one through the two thicknesses), so each distinct
        # combination is evaluated once and reused across layer pairs.
        exact_cache: dict[tuple[float, float, float], np.ndarray] = {}
        grid_cache: dict[float, np.ndarray] = {}
        ui, uj = bi[ef], bj[ef]
        gi_, gj_ = bi[gf], bj[gf]
        for a, la in enumerate(layers):
            for b, lb in enumerate(layers):
                sep = self._separation[(a, b)]
                dz = lb.z_m - la.z_m
                thick = (min(la.thickness_m, lb.thickness_m), max(la.thickness_m, lb.thickness_m))
                key = (sep, *thick)
                if key not in exact_cache:
                    if axis == "x":
                        offset = (flat_cx[uj] - flat_cx[ui], flat_cy[uj] - flat_cy[ui], np.full(ui.size, abs(dz)))
                    else:
                        offset = (flat_cy[uj] - flat_cy[ui], flat_cx[uj] - flat_cx[ui], np.full(ui.size, abs(dz)))
                    exact_cache[key] = closed_form_mutual_inductance_arrays(
                        (flat_l[ui], flat_w[ui], np.full(ui.size, thick[0])),
                        (flat_l[uj], flat_w[uj], np.full(uj.size, thick[1])),
                        offset,
                        check_precision=False,
                    )
                if sep not in grid_cache:
                    grid_cache[sep] = flat_l[gi_] * flat_l[gj_] * self._grid_pair_coupling(self._projection[axis], gi_, gj_, sep)
                exact = exact_cache[key][ei]
                approximate = grid_cache[sep][gi]
                blocks.append(sp.csr_matrix((exact - approximate, (bi, bj)), shape=(count, count)))
                if a == b:
                    self_values[a, bi[diagonal_pairs]] = exact[diagonal_pairs]
        correction = sp.bmat([[blocks[a * len(layers) + b] for b in range(len(layers))] for a in range(len(layers))], format="csr")
        return correction, self_values

    def _build_vertical_correction(self, centre_x: np.ndarray, centre_y: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
        assert self._projection_z is not None
        tensor = self.mesh_grid
        n_rows, n_cols = self.shape
        count = n_rows * n_cols
        hx = np.broadcast_to(tensor.pitch_x_m[None, :], self.shape)
        hy = np.broadcast_to(tensor.pitch_y_m[:, None], self.shape)
        pairs = self._pair_structure(np.ascontiguousarray(centre_x), np.ascontiguousarray(centre_y), hx / 2.0, hy / 2.0)
        bi, bj = pairs["bi"], pairs["bj"]
        ef, ei = pairs["exact_first"], pairs["exact_inverse"]
        gf, gi = pairs["grid_first"], pairs["grid_inverse"]
        fhx, fhy = hx.reshape(-1), hy.reshape(-1)
        fx, fy = centre_x.reshape(-1), centre_y.reshape(-1)
        levels = len(self.vertical_levels)
        blocks = []
        self_values = np.zeros((levels, count))
        diagonal_pairs = bi == bj
        grid_cache: dict[float, np.ndarray] = {}
        ui, uj = bi[ef], bj[ef]
        gi_, gj_ = bi[gf], bj[gf]
        for a in range(levels):
            span_a, z_a = self._vertical_geometry[a]
            for b in range(levels):
                span_b, z_b = self._vertical_geometry[b]
                sep = round(abs(z_b - z_a), 15)
                exact_unique = closed_form_mutual_inductance_arrays(
                    (np.full(ui.size, span_a), fhx[ui], fhy[ui]),
                    (np.full(uj.size, span_b), fhx[uj], fhy[uj]),
                    (np.full(ui.size, z_b - z_a), fx[uj] - fx[ui], fy[uj] - fy[ui]),
                    check_precision=False,
                )
                exact = exact_unique[ei]
                if sep not in grid_cache:
                    grid_cache[sep] = self._grid_pair_coupling(self._projection_z, gi_, gj_, sep)
                approximate = (span_a * span_b * grid_cache[sep])[gi]
                blocks.append(sp.csr_matrix((exact - approximate, (bi, bj)), shape=(count, count)))
                if a == b:
                    self_values[a, bi[diagonal_pairs]] = exact[diagonal_pairs]
        correction = sp.bmat([[blocks[a * levels + b] for b in range(levels)] for a in range(levels)], format="csr")
        return correction, self_values

    # -------------------------------------------------------------------- apply
    @property
    def kernel_bytes(self) -> int:
        spectra = sum(v.nbytes for v in self._spectra.values()) + sum(v.nbytes for v in self._spectra_z.values())
        sparse = sum(m.data.nbytes + m.indices.nbytes for m in self._projection.values())
        sparse += sum(m.data.nbytes + m.indices.nbytes for m in self._correction.values())
        return int(spectra + sparse)

    def _far(self, currents: np.ndarray, axis: str, length: np.ndarray, projection: sp.csr_matrix, spectra_of: Any) -> np.ndarray:
        layer_count = currents.shape[0]
        n = projection.shape[0]
        if n == 0:
            return np.zeros((layer_count, 0))
        grid_shape = (self.grid.nodes_y, self.grid.nodes_x)
        sources = [projection.T @ (currents[l].reshape(-1) * length.reshape(-1)) for l in range(layer_count)]
        spectra = [np.fft.rfft2(src.reshape(grid_shape), s=self.grid.padded) for src in sources]
        out = np.zeros((layer_count, n), dtype=np.float64)
        for target in range(layer_count):
            accumulated = np.zeros_like(spectra[0])
            for source in range(layer_count):
                accumulated += spectra_of(target, source) * spectra[source]
            potential = np.fft.irfft2(accumulated, s=self.grid.padded)[: grid_shape[0], : grid_shape[1]]
            out[target] = length.reshape(-1) * (projection @ potential.reshape(-1))
        return out

    def _apply_axis(self, currents: np.ndarray, axis: str) -> np.ndarray:
        cx, cy, length, _w = self._geometry[axis]
        expected = (len(self.stackup),) + cx.shape
        if currents.shape != expected:
            raise ValueError(f"currents_{axis} must have shape {expected}, got {currents.shape}")
        far = self._far(currents, axis, length, self._projection[axis], lambda t, s_: self._spectra[self._separation[(t, s_)]])
        near = self._correction[axis] @ currents.reshape(-1)
        return (far.reshape(-1) + near).reshape(expected)

    def apply(
        self,
        currents_x: np.ndarray,
        currents_y: np.ndarray,
        currents_z: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flux linkage of every branch for these currents, in webers.

        The current grids carry the operator's full ``(layers, rows, cols)``
        layout; the last column of the x grid and the last row of the y grid
        hold no branch and must be zero.
        """

        rows, cols = self.shape
        layer_count = len(self.stackup)
        for name, array in (("currents_x", currents_x), ("currents_y", currents_y)):
            if array.shape != (layer_count, rows, cols):
                raise ValueError(f"{name} must have shape {(layer_count, rows, cols)}, got {array.shape}")
        flux_x = np.zeros_like(currents_x, dtype=np.float64)
        flux_y = np.zeros_like(currents_y, dtype=np.float64)
        flux_x[:, :, :-1] = self._apply_axis(np.asarray(currents_x[:, :, :-1], dtype=np.float64), "x")
        flux_y[:, :-1, :] = self._apply_axis(np.asarray(currents_y[:, :-1, :], dtype=np.float64), "y")
        if currents_z is None:
            return flux_x, flux_y
        levels = len(self.vertical_levels)
        if currents_z.shape != (levels, rows, cols):
            raise ValueError(f"currents_z must have shape {(levels, rows, cols)}, got {currents_z.shape}")
        if not levels:
            return flux_x, flux_y, np.zeros_like(currents_z)
        assert self._projection_z is not None and self._correction_z is not None
        spans = np.asarray([span for span, _ in self._vertical_geometry])
        out = np.zeros((levels, rows * cols))
        grid_shape = (self.grid.nodes_y, self.grid.nodes_x)
        spectra = [
            np.fft.rfft2((self._projection_z.T @ (currents_z[l].reshape(-1) * spans[l])).reshape(grid_shape), s=self.grid.padded)
            for l in range(levels)
        ]
        for target in range(levels):
            accumulated = np.zeros_like(spectra[0])
            for source in range(levels):
                sep = round(abs(self._vertical_geometry[source][1] - self._vertical_geometry[target][1]), 15)
                accumulated += self._spectra_z[sep] * spectra[source]
            potential = np.fft.irfft2(accumulated, s=self.grid.padded)[: grid_shape[0], : grid_shape[1]]
            out[target] = spans[target] * (self._projection_z @ potential.reshape(-1))
        flux_z = out.reshape(-1) + self._correction_z @ np.asarray(currents_z, dtype=np.float64).reshape(-1)
        return flux_x, flux_y, flux_z.reshape(levels, rows, cols)

    def branch_self_inductance(self, mesh: SheetMesh) -> np.ndarray:
        """Own partial inductance of every in-plane branch, in the mesh's branch order."""

        values = []
        for axis, group in (("x", mesh.branch_x), ("y", mesh.branch_y)):
            n_cols = self._geometry[axis][0].shape[1]
            table = self._self[axis]
            values.extend(float(table[layer, row * n_cols + col]) for layer, row, col in group)
        return np.asarray(values, dtype=np.float64)

    def vertical_self_inductance(self, mesh: SheetMesh) -> np.ndarray:
        """Own partial inductance of every vertical branch, in the mesh's via order."""

        if self._self_z is None:
            return np.zeros(len(mesh.via_branches))
        index_of = {key: position for position, key in enumerate(self.vertical_levels)}
        rows, cols = self.shape
        return np.asarray(
            [self._self_z[index_of[(via.lower_layer, via.upper_layer)], via.row * cols + via.col] for via in mesh.via_branches]
        )

    def dense_matrix(self, axis: str = "x") -> np.ndarray:
        """Assemble the in-plane operator of one axis densely, for checks on small meshes."""

        if axis not in {"x", "y"}:
            raise ValueError("axis must be 'x' or 'y'")
        rows, cols = self.shape
        count = len(self.stackup) * rows * cols
        matrix = np.zeros((count, count))
        basis_x = np.zeros((len(self.stackup), rows, cols))
        basis_y = np.zeros_like(basis_x)
        target = basis_x if axis == "x" else basis_y
        for column in range(count):
            l, r, c = np.unravel_index(column, target.shape)
            if (axis == "x" and c == cols - 1) or (axis == "y" and r == rows - 1):
                continue
            target.flat[column] = 1.0
            out_x, out_y = self.apply(basis_x, basis_y)
            matrix[:, column] = (out_x if axis == "x" else out_y).reshape(-1)
            target.flat[column] = 0.0
        return matrix


__all__ = ["PfftSheetInductanceOperator", "ProjectionGrid", "lagrange_stencil"]
