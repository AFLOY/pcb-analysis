"""The hybrid sheet solve without dense matrices: grid FFT far field, precorrected near field.

Every element's RT0 current is sampled at its quadrature points; the points'
x and y currents are spread onto a uniform grid of pitch ``grid`` by
quadratic Lagrange weights (3 x 3 nodes each), convolved there with the
kernel of each sheet pair by FFT, and gathered back the same way (the
precorrected FFT of Phillips & White, 1997).  Pairs of elements close enough
for that to be inaccurate get the difference between their exact moments
(:func:`.hybrid_rt0.near_moments`, thickness-resolved) and what the grid gave
them, as a sparse correction; the same exact near moments, plus the
resistance, are the near impedance that preconditions GMRES on the
saddle-point system, factored once.

Links (resistive joints between the elements of two sheets: the filaments
of a layer, the vias) are dofs of their own here, so the near system is the
plain saddle point ``[[Z_near, -D], [D^T, 0]]``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from .hybrid_rt0 import MU0, RT0Space, Sheet, element_points, near_moments_over, thickness_gap, thickness_nodes

KAPPA = MU0 / (4.0 * math.pi)


def _native_sheet(name: str):
    """The native sheet module when it has ``name``, else None."""
    from electrical import backend as _backend

    core = _backend.core()
    sheet = getattr(core, "sheet", None) if core is not None else None
    return sheet if sheet is not None and hasattr(sheet, name) else None


def _near_pairs(ca: np.ndarray, da: np.ndarray, cb: np.ndarray, db: np.ndarray, factor: float, floor: float):
    """Element pairs (e, f) with |c_e - c_f| < max(factor max(d_e, d_f), floor), ordered by e then f."""
    sheet = _native_sheet("rt0_near_pairs")
    if sheet is not None:
        from electrical.threads import thread_budget

        e, f = sheet.rt0_near_pairs(np.ascontiguousarray(ca), np.ascontiguousarray(da), np.ascontiguousarray(cb),
                                    np.ascontiguousarray(db), float(factor), float(floor), thread_budget())
        return np.asarray(e, dtype=np.int64), np.asarray(f, dtype=np.int64)
    from scipy.spatial import cKDTree

    radius = max(factor * max(float(da.max()), float(db.max())), floor)
    near = cKDTree(ca).sparse_distance_matrix(cKDTree(cb), radius, output_type="ndarray")
    e, f = near["i"].astype(np.int64), near["j"].astype(np.int64)
    reach = np.maximum(factor * np.maximum(da[e], db[f]), floor)
    keep = near["v"] < reach
    order = np.lexsort((f[keep], e[keep]))
    return e[keep][order], f[keep][order]


def _point_phi(space: RT0Space):
    """Far quadrature points (P, 2) and Phi (P, 3E): each point's weighted [1, r - c] in its element's columns."""
    pts, w = element_points(space.vertices, space.corners, space.kind, order="far")
    count, q = w.shape
    flat = pts.reshape(-1, 2)
    centred = (pts - space.centroid[:, None, :]).reshape(-1, 2)
    phi = np.concatenate([np.ones((len(flat), 1)), centred], axis=1) * w.reshape(-1)[:, None]
    owner = np.repeat(np.arange(count), q)
    rows = np.repeat(np.arange(len(flat)), 3)
    cols = (3 * owner[:, None] + np.arange(3)[None, :]).ravel()
    return flat, sp.csr_matrix((phi.ravel(), (rows, cols)), shape=(len(flat), 3 * count))


@dataclass
class Grid:
    origin: np.ndarray
    pitch: float
    shape: tuple[int, int]

    @classmethod
    def around(cls, points: np.ndarray, pitch: float) -> "Grid":
        lo = points.min(axis=0) - 2 * pitch
        hi = points.max(axis=0) + 2 * pitch
        shape = tuple(int(math.ceil(v)) + 1 for v in (hi - lo) / pitch)
        return cls(lo, pitch, shape)

    def stencil(self, points: np.ndarray) -> sp.csr_matrix:
        """(P, nodes): quadratic Lagrange weights of each point on its nearest 3 x 3 nodes."""
        rel = (points - self.origin) / self.pitch
        centre = np.rint(rel).astype(np.int64)
        u = rel - centre  # in [-0.5, 0.5]

        def weights(t):
            return np.stack([0.5 * t * (t - 1.0), 1.0 - t * t, 0.5 * t * (t + 1.0)], axis=1)

        wx, wy = weights(u[:, 0]), weights(u[:, 1])
        nx, ny = self.shape
        rows, cols, vals = [], [], []
        for i in range(3):
            for j in range(3):
                ix = centre[:, 0] + i - 1
                iy = centre[:, 1] + j - 1
                rows.append(np.arange(len(points)))
                cols.append(ix * ny + iy)
                vals.append(wx[:, i] * wy[:, j])
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(points), nx * ny))

    def node_xy(self, index: np.ndarray) -> np.ndarray:
        nx, ny = self.shape
        return self.origin + self.pitch * np.column_stack([index // ny, index % ny])

    def kernel_spectrum(self, gap: float) -> np.ndarray:
        """The rfft2 of 1/sqrt(r^2 + gap^2) on the doubled grid (aperiodic convolution)."""
        nx, ny = self.shape
        ix = np.r_[0:nx, -nx:0]
        iy = np.r_[0:ny, -ny:0]
        X, Y = np.meshgrid(ix * self.pitch, iy * self.pitch, indexing="ij")
        r2 = X * X + Y * Y + gap * gap
        with np.errstate(divide="ignore"):
            g = np.where(r2 > 0.0, 1.0 / np.sqrt(r2), 0.0)  # the coincident node is precorrected away
        return np.fft.rfft2(g)


class HybridOperator:
    """``Z`` of a set of sheets and links as an operator, and its near part as a sparse matrix."""

    def __init__(self, sheets: list[Sheet], links, frequency: float, grid_pitch: float, near_factor: float = 2.5,
                 close_factor: float = 1.2, near_floor: float = 0.0, taper_radius: float = 2.0e-3,
                 node_ports=(), node_links=(), node_element_links=(), node_link_blocks=(), log=print,
                 correction_floor: float | None = None, grid_reach_pitches: float | None = None,
                 correction_gap: float | None = None, near_same_layer: bool = False,
                 sheet_impedance=None, sheet_couplings=()) -> None:
        """Near zone (the defaults keep the earlier behaviour).  A pair of elements is corrected
        (exact moments for the grid's) within max(near_factor * diameter,
        grid_reach_pitches * grid pitch, correction_floor); grid_reach_pitches
        defaults to 2 near_factor and correction_floor to max(near_floor,
        taper_radius), which ties the corrected zone to the preconditioner's
        taper.  correction_gap: sheet pairs farther apart than this (vertical
        gap) take the grid alone -- across a board's core the kernel is smooth
        on the grid.  near_same_layer: the preconditioner keeps only the sheet
        pairs of one layer (a layer's filaments).  The preconditioner's taper
        must not reach past the corrected zone.

        Surface sheets (hybrid_surface): ``sheet_impedance`` maps a sheet to
        its complex impedance per square, which then replaces rho / t, and
        ``sheet_couplings`` lists (a, b, z) -- sheets on one mesh whose
        currents at the same place couple through z per square (a layer's two
        faces).  Both enter the impedance with the RT0 mass matrix.
        """
        t0 = time.time()
        self.taper_radius = float(taper_radius)
        self.correction_floor = max(float(near_floor), float(taper_radius)) if correction_floor is None else float(correction_floor)
        self.grid_reach_pitches = 2.0 * float(near_factor) if grid_reach_pitches is None else float(grid_reach_pitches)
        self.correction_gap = None if correction_gap is None else float(correction_gap)
        self.near_same_layer = bool(near_same_layer)
        if self.taper_radius > max(self.correction_floor, self.grid_reach_pitches * float(grid_pitch)) + 1e-15:
            raise ValueError("taper_radius reaches past the corrected near zone")
        self.close_factor = float(close_factor)
        self.near_floor = float(near_floor)
        self.sheets = sheets
        self.frequency = float(frequency)
        self.omega = 2.0 * math.pi * self.frequency
        spaces = []
        for s_ in sheets:
            if not any(s_.space is x for x in spaces):
                spaces.append(s_.space)
        self.spaces = spaces
        self.space_of = [next(i for i, x in enumerate(spaces) if x is s_.space) for s_ in sheets]
        allpts = np.vstack([_point_phi(x)[0] for x in spaces])
        self.grid = Grid.around(allpts, grid_pitch)
        # per space: points, Phi, stencil, B_d = Phi C_d^T (P x M), node moments N = S^T Phi
        self._sp = []
        for x in spaces:
            pts, phi = _point_phi(x)
            stencil = self.grid.stencil(pts)
            b = [(phi @ x.component(d).T).tocsr() for d in range(2)]
            self._sp.append(dict(points=pts, phi=phi, stencil=stencil, b=b, node_moments=(stencil.T @ phi).tocsc()))
        sizes = [s_.space.size for s_ in sheets]
        elems = [len(s_.space.kind) for s_ in sheets]
        self.dof0 = np.r_[0, np.cumsum(sizes)]
        self.el0 = np.r_[0, np.cumsum(elems)]
        self.links = links
        nlink = sum(len(l[2]) for l in links)
        # Node ports: a port whose edges meet in one node of unknown potential
        # (a via barrel's end: its hole's wall on every filament of a layer),
        # and links between such nodes (the barrels).
        self.node_ports = list(node_ports)
        self.node_of_port = {p: k for k, p in enumerate(self.node_ports)}
        self.node_links = list(node_links)
        # (port, sheet, elements, impedance): a node joined to elements of a
        # sheet (a barrel's end on a layer without a hole: copper over it)
        self.node_element_links = list(node_element_links)
        n_node_el = sum(len(l[2]) for l in self.node_element_links)
        # Coupled node links: each block is (pairs [(port_a, port_b), ...],
        # resistances (k,), inductance matrix (k, k)) -- a barrel cut into
        # azimuthal strips, coupled to one another.
        # A block may carry a fourth entry, a group label per branch (a barrel's
        # strips share one): the preconditioner then keeps only the couplings
        # within a group, so a block coupling every barrel to every other does
        # not tie the whole board together in the near factor (the operator
        # itself keeps them all).
        self.node_link_blocks = [(list(b[0]), np.asarray(b[1], dtype=float), np.asarray(b[2], dtype=float))
                                 for b in node_link_blocks]
        self._node_link_groups = [np.asarray(b[3]) if len(b) > 3 else None for b in node_link_blocks]
        n_blocks = sum(len(b[0]) for b in self.node_link_blocks)
        self.m_plane = int(self.dof0[-1])
        self.m = self.m_plane + nlink + len(self.node_links) + n_node_el + n_blocks
        self.n_elements = int(self.el0[-1])
        self.n = self.n_elements + len(self.node_ports)

        # Kernel spectra per distinct sheet offset; the far gap is the GMD.
        self._pair_key = {}
        self._spectra = {}
        for a, sa in enumerate(sheets):
            for b, sb in enumerate(sheets):
                gap = thickness_gap(sa.thickness, sa.z, sb.thickness, sb.z)
                key = (self.space_of[a], self.space_of[b], round(sa.thickness, 12), round(sb.thickness, 12), round(abs(sa.z - sb.z), 12))
                self._pair_key[a, b] = (key, gap)
                self._full_spectrum(gap)
        t1 = time.time()

        # Near corrections and the near inductance, per distinct key, in dof space.
        self._corr = {}
        self._near = {}
        # Only a <= b is integrated; (b, a) is its transpose, so the operator is
        # symmetric as the inductance is.  The near moments integrate the outer
        # element by quadrature and the inner one in closed form, which is not
        # symmetric in the pair: between adjacent sliver triangles the two
        # orders differed by up to ~10 %, and MUMPS's LDL^T (the upper triangle)
        # then solved its own matrix only to 1e-2.  A sheet's pairs with itself
        # take the mean of both orders.
        skipped = 0
        for (a, b), (key, gap) in self._pair_key.items():
            if key in self._corr or a > b:
                continue
            sa_, sb_ = sheets[a], sheets[b]
            ia, ib = self.space_of[a], self.space_of[b]
            if self.correction_gap is not None and gap > self.correction_gap:
                shape = (sa_.space.size, sb_.space.size)
                self._corr[key] = sp.csr_matrix(shape)
                self._near[key] = sp.csr_matrix(shape)
                reverse_key = self._pair_key[b, a][0]
                if reverse_key != key:
                    self._corr[reverse_key] = sp.csr_matrix(shape[::-1])
                    self._near[reverse_key] = sp.csr_matrix(shape[::-1])
                skipped += 1
                continue
            exact, approx, kept = self._near_pair(ia, ib, gap, thickness_nodes(sa_.thickness, sa_.z, sb_.thickness, sb_.z),
                                                  near_factor)
            reverse_key = self._pair_key[b, a][0]
            if reverse_key == key and ia == ib:
                exact = ((exact + exact.T) * 0.5).tocsr()
                kept = ((kept + kept.T) * 0.5).tocsr()
            ca = [spaces[ia].component(d) for d in range(2)]
            cb = [spaces[ib].component(d) for d in range(2)]
            self._corr[key] = KAPPA * sum((ca[d] @ (exact - approx) @ cb[d].T) for d in range(2)).tocsr()
            self._near[key] = KAPPA * sum((ca[d] @ kept @ cb[d].T) for d in range(2)).tocsr()
            if reverse_key != key:
                self._corr[reverse_key] = self._corr[key].T.tocsr()
                self._near[reverse_key] = self._near[key].T.tocsr()
        t2 = time.time()

        # Resistance (block diagonal) and links.
        sheet_impedance = dict(sheet_impedance or {})
        self.sheet_couplings = list(sheet_couplings)
        blocks = [s_.space.resistance_sparse(sheet_impedance.get(a, s_.resistivity / s_.thickness))
                  for a, s_ in enumerate(sheets)]
        # A link's fifth entry is its resistance, or (resistance, inductance):
        # a via barrel's own inductance, uncoupled.
        def link_impedance(entry):
            value = entry[4]
            if isinstance(value, tuple):
                r, l_ = value
                return np.broadcast_to(np.asarray(r, dtype=float) + 1j * self.omega * np.asarray(l_, dtype=float), np.shape(entry[2]))
            return np.broadcast_to(np.asarray(value, dtype=float), np.shape(entry[2]))

        link_r = np.concatenate([link_impedance(l) for l in links]) if links else np.zeros(0)
        node_z = np.array([link_impedance((None, None, [0], None, z))[0] for (_a, _b, z) in self.node_links]) \
            if self.node_links else np.zeros(0)
        node_el_z = np.concatenate([link_impedance((None, None, l[2], None, l[3])) for l in self.node_element_links]) \
            if self.node_element_links else np.zeros(0)
        diagonal = np.concatenate([link_r, node_z, node_el_z])
        coupled = [sp.csr_matrix(np.diag(r) + 1j * self.omega * l) if self.omega > 0 else sp.csr_matrix(np.diag(r))
                   for _pairs, r, l in self.node_link_blocks]
        coupled_near = []
        for (_pairs, r, l), groups in zip(self.node_link_blocks, self._node_link_groups):
            if groups is None or self.omega == 0:
                coupled_near.append(sp.csr_matrix(np.diag(r) + 1j * self.omega * l) if self.omega > 0 else sp.csr_matrix(np.diag(r)))
            else:
                same = groups[:, None] == groups[None, :]
                coupled_near.append(sp.csr_matrix(np.diag(r) + 1j * self.omega * np.where(same, l, 0.0)))
        plane_and_links = [sp.csr_matrix(bk) for bk in blocks] + ([sp.diags(diagonal)] if len(diagonal) else [])
        self.resistance = sp.block_diag(plane_and_links + coupled, format="csr")
        self._impedance_near_links = sp.block_diag(plane_and_links + coupled_near, format="csr")
        if self.sheet_couplings:
            rows, cols, vals = [], [], []
            for a, b, z in self.sheet_couplings:
                if sheets[a].space is not sheets[b].space:
                    raise ValueError("coupled surface sheets must share a mesh")
                mass = sheets[a].space.resistance_sparse(1.0).tocoo()
                for (p_, q_) in ((a, b), (b, a)):
                    rows.append(mass.row + self.dof0[p_])
                    cols.append(mass.col + self.dof0[q_])
                    vals.append(z * mass.data)
            extra = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.m, self.m))
            self.resistance = (self.resistance.astype(np.complex128) + extra).tocsr()
            self._impedance_near_links = (self._impedance_near_links.astype(np.complex128) + extra).tocsr()
        # Ports.
        ports = np.full(self.m, -1)
        for a, s_ in enumerate(sheets):
            p = s_.space.port_of_dof
            mapping = np.asarray(s_.port_ids if s_.port_ids else range(max(p.max() + 1, 0)), dtype=np.int64)
            ports[self.dof0[a]:self.dof0[a + 1]] = np.where(p >= 0, mapping[np.maximum(p, 0)] if len(mapping) else -1, -1)
        self.ports = ports
        # Divergence (m x n).
        rows, cols, vals = [], [], []
        for a, s_ in enumerate(sheets):
            dv = s_.space.divergence().tocoo()
            rows.append(dv.row + self.dof0[a])
            cols.append(dv.col + self.el0[a])
            vals.append(dv.data)
        row = self.m_plane
        for a, b, ea, eb, _r in links:
            k = np.arange(row, row + len(ea))
            rows += [k, k]
            cols += [self.el0[a] + np.asarray(ea), self.el0[b] + np.asarray(eb)]
            vals += [np.ones(len(ea)), -np.ones(len(ea))]
            row += len(ea)
        on_node = np.flatnonzero(np.isin(ports, self.node_ports)) if self.node_ports else np.zeros(0, dtype=np.int64)
        if len(on_node):
            rows.append(on_node)
            cols.append(self.n_elements + np.array([self.node_of_port[int(p)] for p in ports[on_node]]))
            vals.append(-np.ones(len(on_node)))
        for k, (pa, pb, _z) in enumerate(self.node_links):
            rows += [np.array([row + k]), np.array([row + k])]
            cols += [np.array([self.n_elements + self.node_of_port[pa]]), np.array([self.n_elements + self.node_of_port[pb]])]
            vals += [np.ones(1), -np.ones(1)]
        row += len(self.node_links)
        for port, sheet, elements, _z in self.node_element_links:
            k = np.arange(row, row + len(elements))
            rows += [k, k]
            cols += [np.full(len(elements), self.n_elements + self.node_of_port[port]), self.el0[sheet] + np.asarray(elements)]
            vals += [np.ones(len(elements)), -np.ones(len(elements))]
            row += len(elements)
        for pairs, _r, _l in self.node_link_blocks:
            for pa, pb in pairs:
                rows += [np.array([row]), np.array([row])]
                cols += [np.array([self.n_elements + self.node_of_port[pa]]), np.array([self.n_elements + self.node_of_port[pb]])]
                vals += [np.ones(1), -np.ones(1)]
                row += 1
        self.divergence = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.m, self.n))
        self.timings = dict(grid_s=round(t1 - t0, 2), near_s=round(t2 - t1, 2), resistance_s=round(time.time() - t2, 2))
        log(f"hybrid operator: {len(sheets)} sheets, {self.m} dofs, {self.n} elements, grid {self.grid.shape}, "
            f"{len(self._corr)} pair kinds ({skipped} beyond correction_gap), near nnz {sum(v.nnz for v in self._near.values())}, "
            f"correction nnz {sum(v.nnz for v in self._corr.values())}, {self.timings}")

    # -- near pairs ------------------------------------------------------------
    def _near_pair(self, ia: int, ib: int, gap: float, nodes, near_factor: float):
        """Exact near moments and the grid's approximation of the same pairs, (3E_a, 3E_b) sparse."""
        sa, sb = self.spaces[ia], self.spaces[ib]
        floor = max(self.grid_reach_pitches * self.grid.pitch, self.correction_floor)
        # The pairs within reach in the plane, once per pair of meshes (every
        # sheet pair of those meshes filters them by its own gap).
        cache = self.__dict__.setdefault("_pair_cache", {})
        if (ia, ib, near_factor) not in cache:
            cache[ia, ib, near_factor] = _near_pairs(sa.centroid, sa.diameter, sb.centroid, sb.diameter, near_factor, floor)
        e, f = cache[ia, ib, near_factor]
        if gap > 0.0 and len(e):
            reach = np.maximum(near_factor * np.maximum(sa.diameter[e], sb.diameter[f]), floor)
            d = sa.centroid[e] - sb.centroid[f]
            keep = np.einsum("pi,pi->p", d, d) + gap * gap < reach * reach
            e, f = e[keep], f[keep]
        if len(e) == 0:
            z = sp.csr_matrix((3 * len(sa.kind), 3 * len(sb.kind)))
            return z, z, z
        # Closest pairs by the closed form; the rest of the near zone by point
        # quadrature on both elements (averaged over the thicknesses too).
        dist = np.linalg.norm(sa.centroid[e] - sb.centroid[f], axis=1)
        close = dist * dist + gap * gap < (self.close_factor * np.maximum(sa.diameter[e], sb.diameter[f])) ** 2
        exact = np.zeros((len(e), 3, 3))
        if np.any(close):
            exact[close] = near_moments_over(sa, e[close], f[close], nodes[0], nodes[1], other=sb)
        if np.any(~close):
            exact[~close] = self._point_moments(ia, ib, e[~close], f[~close], nodes)
        approx = self._grid_moments(ia, ib, e, f, gap)
        shape = (3 * len(sa.kind), 3 * len(sb.kind))

        def to_sparse(vals):
            r = (3 * e[:, None, None] + np.arange(3)[None, :, None]).repeat(3, axis=2)
            c = (3 * f[:, None, None] + np.arange(3)[None, None, :]).repeat(3, axis=1)
            return sp.csr_matrix((vals.ravel(), (r.ravel(), c.ravel())), shape=shape)

        # The preconditioner's copy: cutting the inductance at a distance makes
        # it indefinite (negative eigenvalues down to -0.5 of the diagonal on a
        # 0.4 mm mesh), and GMRES then stalls erratically.  With taper_radius
        # the pairs are weighted by Wendland's compactly supported C2 function
        # of the 3-D centroid distance: a positive definite function, so the
        # weighted (Schur) product of the moments stays positive definite.
        if self.taper_radius > 0.0:
            x = np.sqrt(dist * dist + gap * gap) / self.taper_radius
            weight = np.where(x < 1.0, (1.0 - x) ** 4 * (4.0 * x + 1.0), 0.0)
            kept = exact * weight[:, None, None]
        else:
            kept = exact
        return to_sparse(exact), to_sparse(approx), to_sparse(kept)

    def _point_moments(self, ia: int, ib: int, e: np.ndarray, f: np.ndarray, nodes, chunk: int = 20000) -> np.ndarray:
        """(P, 3, 3) moments of element pairs by point quadrature on both, the kernel averaged over ``nodes``."""
        sa, sb = self.spaces[ia], self.spaces[ib]
        quad = self.__dict__.setdefault("_far_points", {})
        for i, s_ in ((ia, sa), (ib, sb)):
            if i not in quad:
                quad[i] = element_points(s_.vertices, s_.corners, s_.kind, order="far")
        (pa, wa), (pb, wb) = quad[ia], quad[ib]
        h, hw = nodes
        sheet = _native_sheet("rt0_point_moments")
        if sheet is not None:
            from electrical.threads import thread_budget

            return np.asarray(sheet.rt0_point_moments(
                np.ascontiguousarray(pa, dtype=np.float64), np.ascontiguousarray(wa, dtype=np.float64),
                np.ascontiguousarray(sa.centroid), np.ascontiguousarray(pb, dtype=np.float64),
                np.ascontiguousarray(wb, dtype=np.float64), np.ascontiguousarray(sb.centroid),
                np.ascontiguousarray(e, dtype=np.int64), np.ascontiguousarray(f, dtype=np.int64),
                np.ascontiguousarray(h, dtype=np.float64), np.ascontiguousarray(hw, dtype=np.float64), thread_budget()))
        out = np.zeros((len(e), 3, 3))
        for start in range(0, len(e), chunk):
            ee, ff = e[start:start + chunk], f[start:start + chunk]
            xa, xb = pa[ee], pb[ff]  # (P, q, 2)
            fa = np.concatenate([np.ones(xa.shape[:2] + (1,)), xa - sa.centroid[ee][:, None, :]], axis=2) * wa[ee][..., None]
            fb = np.concatenate([np.ones(xb.shape[:2] + (1,)), xb - sb.centroid[ff][:, None, :]], axis=2) * wb[ff][..., None]
            dx = xa[:, :, None, 0] - xb[:, None, :, 0]
            dy = xa[:, :, None, 1] - xb[:, None, :, 1]
            r2 = dx * dx + dy * dy
            g = np.zeros_like(r2)
            for hk, wk in zip(h, hw):
                g += wk / np.sqrt(r2 + hk * hk + 1e-300)
            out[start:start + chunk] = np.matmul(np.matmul(fa.transpose(0, 2, 1), g), fb)
        return out

    def _padded(self, i: int):
        """Each element's grid nodes (E, K), -1 padded, and their [1, a] moments (E, K, 3)."""
        key = ("padded", i)
        if key in self.__dict__:
            return self.__dict__[key]
        nm = self._sp[i]["node_moments"].tocoo()  # (nodes, 3E)
        element = nm.col // 3
        component = nm.col % 3
        order = np.lexsort((nm.row, element))
        element, node, component, value = element[order], nm.row[order], component[order], nm.data[order]
        # one slot per (element, node)
        new = np.r_[True, (element[1:] != element[:-1]) | (node[1:] != node[:-1])]
        slot_id = np.cumsum(new) - 1
        first_slot = np.r_[0, np.flatnonzero(element[1:] != element[:-1]) + 1]
        count = len(self.spaces[i].kind)
        start_of = np.zeros(count, dtype=np.int64)
        start_of[element[first_slot]] = slot_id[first_slot]
        local = slot_id - start_of[element]
        k = int(local.max()) + 1
        index = np.full((count, k), -1, dtype=np.int64)
        moments = np.zeros((count, k, 3))
        index[element, local] = node
        np.add.at(moments, (element, local, component), value)
        self.__dict__[key] = (index, moments)
        return index, moments

    def _grid_moments(self, ia: int, ib: int, e: np.ndarray, f: np.ndarray, gap: float, chunk: int = 4000) -> np.ndarray:
        """(P, 3, 3): N_E^T G N_F, what the grid gives each pair."""
        from electrical import backend as _backend
        from electrical.threads import thread_budget

        idx_a, mom_a = self._padded(ia)
        idx_b, mom_b = self._padded(ib)
        k = max(idx_a.shape[1], idx_b.shape[1])
        if idx_a.shape[1] < k:
            idx_a = np.pad(idx_a, ((0, 0), (0, k - idx_a.shape[1])), constant_values=-1)
            mom_a = np.pad(mom_a, ((0, 0), (0, k - mom_a.shape[1]), (0, 0)))
        if idx_b.shape[1] < k:
            idx_b = np.pad(idx_b, ((0, 0), (0, k - idx_b.shape[1])), constant_values=-1)
            mom_b = np.pad(mom_b, ((0, 0), (0, k - mom_b.shape[1]), (0, 0)))
        core = _backend.core()
        sheet = getattr(core, "sheet", None) if core is not None else None
        if sheet is not None and hasattr(sheet, "rt0_grid_moments"):
            return np.asarray(sheet.rt0_grid_moments(
                np.ascontiguousarray(idx_a), np.ascontiguousarray(mom_a), np.ascontiguousarray(idx_b),
                np.ascontiguousarray(mom_b), float(self.grid.origin[0]), float(self.grid.origin[1]),
                float(self.grid.pitch), int(self.grid.shape[1]), float(gap),
                np.ascontiguousarray(e, dtype=np.int64), np.ascontiguousarray(f, dtype=np.int64), thread_budget()))
        out = np.zeros((len(e), 3, 3))
        for start in range(0, len(e), chunk):
            ee, ff = e[start:start + chunk], f[start:start + chunk]
            xa = self.grid.node_xy(np.maximum(idx_a[ee], 0).ravel()).reshape(len(ee), -1, 2)
            xb = self.grid.node_xy(np.maximum(idx_b[ff], 0).ravel()).reshape(len(ff), -1, 2)
            dx = xa[:, :, None, 0] - xb[:, None, :, 0]
            dy = xa[:, :, None, 1] - xb[:, None, :, 1]
            r2 = dx * dx + dy * dy + gap * gap
            with np.errstate(divide="ignore"):
                g = np.where(r2 > 0.0, 1.0 / np.sqrt(r2), 0.0)
            out[start:start + chunk] = np.matmul(np.matmul(mom_a[ee].transpose(0, 2, 1), g), mom_b[ff])
        return out

    # -- products --------------------------------------------------------------
    def _stacked(self):
        """Distinct real kernel spectra (U, K) and each sheet pair's index into them (S, S)."""
        if "_stack" not in self.__dict__:
            gaps = sorted({round(gap, 15) for (_key, gap) in self._pair_key.values()})
            where = {g: i for i, g in enumerate(gaps)}
            # The kernel is even on the wrapped offsets, so its spectrum is real.
            spectra = np.stack([self._full_spectrum(g).real.ravel() for g in gaps])
            count = len(self.sheets)
            index = np.zeros((count, count), dtype=np.int64)
            for (a, b), (_key, gap) in self._pair_key.items():
                index[a, b] = where[round(gap, 15)]
            groups = {}
            for a, sp_index in enumerate(self.space_of):
                groups.setdefault(sp_index, []).append(a)
            self.__dict__["_stack"] = (spectra, index, groups)
        return self.__dict__["_stack"]

    def inductance_apply(self, x: np.ndarray) -> np.ndarray:
        """L x over the in-plane dofs (links carry none).

        Sheets on one mesh are spread and gathered together (one sparse
        product with a column per sheet, one batched FFT); in Fourier space
        every bin couples the sheets by a small dense matrix.
        """
        nx, ny = self.grid.shape
        spectra, index, groups = self._stacked()
        count = len(self.sheets)
        bins = 4 * nx * ny
        charges = np.zeros((2, count, bins), dtype=np.complex128)
        for sp_index, members in groups.items():
            info = self._sp[sp_index]
            columns = np.stack([x[self.dof0[a]:self.dof0[a + 1]] for a in members], axis=1)
            for d in range(2):
                q = info["stencil"].T @ (info["b"][d] @ columns)  # (nodes, len(members))
                grid = np.zeros((len(members), 2 * nx, 2 * ny), dtype=np.complex128)
                grid[:, :nx, :ny] = q.T.reshape(len(members), nx, ny)
                charges[d, members] = np.fft.fft2(grid, axes=(1, 2)).reshape(len(members), bins)
        y = np.zeros(self.m_plane, dtype=np.complex128)
        for sp_index, members in groups.items():
            info = self._sp[sp_index]
            potentials = np.empty((2, len(members), bins), dtype=np.complex128)
            for k, a in enumerate(members):
                kernel = spectra[index[a]]  # (S, K)
                for d in range(2):
                    potentials[d, k] = np.einsum("bk,bk->k", kernel, charges[d])
            out = np.zeros((self.dof0[members[0] + 1] - self.dof0[members[0]], len(members)), dtype=np.complex128)
            for d in range(2):
                pot = np.fft.ifft2(potentials[d].reshape(len(members), 2 * nx, 2 * ny), axes=(1, 2))[:, :nx, :ny]
                out += info["b"][d].T @ (info["stencil"] @ pot.reshape(len(members), nx * ny).T)
            for k, a in enumerate(members):
                y[self.dof0[a]:self.dof0[a + 1]] = KAPPA * out[:, k]
        for a in range(count):
            for b in range(count):
                key, _gap = self._pair_key[a, b]
                y[self.dof0[a]:self.dof0[a + 1]] += self._corr[key] @ x[self.dof0[b]:self.dof0[b + 1]]
        return y

    def _full_spectrum(self, gap: float) -> np.ndarray:
        k = ("full", round(gap, 15))
        if k not in self._spectra:
            nx, ny = self.grid.shape
            ix = np.r_[0:nx, -nx:0]
            iy = np.r_[0:ny, -ny:0]
            X, Y = np.meshgrid(ix * self.grid.pitch, iy * self.grid.pitch, indexing="ij")
            r2 = X * X + Y * Y + gap * gap
            with np.errstate(divide="ignore"):
                g = np.where(r2 > 0.0, 1.0 / np.sqrt(r2), 0.0)
            self._spectra[k] = np.fft.fft2(g)
        return self._spectra[k]

    def near_inductance(self) -> sp.csr_matrix:
        """The exact near inductance of every sheet pair, (m_plane, m_plane) sparse."""
        rows = []
        for a in range(len(self.sheets)):
            row = []
            for b in range(len(self.sheets)):
                key, _ = self._pair_key[a, b]
                if self.near_same_layer and self.sheets[a].layer != self.sheets[b].layer:
                    row.append(sp.csr_matrix(self._near[key].shape))
                else:
                    row.append(self._near[key])
            rows.append(row)
        return sp.bmat(rows, format="csr")

    def impedance_apply(self, x: np.ndarray) -> np.ndarray:
        y = self.resistance @ x
        if self.omega > 0:
            y = y.astype(np.complex128)
            y[:self.m_plane] += 1j * self.omega * self.inductance_apply(x[:self.m_plane])
        return y

    def near_impedance(self) -> sp.csr_matrix:
        z = self._impedance_near_links.astype(np.complex128)
        if self.omega > 0:
            near = self.near_inductance()
            pad = sp.csr_matrix((self.m - self.m_plane, self.m - self.m_plane))
            z = z + sp.block_diag([1j * self.omega * near, pad], format="csr")
        return z.tocsr()


def solve_ports_fast(op: HybridOperator, voltages: dict[int, complex], tolerance: float = 1e-8, restart: int = 120,
                     maxiter: int = 50, log=print, factor_threads: int | str | None = None):
    """GMRES on [[Z, -D], [D^T, 0]] with fixed port voltages, preconditioned by the near system.

    Returns dof currents, element potentials, the current each port drives
    into the conductor and the iteration count.
    """
    m, n = op.m, op.n
    d = op.divergence
    rhs = np.zeros(m + n, dtype=np.complex128)
    for p, v in voltages.items():
        rhs[:m][op.ports == p] = -v
    t0 = time.time()
    solve_near, close = _near_factor(op.near_impedance(), d, factor_threads)
    t1 = time.time()

    def apply(v):
        i, phi = v[:m], v[m:]
        return np.concatenate([op.impedance_apply(i) - d @ phi, d.T @ i])

    count = [0]

    def counter(_):
        count[0] += 1

    a = spl.LinearOperator((m + n, m + n), matvec=apply, dtype=np.complex128)
    pre = spl.LinearOperator((m + n, m + n), matvec=solve_near, dtype=np.complex128)
    x, info = spl.gmres(a, rhs, M=pre, rtol=tolerance, restart=restart, maxiter=maxiter, callback=counter,
                        callback_type="pr_norm")
    close()
    t2 = time.time()
    op.timings.update(factor_s=round(t1 - t0, 2), gmres_s=round(t2 - t1, 2))
    log(f"near factor {t1 - t0:.1f} s, GMRES {count[0]} iterations {t2 - t1:.1f} s, info {info}")
    current = x[:m]
    driven = {p: -current[op.ports == p].sum() for p in voltages}
    return current, x[m:], driven, count[0]


def _near_factor(near: sp.csr_matrix, d: sp.csr_matrix, threads: int | str | None = None):
    """The near saddle point's solve: MUMPS's quasi-definite LDL^T when the core links it, else SuperLU.

    ``threads``: MUMPS's (``near_factor_threads``: one by default, ``"budget"``).
    """
    from .near_factor import NEAR_REGULARIZATION, MumpsQuasiDefiniteNear, mumps_available, resolve_near_factor_threads

    if mumps_available():
        factor = MumpsQuasiDefiniteNear(near, d, NEAR_REGULARIZATION["mumps_quasi_definite"],
                                        threads=resolve_near_factor_threads(threads))
        return factor.solve, factor.close
    lu = spl.splu(sp.bmat([[near, -d], [d.T, None]], format="csc"), permc_spec="COLAMD")
    return lu.solve, (lambda: None)


def connected_components(op: HybridOperator) -> np.ndarray:
    """The component of every element, joined through dofs (edges and links)."""
    import scipy.sparse.csgraph as cg

    d = op.divergence.tocsc()
    adjacency = (d.T @ d)
    adjacency.data = np.ones_like(adjacency.data)
    _, labels = cg.connected_components(adjacency, directed=False)
    return labels


def solve_injection_fast(op: HybridOperator, injected: np.ndarray, tolerance: float = 1e-9, restart: int = 120,
                         maxiter: int = 50, log=print, factor_threads: int | str | None = None):
    """GMRES with currents fed into elements; one element of every component grounded.

    Returns dof currents, element potentials (0 at the grounds) and the
    iteration count.
    """
    labels = connected_components(op)
    ground = np.array([np.flatnonzero(labels == c)[0] for c in np.unique(labels)])
    keep = np.setdiff1d(np.arange(op.n), ground)
    d = op.divergence[:, keep].tocsr()
    m, n = op.m, len(keep)
    for c in np.unique(labels):
        total = injected[labels == c].sum()
        if abs(total) > 1e-9 * max(np.abs(injected).max(), 1e-30):
            raise ValueError(f"component {c} is fed a net {total} A")
    rhs = np.concatenate([np.zeros(m, dtype=np.complex128), np.asarray(injected, dtype=np.complex128)[keep]])
    t0 = time.time()
    solve_near, close = _near_factor(op.near_impedance(), d, factor_threads)
    t1 = time.time()

    def apply(v):
        i, phi = v[:m], v[m:]
        return np.concatenate([op.impedance_apply(i) - d @ phi, d.T @ i])

    count = [0]
    a = spl.LinearOperator((m + n, m + n), matvec=apply, dtype=np.complex128)
    pre = spl.LinearOperator((m + n, m + n), matvec=solve_near, dtype=np.complex128)
    x, info = spl.gmres(a, rhs, M=pre, rtol=tolerance, restart=restart, maxiter=maxiter,
                        callback=lambda _: count.__setitem__(0, count[0] + 1), callback_type="pr_norm")
    close()
    t2 = time.time()
    log(f"components {len(ground)}, near factor {t1 - t0:.1f} s, GMRES {count[0]} iterations {t2 - t1:.1f} s, info {info}")
    phi = np.zeros(op.n, dtype=np.complex128)
    phi[keep] = x[m:]
    return x[:m], phi, count[0]
