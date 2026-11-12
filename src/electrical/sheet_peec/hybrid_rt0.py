"""Mixed square/triangle sheet PEEC with Raviart-Thomas (RT0) currents (experiment).

A sheet conductor is cut into elements -- coarse axis-aligned squares inside,
triangles along its outline -- and its surface current is the lowest-order
Raviart-Thomas field: one unknown per interior edge, the current crossing it.
On a triangle the edge's basis is ``s (r - r_opposite) / (2 A)`` (the RWG
function), on a rectangle the rooftop ``s ((r - p_opposite) . n) n / A``; both
put one ampere across their edge and none across the others, and both are
linear in ``r``, which is all this module uses:

    K_i(r) = C_i [1, (r - c_E)]^T        (C_i a 2 x 3 matrix on element E)

so every Galerkin integral reduces to moments of the kernel over element pairs,

    W_EF = int_E int_F G(r, r') [1; r - c_E] [1; r' - c_F]^T dA dA'   (3 x 3),

and ``L_ij = mu0/4pi sum_d (C_i W_EF C_j^T)_dd``.  The far moments come from
point quadrature on both elements; the near ones from quadrature on the outer
element and the closed-form potential of a uniform and of a linear source on
the inner polygon (Wilton et al., IEEE Trans. AP 32(3), 1984), so the 1/R
singularity is integrated exactly.  Thickness enters through the kernel
``1 / sqrt(R^2 + g^2)`` with ``g`` the geometric mean distance of the two
sheets' thicknesses along z, exact where the in-plane distances exceed them.

The resistance is the RT0 mass matrix, ``rho / t int K_i . K_j``; the element
potentials are piecewise constant, and Kirchhoff's current law is the RT0
divergence, so the system is the saddle point the uniform sheet solve uses:
``[[Z, -D], [D^T, 0]] [I; phi] = [0; s]``.

This module solves densely: it is the accuracy prototype of the hybrid mesh,
not its fast path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

MU0 = 4.0e-7 * math.pi
GMD_SELF = math.exp(-1.5)  # a segment's geometric mean distance from itself, over its length


# ---------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------

def _triangle_rule(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Barycentric points (n, 3) and weights summing to one on a triangle."""
    if degree <= 1:
        return np.array([[1 / 3, 1 / 3, 1 / 3]]), np.array([1.0])
    if degree == 2:
        b = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
        return b, np.full(3, 1 / 3)
    if degree <= 4:  # Dunavant 6 points, degree 4
        a1, w1 = 0.445948490915965, 0.223381589678011
        a2, w2 = 0.091576213509771, 0.109951743655322
        pts, wts = [], []
        for a, w in ((a1, w1), (a2, w2)):
            c = 1 - 2 * a
            pts += [[c, a, a], [a, c, a], [a, a, c]]
            wts += [w, w, w]
        return np.array(pts), np.array(wts)
    # Dunavant 12 points, degree 6
    groups = (
        (0.063089014491502, 0.050844906370207),
        (0.249286745170910, 0.116786275726379),
    )
    pts, wts = [], []
    for a, w in groups:
        c = 1 - 2 * a
        pts += [[c, a, a], [a, c, a], [a, a, c]]
        wts += [w, w, w]
    a, b, w = 0.053145049844817, 0.310352451033784, 0.082851075618374
    c = 1 - a - b
    for p in ([a, b, c], [a, c, b], [b, a, c], [b, c, a], [c, a, b], [c, b, a]):
        pts.append(p)
        wts.append(w)
    return np.array(pts), np.array(wts)


def _quad_rule(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Points (n^2, 2) in [0, 1]^2 and weights summing to one."""
    x, w = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    X, Y = np.meshgrid(x, x, indexing="ij")
    W = np.outer(w, w)
    return np.column_stack([X.ravel(), Y.ravel()]), W.ravel()


def element_points(vertices: np.ndarray, corners: np.ndarray, kind: np.ndarray, order: str = "far"):
    """Quadrature points of every element: (E, q, 2) points and (E, q) weights (area included).

    ``corners`` (E, 4): vertex indices, the fourth -1 on a triangle; ``kind``
    0 for a triangle, 1 for a rectangle (corners counter-clockwise).  Triangles
    and rectangles get rules of the same point count so the arrays are dense.
    """
    tri_b, tri_w = _triangle_rule(4 if order == "far" else 6)
    quad_x, quad_w = _quad_rule(2 if order == "far" else 3)
    q = max(len(tri_w), len(quad_w))
    count = len(kind)
    points = np.zeros((count, q, 2))
    weights = np.zeros((count, q))
    tri = np.flatnonzero(kind == 0)
    if tri.size:
        v = vertices[corners[tri, :3]]  # (T, 3, 2)
        e1, e2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
        area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
        n = len(tri_w)
        points[tri, :n] = np.einsum("qk,tkd->tqd", tri_b, v)
        weights[tri, :n] = area[:, None] * tri_w[None, :]
        points[tri, n:] = points[tri, :1]
    rect = np.flatnonzero(kind == 1)
    if rect.size:
        v = vertices[corners[rect]]  # (R, 4, 2) counter-clockwise from the lower left
        lo = v.min(axis=1)
        hi = v.max(axis=1)
        size = hi - lo
        n = len(quad_w)
        points[rect, :n] = lo[:, None, :] + quad_x[None, :, :] * size[:, None, :]
        weights[rect, :n] = (size[:, 0] * size[:, 1])[:, None] * quad_w[None, :]
        points[rect, n:] = points[rect, :1]
    return points, weights


# ---------------------------------------------------------------------------
# Closed-form potentials of a polygon (Wilton et al. 1984)
# ---------------------------------------------------------------------------

def polygon_potentials(points: np.ndarray, polygons: np.ndarray, height: np.ndarray | float):
    """``int_P 1/R dA'`` and ``int_P (rho' - rho)/R dA'`` over planar polygons.

    ``points`` (N, 2) observation points projected onto the polygon's plane,
    ``polygons`` (N, V, 2) counter-clockwise vertices, ``height`` the
    observation point's distance from the plane (scalar or (N,)), so
    ``R = sqrt(|rho - rho'|^2 + h^2)``.  Returns ``P0`` (N,) and ``P1`` (N, 2).
    Valid on the polygon, its edges and outside it, at any height.
    """
    points = np.asarray(points, dtype=np.float64)
    polygons = np.asarray(polygons, dtype=np.float64)
    d = np.broadcast_to(np.abs(np.asarray(height, dtype=np.float64)), points.shape[:1])
    start = polygons
    end = np.roll(polygons, -1, axis=1)
    edge = end - start
    length = np.linalg.norm(edge, axis=2)
    keep = length > 0.0
    tangent = edge / np.where(keep, length, 1.0)[..., None]
    normal = np.stack([tangent[..., 1], -tangent[..., 0]], axis=2)  # outward for counter-clockwise
    rel_start = start - points[:, None, :]
    rel_end = end - points[:, None, :]
    t0 = np.einsum("nvd,nvd->nv", rel_start, normal)
    lm = np.einsum("nvd,nvd->nv", rel_start, tangent)
    lp = np.einsum("nvd,nvd->nv", rel_end, tangent)
    d2 = (d * d)[:, None]
    r0sq = t0 * t0 + d2
    rp = np.sqrt(r0sq + lp * lp)
    rm = np.sqrt(r0sq + lm * lm)

    def plus(r, l):
        # R + l without cancellation: R0^2 / (R - l) when l < 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(l >= 0.0, r + l, r0sq / (r - l))

    gp = plus(rp, lp)
    gm = plus(rm, lm)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.log(gp / gm)
    # On an edge's line (R0 = 0) f is infinite where the segment passes under
    # the point, but it is multiplied by t0 = 0 and R0^2 = 0 there.
    tiny = r0sq <= 1e-30 * np.maximum(lp * lp + lm * lm, 1e-300)
    f = np.where(tiny | ~keep, 0.0, f)
    p0 = np.sum(np.where(keep, t0 * f, 0.0), axis=1)
    dd = d[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.arctan2(t0 * lp, r0sq + dd * rp) - np.arctan2(t0 * lm, r0sq + dd * rm)
    p0 = p0 - d * np.sum(np.where(keep, beta, 0.0), axis=1)
    weight = np.where(keep, r0sq * f + lp * rp - lm * rm, 0.0)
    p1 = 0.5 * np.einsum("nv,nvd->nd", weight, normal)
    return p0, p1


# ---------------------------------------------------------------------------
# The mesh's RT0 space
# ---------------------------------------------------------------------------

@dataclass
class RT0Space:
    """Elements, edges and the RT0 basis of one in-plane mesh (shared by every sheet)."""

    vertices: np.ndarray  # (V, 2) metres
    corners: np.ndarray  # (E, 4) vertex indices, -1 pads a triangle; counter-clockwise
    kind: np.ndarray  # (E,) 0 triangle, 1 axis-aligned rectangle
    port_edges: tuple = ()  # per port, an (n, 2) array of boundary vertex pairs made dofs
    area: np.ndarray = field(init=False)
    centroid: np.ndarray = field(init=False)
    diameter: np.ndarray = field(init=False)
    edges: np.ndarray = field(init=False)  # (M, 2) vertex pairs: interior edges, then port edges
    edge_elements: np.ndarray = field(init=False)  # (M, 2) element on each side: current flows [0] -> [1]; -1 outside
    port_of_dof: np.ndarray = field(init=False)  # (M,) the port a boundary dof belongs to, -1 inside
    boundary_edges: np.ndarray = field(init=False)  # (B, 3): vertex, vertex, element
    coefficients: sp.csr_matrix = field(init=False)  # (2 M, 3 E): rows x and y of every C_i

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.corners = np.asarray(self.corners, dtype=np.int64)
        self.kind = np.asarray(self.kind, dtype=np.int64)
        count = len(self.kind)
        sides = np.where(self.kind == 0, 3, 4)
        poly = self.polygons()
        x, y = poly[..., 0], poly[..., 1]
        xn, yn = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
        cross = x * yn - xn * y
        mask = np.arange(4)[None, :] < sides[:, None]
        cross = np.where(mask, cross, 0.0)
        area2 = cross.sum(axis=1)
        if np.any(area2 <= 0.0):
            raise ValueError("every element must be counter-clockwise with a positive area")
        self.area = 0.5 * area2
        cx = np.sum(np.where(mask, (x + xn) * cross, 0.0), axis=1) / (3.0 * area2)
        cy = np.sum(np.where(mask, (y + yn) * cross, 0.0), axis=1) / (3.0 * area2)
        self.centroid = np.column_stack([cx, cy])
        spread = poly - self.centroid[:, None, :]
        self.diameter = 2.0 * np.max(np.where(mask, np.linalg.norm(spread, axis=2), 0.0), axis=1)

        # Edges: an element's k-th edge runs from corner k to corner k+1.
        local = []
        for e in range(count):
            n = sides[e]
            for k in range(n):
                a, b = self.corners[e, k], self.corners[e, (k + 1) % n]
                local.append((min(a, b), max(a, b), e, k))
        local = np.array(local, dtype=np.int64)
        order = np.lexsort((local[:, 2], local[:, 1], local[:, 0]))
        local = local[order]
        key = local[:, 0] * (len(self.vertices) + 1) + local[:, 1]
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        counts = np.diff(np.r_[starts, len(key)])
        if np.any(counts > 2):
            raise ValueError("an edge is shared by more than two elements (non-manifold mesh)")
        interior = starts[counts == 2]
        boundary = starts[counts == 1]
        edges = [local[interior, :2]]
        edge_elements = [np.column_stack([local[interior, 2], local[interior + 1, 2]])]
        edge_local = [np.column_stack([local[interior, 3], local[interior + 1, 3]])]
        ports = [np.full(len(interior), -1)]
        bkey = key[boundary]
        lookup = {int(k): int(i) for k, i in zip(bkey, boundary)}
        taken = set()
        for p, pairs in enumerate(self.port_edges):
            pairs = np.sort(np.asarray(pairs, dtype=np.int64).reshape(-1, 2), axis=1)
            rows = []
            for a, b in pairs:
                k = int(a * (len(self.vertices) + 1) + b)
                if k not in lookup or k in taken:
                    raise ValueError(f"port {p}: ({a}, {b}) is not a free boundary edge")
                taken.add(k)
                rows.append(lookup[k])
            rows = np.array(rows, dtype=np.int64)
            edges.append(local[rows, :2])
            edge_elements.append(np.column_stack([local[rows, 2], np.full(len(rows), -1)]))
            edge_local.append(np.column_stack([local[rows, 3], np.full(len(rows), -1)]))
            ports.append(np.full(len(rows), p))
        self.edges = np.concatenate(edges)
        self.edge_elements = np.concatenate(edge_elements)
        self._edge_local = np.concatenate(edge_local)
        self.port_of_dof = np.concatenate(ports)
        self.boundary_edges = local[boundary][:, :3]
        self.coefficients = self._basis()

    def with_ports(self, *selectors) -> "RT0Space":
        """The same mesh with the boundary edges each selector picks (midpoints (B, 2) -> bool) as ports."""
        mid = 0.5 * (self.vertices[self.boundary_edges[:, 0]] + self.vertices[self.boundary_edges[:, 1]])
        ports = tuple(self.boundary_edges[np.asarray(sel(mid), dtype=bool), :2] for sel in selectors)
        return RT0Space(self.vertices, self.corners, self.kind, port_edges=ports)

    # -- geometry ------------------------------------------------------------
    def polygons(self) -> np.ndarray:
        """(E, 4, 2) element vertices; a triangle repeats its first vertex."""
        c = np.where(self.corners < 0, self.corners[:, :1], self.corners)
        return self.vertices[c]

    @property
    def size(self) -> int:
        return len(self.edges)

    # -- basis ---------------------------------------------------------------
    def _basis(self) -> sp.csr_matrix:
        """C_i for every (edge, side): K = C [1, r - c_E] on that element.

        Rows 2i (x) and 2i+1 (y) of dof i; columns 3E..3E+2 of element E.  An
        edge's two elements each contribute their own block (the dof is the sum).
        """
        rows, cols, vals = [], [], []
        poly = self.polygons()
        for side in (0, 1):
            sign = 1.0 if side == 0 else -1.0  # outward from element [0], inward into [1]
            elements = self.edge_elements[:, side]
            k = self._edge_local[:, side]
            for i, (e, kk) in enumerate(zip(elements, k)):
                if e < 0:
                    continue
                c = self.centroid[e]
                if self.kind[e] == 0:
                    opposite = poly[e, (kk + 2) % 3]
                    scale = sign / (2.0 * self.area[e])
                    m = scale * np.eye(2)
                    offset = m @ (c - opposite)
                else:
                    a, b = poly[e, kk], poly[e, (kk + 1) % 4]
                    t = (b - a) / np.linalg.norm(b - a)
                    n = np.array([t[1], -t[0]])  # outward
                    opposite = poly[e, (kk + 2) % 4]
                    m = sign / self.area[e] * np.outer(n, n)
                    offset = m @ (c - opposite)
                block = np.column_stack([offset, m])  # 2 x 3
                for d in range(2):
                    for j in range(3):
                        if block[d, j] != 0.0:
                            rows.append(2 * i + d)
                            cols.append(3 * e + j)
                            vals.append(block[d, j])
        return sp.csr_matrix((vals, (rows, cols)), shape=(2 * self.size, 3 * len(self.kind)))

    def component(self, d: int) -> sp.csr_matrix:
        """The (M, 3E) rows of the d-th component of every basis function."""
        return self.coefficients[d::2]

    def divergence(self) -> sp.csr_matrix:
        """(M, E): +1 on the element a dof leaves, -1 on the one it enters."""
        m = self.size
        rows = np.repeat(np.arange(m), 2)
        cols = self.edge_elements.ravel()
        vals = np.tile([1.0, -1.0], m)
        inside = cols >= 0
        return sp.csr_matrix((vals[inside], (rows[inside], cols[inside])), shape=(m, len(self.kind)))

    # -- moments -------------------------------------------------------------
    def mass_moments(self) -> np.ndarray:
        """(E, 3, 3): int_E [1; r - c][1; r - c]^T dA (degree 2, exact)."""
        pts, w = element_points(self.vertices, self.corners, self.kind, order="far")
        a = pts - self.centroid[:, None, :]
        phi = np.concatenate([np.ones(a.shape[:2] + (1,)), a], axis=2)
        return np.einsum("eq,eqk,eql->ekl", w, phi, phi)

    def resistance(self, sheet_resistance: float) -> np.ndarray:
        """Dense (M, M) RT0 mass matrix times the sheet resistance rho / t."""
        moments = self.mass_moments()
        block = sp.block_diag(list(moments), format="csr")
        total = np.zeros((self.size, self.size))
        for d in range(2):
            c = self.component(d)
            total += (c @ block @ c.T).toarray()
        return sheet_resistance * total


def kernel_moments(space: RT0Space, gap: float, near_factor: float = 2.5, other: RT0Space | None = None,
                   heights: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """(3E_a, 3E_b) moments W of the sheets' kernel between the elements of two meshes.

    Far pairs by point quadrature on both elements with ``1/sqrt(R^2 + gap^2)``
    (``gap`` the GMD of the two thicknesses: exact once the in-plane distance
    exceeds them); pairs closer (in 3D) than ``near_factor`` times the larger
    diameter by the outer element's finer rule and the inner polygon's closed
    form, averaged over the vertical offsets ``heights = (h, w)`` (from
    :func:`thickness_nodes`) when given, else at ``gap``.  ``other`` defaults
    to ``space`` itself.
    """
    other = space if other is None else other

    def weighted_points(sp_: RT0Space):
        pts, w = element_points(sp_.vertices, sp_.corners, sp_.kind, order="far")
        count, q = w.shape
        flat = pts.reshape(-1, 2)
        centred = (pts - sp_.centroid[:, None, :]).reshape(-1, 2)
        phi = np.concatenate([np.ones((len(flat), 1)), centred], axis=1) * w.reshape(-1)[:, None]
        owner = np.repeat(np.arange(count), q)
        prow = np.repeat(np.arange(len(flat)), 3)
        pcol = (3 * owner[:, None] + np.arange(3)[None, :]).ravel()
        big = sp.csr_matrix((phi.ravel(), (prow, pcol)), shape=(len(flat), 3 * count))
        return flat, big.T.tocsr()

    flat_a, phi_a = weighted_points(space)
    flat_b, phi_b = weighted_points(other)
    moments = np.zeros((phi_a.shape[0], phi_b.shape[0]))
    rows = max(1, int(1.0e8 // max(len(flat_b), 1)))
    for start in range(0, len(flat_a), rows):
        block = flat_a[start:start + rows]
        dx = block[:, None, 0] - flat_b[None, :, 0]
        dy = block[:, None, 1] - flat_b[None, :, 1]
        r2 = dx * dx + dy * dy + gap * gap
        # Coincident points belong to near pairs, which are replaced below.
        with np.errstate(divide="ignore"):
            kernel = np.where(r2 > 0.0, 1.0 / np.sqrt(r2), 0.0)
        partial = np.asarray((phi_b @ kernel.T).T)  # (rows, 3E_b)
        moments += np.asarray(phi_a[:, start:start + rows] @ partial)

    reach = near_factor * np.maximum(space.diameter[:, None], other.diameter[None, :])
    near_e, near_f = [], []
    for start in range(0, len(space.kind), 2000):
        d = np.linalg.norm(space.centroid[start:start + 2000, None, :] - other.centroid[None, :, :], axis=2)
        e, f = np.nonzero(d * d + gap * gap < reach[start:start + 2000] ** 2)
        near_e.append(e + start)
        near_f.append(f)
    near_e = np.concatenate(near_e)
    near_f = np.concatenate(near_f)
    if heights is None:
        exact = near_moments_over(space, near_e, near_f, [gap], [1.0], other=other)
    else:
        exact = near_moments_over(space, near_e, near_f, heights[0], heights[1], other=other)
    for k in range(3):
        for l in range(3):
            moments[3 * near_e + k, 3 * near_f + l] = exact[:, k, l]
    return moments


def near_moments(space: RT0Space, outer: np.ndarray, inner: np.ndarray, gap: float, chunk: int = 20000,
                 other: RT0Space | None = None) -> np.ndarray:
    """(P, 3, 3) moments of element pairs, the inner integral in closed form."""
    other = space if other is None else other
    pts, w = element_points(space.vertices, space.corners, space.kind, order="near")
    poly = other.polygons()
    result = np.zeros((len(outer), 3, 3))
    q = w.shape[1]
    for start in range(0, len(outer), chunk):
        e = outer[start:start + chunk]
        f = inner[start:start + chunk]
        p = pts[e].reshape(-1, 2)  # (Pq, 2)
        wt = w[e].reshape(-1)
        target = np.repeat(f, q)
        p0, p1 = polygon_potentials(p, poly[target], gap)
        a = p - space.centroid[np.repeat(e, q)]
        b_first = p1 + (p - other.centroid[target]) * p0[:, None]  # int_F (r' - c_F) G
        inner_vec = np.concatenate([p0[:, None], b_first], axis=1)  # (Pq, 3)
        outer_vec = np.concatenate([np.ones((len(p), 1)), a], axis=1) * wt[:, None]
        block = np.einsum("pk,pl->pkl", outer_vec, inner_vec).reshape(len(e), q, 3, 3).sum(axis=1)
        result[start:start + chunk] = block
    return result


def near_moments_over(space: RT0Space, outer: np.ndarray, inner: np.ndarray, heights: np.ndarray,
                      weights: np.ndarray, other: RT0Space | None = None, threads: int | None = None) -> np.ndarray:
    """``sum_k weights[k] near_moments(..., heights[k])``, in the C++ core when it is built."""
    other = space if other is None else other
    heights = np.atleast_1d(np.asarray(heights, dtype=np.float64))
    weights = np.atleast_1d(np.asarray(weights, dtype=np.float64))
    from electrical import backend as _backend
    from electrical.threads import thread_budget

    core = _backend.core()
    sheet = getattr(core, "sheet", None) if core is not None else None
    if sheet is not None and hasattr(sheet, "rt0_near_moments") and len(outer):
        pts, w = element_points(space.vertices, space.corners, space.kind, order="near")
        poly = np.ascontiguousarray(other.polygons())
        sides = np.where(other.kind == 0, 3, 4).astype(np.int32)
        return np.asarray(sheet.rt0_near_moments(
            np.ascontiguousarray(pts), np.ascontiguousarray(w), np.ascontiguousarray(space.centroid), poly, sides,
            np.ascontiguousarray(other.centroid), np.ascontiguousarray(outer, dtype=np.int64),
            np.ascontiguousarray(inner, dtype=np.int64), heights, weights,
            thread_budget() if threads is None else int(threads)))
    return sum(wk * near_moments(space, outer, inner, hk, other=other) for hk, wk in zip(heights, weights))


def inductance(space: RT0Space, gap: float, other: RT0Space | None = None,
               heights: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Dense (M_a, M_b) partial inductance between the RT0 currents of two sheets ``gap`` apart."""
    other = space if other is None else other
    moments = kernel_moments(space, gap, other=other, heights=heights)
    total = np.zeros((space.size, other.size))
    for d in range(2):
        ca = space.component(d)
        cb = other.component(d)
        total += (cb @ (ca @ moments).T).T  # C_a W C_b^T
    return MU0 / (4.0 * math.pi) * total


def thickness_nodes(t_a: float, z_a: float, t_b: float, z_b: float, points: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Vertical offsets ``|z - z'|`` and weights for averaging a kernel over two thicknesses.

    ``z - z'`` for z, z' uniform in the two intervals has a piecewise-linear
    (trapezoidal) density; each linear piece, split at zero and graded toward
    it (where the in-plane potential has its kink), gets a Gauss-Legendre rule
    weighted by the density.  The weights sum to one.
    """
    a0, a1 = z_a - t_a / 2, z_a + t_a / 2
    b0, b1 = z_b - t_b / 2, z_b + t_b / 2

    def density(zeta):
        return np.clip(np.minimum(a1, zeta + b1) - np.maximum(a0, zeta + b0), 0.0, None) / (t_a * t_b)

    breaks = sorted({a0 - b1, a0 - b0, a1 - b1, a1 - b0})
    lo, hi = breaks[0], breaks[-1]
    if lo < 0.0 < hi:
        breaks.append(0.0)
    breaks = sorted(set(breaks))
    pieces = []
    for left, right in zip(breaks[:-1], breaks[1:]):
        if right - left <= 0.0:
            continue
        if left >= 0.0 and left == min(abs(left), abs(right)) and left < 0.05 * (right - left):
            cut = left + (right - left) / 8
            pieces += [(left, cut), (cut, right)]
        elif right <= 0.0 and abs(right) < 0.05 * (right - left):
            cut = right - (right - left) / 8
            pieces += [(left, cut), (cut, right)]
        else:
            pieces.append((left, right))
    x, w = np.polynomial.legendre.leggauss(points)
    hs, ws = [], []
    for left, right in pieces:
        zeta = left + (x + 1.0) * 0.5 * (right - left)
        hs.append(np.abs(zeta))
        ws.append(w * 0.5 * (right - left) * density(zeta))
    h = np.concatenate(hs)
    weight = np.concatenate(ws)
    return h, weight / weight.sum()


def thickness_gap(t_a: float, z_a: float, t_b: float, z_b: float) -> float:
    """Geometric mean distance along z of two thickness intervals (centres z, thicknesses t)."""
    if t_a == t_b and z_a == z_b:
        return GMD_SELF * t_a

    def big_f(x: float) -> float:
        return 0.0 if x == 0.0 else 0.5 * x * x * math.log(abs(x)) - 0.75 * x * x

    a0, a1 = z_a - t_a / 2, z_a + t_a / 2
    b0, b1 = z_b - t_b / 2, z_b + t_b / 2
    # int_{a0}^{a1} int_{b0}^{b1} ln|z' - z| dz' dz
    total = big_f(b1 - a0) - big_f(b1 - a1) - big_f(b0 - a0) + big_f(b0 - a1)
    return math.exp(total / (t_a * t_b))


def solve_saddle(impedance: np.ndarray, divergence: sp.spmatrix, injected: np.ndarray, ground: int = 0):
    """Solve ``Z I = D phi`` (the drop along each dof) and ``D^T I = s`` with ``phi[ground] = 0``.

    ``injected`` (E,): the current fed into each element from outside (sums to
    zero).  Returns the dof currents and the element potentials.
    """
    d = divergence.toarray() if sp.issparse(divergence) else np.asarray(divergence)
    m, e = d.shape
    keep = np.r_[0:ground, ground + 1:e]
    dtype = np.result_type(impedance.dtype, np.float64)
    system = np.zeros((m + e - 1, m + e - 1), dtype=dtype)
    system[:m, :m] = impedance
    system[:m, m:] = -d[:, keep]
    system[m:, :m] = d[:, keep].T
    rhs = np.zeros(m + e - 1, dtype=dtype)
    rhs[m:] = np.asarray(injected)[keep]
    x = np.linalg.solve(system, rhs)
    phi = np.zeros(e, dtype=dtype)
    phi[keep] = x[m:]
    return x[:m], phi


def solve_ports(impedance: np.ndarray, divergence: sp.spmatrix, port_of_dof: np.ndarray, voltages) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drive the ports at ``voltages`` (one per port); every port edge is equipotential.

    Port dof i of port p closes ``Z I - D phi = -V_p`` (its drop from the
    element to the port), so the port fixes the potential and no element is
    grounded.  Returns the dof currents, the element potentials and each
    port's total current out of the conductor.
    """
    d = divergence.toarray() if sp.issparse(divergence) else np.asarray(divergence)
    m, e = d.shape
    dtype = np.result_type(impedance.dtype, np.float64)
    system = np.zeros((m + e, m + e), dtype=dtype)
    system[:m, :m] = impedance
    system[:m, m:] = -d
    system[m:, :m] = d.T
    rhs = np.zeros(m + e, dtype=dtype)
    voltages = np.asarray(voltages, dtype=dtype)
    on_port = port_of_dof >= 0
    rhs[:m][on_port] = -voltages[port_of_dof[on_port]]
    x = np.linalg.solve(system, rhs)
    current = x[:m]
    totals = np.array([current[port_of_dof == p].sum() for p in range(len(voltages))])
    return current, x[m:], totals


@dataclass(frozen=True)
class Sheet:
    """One conducting sheet: a mesh at height ``z`` with its thickness and resistivity."""

    space: RT0Space
    z: float
    thickness: float
    resistivity: float
    layer: str
    port_ids: tuple = ()  # global id of each of the space's ports


def assemble(sheets: list[Sheet], links: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]], frequency: float):
    """Impedance, divergence and dof bookkeeping of sheets joined by vertical links.

    ``links``: (sheet a, sheet b, elements in a, elements in b, resistance) --
    resistive joints between element pairs (between the filaments of one
    layer, and vias).  Having no inductance, a link's current is the
    potential difference over its resistance, so the links are eliminated
    into a conductance ``Y`` on the element potentials (Kirchhoff's law
    becomes ``D^T I + Y phi = s``).  Returns ``Z`` (dense, in-plane dofs),
    ``D`` (dense), the global port of each dof (-1 inside), the dof and
    element offset of each sheet, and ``Y`` (sparse).
    """
    sizes = [s_.space.size for s_ in sheets]
    elems = [len(s_.space.kind) for s_ in sheets]
    dof0 = np.r_[0, np.cumsum(sizes)]
    el0 = np.r_[0, np.cumsum(elems)]
    m = dof0[-1]
    n = el0[-1]
    omega = 2.0 * math.pi * frequency
    dtype = np.complex128 if frequency > 0 else np.float64
    z = np.zeros((m, m), dtype=dtype)
    d = np.zeros((m, n))
    ports = np.full(m, -1)
    cache: dict[tuple[int, int, float], np.ndarray] = {}
    for a, sa in enumerate(sheets):
        sl = slice(dof0[a], dof0[a + 1])
        z[sl, sl] += sa.space.resistance(sa.resistivity / sa.thickness)
        d[sl, el0[a]:el0[a + 1]] = sa.space.divergence().toarray()
        port = sa.space.port_of_dof
        mapping = np.asarray(sa.port_ids if sa.port_ids else range(max(port.max() + 1, 0)), dtype=np.int64)
        ports[sl] = np.where(port >= 0, mapping[np.maximum(port, 0)] if len(mapping) else -1, -1)
        if frequency > 0:
            for b in range(a, len(sheets)):
                sb = sheets[b]
                gap = thickness_gap(sa.thickness, sa.z, sb.thickness, sb.z)
                key = (id(sa.space), id(sb.space), round(sa.thickness, 12), round(sb.thickness, 12), round(abs(sa.z - sb.z), 12))
                if key not in cache:
                    nodes = thickness_nodes(sa.thickness, sa.z, sb.thickness, sb.z)
                    cache[key] = inductance(sa.space, gap, other=sb.space, heights=nodes)
                block = 1j * omega * cache[key]
                z[sl, dof0[b]:dof0[b + 1]] += block
                if b != a:
                    z[dof0[b]:dof0[b + 1], sl] += block.T
    rows, cols, vals = [], [], []
    for a, b, ea, eb, resistance in links:
        g = 1.0 / np.broadcast_to(np.asarray(resistance, dtype=np.float64), np.shape(ea))
        ia, ib = el0[a] + np.asarray(ea), el0[b] + np.asarray(eb)
        rows += [ia, ib, ia, ib]
        cols += [ia, ib, ib, ia]
        vals += [g, g, -g, -g]
    y = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)) if rows else sp.csr_matrix((n, n))
    return z, d, ports, dof0, el0, y


def solve_driven(z: np.ndarray, d: np.ndarray, ports: np.ndarray, fixed: dict[int, complex], floating: list[int],
                 y: sp.spmatrix | None = None):
    """Ports in ``fixed`` held at their voltage; the ports in ``floating`` share one free potential
    (a short closing the loop).  Returns dof currents, element potentials and the current each
    fixed port drives into the conductor."""
    m, n = d.shape
    extra = 1 if floating else 0
    dtype = np.result_type(z.dtype, np.complex128)
    size = m + n + extra
    system = np.zeros((size, size), dtype=dtype)
    system[:m, :m] = z
    system[:m, m:m + n] = -d
    system[m:m + n, :m] = d.T
    if y is not None:
        system[m:m + n, m:m + n] += y.toarray()
    rhs = np.zeros(size, dtype=dtype)
    for p, v in fixed.items():
        rhs[:m][ports == p] = -v
    if floating:
        on = np.isin(ports, floating)
        system[:m, m + n][on] = 1.0  # Z I - D phi + V_short = 0 on its dofs
        system[m + n, :m][on] = 1.0  # no net current into the short
    # NumPy's LAPACK is the 64-bit-integer OpenBLAS: SciPy's 32-bit one
    # refuses matrices past 2^31 entries ("Memory error in scipy.linalg.solve").
    x = np.linalg.solve(system, rhs)
    current = x[:m]
    driven = {p: -current[ports == p].sum() for p in fixed}
    return current, x[m:m + n], driven
