"""A sheet conductor's outline as a band of triangles around coarse squares (experiment).

The copper polygon (shapely, metres, holes allowed) is meshed in three parts:

* **the band**: rings offset inward from every boundary loop by the distances
  ``offsets`` (graded from a fraction of the skin depth to the band width),
  each ring a mitred copy of the loop sampled at the same points, so two
  consecutive rings bound a row of quadrilaterals, each cut into two
  triangles.  The rows are thin across the edge and ``along`` long, which is
  how the current crowding at an edge is resolved without isotropic fine
  elements;
* **the squares**: cells of a uniform grid of ``pitch`` (aligned to the
  origin) that lie inside the innermost ring with ``clearance`` to spare; the
  far field of these is what a uniform FFT grid can carry;
* **the gap** between the innermost ring and the squares: a Delaunay
  triangulation of the ring's points and the squares' boundary corners, kept
  inside the gap, refined until every ring segment and square edge is an edge
  of it (a missing ring segment is split in every ring, a square whose edge
  is missing is given back to the gap).

The result is an :class:`~.hybrid_rt0.RT0Space`.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.prepared import prep
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient

from .hybrid_rt0 import RT0Space


@dataclass(frozen=True)
class HybridMesh:
    space: RT0Space
    band_elements: np.ndarray  # indices of the band's triangles
    gap_elements: np.ndarray
    square_elements: np.ndarray
    row_of_element: np.ndarray  # band row (0 at the outline), -1 elsewhere
    outline_area_change: float = 0.0  # meshed area / given area - 1 (vertices dropped by _fit_band)
    band: float = 0.0  # the band width the mesh ended with (halved where it did not fit)


def _clean_loop(coords: np.ndarray, tolerance: float) -> np.ndarray:
    """Open loop without repeated or collinear vertices."""
    pts = np.asarray(coords, dtype=np.float64)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    changed = True
    while changed and len(pts) > 3:
        changed = False
        keep = []
        n = len(pts)
        for i in range(n):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
            ab, bc = b - a, c - b
            if np.linalg.norm(ab) <= tolerance:
                changed = True
                continue
            cross = ab[0] * bc[1] - ab[1] * bc[0]
            if abs(cross) <= tolerance * (np.linalg.norm(ab) + np.linalg.norm(bc)) and np.dot(ab, bc) > 0:
                changed = True
                continue
            keep.append(i)
        pts = pts[keep]
    return pts


def _fit_band(pts: np.ndarray, band: float, minimum: float = 0.1, gentle: float = math.radians(30.0)) -> np.ndarray:
    """Merge chords of curves that the band would collapse (best effort).

    ``minimum``: the fraction of an edge's length it must keep once offset.

    An edge offset by ``d`` into the copper changes length by
    ``-d (tan(a0/2) + tan(a1/2))``, ``a`` the turn at each end (positive at a
    convex corner).  A short chord of an arc beside a corner collapses; its
    vertex with the gentler turn (below ``gentle``) is removed, replacing two
    chords by one.  Sharp corners are never removed: the ring caps their
    offset instead (:meth:`_Loop.ring`).
    """
    pts = np.asarray(pts, dtype=np.float64)
    while len(pts) > 3:
        edge = np.roll(pts, -1, axis=0) - pts
        length = np.linalg.norm(edge, axis=1)
        t = edge / length[:, None]
        prev = np.roll(t, 1, axis=0)
        turn = np.arctan2(prev[:, 0] * t[:, 1] - prev[:, 1] * t[:, 0], np.sum(prev * t, axis=1))  # at vertex i
        shrink = band * (np.tan(turn / 2) + np.tan(np.roll(turn, -1) / 2))
        left = length - shrink
        order = np.argsort(left / length)
        dropped = False
        for worst in order:
            if left[worst] >= minimum * length[worst]:
                break
            a, b = int(worst), int((worst + 1) % len(pts))
            candidates = [v for v in (a, b) if abs(turn[v]) < gentle]
            if candidates:
                drop = min(candidates, key=lambda v: abs(turn[v]))
                pts = np.delete(pts, drop, axis=0)
                dropped = True
                break
        if not dropped:
            return pts
    return pts


class _Loop:
    """One boundary loop, copper on its left, sampled and offset by mitred joins."""

    def __init__(self, vertices: np.ndarray, along: float) -> None:
        self.vertices = vertices
        n = len(vertices)
        edge = np.roll(vertices, -1, axis=0) - vertices
        self.length = np.linalg.norm(edge, axis=1)
        self.tangent = edge / self.length[:, None]
        self.normal = np.column_stack([-self.tangent[:, 1], self.tangent[:, 0]])  # left = copper
        # samples per edge: fractions in [0, 1), the vertex first
        self.fractions = [list(np.arange(max(1, math.ceil(self.length[i] / along))) / max(1, math.ceil(self.length[i] / along))) for i in range(n)]
        # The mitre moves a corner along each edge by d tan(turn/2); capping
        # the offset there keeps every edge at least 10 % of its length, so a
        # sharp corner (a thermal spoke, a zone sliver) tapers the band's rows
        # instead of collapsing them.
        prev = np.roll(self.tangent, 1, axis=0)
        turn = np.arctan2(prev[:, 0] * self.tangent[:, 1] - prev[:, 1] * self.tangent[:, 0], np.sum(prev * self.tangent, axis=1))
        slide = np.abs(np.tan(turn / 2))
        room = 0.45 * np.minimum(self.length, np.roll(self.length, 1))
        with np.errstate(divide="ignore"):
            self.cap = np.where(slide > 1e-12, room / slide, np.inf)

    def split(self, edge: int, index: int) -> None:
        """Halve sample segment ``index`` of edge ``edge``."""
        f = self.fractions[edge]
        nxt = f[index + 1] if index + 1 < len(f) else 1.0
        f.insert(index + 1, 0.5 * (f[index] + nxt))

    def ring(self, distance: float) -> np.ndarray:
        """Points of the loop offset ``distance`` into the copper; same count at any distance."""
        n = len(self.vertices)
        corners = np.empty((n, 2))
        for i in range(n):
            n1, n2 = self.normal[i - 1], self.normal[i]
            denom = 1.0 + float(np.dot(n1, n2))
            if denom < 1e-6:
                raise ValueError("a corner folds back on itself")
            corners[i] = self.vertices[i] + min(distance, self.cap[i]) * (n1 + n2) / denom
        out = []
        for i in range(n):
            a, b = corners[i], corners[(i + 1) % n]
            if distance > 0 and np.dot(b - a, self.tangent[i]) <= 0.0:
                raise ValueError(f"offset {distance:g} m collapses an edge of {self.length[i]:g} m")
            for t in self.fractions[i]:
                out.append(a + t * (b - a))
        return np.array(out)

    def sample_index(self) -> list[tuple[int, int]]:
        return [(i, k) for i in range(len(self.vertices)) for k in range(len(self.fractions[i]))]


def build_hybrid_mesh(
    polygon: Polygon,
    *,
    pitch: float,
    offsets: list[float] | np.ndarray,
    along: float,
    clearance: float | None = None,
    origin: tuple[float, float] = (0.0, 0.0),
    max_rounds: int = 30,
    outline_tolerance: float = 2e-6,
    gap_size: float | None = None,
) -> HybridMesh:
    """Mesh ``polygon`` (metres) as band triangles, gap triangles and grid squares.

    Each boundary loop keeps its own band: at most 0.4 of its distance to the
    nearest other loop (a drill hole among others gets a narrower one, the
    outline keeps its own), halved where its innermost ring still crosses.
    The outline is first simplified to ``outline_tolerance`` (2 um: a 0.2 mm
    drill becomes a polygon of about twenty sides), and the gap between the
    band and the squares gets points every ``gap_size`` (a quarter of the
    pitch) so that the copper between close vias is not a few long triangles.
    """
    return _build(polygon, pitch=pitch, offsets=offsets, along=along, clearance=clearance, origin=origin,
                  max_rounds=max_rounds, outline_tolerance=outline_tolerance, gap_size=gap_size)


def _loop_bands(fitted: list[np.ndarray], band: float, share: float = 0.4) -> list[float]:
    """Each loop's band: at most ``share`` of its distance to the nearest other loop."""
    from shapely.geometry import LinearRing

    rings = [LinearRing(v) for v in fitted]
    tree = shapely.STRtree(rings)
    out = []
    for k, ring in enumerate(rings):
        nearest = band / share
        for j in tree.query(ring.buffer(band / share)):
            if j != k:
                nearest = min(nearest, ring.distance(rings[j]))
        out.append(min(band, share * nearest))
    return out


def _build(polygon, *, pitch, offsets, along, clearance, origin, max_rounds, outline_tolerance=2e-6,
           gap_size=None) -> HybridMesh:
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets[0] != 0.0 or np.any(np.diff(offsets) <= 0):
        raise ValueError("offsets must start at 0 and increase")
    polygon = orient(polygon, sign=1.0)  # exterior counter-clockwise, holes clockwise: copper on the left
    if outline_tolerance > 0.0:
        # chords of arcs (drill holes, rounded pads) within the tolerance of the arc
        polygon = orient(polygon.simplify(outline_tolerance, preserve_topology=True), sign=1.0)
    tol = 1e-9 * max(polygon.bounds[2] - polygon.bounds[0], polygon.bounds[3] - polygon.bounds[1])
    band = float(offsets[-1])
    cleaned = [_clean_loop(np.asarray(polygon.exterior.coords), tol)]
    cleaned += [_clean_loop(np.asarray(r.coords), tol) for r in polygon.interiors]
    bands = _loop_bands(cleaned, band)
    fitted = [_fit_band(v, b) for v, b in zip(cleaned, bands)]
    meshed = Polygon(fitted[0], holes=fitted[1:])
    loops = [_Loop(v, along) for v in fitted]
    clearance = 0.15 * pitch if clearance is None else clearance
    gap_size = 0.25 * pitch if gap_size is None else gap_size
    removed: set[tuple[int, int]] = set()

    def loop_offsets(b):
        kept = offsets[offsets < b * (1 - 1e-9)]
        return np.r_[kept, b] if len(kept) else np.array([0.0, b])

    for _round in range(max_rounds):
        for _attempt in range(8):
            try:
                rings = [[loop.ring(d) for d in loop_offsets(b)] for loop, b in zip(loops, bands)]
            except ValueError:
                bands = [b * 0.5 for b in bands]  # an edge collapses somewhere: narrow every loop once more
                continue
            inner_loops = [r[-1] for r in rings]
            bad = [k for k, ring in enumerate(inner_loops) if not Polygon(ring).is_valid]
            inner = Polygon(inner_loops[0], holes=inner_loops[1:])
            if not bad and not inner.is_valid:
                # rings that cross another ring
                from shapely.geometry import LinearRing

                lr = [LinearRing(r) for r in inner_loops]
                tree = shapely.STRtree(lr)
                bad = sorted({int(k) for k in range(len(lr)) for j in tree.query(lr[k]) if j != k and lr[k].intersects(lr[j])})
            if not bad and inner.is_valid:
                break
            for k in bad:
                bands[k] *= 0.5
        else:
            raise ValueError("the innermost rings do not form a simple polygon")
        squares = _squares(inner.buffer(-clearance, join_style="mitre"), pitch, origin, removed)
        result = _assemble(loops, rings, squares, pitch, origin, inner, gap_size)
        if isinstance(result, HybridMesh):
            return dataclasses.replace(result, outline_area_change=meshed.area / polygon.area - 1.0,
                                       band=float(min(bands)))
        missing_rings, missing_squares = result
        for li, ei, si in missing_rings:
            loops[li].split(ei, si)
        removed |= missing_squares
    raise RuntimeError("the gap triangulation did not conform within max_rounds")


def _squares(region, pitch, origin, removed) -> list[tuple[int, int]]:
    if region.is_empty:
        return []
    x0, y0 = origin
    minx, miny, maxx, maxy = region.bounds
    i0, i1 = math.floor((minx - x0) / pitch), math.ceil((maxx - x0) / pitch)
    j0, j1 = math.floor((miny - y0) / pitch), math.ceil((maxy - y0) / pitch)
    cells = []
    prepared = prep(region)
    for i in range(i0, i1):
        for j in range(j0, j1):
            if (i, j) in removed:
                continue
            cell = box(x0 + i * pitch, y0 + j * pitch, x0 + (i + 1) * pitch, y0 + (j + 1) * pitch)
            if prepared.contains(cell):
                cells.append((i, j))
    return cells


def _assemble(loops, rings, squares, pitch, origin, inner, gap_size=None):
    x0, y0 = origin
    vertices: list[np.ndarray] = []
    corners: list[list[int]] = []
    kinds: list[int] = []
    group: list[int] = []  # 0 band, 1 gap, 2 square
    rows: list[int] = []

    # Band vertices: rings[loop][row][k]
    ring_ids = []
    for li, loop_rings in enumerate(rings):
        ids = []
        # one vertex per position across the loop's rows: a capped corner
        # stays put from the row where its cap binds
        seen: dict[tuple[float, float], int] = {}
        for row in loop_rings:
            row_ids = []
            for pt in row:
                key = (float(pt[0]), float(pt[1]))
                if key not in seen:
                    seen[key] = len(vertices)
                    vertices.append(np.asarray(pt))
                row_ids.append(seen[key])
            ids.append(np.array(row_ids))
        ring_ids.append(ids)
    for li, ids in enumerate(ring_ids):
        for row in range(len(ids) - 1):
            a_ids, b_ids = ids[row], ids[row + 1]
            n = len(a_ids)
            for k in range(n):
                a0, a1 = a_ids[k], a_ids[(k + 1) % n]
                b0, b1 = b_ids[k], b_ids[(k + 1) % n]
                pa0, pa1, pb0, pb1 = (np.asarray(vertices[v]) for v in (a0, a1, b0, b1))
                # the shorter diagonal
                if np.linalg.norm(pb1 - pa0) <= np.linalg.norm(pb0 - pa1):
                    tris = ([a0, a1, b1], [a0, b1, b0])
                else:
                    tris = ([a0, a1, b0], [a1, b1, b0])
                for t in tris:
                    if len(set(t)) < 3:
                        continue  # a capped corner: the quad is a triangle
                    corners.append(t + [-1])
                    kinds.append(0)
                    group.append(0)
                    rows.append(row)

    # Squares: corner vertices by grid index
    grid_id: dict[tuple[int, int], int] = {}

    def gid(i, j):
        key = (i, j)
        if key not in grid_id:
            grid_id[key] = len(vertices)
            vertices.append(np.array([x0 + i * pitch, y0 + j * pitch]))
        return grid_id[key]

    square_set = set(squares)
    for i, j in squares:
        corners.append([gid(i, j), gid(i + 1, j), gid(i + 1, j + 1), gid(i, j + 1)])
        kinds.append(1)
        group.append(2)
        rows.append(-1)

    # The gap's boundary segments: innermost ring segments and the squares' outer edges.
    required: list[tuple[int, int, object]] = []
    for li, ids in enumerate(ring_ids):
        inner_ids = ids[-1]
        samples = loops[li].sample_index()
        n = len(inner_ids)
        for k in range(n):
            if inner_ids[k] != inner_ids[(k + 1) % n]:
                required.append((inner_ids[k], inner_ids[(k + 1) % n], ("ring", li, samples[k])))
    for i, j in squares:
        for (di, dj), (a, b) in {
            (0, -1): ((i, j), (i + 1, j)),
            (1, 0): ((i + 1, j), (i + 1, j + 1)),
            (0, 1): ((i + 1, j + 1), (i, j + 1)),
            (-1, 0): ((i, j + 1), (i, j)),
        }.items():
            if (i + di, j + dj) not in square_set:
                required.append((gid(*a), gid(*b), ("square", (i, j))))

    point_ids = sorted({v for a, b, _ in required for v in (a, b)})
    pts = np.array([vertices[v] for v in point_ids])
    if gap_size:
        # Steiner points inside the gap on a lattice, kept half a step clear of its boundary
        gap_region = inner
        if squares:
            gap_region = inner.difference(shapely.union_all([box(x0 + i * pitch, y0 + j * pitch, x0 + (i + 1) * pitch,
                                                                   y0 + (j + 1) * pitch) for i, j in squares]))
        minx, miny, maxx, maxy = gap_region.bounds
        gx = np.arange(minx + gap_size / 2, maxx, gap_size)
        gy = np.arange(miny + gap_size / 2, maxy, gap_size)
        if len(gx) and len(gy):
            X, Y = np.meshgrid(gx, gy, indexing="ij")
            cand = np.column_stack([X.ravel(), Y.ravel()])
            shrunk = gap_region.buffer(-0.5 * gap_size)
            keep = shapely.contains_xy(shrunk, cand[:, 0], cand[:, 1]) if not shrunk.is_empty else np.zeros(len(cand), bool)
            # A square's edge stays an edge of the triangulation only while the
            # disc on it as diameter holds no other point: keep those discs clear.
            mids = [0.5 * (np.asarray(vertices[a]) + np.asarray(vertices[b])) for a, b, tag in required if tag[0] == "square"]
            apex = []
            if mids and np.any(keep):
                from scipy.spatial import cKDTree

                near = cKDTree(np.array(mids)).query(cand[keep], k=1)[0] < 0.55 * pitch
                idx = np.flatnonzero(keep)
                keep[idx[near]] = False
                # one point facing each free square edge, just outside its disc,
                # so the triangle on it is near isosceles instead of a fan
                for (a, b, tag) in required:
                    if tag[0] != "square":
                        continue
                    pa, pb = np.asarray(vertices[a]), np.asarray(vertices[b])
                    t = (pb - pa) / np.linalg.norm(pb - pa)
                    outward = np.array([t[1], -t[0]])  # square corners run counter-clockwise
                    apex.append(0.5 * (pa + pb) + 0.6 * pitch * outward)
                apex = np.array(apex)
                ok = shapely.contains_xy(shrunk, apex[:, 0], apex[:, 1]) if not shrunk.is_empty else np.zeros(len(apex), bool)
                apex = apex[ok]
                if len(apex):
                    # no lattice point closer than half a step to an apex
                    far = cKDTree(apex).query(cand, k=1)[0] > 0.5 * gap_size
                    keep &= far
                    apex = np.unique(np.round(apex / 1e-9) * 1e-9, axis=0)
            extra = cand[keep]
            if len(apex):
                extra = np.vstack([extra, apex]) if len(extra) else apex
            if len(extra):
                start = len(vertices)
                vertices.extend(extra)
                point_ids = point_ids + list(range(start, start + len(extra)))
                pts = np.vstack([pts, extra])
    gap = inner
    if squares:
        gap = inner.difference(shapely.union_all([box(x0 + i * pitch, y0 + j * pitch, x0 + (i + 1) * pitch, y0 + (j + 1) * pitch) for i, j in squares]))
    tri = Delaunay(pts)
    simplices = np.asarray(tri.simplices)
    centre = pts[simplices].mean(axis=1)
    inside = shapely.contains_xy(gap, centre[:, 0], centre[:, 1])
    tp = pts[simplices]
    area2 = np.abs((tp[:, 1, 0] - tp[:, 0, 0]) * (tp[:, 2, 1] - tp[:, 0, 1]) - (tp[:, 1, 1] - tp[:, 0, 1]) * (tp[:, 2, 0] - tp[:, 0, 0]))
    longest = np.max(np.linalg.norm(tp - np.roll(tp, 1, axis=1), axis=2), axis=1)
    flat = area2 <= 1e-9 * longest * longest  # collinear points on a straight ring
    simplices = simplices[inside & ~flat]
    edge_set = set()
    for s in simplices:
        for a, b in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
            ga, gb = point_ids[a], point_ids[b]
            edge_set.add((min(ga, gb), max(ga, gb)))
    missing_rings, missing_squares = [], set()
    for a, b, tag in required:
        if (min(a, b), max(a, b)) not in edge_set:
            if tag[0] == "ring":
                li, (ei, si) = tag[1], tag[2]
                missing_rings.append((li, ei, si))
            else:
                missing_squares.add(tag[1])
    if missing_rings or missing_squares:
        # split each missing ring segment once, the last sample index first
        missing_rings = sorted(set(missing_rings), key=lambda t: (t[0], t[1], -t[2]))
        return missing_rings, missing_squares

    for s in simplices:
        ids = [point_ids[v] for v in s]
        a, b, c = (np.asarray(vertices[v]) for v in ids)
        if (b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0] < 0:
            ids = [ids[0], ids[2], ids[1]]
        corners.append(ids + [-1])
        kinds.append(0)
        group.append(1)
        rows.append(-1)

    # Orient the band's triangles counter-clockwise too.
    verts = np.array(vertices)
    corners_arr = np.array(corners, dtype=np.int64)
    kinds_arr = np.array(kinds)
    for e in np.flatnonzero(kinds_arr == 0):
        a, b, c = verts[corners_arr[e, :3]]
        if (b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0] < 0:
            corners_arr[e, [1, 2]] = corners_arr[e, [2, 1]]
    space = RT0Space(verts, corners_arr, kinds_arr)
    group_arr = np.array(group)
    return HybridMesh(
        space=space,
        band_elements=np.flatnonzero(group_arr == 0),
        gap_elements=np.flatnonzero(group_arr == 1),
        square_elements=np.flatnonzero(group_arr == 2),
        row_of_element=np.array(rows),
    )


def graded_offsets(first: float, growth: float, width: float) -> np.ndarray:
    """0, first, first*(1+growth), ... up to ``width`` (the last row stretched to end there)."""
    out = [0.0]
    step = first
    while out[-1] + step < width * (1 - 1e-9):
        out.append(out[-1] + step)
        step *= growth
    if width - out[-1] < 0.5 * step / growth and len(out) > 1:
        out[-1] = width
    else:
        out.append(width)
    return np.array(out)


def mesh_region(region, **kwargs) -> HybridMesh:
    """Mesh every polygon of a (Multi)Polygon and join them into one space (disjoint pieces)."""
    parts = list(region.geoms) if hasattr(region, "geoms") else [region]
    meshes = [build_hybrid_mesh(p, **kwargs) for p in parts if p.area > 0.0]
    vertices, corners, kinds, groups, rows = [], [], [], [], []
    offset = 0
    for m in meshes:
        s_ = m.space
        c = s_.corners.copy()
        c[c >= 0] += offset
        vertices.append(s_.vertices)
        corners.append(c)
        kinds.append(s_.kind)
        g = np.zeros(len(s_.kind), dtype=np.int64)
        g[m.gap_elements] = 1
        g[m.square_elements] = 2
        groups.append(g)
        rows.append(m.row_of_element)
        offset += len(s_.vertices)
    space = RT0Space(np.vstack(vertices), np.vstack(corners), np.concatenate(kinds))
    group = np.concatenate(groups)
    return HybridMesh(space=space, band_elements=np.flatnonzero(group == 0), gap_elements=np.flatnonzero(group == 1),
                      square_elements=np.flatnonzero(group == 2), row_of_element=np.concatenate(rows))
