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


def _fit_band(pts: np.ndarray, band: float, minimum: float) -> np.ndarray:
    """Drop vertices until every edge keeps ``minimum`` length when offset ``band`` inward.

    An edge offset by ``d`` into the copper changes length by
    ``-d (tan(a0/2) + tan(a1/2))``, ``a`` the turn at each end (positive at a
    convex corner).  A short edge beside a sharp convex corner -- the first
    chord of an arc meeting a straight side -- collapses; its vertex away from
    the corner is removed, which replaces two chords of the arc by one.
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
        worst = int(np.argmin(left))
        if left[worst] >= minimum:
            return pts
        a, b = worst, (worst + 1) % len(pts)
        drop = a if abs(turn[a]) < abs(turn[b]) else b
        pts = np.delete(pts, drop, axis=0)
    raise ValueError("the outline is too small for the band")


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
            if denom < 0.2:
                raise ValueError("a corner sharper than ~ 100 degrees cannot be mitred")
            corners[i] = self.vertices[i] + distance * (n1 + n2) / denom
        out = []
        for i in range(n):
            a, b = corners[i], corners[(i + 1) % n]
            if distance > 0:
                if np.dot(b - a, self.tangent[i]) <= 0.0:
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
) -> HybridMesh:
    """Mesh ``polygon`` (metres) as band triangles, gap triangles and grid squares."""
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets[0] != 0.0 or np.any(np.diff(offsets) <= 0):
        raise ValueError("offsets must start at 0 and increase")
    polygon = orient(polygon, sign=1.0)  # exterior counter-clockwise, holes clockwise: copper on the left
    tol = 1e-9 * max(polygon.bounds[2] - polygon.bounds[0], polygon.bounds[3] - polygon.bounds[1])
    band = float(offsets[-1])
    fitted = [_fit_band(_clean_loop(np.asarray(polygon.exterior.coords), tol), band, 0.25 * along)]
    fitted += [_fit_band(_clean_loop(np.asarray(r.coords), tol), band, 0.25 * along) for r in polygon.interiors]
    meshed = Polygon(fitted[0], holes=fitted[1:])
    loops = [_Loop(v, along) for v in fitted]
    clearance = 0.15 * pitch if clearance is None else clearance
    removed: set[tuple[int, int]] = set()

    for _round in range(max_rounds):
        rings = [[loop.ring(d) for d in offsets] for loop in loops]  # [loop][row] -> (n, 2)
        inner_loops = [r[-1] for r in rings]
        inner = Polygon(inner_loops[0], holes=inner_loops[1:])
        if not inner.is_valid:
            raise ValueError("the innermost ring is not a simple polygon: reduce the band width")
        squares = _squares(inner.buffer(-clearance, join_style="mitre"), pitch, origin, removed)
        result = _assemble(loops, rings, squares, pitch, origin, inner)
        if isinstance(result, HybridMesh):
            return dataclasses.replace(result, outline_area_change=meshed.area / polygon.area - 1.0)
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


def _assemble(loops, rings, squares, pitch, origin, inner):
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
        for row in loop_rings:
            start = len(vertices)
            vertices.extend(row)
            ids.append(np.arange(start, start + len(row)))
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
