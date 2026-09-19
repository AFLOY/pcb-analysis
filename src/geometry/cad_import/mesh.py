"""Tessellated solids and point-in-solid tests by the generalized winding number.

OpenCASCADE classifies one point at a time through Python, about 25 µs per
point; a board sampled at 0.1 mm with three samples per axis asks for
millions of points.  Tessellating each solid once (``BRepMesh``) turns the
test into a sum of signed solid angles over a few hundred triangles, which
the C++ module evaluates over all points in parallel and NumPy evaluates in
chunks when the module is not built.  Planar solids tessellate exactly;
curved faces carry the linear deflection the tessellation was asked for.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np

try:  # pragma: no cover - depends on the local build
    from . import _voxelize_native as _native
except ImportError:  # pragma: no cover
    _native = None


ClassifyMethod = Literal["auto", "occ", "numpy", "native", "section"]


def native_available() -> bool:
    return _native is not None


def native_threads() -> int:
    value = os.environ.get("PCB_NATIVE_THREADS")
    if value:
        return max(1, int(value))
    return 1


def default_method() -> ClassifyMethod:
    """``PCB_GEOMETRY_CLASSIFY`` overrides; else native when built, else NumPy."""

    flag = os.environ.get("PCB_GEOMETRY_CLASSIFY", "").strip().lower()
    if flag in ("occ", "numpy", "native", "section"):
        return flag  # type: ignore[return-value]
    return "native" if native_available() else "numpy"


def default_plane_method() -> ClassifyMethod:
    """Layer sampling: the exact section rasteriser when built, else the point path."""

    flag = os.environ.get("PCB_GEOMETRY_CLASSIFY", "").strip().lower()
    if flag in ("occ", "numpy", "native", "section"):
        return flag  # type: ignore[return-value]
    return "section" if native_available() else "numpy"


@dataclass(frozen=True)
class TriangleMesh:
    """Outward-oriented triangles ``(n, 3, 3)`` of one closed solid, in metres."""

    triangles_m: np.ndarray
    deflection_m: float

    def __post_init__(self) -> None:
        triangles = np.ascontiguousarray(self.triangles_m, dtype=np.float64)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or triangles.shape[0] < 4:
            raise ValueError("triangles_m must have shape (n >= 4, 3, 3)")
        if not np.all(np.isfinite(triangles)):
            raise ValueError("triangle vertices must be finite")
        object.__setattr__(self, "triangles_m", triangles)

    @property
    def size(self) -> int:
        return int(self.triangles_m.shape[0])

    @property
    def bounds_m(self) -> tuple[np.ndarray, np.ndarray]:
        flat = self.triangles_m.reshape(-1, 3)
        return flat.min(axis=0), flat.max(axis=0)

    def signed_volume_m3(self) -> float:
        """Positive for an outward-oriented closed surface (divergence theorem)."""

        a, b, c = self.triangles_m[:, 0], self.triangles_m[:, 1], self.triangles_m[:, 2]
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    def is_closed(self, tolerance: float = 1.0e-9) -> bool:
        """Every directed edge appears once with its reverse (watertight, consistent)."""

        tri = self.triangles_m
        edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=0)
        scale = max(float(np.max(np.abs(tri))), 1.0)
        key = np.round(edges / (scale * tolerance)).astype(np.int64)
        forward = {tuple(row.reshape(-1)) for row in key}
        reverse = {tuple(row[::-1].reshape(-1)) for row in key}
        return forward == reverse and len(forward) == edges.shape[0]

    # ------------------------------------------------------------ queries
    def winding_numbers(self, points_m: np.ndarray, *, method: ClassifyMethod = "auto", threads: int | None = None) -> np.ndarray:
        points = np.ascontiguousarray(points_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_m must have shape (n, 3)")
        chosen = default_method() if method == "auto" else method
        if chosen == "native":
            if _native is None:
                raise ImportError(
                    "the geometry native extension is not built; run cmake or "
                    "python -m geometry.cad_import.native.build"
                )
            return np.asarray(_native.winding_numbers(points, self.triangles_m, threads or native_threads()))
        if chosen == "numpy":
            return winding_numbers_numpy(points, self.triangles_m)
        raise ValueError("winding numbers are computed by 'numpy' or 'native', not 'occ'")

    def contains(self, points_m: np.ndarray, *, method: ClassifyMethod = "auto", threads: int | None = None, threshold: float = 0.5) -> np.ndarray:
        points = np.ascontiguousarray(points_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_m must have shape (n, 3)")
        chosen = default_method() if method == "auto" else method
        if chosen == "native":
            if _native is None:
                raise ImportError(
                    "the geometry native extension is not built; run cmake or "
                    "python -m geometry.cad_import.native.build"
                )
            return np.asarray(_native.contains(points, self.triangles_m, threshold, threads or native_threads()), dtype=bool)
        lo, hi = self.bounds_m
        inside = np.all((points >= lo) & (points <= hi), axis=1)
        candidates = np.nonzero(inside)[0]
        if candidates.size:
            inside[candidates] = winding_numbers_numpy(points[candidates], self.triangles_m) >= threshold
        return inside


    # ------------------------------------------------------------ sections
    def section_segments(self, z_m: float) -> np.ndarray:
        """Oriented segments ``(m, 2, 2)`` of the surface cut at ``z``, counter-clockwise around the solid."""

        if _native is not None:
            return np.asarray(_native.section_segments(self.triangles_m, float(z_m)))
        return section_segments_numpy(self.triangles_m, float(z_m))


def plane_section_coverage(
    meshes: "list[TriangleMesh]",
    z_m: float,
    *,
    origin_m: tuple[float, float],
    pitch_m: float,
    shape: tuple[int, int],
) -> np.ndarray:
    """Exact fraction of every ``(row, col)`` cell covered by the solids' section at ``z``.

    Rows count upwards from ``origin_m``.  The solids of one layer touch but
    do not overlap, so their coverages add; the sum is clamped to one.
    Needs the native extension; ``section_segments_numpy`` gives the segments
    for checks but no NumPy rasteriser is provided.
    """

    if _native is None:
        raise ImportError(
            "the section rasteriser needs the geometry native extension; run cmake or "
            "python -m geometry.cad_import.native.build"
        )
    rows, cols = (int(axis) for axis in shape)
    sets = []
    for mesh in meshes:
        lo, hi = mesh.bounds_m
        if lo[2] <= z_m <= hi[2]:
            sets.append(mesh.triangles_m)
    if not sets:
        return np.zeros((rows, cols))
    return np.asarray(
        _native.plane_section_coverage(sets, float(z_m), float(origin_m[0]), float(origin_m[1]), float(pitch_m), rows, cols)
    )


def section_segments_numpy(triangles_m: np.ndarray, z_m: float) -> np.ndarray:
    """Reference implementation of the oriented plane section, ``(m, 2, 2)``."""

    tri = np.asarray(triangles_m, dtype=np.float64)
    above = tri[:, :, 2] > z_m
    crossing = np.any(above, axis=1) & ~np.all(above, axis=1)
    out = []
    for t, flags in zip(tri[crossing], above[crossing]):
        pts = []
        for i in range(3):
            p, q = t[i], t[(i + 1) % 3]
            if flags[i] == flags[(i + 1) % 3]:
                continue
            f = (z_m - p[2]) / (q[2] - p[2])
            pts.append(p[:2] + f * (q[:2] - p[:2]))
        if len(pts) < 2:
            continue
        n = np.cross(t[1] - t[0], t[2] - t[0])
        tangent = np.array([-n[1], n[0]])
        p0, p1 = pts[0], pts[1]
        if np.dot(p1 - p0, tangent) < 0.0:
            p0, p1 = p1, p0
        out.append([p0, p1])
    return np.asarray(out, dtype=np.float64).reshape(-1, 2, 2)


def winding_numbers_numpy(points_m: np.ndarray, triangles_m: np.ndarray, *, chunk: int = 4096) -> np.ndarray:
    """Solid angle sum over the triangles divided by ``4π``, chunked over points."""

    points = np.asarray(points_m, dtype=np.float64)
    triangles = np.asarray(triangles_m, dtype=np.float64)
    out = np.empty(points.shape[0])
    a0, b0, c0 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    for start in range(0, points.shape[0], chunk):
        p = points[start : start + chunk][:, None, :]
        a = a0[None] - p
        b = b0[None] - p
        c = c0[None] - p
        la = np.linalg.norm(a, axis=2)
        lb = np.linalg.norm(b, axis=2)
        lc = np.linalg.norm(c, axis=2)
        numerator = np.einsum("ijk,ijk->ij", a, np.cross(b, c))
        denominator = (
            la * lb * lc
            + np.einsum("ijk,ijk->ij", a, b) * lc
            + np.einsum("ijk,ijk->ij", a, c) * lb
            + np.einsum("ijk,ijk->ij", b, c) * la
        )
        out[start : start + chunk] = 2.0 * np.arctan2(numerator, denominator).sum(axis=1) / (4.0 * np.pi)
    return out


__all__ = ["ClassifyMethod", "TriangleMesh", "default_method", "default_plane_method", "native_available", "native_threads", "plane_section_coverage", "section_segments_numpy", "winding_numbers_numpy"]
