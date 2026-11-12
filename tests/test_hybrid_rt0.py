"""The RT0 hybrid sheet prototype: closed-form potentials, basis and DC resistance."""

from __future__ import annotations

import math

import numpy as np
import pytest

from electrical.sheet_peec.hybrid_rt0 import (
    GMD_SELF,
    RT0Space,
    polygon_potentials,
    solve_saddle,
    thickness_gap,
)


def _brute(point, polygon, h, n=400):
    """Midpoint rule on an n x n split of the polygon's bounding box, masked to the polygon."""
    from matplotlib.path import Path  # noqa: PLC0415 - test helper only

    lo, hi = polygon.min(axis=0), polygon.max(axis=0)
    xs = lo[0] + (np.arange(n) + 0.5) * (hi[0] - lo[0]) / n
    ys = lo[1] + (np.arange(n) + 0.5) * (hi[1] - lo[1]) / n
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    inside = Path(polygon).contains_points(pts)
    pts = pts[inside]
    da = (hi[0] - lo[0]) * (hi[1] - lo[1]) / n / n
    rel = pts - point
    r = np.sqrt(np.sum(rel * rel, axis=1) + h * h)
    return np.sum(1 / r) * da, np.sum(rel / r[:, None], axis=0) * da


TRIANGLE = np.array([[0.0, 0.0], [1.0, 0.1], [0.3, 0.8]])


@pytest.mark.parametrize("point,h", [((0.4, 0.3), 0.3), ((1.5, 0.9), 0.0), ((-0.2, 0.4), 0.05), ((0.6, 0.05), 0.2)])
def test_polygon_potentials_match_a_fine_quadrature(point, h) -> None:
    pytest.importorskip("matplotlib")
    p0, p1 = polygon_potentials(np.array([point]), TRIANGLE[None], h)
    b0, b1 = _brute(np.array(point), TRIANGLE, h, n=1200)
    assert p0[0] == pytest.approx(b0, rel=3e-3)
    np.testing.assert_allclose(p1[0], b1, rtol=3e-3, atol=3e-4 * abs(b0))


def test_the_potential_at_a_square_s_centre_is_closed_form() -> None:
    a = 0.7
    square = np.array([[-a, -a], [a, -a], [a, a], [-a, a]])
    p0, p1 = polygon_potentials(np.zeros((1, 2)), square[None], 0.0)
    assert p0[0] == pytest.approx(8 * a * math.log(1 + math.sqrt(2)), rel=1e-12)
    np.testing.assert_allclose(p1[0], 0.0, atol=1e-12)


def test_a_point_on_an_edge_or_a_vertex_is_finite() -> None:
    pts = np.array([[0.5, 0.05], [0.0, 0.0], [1.0, 0.1]])
    p0, p1 = polygon_potentials(pts, np.repeat(TRIANGLE[None], 3, axis=0), 0.0)
    assert np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))
    # continuous across the edge: a point just inside and just outside agree
    q = np.array([[0.5, 0.05 + 1e-7], [0.5, 0.05 - 1e-7]])
    r0, _ = polygon_potentials(q, np.repeat(TRIANGLE[None], 2, axis=0), 0.0)
    assert r0[0] == pytest.approx(r0[1], rel=1e-5)


def test_the_gmd_of_a_thickness_is_its_closed_form() -> None:
    assert thickness_gap(2e-5, 0.0, 2e-5, 0.0) == pytest.approx(GMD_SELF * 2e-5)
    # far apart the GMD tends to the centre distance
    assert thickness_gap(1e-5, 0.0, 1e-5, 1e-3) == pytest.approx(1e-3, rel=1e-4)
    # numerically against a double midpoint sum
    za = (np.arange(400) + 0.5) / 400 * 3e-5
    zb = 3e-5 + (np.arange(400) + 0.5) / 400 * 1e-5
    ln = np.mean(np.log(np.abs(za[:, None] - zb[None, :])))
    assert math.log(thickness_gap(3e-5, 1.5e-5, 1e-5, 3.5e-5)) == pytest.approx(ln, abs=2e-3)


def _strip_space(kind: str, length: int):
    """A length x 1 strip of unit squares, or of two triangles per square."""
    X, Y = np.meshgrid(np.arange(length + 1.0), np.arange(2.0), indexing="ij")
    vertices = np.column_stack([X.ravel(), Y.ravel()])

    def idx(i, j):
        return i * 2 + j

    corners, kinds = [], []
    for i in range(length):
        a, b, c, d = idx(i, 0), idx(i + 1, 0), idx(i + 1, 1), idx(i, 1)
        if kind == "quad":
            corners.append([a, b, c, d])
            kinds.append(1)
        else:
            corners += [[a, b, c, -1], [a, c, d, -1]]
            kinds += [0, 0]
    return RT0Space(vertices, np.array(corners), np.array(kinds))


def _end_to_end(space, sheet_resistance=1.0):
    """1 A into the leftmost element(s), out of the rightmost; the mean potential difference."""
    x = space.centroid[:, 0]
    first = np.flatnonzero(x < x.min() + 0.5)
    last = np.flatnonzero(x > x.max() - 0.5)
    injected = np.zeros(len(x))
    injected[first] = 1.0 / len(first)
    injected[last] = -1.0 / len(last)
    _, phi = solve_saddle(space.resistance(sheet_resistance), space.divergence(), injected)
    return injected @ phi  # power-consistent voltage for 1 A


@pytest.mark.parametrize("kind", ["quad", "tri"])
def test_one_more_square_of_strip_adds_one_square_of_resistance(kind) -> None:
    """RT0 holds the uniform field exactly, so away from the ends a unit square adds rho_s."""
    r3 = _end_to_end(_strip_space(kind, 3), 2.5)
    r5 = _end_to_end(_strip_space(kind, 5), 2.5)
    assert (r5 - r3) / 2 == pytest.approx(2.5, rel=1e-10)


def test_every_basis_function_carries_one_ampere_across_its_edge() -> None:
    space = _strip_space("tri", 3)
    poly = space.polygons()
    for i, (a, b) in enumerate(space.edges):
        p, q = space.vertices[a], space.vertices[b]
        t = (q - p) / np.linalg.norm(q - p)
        n = np.array([t[1], -t[0]])
        e0 = space.edge_elements[i, 0]
        if np.dot(space.centroid[e0] - p, n) > 0:
            n = -n  # pointing out of element [0]
        mid = 0.5 * (p + q)
        for side, e in enumerate(space.edge_elements[i]):
            block = space.coefficients[2 * i:2 * i + 2, 3 * e:3 * e + 3].toarray()
            k = block @ np.r_[1.0, mid - space.centroid[e]]
            assert k @ n * np.linalg.norm(q - p) == pytest.approx(1.0, rel=1e-12)
    del poly


def _sector(r1, r2, span, n=128):
    from shapely.geometry import Polygon

    th = np.linspace(0, span, n + 1)
    outer = np.column_stack([r2 * np.cos(th), r2 * np.sin(th)])
    inner = np.column_stack([r1 * np.cos(th[::-1]), r1 * np.sin(th[::-1])])
    return Polygon(np.vstack([outer, inner]))


def test_the_hybrid_mesh_covers_a_turned_strip_with_squares_inside() -> None:
    from shapely import affinity
    from shapely.geometry import Polygon

    from electrical.sheet_peec.hybrid_mesh import build_hybrid_mesh, graded_offsets

    strip = affinity.rotate(Polygon([(0, -2.4e-3), (6e-3, -2.4e-3), (6e-3, 2.4e-3), (0, 2.4e-3)]), 45, origin=(0, 0))
    mesh = build_hybrid_mesh(strip, pitch=0.4e-3, offsets=graded_offsets(3e-6, 2.0, 0.3e-3), along=0.25e-3)
    space = mesh.space
    assert len(mesh.square_elements) > 50 and len(mesh.gap_elements) > 0
    assert space.area.sum() == pytest.approx(strip.area, rel=1e-12)
    # the boundary is the outline: its edges add up to the perimeter
    b = space.boundary_edges
    length = np.linalg.norm(space.vertices[b[:, 0]] - space.vertices[b[:, 1]], axis=1).sum()
    assert length == pytest.approx(strip.length, rel=1e-12)
    # the first band row is as thin as asked
    first = mesh.band_elements[mesh.row_of_element[mesh.band_elements] == 0]
    assert np.max(space.area[first]) < 3e-6 * 0.25e-3 * 1.01


def test_an_annular_sector_between_equipotential_ports_has_its_closed_form_resistance() -> None:
    from electrical.sheet_peec.hybrid_mesh import build_hybrid_mesh, graded_offsets
    from electrical.sheet_peec.hybrid_rt0 import solve_ports

    r1, r2, span = 2e-3, 4e-3, math.pi / 2
    mesh = build_hybrid_mesh(_sector(r1, r2, span), pitch=0.4e-3, offsets=graded_offsets(3e-6, 2.0, 0.3e-3), along=0.25e-3)

    def angle(c):
        return np.arctan2(c[:, 1], c[:, 0])

    space = mesh.space.with_ports(lambda c: np.abs(angle(c)) < 1e-9, lambda c: np.abs(angle(c) - span) < 1e-9)
    _, _, out = solve_ports(space.resistance(1.0), space.divergence(), space.port_of_dof, [1.0, 0.0])
    resistance = -1.0 / out[0]  # out[0] < 0: the current enters at port 0
    assert resistance == pytest.approx(span / math.log(r2 / r1), rel=2e-3)
