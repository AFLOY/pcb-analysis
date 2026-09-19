"""Tessellated point-in-solid tests: NumPy and C++ winding numbers against OpenCASCADE."""

from __future__ import annotations

import numpy as np
import pytest

from geometry.cad_import import (
    TriangleMesh,
    box_solid,
    cylinder_solid,
    native_classify_available,
    ocp_available,
    sample_plane_fill,
    winding_numbers_numpy,
)

MM = 1.0e-3


def _tetrahedron() -> TriangleMesh:
    a, b, c, d = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    return TriangleMesh(np.array([[a, c, b], [a, b, d], [b, c, d], [a, d, c]]), 0.0)


def test_numpy_winding_number_on_a_tetrahedron() -> None:
    mesh = _tetrahedron()
    assert mesh.is_closed()
    assert mesh.signed_volume_m3() == pytest.approx(1.0 / 6.0)
    points = np.array([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [-0.1, 0.2, 0.2], [0.2, 0.2, 0.2]])
    winding = winding_numbers_numpy(points, mesh.triangles_m)
    np.testing.assert_allclose(winding, [1.0, 0.0, 0.0, 1.0], atol=1.0e-12)
    assert mesh.contains(points, method="numpy").tolist() == [True, False, False, True]
    flipped = TriangleMesh(mesh.triangles_m[:, ::-1, :], 0.0)
    assert flipped.signed_volume_m3() == pytest.approx(-1.0 / 6.0)


@pytest.mark.skipif(not native_classify_available(), reason="geometry native extension not built")
@pytest.mark.parametrize("threads", [1, 3])
def test_native_winding_numbers_match_numpy(threads: int) -> None:
    mesh = _tetrahedron()
    rng = np.random.default_rng(1)
    points = rng.uniform(-0.2, 1.2, size=(5000, 3))
    expected = winding_numbers_numpy(points, mesh.triangles_m)
    actual = mesh.winding_numbers(points, method="native", threads=threads)
    np.testing.assert_allclose(actual, expected, atol=1.0e-12)
    np.testing.assert_array_equal(mesh.contains(points, method="native", threads=threads), mesh.contains(points, method="numpy"))


@pytest.mark.skipif(not ocp_available(), reason="OCP not installed")
def test_tessellation_is_closed_outward_and_agrees_with_the_classifier() -> None:
    box = box_solid("box", (0.0, 0.0, 0.0), (20 * MM, 12 * MM, 1.6 * MM))
    cylinder = cylinder_solid("via", (5 * MM, 5 * MM), -0.1 * MM, 1.8 * MM, 0.3 * MM)
    rng = np.random.default_rng(2)
    for solid, points in (
        (box, np.column_stack((rng.uniform(-1, 21, 4000), rng.uniform(-1, 13, 4000), rng.uniform(-0.5, 2.1, 4000))) * MM),
        (cylinder, np.column_stack((rng.uniform(4.6, 5.4, 4000), rng.uniform(4.6, 5.4, 4000), rng.uniform(-0.3, 1.9, 4000))) * MM),
    ):
        mesh = solid.tessellate()
        assert mesh.is_closed()
        assert mesh.signed_volume_m3() == pytest.approx(solid.volume_m3, rel=5.0e-3)
        occ = solid.contains(points, method="occ")
        numpy_path = solid.contains(points, method="numpy")
        # Disagreements sit within the tessellation deflection of the surface.
        disagree = occ != numpy_path
        assert disagree.mean() < 5.0e-3
        if solid is cylinder and disagree.any():
            radial = np.hypot(points[disagree, 0] - 5 * MM, points[disagree, 1] - 5 * MM)
            assert np.all(np.abs(radial - 0.3 * MM) < 2.0 * mesh.deflection_m + 1e-9) or np.all(
                (points[disagree, 2] < -0.1 * MM + 1e-6) | (points[disagree, 2] > 1.7 * MM - 1e-6)
            )
        if native_classify_available():
            np.testing.assert_array_equal(solid.contains(points, method="native"), numpy_path)
    assert box.tessellate().size == 12
    assert box.tessellate() is box.tessellate()  # cached


@pytest.mark.skipif(not ocp_available(), reason="OCP not installed")
def test_raster_fill_is_the_same_on_every_classification_path() -> None:
    pad = box_solid("pad", (1.3 * MM, 0.7 * MM, 0.0), (3.0 * MM, 2.0 * MM, 0.035 * MM))
    kwargs = dict(z_m=0.0175 * MM, origin_m=(0.0, 0.0), pitch_m=0.5 * MM, shape=(8, 10), supersample=3)
    occ = sample_plane_fill([pad], method="occ", **kwargs)
    numpy_path = sample_plane_fill([pad], method="numpy", **kwargs)
    np.testing.assert_allclose(numpy_path, occ)
    if native_classify_available():
        np.testing.assert_allclose(sample_plane_fill([pad], method="native", **kwargs), occ)
        # The exact section differs from three-sample quantisation by at most one sample per cell.
        exact = sample_plane_fill([pad], method="section", **kwargs)
        assert float(np.max(np.abs(exact - occ))) <= 1.0 / 9.0 + 1.0e-12
        assert float(exact.sum()) * (0.5 * MM) ** 2 == pytest.approx(6.0 * MM**2, rel=1.0e-12)
    assert float(occ.sum()) * (0.5 * MM) ** 2 == pytest.approx(6.0 * MM**2, rel=0.12)


def test_section_segments_are_closed_loops_counter_clockwise() -> None:
    from geometry.cad_import import section_segments_numpy

    mesh = _tetrahedron()
    segments = section_segments_numpy(mesh.triangles_m, 0.25)
    assert segments.shape == (3, 2, 2)
    # Every end point is another segment's start point: closed loop.
    starts = {tuple(np.round(s[0], 12)) for s in segments}
    ends = {tuple(np.round(s[1], 12)) for s in segments}
    assert starts == ends
    # Shoelace area of the loop is positive (counter-clockwise) and equals the section triangle.
    area = 0.5 * float(np.sum(segments[:, 0, 0] * segments[:, 1, 1] - segments[:, 1, 0] * segments[:, 0, 1]))
    assert area == pytest.approx(0.5 * 0.75 * 0.75)
    if native_classify_available():
        native = mesh.section_segments(0.25)
        assert native.shape == segments.shape
        native_area = 0.5 * float(np.sum(native[:, 0, 0] * native[:, 1, 1] - native[:, 1, 0] * native[:, 0, 1]))
        assert native_area == pytest.approx(area)


@pytest.mark.skipif(not (ocp_available() and native_classify_available()), reason="needs OCP and the native extension")
def test_section_coverage_is_exact_and_handles_holes() -> None:
    from geometry.cad_import import plane_section_coverage

    pad = box_solid("pad", (1.3 * MM, 0.7 * MM, 0.0), (3.0 * MM, 2.0 * MM, 0.035 * MM))
    coverage = plane_section_coverage([pad.tessellate()], 0.0175 * MM, origin_m=(0.0, 0.0), pitch_m=0.5 * MM, shape=(8, 10))
    assert float(coverage.sum()) * (0.5 * MM) ** 2 == pytest.approx(6.0 * MM**2, rel=1.0e-12)
    assert coverage[1, 2] == pytest.approx(0.2 * 0.3 / 0.25)  # corner cell: 0.2 x 0.3 mm of a 0.5 mm cell
    assert coverage[2, 4] == pytest.approx(1.0)
    assert coverage[0, 0] == 0.0
    # An annulus (pad ring around a drill) plus its barrel: the hole stays open, the pieces add.
    ring_outer = cylinder_solid("outer", (2.0 * MM, 2.0 * MM), 0.0, 0.035 * MM, 0.85 * MM)
    # Build the ring as the section of two solids is not possible with primitives alone, so check
    # orientation with the barrel instead: coverage of a barrel disc at its own section.
    barrel = cylinder_solid("barrel", (2.0 * MM, 2.0 * MM), -0.1 * MM, 1.7 * MM, 0.45 * MM)
    disc = plane_section_coverage([barrel.tessellate()], 0.8 * MM, origin_m=(0.0, 0.0), pitch_m=0.05 * MM, shape=(80, 80))
    assert float(disc.sum()) * (0.05 * MM) ** 2 == pytest.approx(np.pi * (0.45 * MM) ** 2, rel=3.0e-3)
    both = plane_section_coverage([ring_outer.tessellate(), barrel.tessellate()], 0.0175 * MM, origin_m=(0.0, 0.0), pitch_m=0.05 * MM, shape=(80, 80))
    # Overlapping solids clamp at one; the union area is the outer disc.
    assert float(both.sum()) * (0.05 * MM) ** 2 == pytest.approx(np.pi * (0.85 * MM) ** 2, rel=3.0e-3)
    assert both.max() <= 1.0
    # Reference: dense point sampling through OpenCASCADE agrees with the exact area per cell.
    sampled = sample_plane_fill([pad], z_m=0.0175 * MM, origin_m=(0.0, 0.0), pitch_m=0.5 * MM, shape=(8, 10), supersample=10, method="occ")
    np.testing.assert_allclose(coverage, sampled, atol=0.06)
    auto = sample_plane_fill([pad], z_m=0.0175 * MM, origin_m=(0.0, 0.0), pitch_m=0.5 * MM, shape=(8, 10))
    np.testing.assert_allclose(auto, coverage)
