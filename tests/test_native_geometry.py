"""Tessellated point-in-solid tests: NumPy and C++ winding numbers against OpenCASCADE."""

from __future__ import annotations

import numpy as np
import pytest

from geometry.step_voxelize import (
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
    occ = sample_plane_fill([pad], **kwargs)
    numpy_path = sample_plane_fill([pad], method="numpy", **kwargs)
    np.testing.assert_allclose(numpy_path, occ)
    if native_classify_available():
        np.testing.assert_allclose(sample_plane_fill([pad], method="native", **kwargs), occ)
    assert float(occ.sum()) * (0.5 * MM) ** 2 == pytest.approx(6.0 * MM**2, rel=0.12)
