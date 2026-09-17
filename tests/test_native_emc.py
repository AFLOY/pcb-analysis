"""C++ direct dipole summation against the array path (opt-in)."""

from __future__ import annotations

import numpy as np
import pytest

from emc.tiled_dipole_superposition import (
    CurrentDipoles,
    SphereSampling,
    evaluate_fields,
    far_field_pattern,
    scan_plane,
)
from emc.tiled_dipole_superposition.native_dipole import native_available

pytestmark = pytest.mark.skipif(
    not native_available(),
    reason="emc native extension not built; run python -m emc.tiled_dipole_superposition.native.build",
)


def _sources(count: int, seed: int = 5) -> CurrentDipoles:
    rng = np.random.default_rng(seed)
    moment = (rng.normal(size=(count, 3)) + 1j * rng.normal(size=(count, 3))) * 1e-3
    return CurrentDipoles(rng.normal(scale=0.02, size=(count, 3)), moment)


@pytest.mark.parametrize("threads", [1, 3])
def test_native_near_field_matches_the_array_path(threads) -> None:
    sources = _sources(123)
    points = scan_plane(np.linspace(-0.03, 0.03, 9), np.linspace(-0.02, 0.02, 7), 0.005)
    reference = evaluate_fields(sources, points, 150e6, native=False)
    native = evaluate_fields(sources, points, 150e6, native=True, native_threads=threads)
    np.testing.assert_allclose(native.magnetic_a_per_m, reference.magnetic_a_per_m, rtol=1e-12, atol=0)
    np.testing.assert_allclose(native.electric_v_per_m, reference.electric_v_per_m, rtol=1e-12, atol=0)


def test_native_zero_frequency_is_biot_savart_and_refuses_the_electric_field() -> None:
    sources = _sources(40)
    points = np.array([[0.0, 0.1, 0.0], [0.05, 0.0, 0.02]])
    reference = evaluate_fields(sources, points, 0.0, electric=False, native=False)
    native = evaluate_fields(sources, points, 0.0, electric=False, native=True)
    np.testing.assert_allclose(native.magnetic_a_per_m, reference.magnetic_a_per_m, rtol=1e-12)
    assert native.electric_v_per_m is None
    with pytest.raises(ValueError, match="zero"):
        evaluate_fields(sources, points, 0.0, native=True)


def test_native_far_field_matches_pattern_power_and_peak() -> None:
    sources = _sources(200).with_ground_plane_images(-0.05)
    sampling = SphereSampling.gauss_legendre(24, 48)
    reference = far_field_pattern(sources, 400e6, sampling=sampling, native=False)
    native = far_field_pattern(sources, 400e6, sampling=sampling, native=True, native_threads=2)
    np.testing.assert_allclose(native.electric_v_per_m, reference.electric_v_per_m, rtol=1e-11, atol=0)
    assert native.radiated_power_w == pytest.approx(reference.radiated_power_w, rel=1e-12)
    assert native.max_field_dbuv_per_m == pytest.approx(reference.max_field_dbuv_per_m, abs=1e-9)


def test_native_rejects_coincident_points_and_unsupported_requests() -> None:
    sources = _sources(3)
    with pytest.raises(ValueError, match="coincides"):
        evaluate_fields(sources, sources.position_m[:1], 100e6, native=True)
    with pytest.raises(ValueError, match="complex128"):
        evaluate_fields(sources, np.array([[1.0, 0.0, 0.0]]), 100e6, native=True, dtype=np.complex64)
    with pytest.raises(ValueError, match="complex128"):
        far_field_pattern(sources, 100e6, native=True, dtype=np.complex64)
