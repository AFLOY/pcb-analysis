from __future__ import annotations

import numpy as np
import pytest

from emc.tiled_dipole_superposition import (
    CISPR32_CLASS_A,
    CISPR32_CLASS_B,
    FCC_PART15_CLASS_A,
    FCC_PART15_CLASS_B,
    FREE_SPACE_IMPEDANCE_OHM,
    CurrentDipoles,
    SphereSampling,
    db_microvolt_per_m,
    dipole_moments,
    emission_margin,
    evaluate_fields,
    far_field_pattern,
    scan_plane,
    wavenumber_per_m,
)

ETA = FREE_SPACE_IMPEDANCE_OHM


def _square_loop(side_m: float, current_a: float, segments: int = 100) -> CurrentDipoles:
    step = side_m / segments
    positions = []
    moments = []
    for index in range(segments):
        t = -side_m / 2 + (index + 0.5) * step
        positions += [[t, -side_m / 2, 0.0], [side_m / 2, t, 0.0], [-t, side_m / 2, 0.0], [-side_m / 2, -t, 0.0]]
        moments += [[current_a * step, 0, 0], [0, current_a * step, 0], [-current_a * step, 0, 0], [0, -current_a * step, 0]]
    return CurrentDipoles(np.asarray(positions), np.asarray(moments))


def test_single_dipole_radiated_power_and_peak_field_match_the_closed_form() -> None:
    frequency = 100.0e6
    k = wavenumber_per_m(frequency)
    moment = 1.0e-3
    dipole = CurrentDipoles([[0.0, 0.0, 0.0]], [[0.0, 0.0, moment]])
    sampling = SphereSampling.gauss_legendre(polar_nodes=33, azimuth_nodes=8)  # odd count puts a node at θ = 90°

    pattern = far_field_pattern(dipole, frequency, distance_m=10.0, sampling=sampling)

    assert pattern.radiated_power_w == pytest.approx(ETA * k**2 * moment**2 / (12 * np.pi), rel=1e-12)
    assert pattern.max_field_v_per_m == pytest.approx(ETA * k * moment / (4 * np.pi * 10.0), rel=1e-9)
    assert pattern.directivity_dbi == pytest.approx(10 * np.log10(1.5), abs=1e-9)
    # A z-directed dipole radiates only E_theta.
    np.testing.assert_allclose(np.abs(pattern.e_phi_v_per_m), 0.0, atol=1e-16)
    scaled = pattern.scaled_to_distance(3.0)
    assert scaled.max_field_v_per_m == pytest.approx(pattern.max_field_v_per_m * 10.0 / 3.0)


def test_exact_fields_approach_the_far_field_and_the_wave_impedance() -> None:
    frequency = 300.0e6
    k = wavenumber_per_m(frequency)
    dipole = CurrentDipoles([[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0e-3]])
    distance = 20.0  # kr ≈ 126
    samples = evaluate_fields(dipole, np.array([[distance, 0.0, 0.0]]), frequency)

    electric = samples.electric_v_per_m[0]
    magnetic = samples.magnetic_a_per_m[0]
    far_magnitude = ETA * k * 1.0e-3 / (4 * np.pi * distance)
    assert abs(electric[2]) == pytest.approx(far_magnitude, rel=2.0 / (k * distance) ** 2)
    # Transverse fields, outward Poynting vector, wave impedance to O(1/(kr)²).
    # On the +x axis θ̂ = -ẑ and φ̂ = +ŷ, so E_z / H_y = -E_θ / H_φ = -η.
    assert abs(electric[0]) < 1e-9 * far_magnitude and abs(electric[1]) < 1e-9 * far_magnitude
    assert electric[2] / magnetic[1] == pytest.approx(-ETA, rel=2.0 / (k * distance) ** 2)
    poynting = 0.5 * np.real(np.cross(electric, np.conj(magnetic)))
    assert poynting[0] > 0.0


def test_zero_frequency_magnetic_field_is_biot_savart() -> None:
    xs = np.linspace(-1.0, 1.0, 4001)
    step = xs[1] - xs[0]
    wire = CurrentDipoles(
        np.column_stack((xs, np.zeros_like(xs), np.zeros_like(xs))),
        np.column_stack((np.full_like(xs, step), np.zeros_like(xs), np.zeros_like(xs))),
    )
    rho = 0.01
    samples = evaluate_fields(
        wire, np.array([[0.0, rho, 0.0], [0.0, 0.0, rho]]), 0.0, electric=False, tile_points=1
    )
    expected = 1.0 / (2 * np.pi * rho)
    np.testing.assert_allclose(samples.magnetic_magnitude_a_per_m, expected, rtol=1e-4)
    # Right-hand rule: current along +x, point on +y gives H along +z.
    assert samples.magnetic_a_per_m[0][2].real == pytest.approx(expected, rel=1e-4)
    assert samples.electric_v_per_m is None
    with pytest.raises(ValueError, match="undefined at zero frequency"):
        evaluate_fields(wire, np.array([[0.0, rho, 0.0]]), 0.0)


def test_small_loop_radiates_as_its_magnetic_dipole_moment() -> None:
    frequency = 100.0e6
    k = wavenumber_per_m(frequency)
    side, current = 0.01, 1.0
    loop = _square_loop(side, current)

    pattern = far_field_pattern(loop, frequency)
    moments = dipole_moments(loop, frequency)

    magnetic_moment = current * side**2
    np.testing.assert_allclose(moments.magnetic_a_m2, [0.0, 0.0, magnetic_moment], atol=1e-15)
    np.testing.assert_allclose(moments.electric_a_m, 0.0, atol=1e-15)
    expected = ETA * k**4 * magnetic_moment**2 / (12 * np.pi)
    assert moments.magnetic_radiated_power_w == pytest.approx(expected, rel=1e-12)
    assert pattern.radiated_power_w == pytest.approx(expected, rel=1e-3)
    # A horizontal loop radiates E_phi in its plane and nothing along its axis.
    in_plane = np.argmin(np.abs(pattern.sampling.theta_rad - np.pi / 2))
    assert abs(pattern.e_phi_v_per_m[in_plane]) > 100 * abs(pattern.e_theta_v_per_m[in_plane])
    # Along the loop axis the field vanishes as sin θ; the nearest Gauss node is ~4°.
    nearest_axis = int(np.argmin(pattern.sampling.theta_rad))
    assert pattern.magnitude_v_per_m[nearest_axis] == pytest.approx(
        pattern.max_field_v_per_m * np.sin(pattern.sampling.theta_rad[nearest_axis]), rel=2e-2
    )


def test_ground_plane_images_cancel_the_tangential_electric_field() -> None:
    frequency = 200.0e6
    sources = CurrentDipoles([[0.0, 0.0, 0.01], [0.02, 0.0, 0.03]], [[1e-3, 0, 0], [0, 2e-3, 1e-3]])
    imaged = sources.with_ground_plane_images(0.0)
    assert imaged.count == 4
    np.testing.assert_allclose(imaged.position_m[2:, 2], [-0.01, -0.03])
    np.testing.assert_allclose(imaged.moment_a_m[2:], [[-1e-3, 0, 0], [0, -2e-3, 1e-3]])

    points = np.array([[0.3, 0.2, 0.0], [-0.1, 0.05, 0.0]])
    samples = evaluate_fields(imaged, points, frequency)
    scale = np.max(np.abs(samples.electric_v_per_m))
    np.testing.assert_allclose(samples.electric_v_per_m[:, :2], 0.0, atol=1e-12 * scale)
    with pytest.raises(ValueError, match="one side"):
        CurrentDipoles([[0, 0, 0.01], [0, 0, -0.01]], [[1e-3, 0, 0]] * 2).with_ground_plane_images(0.0)


def test_tiling_and_dtype_do_not_change_the_answer() -> None:
    rng = np.random.default_rng(0)
    sources = CurrentDipoles(rng.normal(scale=0.02, size=(37, 3)), rng.normal(size=(37, 3)) * 1e-3)
    points = scan_plane(np.linspace(-0.05, 0.05, 7), np.linspace(-0.04, 0.04, 5), 0.01)
    assert points.shape == (35, 3)
    reference = evaluate_fields(sources, points, 150e6)
    tiled = evaluate_fields(sources, points, 150e6, tile_points=4)
    np.testing.assert_allclose(tiled.magnetic_a_per_m, reference.magnetic_a_per_m, rtol=1e-12)
    np.testing.assert_allclose(tiled.electric_v_per_m, reference.electric_v_per_m, rtol=1e-12)
    single = evaluate_fields(sources, points, 150e6, dtype=np.complex64)
    np.testing.assert_allclose(single.magnetic_a_per_m, reference.magnetic_a_per_m, rtol=1e-4)
    pattern = far_field_pattern(sources, 150e6, tile_directions=5)
    np.testing.assert_allclose(
        pattern.magnitude_v_per_m, far_field_pattern(sources, 150e6).magnitude_v_per_m, rtol=1e-12
    )


def test_limit_lines_and_margins() -> None:
    assert CISPR32_CLASS_B.limit_dbuv_per_m(100e6) == 30.0
    assert CISPR32_CLASS_B.limit_dbuv_per_m(500e6) == 37.0
    assert CISPR32_CLASS_A.limit_dbuv_per_m(100e6) == 40.0
    assert CISPR32_CLASS_B.limit_dbuv_per_m(100e6, distance_m=3.0) == pytest.approx(30.0 + 20 * np.log10(10 / 3))
    assert CISPR32_CLASS_B.limit_dbuv_per_m(2e9, detector="average") == 50.0
    assert FCC_PART15_CLASS_B.limit_dbuv_per_m(100e6) == pytest.approx(43.52, abs=0.01)
    assert FCC_PART15_CLASS_B.limit_dbuv_per_m(500e6) == pytest.approx(46.02, abs=0.01)
    assert FCC_PART15_CLASS_A.limit_dbuv_per_m(500e6) == pytest.approx(46.44, abs=0.01)
    assert FCC_PART15_CLASS_B.published_distance_m(100e6) == 3.0
    with pytest.raises(ValueError, match="no quasi-peak limit"):
        CISPR32_CLASS_B.limit_dbuv_per_m(10e6)

    assert db_microvolt_per_m(1e-6) == pytest.approx(0.0)
    margin = emission_margin(10e-6, 100e6, CISPR32_CLASS_B, distance_m=10.0)
    assert margin.predicted_dbuv_per_m == pytest.approx(20.0)
    assert margin.margin_db == pytest.approx(10.0)
    assert margin.compliant
    assert not emission_margin(100e-6, 100e6, CISPR32_CLASS_B, distance_m=10.0).compliant


def test_cuda_backend_matches_numpy() -> None:
    pytest.importorskip("cupy")
    from emc.tiled_dipole_superposition import array_namespace

    try:
        xp = array_namespace("cuda")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert xp is not np
    rng = np.random.default_rng(5)
    sources = CurrentDipoles(rng.normal(scale=0.02, size=(300, 3)), rng.normal(size=(300, 3)) * 1e-3)
    points = scan_plane(np.linspace(-0.05, 0.05, 20), np.linspace(-0.04, 0.04, 15), 0.005)
    cpu = evaluate_fields(sources, points, 200e6)
    gpu = evaluate_fields(sources, points, 200e6, backend="cuda", tile_points=64)
    np.testing.assert_allclose(gpu.magnetic_a_per_m, cpu.magnetic_a_per_m, rtol=1e-10)
    np.testing.assert_allclose(gpu.electric_v_per_m, cpu.electric_v_per_m, rtol=1e-10)
    cpu_pattern = far_field_pattern(sources, 200e6)
    gpu_pattern = far_field_pattern(sources, 200e6, backend="cuda")
    np.testing.assert_allclose(gpu_pattern.magnitude_v_per_m, cpu_pattern.magnitude_v_per_m, rtol=1e-10)
    assert gpu_pattern.radiated_power_w == pytest.approx(cpu_pattern.radiated_power_w, rel=1e-10)
