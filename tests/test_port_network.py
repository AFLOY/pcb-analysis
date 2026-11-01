"""Port networks: realization of a pad-port impedance, SPICE export, lumped closure, port moments."""

from __future__ import annotations

import math

import numpy as np
import pytest

from electrical.port_network import (
    Branch,
    SeriesRLC,
    capacitor,
    close_kcl,
    conductance_branches,
    loop_impedance,
    mesh_matrices,
    nodal_voltages,
    port_block,
    realize_port_network,
    subckt_text,
    window_moments,
)

# --- realization ------------------------------------------------------------


def test_realization_reproduces_the_matrix_and_is_resistive_at_dc() -> None:
    z = np.array([[2.0 + 1.0j, 0.5 + 0.2j], [0.5 + 0.2j, 3.0 + 1.5j]])
    network = realize_port_network(z, ("A.1", "B.1"), "R.1", 1e5)
    np.testing.assert_allclose(network.impedance_matrix(), z)
    np.testing.assert_allclose(network.impedance_matrix(0.0), z.real)
    assert [(r.pad_a, r.pad_b) for r in network.resistors] == [("A.1", "B.1"), ("A.1", "R.1"), ("B.1", "R.1")]
    assert set(network.self_inductance_h) == {"A.1", "B.1"}
    assert len(network.couplings) == 1
    assert network.flags == ()


def test_a_shared_copper_path_gives_a_negative_resistor_that_is_kept() -> None:
    z = np.array([[1.0 + 1.0j, 1.2 + 0.5j], [1.2 + 0.5j, 2.0 + 1.0j]])
    network = realize_port_network(z, ("A.1", "B.1"), "R.1", 1e5)
    np.testing.assert_allclose(network.impedance_matrix(), z)
    assert any("negative_resistance" in r.flags for r in network.resistors)
    assert "resistance_matrix_not_positive_definite" not in network.flags


def test_an_indefinite_inductance_matrix_drops_the_couplings_and_says_so() -> None:
    z = np.array([[1.0 + 1.0j, 0.1 + 1.5j], [0.1 + 1.5j, 1.0 + 1.0j]])
    network = realize_port_network(z, ("A.1", "B.1"), "R.1", 1e5)
    assert "inductance_matrix_not_positive_definite" in network.flags
    assert network.couplings == ()
    np.testing.assert_allclose(network.impedance_matrix().real, z.real)


def test_a_dc_matrix_realizes_as_resistors_alone() -> None:
    network = realize_port_network(np.array([[2.0]]), ("A",), "R", 0.0)
    assert network.self_inductance_h == {} and network.couplings == ()
    assert [(r.pad_a, r.pad_b, r.resistance_ohm) for r in network.resistors] == [("A", "R", pytest.approx(2.0))]


def test_conductance_branches_of_a_full_and_a_reduced_matrix() -> None:
    full = np.array([[3.0, -1.0, -2.0], [-1.0, 1.5, -0.5], [-2.0, -0.5, 2.5]])
    assert conductance_branches(full, reduced=False) == [(0, 1, 1.0), (0, 2, 2.0), (1, 2, 0.5)]
    reduced = full[1:, 1:]
    assert conductance_branches(reduced, reduced=True) == [(0, 1, 0.5), (0, None, 1.0), (1, None, 2.0)]


# --- SPICE export -------------------------------------------------------------


def test_subckt_text_names_the_ports_and_elements_after_the_pads() -> None:
    z = np.array([[2.0 + 1.0j, 0.5 + 0.2j], [0.5 + 0.2j, 3.0 + 1.5j]])
    text = subckt_text(realize_port_network(z, ("A.1", "B.1"), "R.1", 1e5), "copper")
    lines = text.splitlines()
    assert lines[1] == ".subckt copper R_1 A_1 B_1"
    assert lines[-1] == ".ends copper"
    cards = [line.split()[0] for line in lines[2:-1]]
    assert cards == ["L_A_1", "L_B_1", "K_A_1__B_1", "R_A_1__B_1", "R_A_1__R_1", "R_B_1__R_1"]
    # the resistors join the inner nodes and the reference; the inductors the pads and the inner nodes
    assert "R_A_1__R_1 A_1_m R_1 " in text and "L_A_1 A_1 A_1_m " in text


def test_a_resistive_subckt_puts_its_resistors_on_the_pads() -> None:
    text = subckt_text(realize_port_network(np.array([[2.0]]), ("A",), "R", 0.0), "r")
    assert "R_A__R A R 2.000000e+00" in text and "L_" not in text


# --- lumped closure -----------------------------------------------------------


def test_series_rlc_impedance() -> None:
    omega = 2 * math.pi * 1e6
    part = capacitor(1e-6, 0.01, 1e-9)
    assert part.impedance(omega) == pytest.approx(complex(0.01, omega * 1e-9 - 1 / (omega * 1e-6)), rel=1e-12)
    assert SeriesRLC(1.0, 2e-9).impedance(omega) == pytest.approx(complex(1.0, omega * 2e-9))


def test_nodal_voltages_of_a_divider_and_its_reference() -> None:
    # 1 A into node 1; node 1 -- 2 ohm -- node 2 -- (3 ohm || 1 uF) -- node 0 (reference)
    f = 1e5
    omega = 2 * math.pi * f
    branches = [Branch(1, 2, SeriesRLC(2.0)), Branch(2, 0, SeriesRLC(3.0)), Branch(2, 0, capacitor(1e-6, 0.0, 0.0))]
    v = nodal_voltages(3, branches, [0, 1, 0], 0, f)
    z_shunt = 1.0 / (1.0 / 3.0 + 1j * omega * 1e-6)
    assert v[0] == 0
    assert v[2] == pytest.approx(z_shunt, rel=1e-12)
    assert v[1] == pytest.approx(2.0 + z_shunt, rel=1e-12)


def test_a_floating_node_is_singular() -> None:
    with pytest.raises(np.linalg.LinAlgError):
        nodal_voltages(3, [Branch(1, 0, SeriesRLC(1.0))], [0, 0, 1], 0, 1e3)


def test_a_zero_impedance_branch_is_kept_as_a_short() -> None:
    v = nodal_voltages(3, [Branch(1, 2, SeriesRLC()), Branch(2, 0, SeriesRLC(5.0))], [0, 1, 0], 0, 1e3)
    assert v[1] == pytest.approx(5.0 + 1e-9)


def test_loop_impedance_of_two_meshes_through_shared_copper() -> None:
    f = 30e6
    omega = 2 * math.pi * f
    r = np.array([[2e-3, 1e-3], [1e-3, 3e-3]])
    l = np.array([[4e-9, 2e-9], [2e-9, 6e-9]])
    parts = [capacitor(10e-9, 0.05, 0.7e-9), None]
    z, currents = loop_impedance(r, l, parts, f)
    zc = np.diag([parts[0].impedance(omega), 0.0])
    expected = np.linalg.solve(r + 1j * omega * l + zc, np.ones(2))
    np.testing.assert_allclose(currents, expected, rtol=1e-12)
    assert z == pytest.approx(1.0 / expected.sum(), rel=1e-12)
    # one mesh alone is its own impedance
    z1, _ = loop_impedance(r[:1, :1], l[:1, :1], [None], f)
    assert z1 == pytest.approx(complex(2e-3, omega * 4e-9), rel=1e-12)


def test_mesh_matrices_symmetrise_and_split_r_and_l() -> None:
    f = 1e6
    omega = 2 * math.pi * f
    z = np.array([[1 + 1j * omega * 2e-9, 0.2 + 1j * omega * 1e-9], [0.4 + 1j * omega * 1e-9, 2 + 1j * omega * 3e-9]])
    r, l = mesh_matrices(z, f)
    np.testing.assert_allclose(r, [[1, 0.3], [0.3, 2]])
    np.testing.assert_allclose(l, [[2e-9, 1e-9], [1e-9, 3e-9]], rtol=1e-12)


# --- port moments ---------------------------------------------------------------


def _nonuniform_time(start: float, stop: float, count: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    inner = np.sort(rng.uniform(start, stop, count - 2))
    return np.concatenate(([start], inner, [stop]))


def test_window_moments_match_the_analytic_averages_of_a_sine_on_dc() -> None:
    f = 300e3
    t = _nonuniform_time(0.0, 3.0 / f, 6000)
    a, b, c = 2.0, 0.8, 1.5
    i_1 = a + b * np.sin(2 * math.pi * f * t)
    i_2 = c * np.cos(2 * math.pi * f * t)
    currents = np.stack((i_1, i_2, -(i_1 + i_2)), axis=1)
    start, stop = 1.0 / f, 2.0 / f
    m = window_moments(("p1", "p2", "p3"), t, currents, start, stop, fundamental_hz=f, harmonics=3)

    np.testing.assert_allclose(m.mean_a, (a, 0.0, -a), atol=2e-4)
    expected = np.array([[a * a + b * b / 2, 0.0, -(a * a + b * b / 2)],
                         [0.0, c * c / 2, -c * c / 2],
                         [-(a * a + b * b / 2), -c * c / 2, a * a + b * b / 2 + c * c / 2]])
    np.testing.assert_allclose(m.second_moment_a2, expected, atol=3e-4)
    assert len(m.harmonics) == 3
    np.testing.assert_allclose(m.harmonics[0], (-1j * b, c, 1j * b - c), atol=3e-4)
    np.testing.assert_allclose(m.harmonics[1], 0.0, atol=3e-4)
    np.testing.assert_allclose(m.ripple_variance_a2, (b * b / 2, c * c / 2, b * b / 2 + c * c / 2), atol=3e-4)
    assert m.samples == int(np.sum((t > start) & (t < stop))) + 2


def test_a_partial_window_has_moments_but_no_harmonics() -> None:
    f = 300e3
    t = np.linspace(0.0, 2.0 / f, 2001)
    currents = np.stack((np.sin(2 * math.pi * f * t), -np.sin(2 * math.pi * f * t)), axis=1)
    m = window_moments(("a", "b"), t, currents, 0.0, 0.455 / f, fundamental_hz=f, harmonics=5)
    assert m.harmonics == ()
    with pytest.raises(ValueError, match="outside the simulated"):
        window_moments(("a", "b"), t, currents, 0.0, 3.0 / f)
    with pytest.raises(ValueError, match="positive length"):
        window_moments(("a", "b"), t, currents, 1.0 / f, 1.0 / f)


def test_kcl_closure_is_one_projection_shared_by_means_moments_and_harmonics() -> None:
    f = 1e3
    t = np.linspace(0.0, 1.0 / f, 4001)
    i_a = 2.0 + 0.5 * np.sin(2 * math.pi * f * t)
    i_b = 1.0 + 0.2 * np.cos(2 * math.pi * f * t)
    leak = 1.0e-4
    i_c = -(i_a + i_b) + leak
    quiet = np.full_like(t, 1e-12)
    raw = window_moments(("A", "B", "C", "Q"), t, np.stack((i_a, i_b, i_c, quiet), axis=1), 0.0, 1.0 / f,
                         fundamental_hz=f, harmonics=2)
    closed = close_kcl(raw, {"A": "net", "B": "net", "C": "net", "Q": "net"})

    active = [0, 1, 2]
    assert math.fsum(closed.mean_a[active]) == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_allclose(closed.second_moment_a2[active].sum(axis=0), 0.0, atol=1e-14)
    for h in closed.harmonics:
        assert abs(h[active].sum()) < 1e-14
    assert closed.mean_a[3] == raw.mean_a[3]
    weights = np.abs(raw.mean_a[active]) / np.sum(np.abs(raw.mean_a[active]))
    np.testing.assert_allclose(closed.mean_a[active], raw.mean_a[active] - leak * weights, atol=1e-12)
    assert closed.closure_by_group["net"]["raw_residual_a"] == pytest.approx(leak, rel=1e-6)
    assert np.all(np.linalg.eigvalsh(closed.second_moment_a2) > -1e-12)

    mean, second, harmonics = port_block(closed, ("B", "A"))
    assert mean[0] == closed.mean_a[1] and mean[1] == closed.mean_a[0]
    assert second.shape == (2, 2) and len(harmonics) == 2
    assert closed.as_dict()["closure_by_group"]["net"]["carried_a"] > 0.0

    bad = window_moments(("A", "C"), t, np.stack((i_a, -0.5 * i_a), axis=1), 0.0, 1.0 / f)
    with pytest.raises(ValueError, match="does not close"):
        close_kcl(bad, {"A": "net", "C": "net"})
