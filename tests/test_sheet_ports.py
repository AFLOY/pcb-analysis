"""Pad-port impedance across conductors from one lockstep sheet solve."""

from __future__ import annotations

import numpy as np
import pytest

from electrical.port_network import realize_port_network, subckt_text
from electrical.sheet_peec import (
    PortPad,
    SheetInductanceOperator,
    SheetLayer,
    SheetMesh,
    SheetStackup,
    Terminal,
    ViaBranch,
    merge_pads,
    solve_port_impedance,
    solve_sheet_case,
)

PITCH_M, COPPER_M, RESISTIVITY = 2e-4, 3.5e-5, 1.724e-8
LENGTH, WIDTH, FREQUENCY = 40, 5, 1e6


def _stackup() -> SheetStackup:
    return SheetStackup((SheetLayer("F.Cu", 0.0, COPPER_M, RESISTIVITY),
                         SheetLayer("In1.Cu", -2e-4, COPPER_M, RESISTIVITY)))


def _mesh(vias=()):
    rows, cols = WIDTH + 2, LENGTH + 3
    occupancy = np.zeros((2, rows, cols), dtype=bool)
    occupancy[:, 1:1 + WIDTH, 1:1 + LENGTH] = True
    return SheetMesh((rows, cols), PITCH_M, _stackup(), occupancy, vias=vias), (rows, cols)


def _cells(col: int) -> tuple:
    return tuple((row, col) for row in range(1, 1 + WIDTH))


PADS = [PortPad("top_near", 0, _cells(1)), PortPad("top_far", 0, _cells(LENGTH)),
        PortPad("bottom_near", 1, _cells(1)), PortPad("bottom_far", 1, _cells(LENGTH)),
        PortPad("top_mid", 0, _cells(LENGTH // 2))]


def test_the_matrix_is_reciprocal_and_couples_the_two_conductors() -> None:
    mesh, shape = _mesh()
    operator = SheetInductanceOperator(shape, PITCH_M, _stackup())
    result = solve_port_impedance(
        mesh, operator, PADS, {"top_near": "top_far", "top_mid": "top_far", "bottom_near": "bottom_far"},
        frequency_hz=FREQUENCY,
    )
    assert result.ports == ("top_near", "top_mid", "bottom_near")
    assert result.reference_pads == ("top_far", "bottom_far")
    assert all(result.converged)
    z = result.z_ohm
    np.testing.assert_allclose(z, z.T, rtol=1e-6)
    omega = 2 * np.pi * FREQUENCY
    inductance = z.imag / omega
    assert inductance[0, 2] > 0.8 * inductance[0, 0]       # go and return couple strongly
    assert 0.4 < inductance[1, 1] / inductance[0, 0] < 0.6  # half the strip, about half its inductance
    assert np.all(np.linalg.eigvalsh(inductance) > 0.0)

    # the loop closed by vias at the far end is the port matrix's go minus return
    loop_from_ports = z[0, 0] - z[0, 2] - z[2, 0] + z[2, 2]
    vias = tuple(ViaBranch(row, LENGTH, 1, 0, 1e-6) for row, _ in _cells(LENGTH))
    closed_mesh, _ = _mesh(vias)
    closed = solve_sheet_case(
        closed_mesh, SheetInductanceOperator(shape, PITCH_M, _stackup(), vertical_levels=closed_mesh.vertical_levels),
        [Terminal("top_near", 0, _cells(1), 1.0), Terminal("bottom_near", 1, _cells(1), -1.0)],
        frequency_hz=FREQUENCY,
    )

    def mean(layer):
        return np.mean([closed.node_voltage[closed_mesh.node_index[(layer, r, c)]] for r, c in _cells(1)])

    loop_closed = mean(0) - mean(1)
    assert loop_from_ports.imag / loop_closed.imag == pytest.approx(1.0, abs=0.02)
    assert loop_from_ports.real / loop_closed.real == pytest.approx(1.0, abs=0.01)


def test_bad_port_statements_are_refused() -> None:
    mesh, shape = _mesh()
    operator = SheetInductanceOperator(shape, PITCH_M, _stackup())
    with pytest.raises(ValueError, match="not given"):
        solve_port_impedance(mesh, operator, PADS, {"top_near": "nowhere"}, frequency_hz=FREQUENCY)
    with pytest.raises(ValueError, match="cannot also be a port"):
        solve_port_impedance(mesh, operator, PADS, {"top_near": "top_far", "top_far": "top_mid"},
                             frequency_hz=FREQUENCY)
    with pytest.raises(ValueError, match="close its own"):
        # a port whose reference is on the other conductor drives a net current into each
        solve_port_impedance(mesh, operator, PADS, {"top_near": "bottom_far"}, frequency_hz=FREQUENCY)


def test_the_solved_matrix_realizes_as_a_circuit_of_two_conductors() -> None:
    mesh, shape = _mesh()
    operator = SheetInductanceOperator(shape, PITCH_M, _stackup())
    result = solve_port_impedance(mesh, operator, PADS, {"top_near": "top_far", "bottom_near": "bottom_far"},
                                  frequency_hz=FREQUENCY)
    network = realize_port_network(result.z_ohm, result.ports, result.references, result.frequency_hz)
    assert network.flags == ()
    assert len(network.couplings) == 1  # across the conductors
    assert all({x.pad_a, x.pad_b} in ({"top_near", "top_far"}, {"bottom_near", "bottom_far"}) for x in network.resistors)
    np.testing.assert_allclose(network.impedance_matrix().imag, result.z_ohm.imag, rtol=1e-9)
    # the eddy loss the driven strip induces in the other is a mutual resistance; a resistor
    # between conductors would join them at DC, so it is left out and recorded
    assert network.dropped_cross_resistance_ohm == pytest.approx(abs(result.z_ohm[0, 1].real))
    z, rebuilt = result.z_ohm, network.impedance_matrix()
    loop, loop_rebuilt = z[0, 0] - 2 * z[0, 1] + z[1, 1], rebuilt[0, 0] - 2 * rebuilt[0, 1] + rebuilt[1, 1]
    assert loop_rebuilt.imag == pytest.approx(loop.imag, rel=1e-9)
    assert loop_rebuilt.real - loop.real == pytest.approx(2 * z[0, 1].real, rel=1e-9)
    assert subckt_text(network, "strips").splitlines()[1] == ".subckt strips top_far bottom_far top_near bottom_near"


def _pins(rows, col):
    return tuple((row, col) for row in rows)


def test_merged_pins_lose_only_the_division_between_them() -> None:
    # one strip, its return across the far end; a part's two pins on one net at the near end
    mesh, shape = _mesh()
    operator = SheetInductanceOperator(shape, PITCH_M, _stackup())
    far = PortPad("far", 0, _cells(LENGTH))

    def merged_and_tied(pin_a, pin_b):
        split = solve_port_impedance(mesh, operator, [pin_a, pin_b, far], {pin_a.name: "far", pin_b.name: "far"},
                                     frequency_hz=FREQUENCY)
        merged = solve_port_impedance(mesh, operator, [merge_pads("q1_s", [pin_a, pin_b]), far], {"q1_s": "far"},
                                      frequency_hz=FREQUENCY)
        # the pins tied together outside the copper: the split matrix's ports in parallel
        return merged.z_ohm[0, 0], 1.0 / np.linalg.inv(split.z_ohm).sum()

    # pins side by side across the current's path, as a package's leads are: the
    # even spread over their cells is the division they take when tied (3e-4 here)
    pin_a, pin_b = PortPad("q1_s1", 0, _pins((1,), 1)), PortPad("q1_s2", 0, _pins((3, 4, 5), 1))
    assert merge_pads("q1_s", [pin_a, pin_b]).cells == pin_a.cells + pin_b.cells
    merged, tied = merged_and_tied(pin_a, pin_b)
    assert merged.real == pytest.approx(tied.real, rel=1e-3)
    assert merged.imag == pytest.approx(tied.imag, rel=1e-3)
    # pins 1 mm apart along the path: tied, the copper between them carries a
    # circulating current that merging cannot represent (6 % in R, 9 % in L here)
    merged, tied = merged_and_tied(PortPad("q1_s1", 0, _pins((1,), 1)), PortPad("q1_s2", 0, _pins((5,), 6)))
    assert abs(merged.imag / tied.imag - 1.0) > 0.03


def test_a_pad_over_two_layers() -> None:
    # the strips joined by vias across both ends: one conductor; a plated pad over
    # both layers at the near end, the far end of the top strip its return
    ends = tuple(ViaBranch(row, col, 1, 0, 1e-6) for col in (2, LENGTH - 1) for row, _ in _cells(col))
    mesh, shape = _mesh(ends)
    operator = SheetInductanceOperator(shape, PITCH_M, _stackup(), vertical_levels=mesh.vertical_levels)
    plated = PortPad("j1", 0, _cells(1), ((1, _cells(1)),))
    pads = [plated, PortPad("mid", 1, _cells(LENGTH // 2)), PortPad("far", 0, _cells(LENGTH))]
    result = solve_port_impedance(mesh, operator, pads, {"j1": "far", "mid": "far"}, frequency_hz=FREQUENCY)
    assert all(result.converged)
    np.testing.assert_allclose(result.z_ohm, result.z_ohm.T, rtol=1e-6)


def test_a_pad_lists_each_layer_once() -> None:
    with pytest.raises(ValueError, match="twice"):
        PortPad("x", 0, _cells(1), ((0, _cells(2)),))
    with pytest.raises(ValueError, match="no cells"):
        PortPad("x", 0, ())
