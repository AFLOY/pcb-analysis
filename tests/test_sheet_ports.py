"""Pad-port impedance across conductors from one lockstep sheet solve."""

from __future__ import annotations

import numpy as np
import pytest

from electrical.sheet_peec import (
    PortPad,
    SheetInductanceOperator,
    SheetLayer,
    SheetMesh,
    SheetStackup,
    Terminal,
    ViaBranch,
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
