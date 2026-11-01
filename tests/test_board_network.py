"""Copper and every part that joins its nets, solved together with each part's own current."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from electrical.port_network import (
    BoardNetwork,
    CurrentSource,
    Part,
    SeriesRLC,
    TabulatedImpedance,
    Tie,
    VoltageSource,
    capacitor,
)
from electrical.sheet_peec import (
    PortPad,
    SheetInductanceOperator,
    SheetLayer,
    SheetMesh,
    SheetStackup,
    solve_port_impedance,
)

PITCH_M, COPPER_M, RESISTIVITY = 2e-4, 3.5e-5, 1.724e-8
LENGTH, WIDTH = 40, 5


@dataclass(frozen=True)
class Copper:
    ports: tuple
    references: dict
    z_ohm: np.ndarray
    frequency_hz: float


def _strips(frequencies):
    """VIN on F.Cu over PGND on In1.Cu: the switch at the near end, capacitors at the middle and far end."""
    stackup = SheetStackup((SheetLayer("F.Cu", 0.0, COPPER_M, RESISTIVITY),
                            SheetLayer("In1.Cu", -2e-4, COPPER_M, RESISTIVITY)))
    rows, cols = WIDTH + 2, LENGTH + 3
    occupancy = np.zeros((2, rows, cols), dtype=bool)
    occupancy[:, 1:1 + WIDTH, 1:1 + LENGTH] = True
    mesh = SheetMesh((rows, cols), PITCH_M, stackup, occupancy)
    operator = SheetInductanceOperator((rows, cols), PITCH_M, stackup)

    def cells(col):
        return tuple((row, col) for row in range(1, 1 + WIDTH))

    pads = [PortPad(f"{net}.{where}", layer, cells(col))
            for net, layer in (("VIN", 0), ("PGND", 1))
            for where, col in (("sw", 1), ("mid", LENGTH // 2), ("far", LENGTH))]
    references = {"VIN.mid": "VIN.sw", "VIN.far": "VIN.sw", "PGND.mid": "PGND.sw", "PGND.far": "PGND.sw"}
    return [solve_port_impedance(mesh, operator, pads, references, frequency_hz=f) for f in frequencies]


@pytest.fixture(scope="module")
def strips():
    return _strips((1e6, 30e6))


def test_every_capacitor_keeps_its_own_current_and_the_switch_current_closes_through_them(strips) -> None:
    caps = [Part("C_mid", "VIN.mid", "PGND.mid", capacitor(1e-6, 2e-3, 0.5e-9)),
            Part("C_far", "VIN.far", "PGND.far", capacitor(1e-6, 2e-3, 0.5e-9))]
    switch = CurrentSource("Q", "PGND.sw", "VIN.sw", {1e6: 1.0, 30e6: 1.0})  # 1 A out of PGND into VIN at the switch
    network = BoardNetwork(strips, caps, ground="PGND.sw", current_sources=[switch])
    spectrum = network.solve_spectrum()
    for solution in spectrum.solutions:
        assert solution.solve_residual < 1e-9
        # the copper conductors are apart: all of the switch current returns through the two capacitors
        total = solution.part_current_a["C_mid"] + solution.part_current_a["C_far"]
        assert total == pytest.approx(1.0, abs=1e-9)
    near_share = np.abs(spectrum.part_current_a("C_mid"))
    # at 30 MHz the capacitors are inductive and the nearer one, in the smaller loop, takes more
    assert near_share[1] > 0.6 and near_share[1] > near_share[0] - 1e-3


def test_a_tie_at_the_far_end_gives_the_loop_of_the_port_matrix(strips) -> None:
    copper = strips[1]
    network = BoardNetwork([copper], [Part("short", "VIN.far", "PGND.far", Tie())], ground="PGND.sw",
                           current_sources=[CurrentSource("Q", "PGND.sw", "VIN.sw", {copper.frequency_hz: 1.0})])
    solution = network.solve(copper.frequency_hz)
    loop = solution.pad_voltage_v["VIN.sw"] - solution.pad_voltage_v["PGND.sw"]
    i = {p: k for k, p in enumerate(copper.ports)}
    z = copper.z_ohm
    vf, gf = i["VIN.far"], i["PGND.far"]
    # 1 A from the switch out along VIN to the far end, across the tie, and back along PGND
    expected = z[vf, vf] - z[vf, gf] - z[gf, vf] + z[gf, gf]
    assert loop == pytest.approx(expected, rel=1e-9)
    assert solution.part_current_a["short"] == pytest.approx(1.0, rel=1e-9)


def test_a_voltage_source_and_a_resistor_on_trivial_copper() -> None:
    copper = Copper(("A.2",), {"A.2": "A.1"}, np.array([[1e-3 + 0j]]), 1e3)
    network = BoardNetwork([copper], [Part("R", "A.2", "G", SeriesRLC(9e-3))], ground="G",
                           voltage_sources=[VoltageSource("V", "A.1", "G", {1e3: 1.0})])
    solution = network.solve(1e3)
    assert solution.part_current_a["R"] == pytest.approx(100.0, rel=1e-12)
    assert solution.source_current_a["V"] == pytest.approx(-100.0, rel=1e-12)  # it delivers 100 A out of a
    assert solution.copper_port_current_a["A.2"] == pytest.approx(-100.0, rel=1e-12)


def test_what_the_network_refuses() -> None:
    copper = Copper(("A.2",), {"A.2": "A.1"}, np.array([[1e-3 + 0j]]), 1e3)
    with pytest.raises(ValueError, match="no copper matrix"):
        BoardNetwork([copper], [Part("R", "A.2", "A.1", SeriesRLC(1.0))], ground="A.1").solve(2e3)
    floating = BoardNetwork([copper], [Part("R", "B.1", "B.2", SeriesRLC(1.0))], ground="A.1")
    with pytest.raises(np.linalg.LinAlgError, match="no path"):
        floating.solve(1e3)
    with pytest.raises(ValueError, match="names used twice"):
        BoardNetwork([copper], [Part("R", "A.2", "A.1", SeriesRLC(1.0)), Part("R", "A.1", "A.2", SeriesRLC(1.0))],
                     ground="A.1")


def test_a_tabulated_impedance_follows_its_curve_and_refuses_outside_it() -> None:
    part = capacitor(1e-6, 5e-3, 1e-9)
    f = np.logspace(3, 8, 51)
    table = TabulatedImpedance(tuple(f), tuple(part.impedance(2 * math.pi * v) for v in f))
    for v in (f[0], f[17], f[-1]):
        assert table.impedance(2 * math.pi * v) == pytest.approx(part.impedance(2 * math.pi * v), rel=1e-9)
    between = math.sqrt(f[10] * f[11])
    assert abs(table.impedance(2 * math.pi * between)) == pytest.approx(abs(part.impedance(2 * math.pi * between)), rel=0.02)
    with pytest.raises(ValueError, match="outside the table"):
        table.impedance(2 * math.pi * 2e8)


def _caps():
    return [Part("C_mid", "VIN.mid", "PGND.mid", capacitor(1e-6, 2e-3, 0.5e-9)),
            Part("C_far", "VIN.far", "PGND.far", capacitor(10e-6, 5e-3, 1.0e-9))]


def test_cases_and_harmonics_solve_together_as_each_alone(strips) -> None:
    # topology only: the switch and a ripple on the input, each case gives the spectra
    switch = CurrentSource("Q", "PGND.sw", "VIN.sw", {})
    source = VoltageSource("VIN", "VIN.far", "PGND.far", {})
    network = BoardNetwork(strips, _caps(), ground="PGND.sw", current_sources=[switch], voltage_sources=[source])
    cases = {
        "light": {"Q": {1e6: 0.5, 30e6: 0.05j}},
        "heavy": {"Q": {1e6: 2.0 - 0.3j, 30e6: 0.2 + 0.1j}, "VIN": {1e6: 0.01}},
        "ripple_only": {"VIN": {30e6: 1e-3}},
    }
    spectra = network.solve_cases(cases)
    assert set(spectra) == set(cases)
    for name, case in cases.items():
        alone = BoardNetwork(
            strips, _caps(), ground="PGND.sw",
            current_sources=[CurrentSource("Q", "PGND.sw", "VIN.sw", dict(case.get("Q", {})))],
            voltage_sources=[VoltageSource("VIN", "VIN.far", "PGND.far", dict(case.get("VIN", {})))],
        ).solve_spectrum()
        assert spectra[name].frequency_hz == (1e6, 30e6)
        for part in ("C_mid", "C_far"):
            np.testing.assert_allclose(spectra[name].part_current_a(part), alone.part_current_a(part),
                                       rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(spectra[name].pad_voltage_v("VIN.sw"), alone.pad_voltage_v("VIN.sw"),
                                   rtol=1e-12, atol=1e-18)
    # a frequency a case does not list carries nothing in it
    quiet = spectra["ripple_only"].solutions[0]
    assert all(v == 0 for v in quiet.part_current_a.values())
    # the network is linear: the heavy case's switch alone plus its ripple alone
    heavy, parts = spectra["heavy"].solutions[0], network.transfer(1e6)
    q_alone = parts.apply({"Q": 2.0 - 0.3j})
    v_alone = parts.apply({"VIN": 0.01})
    for part in ("C_mid", "C_far"):
        assert heavy.part_current_a[part] == pytest.approx(
            q_alone.part_current_a[part] + v_alone.part_current_a[part], rel=1e-12)


def test_a_transfer_is_one_factorisation_per_frequency(strips) -> None:
    network = BoardNetwork(strips, _caps(), ground="PGND.sw",
                           current_sources=[CurrentSource("Q", "PGND.sw", "VIN.sw", {1e6: 1.0})])
    transfer = network.transfer(1e6)
    assert network.transfer(1e6) is transfer
    assert transfer.sources == ("Q",)
    assert transfer.part_current_a.shape == (2, 1)
    assert transfer.solve_residual < 1e-9
    # the share of the switch current each capacitor takes, per ampere: the columns sum to the whole
    assert transfer.part_current_a[:, 0].sum() == pytest.approx(1.0, abs=1e-9)
    many = network.solve_cases({f"load_{k}": {"Q": {1e6: 0.1 * k}} for k in range(1, 50)})
    assert network.transfer(1e6) is transfer  # the cases reused it
    assert many["load_30"].part_current_a("C_mid")[0] == pytest.approx(3.0 * transfer.part_current_a[0, 0], rel=1e-12)


def test_what_cases_refuse(strips) -> None:
    network = BoardNetwork(strips, _caps(), ground="PGND.sw",
                           current_sources=[CurrentSource("Q", "PGND.sw", "VIN.sw", {})])
    with pytest.raises(ValueError, match="does not have"):
        network.solve_cases({"bad": {"nope": {1e6: 1.0}}})
    with pytest.raises(ValueError, match="no copper matrix"):
        network.solve_cases({"third_harmonic": {"Q": {3e6: 1.0}}})
