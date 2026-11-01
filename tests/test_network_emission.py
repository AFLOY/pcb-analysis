"""The emission of a board network: superposed copper current and part currents, harmonic by harmonic."""

from __future__ import annotations

import numpy as np
import pytest

from electrical.port_network import BoardNetwork, CurrentSource, Part, capacitor, refine_frequencies
from electrical.sheet_peec import (
    PortImpedanceProblem,
    PortPad,
    SheetInductanceOperator,
    SheetLayer,
    SheetMesh,
    SheetStackup,
    Terminal,
    solve_port_impedance,
    solve_port_impedance_problem,
    solve_sheet_case,
)
from emc.tiled_dipole_superposition import (
    CISPR32_CLASS_B,
    dipoles_from_sheet_peec,
    evaluate_fields,
    far_field_pattern,
    network_dipoles,
    network_emission,
    path_dipoles,
    scan_plane,
)
from test_port_impedance_contract import _problem

PITCH, COPPER, RHO, GAP = 2e-4, 3.5e-5, 1.724e-8, 2e-4
LENGTH, WIDTH = 40, 5
ROWS, COLS = WIDTH + 2, LENGTH + 3
LAYER_Z = {0: 0.0, 1: -GAP}


def _cells(col):
    return tuple((row, col) for row in range(1, 1 + WIDTH))


def _mesh():
    stackup = SheetStackup((SheetLayer("F.Cu", 0.0, COPPER, RHO), SheetLayer("In1.Cu", -GAP, COPPER, RHO)))
    occupancy = np.zeros((2, ROWS, COLS), dtype=bool)
    occupancy[:, 1:1 + WIDTH, 1:1 + LENGTH] = True
    mesh = SheetMesh((ROWS, COLS), PITCH, stackup, occupancy)
    return mesh, SheetInductanceOperator((ROWS, COLS), PITCH, stackup)


PADS = [PortPad(f"{net}.{where}", layer, _cells(col))
        for net, layer in (("VIN", 0), ("PGND", 1))
        for where, col in (("sw", 1), ("mid", LENGTH // 2), ("far", LENGTH))]
REFERENCES = {"VIN.mid": "VIN.sw", "VIN.far": "VIN.sw", "PGND.mid": "PGND.sw", "PGND.far": "PGND.sw"}
POSITIONS = {pad.name: (float(np.mean([(c + 0.5) * PITCH for _, c in pad.cells])),
                        float(np.mean([(r + 0.5) * PITCH for r, _ in pad.cells])), LAYER_Z[pad.layer]) for pad in PADS}


def _network(coppers, spectrum):
    parts = [Part("C_mid", "VIN.mid", "PGND.mid", capacitor(1e-6, 2e-3, 0.5e-9)),
             Part("C_far", "VIN.far", "PGND.far", capacitor(10e-6, 5e-3, 1.0e-9))]
    return BoardNetwork(coppers, parts, ground="PGND.sw",
                        current_sources=[CurrentSource("Q", "PGND.sw", "VIN.sw", spectrum)])


@pytest.fixture(scope="module")
def copper():
    mesh, operator = _mesh()
    solved = {f: solve_port_impedance(mesh, operator, PADS, REFERENCES, frequency_hz=f, keep_solutions=True)
              for f in (1e6, 30e6, 100e6)}
    return mesh, operator, solved


def test_the_superposed_network_radiates_as_the_same_currents_solved_directly(copper) -> None:
    mesh, operator, solved = copper
    f = 30e6
    network = _network(list(solved.values()), {f: 1.0})
    solution = network.solve(f)
    assert solution.source_current_a["Q"] == 1.0  # a current source reports its own current
    superposed = network_dipoles(network, solution, solved[f], mesh, POSITIONS)

    # the same port currents injected directly, one sheet solve
    injected = {pad.name: 0j for pad in PADS}
    for port, current in solution.copper_port_current_a.items():
        injected[port] += current
        injected[REFERENCES[port]] -= current
    by_name = {pad.name: pad for pad in PADS}
    direct = solve_sheet_case(mesh, operator, [Terminal(n, by_name[n].layer, by_name[n].cells, i)
                                               for n, i in injected.items()], frequency_hz=f, tolerance=1e-10)
    ends = {item.name: (item.a, item.b) for item in (*network.parts, *network.current_sources)}
    reference = dipoles_from_sheet_peec(mesh, direct).concatenate(
        path_dipoles({**solution.part_current_a, **solution.source_current_a}, ends, POSITIONS))

    points = scan_plane(np.linspace(-2e-3, 10e-3, 7), np.linspace(-2e-3, 3e-3, 5), 3e-3)
    near_a = evaluate_fields(superposed, points, f).magnetic_a_per_m
    near_b = evaluate_fields(reference, points, f).magnetic_a_per_m
    assert np.abs(near_a - near_b).max() < 1e-6 * np.abs(near_b).max()
    far_a, far_b = far_field_pattern(superposed, f, distance_m=3.0), far_field_pattern(reference, f, distance_m=3.0)
    assert far_a.max_field_v_per_m == pytest.approx(far_b.max_field_v_per_m, rel=1e-6)
    assert far_a.radiated_power_w == pytest.approx(far_b.radiated_power_w, rel=1e-6)


def test_the_parts_close_the_copper_s_loops(copper) -> None:
    mesh, _, solved = copper
    network = _network(list(solved.values()), {f: 1.0 for f in solved})
    for f in solved:
        dipoles = network_dipoles(network, network.solve(f), solved[f], mesh, POSITIONS)
        # every pad takes from the copper what the parts deliver: the moments sum to zero
        total = np.abs(dipoles.moment_a_m.sum(axis=0)).max()
        assert total < 1e-7 * np.abs(dipoles.moment_a_m).sum()


def test_emission_per_harmonic_with_a_limit_where_it_has_one(copper) -> None:
    mesh, _, solved = copper
    network = _network(list(solved.values()), {})
    spectra = network.solve_cases({"run": {"Q": {1e6: 2.0, 30e6: 0.3, 100e6: 0.1}}})
    points = scan_plane(np.linspace(0.0, 8e-3, 5), np.linspace(0.0, 1.4e-3, 3), 2e-3)
    emission = network_emission(network, spectra["run"].solutions, solved, mesh, POSITIONS,
                                distance_m=3.0, limit=CISPR32_CLASS_B, scan_points_m=points)
    assert [e.frequency_hz for e in emission] == [1e6, 30e6, 100e6]
    assert emission[0].margin is None  # CISPR 32 has no radiated line below 30 MHz
    assert all(e.margin is not None and e.margin.distance_m == 3.0 for e in emission[1:])
    assert all(e.near_field_max_h_a_per_m > 0.0 and e.far_field_v_per_m > 0.0 for e in emission)
    # a small loop: the field falls as its current and rises as f^2
    ratio = emission[2].far_field_v_per_m / emission[1].far_field_v_per_m
    loop = abs(spectra["run"].solutions[2].copper_port_current_a["VIN.far"]) / abs(
        spectra["run"].solutions[1].copper_port_current_a["VIN.far"])
    assert ratio == pytest.approx(loop * (100 / 30) ** 2, rel=0.2)


def test_what_the_emission_refuses(copper) -> None:
    mesh, operator, solved = copper
    sweep = refine_frequencies(lambda f: solve_port_impedance(mesh, operator, PADS, REFERENCES, frequency_hz=f),
                               [1e6, 30e6], tolerance=0.05, maximum_solves=4)
    network = _network(sweep, {})
    interpolated = network.transfer(5e6).apply({"Q": 1.0})
    with pytest.raises(ValueError, match="interpolated"):
        network_dipoles(network, interpolated, solved[1e6], mesh, POSITIONS)
    plain = solve_port_impedance(mesh, operator, PADS, REFERENCES, frequency_hz=1e6)
    on_plain = _network([plain], {1e6: 1.0})
    with pytest.raises(ValueError, match="keep_solutions"):
        network_dipoles(on_plain, on_plain.solve(1e6), plain, mesh, POSITIONS)
    with pytest.raises(ValueError, match="no position"):
        network_dipoles(on_plain, on_plain.solve(1e6), solved[1e6], mesh, {})


def test_a_graded_board_through_the_contract_closes_its_loops() -> None:
    value = _problem(frequencies=(30e6,), vias=False)
    widths = [0.4] + [0.2] * 41 + [0.4]  # graded: the pFFT operator solves it
    edges = np.concatenate([[0.0], np.cumsum(widths)])
    value["grid"] = {"rows": 7, "columns": 43, "x_edges_mm": edges.tolist(),
                     "y_edges_mm": (np.arange(8) * 0.2).tolist()}
    result = solve_port_impedance_problem(value, {"maximum_filaments": 1, "relative_tolerance": 1e-10},
                                          keep_solutions=True)
    problem = PortImpedanceProblem.from_mapping(value)
    positions = problem.pad_positions_m()
    assert set(positions) == set(problem.pad_groups())
    network = BoardNetwork(result.impedances,
                           [Part("C1", "VIN.far", "GND.far", capacitor(1e-6, 2e-3, 0.5e-9))], ground="GND.near",
                           current_sources=[CurrentSource("Q", "GND.near", "VIN.near", {30e6: 1.0})])
    solution = network.solve(30e6)
    mesh = result.meshes[0]
    assert mesh.pitch_m is None
    dipoles = network_dipoles(network, solution, result.impedances[0], mesh, positions)
    assert np.abs(dipoles.moment_a_m.sum(axis=0)).max() < 1e-7 * np.abs(dipoles.moment_a_m).sum()
