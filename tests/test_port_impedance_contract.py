"""port-impedance-problem/v1: routed copper's pads as ports, as a serialisable mapping."""

from __future__ import annotations

import json

import numpy as np
import pytest

from electrical.port_network import BoardNetwork, CurrentSource, Part, capacitor
from electrical.sheet_peec import (
    PORT_IMPEDANCE_PROBLEM_SCHEMA,
    PortImpedanceProblem,
    PortImpedanceResult,
    PortPad,
    SheetInductanceOperator,
    SheetLayer,
    SheetMesh,
    SheetStackup,
    ViaBranch,
    merge_pads,
    solve_port_impedance,
    solve_port_impedance_problem,
)

PITCH_MM, COPPER_MM, GAP_MM, RHO = 0.2, 0.035, 0.2, 1.724e-8
LENGTH, WIDTH = 40, 5
ROWS, COLUMNS = WIDTH + 2, LENGTH + 3
VIA_OHM = 1e-6
ONE_FILAMENT = {"maximum_filaments": 1, "relative_tolerance": 1e-10}


def _cells(layer: str, col: int, rows=range(1, 1 + WIDTH)) -> list[dict]:
    return [{"layer": layer, "x": col, "y": row} for row in rows]


def _problem(*, frequencies=(1e6,), pads=None, ports=None, vias=False, merge_pins=False) -> dict:
    strip = [{"x": x, "y": y} for y in range(1, 1 + WIDTH) for x in range(1, 1 + LENGTH)]
    connections = []
    if vias:
        connections = [{"name": f"via_{row}", "cell": {"x": LENGTH, "y": row},
                        "segments": [{"upper_layer": "F.Cu", "lower_layer": "In1.Cu",
                                      "resistance_ohm": VIA_OHM, "length_mm": GAP_MM}]}
                       for row in range(1, 1 + WIDTH)]
    return {
        "schema": PORT_IMPEDANCE_PROBLEM_SCHEMA,
        "name": "two_strips",
        "grid": {"rows": ROWS, "columns": COLUMNS, "pitch_mm": PITCH_MM},
        "layers": [
            {"name": "F.Cu", "order": 0, "center_z_mm": 0.0, "thickness_mm": COPPER_MM, "resistivity_ohm_m": RHO},
            {"name": "In1.Cu", "order": 1, "center_z_mm": -GAP_MM, "thickness_mm": COPPER_MM,
             "resistivity_ohm_m": RHO},
        ],
        "copper_by_layer": {"F.Cu": strip, "In1.Cu": strip},
        "vertical_connections": connections,
        "frequencies_hz": list(frequencies),
        "pads": pads if pads is not None else [
            {"name": "VIN.near", "part": "J1", "net": "VIN", "cells": _cells("F.Cu", 1)},
            {"name": "VIN.far", "part": "C1", "net": "VIN", "cells": _cells("F.Cu", LENGTH)},
            {"name": "GND.near", "part": "J1", "net": "GND", "cells": _cells("In1.Cu", 1)},
            {"name": "GND.far", "part": "C1", "net": "GND", "cells": _cells("In1.Cu", LENGTH)},
        ],
        "ports": ports if ports is not None else [
            {"pad": "VIN.near", "reference": "VIN.far"}, {"pad": "GND.near", "reference": "GND.far"},
        ],
        "merge_pins": merge_pins,
    }


def _direct(vias=False):
    """The same copper built as a SheetMesh by hand."""
    stackup = SheetStackup((SheetLayer("F.Cu#0", 0.0, COPPER_MM * 1e-3, RHO),
                            SheetLayer("In1.Cu#0", -GAP_MM * 1e-3, COPPER_MM * 1e-3, RHO)))
    occupancy = np.zeros((2, ROWS, COLUMNS), dtype=bool)
    occupancy[:, 1:1 + WIDTH, 1:1 + LENGTH] = True
    branches = tuple(ViaBranch(row, LENGTH, 1, 0, VIA_OHM) for row in range(1, 1 + WIDTH)) if vias else ()
    mesh = SheetMesh((ROWS, COLUMNS), PITCH_MM * 1e-3, stackup, occupancy, vias=branches)
    operator = SheetInductanceOperator(mesh.shape, PITCH_MM * 1e-3, stackup, vertical_levels=mesh.vertical_levels)
    return mesh, operator


def _pad(name, layer, col, rows=range(1, 1 + WIDTH)):
    return PortPad(name, layer, tuple((row, col) for row in rows))


def test_the_contract_gives_the_matrix_of_the_same_copper_built_by_hand() -> None:
    result = solve_port_impedance_problem(_problem(frequencies=(1e7, 1e6)), {**ONE_FILAMENT, "warm_start": False})
    assert result.ports == ("VIN.near", "GND.near")
    assert result.references == {"VIN.near": "VIN.far", "GND.near": "GND.far"}
    assert result.frequencies_hz == (1e6, 1e7)  # ascending
    assert result.metrics["meshes_built"] == 1  # one filament per layer: both frequencies share the mesh
    mesh, operator = _direct()
    pads = [_pad("VIN.near", 0, 1), _pad("VIN.far", 0, LENGTH), _pad("GND.near", 1, 1), _pad("GND.far", 1, LENGTH)]
    for frequency in (1e6, 1e7):
        direct = solve_port_impedance(mesh, operator, pads, {"VIN.near": "VIN.far", "GND.near": "GND.far"},
                                      frequency_hz=frequency, tolerance=1e-10, max_iterations=200, restart=120)
        np.testing.assert_array_equal(result.at(frequency).z_ohm, direct.z_ohm)


def test_a_frequency_starts_from_the_nearest_one_solved_on_its_mesh() -> None:
    frequencies = (1e6, 1e8, 1e7)
    cold = solve_port_impedance_problem(_problem(frequencies=frequencies), {**ONE_FILAMENT, "warm_start": False})
    warm = solve_port_impedance_problem(_problem(frequencies=frequencies), ONE_FILAMENT)
    started = {r["frequency_hz"]: r["started_from_hz"] for r in warm.metrics["frequencies"]}
    assert started == {1e6: None, 1e7: 1e6, 1e8: 1e7}  # solved in ascending order
    assert all(r["started_from_hz"] is None for r in cold.metrics["frequencies"])
    assert all(z.solutions is None for z in warm.impedances)  # kept to start from, not returned
    for a, b in zip(cold.impedances, warm.impedances):
        assert np.abs(a.z_ohm - b.z_ohm).max() < 1e-7 * np.abs(a.z_ohm).max()
    later = [r["maximum_iterations"] for r in warm.metrics["frequencies"][1:]]
    assert sum(later) < sum(r["maximum_iterations"] for r in cold.metrics["frequencies"][1:])


def test_a_port_solve_started_from_another_frequency() -> None:
    mesh, operator = _direct()
    pads = [_pad("VIN.near", 0, 1), _pad("VIN.far", 0, LENGTH), _pad("GND.near", 1, 1), _pad("GND.far", 1, LENGTH)]
    references = {"VIN.near": "VIN.far", "GND.near": "GND.far"}
    at_10 = solve_port_impedance(mesh, operator, pads, references, frequency_hz=1e7, keep_solutions=True)
    cold = solve_port_impedance(mesh, operator, pads, references, frequency_hz=3e7)
    warm = solve_port_impedance(mesh, operator, pads, references, frequency_hz=3e7, initial=at_10)
    assert np.abs(warm.z_ohm - cold.z_ohm).max() < 1e-7 * np.abs(cold.z_ohm).max()
    assert max(warm.iterations) < max(cold.iterations)
    # its own solution is a start that needs no iteration
    again = solve_port_impedance(mesh, operator, pads, references, frequency_hz=1e7, initial=at_10)
    assert again.iterations == (0, 0)
    np.testing.assert_allclose(again.z_ohm, at_10.z_ohm, rtol=1e-12)
    # one port: the excitation solved alone, not in lockstep
    one = {"VIN.near": "VIN.far"}
    alone_10 = solve_port_impedance(mesh, operator, pads, one, frequency_hz=1e7, keep_solutions=True)
    alone_cold = solve_port_impedance(mesh, operator, pads, one, frequency_hz=3e7)
    alone_warm = solve_port_impedance(mesh, operator, pads, one, frequency_hz=3e7, initial=alone_10)
    assert abs(alone_warm.z_ohm[0, 0] - alone_cold.z_ohm[0, 0]) < 1e-7 * abs(alone_cold.z_ohm[0, 0])
    assert alone_warm.iterations[0] < alone_cold.iterations[0]
    with pytest.raises(ValueError, match="keep_solutions"):
        solve_port_impedance(mesh, operator, pads, references, frequency_hz=3e7, initial=cold)
    with pytest.raises(ValueError, match="same ports"):
        solve_port_impedance(mesh, operator, pads, {"VIN.near": "VIN.far"}, frequency_hz=3e7, initial=at_10)


def test_vertical_connections_and_a_pad_over_two_layers() -> None:
    # the strips joined by vias at the far end: one conductor, a plated pad at the near end
    pads = [{"name": "J1.1", "cells": _cells("F.Cu", 1) + _cells("In1.Cu", 1)},
            {"name": "TP1", "cells": _cells("F.Cu", LENGTH // 2)},
            {"name": "C1.2", "cells": _cells("In1.Cu", LENGTH // 2)}]
    ports = [{"pad": "J1.1", "reference": "C1.2"}, {"pad": "TP1", "reference": "C1.2"}]
    result = solve_port_impedance_problem(_problem(pads=pads, ports=ports, vias=True), ONE_FILAMENT)
    mesh, operator = _direct(vias=True)
    direct = solve_port_impedance(
        mesh, operator,
        [PortPad("J1.1", 0, _pad("x", 0, 1).cells, ((1, _pad("x", 1, 1).cells),)),
         _pad("TP1", 0, LENGTH // 2), _pad("C1.2", 1, LENGTH // 2)],
        {"J1.1": "C1.2", "TP1": "C1.2"}, frequency_hz=1e6, tolerance=1e-10, max_iterations=200, restart=120,
    )
    np.testing.assert_array_equal(result.at(1e6).z_ohm, direct.z_ohm)


def test_merged_pins_are_one_port_named_by_part_and_net() -> None:
    pads = [{"name": "Q1.1", "part": "Q1", "net": "VIN", "cells": _cells("F.Cu", 1, rows=(1,))},
            {"name": "Q1.2", "part": "Q1", "net": "VIN", "cells": _cells("F.Cu", 1, rows=(3, 4, 5))},
            {"name": "C1.1", "part": "C1", "net": "VIN", "cells": _cells("F.Cu", LENGTH)}]
    ports = [{"pad": "Q1.1", "reference": "C1.1"}, {"pad": "Q1.2", "reference": "C1.1"}]
    copper = {"F.Cu": _problem()["copper_by_layer"]["F.Cu"]}

    def one_layer(merge):
        value = _problem(pads=pads, ports=ports, merge_pins=merge)
        value["layers"] = value["layers"][:1]
        value["copper_by_layer"] = copper
        return value

    split = solve_port_impedance_problem(one_layer(False), ONE_FILAMENT)
    assert split.ports == ("Q1.1", "Q1.2")
    merged = solve_port_impedance_problem(one_layer(True), ONE_FILAMENT)
    assert merged.ports == ("Q1:VIN",)
    assert merged.references == {"Q1:VIN": "C1:VIN"}
    assert merged.pad_groups == {"Q1:VIN": ("Q1.1", "Q1.2"), "C1:VIN": ("C1.1",)}

    stackup = SheetStackup((SheetLayer("F.Cu#0", 0.0, COPPER_MM * 1e-3, RHO),))
    occupancy = np.zeros((1, ROWS, COLUMNS), dtype=bool)
    occupancy[:, 1:1 + WIDTH, 1:1 + LENGTH] = True
    mesh = SheetMesh((ROWS, COLUMNS), PITCH_MM * 1e-3, stackup, occupancy)
    operator = SheetInductanceOperator(mesh.shape, PITCH_MM * 1e-3, stackup)
    direct = solve_port_impedance(
        mesh, operator,
        [merge_pads("Q1:VIN", [_pad("Q1.1", 0, 1, rows=(1,)), _pad("Q1.2", 0, 1, rows=(3, 4, 5))]),
         _pad("C1:VIN", 0, LENGTH)],
        {"Q1:VIN": "C1:VIN"}, frequency_hz=1e6, tolerance=1e-10, max_iterations=200, restart=120,
    )
    np.testing.assert_array_equal(merged.at(1e6).z_ohm, direct.z_ohm)
    # the pins side by side: merged, the port is the split ports in parallel within 0.1 %
    together = 1.0 / np.linalg.inv(split.at(1e6).z_ohm).sum()
    assert merged.at(1e6).z_ohm[0, 0] == pytest.approx(together, rel=1e-3)


def test_oblong_cells_are_solved_as_oblong() -> None:
    # Equal 0.2 x 0.4 mm cells, against the same copper on 0.2 mm squares: the
    # convolution operator takes one pitch, so these go the graded way (the
    # inductance was 4.98 nH when they were taken as 0.2 mm squares).
    def strip(row_mm, rows_per_cell):
        width = WIDTH * rows_per_cell
        value = _problem()
        cells = [{"x": x, "y": y} for y in range(1, 1 + width) for x in range(1, 1 + LENGTH)]
        value["copper_by_layer"] = {"F.Cu": cells, "In1.Cu": cells}
        value["grid"] = {"rows": width + 2, "columns": COLUMNS,
                         "x_edges_mm": (np.arange(COLUMNS + 1) * PITCH_MM).tolist(),
                         "y_edges_mm": (np.arange(width + 3) * row_mm).tolist()}
        for pad in value["pads"]:
            pad["cells"] = _cells(pad["cells"][0]["layer"], pad["cells"][0]["x"], rows=range(1, 1 + width))
        problem = PortImpedanceProblem.from_mapping(value)
        return problem, solve_port_impedance_problem(problem, ONE_FILAMENT).at(1e6).z_ohm[0, 0]

    oblong, z_oblong = strip(0.4, 1)
    square, z_square = strip(0.2, 2)
    assert oblong.pitch_mm is None and square.pitch_mm == pytest.approx(0.2)
    henry = lambda z: z.imag / (2 * np.pi * 1e6)
    assert henry(z_oblong) == pytest.approx(henry(z_square), rel=0.01)
    assert z_oblong.real == pytest.approx(z_square.real, rel=0.05)


def test_frequencies_that_cut_the_copper_differently_get_their_own_mesh() -> None:
    result = solve_port_impedance_problem(_problem(frequencies=(1e6, 3e7)), {"maximum_filaments": 8})
    records = result.metrics["frequencies"]
    assert [r["mesh_reused"] for r in records] == [False, False]
    assert records[1]["sheet_layers"] > records[0]["sheet_layers"]
    assert result.metrics["meshes_built"] == 2
    assert all(r["converged"] for r in records)


def test_the_result_round_trips_through_json_and_closes_a_network() -> None:
    result = solve_port_impedance_problem(_problem(frequencies=(1e6, 1e7)), ONE_FILAMENT)
    again = PortImpedanceResult.from_mapping(json.loads(json.dumps(result.to_mapping())))
    assert again.ports == result.ports and again.references == result.references
    for a, b in zip(again.impedances, result.impedances):
        np.testing.assert_array_equal(a.z_ohm, b.z_ohm)

    # a switch's current at the near end, a capacitor across the far end
    network = BoardNetwork(
        again.impedances, [Part("C1", "VIN.far", "GND.far", capacitor(1e-6, 2e-3, 0.5e-9))], ground="GND.far",
        current_sources=[CurrentSource("SW", "GND.near", "VIN.near", {1e6: 1.0, 1e7: 1.0})],
    )
    spectrum = network.solve_spectrum()
    np.testing.assert_allclose(spectrum.part_current_a("C1"), [1.0, 1.0], rtol=1e-9)  # all of it closes through C1


def test_bad_problems_are_refused() -> None:
    with pytest.raises(ValueError, match="schema"):
        PortImpedanceProblem.from_mapping({**_problem(), "schema": "current-field-problem/v2"})
    with pytest.raises(ValueError, match="not listed"):
        PortImpedanceProblem.from_mapping(_problem(ports=[{"pad": "VIN.near", "reference": "nowhere"}]))
    with pytest.raises(ValueError, match="not conductor"):
        PortImpedanceProblem.from_mapping(_problem(pads=[{"name": "X", "cells": [{"layer": "F.Cu", "x": 0, "y": 0}]}]))
    with pytest.raises(ValueError, match="positive"):
        PortImpedanceProblem.from_mapping(_problem(frequencies=(0.0,)))
    same_part = [{"name": "U1.1", "part": "U1", "net": "VIN", "cells": _cells("F.Cu", 1)},
                 {"name": "U1.2", "part": "U1", "net": "VIN", "cells": _cells("F.Cu", LENGTH)}]
    with pytest.raises(ValueError, match="merge into one pad"):
        PortImpedanceProblem.from_mapping(
            _problem(pads=same_part, ports=[{"pad": "U1.1", "reference": "U1.2"}], merge_pins=True))
    with pytest.raises(ValueError, match="'cpu' or 'cuda'"):
        solve_port_impedance_problem(_problem(), {"execution_backend": "rocm"})
