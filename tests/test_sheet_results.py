import unittest

import numpy as np

from electrical.sheet_peec.sheet_operator import (
    SheetInductanceOperator,
    SheetLayer,
    SheetStackup,
)
from electrical.sheet_peec.sheet_peec import SheetMesh, Terminal, ViaBranch, solve_sheet_case
from electrical.sheet_peec.sheet_results import (
    cell_current_density,
    sheet_fields,
    vertical_currents,
)

PITCH_M = 2e-4
COPPER_M = 3.5e-5
RESISTIVITY = 1.724e-8


def strip(rows, cols):
    """A uniform strip fed across its whole width at both ends."""
    stackup = SheetStackup((SheetLayer("F.Cu", 0.0, COPPER_M, RESISTIVITY),))
    mesh = SheetMesh(
        (rows, cols), PITCH_M, stackup, np.ones((1, rows, cols), dtype=bool)
    )
    operator = SheetInductanceOperator((rows, cols), PITCH_M, stackup)
    terminals = [
        Terminal("in", 0, tuple((row, 0) for row in range(rows)), 1.0),
        Terminal("out", 0, tuple((row, cols - 1) for row in range(rows)), -1.0),
    ]
    return mesh, operator, terminals


class UniformStripTests(unittest.TestCase):
    """A strip carrying a known current has a known everything."""

    def setUp(self):
        self.rows, self.cols = 9, 13
        self.mesh, self.operator, self.terminals = strip(self.rows, self.cols)
        self.solution = solve_sheet_case(
            self.mesh, self.operator, self.terminals, frequency_hz=0.0
        )
        self.fields = sheet_fields(self.mesh, self.solution, self.terminals)

    def test_the_density_is_the_current_over_the_cross_section(self):
        width_mm = self.rows * PITCH_M * 1e3
        expected = 1.0 / (width_mm * COPPER_M * 1e3)
        for row in range(self.rows):
            value = self.fields.current_density[(0, row, self.cols // 2)]
            self.assertAlmostEqual(value, expected, delta=expected * 1e-9)

    def test_the_loss_is_the_strips_resistance_times_the_current_squared(self):
        width = self.rows * PITCH_M
        length = (self.cols - 1) * PITCH_M
        resistance = RESISTIVITY * length / (width * COPPER_M)
        self.assertAlmostEqual(
            self.fields.metrics["i2r_loss_w"], resistance, delta=resistance * 1e-9
        )
        self.assertAlmostEqual(
            self.fields.metrics["voltage_span_v"],
            resistance,
            delta=resistance * 1e-9,
        )

    def test_the_current_closes(self):
        self.assertLess(self.fields.metrics["current_closure_error_a"], 1e-12)

    def test_the_metrics_a_gate_reads_are_all_present(self):
        for key in (
            "bulk_p99_current_density_a_per_mm2",
            "voltage_span_v",
            "voltage_span_definition",
            "max_vertical_connection_current_a",
            "current_closure_error_a",
            "i2r_loss_w",
            "max_current_density_a_per_mm2",
            "conductive_node_count",
            "converged",
        ):
            self.assertIn(key, self.fields.metrics)
        self.assertEqual(
            self.fields.metrics["conductive_node_count"], self.rows * self.cols
        )


class DensityConventionTests(unittest.TestCase):
    def test_the_two_axes_combine_as_a_magnitude_not_a_sum(self):
        # A cell carrying equal current along both axes carries sqrt(2) times
        # one of them, not twice.  Adding them would report a corner as twice
        # what it is; subtracting would report opposed currents as none.
        stackup = SheetStackup((SheetLayer("F.Cu", 0.0, COPPER_M, RESISTIVITY),))
        occupancy = np.zeros((1, 3, 3), dtype=bool)
        occupancy[0, 1, :] = True
        occupancy[0, :, 1] = True
        mesh = SheetMesh((3, 3), PITCH_M, stackup, occupancy)

        solution = type(
            "Fake",
            (),
            {
                "branch_current": np.zeros(mesh.branch_count, dtype=complex),
                "frequency_hz": 0.0,
            },
        )()
        # One unit along x into the centre, one unit along y out of it.
        for index, (_layer, row, col) in enumerate(mesh.branch_x):
            if (row, col) == (1, 0):
                solution.branch_current[index] = 1.0
        offset = len(mesh.branch_x)
        for index, (_layer, row, col) in enumerate(mesh.branch_y):
            if (row, col) == (1, 1):
                solution.branch_current[offset + index] = 1.0

        density = cell_current_density(mesh, solution)
        area = (PITCH_M * 1e3) * (COPPER_M * 1e3)
        # The centre sees half a unit on each axis, combined as a magnitude.
        self.assertAlmostEqual(
            density[(0, 1, 1)], np.hypot(0.5, 0.5) / area, places=9
        )

    def test_an_edge_cell_averages_over_the_branch_that_is_missing(self):
        # The branch that is not there carries nothing, so the current through
        # the cell is half what the one branch carries -- not all of it.
        stackup = SheetStackup((SheetLayer("F.Cu", 0.0, COPPER_M, RESISTIVITY),))
        occupancy = np.zeros((1, 1, 3), dtype=bool)
        occupancy[0, 0, :] = True
        mesh = SheetMesh((1, 3), PITCH_M, stackup, occupancy)
        solution = type(
            "Fake",
            (),
            {
                "branch_current": np.ones(mesh.branch_count, dtype=complex),
                "frequency_hz": 0.0,
            },
        )()
        density = cell_current_density(mesh, solution)
        area = (PITCH_M * 1e3) * (COPPER_M * 1e3)
        self.assertAlmostEqual(density[(0, 0, 0)], 0.5 / area, places=9)
        self.assertAlmostEqual(density[(0, 0, 1)], 1.0 / area, places=9)
        self.assertAlmostEqual(density[(0, 0, 2)], 0.5 / area, places=9)


class TerminalExclusionTests(unittest.TestCase):
    def test_the_bulk_percentile_leaves_the_terminal_cells_out(self):
        # A lead's own singularity is not a property of the shape being judged,
        # so the bulk figure has to be below the overall maximum where the
        # terminals are where the current crowds.
        rows, cols = 7, 11
        stackup = SheetStackup((SheetLayer("F.Cu", 0.0, COPPER_M, RESISTIVITY),))
        mesh = SheetMesh(
            (rows, cols), PITCH_M, stackup, np.ones((1, rows, cols), dtype=bool)
        )
        operator = SheetInductanceOperator((rows, cols), PITCH_M, stackup)
        terminals = [
            Terminal("in", 0, ((rows // 2, 0),), 1.0),
            Terminal("out", 0, ((rows // 2, cols - 1),), -1.0),
        ]
        solution = solve_sheet_case(mesh, operator, terminals, frequency_hz=0.0)
        fields = sheet_fields(mesh, solution, terminals)
        self.assertEqual(fields.metrics["terminal_cell_count"], 2)
        self.assertLess(
            fields.metrics["bulk_p99_current_density_a_per_mm2"],
            fields.metrics["max_current_density_a_per_mm2"],
        )
        self.assertEqual(len(fields.terminal_cells), 2)


class VerticalCurrentTests(unittest.TestCase):
    def test_the_only_crossing_carries_the_whole_current(self):
        rows, cols = 5, 9
        stackup = SheetStackup(
            (
                SheetLayer("F.Cu", 0.0, COPPER_M, RESISTIVITY),
                SheetLayer("B.Cu", -1.545e-3, COPPER_M, RESISTIVITY),
            )
        )
        via = ViaBranch(rows // 2, cols // 2, 0, 1, resistance_ohm=1e-3)
        mesh = SheetMesh(
            (rows, cols),
            PITCH_M,
            stackup,
            np.ones((2, rows, cols), dtype=bool),
            vias=(via,),
        )
        operator = SheetInductanceOperator(
            (rows, cols), PITCH_M, stackup, vertical_levels=mesh.vertical_levels
        )
        terminals = [
            Terminal("in", 0, ((rows // 2, 0),), 1.0),
            Terminal("out", 1, ((rows // 2, cols - 1),), -1.0),
        ]
        solution = solve_sheet_case(mesh, operator, terminals, frequency_hz=0.0)
        fields = sheet_fields(mesh, solution, terminals)
        crossing = vertical_currents(mesh, solution)
        self.assertEqual(len(crossing), 1)
        self.assertAlmostEqual(next(iter(crossing.values())), 1.0, places=9)
        self.assertAlmostEqual(
            fields.metrics["max_vertical_connection_current_a"], 1.0, places=9
        )

    def test_a_mesh_with_no_crossing_reports_none(self):
        mesh, operator, terminals = strip(5, 7)
        solution = solve_sheet_case(mesh, operator, terminals, frequency_hz=0.0)
        fields = sheet_fields(mesh, solution, terminals)
        self.assertEqual(fields.metrics["max_vertical_connection_current_a"], 0.0)


if __name__ == "__main__":
    unittest.main()
