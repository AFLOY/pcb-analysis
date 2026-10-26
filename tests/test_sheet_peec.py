import unittest

import numpy as np

from electrical.sheet_peec.sheet_operator import (
    SheetInductanceOperator,
    SheetLayer,
    SheetStackup,
)
from electrical.sheet_peec.sheet_inductance import (
    mutual_partial_inductance,
    vertical_cell,
)
from electrical.sheet_peec.sheet_peec import (
    SheetMesh,
    Terminal,
    ViaBranch,
    solve_sheet_case,
    solve_sheet_cases,
    via_resistance,
)

PITCH_M = 2e-4
COPPER_M = 3.5e-5
RESISTIVITY = 1.724e-8
CORE_GAP_M = 1.545e-3


def one_layer():
    return SheetStackup((SheetLayer("F.Cu", 0.0, COPPER_M, RESISTIVITY),))


def two_layers():
    return SheetStackup(
        (
            SheetLayer("F.Cu", 0.0, COPPER_M, RESISTIVITY),
            SheetLayer("B.Cu", -CORE_GAP_M, COPPER_M, RESISTIVITY),
        )
    )


def full_mesh(stackup, rows, cols, vias=()):
    occupancy = np.ones((len(stackup), rows, cols), dtype=bool)
    return SheetMesh((rows, cols), PITCH_M, stackup, occupancy, vias=vias)


def branch_by_branch_near_inductance(operator, mesh, radius_cells=1):
    """The near-field inductance read off the tables one branch at a time."""
    import scipy.sparse as sp

    rows, cols = operator.shape
    padded_rows, padded_cols = operator.padded
    tables = {key: np.fft.irfft2(spectrum, s=operator.padded) for key, spectrum in operator._kernels.items()}
    tables_z = {key: np.fft.irfft2(spectrum, s=operator.padded) for key, spectrum in operator._kernels_z.items()}
    entries_i, entries_j, values = [], [], []
    span = range(-radius_cells, radius_cells + 1)

    def walk(index, levels_count, table_of):
        for (level, row, col), source in index.items():
            for other in range(levels_count):
                table = table_of(level, other)
                for dr in span:
                    for dc in span:
                        target = index.get((other, row + dr, col + dc))
                        if target is not None:
                            entries_i.append(source)
                            entries_j.append(target)
                            values.append(float(table[dr % padded_rows, dc % padded_cols]))

    offset = 0
    for axis, group in (("x", mesh.branch_x), ("y", mesh.branch_y)):
        index = {branch: position + offset for position, branch in enumerate(group)}
        walk(index, len(operator.stackup), lambda a, b, axis=axis: tables[(axis, min(a, b), max(a, b))])
        offset += len(group)
    levels = list(operator.vertical_levels)
    index_of = {key: position for position, key in enumerate(levels)}
    index = {
        (index_of[(via.lower_layer, via.upper_layer)], via.row, via.col): position + offset
        for position, via in enumerate(mesh.via_branches)
    }
    walk(index, len(levels), lambda a, b: tables_z[(min(a, b), max(a, b))])
    count = mesh.branch_count
    return sp.csr_matrix((values, (entries_i, entries_j)), shape=(count, count))


def resistor_network_voltages(rows, cols, source, sink):
    """Solve the same grid as a plain resistor network, independently.

    Nothing here shares code with the sheet solver: the conductances are formed
    directly from the sheet resistance and the system is solved densely.  At
    zero frequency the two have to agree exactly.
    """
    sheet_resistance = RESISTIVITY / COPPER_M
    count = rows * cols

    def index(row, col):
        return row * cols + col

    conductance = np.zeros((count, count))
    for row in range(rows):
        for col in range(cols):
            for step_row, step_col in ((0, 1), (1, 0)):
                other_row, other_col = row + step_row, col + step_col
                if other_row < rows and other_col < cols:
                    first, second = index(row, col), index(other_row, other_col)
                    value = 1.0 / sheet_resistance
                    conductance[first, first] += value
                    conductance[second, second] += value
                    conductance[first, second] -= value
                    conductance[second, first] -= value
    injected = np.zeros(count)
    injected[index(*source)] = 1.0
    injected[index(*sink)] = -1.0
    reduced = np.linalg.solve(
        np.delete(np.delete(conductance, 0, 0), 0, 1), np.delete(injected, 0)
    )
    return np.concatenate(([0.0], reduced))


def dense_reference(mesh, operator, terminals, frequency_hz):
    """Solve the same case with the inductance assembled as a dense matrix."""
    count = mesh.branch_count
    impedance = np.zeros((count, count), dtype=complex)
    basis = np.zeros(count)
    has_vertical = bool(mesh.vertical_levels)
    for column in range(count):
        basis[column] = 1.0
        grid_x, grid_y = mesh.scatter(basis)
        if has_vertical:
            grid_z = mesh.scatter_vertical(basis)
            flux_x, flux_y, flux_z = operator.apply(grid_x, grid_y, grid_z)
        else:
            flux_x, flux_y = operator.apply(grid_x, grid_y)
            flux_z = None
        gathered = mesh.gather(flux_x, flux_y)
        if flux_z is not None:
            mesh.gather_vertical(flux_z, gathered)
        impedance[:, column] = gathered
        basis[column] = 0.0
    omega = 2.0 * np.pi * frequency_hz
    impedance *= 1j * omega
    impedance += np.diag(mesh.resistances().astype(complex))

    incidence = mesh.incidence().toarray()
    keep = np.ones(mesh.node_count, dtype=bool)
    keep[0] = False
    reduced = incidence[:, keep]
    injected = np.zeros(mesh.node_count, dtype=complex)
    for terminal in terminals:
        share = terminal.current_a / len(terminal.cells)
        for row, col in terminal.cells:
            injected[mesh.node_index[(terminal.layer, row, col)]] += share
    admittance = np.linalg.inv(impedance)
    solution = np.linalg.solve(reduced.T @ admittance @ reduced, injected[keep])
    voltage = np.zeros(mesh.node_count, dtype=complex)
    voltage[keep] = solution
    return voltage


class DirectCurrentTests(unittest.TestCase):
    """At zero frequency this is a resistor network and nothing else."""

    def test_a_plain_sheet_reproduces_a_resistor_network_exactly(self):
        rows, cols = 6, 9
        stackup = one_layer()
        mesh = full_mesh(stackup, rows, cols)
        operator = SheetInductanceOperator((rows, cols), PITCH_M, stackup)
        terminals = [
            Terminal("in", 0, ((rows // 2, 0),), 1.0),
            Terminal("out", 0, ((rows // 2, cols - 1),), -1.0),
        ]
        solution = solve_sheet_case(mesh, operator, terminals, frequency_hz=0.0)
        self.assertTrue(solution.converged)

        expected = resistor_network_voltages(
            rows, cols, (rows // 2, 0), (rows // 2, cols - 1)
        )
        got = np.array(
            [
                solution.node_voltage[mesh.node_index[(0, row, col)]].real
                for row in range(rows)
                for col in range(cols)
            ]
        )
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-18)

    def test_the_only_path_between_layers_carries_the_whole_current(self):
        rows, cols = 6, 9
        stackup = two_layers()
        via = ViaBranch(rows // 2, cols // 2, 0, 1, resistance_ohm=1e-3)
        mesh = full_mesh(stackup, rows, cols, vias=(via,))
        operator = SheetInductanceOperator(
            (rows, cols),
            PITCH_M,
            stackup,
            vertical_levels=mesh.vertical_levels,
        )
        terminals = [
            Terminal("in", 0, ((rows // 2, 0),), 1.0),
            Terminal("out", 1, ((rows // 2, cols - 1),), -1.0),
        ]
        solution = solve_sheet_case(mesh, operator, terminals, frequency_hz=0.0)
        self.assertTrue(solution.converged)
        self.assertAlmostEqual(solution.branch_current[-1].real, 1.0, places=9)

    def test_a_wider_conductor_drops_less_voltage(self):
        stackup = one_layer()
        spans = []
        for rows in (3, 9):
            mesh = full_mesh(stackup, rows, 12)
            operator = SheetInductanceOperator((rows, 12), PITCH_M, stackup)
            terminals = [
                Terminal("in", 0, tuple((row, 0) for row in range(rows)), 1.0),
                Terminal("out", 0, tuple((row, 11) for row in range(rows)), -1.0),
            ]
            spans.append(
                solve_sheet_case(
                    mesh, operator, terminals, frequency_hz=0.0
                ).voltage_span_v()
            )
        self.assertLess(spans[1], spans[0])
        # Three times the width, a third of the drop.
        self.assertAlmostEqual(spans[0] / spans[1], 3.0, places=6)


class AlternatingCurrentTests(unittest.TestCase):
    """Above zero frequency the inductance is applied by transform, not formed."""

    def setUp(self):
        self.rows, self.cols = 5, 6
        self.stackup = two_layers()
        via = ViaBranch(2, 3, 0, 1, resistance_ohm=1e-3, inductance_h=1e-9)
        self.mesh = full_mesh(self.stackup, self.rows, self.cols, vias=(via,))
        self.operator = SheetInductanceOperator(
            (self.rows, self.cols),
            PITCH_M,
            self.stackup,
            vertical_levels=self.mesh.vertical_levels,
        )
        self.terminals = [
            Terminal("in", 0, ((2, 0),), 2.0),
            Terminal("out", 1, ((2, self.cols - 1),), -2.0),
        ]

    def test_the_transform_path_matches_a_dense_direct_solve(self):
        for frequency in (0.0, 3e5, 1e8):
            with self.subTest(frequency=frequency):
                solution = solve_sheet_case(
                    self.mesh, self.operator, self.terminals, frequency_hz=frequency
                )
                self.assertTrue(solution.converged)
                expected = dense_reference(
                    self.mesh, self.operator, self.terminals, frequency
                )
                scale = max(np.abs(expected).max(), 1e-30)
                self.assertLess(
                    np.abs(solution.node_voltage - expected).max() / scale, 1e-9
                )

    def test_the_departure_from_the_resistive_answer_is_first_order_in_frequency(self):
        # The reactance enters as j*omega*L against a fixed R, so the potentials
        # have to leave the resistive ones in proportion to the frequency, and
        # to return to them as it goes to zero.  Asserting the slope says more
        # than asserting a threshold: it is the statement that the inductance
        # enters where it should and with the sign it should.
        direct = solve_sheet_case(
            self.mesh, self.operator, self.terminals, frequency_hz=0.0
        )
        scale = np.abs(direct.node_voltage).max()
        departures = []
        for frequency in (1.0, 10.0, 100.0):
            solution = solve_sheet_case(
                self.mesh, self.operator, self.terminals, frequency_hz=frequency
            )
            departures.append(
                float(np.abs(solution.node_voltage - direct.node_voltage).max() / scale)
            )
        self.assertLess(departures[0], 1e-5)
        for smaller, larger in zip(departures, departures[1:]):
            self.assertAlmostEqual(larger / smaller, 10.0, places=3)

    def test_the_near_impedance_keeps_the_couplings_on_a_branch_s_own_layer(self):
        # The layers above and below are left to the Krylov iterations; within
        # a layer (and among the vertical branches) the entries are the
        # operator's near inductance unchanged.
        from electrical.sheet_peec.sheet_peec import _branch_layers, _near_impedance

        omega = 2e6
        resistance = self.mesh.resistances()
        active = np.ones(self.mesh.branch_count, dtype=bool)
        near = _near_impedance(self.mesh, self.operator, omega, resistance, active).tocoo()
        layer = _branch_layers(self.mesh)
        self.assertTrue(np.all(layer[near.row] == layer[near.col]))
        full = self.operator.near_inductance(self.mesh).toarray()
        self.assertTrue(np.any(full[layer[:, None] != layer[None, :]] != 0.0))  # there was something to drop
        expected = np.where(layer[:, None] == layer[None, :], full, 0.0)
        np.testing.assert_array_equal(
            _near_impedance(self.mesh, self.operator, omega, resistance, active).toarray(),
            np.diag(resistance) + 1j * omega * expected,
        )

    def test_the_block_preconditioner_reaches_the_near_preconditioner_solution(self):
        # Block elimination with Z_near factored alone and the Schur complement
        # approximated by the diagonal variant's nodal admittance: the same
        # fixed point as the saddle LU, at a fraction of its memory.
        near = solve_sheet_case(
            self.mesh, self.operator, self.terminals, frequency_hz=3e5, preconditioner="near"
        )
        block = solve_sheet_case(
            self.mesh, self.operator, self.terminals, frequency_hz=3e5, preconditioner="block"
        )
        self.assertTrue(near.converged and block.converged)
        scale = float(np.abs(near.node_voltage).max())
        self.assertLess(float(np.abs(block.node_voltage - near.node_voltage).max()) / scale, 1e-7)
        self.assertLess(
            float(np.abs(block.branch_current - near.branch_current).max())
            / float(np.abs(near.branch_current).max()),
            1e-7,
        )
        self.assertLessEqual(block.iterations, 10 * max(near.iterations, 1))
        with self.assertRaises(ValueError):
            solve_sheet_case(self.mesh, self.operator, self.terminals, frequency_hz=3e5, preconditioner="exact")
        # auto: the small fixture takes the near path and reproduces it exactly.
        auto = solve_sheet_case(
            self.mesh, self.operator, self.terminals, frequency_hz=3e5, preconditioner="auto"
        )
        self.assertEqual(auto.iterations, near.iterations)
        np.testing.assert_allclose(auto.node_voltage, near.node_voltage, rtol=0.0, atol=1e-12 * scale)
        from electrical.sheet_peec.sheet_peec import AUTO_BLOCK_FROM_UNKNOWNS, choose_preconditioner
        self.assertEqual(choose_preconditioner(AUTO_BLOCK_FROM_UNKNOWNS - 1), "near")
        self.assertEqual(choose_preconditioner(AUTO_BLOCK_FROM_UNKNOWNS), "block")

    def test_the_near_block_in_strips_reaches_the_same_solution(self):
        # Dropping the couplings between strips weakens the preconditioner,
        # not the system: GMRES reaches the whole-block solution.
        whole = solve_sheet_case(
            self.mesh, self.operator, self.terminals, frequency_hz=3e5, preconditioner="block", near_tiles=1
        )
        scale = float(np.abs(whole.branch_current).max())
        for tiles in (2, 3, 7):
            strips = solve_sheet_case(
                self.mesh, self.operator, self.terminals, frequency_hz=3e5, preconditioner="block",
                near_tiles=tiles,
            )
            self.assertTrue(strips.converged)
            self.assertLess(float(np.abs(strips.branch_current - whole.branch_current).max()) / scale, 1e-7)
            self.assertLessEqual(strips.iterations, 3 * max(whole.iterations, 1))
        from electrical.sheet_peec.sheet_peec import DEFAULT_NEAR_TILES

        default = solve_sheet_case(
            self.mesh, self.operator, self.terminals, frequency_hz=3e5, preconditioner="block"
        )
        stated = solve_sheet_case(
            self.mesh, self.operator, self.terminals, frequency_hz=3e5, preconditioner="block",
            near_tiles=DEFAULT_NEAR_TILES,
        )
        self.assertEqual(default.branch_current.tobytes(), stated.branch_current.tobytes())
        with self.assertRaises(ValueError):
            solve_sheet_case(self.mesh, self.operator, self.terminals, frequency_hz=3e5, near_tiles=0)

    def test_the_near_strips_are_factored_on_the_calling_thread(self):
        # SciPy's splu run on a worker thread kept its memory after its factors
        # were freed; the strips are factored where the solve was called.
        import threading
        import unittest.mock

        import scipy.sparse.linalg as spla

        from electrical.threads import thread_budget_scope

        calling = threading.get_ident()
        seen = []
        original = spla.splu

        def recording(*args, **kwargs):
            seen.append(threading.get_ident())
            return original(*args, **kwargs)

        with thread_budget_scope(4), unittest.mock.patch.object(spla, "splu", recording):
            solve_sheet_case(
                self.mesh, self.operator, self.terminals, frequency_hz=3e5, preconditioner="block", near_tiles=3
            )
        self.assertTrue(seen)
        self.assertEqual(set(seen), {calling})

    def test_the_near_strips_give_the_same_bits_at_every_thread_budget(self):
        from electrical.threads import thread_budget_scope

        results = []
        for threads in (1, 2, 4):
            with thread_budget_scope(threads):
                solution = solve_sheet_case(
                    self.mesh, self.operator, self.terminals, frequency_hz=3e5, preconditioner="block",
                    near_tiles=3,
                )
            results.append(solution.branch_current.tobytes() + solution.node_voltage.tobytes())
        self.assertTrue(all(result == results[0] for result in results))

    def _excitations(self):
        cols = self.cols
        return [
            self.terminals,
            [Terminal("in", 0, ((1, 0), (3, 0)), 1.5), Terminal("out", 1, ((2, cols - 1),), -1.5)],
            [Terminal("in", 1, ((0, 2),), 0.7), Terminal("out", 0, ((4, 3),), -0.7)],
        ]

    def test_excitations_solved_together_give_the_bits_they_give_alone(self):
        for preconditioner in ("near", "block", "diagonal"):
            for frequency in (0.0, 3e5):
                with self.subTest(preconditioner=preconditioner, frequency=frequency):
                    sets = self._excitations()
                    together = solve_sheet_cases(
                        self.mesh, self.operator, sets, frequency_hz=frequency, preconditioner=preconditioner,
                        restart=7,
                    )
                    for terminals, solution in zip(sets, together):
                        alone = solve_sheet_case(
                            self.mesh, self.operator, terminals, frequency_hz=frequency,
                            preconditioner=preconditioner, restart=7,
                        )
                        self.assertIs(solution.converged, True)  # a Python bool: results go to JSON
                        self.assertEqual(solution.iterations, alone.iterations)
                        self.assertEqual(solution.branch_current.tobytes(), alone.branch_current.tobytes())
                        self.assertEqual(solution.node_voltage.tobytes(), alone.node_voltage.tobytes())

    def test_a_group_cut_to_fit_the_memory_gives_the_same_bits_in_order(self):
        # With room for one or two columns the group is solved in cuts of
        # that width (a cut of one alone); every excitation keeps its bits
        # and its place, and reports the width it shared.
        import electrical.sheet_peec.sheet_peec as sheet

        sets = self._excitations() + self._excitations()[:2]
        whole = solve_sheet_cases(self.mesh, self.operator, sets, frequency_hz=3e5, preconditioner="block", restart=7)
        self.assertEqual([s.lockstep_columns for s in whole], [len(sets)] * len(sets))
        size = self.mesh.branch_count + self.mesh.node_count
        column = sheet.lockstep_column_bytes(size, 7, self.operator)
        original = sheet._available_memory_bytes
        try:
            for room, widths in ((1, [1, 1, 1, 1, 1]), (2, [2, 2, 2, 2, 1])):
                sheet._available_memory_bytes = lambda room=room: int(room * column / sheet.LOCKSTEP_MEMORY_FRACTION)
                cut = solve_sheet_cases(self.mesh, self.operator, sets, frequency_hz=3e5, preconditioner="block", restart=7)
                self.assertEqual([s.lockstep_columns for s in cut], widths)
                for a, b in zip(whole, cut):
                    self.assertEqual(a.branch_current.tobytes(), b.branch_current.tobytes())
                    self.assertEqual(a.node_voltage.tobytes(), b.node_voltage.tobytes())
                    self.assertEqual(a.iterations, b.iterations)
            sheet._available_memory_bytes = lambda: None
            self.assertEqual(sheet.lockstep_width(5, column), 5)
            sheet._available_memory_bytes = lambda: 0
            self.assertEqual(sheet.lockstep_width(5, column), 1)
        finally:
            sheet._available_memory_bytes = original

    def test_a_lockstep_column_counts_its_basis_and_its_transforms(self):
        import electrical.sheet_peec.sheet_peec as sheet

        size = 1000
        rows, cols = self.operator.shape
        layers, levels = len(self.operator.stackup), len(self.operator.vertical_levels)
        counted = 16 * size * (60 + 1 + sheet.LOCKSTEP_WORK_VECTORS) + 2 * (2 * layers + levels) * (
            sheet.LOCKSTEP_FFT_BYTES_PER_CELL * rows * cols
        )
        margin = sheet.LOCKSTEP_ESTIMATE_MARGIN
        self.assertEqual(sheet.lockstep_column_bytes(size, 60, self.operator), int(margin * counted))
        self.assertEqual(
            sheet.lockstep_column_bytes(size, 60, object()), int(margin * 16 * size * (61 + sheet.LOCKSTEP_WORK_VECTORS))
        )

    def test_excitations_driving_different_components_are_solved_apart(self):
        occupancy = np.ones((2, self.rows, self.cols), dtype=bool)
        occupancy[:, :, 3] = False  # two halves, joined only through no copper
        mesh = SheetMesh((self.rows, self.cols), PITCH_M, self.stackup, occupancy)
        operator = SheetInductanceOperator((self.rows, self.cols), PITCH_M, self.stackup)
        left = [Terminal("in", 0, ((1, 0),), 1.0), Terminal("out", 0, ((3, 2),), -1.0)]
        right = [Terminal("in", 0, ((1, 4),), 1.0), Terminal("out", 0, ((3, 5),), -1.0)]
        together = solve_sheet_cases(mesh, operator, [left, right, left], frequency_hz=3e5)
        for terminals, solution in zip([left, right, left], together):
            alone = solve_sheet_case(mesh, operator, terminals, frequency_hz=3e5)
            self.assertEqual(solution.node_voltage.tobytes(), alone.node_voltage.tobytes())
        self.assertNotEqual(together[0].grounded_node, together[1].grounded_node)

    def test_the_two_span_definitions_are_named(self):
        direct = solve_sheet_case(
            self.mesh, self.operator, self.terminals, frequency_hz=0.0
        )
        alternating = solve_sheet_case(
            self.mesh, self.operator, self.terminals, frequency_hz=3e5
        )
        self.assertEqual(direct.voltage_span_definition, "signed_max_minus_min")
        self.assertEqual(
            alternating.voltage_span_definition,
            "maximum_phasor_magnitude_from_reference",
        )

    def test_impedance_grows_with_frequency(self):
        # Compared under one definition, so the growth is the physics and not
        # the change of convention at zero.
        spans = [
            float(
                np.abs(
                    solve_sheet_case(
                        self.mesh,
                        self.operator,
                        self.terminals,
                        frequency_hz=frequency,
                    ).node_voltage
                ).max()
            )
            for frequency in (1.0, 3e5, 1e7, 1e8)
        ]
        self.assertTrue(all(a < b for a, b in zip(spans, spans[1:])))


class MeshTests(unittest.TestCase):
    def test_branches_exist_only_between_two_copper_cells(self):
        stackup = one_layer()
        occupancy = np.zeros((1, 3, 3), dtype=bool)
        occupancy[0, 1, :] = True
        mesh = SheetMesh((3, 3), PITCH_M, stackup, occupancy)
        self.assertEqual(mesh.node_count, 3)
        self.assertEqual(len(mesh.branch_x), 2)
        self.assertEqual(len(mesh.branch_y), 0)

    def test_adjacent_cells_of_two_regions_are_not_joined(self):
        # two nets a clearance narrower than a cell apart sit in adjacent cells
        stackup = one_layer()
        occupancy = np.ones((1, 1, 4), dtype=bool)
        region = np.array([[[1, 1, 2, 2]]])
        mesh = SheetMesh((1, 4), PITCH_M, stackup, occupancy, region=region)
        self.assertEqual(mesh.branch_x, [(0, 0, 0), (0, 0, 2)])
        # a region of 0 (a bridge over a part) joins either side
        bridged = SheetMesh((1, 4), PITCH_M, stackup, occupancy, region=np.array([[[1, 0, 2, 2]]]))
        self.assertEqual(len(bridged.branch_x), 3)
        with self.assertRaises(ValueError):
            SheetMesh((1, 4), PITCH_M, stackup, occupancy, region=np.zeros((1, 1, 3), dtype=int))

    def test_a_via_whose_end_was_cut_away_is_not_a_branch(self):
        stackup = two_layers()
        occupancy = np.ones((2, 3, 3), dtype=bool)
        occupancy[1, 1, 1] = False
        mesh = SheetMesh(
            (3, 3),
            PITCH_M,
            stackup,
            occupancy,
            vias=(ViaBranch(1, 1, 0, 1, resistance_ohm=1e-3),),
        )
        self.assertEqual(mesh.via_branches, ())

    def test_a_case_that_does_not_close_is_refused(self):
        stackup = one_layer()
        mesh = full_mesh(stackup, 4, 4)
        operator = SheetInductanceOperator((4, 4), PITCH_M, stackup)
        with self.assertRaises(ValueError) as caught:
            solve_sheet_case(
                mesh,
                operator,
                [Terminal("in", 0, ((0, 0),), 1.0), Terminal("out", 0, ((3, 3),), -0.5)],
                frequency_hz=0.0,
            )
        self.assertIn("sum to zero", str(caught.exception))

    def test_the_sheet_resistance_of_a_layer_does_not_depend_on_the_pitch(self):
        layer = SheetLayer("F.Cu", 0.0, COPPER_M, RESISTIVITY)
        self.assertAlmostEqual(layer.sheet_resistance_ohm, RESISTIVITY / COPPER_M)

    def test_a_plated_barrel_resistance_comes_from_its_annulus(self):
        # A 0.3mm hole with 25um of plating through a 1.5mm board.
        value = via_resistance(1.5e-3, 3e-4, 2.5e-5)
        self.assertGreater(value, 5e-4)
        self.assertLess(value, 2e-3)
        # Twice the plating, roughly half the resistance.
        thicker = via_resistance(1.5e-3, 3e-4, 5e-5)
        self.assertLess(thicker, value)


class ConnectedComponentTests(unittest.TestCase):
    """A conductor in pieces has one gauge per piece, or none at all."""

    def _two_islands(self):
        stackup = one_layer()
        occupancy = np.zeros((1, 3, 7), dtype=bool)
        occupancy[0, :, 0:3] = True
        occupancy[0, :, 4:7] = True
        mesh = SheetMesh((3, 7), PITCH_M, stackup, occupancy)
        operator = SheetInductanceOperator((3, 7), PITCH_M, stackup)
        return mesh, operator

    def test_an_undriven_island_does_not_make_the_system_singular(self):
        # Grounding one node and assuming the copper is one piece returned NaN
        # with a warning the moment shaping cut an island off a plane.
        mesh, operator = self._two_islands()
        terminals = [
            Terminal("in", 0, ((1, 0),), 1.0),
            Terminal("out", 0, ((1, 2),), -1.0),
        ]
        for frequency in (0.0, 3e5):
            with self.subTest(frequency=frequency):
                solution = solve_sheet_case(
                    mesh, operator, terminals, frequency_hz=frequency
                )
                self.assertTrue(solution.converged)
                self.assertTrue(np.isfinite(solution.node_voltage).all())
                self.assertEqual(solution.undriven_nodes, 9)
                injected = np.zeros(mesh.node_count, dtype=np.complex128)
                for terminal in terminals:
                    node = mesh.node_index[
                        (terminal.layer, *terminal.cells[0])
                    ]
                    injected[node] += terminal.current_a
                closure = (
                    mesh.incidence().T @ solution.branch_current - injected
                )
                self.assertLess(float(np.abs(closure).max()), 1e-8)
                for index, (_layer, _row, col) in enumerate(mesh.branch_x):
                    if col >= 4:
                        self.assertEqual(solution.branch_current[index], 0.0j)

    def test_a_case_split_across_two_islands_is_refused(self):
        mesh, operator = self._two_islands()
        with self.assertRaises(ValueError) as caught:
            solve_sheet_case(
                mesh,
                operator,
                [
                    Terminal("in", 0, ((1, 0),), 1.0),
                    Terminal("out", 0, ((1, 5),), -1.0),
                ],
                frequency_hz=0.0,
            )
        self.assertIn("close its own current", str(caught.exception))

    def test_a_case_driving_nothing_is_refused(self):
        mesh, operator = self._two_islands()
        with self.assertRaises(ValueError) as caught:
            solve_sheet_case(
                mesh, operator, [Terminal("x", 0, ((1, 0),), 0.0)], frequency_hz=0.0
            )
        self.assertIn("carries a terminal", str(caught.exception))

    def test_one_piece_of_copper_reports_no_undriven_nodes(self):
        stackup = one_layer()
        mesh = full_mesh(stackup, 5, 6)
        operator = SheetInductanceOperator((5, 6), PITCH_M, stackup)
        solution = solve_sheet_case(
            mesh,
            operator,
            [
                Terminal("in", 0, ((2, 0),), 1.0),
                Terminal("out", 0, ((2, 5),), -1.0),
            ],
            frequency_hz=0.0,
        )
        self.assertEqual(solution.undriven_nodes, 0)


class MixedThicknessTests(unittest.TestCase):
    """A filament stack makes layers of unequal thickness the normal case."""

    def test_a_stackup_of_unequal_thicknesses_solves_end_to_end(self):
        stackup = SheetStackup(
            (
                SheetLayer("thin", 0.0, 3e-5, RESISTIVITY),
                SheetLayer("thick", -1e-4, 1.2e-4, RESISTIVITY),
            )
        )
        # Joined at both ends, so the current has a route through either layer
        # and divides by conductance.  With one via in the middle it would not:
        # everything upstream of it stays on the layer it entered.
        mesh = full_mesh(stackup, 4, 6, vias=tuple(
            ViaBranch(row, col, 0, 1, resistance_ohm=1e-5)
            for col in (0, 5)
            for row in range(4)
        ))
        operator = SheetInductanceOperator(
            (4, 6), PITCH_M, stackup, vertical_levels=mesh.vertical_levels
        )
        terminals = [
            Terminal("in", 0, ((2, 0),), 1.0),
            Terminal("out", 1, ((2, 5),), -1.0),
        ]
        for frequency in (0.0, 3e5):
            with self.subTest(frequency=frequency):
                solution = solve_sheet_case(
                    mesh, operator, terminals, frequency_hz=frequency
                )
                self.assertTrue(solution.converged)
        # The thicker layer has a quarter of the sheet resistance, so mid-span
        # it takes the larger share.
        solution = solve_sheet_case(mesh, operator, terminals, frequency_hz=0.0)
        by_layer = [0.0, 0.0]
        for index, (layer, _row, col) in enumerate(mesh.branch_x):
            if col == 2:
                by_layer[layer] += abs(solution.branch_current[index])
        self.assertGreater(by_layer[1], by_layer[0])
        self.assertAlmostEqual(
            by_layer[1] / by_layer[0],
            stackup.layers[0].sheet_resistance_ohm
            / stackup.layers[1].sheet_resistance_ohm,
            delta=0.5,
        )

class VerticalOperatorTests(unittest.TestCase):
    """Vertical branches couple to each other and to no in-plane branch."""

    def _stack(self):
        return SheetStackup(
            (
                SheetLayer("a", 0.0, COPPER_M, RESISTIVITY),
                SheetLayer("b", -3e-4, COPPER_M, RESISTIVITY),
                SheetLayer("c", -6e-4, COPPER_M, RESISTIVITY),
            )
        )

    def test_the_transform_matches_the_closed_form_directly(self):
        stackup = self._stack()
        levels = ((0, 1), (1, 2))
        rows, cols = 5, 4
        operator = SheetInductanceOperator(
            (rows, cols), PITCH_M, stackup, vertical_levels=levels
        )
        spans = [abs(stackup.separation_m(a, b)) for a, b in levels]
        centres = [
            (stackup.layers[a].z_m + stackup.layers[b].z_m) / 2.0 for a, b in levels
        ]

        def direct(level_a, cell_a, level_b, cell_b):
            return mutual_partial_inductance(
                vertical_cell(spans[level_a], PITCH_M),
                vertical_cell(spans[level_b], PITCH_M),
                (
                    centres[level_b] - centres[level_a],
                    (cell_b[1] - cell_a[1]) * PITCH_M,
                    (cell_b[0] - cell_a[0]) * PITCH_M,
                ),
                scale_m=PITCH_M,
            )

        currents = np.zeros((len(levels), rows, cols))
        currents[0, 2, 2] = 1.0
        zeros = np.zeros((len(stackup), rows, cols))
        _x, _y, flux = operator.apply(zeros, zeros, currents)
        for level, cell in ((0, (2, 2)), (0, (2, 3)), (0, (3, 2)), (1, (2, 2))):
            with self.subTest(level=level, cell=cell):
                self.assertAlmostEqual(
                    flux[level][cell],
                    direct(0, (2, 2), level, cell),
                    delta=abs(flux[0, 2, 2]) * 1e-9,
                )

    def test_vertical_current_produces_no_in_plane_flux(self):
        stackup = self._stack()
        rows, cols = 5, 4
        operator = SheetInductanceOperator(
            (rows, cols), PITCH_M, stackup, vertical_levels=((0, 1),)
        )
        currents = np.zeros((1, rows, cols))
        currents[0, 2, 2] = 1.0
        zeros = np.zeros((len(stackup), rows, cols))
        flux_x, flux_y, _z = operator.apply(zeros, zeros, currents)
        self.assertEqual(np.abs(flux_x).max(), 0.0)
        self.assertEqual(np.abs(flux_y).max(), 0.0)

    def test_the_neighbour_coupling_is_a_large_fraction_of_the_self_term(self):
        # This is why the self term cannot be added on its own: a neighbouring
        # column carries roughly half as much coupling as the branch itself, and
        # redistribution currents in neighbouring columns often oppose one
        # another, so the mutual terms cancel much of what the self terms claim.
        stackup = self._stack()
        operator = SheetInductanceOperator(
            (8, 8), PITCH_M, stackup, vertical_levels=((0, 1),)
        )
        currents = np.zeros((1, 8, 8))
        currents[0, 4, 4] = 1.0
        zeros = np.zeros((len(stackup), 8, 8))
        _x, _y, flux = operator.apply(zeros, zeros, currents)
        ratio = flux[0, 4, 5] / flux[0, 4, 4]
        self.assertGreater(ratio, 0.3)
        self.assertLess(ratio, 0.8)

    def test_the_near_inductance_is_the_entry_by_entry_table_lookup(self):
        # Four layers, holes, vias over more level pairs than there are
        # layers and one via listed twice: the matrix must be what reading the tables branch by branch
        # gives, to the bit and in the same order within each row.
        stackup = SheetStackup(
            tuple(
                SheetLayer(name, z, COPPER_M, RESISTIVITY)
                for name, z in (("F.Cu", 0.0), ("In1.Cu", -2e-4), ("In2.Cu", -1.3e-3), ("B.Cu", -CORE_GAP_M))
            )
        )
        rows, cols = 9, 11
        occupancy = np.random.default_rng(3).random((4, rows, cols)) > 0.25
        vias = [
            ViaBranch(2, 3, 1, 0, 1e-3),
            ViaBranch(2, 4, 2, 1, 1e-3),
            ViaBranch(6, 7, 3, 0, 1e-3),
            ViaBranch(6, 8, 3, 0, 1e-3),
            ViaBranch(4, 5, 3, 2, 1e-3),
            ViaBranch(5, 5, 2, 0, 1e-3),
        ]
        for via in vias:
            occupancy[:, via.row, via.col] = True
        vias.append(vias[0])
        mesh = SheetMesh((rows, cols), PITCH_M, stackup, occupancy, vias=tuple(vias))
        operator = SheetInductanceOperator((rows, cols), PITCH_M, stackup, vertical_levels=mesh.vertical_levels)
        near = operator.near_inductance(mesh)
        expected = branch_by_branch_near_inductance(operator, mesh)
        for name in ("data", "indices", "indptr"):
            np.testing.assert_array_equal(getattr(near, name), getattr(expected, name))

    def test_a_mesh_joining_layers_the_operator_does_not_know_is_refused(self):
        stackup = two_layers()
        mesh = full_mesh(
            stackup, 4, 4, vias=(ViaBranch(2, 2, 0, 1, resistance_ohm=1e-3),)
        )
        operator = SheetInductanceOperator((4, 4), PITCH_M, stackup)
        with self.assertRaises(ValueError) as caught:
            solve_sheet_case(
                mesh,
                operator,
                [
                    Terminal("in", 0, ((2, 0),), 1.0),
                    Terminal("out", 1, ((2, 3),), -1.0),
                ],
                frequency_hz=3e5,
            )
        self.assertIn("vertical_levels=mesh.vertical_levels", str(caught.exception))

    def test_a_mesh_without_vertical_branches_needs_no_vertical_operator(self):
        stackup = one_layer()
        mesh = full_mesh(stackup, 4, 5)
        self.assertEqual(mesh.vertical_levels, ())
        operator = SheetInductanceOperator((4, 5), PITCH_M, stackup)
        solution = solve_sheet_case(
            mesh,
            operator,
            [
                Terminal("in", 0, ((2, 0),), 1.0),
                Terminal("out", 0, ((2, 4),), -1.0),
            ],
            frequency_hz=3e5,
        )
        self.assertTrue(solution.converged)

    def test_the_vertical_operator_lowers_the_resistance_ratio_below_its_ceiling(self):
        # Omitting the coupling let the answer sit above the loss the mesh can
        # hold given the exact current profile, which is not a value a converged
        # solve can produce.  With the operator it lands below it.
        from electrical.sheet_peec.skin_filaments import filament_links, graded_filaments

        thickness, frequency = 3.5e-3, 3e5
        rows, cols = 6, 12
        cut = graded_filaments(thickness, 0.0, frequency)
        stackup = SheetStackup(cut.layers("inlay", RESISTIVITY))
        occupancy = np.ones((len(stackup), rows, cols), dtype=bool)
        mesh = SheetMesh(
            (rows, cols),
            PITCH_M,
            stackup,
            occupancy,
            vias=filament_links(cut, (rows, cols), PITCH_M),
        )
        operator = SheetInductanceOperator(
            (rows, cols),
            PITCH_M,
            stackup,
            vertical_levels=mesh.vertical_levels,
        )
        self.assertEqual(len(mesh.vertical_levels), len(cut) - 1)
        total = cut.total_thickness_m
        terminals = []
        for layer, thick in enumerate(cut.thicknesses_m):
            share = thick / total
            terminals.append(
                Terminal(f"in{layer}", layer, tuple((r, 0) for r in range(rows)), share)
            )
            terminals.append(
                Terminal(
                    f"out{layer}",
                    layer,
                    tuple((r, cols - 1) for r in range(rows)),
                    -share,
                )
            )

        def middle_drop(frequency_hz):
            solution = solve_sheet_case(
                mesh, operator, terminals, frequency_hz=frequency_hz,
                tolerance=1e-8, restart=120, max_iterations=40,
            )
            def column(index):
                return np.mean(
                    [
                        solution.node_voltage[mesh.node_index[(layer, row, index)]]
                        for layer in range(len(stackup))
                        for row in range(rows)
                    ]
                )
            return column(cols // 4) - column(3 * cols // 4)

        ratio = (middle_drop(frequency) / middle_drop(0.0)).real
        # Above one because the inlay is many skin depths thick, and below the
        # measured ceiling for this pitch and filament cut.
        self.assertGreater(ratio, 1.5)
        self.assertLess(ratio, 4.93)

if __name__ == "__main__":
    unittest.main()
