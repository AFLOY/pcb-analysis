import math
import unittest

import numpy as np

from electrical.dice_peec.sheet_inductance import (
    NEAR_RADIUS_CELLS,
    CellGeometry,
    bar_self_inductance,
    build_kernel,
    closed_form_mutual_inductance,
    closed_form_precision,
    far_field_mutual_inductance,
    filament_mutual_inductance,
    mutual_partial_inductance,
    self_partial_inductance,
)

# The mesh these kernels are built for: a 0.2mm grid on 35um copper, with the
# board's own core between the two outer layers.
PCB_CELL = CellGeometry(length_m=2e-4, width_m=2e-4, thickness_m=3.5e-5)
CORE_GAP_M = 1.545e-3


class ClosedFormAgainstKnownResults(unittest.TestCase):
    """Check the closed form against formulas derived independently of it."""

    def test_self_inductance_matches_grovers_bar_formula(self):
        # Grover's expression is asymptotic in the bar's slenderness, so the
        # agreement should be good and should improve with the aspect ratio.
        for width in (1e-3, 1e-4, 1e-5):
            cell = CellGeometry(1e-2, width, width)
            closed = self_partial_inductance(cell)
            grover = bar_self_inductance(1e-2, width, width)
            self.assertLess(abs(closed - grover) / grover, 2e-3)

    def test_mutual_inductance_matches_the_exact_filament_formula(self):
        # Two thin bars far enough apart that their cross-section is
        # irrelevant reduce to two filaments, for which the mutual inductance
        # is known exactly.
        # Separations kept inside the range where the closed form is still
        # well conditioned for a bar this slender; past that the entry point
        # switches to the far-field limit, which this comparison would then be
        # testing instead.
        cell = CellGeometry(1e-2, 1e-5, 1e-5)
        for separation in (3e-4, 1e-3, 3e-3):
            closed = closed_form_mutual_inductance(cell, cell, (0.0, 0.0, separation))
            exact = filament_mutual_inductance(1e-2, separation)
            self.assertLess(abs(closed - exact) / exact, 1e-4)

    def test_self_inductance_is_finite_and_dominates_every_mutual_term(self):
        own = self_partial_inductance(PCB_CELL)
        self.assertTrue(math.isfinite(own))
        self.assertGreater(own, 0.0)
        for offset in ((2e-4, 0, 0), (0, 2e-4, 0), (0, 0, CORE_GAP_M), (2e-4, 2e-4, 0)):
            self.assertLess(
                closed_form_mutual_inductance(PCB_CELL, PCB_CELL, offset), own
            )

    def test_the_operator_is_reciprocal(self):
        a = CellGeometry(2e-4, 3e-4, 3.5e-5)
        b = CellGeometry(2.5e-4, 2e-4, 7e-5)
        offset = (3e-4, -1e-4, CORE_GAP_M)
        forward = closed_form_mutual_inductance(a, b, offset)
        reverse = closed_form_mutual_inductance(
            b, a, tuple(-value for value in offset)
        )
        self.assertAlmostEqual(forward, reverse, delta=abs(forward) * 1e-9)

    def test_coupling_falls_with_separation_in_every_direction(self):
        own = self_partial_inductance(PCB_CELL)
        along = [
            closed_form_mutual_inductance(PCB_CELL, PCB_CELL, (n * 2e-4, 0, 0))
            for n in range(1, 8)
        ]
        across = [
            closed_form_mutual_inductance(PCB_CELL, PCB_CELL, (0, n * 2e-4, 0))
            for n in range(1, 8)
        ]
        self.assertTrue(all(a > b for a, b in zip(along, along[1:])))
        self.assertTrue(all(a > b for a, b in zip(across, across[1:])))
        self.assertLess(along[0], own)

    def test_a_cell_couples_differently_along_and_across_its_current(self):
        # A cell longer than it is wide couples more strongly to the cell
        # beside it, which is nearer, than to the one it points at.  A square
        # cell makes the two offsets the same geometry, so the asymmetry only
        # shows on a rectangular one.
        cell = CellGeometry(length_m=3e-4, width_m=1e-4, thickness_m=3.5e-5)
        along = closed_form_mutual_inductance(cell, cell, (3e-4, 0.0, 0.0))
        across = closed_form_mutual_inductance(cell, cell, (0.0, 1e-4, 0.0))
        self.assertGreater(across, along)
        square = closed_form_mutual_inductance(
            PCB_CELL, PCB_CELL, (2e-4, 0.0, 0.0)
        )
        self.assertAlmostEqual(
            square,
            closed_form_mutual_inductance(PCB_CELL, PCB_CELL, (0.0, 2e-4, 0.0)),
            delta=square * 1e-9,
        )


class CancellationTests(unittest.TestCase):
    """The closed form is exact in exact arithmetic; this is about the other kind."""

    def test_precision_survives_across_the_near_region_and_is_lost_beyond_it(self):
        near = closed_form_precision(
            PCB_CELL, PCB_CELL, (NEAR_RADIUS_CELLS * 2e-4, 0.0, 0.0)
        )
        far = closed_form_precision(PCB_CELL, PCB_CELL, (150 * 2e-4, 0.0, 0.0))
        self.assertGreater(near, 1e-9)
        self.assertLess(far, 1e-12)

    def test_the_closed_form_refuses_rather_than_returning_noise(self):
        with self.assertRaises(ValueError) as caught:
            closed_form_mutual_inductance(PCB_CELL, PCB_CELL, (150 * 2e-4, 0.0, 0.0))
        self.assertIn("cancellation", str(caught.exception))

    def test_the_entry_point_switches_before_precision_is_lost(self):
        # Same offset, through the entry point rather than the closed form.
        value = mutual_partial_inductance(PCB_CELL, PCB_CELL, (150 * 2e-4, 0.0, 0.0))
        self.assertTrue(math.isfinite(value))
        self.assertGreater(value, 0.0)

    def test_the_two_regimes_agree_where_they_meet(self):
        # The crossover is only defensible if the far form is already accurate
        # where the near form is still trustworthy.
        offset = ((NEAR_RADIUS_CELLS + 1) * 2e-4, 0.0, 0.0)
        near = closed_form_mutual_inductance(
            PCB_CELL, PCB_CELL, offset, check_precision=False
        )
        far = far_field_mutual_inductance(PCB_CELL, PCB_CELL, offset)
        self.assertLess(abs(near - far) / near, 1e-3)


class KernelTableTests(unittest.TestCase):
    def test_the_table_is_indexed_by_wrapped_signed_offset(self):
        kernel = build_kernel((16, 16), PCB_CELL, 0.0)
        self.assertEqual(kernel.shape, (16, 16))
        self.assertTrue(np.isfinite(kernel).all())
        self.assertAlmostEqual(kernel[0, 0], self_partial_inductance(PCB_CELL))
        # Entry -1 holds the offset of minus one cell, which for a symmetric
        # cell equals the offset of plus one.
        self.assertAlmostEqual(kernel[0, -1], kernel[0, 1])
        self.assertAlmostEqual(kernel[-1, 0], kernel[1, 0])

    def test_a_square_cell_gives_a_kernel_symmetric_in_its_two_axes(self):
        # The y-directed operator is this table transposed, which is only
        # meaningful if the convention holds.
        kernel = build_kernel((12, 12), PCB_CELL, 0.0)
        np.testing.assert_allclose(kernel, kernel.T, rtol=1e-8, atol=0.0)

    def test_a_rectangular_cell_is_not_symmetric_in_its_two_axes(self):
        kernel = build_kernel((12, 12), CellGeometry(3e-4, 1e-4, 3.5e-5), 0.0)
        # atol has to be zero: the default swamps quantities of order 1e-11
        # and would call any two inductance tables equal.
        self.assertFalse(np.allclose(kernel, kernel.T, atol=0.0))
        self.assertGreater(kernel[1, 0], kernel[0, 1])

    def test_separating_the_layers_weakens_every_coupling(self):
        same = build_kernel((16, 16), PCB_CELL, 0.0)
        apart = build_kernel((16, 16), PCB_CELL, CORE_GAP_M)
        self.assertTrue((apart > 0.0).all())
        self.assertTrue((apart <= same + 1e-18).all())
        # The self term is the one the gap changes most.
        self.assertLess(apart[0, 0], 0.1 * same[0, 0])

    def test_the_table_is_positive_and_falls_away_from_the_origin(self):
        kernel = build_kernel((64, 64), PCB_CELL, 0.0)
        self.assertTrue((kernel > 0.0).all())
        row = kernel[0, 1:32]
        self.assertTrue(all(a > b for a, b in zip(row, row[1:])))

    def test_a_board_sized_table_stays_finite_through_the_crossover(self):
        kernel = build_kernel((174, 174), PCB_CELL, CORE_GAP_M)
        self.assertTrue(np.isfinite(kernel).all())
        self.assertTrue((kernel > 0.0).all())


if __name__ == "__main__":
    unittest.main()
