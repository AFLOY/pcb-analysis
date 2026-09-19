import math
import unittest

import numpy as np

from electrical.sheet_peec.skin_filaments import (
    FilamentStack,
    filament_links,
    graded_filaments,
    resistance_ratio,
    skin_depth_m,
    slab_surface_impedance,
    uniform_through_thickness,
)

COPPER_FOIL_M = 3.5e-5
INLAY_M = 3.5e-3
SWITCHING_HZ = 3e5


class SkinDepthTests(unittest.TestCase):
    def test_copper_skin_depth_matches_the_textbook_values(self):
        # sqrt(rho / (pi f mu0)) for copper, to the figures usually quoted.
        self.assertAlmostEqual(skin_depth_m(1e3) * 1e3, 2.09, places=2)
        self.assertAlmostEqual(skin_depth_m(1e5) * 1e3, 0.209, places=3)
        self.assertAlmostEqual(skin_depth_m(SWITCHING_HZ) * 1e6, 120.7, places=1)

    def test_direct_current_has_no_skin_depth(self):
        self.assertEqual(skin_depth_m(0.0), math.inf)

    def test_foil_is_uniform_at_the_switching_frequency_and_the_inlay_is_not(self):
        self.assertTrue(uniform_through_thickness(COPPER_FOIL_M, SWITCHING_HZ))
        self.assertFalse(uniform_through_thickness(INLAY_M, SWITCHING_HZ))
        # The inlay stops being uniform in the low hundreds of hertz.
        self.assertTrue(uniform_through_thickness(INLAY_M, 50.0))
        self.assertFalse(uniform_through_thickness(INLAY_M, 500.0))


class SlabImpedanceTests(unittest.TestCase):
    """The analytic reference the filament cut has to reproduce."""

    def test_the_zero_frequency_limit_is_the_sheet_resistance(self):
        impedance = slab_surface_impedance(INLAY_M, 0.0)
        self.assertAlmostEqual(impedance.real, 1.724e-8 / INLAY_M)
        self.assertEqual(impedance.imag, 0.0)
        self.assertAlmostEqual(resistance_ratio(INLAY_M, 0.0), 1.0)

    def test_a_slab_many_skin_depths_thick_conducts_a_skin_depth_at_each_face(self):
        depth = skin_depth_m(1e7)
        impedance = slab_surface_impedance(INLAY_M, 1e7)
        self.assertAlmostEqual(impedance.real, 1.724e-8 / (2.0 * depth), places=9)
        # Equal real and imaginary parts is the signature of the deep limit.
        self.assertAlmostEqual(impedance.real, impedance.imag, places=9)

    def test_the_resistance_ratio_grows_with_frequency(self):
        ratios = [resistance_ratio(INLAY_M, f) for f in (0.0, 1e4, 1e5, SWITCHING_HZ)]
        self.assertTrue(all(a < b for a, b in zip(ratios, ratios[1:])))
        # At the switching frequency the inlay carries its current in a small
        # fraction of its thickness, so its resistance is an order of magnitude
        # above the direct-current value.  Treating it as one filament would
        # miss that by the same factor.
        self.assertGreater(ratios[-1], 10.0)

    def test_foil_barely_departs_from_its_direct_current_resistance(self):
        self.assertLess(resistance_ratio(COPPER_FOIL_M, SWITCHING_HZ), 1.01)


class GradingTests(unittest.TestCase):
    def test_a_conductor_thin_against_the_skin_depth_stays_one_filament(self):
        stack = graded_filaments(COPPER_FOIL_M, 0.0, SWITCHING_HZ)
        self.assertEqual(len(stack), 1)
        self.assertAlmostEqual(stack.thicknesses_m[0], COPPER_FOIL_M)
        self.assertAlmostEqual(stack.centers_m[0], 0.0)

    def test_the_cut_conserves_the_conductor(self):
        for frequency in (1e3, 1e4, 1e5, SWITCHING_HZ, 1e6):
            stack = graded_filaments(INLAY_M, 0.0, frequency)
            self.assertAlmostEqual(stack.total_thickness_m, INLAY_M, places=12)

    def test_the_faces_are_resolved_and_the_middle_is_not(self):
        stack = graded_filaments(INLAY_M, 0.0, SWITCHING_HZ)
        depth = skin_depth_m(SWITCHING_HZ)
        self.assertLessEqual(stack.thicknesses_m[0], depth / 2.0 + 1e-12)
        self.assertLessEqual(stack.thicknesses_m[-1], depth / 2.0 + 1e-12)
        middle = len(stack) // 2
        self.assertGreater(stack.thicknesses_m[middle], depth)

    def test_the_cut_is_symmetric_about_the_conductors_mid_plane(self):
        stack = graded_filaments(INLAY_M, -1.0e-3, SWITCHING_HZ)
        self.assertEqual(
            [round(value, 12) for value in stack.thicknesses_m],
            [round(value, 12) for value in reversed(stack.thicknesses_m)],
        )
        self.assertAlmostEqual(
            sum(
                thickness * center
                for thickness, center in zip(stack.thicknesses_m, stack.centers_m)
            )
            / INLAY_M,
            -1.0e-3,
            places=12,
        )

    def test_the_filaments_tile_the_conductor_without_gap_or_overlap(self):
        stack = graded_filaments(INLAY_M, 0.5e-3, SWITCHING_HZ)
        top = 0.5e-3 + INLAY_M / 2.0
        for center, thickness in zip(stack.centers_m, stack.thicknesses_m):
            self.assertAlmostEqual(center + thickness / 2.0, top, places=12)
            top = center - thickness / 2.0
        self.assertAlmostEqual(top, 0.5e-3 - INLAY_M / 2.0, places=12)

    def test_a_higher_frequency_takes_more_filaments(self):
        counts = [len(graded_filaments(INLAY_M, 0.0, f)) for f in (1e3, 1e4, 1e5, 1e6)]
        self.assertTrue(all(a <= b for a, b in zip(counts, counts[1:])))
        self.assertGreater(counts[-1], counts[0])

    def test_the_cut_is_bounded(self):
        stack = graded_filaments(INLAY_M, 0.0, 1e9, maximum_filaments=8)
        self.assertLessEqual(len(stack), 8)
        self.assertAlmostEqual(stack.total_thickness_m, INLAY_M, places=12)

    def test_the_filaments_become_stackup_layers(self):
        stack = graded_filaments(INLAY_M, 0.0, SWITCHING_HZ)
        layers = stack.layers("inlay")
        self.assertEqual(len(layers), len(stack))
        self.assertEqual(layers[0].name, "inlay#0")
        # Front to back: the first filament sits highest.
        self.assertTrue(all(a.z_m > b.z_m for a, b in zip(layers, layers[1:])))
        for layer, thickness in zip(layers, stack.thicknesses_m):
            self.assertAlmostEqual(layer.thickness_m, thickness)

    def test_bad_parameters_are_refused(self):
        for kwargs in (
            {"cells_per_skin_depth": 0.0},
            {"growth": 0.5},
            {"maximum_filaments": 0},
        ):
            with self.assertRaises(ValueError):
                graded_filaments(INLAY_M, 0.0, SWITCHING_HZ, **kwargs)
        with self.assertRaises(ValueError):
            graded_filaments(-1.0, 0.0, SWITCHING_HZ)



class FilamentLinkTests(unittest.TestCase):
    """Cutting a conductor into filaments must not cut the conductor."""

    def test_every_neighbouring_pair_is_joined_at_every_cell(self):
        stack = graded_filaments(INLAY_M, 0.0, SWITCHING_HZ)
        links = filament_links(stack, (4, 5), 2e-4)
        self.assertEqual(len(links), (len(stack) - 1) * 4 * 5)
        joined = {(link.upper_layer, link.lower_layer) for link in links}
        self.assertEqual(
            joined, {(index, index + 1) for index in range(len(stack) - 1)}
        )

    def test_a_link_carries_the_copper_between_two_mid_planes(self):
        stack = graded_filaments(INLAY_M, 0.0, SWITCHING_HZ)
        pitch = 2e-4
        links = filament_links(stack, (1, 1), pitch)
        for index, link in enumerate(links):
            span = (
                stack.thicknesses_m[index] + stack.thicknesses_m[index + 1]
            ) / 2.0
            self.assertAlmostEqual(
                link.resistance_ohm, 1.724e-8 * span / (pitch * pitch)
            )

    def test_links_follow_the_copper_when_cells_are_cut_away(self):
        stack = graded_filaments(INLAY_M, 0.0, SWITCHING_HZ)
        occupancy = np.ones((len(stack), 3, 3), dtype=bool)
        occupancy[2, 1, 1] = False
        links = filament_links(stack, (3, 3), 2e-4, occupancy=occupancy)
        self.assertEqual(len(links), (len(stack) - 1) * 9 - 2)

    def test_a_conductor_left_as_one_filament_needs_no_links(self):
        stack = graded_filaments(COPPER_FOIL_M, 0.0, SWITCHING_HZ)
        self.assertEqual(filament_links(stack, (4, 4), 2e-4), ())

if __name__ == "__main__":
    unittest.main()
