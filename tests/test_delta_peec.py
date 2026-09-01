import unittest

import numpy as np

from electrical.dice_peec.delta_peec import DeltaQuadraticScorer, FFTInteraction2D, SparseDelta
from electrical.dice_peec.lowmem_peec import approximate_energy, exact_energy


class DeltaPeecTests(unittest.TestCase):
    def test_delta_identity_matches_full_fft(self):
        rng = np.random.default_rng(4)
        shape = (24, 20)
        base = rng.normal(size=shape)
        delta = SparseDelta.from_changes(
            [(2, 3, 0.7), (7, 11, -1.2), (2, 3, 0.1), (18, 4, 0.4)]
        )
        scorer = DeltaQuadraticScorer(FFTInteraction2D(shape), base)
        self.assertTrue(
            np.isclose(scorer.energy(delta), scorer.full_energy(delta), rtol=1e-11)
        )

    def test_duplicate_changes_are_merged(self):
        delta = SparseDelta.from_changes([(1, 2, 1.0), (1, 2, -0.25)])
        self.assertEqual(delta.size, 1)
        self.assertEqual(delta.values[0], 0.75)

    def test_unit_blocks_reproduce_exact_energy(self):
        points = np.asarray([[2, 3], [2, 4], [7, 9], [11, 1]], dtype=np.float64)
        values = np.asarray([1.0, -0.4, 0.8, 1.2])
        reference = exact_energy(points, values)
        estimate = approximate_energy(
            points, values, block_size=1, near_radius=0, order=0,
            storage_dtype=np.float64
        )
        self.assertTrue(np.isclose(reference, estimate, rtol=1e-12))


if __name__ == "__main__":
    unittest.main()
