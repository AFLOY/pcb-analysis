import unittest

from electrical.dice_peec.pypeec_memory import (
    COMPLEX_BYTES,
    describe_estimate,
    estimate_pypeec_memory,
)


def geometry(n, *, conductive=None, overlap_domains=False):
    """Build the part of a PyPEEC geometry the estimator reads."""
    nx, ny, nz = n
    total = nx * ny * nz
    count = total if conductive is None else conductive
    body = list(range(count))
    domains = {"copper_body": body}
    if overlap_domains:
        # A voxel claimed by two domains is still one voxel.
        domains["terminal_000"] = body[: max(1, count // 2)]
    return {"data_voxelize": {"param": {"n": [nx, ny, nz]}, "domain_index": domains}}


ELECTRIC = {"material_def": {"copper": {"material_type": "electric"}}}
MAGNETIC = {
    "material_def": {
        "copper": {"material_type": "electric"},
        "core": {"material_type": "magnetic"},
    }
}
SPLIT_ON = {"dense_options": {"split": True}}


class VoxelBoxTests(unittest.TestCase):
    def test_operator_bytes_match_pypeec_own_footprint_expression(self):
        # PyPEEC 5.8.0 logs nnz = (2nx)(2ny)(2nz)*nd_in at complex128.  An
        # electric-only problem prepares the scalar potential operator and the
        # three-component inductance operator, so nd_in totals four.
        estimate = estimate_pypeec_memory(geometry((10, 8, 4)), ELECTRIC, SPLIT_ON)
        embedded = 20 * 16 * 8
        self.assertEqual(estimate.embedded_count, embedded)
        self.assertEqual(estimate.fft_tensor_bytes, COMPLEX_BYTES * embedded * 4)

    def test_magnetic_domain_adds_the_coupling_operator(self):
        electric = estimate_pypeec_memory(geometry((10, 8, 4)), ELECTRIC, SPLIT_ON)
        magnetic = estimate_pypeec_memory(geometry((10, 8, 4)), MAGNETIC, SPLIT_ON)
        self.assertNotIn("coupling", electric.channels)
        self.assertEqual(magnetic.channels["coupling"], 3)
        self.assertEqual(
            magnetic.fft_tensor_bytes, electric.fft_tensor_bytes // 4 * 7
        )

    def test_height_drives_the_operators_while_copper_drives_the_vectors(self):
        # This is the shape of the multilayer problem: the conductor barely
        # grows while the bounding box grows with the board's height.  An
        # estimate that scaled off either one alone would be wrong.
        flat = estimate_pypeec_memory(
            geometry((100, 100, 1), conductive=5_000), ELECTRIC, SPLIT_ON
        )
        tall = estimate_pypeec_memory(
            geometry((100, 100, 45), conductive=10_000), ELECTRIC, SPLIT_ON
        )
        self.assertEqual(tall.fft_tensor_bytes, flat.fft_tensor_bytes * 45)
        self.assertEqual(tall.krylov_bytes, flat.krylov_bytes * 2)
        self.assertGreater(tall.required_bytes, 10 * flat.required_bytes)

    def test_conductive_count_ignores_domain_overlap(self):
        plain = estimate_pypeec_memory(
            geometry((10, 10, 2), conductive=50), ELECTRIC, SPLIT_ON
        )
        overlapping = estimate_pypeec_memory(
            geometry((10, 10, 2), conductive=50, overlap_domains=True),
            ELECTRIC,
            SPLIT_ON,
        )
        self.assertEqual(plain.conductive_count, 50)
        self.assertEqual(overlapping.conductive_count, 50)

    def test_split_off_allocates_over_the_box_not_the_unknowns(self):
        box = (60, 60, 40)
        split = estimate_pypeec_memory(
            geometry(box, conductive=1_000), ELECTRIC, {"dense_options": {"split": True}}
        )
        combined = estimate_pypeec_memory(
            geometry(box, conductive=1_000),
            ELECTRIC,
            {"dense_options": {"split": False}},
        )
        self.assertGreater(
            combined.product_workspace_bytes, 50 * split.product_workspace_bytes
        )

    def test_restart_length_is_read_from_the_solver_options(self):
        tolerance = {
            "dense_options": {"split": True},
            "solver_options": {"direct_options": {"n_inner": 5}},
        }
        short = estimate_pypeec_memory(
            geometry((10, 10, 2), conductive=100), ELECTRIC, tolerance
        )
        default = estimate_pypeec_memory(
            geometry((10, 10, 2), conductive=100), ELECTRIC, SPLIT_ON
        )
        self.assertLess(short.krylov_bytes, default.krylov_bytes)
        self.assertIn("keeps 5 vectors", " ".join(short.assumptions))

    def test_required_bytes_holds_back_a_share_for_cufft(self):
        estimate = estimate_pypeec_memory(
            geometry((10, 10, 2)), ELECTRIC, SPLIT_ON, unmeasured_fraction=0.5
        )
        self.assertEqual(
            estimate.required_bytes, int(estimate.countable_bytes * 1.5)
        )
        zero = estimate_pypeec_memory(
            geometry((10, 10, 2)), ELECTRIC, SPLIT_ON, unmeasured_fraction=0.0
        )
        self.assertEqual(zero.required_bytes, zero.countable_bytes)

    def test_malformed_boxes_are_refused(self):
        with self.assertRaises(ValueError):
            estimate_pypeec_memory(geometry((10, 10, 0)))
        with self.assertRaises(ValueError):
            estimate_pypeec_memory(
                {"data_voxelize": {"param": {"n": [4, 4]}, "domain_index": {}}}
            )
        with self.assertRaises(KeyError):
            estimate_pypeec_memory({"name": "board"})
        with self.assertRaises(ValueError):
            estimate_pypeec_memory(geometry((4, 4, 4)), unmeasured_fraction=-0.1)

    def test_description_names_the_box_and_the_total(self):
        text = describe_estimate(
            estimate_pypeec_memory(geometry((10, 8, 4)), ELECTRIC, SPLIT_ON)
        )
        self.assertIn("10x8x4", text)
        self.assertIn("20x16x8", text)
        self.assertIn("MiB", text)


if __name__ == "__main__":
    unittest.main()
