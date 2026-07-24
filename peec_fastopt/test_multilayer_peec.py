import unittest
from unittest.mock import patch

import numpy as np

from peec_fastopt.delta_peec import DeltaQuadraticScorer, FFTInteraction2D, SparseDelta
from peec_fastopt.layout_ops import CandidateEdit, SegmentOp, ViaOp, compile_candidate
from peec_fastopt.lowmem_25d import (
    approximate_multilayer_energy,
    exact_multilayer_energy,
    fidelity_cascade_scores,
)
from peec_fastopt.multilayer_peec import (
    FFTInteraction25D,
    MultilayerDeltaScorer,
    SparseDeltaML,
    ViaSet,
    ViaSpec,
)
from peec_fastopt.stackup import Stackup


class StackupTests(unittest.TestCase):
    def test_dual_sided_and_index(self):
        stack = Stackup.dual_sided(board_thickness_mm=1.6)
        self.assertEqual(stack.n_layers, 2)
        self.assertEqual(stack.index("F.Cu"), 0)
        self.assertEqual(stack.index("B.Cu"), 1)
        self.assertAlmostEqual(stack.separation_m(0, 1), 1.6e-3)

    def test_invalid_stackup_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            Stackup.dual_sided(board_thickness_mm=0.0)
        with self.assertRaises(ValueError):
            Stackup(layer_names=("F.Cu",), z_mm=(float("nan"),))


class MultilayerOperatorTests(unittest.TestCase):
    def test_single_layer_matches_2d_operator(self):
        rng = np.random.default_rng(5)
        shape = (16, 14)
        stack = Stackup(layer_names=("F.Cu",), z_mm=(0.0,))
        base2d = rng.normal(size=shape)
        base3d = base2d[None, :, :]
        op2d = FFTInteraction2D(shape, softening=0.75)
        op25 = FFTInteraction25D(
            shape, stack, cell_size_m=0.2e-3, softening=0.75
        )
        field2d = op2d.apply(base2d)
        field25 = op25.apply(base3d)[0]
        np.testing.assert_allclose(field25, field2d, rtol=1e-11, atol=1e-11)
        self.assertTrue(
            np.isclose(op25.energy(base3d), op2d.energy(base2d), rtol=1e-11)
        )

    def test_interlayer_kernel_uses_stackup_separation(self):
        shape = (8, 8)
        stack = Stackup(layer_names=("F.Cu", "B.Cu"), z_mm=(0.0, -1.6))
        op = FFTInteraction25D(shape, stack, cell_size_m=0.2e-3, softening=0.5)
        same = op.kernel_value(0, 0, np.asarray(3.0), np.asarray(4.0))
        cross = op.kernel_value(0, 1, np.asarray(3.0), np.asarray(4.0))
        # Same in-plane distance; cross-layer must be weaker (larger |r|).
        self.assertLess(float(cross), float(same))

    def test_delta_identity_multilayer(self):
        rng = np.random.default_rng(9)
        shape = (18, 16)
        stack = Stackup.dual_sided(board_thickness_mm=1.6)
        base = rng.normal(size=(2, shape[0], shape[1])) * 0.1
        # Put copper on both layers.
        base[0, 3:8, 4:10] += 1.0
        base[1, 10:14, 2:7] += 1.0
        delta = SparseDeltaML.from_changes(
            [
                (0, 4, 5, 0.5),
                (0, 5, 6, -0.25),
                (1, 11, 3, 0.8),
                (1, 12, 4, -0.4),
            ]
        )
        vias = ViaSet(
            vias=(
                ViaSpec(
                    row=6,
                    col=5,
                    layer_from=0,
                    layer_to=1,
                    inductance_h=2e-9,
                    resistance_ohm=0.01,
                ),
            )
        )
        scorer = MultilayerDeltaScorer(
            FFTInteraction25D(shape, stack, cell_size_m=0.2e-3),
            base,
            vias=vias,
            frequency_hz=3e5,
        )
        incremental = scorer.energy(delta)
        full = scorer.full_energy(delta)
        self.assertTrue(np.isclose(incremental, full, rtol=1e-10, atol=1e-10))

    def test_via_set_change_identity(self):
        shape = (12, 12)
        stack = Stackup.dual_sided()
        base = np.zeros((2, *shape), dtype=np.float64)
        base[0, 4, 4] = 1.0
        base[1, 4, 4] = 1.0
        base_vias = ViaSet()
        candidate_vias = ViaSet(
            vias=(
                ViaSpec(
                    row=4,
                    col=4,
                    layer_from=0,
                    layer_to=1,
                    inductance_h=1e-9,
                    important=True,
                ),
            )
        )
        scorer = MultilayerDeltaScorer(
            FFTInteraction25D(shape, stack, cell_size_m=0.2e-3),
            base,
            vias=base_vias,
            frequency_hz=1e6,
        )
        delta = SparseDeltaML.empty()
        with_via = scorer.energy(delta, vias=candidate_vias)
        self.assertTrue(
            np.isclose(
                with_via,
                scorer.full_energy(delta, vias=candidate_vias),
                rtol=1e-12,
            )
        )
        # Lumped via energy is strictly positive when both pads are occupied.
        self.assertGreater(with_via, scorer.base_energy)

    def test_sparse_score_does_not_materialize_full_delta_volume(self):
        shape = (10, 10)
        stack = Stackup.dual_sided()
        scorer = MultilayerDeltaScorer(
            FFTInteraction25D(shape, stack), np.zeros((2, *shape))
        )
        delta = SparseDeltaML.from_changes([(0, 2, 3, 1.0)])
        with patch.object(
            SparseDeltaML,
            "dense",
            side_effect=AssertionError("sparse scoring called dense()"),
        ):
            self.assertTrue(np.isfinite(scorer.energy(delta)))

    def test_identical_vias_are_idempotent_not_fourfold(self):
        via = ViaSpec(row=2, col=3, layer_from=0, layer_to=1)
        deduplicated = ViaSet.from_iterable((via, via))
        self.assertEqual(len(deduplicated), 1)
        self.assertAlmostEqual(
            deduplicated.vias[0].lumped_weight(1e6), via.lumped_weight(1e6)
        )
        with self.assertRaises(ValueError):
            ViaSet.from_iterable(
                (
                    via,
                    ViaSpec(
                        row=2,
                        col=3,
                        layer_from=0,
                        layer_to=1,
                        inductance_h=2e-9,
                    ),
                )
            )


class LayoutOpsTests(unittest.TestCase):
    def test_compile_segment_and_via(self):
        stack = Stackup.dual_sided()
        edit = CandidateEdit(
            segments=[
                SegmentOp(
                    action="add",
                    layer="F.Cu",
                    cells=((1, 1), (1, 2), (1, 3)),
                    value=1.0,
                ),
                SegmentOp(
                    action="remove",
                    layer="B.Cu",
                    cells=((5, 5),),
                    value=1.0,
                ),
            ],
            vias=[
                ViaOp(
                    action="add",
                    row=1,
                    col=3,
                    layer_from="F.Cu",
                    layer_to="B.Cu",
                    important=True,
                    touch_pads=True,
                )
            ],
        )
        compiled = compile_candidate(edit, stack)
        self.assertTrue(compiled.high_risk_topology)
        self.assertGreater(compiled.occupancy_delta.size, 0)
        self.assertEqual(len(compiled.vias), 1)
        # Geometry is a union: the segment endpoint and via pad on F.Cu occupy
        # the same cell once, while B.Cu receives the other pad.
        dense = compiled.occupancy_delta.dense(2, (8, 8))
        self.assertAlmostEqual(dense[0, 1, 3], 1.0)
        self.assertAlmostEqual(dense[1, 1, 3], 1.0)  # via pad only
        self.assertAlmostEqual(dense[1, 5, 5], -1.0)

    def test_remove_via_from_base(self):
        stack = Stackup.dual_sided()
        base = ViaSet(
            vias=(
                ViaSpec(row=2, col=2, layer_from=0, layer_to=1),
                ViaSpec(row=3, col=3, layer_from=0, layer_to=1),
            )
        )
        edit = CandidateEdit(
            vias=[ViaOp(action="remove", row=2, col=2, layer_from=0, layer_to=1)]
        )
        compiled = compile_candidate(edit, stack, base_vias=base)
        self.assertEqual(len(compiled.vias), 1)
        self.assertEqual(compiled.vias.vias[0].normalized()[:2], (3, 3))
        self.assertTrue(compiled.high_risk_topology)

    def test_untouched_base_via_does_not_promote_segment_only_edit(self):
        stack = Stackup.dual_sided()
        base = ViaSet(
            vias=(ViaSpec(row=2, col=2, layer_from=0, layer_to=1, important=True),)
        )
        edit = CandidateEdit(
            segments=[SegmentOp(action="add", layer=0, cells=((4, 4),))]
        )
        compiled = compile_candidate(edit, stack, base_vias=base)
        self.assertFalse(compiled.high_risk_topology)


class Lowmem25DTests(unittest.TestCase):
    def test_block_size_one_matches_exact(self):
        stack = Stackup.dual_sided(board_thickness_mm=1.6)
        volume = np.zeros((2, 10, 10), dtype=np.float64)
        volume[0, 2, 3] = 1.0
        volume[0, 2, 4] = -0.4
        volume[1, 7, 8] = 0.8
        volume[1, 6, 1] = 1.2
        cell = 0.2e-3
        exact = exact_multilayer_energy(volume, stack, cell_size_m=cell)
        approx = approximate_multilayer_energy(
            volume,
            stack,
            cell_size_m=cell,
            block_size=1,
            near_radius=0.0,
            order=0,
            storage_dtype=np.float64,
        )
        self.assertTrue(np.isclose(exact, approx, rtol=1e-11))

    def test_fidelity_cascade_keys(self):
        stack = Stackup.dual_sided()
        volume = np.zeros((2, 20, 20))
        volume[0, 5:8, 5:9] = 1.0
        volume[1, 12:15, 10:14] = 0.7
        scores = fidelity_cascade_scores(volume, stack, cell_size_m=0.2e-3)
        self.assertIn("coarse_radius8", scores)
        self.assertIn("split_error", scores)
        self.assertGreaterEqual(scores["split_error"], 0.0)

    def test_invalid_lowmem_geometry_is_rejected(self):
        stack = Stackup.dual_sided()
        volume = np.zeros((2, 2, 2))
        with self.assertRaises(ValueError):
            exact_multilayer_energy(volume, stack, cell_size_m=0.0)
        with self.assertRaises(ValueError):
            approximate_multilayer_energy(
                volume,
                stack,
                cell_size_m=0.2e-3,
                block_size=0,
                near_radius=1.0,
            )


class SparseDeltaMLTests(unittest.TestCase):
    def test_2d_style_changes_default_layer(self):
        delta = SparseDeltaML.from_changes([(1, 2, 0.5), (1, 2, 0.25)])
        self.assertEqual(delta.size, 1)
        self.assertEqual(int(delta.layers[0]), 0)
        self.assertAlmostEqual(float(delta.values[0]), 0.75)

    def test_negative_coordinates_are_rejected_instead_of_wrapping(self):
        delta = SparseDeltaML.from_changes([(0, -1, 0, 1.0)])
        with self.assertRaises(IndexError):
            delta.dense(1, (2, 2))
        with self.assertRaises(TypeError):
            SparseDeltaML.from_changes([(0, 1.5, 0, 1.0)])

    def test_2d_delta_path_still_consistent(self):
        # Guard: classic single-layer API remains the production scalar path.
        rng = np.random.default_rng(1)
        shape = (12, 10)
        base = rng.normal(size=shape)
        delta = SparseDelta.from_changes([(2, 3, 0.5), (4, 5, -0.2)])
        scorer = DeltaQuadraticScorer(FFTInteraction2D(shape), base)
        self.assertTrue(
            np.isclose(scorer.energy(delta), scorer.full_energy(delta), rtol=1e-11)
        )


if __name__ == "__main__":
    unittest.main()
