import unittest

import numpy as np

from peec_fastopt.cuda_delta import CudaDeltaQuadraticScorer
from peec_fastopt.cuda_pypeec import CudaPeecConfig, cupy_tolerance
from peec_fastopt.delta_peec import SparseDelta
from peec_fastopt.plane_opt_benchmark import (
    _select_candidate,
    ranking_consistent,
    relative_difference,
)


class CudaConfigurationTests(unittest.TestCase):
    def test_cupy_tolerance_isolated_from_cpu_configuration(self):
        original = {
            "dense_options": {
                "method": "fft",
                "fft_options": {"library": "SciPy", "scipy_worker": -1},
            }
        }
        configured = cupy_tolerance(original)
        self.assertEqual(original["dense_options"]["fft_options"]["library"], "SciPy")
        self.assertEqual(configured["dense_options"]["fft_options"]["library"], "CuPy")
        self.assertEqual(configured["dense_options"]["fft_options"]["scipy_worker"], 0)

    def test_runtime_configuration_validates_precision_and_reserve(self):
        config = CudaPeecConfig.from_mapping({
            "device_id": 2,
            "precision": "complex128",
            "memory_reserve_fraction": 0.2,
            "release_pool_after_solve": False,
        })
        self.assertEqual(config.device_id, 2)
        self.assertEqual(config.precision, "complex128")
        self.assertEqual(config.memory_reserve_fraction, 0.2)
        self.assertTrue(config.cache_voxel)
        self.assertEqual(config.voxel_cache_entries, 8)
        self.assertFalse(config.release_pool_after_solve)
        self.assertTrue(CudaPeecConfig.from_mapping({}).release_pool_after_solve)
        with self.assertRaises(ValueError):
            CudaPeecConfig.from_mapping({"precision": "float16"})
        with self.assertRaises(ValueError):
            CudaPeecConfig.from_mapping({"memory_reserve_fraction": 1.0})
        with self.assertRaises(ValueError):
            CudaPeecConfig.from_mapping({"voxel_cache_entries": 0})


class CudaDeltaPackingTests(unittest.TestCase):
    def test_variable_size_candidates_are_packed_with_offsets(self):
        candidates = [
            SparseDelta.from_changes([(1, 2, 0.5), (3, 4, -1.0)]),
            SparseDelta.from_changes([]),
            SparseDelta.from_changes([(5, 6, 2.0)]),
        ]
        rows, cols, values, offsets, count = CudaDeltaQuadraticScorer._pack(candidates)
        self.assertEqual(count, 3)
        np.testing.assert_array_equal(offsets, [0, 2, 2, 3])
        np.testing.assert_array_equal(rows, [1, 3, 5])
        np.testing.assert_array_equal(cols, [2, 4, 6])
        np.testing.assert_allclose(values, [0.5, -1.0, 2.0])

    def test_gpu_scores_match_cpu_reference_when_cuda_is_available(self):
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() < 1:
                self.skipTest("no CUDA device")
        except Exception as error:
            self.skipTest(f"CUDA unavailable: {error}")
        from peec_fastopt.delta_peec import DeltaQuadraticScorer, FFTInteraction2D

        rng = np.random.default_rng(3)
        base = rng.normal(size=(24, 20)).astype(np.float32)
        candidates = [
            SparseDelta.from_changes([(2, 3, 0.7), (7, 11, -1.2)]),
            SparseDelta.from_changes([]),
            SparseDelta.from_changes([(18, 4, 0.4)]),
        ]
        reference = DeltaQuadraticScorer(FFTInteraction2D(base.shape), base)
        expected = np.asarray([reference.energy(item) for item in candidates])
        actual = CudaDeltaQuadraticScorer(base).energy_many(candidates)
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


class BenchmarkAcceptanceTests(unittest.TestCase):
    def test_relative_difference_and_meaningful_ranking(self):
        self.assertAlmostEqual(relative_difference(100.0, 99.0), 0.01)
        cpu = {"a": {"x": 1.0}, "b": {"x": 2.0}, "tie": {"x": 1.005}}
        cuda = {"a": {"x": 1.001}, "b": {"x": 1.999}, "tie": {"x": 0.999}}
        self.assertTrue(ranking_consistent(cpu, cuda, "x"))
        cuda["b"]["x"] = 0.5
        self.assertFalse(ranking_consistent(cpu, cuda, "x"))

    def test_candidate_selection_prefers_valid_or_named_candidate(self):
        invalid = {"name": "invalid", "valid": False, "mask_runs": {}}
        valid = {"name": "valid", "valid": True, "mask_runs": {}}
        data = {"candidates": [invalid, valid]}
        self.assertIs(_select_candidate(data, None), valid)
        self.assertIs(_select_candidate(data, "invalid"), invalid)


if __name__ == "__main__":
    unittest.main()
