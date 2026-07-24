import unittest
from types import SimpleNamespace

import numpy as np

from peec_fastopt.cuda_delta import CudaDeltaQuadraticScorer
from peec_fastopt.cuda_pypeec import (
    CudaPeecConfig,
    CudaPyPeecExecutor,
    _pool_limit_for_solve,
    clear_cuda_caches,
    cupy_tolerance,
)
from peec_fastopt.delta_peec import SparseDelta
from peec_fastopt._bench_utils import (
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
        with self.assertRaises(ValueError):
            CudaPeecConfig.from_mapping({"device_id": -1})

    def test_pool_limit_accounts_for_existing_pool_and_caller_cap(self):
        gib = 1024**3
        calculated = _pool_limit_for_solve(
            total_bytes=4 * gib,
            free_bytes=1 * gib,
            pool_total_bytes=2 * gib,
            reserve_bytes=int(0.4 * gib),
            existing_limit=0,
        )
        self.assertEqual(calculated, 3 * gib - int(0.4 * gib))
        capped = _pool_limit_for_solve(
            total_bytes=4 * gib,
            free_bytes=1 * gib,
            pool_total_bytes=2 * gib,
            reserve_bytes=int(0.4 * gib),
            existing_limit=1 * gib,
        )
        self.assertEqual(capped, 1 * gib)

    def test_executor_restores_pool_limit_and_reuses_host_voxel_cache(self):
        class FakeDevice:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        class FakePool:
            def __init__(self):
                self.limit = 0
                self.limit_calls = []
                self.free_calls = 0

            def get_limit(self):
                return self.limit

            def set_limit(self, *, size):
                self.limit = int(size)
                self.limit_calls.append(self.limit)

            def total_bytes(self):
                return 100

            def used_bytes(self):
                return 10

            def free_all_blocks(self):
                self.free_calls += 1

        class FakeRuntime:
            @staticmethod
            def memGetInfo():
                return 800, 1000

            @staticmethod
            def getDeviceProperties(_device_id):
                return {"name": b"Fake GPU"}

            @staticmethod
            def runtimeGetVersion():
                return 13030

        pool = FakePool()
        pinned_pool = FakePool()
        stream = SimpleNamespace(synchronize=lambda: None)
        cuda = SimpleNamespace(
            Device=lambda _device_id: FakeDevice(),
            Stream=SimpleNamespace(null=stream),
            runtime=FakeRuntime(),
            memory=SimpleNamespace(OutOfMemoryError=MemoryError),
        )
        fake_cupy = SimpleNamespace(
            __version__="14.test",
            cuda=cuda,
            get_default_memory_pool=lambda: pool,
            get_default_pinned_memory_pool=lambda: pinned_pool,
        )

        class FakePyPeec:
            multiply_fft = SimpleNamespace(SET=True)

            def __init__(self):
                self.mesher_calls = 0

            def run_mesher_data(self, geometry):
                self.mesher_calls += 1
                return {"voxelized": geometry["name"]}

            @staticmethod
            def run_solver_data(_voxel, _problem, tolerance):
                assert tolerance["dense_options"]["fft_options"]["library"] == "CuPy"
                return {
                    "status": True,
                    "data_sweep": {
                        "target": {
                            "solution_ok": True,
                            "solver_status": {"n_iter": 3, "residuum_val": 1e-8},
                        }
                    },
                }

        clear_cuda_caches()
        fake_pypeec = FakePyPeec()
        executor = CudaPyPeecExecutor(
            {"memory_reserve_fraction": 0.1},
            cupy_module=fake_cupy,
            pypeec_module=fake_pypeec,
        )
        geometry = {"name": "board"}
        tolerance = {"dense_options": {"fft_options": {}}}
        first = executor.execute(geometry, {}, tolerance)
        second = executor.execute(geometry, {}, tolerance)

        self.assertFalse(first.voxel_cache_hit)
        self.assertTrue(second.voxel_cache_hit)
        self.assertEqual(fake_pypeec.mesher_calls, 1)
        self.assertEqual(pool.limit_calls, [800, 0, 800, 0])
        self.assertEqual(pool.free_calls, 2)
        self.assertEqual(pinned_pool.free_calls, 2)
        self.assertEqual(first.device_name, "Fake GPU")
        self.assertTrue(first.execution_report.converged)


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

    def test_packing_normalizes_cuda_index_abi_and_validates_coordinates(self):
        candidate = SparseDelta(
            rows=np.asarray([1], dtype=np.int64),
            cols=np.asarray([2], dtype=np.int64),
            values=np.asarray([0.5]),
        )
        rows, cols, _, _, _ = CudaDeltaQuadraticScorer._pack([candidate])
        self.assertEqual(rows.dtype, np.int32)
        self.assertEqual(cols.dtype, np.int32)
        with self.assertRaises(IndexError):
            CudaDeltaQuadraticScorer._validate_indices(
                np.asarray([-1], dtype=np.int32),
                np.asarray([0], dtype=np.int32),
                (4, 4),
            )
        malformed = SparseDelta(
            rows=np.asarray([1.5]),
            cols=np.asarray([2]),
            values=np.asarray([0.5]),
        )
        with self.assertRaises(TypeError):
            CudaDeltaQuadraticScorer._pack([malformed])

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

        # A CUDA tie does not preserve a meaningful descending CPU order.
        descending = {"a": {"x": 2.0}, "b": {"x": 1.0}}
        tied = {"a": {"x": 1.5}, "b": {"x": 1.5}}
        self.assertFalse(ranking_consistent(descending, tied, "x"))

    def test_candidate_selection_prefers_valid_or_named_candidate(self):
        invalid = {"name": "invalid", "valid": False, "mask_runs": {}}
        valid = {"name": "valid", "valid": True, "mask_runs": {}}
        data = {"candidates": [invalid, valid]}
        self.assertIs(_select_candidate(data, None), valid)
        self.assertIs(_select_candidate(data, "invalid"), invalid)


if __name__ == "__main__":
    unittest.main()
