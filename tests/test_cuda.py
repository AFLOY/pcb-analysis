import unittest
from types import SimpleNamespace

import numpy as np

from electrical.dice_peec.cuda_delta import CudaDeltaQuadraticScorer
from electrical.dice_peec.cuda_pypeec import (
    CudaPeecConfig,
    CudaPeecMemoryError,
    CudaPeecSolveError,
    CudaPyPeecExecutor,
    _pool_limit_for_solve,
    clear_cuda_caches,
    cupy_tolerance,
    voxel_cache_bytes,
)
from electrical.dice_peec.delta_peec import SparseDelta
from electrical.dice_peec._bench_utils import (
    _select_candidate,
    ranking_consistent,
    relative_difference,
)


def _geometry(n, *, conductive):
    """Build a geometry carrying the voxel box the estimator has to read."""
    nx, ny, nz = n
    return {
        "data_voxelize": {
            "param": {"n": [nx, ny, nz], "d": [1e-4, 1e-4, 3.5e-5], "c": [0, 0, 0]},
            "domain_index": {"copper_body": list(range(conductive))},
        }
    }


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
        self.assertTrue(config.preflight_memory)
        self.assertEqual(config.memory_reserve_fraction, 0.2)
        self.assertTrue(config.cache_voxel)
        self.assertEqual(config.voxel_cache_entries, 8)
        self.assertFalse(config.release_pool_after_solve)
        self.assertTrue(CudaPeecConfig.from_mapping({}).release_pool_after_solve)
        with self.assertRaises(ValueError):
            CudaPeecConfig.from_mapping({"precision": "float16"})
        # complex64 was accepted while it changed nothing.  PyPEEC 5.8 solves
        # in complex128, so the request has to be refused rather than recorded.
        with self.assertRaises(ValueError):
            CudaPeecConfig.from_mapping({"precision": "complex64"})
        with self.assertRaises(ValueError):
            cupy_tolerance({}, precision="complex64")
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
                # Byte-scale token values used to make the model too large for
                # the device, which the preflight now correctly refuses.  Give
                # the fake device a plausible size instead.
                return 800 * 1024**2, 1000 * 1024**2

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
                return {
                    "voxelized": geometry["data_voxelize"]["param"]["n"],
                    "payload": np.zeros(64, dtype=np.float64),
                }

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
        geometry = _geometry((8, 6, 4), conductive=40)
        tolerance = {"dense_options": {"split": True, "fft_options": {}}}
        first = executor.execute(geometry, {}, tolerance)
        second = executor.execute(geometry, {}, tolerance)

        self.assertFalse(first.voxel_cache_hit)
        self.assertTrue(second.voxel_cache_hit)
        self.assertEqual(fake_pypeec.mesher_calls, 1)
        # Two solves, each setting a positive cap and restoring the caller's
        # unlimited policy afterwards.
        self.assertEqual(len(pool.limit_calls), 4)
        self.assertEqual(pool.limit_calls[1::2], [0, 0])
        self.assertTrue(all(value > 0 for value in pool.limit_calls[0::2]))
        self.assertEqual(pool.free_calls, 2)
        self.assertEqual(pinned_pool.free_calls, 2)
        self.assertEqual(first.device_name, "Fake GPU")
        self.assertTrue(first.execution_report.converged)
        # The estimate has to reach the metrics, and a host cache keyed by the
        # geometry alone has to account for what it is holding.
        self.assertIsNotNone(first.estimate)
        self.assertEqual(first.estimate.box, (8, 6, 4))
        self.assertGreater(voxel_cache_bytes(), 0)
        self.assertTrue(first.metrics()["precision_request_honored"])


class FakePlanCache:
    """Stand in for CuPy's per-thread FFT plan cache."""

    def __init__(self, size=16, memsize=-1, curr_bytes=0):
        self.size = size
        self.memsize = memsize
        self.curr_bytes = curr_bytes
        self.cleared = 0

    def get_size(self):
        return self.size

    def set_size(self, value):
        self.size = int(value)

    def get_memsize(self):
        return self.memsize

    def set_memsize(self, value):
        self.memsize = int(value)

    def get_curr_size_bytes(self):
        return self.curr_bytes

    def clear(self):
        self.cleared += 1


def _fake_cuda_stack(*, free_mib=800, total_mib=1000, plan_cache=None, solver=None):
    """Build the minimum CuPy and PyPEEC surface the executor touches."""

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

    pool = FakePool()
    pinned = FakePool()
    cache = plan_cache if plan_cache is not None else FakePlanCache()
    cuda = SimpleNamespace(
        Device=lambda _device_id: FakeDevice(),
        Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None)),
        runtime=SimpleNamespace(
            memGetInfo=lambda: (free_mib * 1024**2, total_mib * 1024**2),
            getDeviceProperties=lambda _d: {"name": b"Fake GPU"},
            runtimeGetVersion=lambda: 13030,
        ),
        memory=SimpleNamespace(OutOfMemoryError=MemoryError),
    )
    fake_cupy = SimpleNamespace(
        __version__="14.test",
        cuda=cuda,
        fft=SimpleNamespace(config=SimpleNamespace(get_plan_cache=lambda: cache)),
        get_default_memory_pool=lambda: pool,
        get_default_pinned_memory_pool=lambda: pinned,
    )

    def default_solver(_voxel, _problem, _tolerance):
        return {
            "status": True,
            "data_sweep": {
                "target": {
                    "solution_ok": True,
                    "solver_status": {"n_iter": 3, "residuum_val": 1e-8},
                }
            },
        }

    fake_pypeec = SimpleNamespace(
        multiply_fft=SimpleNamespace(SET=True),
        run_mesher_data=lambda geometry: {"n": geometry["data_voxelize"]["param"]["n"]},
        run_solver_data=solver or default_solver,
    )
    return fake_cupy, fake_pypeec, pool, pinned, cache


class CudaPreflightTests(unittest.TestCase):
    def setUp(self):
        clear_cuda_caches()

    def test_a_model_too_tall_for_the_device_is_refused_before_it_allocates(self):
        cupy_module, pypeec_module, pool, _pinned, _cache = _fake_cuda_stack(
            free_mib=64, total_mib=128
        )
        executor = CudaPyPeecExecutor(
            {}, cupy_module=cupy_module, pypeec_module=pypeec_module
        )
        # A board-height box on a small device: the operators alone exceed it.
        geometry = _geometry((162, 152, 45), conductive=29_000)
        with self.assertRaises(CudaPeecMemoryError) as caught:
            executor.execute(geometry, {}, {"dense_options": {"split": True}})
        message = str(caught.exception)
        self.assertIn("324x304x90", message)
        self.assertIn("Assumptions", message)
        self.assertIsNotNone(caught.exception.estimate)
        # Refused means refused: no pool policy was applied on the way out.
        self.assertEqual(pool.limit_calls, [])

    def test_the_same_model_is_accepted_when_the_device_can_hold_it(self):
        cupy_module, pypeec_module, pool, _pinned, _cache = _fake_cuda_stack(
            free_mib=4096, total_mib=6144
        )
        executor = CudaPyPeecExecutor(
            {}, cupy_module=cupy_module, pypeec_module=pypeec_module
        )
        geometry = _geometry((162, 152, 45), conductive=29_000)
        result = executor.execute(geometry, {}, {"dense_options": {"split": True}})
        self.assertEqual(result.estimate.box, (162, 152, 45))
        self.assertTrue(pool.limit_calls)

    def test_preflight_can_be_turned_off(self):
        cupy_module, pypeec_module, _pool, _pinned, _cache = _fake_cuda_stack(
            free_mib=64, total_mib=128
        )
        executor = CudaPyPeecExecutor(
            {"preflight_memory": False},
            cupy_module=cupy_module,
            pypeec_module=pypeec_module,
        )
        geometry = _geometry((162, 152, 45), conductive=29_000)
        result = executor.execute(geometry, {}, {"dense_options": {"split": True}})
        self.assertIsNotNone(result.estimate)

    def test_a_geometry_without_a_box_is_solved_rather_than_refused(self):
        cupy_module, pypeec_module, _pool, _pinned, _cache = _fake_cuda_stack()
        pypeec_module.run_mesher_data = lambda geometry: {"voxelized": True}
        executor = CudaPyPeecExecutor(
            {}, cupy_module=cupy_module, pypeec_module=pypeec_module
        )
        result = executor.execute({"name": "board"}, {}, {})
        self.assertIsNone(result.estimate)

    def test_the_plan_cache_is_bounded_for_the_solve_and_restored_after(self):
        cache = FakePlanCache(size=16, memsize=-1, curr_bytes=7 * 1024**2)
        seen = {}

        def solver(_voxel, _problem, _tolerance):
            seen["size"] = cache.get_size()
            return {
                "status": True,
                "data_sweep": {
                    "target": {
                        "solution_ok": True,
                        "solver_status": {"n_iter": 1, "residuum_val": 0.0},
                    }
                },
            }

        cupy_module, pypeec_module, _pool, _pinned, _cache = _fake_cuda_stack(
            plan_cache=cache, solver=solver
        )
        executor = CudaPyPeecExecutor(
            {"fft_plan_cache_entries": 2, "fft_plan_cache_bytes": 128 * 1024**2},
            cupy_module=cupy_module,
            pypeec_module=pypeec_module,
        )
        result = executor.execute(
            _geometry((8, 6, 4), conductive=40), {}, {"dense_options": {}}
        )
        self.assertEqual(seen["size"], 2)
        self.assertEqual(cache.get_size(), 16)
        self.assertEqual(cache.get_memsize(), -1)
        self.assertEqual(result.fft_plan_cache_bytes, 7 * 1024**2)

    def test_a_cufft_allocation_failure_is_reported_as_running_out_of_memory(self):
        class CUFFTError(RuntimeError):
            pass

        def solver(*_args):
            raise CUFFTError("CUFFT_ALLOC_FAILED")

        cupy_module, pypeec_module, _pool, _pinned, _cache = _fake_cuda_stack(
            solver=solver
        )
        executor = CudaPyPeecExecutor(
            {}, cupy_module=cupy_module, pypeec_module=pypeec_module
        )
        with self.assertRaises(CudaPeecSolveError) as caught:
            executor.execute(
                _geometry((8, 6, 4), conductive=40), {}, {"dense_options": {}}
            )
        message = str(caught.exception)
        self.assertIn("ran out of memory", message)
        self.assertIn("plan cache held", message)
        self.assertIn("8x6x4", message)

    def test_an_unrelated_failure_is_not_reported_as_memory(self):
        def solver(*_args):
            raise RuntimeError("singular matrix")

        cupy_module, pypeec_module, _pool, _pinned, _cache = _fake_cuda_stack(
            solver=solver
        )
        executor = CudaPyPeecExecutor(
            {}, cupy_module=cupy_module, pypeec_module=pypeec_module
        )
        with self.assertRaises(CudaPeecSolveError) as caught:
            executor.execute(
                _geometry((8, 6, 4), conductive=40), {}, {"dense_options": {}}
            )
        self.assertNotIn("ran out of memory", str(caught.exception))

    def test_the_voxel_cache_honours_a_byte_budget(self):
        cupy_module, pypeec_module, _pool, _pinned, _cache = _fake_cuda_stack()
        pypeec_module.run_mesher_data = lambda geometry: {
            "payload": np.zeros(4096, dtype=np.float64)
        }
        executor = CudaPyPeecExecutor(
            {"voxel_cache_max_bytes": 40_000},
            cupy_module=cupy_module,
            pypeec_module=pypeec_module,
        )
        for index in range(6):
            executor.execute(
                _geometry((8, 6, 4 + index), conductive=40), {}, {"dense_options": {}}
            )
        self.assertLessEqual(voxel_cache_bytes(), 40_000)
        self.assertGreater(voxel_cache_bytes(), 0)

    def test_closing_releases_the_pool_and_the_plans(self):
        cache = FakePlanCache()
        cupy_module, pypeec_module, pool, pinned, _cache = _fake_cuda_stack(
            plan_cache=cache
        )
        with CudaPyPeecExecutor(
            {"release_pool_after_solve": False},
            cupy_module=cupy_module,
            pypeec_module=pypeec_module,
        ) as executor:
            executor.execute(
                _geometry((8, 6, 4), conductive=40), {}, {"dense_options": {}}
            )
            self.assertEqual(pool.free_calls, 0)
        self.assertEqual(pool.free_calls, 1)
        self.assertEqual(pinned.free_calls, 1)
        self.assertEqual(cache.cleared, 1)


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
        from electrical.dice_peec.delta_peec import DeltaQuadraticScorer, FFTInteraction2D

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
