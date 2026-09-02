"""Optional CUDA calibration backend using CuPy.

This is not the final PEEC executor. It probes real VRAM behavior and measures
representative 2D FFT allocations/timings so the backend-independent controller
can replace analytic assumptions with device measurements.
"""

from __future__ import annotations

import sys

from .controller import ExecutionPlan, ExecutionReport, HardwareTelemetry, ProblemProfile


class CuPyCalibrationBackend:
    def __init__(self, platform: str | None = None, repeats: int = 5):
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError(
                "CuPy is required on the CUDA machine; install the wheel matching CUDA"
            ) from exc
        self.cp = cp
        self.repeats = int(repeats)
        self.platform = platform or ("windows" if sys.platform.startswith("win") else "linux")

    def probe(self) -> HardwareTelemetry:
        free_bytes, total_bytes = self.cp.cuda.runtime.memGetInfo()
        return HardwareTelemetry(
            total_bytes=int(total_bytes),
            free_bytes=int(free_bytes),
            platform=self.platform,
            backend="cupy-cuda",
        )

    def execute(self, plan: ExecutionPlan, problem: ProblemProfile) -> ExecutionReport:
        cp = self.cp
        telemetry = self.probe()
        if plan.estimated_bytes > plan.memory_budget_bytes or plan.estimated_bytes > telemetry.free_bytes:
            return ExecutionReport(0, 0.0, converged=False, oom=True)

        pool = cp.get_default_memory_pool()
        baseline = int(pool.used_bytes())
        dtype = cp.float64 if plan.field_precision == "complex128" else cp.float32
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        arrays = []
        spectra = []
        try:
            arrays = [
                cp.zeros((plan.tile_size, plan.tile_size), dtype=dtype)
                for _ in range(max(1, plan.candidate_batch))
            ]
            start.record()
            for _ in range(self.repeats):
                spectra = [cp.fft.rfft2(array) for array in arrays]
                arrays = [cp.fft.irfft2(spec, s=array.shape) for spec, array in zip(spectra, arrays)]
            end.record()
            end.synchronize()
            elapsed_ms = float(cp.cuda.get_elapsed_time(start, end)) / self.repeats
            peak = int(pool.used_bytes()) - baseline
            return ExecutionReport(
                peak_bytes=max(peak, 1),
                elapsed_ms=elapsed_ms,
                converged=True,
                memory_complete=False,
            )
        except cp.cuda.memory.OutOfMemoryError:
            return ExecutionReport(0, 0.0, converged=False, oom=True)
        finally:
            del arrays
            del spectra
            pool.free_all_blocks()
