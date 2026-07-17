"""Measured PyPEEC execution using its CuPy/cuFFT backend.

The module deliberately owns only CUDA runtime policy and telemetry.  PyPEEC
continues to own voxel PEEC assembly and the coupled iterative solve, which
keeps the CUDA result directly comparable with the trusted CPU path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .controller import ExecutionReport, HardwareTelemetry


class CudaUnavailableError(RuntimeError):
    """Raised when the configured CUDA runtime cannot be used."""


class CudaPeecSolveError(RuntimeError):
    """Raised when a CUDA PEEC execution fails after runtime initialization."""


@dataclass(frozen=True)
class CudaPeecConfig:
    device_id: int = 0
    precision: str = "auto"
    memory_reserve_fraction: float = 0.10
    cache_voxel: bool = True
    voxel_cache_entries: int = 8

    @classmethod
    def from_mapping(cls, settings: dict[str, Any] | None) -> "CudaPeecConfig":
        values = settings or {}
        precision = str(values.get("precision", "auto"))
        if precision not in {"auto", "complex64", "complex128"}:
            raise ValueError(
                "current_field_solver.cuda_peec.precision must be auto, "
                "complex64, or complex128"
            )
        reserve = float(values.get("memory_reserve_fraction", 0.10))
        if not 0.0 <= reserve < 1.0:
            raise ValueError("memory_reserve_fraction must be in [0, 1)")
        cache_entries = int(values.get("voxel_cache_entries", 8))
        if cache_entries < 1:
            raise ValueError("voxel_cache_entries must be positive")
        return cls(
            device_id=int(values.get("device_id", 0)),
            precision=precision,
            memory_reserve_fraction=reserve,
            cache_voxel=bool(values.get("cache_voxel", True)),
            voxel_cache_entries=cache_entries,
        )


@dataclass(frozen=True)
class CudaPeecResult:
    voxel: dict[str, Any]
    solution: dict[str, Any]
    execution_report: ExecutionReport
    telemetry: HardwareTelemetry
    device_name: str
    cuda_runtime_version: int
    cupy_version: str
    mesher_elapsed_ms: float
    solver_elapsed_ms: float
    total_elapsed_ms: float
    requested_precision: str
    effective_precision: str
    voxel_cache_hit: bool

    def metrics(self) -> dict[str, Any]:
        return {
            "cuda_device_name": self.device_name,
            "cuda_device_id": int(self.telemetry.backend.rsplit(":", 1)[-1]),
            "cuda_runtime_version": self.cuda_runtime_version,
            "cupy_version": self.cupy_version,
            "requested_precision": self.requested_precision,
            "effective_precision": self.effective_precision,
            "voxel_cache_hit": self.voxel_cache_hit,
            "peak_device_bytes": self.execution_report.peak_bytes,
            "memory_measurement_complete": self.execution_report.memory_complete,
            "mesher_elapsed_ms": self.mesher_elapsed_ms,
            "solver_elapsed_ms": self.solver_elapsed_ms,
            "total_elapsed_ms": self.total_elapsed_ms,
        }


_VOXEL_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_VOXEL_CACHE_LOCK = threading.RLock()


def _geometry_key(geometry: dict[str, Any], device_id: int = 0) -> str:
    payload = json.dumps(
        geometry, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest() + f":{device_id}"

def _to_device(obj: Any, cp: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_device(v, cp) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, cp) for v in obj]
    if hasattr(obj, "shape") and hasattr(obj, "dtype") and not isinstance(obj, cp.ndarray):
        return cp.asarray(obj)
    return obj


def clear_cuda_caches() -> None:
    """Clear process-local geometry caches, primarily for tests and audits."""
    with _VOXEL_CACHE_LOCK:
        _VOXEL_CACHE.clear()


def cupy_tolerance(
    tolerance: dict[str, Any], *, precision: str = "auto"
) -> dict[str, Any]:
    """Return an isolated PyPEEC tolerance tree selecting CuPy FFTs.

    PyPEEC 5.8 controls the FFT implementation through this nested value.  It
    currently owns the solver dtype, so the requested mixed-precision policy
    is recorded but complex128 remains the effective physical-solve dtype.
    """

    if precision not in {"auto", "complex64", "complex128"}:
        raise ValueError(f"unsupported CUDA PEEC precision: {precision}")
    configured = copy.deepcopy(tolerance)
    dense = configured.setdefault("dense_options", {})
    dense["method"] = "fft"
    fft = dense.setdefault("fft_options", {})
    fft["library"] = "CuPy"
    # CPU worker settings do not apply to CuPy and can otherwise make runtime
    # metadata misleading.
    fft["scipy_worker"] = 0
    return configured


class CudaPyPeecExecutor:
    """Run a physical PyPEEC solve with CUDA and collect device telemetry."""

    def __init__(
        self,
        settings: dict[str, Any] | None = None,
        *,
        cupy_module: Any | None = None,
        pypeec_module: Any | None = None,
    ) -> None:
        self.config = CudaPeecConfig.from_mapping(settings)
        self.cp = cupy_module or self._import_cupy()
        self.pypeec = pypeec_module or self._import_pypeec()

    @staticmethod
    def _import_cupy() -> Any:
        try:
            import cupy as cp
        except (ImportError, OSError) as exc:
            raise CudaUnavailableError(
                "CuPy is unavailable; install peec-fastopt[cuda] with the "
                "wheel selected by the peec-fastopt cuda extra"
            ) from exc
        return cp

    @staticmethod
    def _import_pypeec() -> Any:
        try:
            import pypeec
        except (ImportError, OSError) as exc:
            raise CudaUnavailableError(
                "PyPEEC 5.8 is required for the CUDA PEEC backend"
            ) from exc
        return pypeec

    def _device(self) -> Any:
        return self.cp.cuda.Device(self.config.device_id)

    def probe(self) -> HardwareTelemetry:
        try:
            with self._device():
                free_bytes, total_bytes = self.cp.cuda.runtime.memGetInfo()
        except Exception as exc:  # CuPy exposes runtime-specific subclasses.
            raise CudaUnavailableError(
                f"CUDA device {self.config.device_id} is unavailable: {exc}"
            ) from exc
        platform = "windows" if sys.platform.startswith("win") else "linux"
        return HardwareTelemetry(
            total_bytes=int(total_bytes),
            free_bytes=int(free_bytes),
            platform=platform,
            backend=f"cupy-pypeec:{self.config.device_id}",
        )

    def execute(
        self,
        geometry: dict[str, Any],
        problem: dict[str, Any],
        tolerance: dict[str, Any],
    ) -> CudaPeecResult:
        cp = self.cp
        telemetry = self.probe()
        reserve = int(telemetry.total_bytes * self.config.memory_reserve_fraction)
        usable = min(telemetry.free_bytes, telemetry.total_bytes - reserve)
        if usable <= 0:
            raise CudaUnavailableError(
                "CUDA memory reserve leaves no usable device allocation"
            )

        configured_tolerance = cupy_tolerance(
            tolerance, precision=self.config.precision
        )
        # PyPEEC 5.8 caches its FFT implementation in module globals after
        # the first solve. Reset the flag so a prior SciPy reference solve
        # cannot silently make this CUDA request execute on the CPU.
        try:
            from pypeec.lib_matrix import multiply_fft

            multiply_fft.SET = False
        except (ImportError, AttributeError) as exc:
            raise CudaPeecSolveError(
                "PyPEEC 5.8 FFT backend reset hook is unavailable"
            ) from exc
        pool = cp.get_default_memory_pool()
        baseline_free = int(telemetry.free_bytes)
        total_start = time.perf_counter()
        with self._device():
            # CuPy memory-pool limits are per device.  Read, apply, and restore
            # the limit inside the selected context so a nonzero device_id
            # cannot leak policy onto another GPU.
            old_limit = int(pool.get_limit())
            try:
                pool.set_limit(size=int(usable))
                cache_key = _geometry_key(geometry, self.config.device_id)
                voxel = None
                cache_hit = False
                if self.config.cache_voxel:
                    with _VOXEL_CACHE_LOCK:
                        voxel = _VOXEL_CACHE.get(cache_key)
                        cache_hit = voxel is not None
                        if cache_hit:
                            _VOXEL_CACHE.move_to_end(cache_key)
                if cache_hit:
                    mesher_ms = 0.0
                else:
                    mesher_start = time.perf_counter()
                    voxel = self.pypeec.run_mesher_data(geometry)
                    voxel = _to_device(voxel, cp)
                    mesher_ms = (time.perf_counter() - mesher_start) * 1000.0
                    if self.config.cache_voxel:
                        with _VOXEL_CACHE_LOCK:
                            _VOXEL_CACHE[cache_key] = voxel
                            _VOXEL_CACHE.move_to_end(cache_key)
                            while len(_VOXEL_CACHE) > self.config.voxel_cache_entries:
                                _VOXEL_CACHE.popitem(last=False)

                cp.cuda.Stream.null.synchronize()
                solver_start = time.perf_counter()
                solution = self.pypeec.run_solver_data(
                    voxel, problem, configured_tolerance
                )
                cp.cuda.Stream.null.synchronize()
                solver_ms = (time.perf_counter() - solver_start) * 1000.0
                free_after, _ = cp.cuda.runtime.memGetInfo()
                device_growth = max(0, baseline_free - int(free_after))
                peak_bytes = max(int(pool.total_bytes()), device_growth, 1)
            except Exception as exc:
                oom_type = getattr(cp.cuda.memory, "OutOfMemoryError", ())
                if oom_type and isinstance(exc, oom_type):
                    report = ExecutionReport(
                        peak_bytes=0,
                        elapsed_ms=(time.perf_counter() - total_start) * 1000.0,
                        converged=False,
                        oom=True,
                        memory_complete=False,
                    )
                    raise CudaPeecSolveError(
                        f"CUDA PEEC ran out of memory: {report}"
                    ) from exc
                raise CudaPeecSolveError(
                    f"CUDA PEEC execution failed: {exc}"
                ) from exc
            finally:
                # A zero limit means unlimited; restore the caller's policy.
                pool.set_limit(size=old_limit)

        total_ms = (time.perf_counter() - total_start) * 1000.0
        sweep = solution.get("data_sweep", {}).get("target", {})
        solver_status = sweep.get("solver_status", {})
        converged = bool(solution.get("status")) and bool(sweep.get("solution_ok"))
        report = ExecutionReport(
            peak_bytes=peak_bytes,
            elapsed_ms=total_ms,
            iterations=int(solver_status.get("n_iter", 0)),
            relative_residual=float(solver_status.get("residuum_val", 0.0)),
            converged=converged,
            oom=False,
            stagnated=not converged,
            precision_gap=0.0,
            # cuFFT can allocate workspaces outside CuPy's pool.  Keep this
            # false until CUPTI/NVML peak sampling is added.
            memory_complete=False,
        )
        props = cp.cuda.runtime.getDeviceProperties(self.config.device_id)
        raw_name = props.get("name", b"unknown")
        name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        return CudaPeecResult(
            voxel=voxel,
            solution=solution,
            execution_report=report,
            telemetry=telemetry,
            device_name=name,
            cuda_runtime_version=int(cp.cuda.runtime.runtimeGetVersion()),
            cupy_version=str(cp.__version__),
            mesher_elapsed_ms=mesher_ms,
            solver_elapsed_ms=solver_ms,
            total_elapsed_ms=total_ms,
            requested_precision=self.config.precision,
            effective_precision="complex128",
            voxel_cache_hit=cache_hit,
        )
