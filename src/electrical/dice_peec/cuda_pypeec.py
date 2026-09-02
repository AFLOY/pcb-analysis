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
from .pypeec_memory import (
    DEFAULT_UNMEASURED_FRACTION,
    MemoryEstimate,
    describe_estimate,
    estimate_pypeec_memory,
)


class CudaUnavailableError(RuntimeError):
    """Raised when the configured CUDA runtime cannot be used."""


class CudaPeecMemoryError(CudaUnavailableError):
    """Raised when a solve is predicted, or found, not to fit on the device.

    A subclass of :class:`CudaUnavailableError` so that a caller configured to
    fall back to the CPU keeps falling back: a model too large for this device
    is exactly the case that fallback exists for.
    """

    def __init__(self, message: str, estimate: "MemoryEstimate | None" = None) -> None:
        super().__init__(message)
        self.estimate = estimate


class CudaPeecSolveError(RuntimeError):
    """Raised when a CUDA PEEC execution fails after runtime initialization."""


@dataclass(frozen=True)
class CudaPeecConfig:
    device_id: int = 0
    precision: str = "auto"
    memory_reserve_fraction: float = 0.10
    cache_voxel: bool = True
    voxel_cache_entries: int = 8
    # Host bytes the meshed-voxel cache may hold.  Entry counts stopped being a
    # useful budget once a model could span a board's height instead of one
    # copper layer: eight entries of a tall mesh is a different quantity of
    # memory from eight entries of a flat one.
    voxel_cache_max_bytes: int = 512 * 1024 * 1024
    # Return unused CuPy blocks to the driver after each solve.  Keeps free
    # VRAM higher across multi-case epochs at a small allocator cost.
    release_pool_after_solve: bool = True
    # Refuse a solve the estimate says cannot fit, instead of discovering it
    # part-way through an allocation.  Turn off to let the device decide.
    preflight_memory: bool = True
    unmeasured_memory_fraction: float = DEFAULT_UNMEASURED_FRACTION
    # CuPy's FFT plan cache holds cuFFT workspaces outside the pool this
    # executor caps.  A tall box makes those plans large, so the cache is
    # bounded here rather than left at CuPy's default.  None leaves CuPy's
    # policy alone; a negative byte budget means unlimited, as CuPy defines it.
    fft_plan_cache_entries: int | None = 4
    fft_plan_cache_bytes: int | None = None

    @classmethod
    def from_mapping(cls, settings: dict[str, Any] | None) -> "CudaPeecConfig":
        values = settings or {}
        precision = str(values.get("precision", "auto"))
        if precision not in {"auto", "complex128"}:
            raise ValueError(
                "current_field_solver.cuda_peec.precision must be auto or "
                "complex128. complex64 was accepted while it did nothing: "
                "PyPEEC 5.8 allocates its FFT tensors and solution vectors as "
                "complex128 and offers no way to ask for anything narrower, so "
                "requesting it bought no memory and named a policy that was "
                "never applied"
            )
        reserve = float(values.get("memory_reserve_fraction", 0.10))
        if not 0.0 <= reserve < 1.0:
            raise ValueError("memory_reserve_fraction must be in [0, 1)")
        cache_entries = int(values.get("voxel_cache_entries", 8))
        if cache_entries < 1:
            raise ValueError("voxel_cache_entries must be positive")
        cache_bytes = int(values.get("voxel_cache_max_bytes", 512 * 1024 * 1024))
        if cache_bytes < 0:
            raise ValueError("voxel_cache_max_bytes must be non-negative")
        device_id = int(values.get("device_id", 0))
        if device_id < 0:
            raise ValueError("device_id must be non-negative")
        unmeasured = float(
            values.get("unmeasured_memory_fraction", DEFAULT_UNMEASURED_FRACTION)
        )
        if not 0.0 <= unmeasured < 4.0:
            raise ValueError("unmeasured_memory_fraction must be in [0, 4)")
        plan_entries = values.get("fft_plan_cache_entries", 4)
        if plan_entries is not None:
            plan_entries = int(plan_entries)
            if plan_entries < 0:
                raise ValueError("fft_plan_cache_entries must be non-negative")
        plan_bytes = values.get("fft_plan_cache_bytes")
        if plan_bytes is not None:
            plan_bytes = int(plan_bytes)
        return cls(
            device_id=device_id,
            precision=precision,
            memory_reserve_fraction=reserve,
            cache_voxel=bool(values.get("cache_voxel", True)),
            voxel_cache_entries=cache_entries,
            voxel_cache_max_bytes=cache_bytes,
            release_pool_after_solve=bool(
                values.get("release_pool_after_solve", True)
            ),
            preflight_memory=bool(values.get("preflight_memory", True)),
            unmeasured_memory_fraction=unmeasured,
            fft_plan_cache_entries=plan_entries,
            fft_plan_cache_bytes=plan_bytes,
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
    estimate: MemoryEstimate | None = None
    fft_plan_cache_bytes: int = 0

    def metrics(self) -> dict[str, Any]:
        metrics = {
            "cuda_device_name": self.device_name,
            "cuda_device_id": int(self.telemetry.backend.rsplit(":", 1)[-1]),
            "cuda_runtime_version": self.cuda_runtime_version,
            "cupy_version": self.cupy_version,
            "requested_precision": self.requested_precision,
            "effective_precision": self.effective_precision,
            # PyPEEC owns the solver dtype, so a request for anything other
            # than complex128 cannot be honoured.  Say so rather than let the
            # requested value read as the applied one.
            "precision_request_honored": (
                self.effective_precision == self.requested_precision
                or self.requested_precision == "auto"
            ),
            "voxel_cache_hit": self.voxel_cache_hit,
            "peak_device_bytes": self.execution_report.peak_bytes,
            "memory_measurement_complete": self.execution_report.memory_complete,
            "fft_plan_cache_bytes": self.fft_plan_cache_bytes,
            "mesher_elapsed_ms": self.mesher_elapsed_ms,
            "solver_elapsed_ms": self.solver_elapsed_ms,
            "total_elapsed_ms": self.total_elapsed_ms,
        }
        if self.estimate is not None:
            metrics.update(self.estimate.as_metrics())
            peak = self.execution_report.peak_bytes
            if peak > 0:
                metrics["estimate_over_measured_ratio"] = (
                    self.estimate.required_bytes / peak
                )
        return metrics


@dataclass
class _CachedVoxel:
    voxel: dict[str, Any]
    host_bytes: int


_VOXEL_CACHE: OrderedDict[str, _CachedVoxel] = OrderedDict()
_VOXEL_CACHE_LOCK = threading.RLock()


def _host_bytes(value: Any, _depth: int = 0) -> int:
    """Approximate what a meshed voxel tree costs in host memory.

    Only the array payloads are counted; the dict scaffolding around them is
    noise beside a mesh over a board-height box.  The recursion is bounded
    because the tree PyPEEC returns is not deep and a runaway walk would cost
    more than the accounting saves.
    """
    if _depth > 8:
        return 0
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    if isinstance(value, dict):
        return sum(_host_bytes(item, _depth + 1) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_host_bytes(item, _depth + 1) for item in value)
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    return 0


def _pool_limit_for_solve(
    *,
    total_bytes: int,
    free_bytes: int,
    pool_total_bytes: int,
    reserve_bytes: int,
    existing_limit: int,
) -> int:
    """Return an absolute CuPy-pool cap while preserving free VRAM reserve.

    ``memGetInfo().free`` excludes blocks already acquired by CuPy's pool, so
    using it directly as the pool's absolute limit can place an active pool
    below its current size.  Add those blocks back, subtract allocations owned
    outside the pool and the requested reserve, then honor a stricter limit set
    by the caller.
    """

    values = (
        total_bytes,
        free_bytes,
        pool_total_bytes,
        reserve_bytes,
        existing_limit,
    )
    if any(int(value) < 0 for value in values):
        raise ValueError("CUDA memory counters and limits must be non-negative")
    total = int(total_bytes)
    free = min(int(free_bytes), total)
    pool_total = min(int(pool_total_bytes), total)
    reserve = min(int(reserve_bytes), total)
    non_pool = max(0, total - free - pool_total)
    policy_limit = max(0, total - reserve - non_pool)
    if existing_limit:
        policy_limit = min(policy_limit, int(existing_limit))
    return int(policy_limit)


def _geometry_key(geometry: dict[str, Any]) -> str:
    """Key a meshed voxel tree by the geometry that produced it.

    The device is deliberately not part of the key.  The mesh is held on the
    host and PyPEEC builds it without touching a GPU, so keying by device would
    mesh the same board once per device for no gain.
    """
    payload = json.dumps(
        geometry, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _describe(estimate: MemoryEstimate | None) -> str:
    """Describe an estimate for a message, or say that none could be made."""
    return "no memory estimate" if estimate is None else describe_estimate(estimate)


def _trim_voxel_cache(max_entries: int, max_bytes: int) -> None:
    """Evict oldest entries until both budgets hold.

    The byte budget is the one that matters for a tall model, but it is checked
    second so that a single entry larger than the whole budget is still kept:
    evicting it would mesh the same board again on the next case and gain
    nothing.
    """
    while len(_VOXEL_CACHE) > max_entries:
        _VOXEL_CACHE.popitem(last=False)
    while len(_VOXEL_CACHE) > 1:
        total = sum(entry.host_bytes for entry in _VOXEL_CACHE.values())
        if total <= max_bytes:
            break
        _VOXEL_CACHE.popitem(last=False)


def voxel_cache_bytes() -> int:
    """Report what the meshed-voxel cache is holding on the host."""
    with _VOXEL_CACHE_LOCK:
        return sum(entry.host_bytes for entry in _VOXEL_CACHE.values())


def clear_cuda_caches() -> None:
    """Clear process-local geometry caches, primarily for tests and audits."""
    with _VOXEL_CACHE_LOCK:
        _VOXEL_CACHE.clear()


def cupy_tolerance(
    tolerance: dict[str, Any], *, precision: str = "auto"
) -> dict[str, Any]:
    """Return an isolated PyPEEC tolerance tree selecting CuPy FFTs.

    PyPEEC 5.8 controls the FFT implementation through this nested value.  It
    also owns the solver dtype: ``lib_matrix/multiply_fft.py`` allocates its
    prepared tensors and its products as ``complex128`` unconditionally.  There
    is therefore no narrower precision to select here, and the argument exists
    only to reject a request this adapter cannot carry out.
    """

    if precision not in {"auto", "complex128"}:
        raise ValueError(
            f"unsupported CUDA PEEC precision: {precision}. PyPEEC 5.8 solves "
            "in complex128 and exposes no way to ask for less"
        )
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
        self.cp = cupy_module if cupy_module is not None else self._import_cupy()
        self.pypeec = (
            pypeec_module if pypeec_module is not None else self._import_pypeec()
        )
        self._multiply_fft = getattr(self.pypeec, "multiply_fft", None)

    @staticmethod
    def _import_cupy() -> Any:
        try:
            import cupy as cp
        except (ImportError, OSError) as exc:
            raise CudaUnavailableError(
                "CuPy is unavailable; install pcb-analysis[cuda] with the "
                "wheel selected by the pcb-analysis cuda extra"
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

    def __enter__(self) -> "CudaPyPeecExecutor":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release what this executor holds on the device.

        An optimizer epoch solves many cases on one geometry, and keeping the
        pool and the FFT plans across them is what makes a tall box affordable
        the second time.  Constructing an executor per case, as callers do
        today, throws that away; a caller that holds one open for an epoch can
        close it here instead of at every solve.
        """
        try:
            with self._device():
                self.cp.get_default_memory_pool().free_all_blocks()
                self.cp.get_default_pinned_memory_pool().free_all_blocks()
                cache = self._plan_cache()
                if cache is not None:
                    cache.clear()
        except Exception:  # noqa: BLE001 - closing must not raise
            pass

    def _device(self) -> Any:
        return self.cp.cuda.Device(self.config.device_id)

    def _plan_cache(self) -> Any | None:
        """Reach CuPy's per-thread FFT plan cache, if this CuPy exposes one."""
        try:
            return self.cp.fft.config.get_plan_cache()
        except (AttributeError, RuntimeError):
            return None

    def _plan_cache_bytes(self) -> int:
        cache = self._plan_cache()
        if cache is None:
            return 0
        try:
            return int(cache.get_curr_size_bytes())
        except (AttributeError, RuntimeError, TypeError):
            return 0

    def _is_memory_error(self, exc: BaseException) -> bool:
        """Say whether a failure is the device running out of room.

        CuPy's own out-of-memory type is the obvious case.  cuFFT is the one
        that matters on a tall box: its plan workspaces are allocated outside
        the pool, so exhausting them raises a cuFFT allocation failure rather
        than CuPy's error, and treating that as a generic fault loses both the
        fallback and the telemetry.
        """
        oom_type = getattr(self.cp.cuda.memory, "OutOfMemoryError", ())
        if oom_type and isinstance(exc, oom_type):
            return True
        name = type(exc).__name__
        if "OutOfMemory" in name or "CUFFTError" in name or "CuFFTError" in name:
            text = str(exc).upper()
            return (
                "CUFFTERROR" in name.upper()
                or "ALLOC" in text
                or "MEMORY" in text
                or "OUT OF" in text
            )
        return False

    def estimate(
        self,
        geometry: dict[str, Any],
        problem: dict[str, Any] | None = None,
        tolerance: dict[str, Any] | None = None,
    ) -> MemoryEstimate | None:
        """Predict this solve's device memory without touching the device.

        Returns ``None`` when the geometry states no voxel box.  Being unable
        to predict is not a reason to refuse: the prediction exists to stop a
        solve that will not fit, and a geometry this cannot read is one PyPEEC
        will reject on its own terms, with its own message.
        """
        try:
            return estimate_pypeec_memory(
                geometry,
                problem,
                tolerance,
                unmeasured_fraction=self.config.unmeasured_memory_fraction,
            )
        except (KeyError, TypeError, ValueError):
            return None

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

        estimate = self.estimate(geometry, problem, tolerance)
        if self.config.preflight_memory and estimate is not None:
            budget = telemetry.free_bytes - reserve
            if estimate.required_bytes > budget:
                raise CudaPeecMemoryError(
                    "CUDA PEEC solve is predicted not to fit: "
                    f"{describe_estimate(estimate)}; the device offers "
                    f"{budget / (1024 ** 2):.1f} MiB after a "
                    f"{self.config.memory_reserve_fraction:.0%} reserve. "
                    f"Assumptions: {'; '.join(estimate.assumptions)}",
                    estimate,
                )

        configured_tolerance = cupy_tolerance(
            tolerance, precision=self.config.precision
        )
        # PyPEEC 5.8 caches its FFT implementation in module globals after
        # the first solve. Reset the flag so a prior SciPy reference solve
        # cannot silently make this CUDA request execute on the CPU.
        try:
            multiply_fft = self._multiply_fft
            if multiply_fft is None:
                from pypeec.lib_matrix import multiply_fft

            multiply_fft.SET = False
        except (ImportError, AttributeError) as exc:
            raise CudaPeecSolveError(
                "PyPEEC 5.8 FFT backend reset hook is unavailable"
            ) from exc
        pool = cp.get_default_memory_pool()
        pinned_pool = cp.get_default_pinned_memory_pool()
        total_start = time.perf_counter()
        with self._device():
            # CuPy memory-pool limits are per device.  Read, apply, and restore
            # the limit inside the selected context so a nonzero device_id
            # cannot leak policy onto another GPU.
            old_limit = int(pool.get_limit())
            plan_cache = self._plan_cache()
            old_plan_size: int | None = None
            old_plan_bytes: int | None = None
            try:
                # A tall box makes each cuFFT plan large, and those workspaces
                # live outside the pool capped below.  Bound the cache so the
                # reserve means something; restore the caller's policy after.
                if plan_cache is not None:
                    if self.config.fft_plan_cache_entries is not None:
                        old_plan_size = int(plan_cache.get_size())
                        plan_cache.set_size(self.config.fft_plan_cache_entries)
                    if self.config.fft_plan_cache_bytes is not None:
                        old_plan_bytes = int(plan_cache.get_memsize())
                        plan_cache.set_memsize(self.config.fft_plan_cache_bytes)
                pool_limit = _pool_limit_for_solve(
                    total_bytes=telemetry.total_bytes,
                    free_bytes=telemetry.free_bytes,
                    pool_total_bytes=int(pool.total_bytes()),
                    reserve_bytes=reserve,
                    existing_limit=old_limit,
                )
                if pool_limit <= int(pool.used_bytes()):
                    raise CudaPeecMemoryError(
                        "CUDA memory reserve leaves no pool headroom for the "
                        f"solve; {_describe(estimate)}",
                        estimate,
                    )
                pool.set_limit(size=pool_limit)
                cache_key = _geometry_key(geometry)
                voxel = None
                cache_hit = False
                if self.config.cache_voxel:
                    with _VOXEL_CACHE_LOCK:
                        cached = _VOXEL_CACHE.get(cache_key)
                        cache_hit = cached is not None
                        if cache_hit:
                            # Defensive copy: PyPEEC may mutate geometry views.
                            voxel = copy.deepcopy(cached.voxel)
                            _VOXEL_CACHE.move_to_end(cache_key)
                if cache_hit:
                    mesher_ms = 0.0
                else:
                    mesher_start = time.perf_counter()
                    # Keep mesher output on the host.  PyPEEC 5.8 indexes
                    # domain_def with NumPy and only moves FFT products to
                    # CuPy; pushing the whole voxel tree to the device breaks
                    # material indexing and wastes VRAM.
                    voxel = self.pypeec.run_mesher_data(geometry)
                    mesher_ms = (time.perf_counter() - mesher_start) * 1000.0
                    if self.config.cache_voxel:
                        stored = copy.deepcopy(voxel)
                        entry = _CachedVoxel(stored, _host_bytes(stored))
                        with _VOXEL_CACHE_LOCK:
                            _VOXEL_CACHE[cache_key] = entry
                            _VOXEL_CACHE.move_to_end(cache_key)
                            _trim_voxel_cache(
                                self.config.voxel_cache_entries,
                                self.config.voxel_cache_max_bytes,
                            )

                cp.cuda.Stream.null.synchronize()
                free_before_solve, _ = cp.cuda.runtime.memGetInfo()
                pool_before_solve = int(pool.total_bytes())
                solver_start = time.perf_counter()
                solution = self.pypeec.run_solver_data(
                    voxel, problem, configured_tolerance
                )
                cp.cuda.Stream.null.synchronize()
                solver_ms = (time.perf_counter() - solver_start) * 1000.0
                free_after, _ = cp.cuda.runtime.memGetInfo()
                pool_after = int(pool.total_bytes())
                device_growth = max(0, int(free_before_solve) - int(free_after))
                pool_growth = max(0, pool_after - pool_before_solve)
                # Prefer the larger of pool growth and free-memory drop for the
                # solve window; still incomplete vs cuFFT workspaces outside pool.
                peak_bytes = max(
                    pool_growth,
                    device_growth,
                    int(pool.used_bytes()),
                    1,
                )
            except CudaUnavailableError:
                raise
            except Exception as exc:
                if self._is_memory_error(exc):
                    report = ExecutionReport(
                        peak_bytes=0,
                        elapsed_ms=(time.perf_counter() - total_start) * 1000.0,
                        converged=False,
                        oom=True,
                        memory_complete=False,
                    )
                    raise CudaPeecSolveError(
                        f"CUDA PEEC ran out of memory: {report}; "
                        f"{_describe(estimate)}; plan cache held "
                        f"{self._plan_cache_bytes() / (1024 ** 2):.1f} MiB; "
                        f"free VRAM at entry "
                        f"{telemetry.free_bytes / (1024 ** 2):.1f} MiB"
                    ) from exc
                raise CudaPeecSolveError(
                    f"CUDA PEEC execution failed: {exc}"
                ) from exc
            finally:
                if plan_cache is not None:
                    if old_plan_size is not None:
                        plan_cache.set_size(old_plan_size)
                    if old_plan_bytes is not None:
                        plan_cache.set_memsize(old_plan_bytes)
                # A zero limit means unlimited; restore the caller's policy.
                pool.set_limit(size=old_limit)
                if self.config.release_pool_after_solve:
                    # Drop orphan blocks held by the pool so multi-case
                    # optimizer epochs do not accumulate free-but-reserved VRAM.
                    pool.free_all_blocks()
                    pinned_pool.free_all_blocks()

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
            estimate=estimate,
            fft_plan_cache_bytes=self._plan_cache_bytes(),
        )
