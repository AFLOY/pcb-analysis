"""Mixed-precision iterative refinement with a matrix-free FP32 inner PCG."""

from __future__ import annotations

import importlib

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np

from .runtime import LowPrecisionRuntime


class MatrixFreeMPIRSystem(Protocol):
    """Split-precision operator contract.

    The high path consumes and returns host FP64 arrays.  The low path operates
    on the device-native arrays owned by ``runtime``.  No assembled global
    matrix is part of this interface.

    ``diagonal_low`` feeds the default Jacobi preconditioner.  A system may
    additionally define ``precondition_low(vector) -> vector``, an SPD (for
    PCG) approximate inverse applied on the low-precision runtime; when present
    it replaces the Jacobi scaling in every inner solver.
    """

    size: int
    runtime: LowPrecisionRuntime
    high_dtype: Any
    inner_solver: Literal["pcg", "gmres"]

    def apply_high(self, vector: np.ndarray) -> np.ndarray: ...

    def apply_low(self, vector: Any) -> Any: ...

    def diagonal_low(self) -> Any: ...


def _low_preconditioner(system: MatrixFreeMPIRSystem) -> Any:
    """Return the low-precision preconditioner action of ``system``."""

    custom = getattr(system, "precondition_low", None)
    if custom is not None:
        return custom
    runtime = system.runtime
    diagonal = system.diagonal_low()
    return lambda vector: runtime.divide(vector, diagonal)


@dataclass(frozen=True)
class MPIRConfig:
    """Accuracy and work limits for the nested solve."""

    relative_tolerance: float = 1.0e-10
    absolute_tolerance: float = 0.0
    inner_relative_tolerance: float = 2.0e-3
    max_outer_iterations: int = 8
    max_inner_iterations: int = 200
    gmres_restart: int = 24

    def __post_init__(self) -> None:
        for name in (
            "relative_tolerance",
            "absolute_tolerance",
            "inner_relative_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.relative_tolerance == 0.0 and self.absolute_tolerance == 0.0:
            raise ValueError("at least one outer tolerance must be positive")
        if not 0.0 < self.inner_relative_tolerance < 1.0:
            raise ValueError("inner_relative_tolerance must be between zero and one")
        if self.max_outer_iterations < 1 or self.max_inner_iterations < 1:
            raise ValueError("iteration limits must be positive")
        if self.gmres_restart < 2:
            raise ValueError("gmres_restart must be at least two")


@dataclass(frozen=True)
class MPIRStep:
    outer_iteration: int
    high_relative_residual: float
    inner_iterations: int
    inner_relative_residual: float


@dataclass(frozen=True)
class MPIRResult:
    solution: np.ndarray
    converged: bool
    outer_iterations: int
    inner_iterations: int
    relative_residual: float
    high_operator_applications: int
    low_operator_applications: int
    low_runtime: str
    history: tuple[MPIRStep, ...]


def _inner_pcg(
    system: MatrixFreeMPIRSystem,
    rhs_high: np.ndarray,
    config: MPIRConfig,
) -> tuple[np.ndarray, int, float, int]:
    """Approximately solve one correction equation entirely in FP32."""

    runtime = system.runtime
    rhs = runtime.from_host(rhs_high)
    correction = runtime.zeros_like(rhs)
    residual = runtime.copy(rhs)
    rhs_norm = runtime.norm(rhs)
    if rhs_norm == 0.0:
        return np.zeros_like(rhs_high), 0, 0.0, 0

    precondition = _low_preconditioner(system)
    preconditioned = precondition(residual)
    direction = runtime.copy(preconditioned)
    rz = float(np.real(runtime.dot(residual, preconditioned)))
    applications = 0
    relative_residual = 1.0

    for iteration in range(1, config.max_inner_iterations + 1):
        applied = system.apply_low(direction)
        applications += 1
        curvature = float(np.real(runtime.dot(direction, applied)))
        if not np.isfinite(curvature) or curvature <= 0.0:
            raise RuntimeError(
                "inner PCG requires a finite symmetric positive-definite operator"
            )
        alpha = rz / curvature
        correction = runtime.axpy(alpha, direction, correction)
        residual = runtime.axpy(-alpha, applied, residual)
        relative_residual = runtime.norm(residual) / rhs_norm
        if relative_residual <= config.inner_relative_tolerance:
            return (
                runtime.to_host(correction),
                iteration,
                relative_residual,
                applications,
            )

        preconditioned = precondition(residual)
        next_rz = float(np.real(runtime.dot(residual, preconditioned)))
        if not np.isfinite(next_rz) or rz == 0.0:
            break
        beta = next_rz / rz
        direction = runtime.axpy(beta, direction, preconditioned)
        rz = next_rz

    return (
        runtime.to_host(correction),
        iteration,
        relative_residual,
        applications,
    )


def _inner_gmres(
    system: MatrixFreeMPIRSystem,
    rhs_high: np.ndarray,
    config: MPIRConfig,
) -> tuple[np.ndarray, int, float, int]:
    """Solve one correction with restarted complex64 right-Jacobi GMRES."""

    if getattr(system.runtime, "is_cuda", False):
        return _inner_gmres_cuda(system, rhs_high, config)

    # A system may run the whole restarted cycle natively (fused C++ operator,
    # Gram-Schmidt, and Givens updates).  It returns ``None`` to decline.
    native = getattr(system, "native_inner_gmres", None)
    if native is not None:
        native_result = native(rhs_high, config)
        if native_result is not None:
            correction, iterations, relative_residual, applications = native_result
            return (
                np.asarray(correction),
                iterations,
                relative_residual,
                applications,
            )

    runtime = system.runtime
    rhs = runtime.from_host(rhs_high)
    correction = runtime.zeros_like(rhs)
    precondition = _low_preconditioner(system)
    rhs_norm = runtime.norm(rhs)
    if rhs_norm == 0.0:
        return np.zeros_like(rhs_high), 0, 0.0, 0

    residual = runtime.copy(rhs)
    applications = 0
    total_iterations = 0
    relative_residual = 1.0

    while total_iterations < config.max_inner_iterations:
        # Recompute the low-precision residual at a restart boundary.  This is
        # the inner analogue of a reliable update and limits Arnoldi drift.
        if total_iterations:
            applied_correction = system.apply_low(correction)
            applications += 1
            residual = runtime.axpy(-1.0, applied_correction, rhs)
        beta = runtime.norm(residual)
        relative_residual = beta / rhs_norm
        if relative_residual <= config.inner_relative_tolerance:
            break

        cycle = min(
            config.gmres_restart,
            config.max_inner_iterations - total_iterations,
        )
        basis: list[Any] = []
        preconditioned_basis: list[Any] = []
        first = runtime.axpy(1.0 / beta, residual, runtime.zeros_like(residual))
        basis.append(first)
        hessenberg = np.zeros((cycle + 1, cycle), dtype=np.complex64)
        right_hand = np.zeros(cycle + 1, dtype=np.complex64)
        right_hand[0] = np.complex64(beta)
        accepted = 0
        coefficients = np.zeros(0, dtype=np.complex64)

        for column in range(cycle):
            preconditioned = precondition(basis[column])
            preconditioned_basis.append(preconditioned)
            candidate = system.apply_low(preconditioned)
            applications += 1

            # Modified Gram-Schmidt reductions consume complex64 vectors.  The
            # tiny Hessenberg least-squares problem stays on the host; it does
            # not use scarce accelerator FP64 execution units.
            for row in range(column + 1):
                coefficient = runtime.dot(basis[row], candidate)
                hessenberg[row, column] = np.complex64(coefficient)
                candidate = runtime.axpy(-coefficient, basis[row], candidate)
            next_norm = runtime.norm(candidate)
            hessenberg[column + 1, column] = np.complex64(next_norm)
            if next_norm > np.finfo(np.float32).eps * beta:
                basis.append(
                    runtime.axpy(
                        1.0 / next_norm,
                        candidate,
                        runtime.zeros_like(candidate),
                    )
                )

            accepted = column + 1
            coefficients, *_ = np.linalg.lstsq(
                hessenberg[: accepted + 1, :accepted].astype(np.complex128),
                right_hand[: accepted + 1].astype(np.complex128),
                rcond=None,
            )
            estimate = right_hand[: accepted + 1] - (
                hessenberg[: accepted + 1, :accepted] @ coefficients
            )
            relative_residual = float(np.linalg.norm(estimate)) / rhs_norm
            total_iterations += 1
            if (
                relative_residual <= config.inner_relative_tolerance
                or next_norm <= np.finfo(np.float32).eps * beta
            ):
                break

        for coefficient, direction in zip(coefficients, preconditioned_basis):
            correction = runtime.axpy(coefficient, direction, correction)
        if relative_residual <= config.inner_relative_tolerance:
            break

    return (
        runtime.to_host(correction),
        total_iterations,
        relative_residual,
        applications,
    )


class _CudaGmresKernels:
    """cuBLAS handles and the small device kernels of the CUDA inner cycle."""

    def __init__(self, xp: Any) -> None:
        self.xp = xp
        self.cublas = importlib.import_module("cupy_backends.cuda.libs.cublas")
        self.runtime_api = xp.cuda.runtime
        self.handle = xp.cuda.device.get_cublas_handle()
        self.one = np.ones(1, dtype=np.complex64)
        self.zero = np.zeros(1, dtype=np.complex64)
        self.minus_one = -np.ones(1, dtype=np.complex64)
        self.nrm2 = importlib.import_module(f"{xp.__name__}.cublas").nrm2
        # out = src * (1 / norm) with the norm read from device memory, so the
        # next Arnoldi column can be enqueued before the host sees the norm.
        self.normalise = xp.ElementwiseKernel(
            "raw T src, raw float32 scratch, int32 slot",
            "T out",
            "out = src[i] * (1.0f / scratch[slot])",
            "mpir_gmres_normalise",
        )

    def gemv_conj(self, basis: Any, active: int, size: int, vector: Any, out: Any) -> None:
        """out = conj(basis[:active]) . vector, reading the basis in place."""

        self.cublas.cgemv(
            self.handle, self.cublas.CUBLAS_OP_C, size, active,
            self.one.ctypes.data, basis.data.ptr, size,
            vector.data.ptr, 1, self.zero.ctypes.data, out.data.ptr, 1,
        )

    def gemv_update(
        self, basis: Any, active: int, size: int, coefficients: Any, vector: Any, sign: np.ndarray
    ) -> None:
        """vector += sign * basis[:active].T . coefficients, in place."""

        self.cublas.cgemv(
            self.handle, self.cublas.CUBLAS_OP_N, size, active,
            sign.ctypes.data, basis.data.ptr, size,
            coefficients.data.ptr, 1, self.one.ctypes.data, vector.data.ptr, 1,
        )


def _inner_gmres_cuda(
    system: MatrixFreeMPIRSystem,
    rhs_high: np.ndarray,
    config: MPIRConfig,
) -> tuple[np.ndarray, int, float, int]:
    """CUDA GMRES with device-resident bases and pipelined CGS2 columns.

    Modified Gram-Schmidt maps poorly to a GPU because Arnoldi column ``j``
    performs ``j`` separate dot products and host-visible scalar decisions.
    Two-pass classical Gram-Schmidt (CGS2) provides comparable orthogonality
    using four cuBLAS GEMV calls per column on the stored basis, read in
    place without a conjugated copy or temporaries.  The candidate is
    normalised on the device from the device-resident norm, so the GPU work of
    column ``j+1`` is enqueued before the host waits for column ``j``'s
    projections; the host then reduces column ``j`` with complex128 Givens
    rotations while the device runs column ``j+1``.  Each column has its own
    scratch row and event, and the pinned host copy of that row is the only
    host-device synchronisation per column.  A column enqueued after the
    cycle has already converged is discarded; its operator application is
    still counted because it ran.
    """

    runtime = system.runtime
    xp = runtime.namespace
    kernels = _CudaGmresKernels(xp)
    rhs = runtime.from_host(rhs_high)
    correction = runtime.zeros_like(rhs)
    size = int(rhs.size)
    rhs_norm = runtime.norm(rhs)
    if rhs_norm == 0.0:
        return np.zeros_like(rhs_high), 0, 0.0, 0

    custom_precondition = getattr(system, "precondition_low", None)
    diagonal = None if custom_precondition is not None else system.diagonal_low()

    restart = config.gmres_restart
    basis = xp.empty((restart + 1, size), dtype=runtime.dtype)
    preconditioned_basis = xp.empty((restart, size), dtype=runtime.dtype)
    # Per column: first-pass projections, second-pass projections, then the
    # candidate norm in the real part of the last complex64 slot.
    row_length = 2 * (restart + 1) + 1
    second_offset = restart + 1
    norm_slot = 2 * (restart + 1)
    scratch = xp.empty((restart, row_length), dtype=runtime.dtype)
    scratch_real = scratch.view(xp.float32)
    pinned = xp.cuda.alloc_pinned_memory(scratch.nbytes)
    # The pinned allocation is rounded up; view only the used prefix.
    host_scratch = np.frombuffer(
        pinned, dtype=np.complex64, count=restart * row_length
    ).reshape(restart, row_length)
    events = [xp.cuda.Event(block=False, disable_timing=True) for _ in range(restart)]
    stream = xp.cuda.get_current_stream()
    eps32 = float(np.finfo(np.float32).eps)

    residual = runtime.copy(rhs)
    applications = 0
    total_iterations = 0
    relative_residual = 1.0

    def enqueue(column: int) -> None:
        """Queue all device work of one Arnoldi column; no host wait."""

        active = column + 1
        if diagonal is not None:
            xp.divide(basis[column], diagonal, out=preconditioned_basis[column])
        else:
            preconditioned_basis[column] = custom_precondition(basis[column])
        candidate = system.apply_low(preconditioned_basis[column])
        row = scratch[column]
        first = row[:active]
        second = row[second_offset : second_offset + active]
        kernels.gemv_conj(basis, active, size, candidate, first)
        kernels.gemv_update(basis, active, size, first, candidate, kernels.minus_one)
        kernels.gemv_conj(basis, active, size, candidate, second)
        kernels.gemv_update(basis, active, size, second, candidate, kernels.minus_one)
        norm_view = scratch_real[column, 2 * norm_slot : 2 * norm_slot + 1].reshape(())
        kernels.nrm2(candidate, out=norm_view)
        kernels.normalise(candidate, scratch_real[column], np.int32(2 * norm_slot), basis[active])
        kernels.runtime_api.memcpyAsync(
            pinned.ptr + column * row_length * 8,
            row.data.ptr,
            row_length * 8,
            kernels.runtime_api.memcpyDeviceToHost,
            stream.ptr,
        )
        events[column].record(stream)

    while total_iterations < config.max_inner_iterations:
        if total_iterations:
            applied_correction = system.apply_low(correction)
            applications += 1
            residual = runtime.axpy(-1.0, applied_correction, rhs)
        beta = runtime.norm(residual)
        relative_residual = beta / rhs_norm
        if relative_residual <= config.inner_relative_tolerance:
            break

        cycle = min(restart, config.max_inner_iterations - total_iterations)
        xp.multiply(residual, np.float32(1.0 / beta), out=basis[0])
        hessenberg = np.zeros((cycle + 1, cycle), dtype=np.complex128)
        givens_c: list[complex] = [0j] * cycle
        givens_s: list[float] = [0.0] * cycle
        right_hand: list[complex] = [0j] * (cycle + 1)
        right_hand[0] = complex(float(np.float32(beta)))
        accepted = 0
        breakdown = False

        enqueue(0)
        applications += 1
        for column in range(cycle):
            active = column + 1
            if active < cycle:
                enqueue(active)
                applications += 1
            events[column].synchronize()
            host_row = host_scratch[column]

            projection = (
                host_row[:active] + host_row[second_offset : second_offset + active]
            ).astype(np.complex64)
            next_norm = float(np.float32(host_row[norm_slot].real))
            breakdown = not (next_norm > eps32 * beta)

            # Givens rotations on the new column in plain complex arithmetic.
            values = projection.astype(np.complex128).tolist() + [complex(next_norm)]
            for row in range(column):
                a = values[row]
                b = values[row + 1]
                values[row] = givens_c[row].conjugate() * a + givens_s[row] * b
                values[row + 1] = -givens_s[row] * a + givens_c[row] * b
            a = values[column]
            b = values[active]
            norm_a = abs(a)
            radius = float(np.hypot(norm_a, abs(b)))
            if radius == 0.0:
                givens_c[column] = 1.0 + 0j
                givens_s[column] = 0.0
            elif norm_a == 0.0:
                givens_c[column] = 0j
                givens_s[column] = 1.0
            else:
                givens_c[column] = (a / norm_a) * (norm_a / radius)
                givens_s[column] = abs(b) / radius
            values[column] = givens_c[column].conjugate() * a + givens_s[column] * b
            values[active] = 0j
            hessenberg[: active + 1, column] = values
            g0 = right_hand[column]
            right_hand[column] = givens_c[column].conjugate() * g0
            right_hand[active] = -givens_s[column] * g0

            accepted = active
            relative_residual = abs(right_hand[accepted]) / rhs_norm
            total_iterations += 1
            if relative_residual <= config.inner_relative_tolerance or breakdown:
                break

        # Back substitution R y = g, then correction += Z y in one GEMV.  Wait
        # for any speculative column first so its writes do not overlap ours.
        stream.synchronize()
        coefficients = np.zeros(accepted, dtype=np.complex128)
        for row in range(accepted - 1, -1, -1):
            acc = right_hand[row]
            for col in range(row + 1, accepted):
                acc -= hessenberg[row, col] * coefficients[col]
            coefficients[row] = acc / hessenberg[row, row]
        coefficients_low = runtime.from_host(np.asarray(coefficients, dtype=np.complex64))
        kernels.gemv_update(
            preconditioned_basis, accepted, size, coefficients_low, correction, kernels.one
        )
        if relative_residual <= config.inner_relative_tolerance:
            break

    return (
        runtime.to_host(correction),
        total_iterations,
        relative_residual,
        applications,
    )


def solve_mpir(
    system: MatrixFreeMPIRSystem,
    rhs: np.ndarray,
    *,
    config: MPIRConfig | None = None,
    initial_guess: np.ndarray | None = None,
) -> MPIRResult:
    """Solve ``A x = rhs`` with host-FP64 refinement and device-FP32 PCG.

    A high-precision matrix-free residual is the reliable update.  Each outer
    step solves only the correction equation at low precision, so nearly all
    operator applications use FP32 while attainable accuracy is governed by
    the repeated FP64 residual, not by a single FP32 solve.
    """

    config = config or MPIRConfig()
    high_dtype = getattr(system, "high_dtype", np.float64)
    rhs_high = np.asarray(rhs, dtype=high_dtype).reshape(-1)
    if rhs_high.size != system.size:
        raise ValueError(f"rhs has size {rhs_high.size}, expected {system.size}")
    if not np.all(np.isfinite(rhs_high)):
        raise ValueError("rhs must contain only finite values")

    if initial_guess is None:
        solution = np.zeros(system.size, dtype=high_dtype)
    else:
        solution = np.asarray(initial_guess, dtype=high_dtype).reshape(-1).copy()
        if solution.size != system.size:
            raise ValueError(
                f"initial_guess has size {solution.size}, expected {system.size}"
            )

    rhs_norm = float(np.linalg.norm(rhs_high))
    scale = rhs_norm if rhs_norm > 0.0 else 1.0
    target = config.absolute_tolerance + config.relative_tolerance * scale
    history: list[MPIRStep] = []
    high_applications = 0
    low_applications = 0
    total_inner = 0

    for outer in range(config.max_outer_iterations + 1):
        residual = rhs_high - system.apply_high(solution)
        high_applications += 1
        residual_norm = float(np.linalg.norm(residual))
        relative_residual = residual_norm / scale
        if residual_norm <= target:
            return MPIRResult(
                solution=solution,
                converged=True,
                outer_iterations=outer,
                inner_iterations=total_inner,
                relative_residual=relative_residual,
                high_operator_applications=high_applications,
                low_operator_applications=low_applications,
                low_runtime=system.runtime.name,
                history=tuple(history),
            )
        if outer == config.max_outer_iterations:
            break

        inner_solver = getattr(system, "inner_solver", "pcg")
        if inner_solver == "pcg":
            correction, inner_count, inner_residual, applications = _inner_pcg(
                system, residual, config
            )
        elif inner_solver == "gmres":
            correction, inner_count, inner_residual, applications = _inner_gmres(
                system, residual, config
            )
        else:
            raise ValueError(f"unsupported inner solver {inner_solver!r}")
        solution += correction
        total_inner += inner_count
        low_applications += applications
        history.append(
            MPIRStep(
                outer_iteration=outer + 1,
                high_relative_residual=relative_residual,
                inner_iterations=inner_count,
                inner_relative_residual=inner_residual,
            )
        )

    return MPIRResult(
        solution=solution,
        converged=False,
        outer_iterations=config.max_outer_iterations,
        inner_iterations=total_inner,
        relative_residual=relative_residual,
        high_operator_applications=high_applications,
        low_operator_applications=low_applications,
        low_runtime=system.runtime.name,
        history=tuple(history),
    )
