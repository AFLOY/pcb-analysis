"""Mixed-precision iterative refinement with a matrix-free FP32 inner PCG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np

from .runtime import LowPrecisionRuntime


class MatrixFreeMPIRSystem(Protocol):
    """Split-precision operator contract.

    The high path consumes and returns host FP64 arrays.  The low path operates
    on the device-native arrays owned by ``runtime``.  No assembled global
    matrix is part of this interface.
    """

    size: int
    runtime: LowPrecisionRuntime
    high_dtype: Any
    inner_solver: Literal["pcg", "gmres"]

    def apply_high(self, vector: np.ndarray) -> np.ndarray: ...

    def apply_low(self, vector: Any) -> Any: ...

    def diagonal_low(self) -> Any: ...


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

    diagonal = system.diagonal_low()
    preconditioned = runtime.divide(residual, diagonal)
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

        preconditioned = runtime.divide(residual, diagonal)
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

    runtime = system.runtime
    rhs = runtime.from_host(rhs_high)
    correction = runtime.zeros_like(rhs)
    diagonal = system.diagonal_low()
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
            preconditioned = runtime.divide(basis[column], diagonal)
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


def _inner_gmres_cuda(
    system: MatrixFreeMPIRSystem,
    rhs_high: np.ndarray,
    config: MPIRConfig,
) -> tuple[np.ndarray, int, float, int]:
    """CUDA GMRES with device-resident bases and batched CGS2 projection.

    Modified Gram-Schmidt maps poorly to a GPU because Arnoldi column ``j``
    performs ``j`` separate dot products and host-visible scalar decisions.
    Two-pass classical Gram-Schmidt (CGS2) provides comparable orthogonality
    using four GEMV operations per column.  Only the small Hessenberg column is
    copied to the host for the complex128 least-squares problem.
    """

    runtime = system.runtime
    xp = runtime.namespace
    rhs = runtime.from_host(rhs_high)
    correction = runtime.zeros_like(rhs)
    diagonal = system.diagonal_low()
    rhs_norm = runtime.norm(rhs)
    if rhs_norm == 0.0:
        return np.zeros_like(rhs_high), 0, 0.0, 0

    residual = runtime.copy(rhs)
    applications = 0
    total_iterations = 0
    relative_residual = 1.0

    while total_iterations < config.max_inner_iterations:
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
        basis = xp.empty((cycle + 1, rhs.size), dtype=runtime.dtype)
        preconditioned_basis = xp.empty(
            (cycle, rhs.size), dtype=runtime.dtype
        )
        basis[0] = runtime.axpy(
            1.0 / beta, residual, runtime.zeros_like(residual)
        )
        hessenberg = np.zeros((cycle + 1, cycle), dtype=np.complex64)
        right_hand = np.zeros(cycle + 1, dtype=np.complex64)
        right_hand[0] = np.complex64(beta)
        accepted = 0
        coefficients = np.zeros(0, dtype=np.complex128)

        for column in range(cycle):
            preconditioned_basis[column] = runtime.divide(
                basis[column], diagonal
            )
            candidate = system.apply_low(preconditioned_basis[column])
            applications += 1

            active_basis = basis[: column + 1]
            first_projection = xp.matmul(active_basis.conj(), candidate)
            candidate = candidate - xp.matmul(first_projection, active_basis)
            second_projection = xp.matmul(active_basis.conj(), candidate)
            candidate = candidate - xp.matmul(second_projection, active_basis)
            projection = runtime.to_host(first_projection + second_projection)
            hessenberg[: column + 1, column] = np.asarray(
                projection, dtype=np.complex64
            )

            next_norm = runtime.norm(candidate)
            hessenberg[column + 1, column] = np.complex64(next_norm)
            if next_norm > np.finfo(np.float32).eps * beta:
                basis[column + 1] = runtime.axpy(
                    1.0 / next_norm,
                    candidate,
                    runtime.zeros_like(candidate),
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

        coefficients_low = runtime.from_host(
            np.asarray(coefficients, dtype=np.complex64)
        )
        update = xp.matmul(
            coefficients_low, preconditioned_basis[:accepted]
        )
        correction = runtime.axpy(1.0, update, correction)
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
