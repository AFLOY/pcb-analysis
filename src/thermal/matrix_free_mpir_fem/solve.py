"""Solve the conduction problem with MPIR and report the heat budget."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from electrical.matrix_free_mpir_fem.runtime import LowPrecisionRuntime, RuntimeBackend
from electrical.matrix_free_mpir_fem.solver import MPIRConfig, MPIRResult, solve_mpir

from .mesh import Preconditioner
from .operator import MatrixFreeThermalOperator
from .problem import ThermalConductionProblem


@dataclass(frozen=True)
class ThermalConductionSolution:
    """Nodal temperatures and the heat budget of one steady solve."""

    temperature_k: np.ndarray
    heat_flux_w_per_m2: np.ndarray
    max_temperature_k: float
    min_temperature_k: float
    total_heat_input_w: float
    convective_heat_w: np.ndarray
    fixed_temperature_heat_w: float
    heat_balance_error_w: float
    solve: MPIRResult
    radiative_heat_w: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    radiation_iterations: int = 0
    radiation_converged: bool = True
    radiation_change_k: float = 0.0
    stored_heat_w: float = 0.0


def solve_thermal_conduction(
    problem: ThermalConductionProblem,
    *,
    config: MPIRConfig | None = None,
    runtime: LowPrecisionRuntime | None = None,
    backend: RuntimeBackend | None = None,
    device_id: int = 0,
    initial_temperature_k: np.ndarray | float | None = None,
    preconditioner: Preconditioner = "two-level",
    coarse_block_nodes: int | None = None,
    reference_temperature_k: float | None = None,
    native: bool | None = None,
    native_threads: int | None = None,
    radiation_max_iterations: int = 25,
    radiation_tolerance_k: float = 1.0e-4,
) -> ThermalConductionSolution:
    """Solve steady heat conduction in a layered PCB with matrix-free MPIR.

    The solver works on the rise above ``reference_temperature_k`` (default:
    the first convective ambient, else the mean fixed temperature) so the
    outer FP64 residual is measured against the heat load rather than against
    ``300 K`` of absolute temperature.  ``initial_temperature_k`` warm-starts
    the outer refinement; a previous solution shortens the solve.  Without it
    the solve starts from the reference temperature.

    With ``problem.radiation`` the problem is nonlinear: each radiating face is
    replaced by its Newton linearisation at the current temperature, the
    linear problem is solved, and the two repeat until the nodal temperatures
    move by less than ``radiation_tolerance_k`` (Newton's method on the whole
    problem, since conduction is linear).  The first linearisation uses the
    warm start, else the radiative ambient.  ``radiative_heat_w`` then holds
    the heat each radiation boundary removes and ``convective_heat_w`` only
    the true convection boundaries.
    """

    return _solve(
        problem, config=config, runtime=runtime, backend=backend, device_id=device_id,
        initial_temperature_k=initial_temperature_k, preconditioner=preconditioner,
        coarse_block_nodes=coarse_block_nodes, reference_temperature_k=reference_temperature_k,
        native=native, native_threads=native_threads,
        radiation_max_iterations=radiation_max_iterations, radiation_tolerance_k=radiation_tolerance_k,
    )


def _solve(
    problem: ThermalConductionProblem,
    *,
    config: MPIRConfig | None = None,
    runtime: LowPrecisionRuntime | None = None,
    backend: RuntimeBackend | None = None,
    device_id: int = 0,
    initial_temperature_k: np.ndarray | float | None = None,
    preconditioner: Preconditioner = "two-level",
    coarse_block_nodes: int | None = None,
    reference_temperature_k: float | None = None,
    native: bool | None = None,
    native_threads: int | None = None,
    radiation_max_iterations: int = 25,
    radiation_tolerance_k: float = 1.0e-4,
    capacity_per_s: np.ndarray | None = None,
    previous_temperature_k: np.ndarray | None = None,
) -> ThermalConductionSolution:
    """Steady solve, or one backward-Euler step when ``capacity_per_s`` is given."""

    linear_options = dict(
        config=config, runtime=runtime, backend=backend, device_id=device_id,
        preconditioner=preconditioner, coarse_block_nodes=coarse_block_nodes,
        reference_temperature_k=reference_temperature_k, native=native, native_threads=native_threads,
        capacity_per_s=capacity_per_s, previous_temperature_k=previous_temperature_k,
    )
    if not problem.radiation:
        return _solve_linear(problem, initial_temperature_k=initial_temperature_k, **linear_options)
    if radiation_max_iterations < 1:
        raise ValueError("radiation_max_iterations must be positive")
    if not radiation_tolerance_k > 0.0:
        raise ValueError("radiation_tolerance_k must be positive")

    mesh = problem.mesh
    if initial_temperature_k is None:
        start = float(problem.radiation[0].mean_ambient_k())
        if problem.convection:
            start = problem.convection[0].mean_ambient_k()
        current = np.full(mesh.node_shape, start)
    else:
        guess = np.asarray(initial_temperature_k, dtype=np.float64)
        current = np.full(mesh.node_shape, float(guess)) if guess.ndim == 0 else guess.reshape(mesh.node_shape).copy()
        current = np.where(np.isfinite(current), current, problem.radiation[0].mean_ambient_k())
    convection_count = len(problem.convection)
    solution: ThermalConductionSolution | None = None
    change = float("inf")
    converged = False
    for iteration in range(1, radiation_max_iterations + 1):
        linearised = tuple(boundary.linearize(mesh, current) for boundary in problem.radiation)
        linear = dataclasses.replace(
            problem, convection=tuple(problem.convection) + linearised, radiation=()
        )
        solution = _solve_linear(linear, initial_temperature_k=current, **linear_options)
        proposed = np.where(np.isfinite(solution.temperature_k), solution.temperature_k, current)
        change = float(np.max(np.abs(proposed - current)))
        current = proposed
        if change <= radiation_tolerance_k:
            converged = True
            break
    assert solution is not None
    heat = solution.convective_heat_w
    return dataclasses.replace(
        solution,
        convective_heat_w=heat[:convection_count],
        radiative_heat_w=heat[convection_count:],
        radiation_iterations=iteration,
        radiation_converged=converged and solution.solve.converged,
        radiation_change_k=change,
    )


def _solve_linear(
    problem: ThermalConductionProblem,
    *,
    config: MPIRConfig | None,
    runtime: LowPrecisionRuntime | None,
    backend: RuntimeBackend | None,
    device_id: int,
    initial_temperature_k: np.ndarray | float | None,
    preconditioner: Preconditioner,
    coarse_block_nodes: int | None,
    reference_temperature_k: float | None,
    native: bool | None,
    native_threads: int | None,
    capacity_per_s: np.ndarray | None = None,
    previous_temperature_k: np.ndarray | None = None,
) -> ThermalConductionSolution:
    operator = MatrixFreeThermalOperator(
        problem,
        runtime=runtime,
        backend=backend,
        device_id=device_id,
        preconditioner=preconditioner,
        coarse_block_nodes=coarse_block_nodes,
        native=native,
        native_threads=native_threads,
        capacity_per_s=capacity_per_s,
    )
    previous = None
    if previous_temperature_k is not None:
        previous = np.asarray(previous_temperature_k, dtype=np.float64).reshape(-1)
        previous = np.where(np.isfinite(previous), previous, 0.0)
    reference = (
        operator.default_reference_temperature()
        if reference_temperature_k is None
        else float(reference_temperature_k)
    )
    if not np.isfinite(reference):
        raise ValueError("reference_temperature_k must be finite")
    rhs = operator.build_rhs(reference, previous)
    if initial_temperature_k is None:
        initial = None
    else:
        guess = np.asarray(initial_temperature_k, dtype=np.float64)
        if guess.ndim == 0:
            initial = np.full(operator.size, float(guess))
        else:
            initial = guess.reshape(-1).copy()
        if initial.size != operator.size:
            raise ValueError("initial_temperature_k must hold one value per node")
        initial = np.where(
            operator.fixed_nodes.reshape(-1),
            operator.fixed_temperature_k.reshape(-1) + reference * ~operator.active_nodes.reshape(-1),
            initial,
        ) - reference
    if config is None:
        # A stiff copper/laminate stack lets the FP32 inner solve buy only
        # about one decade per outer step; give the FP64 refinement room.
        config = MPIRConfig(max_outer_iterations=16)
    result = solve_mpir(operator, rhs, config=config, initial_guess=initial)
    free = operator.free_nodes.reshape(-1)
    if not result.converged and np.any(free):
        # The FP64 residual of ``K theta`` floors at eps * ||K|| * ||theta||.  A
        # plate that sits far above the reference, almost uniformly, hits that
        # floor before the requested tolerance.  Re-reference to the mean rise
        # and continue from the current iterate; the remaining unknown is the
        # small in-plane variation, whose residual is accurate.
        shift = float(np.mean(result.solution[free]))
        if abs(shift) > 0.0:
            reference += shift
            rhs = operator.build_rhs(reference, previous)
            resumed = solve_mpir(
                operator, rhs, config=config, initial_guess=result.solution - shift
            )
            result = MPIRResult(
                solution=resumed.solution,
                converged=resumed.converged,
                outer_iterations=result.outer_iterations + resumed.outer_iterations,
                inner_iterations=result.inner_iterations + resumed.inner_iterations,
                relative_residual=resumed.relative_residual,
                high_operator_applications=result.high_operator_applications
                + resumed.high_operator_applications,
                low_operator_applications=result.low_operator_applications
                + resumed.low_operator_applications,
                low_runtime=resumed.low_runtime,
                history=result.history + resumed.history,
            )
    temperature = result.solution.reshape(problem.mesh.node_shape) + reference

    load = operator.nodal_load()
    residual = operator.unconstrained_residual(temperature, previous)
    fixed = operator.fixed_nodes.reshape(-1)
    convective = operator.convective_heat(temperature)
    fixed_heat = -float(np.sum(residual[fixed]))
    stored = float(np.sum(operator.stored_heat_w(temperature, previous)))
    total_input = float(np.sum(load))
    heat_flux = operator.element_heat_flux(temperature)
    active_nodes = operator.active_nodes
    if not problem.mesh.is_full:
        heat_flux = np.where(problem.mesh.active[..., None], heat_flux, 0.0)  # type: ignore[index]
        temperature = np.where(active_nodes, temperature, np.nan)
    return ThermalConductionSolution(
        temperature_k=temperature,
        heat_flux_w_per_m2=heat_flux,
        max_temperature_k=float(np.nanmax(temperature)),
        min_temperature_k=float(np.nanmin(temperature)),
        total_heat_input_w=total_input,
        convective_heat_w=convective,
        fixed_temperature_heat_w=fixed_heat,
        heat_balance_error_w=total_input - float(np.sum(convective)) - fixed_heat - stored,
        solve=result,
        stored_heat_w=stored,
    )
