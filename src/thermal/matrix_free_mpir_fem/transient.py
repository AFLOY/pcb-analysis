"""Transient conduction by backward Euler on the steady operator.

With the element heat capacity ``ρ c`` lumped onto the nodes (``C``), one
implicit step reads

```text
(K + R + C / Δt) T_{n+1} = q + R T_amb + C / Δt T_n,
```

so the transient operator is the steady one with ``C / Δt`` added to the
diagonal, exactly where the Robin conductance already sits.  Nothing else
changes: the same matrix-free kernels (NumPy, CuPy, C++), the same two-level
preconditioner, the same radiation Newton loop inside each step, and the
step is better conditioned than the steady solve the larger ``C / Δt`` is.

Backward Euler is first order in ``Δt`` and unconditionally stable, so a
schedule that grows geometrically from a first step resolving the fastest
time constant of interest to a last step of the order of the slowest one
walks from ``t = 0`` to the steady state in a few dozen steps
(``TimeSchedule.geometric``); ``until_steady`` stops once the field moves
less than ``steady_tolerance_k_per_s`` per second.  ``TimeSchedule.uniform``
gives constant steps.  The operator is rebuilt only when ``Δt`` changes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from electrical.matrix_free_mpir_fem.runtime import LowPrecisionRuntime, RuntimeBackend
from electrical.matrix_free_mpir_fem.solver import MPIRConfig

from .mesh import Preconditioner
from .problem import ThermalConductionProblem
from .solve import ThermalConductionSolution, _solve


@dataclass(frozen=True)
class TimeSchedule:
    """Time points ``0 = t_0 < t_1 < ... < t_N`` of a transient run, in seconds."""

    times_s: tuple[float, ...]

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=np.float64).reshape(-1)
        if times.size < 2 or times[0] != 0.0:
            raise ValueError("times_s must start at 0 and hold at least one step")
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            raise ValueError("times_s must be finite and strictly increasing")
        object.__setattr__(self, "times_s", tuple(times.tolist()))

    @property
    def steps_s(self) -> np.ndarray:
        return np.diff(np.asarray(self.times_s))

    @property
    def end_s(self) -> float:
        return self.times_s[-1]

    @classmethod
    def uniform(cls, step_s: float, end_s: float) -> "TimeSchedule":
        if step_s <= 0.0 or end_s <= 0.0:
            raise ValueError("step_s and end_s must be positive")
        count = int(np.ceil(end_s / step_s - 1.0e-9))
        return cls(tuple(np.linspace(0.0, count * step_s, count + 1).tolist()))

    @classmethod
    def geometric(
        cls, first_step_s: float, end_s: float, *, growth: float = 1.5, max_step_s: float | None = None
    ) -> "TimeSchedule":
        """Steps growing by ``growth`` from ``first_step_s``, the last one trimmed to ``end_s``."""

        if first_step_s <= 0.0 or end_s <= first_step_s * 0.0 or end_s <= 0.0:
            raise ValueError("first_step_s and end_s must be positive")
        if growth < 1.0:
            raise ValueError("growth must be at least one")
        times = [0.0]
        step = float(first_step_s)
        while times[-1] < end_s - 1.0e-12 * end_s:
            times.append(min(times[-1] + step, end_s))
            step *= growth
            if max_step_s is not None:
                step = min(step, float(max_step_s))
        return cls(tuple(times))


@dataclass(frozen=True)
class TransientStep:
    index: int
    time_s: float
    step_s: float
    max_temperature_k: float
    max_change_k: float
    stored_heat_w: float
    convective_heat_w: float
    radiative_heat_w: float
    heat_balance_error_w: float
    inner_iterations: int
    radiation_iterations: int
    converged: bool


@dataclass(frozen=True)
class TransientThermalSolution:
    """Temperature history of a backward-Euler run and the last step's budget."""

    times_s: np.ndarray
    temperature_k: np.ndarray
    """``(len(times_s), *node_shape)`` when every step is stored, else the final field only ``(1, ...)``."""
    history: tuple[TransientStep, ...]
    final: ThermalConductionSolution
    reached_steady: bool

    @property
    def max_temperature_k(self) -> np.ndarray:
        return np.asarray([step.max_temperature_k for step in self.history])

    @property
    def final_temperature_k(self) -> np.ndarray:
        return self.temperature_k[-1]


def solve_thermal_transient(
    problem: ThermalConductionProblem,
    schedule: TimeSchedule,
    *,
    initial_temperature_k: np.ndarray | float | None = None,
    store: str = "all",
    until_steady: bool = False,
    steady_tolerance_k_per_s: float = 1.0e-3,
    config: MPIRConfig | None = None,
    runtime: LowPrecisionRuntime | None = None,
    backend: RuntimeBackend | None = None,
    device_id: int = 0,
    preconditioner: Preconditioner = "two-level",
    coarse_block_nodes: int | None = None,
    native: bool | None = None,
    native_threads: int | None = None,
    radiation_max_iterations: int = 25,
    radiation_tolerance_k: float = 1.0e-4,
) -> TransientThermalSolution:
    """March ``problem`` from ``initial_temperature_k`` through ``schedule`` by backward Euler.

    The mesh must carry ``volumetric_heat_capacity_j_per_m3_k``.  The initial
    field defaults to the first convective (else radiative) ambient, else the
    mean fixed temperature.  ``store="all"`` keeps every field, ``"final"``
    only the last.  With ``until_steady`` the march stops early once
    ``max |T_{n+1} - T_n| / Δt`` falls below ``steady_tolerance_k_per_s``.
    """

    mesh = problem.mesh
    if store not in ("all", "final"):
        raise ValueError("store must be 'all' or 'final'")
    if steady_tolerance_k_per_s <= 0.0:
        raise ValueError("steady_tolerance_k_per_s must be positive")
    capacity = mesh.nodal_heat_capacity_j_per_k().reshape(-1)

    if initial_temperature_k is None:
        if problem.convection:
            start = problem.convection[0].mean_ambient_k()
        elif problem.radiation:
            start = problem.radiation[0].mean_ambient_k()
        else:
            mask = problem.fixed_temperature_mask
            start = float(np.mean(problem.fixed_temperature_k[mask]))
        current = np.full(mesh.node_shape, start)
    else:
        guess = np.asarray(initial_temperature_k, dtype=np.float64)
        current = np.full(mesh.node_shape, float(guess)) if guess.ndim == 0 else guess.reshape(mesh.node_shape).copy()
        if not np.all(np.isfinite(current[mesh.active_nodes])):
            raise ValueError("initial_temperature_k must be finite on active nodes")
    # Fixed nodes start at their prescribed value; inactive nodes are NaN in the output.
    current = np.where(problem.fixed_temperature_mask, problem.fixed_temperature_k, current)
    active = mesh.active_nodes
    fields = [np.where(active, current, np.nan)]
    times = [0.0]
    history: list[TransientStep] = []
    final: ThermalConductionSolution | None = None
    reached_steady = False
    options = dict(
        config=config, runtime=runtime, backend=backend, device_id=device_id,
        preconditioner=preconditioner, coarse_block_nodes=coarse_block_nodes,
        native=native, native_threads=native_threads,
        radiation_max_iterations=radiation_max_iterations, radiation_tolerance_k=radiation_tolerance_k,
    )
    for index, (time_s, step_s) in enumerate(zip(schedule.times_s[1:], schedule.steps_s), start=1):
        final = _solve(
            problem,
            initial_temperature_k=current,
            capacity_per_s=capacity / float(step_s),
            previous_temperature_k=current,
            **options,
        )
        proposed = np.where(np.isfinite(final.temperature_k), final.temperature_k, current)
        change = float(np.max(np.abs(proposed - current)[active])) if np.any(active) else 0.0
        current = proposed
        times.append(float(time_s))
        if store == "all":
            fields.append(np.where(active, current, np.nan))
        else:
            fields = [np.where(active, current, np.nan)]
        history.append(
            TransientStep(
                index=index,
                time_s=float(time_s),
                step_s=float(step_s),
                max_temperature_k=float(np.nanmax(fields[-1])),
                max_change_k=change,
                stored_heat_w=final.stored_heat_w,
                convective_heat_w=float(np.sum(final.convective_heat_w)),
                radiative_heat_w=float(np.sum(final.radiative_heat_w)),
                heat_balance_error_w=final.heat_balance_error_w,
                inner_iterations=final.solve.inner_iterations,
                radiation_iterations=final.radiation_iterations,
                converged=final.solve.converged and final.radiation_converged,
            )
        )
        if until_steady and change / float(step_s) <= steady_tolerance_k_per_s:
            reached_steady = True
            break
    assert final is not None
    stored_times = np.asarray(times) if store == "all" else np.asarray(times[-1:])
    return TransientThermalSolution(
        times_s=stored_times,
        temperature_k=np.stack(fields),
        history=tuple(history),
        final=final,
        reached_steady=reached_steady,
    )


__all__ = ["TimeSchedule", "TransientStep", "TransientThermalSolution", "solve_thermal_transient"]
