"""Execution backend contracts and a deterministic CPU-side simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .controller import (
    ExecutionPlan,
    ExecutionReport,
    HardwareTelemetry,
    ProblemProfile,
)


class CalibrationBackend(Protocol):
    def probe(self) -> HardwareTelemetry: ...

    def execute(self, plan: ExecutionPlan, problem: ProblemProfile) -> ExecutionReport: ...


@dataclass
class SyntheticBackend:
    """Predictable backend for controller tests without a CUDA device."""

    total_bytes: int
    platform: str = "windows"
    free_fraction: float = 0.90
    memory_multiplier: float = 1.08
    force_oom: bool = False
    force_stagnation: bool = False

    def probe(self) -> HardwareTelemetry:
        return HardwareTelemetry(
            total_bytes=self.total_bytes,
            free_bytes=int(self.total_bytes * self.free_fraction),
            platform=self.platform,
            backend="synthetic",
        )

    def execute(self, plan: ExecutionPlan, problem: ProblemProfile) -> ExecutionReport:
        peak = int(plan.estimated_bytes * self.memory_multiplier)
        oom = self.force_oom or peak > self.probe().free_bytes
        work = (
            problem.nx * problem.ny * max(problem.layers, 1)
            / max(plan.tile_size**2 * plan.candidate_batch, 1)
        )
        elapsed = 0.08 * work * max(1, 8 / max(plan.kernel_batch, 1))
        if plan.host_offload:
            elapsed *= 1.8
        iterations = 0 if plan.solver == "none" else (34 if plan.solver == "gmres" else 52)
        stagnated = self.force_stagnation and plan.solver.startswith("bicgstab")
        return ExecutionReport(
            peak_bytes=peak,
            elapsed_ms=elapsed,
            iterations=iterations,
            relative_residual=2e-3 if stagnated else plan.tolerance * 0.5,
            converged=not (oom or stagnated),
            oom=oom,
            stagnated=stagnated,
            precision_gap=0.0,
        )

