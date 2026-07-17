"""Runtime controller for memory-bounded, error-driven DICE-PEEC execution.

The controller is backend independent. It chooses a fidelity level and a
resource plan from live memory telemetry, measured execution reports, and
candidate error intervals. CUDA-specific calibration is implemented by a
separate backend.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from math import ceil
from typing import Iterable


MIB = 1024**2
GIB = 1024**3


class FidelityStage(IntEnum):
    DELTA = 0
    NEAR_COARSE = 1
    NEAR_FINE = 2
    CORRECTION = 3
    REFINED = 4


@dataclass(frozen=True)
class HardwareTelemetry:
    total_bytes: int
    free_bytes: int
    platform: str = "linux"
    backend: str = "synthetic"


@dataclass(frozen=True)
class ProblemProfile:
    nx: int
    ny: int
    layers: int
    unknowns: int
    candidate_count: int
    changed_cells_mean: int
    frequency_hz: float = 1e6


@dataclass(frozen=True)
class ErrorBudget:
    target_relative: float = 1e-3
    shortlist_fraction: float = 0.20
    audit_fraction: float = 0.01
    required_top_recall: float = 0.99


@dataclass(frozen=True)
class CandidateEstimate:
    candidate_id: str
    score: float
    split_error: float = 0.0
    iterative_error: float = 0.0
    precision_error: float = 0.0
    mesh_error: float = 0.0
    high_risk_topology: bool = False

    @property
    def absolute_error(self) -> float:
        return (
            abs(self.split_error)
            + abs(self.iterative_error)
            + abs(self.precision_error)
            + abs(self.mesh_error)
        )

    @property
    def interval(self) -> tuple[float, float]:
        error = self.absolute_error
        return self.score - error, self.score + error


@dataclass(frozen=True)
class ExecutionPlan:
    stage: FidelityStage
    tile_size: int
    candidate_batch: int
    kernel_batch: int
    near_radius: int
    far_block_width: int
    far_order: int
    field_precision: str
    reduction_precision: str
    solver: str
    restart: int
    tolerance: float
    host_offload: bool
    estimated_bytes: int
    memory_budget_bytes: int


@dataclass(frozen=True)
class ExecutionReport:
    peak_bytes: int
    elapsed_ms: float
    iterations: int = 0
    relative_residual: float = 0.0
    converged: bool = True
    oom: bool = False
    stagnated: bool = False
    precision_gap: float = 0.0
    memory_complete: bool = True


@dataclass
class PolicyState:
    """Mutable accuracy policy learned from exact audits."""

    shortlist_fraction: float = 0.20
    coarse_near_radius: int = 8
    fine_near_radius: int = 16
    audits: int = 0
    misses: int = 0
    stable_audits: int = 0

    @property
    def observed_recall(self) -> float:
        if self.audits == 0:
            return 1.0
        return 1.0 - self.misses / self.audits


class RuntimeModel:
    """Online correction factors for analytic memory and time estimates."""

    def __init__(self, alpha: float = 0.25):
        self.alpha = float(alpha)
        self.memory_scale = 1.0
        self.time_ms: dict[tuple, float] = {}

    def update_memory(self, estimated_bytes: int, observed_bytes: int) -> None:
        if estimated_bytes <= 0 or observed_bytes <= 0:
            return
        ratio = observed_bytes / estimated_bytes
        self.memory_scale = (1.0 - self.alpha) * self.memory_scale + self.alpha * ratio
        self.memory_scale = min(max(self.memory_scale, 0.5), 3.0)

    def update_time(self, plan: ExecutionPlan, elapsed_ms: float) -> None:
        key = self.signature(plan)
        previous = self.time_ms.get(key, elapsed_ms)
        self.time_ms[key] = (1.0 - self.alpha) * previous + self.alpha * elapsed_ms

    @staticmethod
    def signature(plan: ExecutionPlan) -> tuple:
        return (
            int(plan.stage), plan.tile_size, plan.candidate_batch,
            plan.kernel_batch, plan.solver, plan.field_precision,
        )


class MemoryEstimator:
    """Conservative model used before CUDA reports replace assumptions."""

    def __init__(self, runtime_model: RuntimeModel):
        self.runtime_model = runtime_model

    @staticmethod
    def safe_budget(telemetry: HardwareTelemetry) -> int:
        platform = telemetry.platform.lower()
        free_fraction = 0.65 if platform in {"windows", "wddm"} else 0.78
        total_fraction = 0.72 if platform in {"windows", "wddm"} else 0.82
        return int(min(
            telemetry.free_bytes * free_fraction,
            telemetry.total_bytes * total_fraction,
        ))

    def estimate(
        self,
        problem: ProblemProfile,
        *,
        stage: FidelityStage,
        tile_size: int,
        candidate_batch: int,
        kernel_batch: int,
        near_radius: int,
        solver: str,
        restart: int,
        field_precision: str,
        host_offload: bool,
    ) -> int:
        if stage == FidelityStage.DELTA:
            sparse_candidates = (
                candidate_batch * problem.changed_cells_mean * 24
            )
            return int((96 * MIB + sparse_candidates) * self.runtime_model.memory_scale)

        complex_bytes = 16 if field_precision == "complex128" else 8
        padded_x = 2 * tile_size
        padded_y_rfft = tile_size + 1
        spectrum_cells = padded_x * padded_y_rfft
        active_layers = min(max(problem.layers, 1), 8)

        resident_spectra = 2 * active_layers + kernel_batch + 3
        fft_arrays = resident_spectra * spectrum_cells * complex_bytes
        fft_workspace = int(0.55 * fft_arrays) + 64 * MIB

        halo = tile_size + 2 * near_radius
        tile_cells = halo * halo * active_layers
        local_unknowns = min(problem.unknowns, max(tile_cells * 3, 1))
        tile_geometry = tile_cells * 28
        preconditioner = local_unknowns * (48 if stage < FidelityStage.REFINED else 80)

        if solver == "none":
            solver_vectors = 2
        elif solver.startswith("bicgstab") or solver.startswith("idr"):
            solver_vectors = 10
        else:
            solver_vectors = max(restart + 5, 10)
        if host_offload:
            solver_vectors = min(solver_vectors, 8)
        krylov = solver_vectors * local_unknowns * complex_bytes

        accepted_cache = 4 * local_unknowns * complex_bytes
        sparse_candidates = candidate_batch * problem.changed_cells_mean * 24
        fixed_runtime = 160 * MIB
        total = (
            fft_arrays + fft_workspace + tile_geometry + preconditioner
            + krylov + accepted_cache + sparse_candidates + fixed_runtime
        )
        return int(total * self.runtime_model.memory_scale)


class DynamicController:
    """Selects the cheapest safe plan and escalates only on evidence."""

    TILE_OPTIONS = (512, 384, 256, 192, 128, 96, 64)

    def __init__(self, error_budget: ErrorBudget | None = None):
        self.error_budget = error_budget or ErrorBudget()
        self.policy = PolicyState(
            shortlist_fraction=self.error_budget.shortlist_fraction
        )
        self.runtime = RuntimeModel()
        self.memory = MemoryEstimator(self.runtime)

    def make_plan(
        self,
        problem: ProblemProfile,
        telemetry: HardwareTelemetry,
        stage: FidelityStage,
        previous_report: ExecutionReport | None = None,
        previous_plan: ExecutionPlan | None = None,
    ) -> ExecutionPlan:
        budget = self.memory.safe_budget(telemetry)
        defaults = self._stage_defaults(stage, previous_report)

        max_tile = min(max(problem.nx, problem.ny), self.TILE_OPTIONS[0])
        tile_options = [v for v in self.TILE_OPTIONS if v <= max_tile]
        if previous_report and previous_report.oom and previous_plan:
            tile_options = [v for v in tile_options if v < previous_plan.tile_size]
            if not tile_options:
                tile_options = [64]

        candidates: list[ExecutionPlan] = []
        for host_offload in (False, True):
            for tile in tile_options:
                for kernel_batch in defaults["kernel_batches"]:
                    for candidate_batch in defaults["candidate_batches"]:
                        estimated = self.memory.estimate(
                            problem,
                            stage=stage,
                            tile_size=tile,
                            candidate_batch=candidate_batch,
                            kernel_batch=kernel_batch,
                            near_radius=defaults["near_radius"],
                            solver=defaults["solver"],
                            restart=defaults["restart"],
                            field_precision=defaults["field_precision"],
                            host_offload=host_offload,
                        )
                        if estimated <= budget:
                            candidates.append(ExecutionPlan(
                                stage=stage,
                                tile_size=tile,
                                candidate_batch=candidate_batch,
                                kernel_batch=kernel_batch,
                                near_radius=defaults["near_radius"],
                                far_block_width=defaults["far_block_width"],
                                far_order=1,
                                field_precision=defaults["field_precision"],
                                reduction_precision="float64",
                                solver=defaults["solver"],
                                restart=defaults["restart"],
                                tolerance=defaults["tolerance"],
                                host_offload=host_offload,
                                estimated_bytes=estimated,
                                memory_budget_bytes=budget,
                            ))
            if candidates:
                break

        if not candidates:
            raise MemoryError(
                f"no safe plan: budget={budget / MIB:.1f} MiB; "
                "reduce the problem ROI or enable CPU-only fallback"
            )
        return max(candidates, key=self._throughput_score)

    def observe(self, plan: ExecutionPlan, report: ExecutionReport) -> None:
        if not report.oom and report.memory_complete:
            self.runtime.update_memory(plan.estimated_bytes, report.peak_bytes)
        if not report.oom:
            self.runtime.update_time(plan, report.elapsed_ms)

    def record_accuracy_audit(
        self,
        *,
        truly_promising: bool,
        was_shortlisted: bool,
        coarse_fine_relative_gap: float,
    ) -> None:
        """Adapt fidelity thresholds from occasional higher-accuracy audits."""
        self.policy.audits += 1
        missed = truly_promising and not was_shortlisted
        if missed:
            self.policy.misses += 1
            self.policy.stable_audits = 0
            self.policy.shortlist_fraction = min(
                0.50, self.policy.shortlist_fraction + 0.05
            )
            self.policy.coarse_near_radius = min(
                self.policy.fine_near_radius, self.policy.coarse_near_radius * 2
            )
            self.policy.fine_near_radius = min(64, self.policy.fine_near_radius * 2)
            return

        if coarse_fine_relative_gap > self.error_budget.target_relative:
            self.policy.stable_audits = 0
            self.policy.coarse_near_radius = min(32, self.policy.coarse_near_radius * 2)
            self.policy.fine_near_radius = min(64, self.policy.fine_near_radius * 2)
        else:
            self.policy.stable_audits += 1

        # Relax only after a long stable window and never below calibrated defaults.
        if self.policy.stable_audits >= 100:
            self.policy.shortlist_fraction = max(
                self.error_budget.shortlist_fraction,
                self.policy.shortlist_fraction - 0.02,
            )
            if coarse_fine_relative_gap < self.error_budget.target_relative * 0.25:
                self.policy.coarse_near_radius = max(8, self.policy.coarse_near_radius // 2)
                self.policy.fine_near_radius = max(16, self.policy.fine_near_radius // 2)
            self.policy.stable_audits = 0

    def select_for_promotion(
        self,
        estimates: Iterable[CandidateEstimate],
    ) -> list[str]:
        items = sorted(estimates, key=lambda item: item.score)
        if not items:
            return []
        shortlist_count = max(1, ceil(len(items) * self.policy.shortlist_fraction))
        promoted = {item.candidate_id for item in items[:shortlist_count]}
        best = items[0]
        best_low, best_high = best.interval
        for item in items:
            low, high = item.interval
            intervals_overlap = not (high < best_low or low > best_high)
            relative_error = item.absolute_error / max(abs(item.score), 1e-30)
            if (
                item.high_risk_topology
                or intervals_overlap
                or relative_error > self.error_budget.target_relative
            ):
                promoted.add(item.candidate_id)
        return sorted(promoted)

    @staticmethod
    def next_stage(stage: FidelityStage) -> FidelityStage:
        return FidelityStage(min(int(stage) + 1, int(FidelityStage.REFINED)))

    @staticmethod
    def _throughput_score(plan: ExecutionPlan) -> tuple:
        # Prefer on-device execution, then useful parallel work and tile size.
        return (
            not plan.host_offload,
            plan.candidate_batch * plan.kernel_batch * plan.tile_size**2,
            -plan.estimated_bytes,
        )

    def _stage_defaults(
        self, stage: FidelityStage, report: ExecutionReport | None
    ) -> dict:
        if stage == FidelityStage.DELTA:
            return dict(
                near_radius=0, far_block_width=2, field_precision="complex64",
                solver="none", restart=0, tolerance=1e-2,
                candidate_batches=(64, 32, 16, 8, 4, 2, 1),
                kernel_batches=(1,),
            )
        if stage == FidelityStage.NEAR_COARSE:
            return dict(
                near_radius=self.policy.coarse_near_radius,
                far_block_width=2, field_precision="complex64",
                solver="none", restart=0, tolerance=1e-3,
                candidate_batches=(32, 16, 8, 4, 2, 1),
                kernel_batches=(8, 4, 2, 1),
            )
        if stage == FidelityStage.NEAR_FINE:
            return dict(
                near_radius=self.policy.fine_near_radius,
                far_block_width=2, field_precision="complex64",
                solver="none", restart=0, tolerance=2e-4,
                candidate_batches=(16, 8, 4, 2, 1),
                kernel_batches=(8, 4, 2, 1),
            )

        stagnated = bool(report and (report.stagnated or not report.converged))
        precision_bad = bool(report and report.precision_gap > 1e-5)
        if stage == FidelityStage.CORRECTION and not stagnated:
            solver, restart = "bicgstab2", 0
        else:
            solver, restart = "gmres", 12 if stage == FidelityStage.CORRECTION else 20
        return dict(
            near_radius=(
                self.policy.fine_near_radius
                if stage == FidelityStage.CORRECTION
                else min(64, self.policy.fine_near_radius * 2)
            ),
            far_block_width=2 if stage == FidelityStage.CORRECTION else 1,
            field_precision="complex128" if precision_bad and stage == FidelityStage.CORRECTION else "complex64",
            solver=solver,
            restart=restart,
            tolerance=1e-4 if stage == FidelityStage.CORRECTION else 1e-6,
            candidate_batches=(2, 1) if stage == FidelityStage.CORRECTION else (1,),
            kernel_batches=(4, 2, 1),
        )
