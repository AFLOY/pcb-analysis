"""What a CUDA voxel solve saw of its device and what it cost."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareTelemetry:
    """Device memory at the time of a probe."""

    total_bytes: int
    free_bytes: int
    platform: str = "linux"
    backend: str = "synthetic"


@dataclass(frozen=True)
class ExecutionReport:
    """Peak memory, wall time and convergence of one solve."""

    peak_bytes: int
    elapsed_ms: float
    iterations: int = 0
    relative_residual: float = 0.0
    converged: bool = True
    oom: bool = False
    stagnated: bool = False
    precision_gap: float = 0.0
    memory_complete: bool = True
