"""Shared DICE-PEEC benchmark utility functions."""

from __future__ import annotations

import math
from typing import Any


def relative_difference(reference: float, observed: float) -> float:
    reference = float(reference)
    observed = float(observed)
    if not math.isfinite(reference) or not math.isfinite(observed):
        return math.inf
    scale = max(abs(reference), 1e-30)
    return abs(observed - reference) / scale


def ranking_consistent(
    cpu: dict[str, dict[str, Any]],
    cuda: dict[str, dict[str, Any]],
    metric: str,
    *,
    separation: float = 0.01,
) -> bool:
    """Check every meaningfully separated CPU pair retains its order."""
    if not math.isfinite(separation) or separation < 0.0:
        raise ValueError("separation must be finite and non-negative")
    if set(cpu) != set(cuda):
        return False
    names = sorted(cpu)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            cpu_left = float(cpu[left][metric])
            cpu_right = float(cpu[right][metric])
            cuda_left = float(cuda[left][metric])
            cuda_right = float(cuda[right][metric])
            if not all(
                math.isfinite(value)
                for value in (cpu_left, cpu_right, cuda_left, cuda_right)
            ):
                return False
            scale = max(abs(cpu_left), abs(cpu_right), 1e-30)
            if abs(cpu_left - cpu_right) / scale <= separation:
                continue
            cpu_delta = cpu_left - cpu_right
            cuda_delta = cuda_left - cuda_right
            if cuda_delta == 0.0 or (cpu_delta < 0.0) != (cuda_delta < 0.0):
                return False
    return True


def _select_candidate(
    data: dict[str, Any], candidate_name: str | None
) -> dict[str, Any]:
    if "mask_runs" in data:
        return data
    if "candidate" in data:
        return data["candidate"]
    candidates = list(data.get("candidates", []))
    if candidate_name is not None:
        for candidate in candidates:
            if candidate.get("name") == candidate_name:
                return candidate
        raise ValueError(f"candidate not found: {candidate_name}")
    valid = [candidate for candidate in candidates if candidate.get("valid")]
    if valid:
        return valid[0]
    if candidates:
        return candidates[0]
    raise ValueError("candidate file does not contain mask_runs or candidates")
