"""Benchmark plane_opt CPU PyPEEC against the CUDA PEEC backend."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable


METRIC_FIELDS = (
    "voltage_span_v",
    "i2r_loss_w",
    "magnetic_energy_j",
    "bulk_p99_current_density_a_per_mm2",
)


def relative_difference(reference: float, observed: float) -> float:
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
    names = sorted(cpu)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            cpu_left = float(cpu[left][metric])
            cpu_right = float(cpu[right][metric])
            scale = max(abs(cpu_left), abs(cpu_right), 1e-30)
            if abs(cpu_left - cpu_right) / scale <= separation:
                continue
            cuda_left = float(cuda[left][metric])
            cuda_right = float(cuda[right][metric])
            if (cpu_left < cpu_right) != (cuda_left < cuda_right):
                return False
    return True


def _timed_solve(
    solve: Callable[..., Any],
    grid: Any,
    case: dict[str, Any],
    config: dict[str, Any],
    *,
    warmups: int,
    repeats: int,
) -> tuple[dict[str, Any], list[float]]:
    for _ in range(warmups):
        solve(grid, case, config)
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = solve(grid, case, config)
        times.append((time.perf_counter() - start) * 1000.0)
    assert result is not None
    return result.metrics, times


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


def benchmark(
    extract: Path,
    config_path: Path,
    *,
    candidate_path: Path | None = None,
    candidate_name: str | None = None,
    warmups: int = 1,
    repeats: int = 10,
    case_names: set[str] | None = None,
) -> dict[str, Any]:
    from plane_opt.current_field_backend import solve_current_field_case
    from plane_opt.geometry_metrics import CopperGrid
    from plane_opt.topology_holes import apply_mask_runs
    from plane_opt.topology_domain import ROLES

    data = json.loads(extract.read_text(encoding="utf-8"))
    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    grid = CopperGrid(data, float(base_config["grid_resolution_mm"]))
    if candidate_path is not None:
        candidate_data = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate = _select_candidate(candidate_data, candidate_name)
        for role in ROLES:
            apply_mask_runs(grid, role, candidate["mask_runs"][role])
    cases = [
        case
        for case in base_config["scenarios"]
        if case_names is None or str(case["name"]) in case_names
    ]
    if not cases:
        raise ValueError("no benchmark scenarios selected")

    configurations = {}
    for backend in ("pypeec", "cuda_peec"):
        configured = copy.deepcopy(base_config)
        configured.setdefault("current_field_solver", {})["backend"] = backend
        if backend == "cuda_peec":
            cuda = configured["current_field_solver"].setdefault("cuda_peec", {})
            # Benchmarking must never report CPU fallback as CUDA timing.
            cuda["fallback_backend"] = None
        configurations[backend] = configured

    results: dict[str, dict[str, Any]] = {"pypeec": {}, "cuda_peec": {}}
    all_times: dict[str, list[float]] = {"pypeec": [], "cuda_peec": []}
    for case in cases:
        name = str(case["name"])
        for backend in ("pypeec", "cuda_peec"):
            metrics, times = _timed_solve(
                solve_current_field_case,
                grid,
                case,
                configurations[backend],
                warmups=warmups,
                repeats=repeats,
            )
            all_times[backend].extend(times)
            results[backend][name] = {
                "median_ms": statistics.median(times),
                "samples_ms": times,
                "metrics": metrics,
            }

    cpu_metrics = {
        name: values["metrics"] for name, values in results["pypeec"].items()
    }
    cuda_metrics = {
        name: values["metrics"] for name, values in results["cuda_peec"].items()
    }
    metric_differences = {
        name: {
            metric: relative_difference(
                float(cpu_metrics[name][metric]), float(cuda_metrics[name][metric])
            )
            for metric in METRIC_FIELDS
        }
        for name in cpu_metrics
    }
    cpu_median = statistics.median(all_times["pypeec"])
    cuda_median = statistics.median(all_times["cuda_peec"])
    case_speedups = {
        name: results["pypeec"][name]["median_ms"]
        / max(results["cuda_peec"][name]["median_ms"], 1e-30)
        for name in cpu_metrics
    }
    max_metric_difference = max(
        difference
        for fields in metric_differences.values()
        for difference in fields.values()
    )
    closure_ok = all(
        float(metrics["current_closure_error_a"]) <= 1e-6
        for metrics in cuda_metrics.values()
    )
    rankings = {
        metric: ranking_consistent(cpu_metrics, cuda_metrics, metric)
        for metric in METRIC_FIELDS
    }
    speedup = cpu_median / max(cuda_median, 1e-30)
    acceptance = {
        "speedup_at_least_2x": speedup >= 2.0,
        "no_case_regression_over_10pct": all(
            value >= 1.0 / 1.10 for value in case_speedups.values()
        ),
        "metrics_within_1pct": max_metric_difference <= 0.01,
        "current_closure_within_1e_6_a": closure_ok,
        "rankings_consistent": all(rankings.values()),
    }
    acceptance["passed"] = all(acceptance.values())
    return {
        "schema_version": 1,
        "extract": str(extract),
        "candidate": str(candidate_path) if candidate_path else None,
        "candidate_name": candidate_name,
        "config": str(config_path),
        "warmups": warmups,
        "repeats": repeats,
        "case_count": len(cases),
        "cpu_median_ms": cpu_median,
        "cuda_median_ms": cuda_median,
        "speedup": speedup,
        "case_speedups": case_speedups,
        "metric_relative_differences": metric_differences,
        "ranking_checks": rankings,
        "acceptance": acceptance,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be non-negative and repeats must be positive")
    report = benchmark(
        args.extract,
        args.config,
        candidate_path=args.candidate,
        candidate_name=args.candidate_name,
        warmups=args.warmups,
        repeats=args.repeats,
        case_names=set(args.case) if args.case else None,
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
