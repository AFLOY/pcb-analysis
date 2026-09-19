"""Benchmark plane_opt CPU PyPEEC against the CUDA PEEC backend."""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable


from electrical.dice_peec.benchmark import (
    _select_candidate,
    ranking_consistent,
    relative_difference,
)


METRIC_FIELDS = (
    "voltage_span_v",
    "i2r_loss_w",
    "magnetic_energy_j",
    "bulk_p99_current_density_a_per_mm2",
)


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
    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    # plane_opt is a separate project and is not a dependency of this package,
    # so it is imported at call time and only for this benchmark.
    from plane_opt.configuration import load_adopted_pipeline_config
    from plane_opt.domain.topology import (
        pipeline_role_context,
        resolve_pipeline_roles,
    )
    from plane_opt.geometry.connectivity import apply_mask_runs
    from plane_opt.geometry.metrics import CopperGrid
    from plane_opt.physics.current_field import solve_current_field_case

    data = json.loads(extract.read_text(encoding="utf-8"))
    # A board file states only what is specific to that board; the algorithm
    # defaults, and so grid_resolution_mm, come from plane_opt itself. Reading
    # the file directly would give a configuration missing most of its keys.
    base_config = load_adopted_pipeline_config(config_path)
    # Roles come from the configuration rather than from a module-level
    # constant: plane_opt resolves them per board and exposes them only inside
    # a role context, where an unset context reads as empty rather than as some
    # other board's roles.
    roles = resolve_pipeline_roles(base_config)
    grid = CopperGrid(data, float(base_config["grid_resolution_mm"]))
    if candidate_path is not None:
        candidate_data = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate = _select_candidate(candidate_data, candidate_name)
        missing = sorted(set(roles) - set(candidate["mask_runs"]))
        if missing:
            raise ValueError(f"candidate is missing masks for roles: {missing}")
        for role in roles:
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
            with pipeline_role_context(roles):
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
        abs(float(metrics["current_closure_error_a"])) <= 1e-6
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
