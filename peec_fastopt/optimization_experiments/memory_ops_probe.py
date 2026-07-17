"""Probe CUDA memory behavior of the physical PyPEEC executor.

Measures:
1. controller plans under live free/total VRAM
2. sequential multi-case solves: free VRAM, CuPy pool, reported peaks
3. voxel-cache on vs off (memory retention vs remesh cost)
4. post-solve pool free_all_blocks effect
5. memory_reserve_fraction impact on usable limit
6. OOM recovery path under artificially tight pool limits
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _mem_snapshot(cp: Any, label: str) -> dict[str, Any]:
    free_b, total_b = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    pinned = cp.get_default_pinned_memory_pool()
    return {
        "label": label,
        "free_bytes": int(free_b),
        "total_bytes": int(total_b),
        "used_device_bytes": int(total_b - free_b),
        "pool_used_bytes": int(pool.used_bytes()),
        "pool_total_bytes": int(pool.total_bytes()),
        "pool_limit_bytes": int(pool.get_limit()),
        "pinned_n_free_blocks": int(pinned.n_free_blocks()),
    }


def _mib(n: int) -> float:
    return n / (1024**2)


def run_controller_live() -> dict[str, Any]:
    from peec_fastopt.controller import DynamicController, FidelityStage, ProblemProfile
    from peec_fastopt.cupy_calibrator import CuPyCalibrationBackend

    backend = CuPyCalibrationBackend(repeats=5)
    telemetry = backend.probe()
    problem = ProblemProfile(
        nx=1024,
        ny=1024,
        layers=4,
        unknowns=2_000_000,
        candidate_count=1000,
        changed_cells_mean=32,
    )
    controller = DynamicController()
    stages = []
    for stage in FidelityStage:
        plan = controller.make_plan(problem, telemetry, stage)
        stages.append(
            {
                "stage": stage.name,
                "tile_size": plan.tile_size,
                "candidate_batch": plan.candidate_batch,
                "kernel_batch": plan.kernel_batch,
                "solver": plan.solver,
                "field_precision": plan.field_precision,
                "estimated_mib": round(_mib(plan.estimated_bytes), 2),
                "budget_mib": round(_mib(plan.memory_budget_bytes), 2),
                "headroom_mib": round(
                    _mib(plan.memory_budget_bytes - plan.estimated_bytes), 2
                ),
                "fits": plan.estimated_bytes <= plan.memory_budget_bytes,
            }
        )
    return {
        "telemetry_mib": {
            "total": round(_mib(telemetry.total_bytes), 2),
            "free": round(_mib(telemetry.free_bytes), 2),
            "platform": telemetry.platform,
        },
        "stages": stages,
        "all_fit": all(s["fits"] for s in stages),
    }


def _load_board(
    extract: Path,
    config_path: Path,
    candidate_path: Path | None,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    from plane_opt.geometry_metrics import CopperGrid
    from plane_opt.topology_domain import ROLES
    from plane_opt.topology_holes import apply_mask_runs

    data = json.loads(extract.read_text(encoding="utf-8"))
    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    grid = CopperGrid(data, float(base_config["grid_resolution_mm"]))
    if candidate_path is not None:
        candidate_data = json.loads(candidate_path.read_text(encoding="utf-8"))
        if "mask_runs" in candidate_data:
            candidate = candidate_data
        elif "candidate" in candidate_data:
            candidate = candidate_data["candidate"]
        else:
            candidates = list(candidate_data.get("candidates", []))
            valid = [c for c in candidates if c.get("valid")]
            candidate = valid[0] if valid else candidates[0]
        for role in ROLES:
            apply_mask_runs(grid, role, candidate["mask_runs"][role])
    cases = list(base_config["scenarios"])
    return grid, cases, base_config


def run_physical_memory_matrix(
    extract: Path,
    config_path: Path,
    candidate_path: Path | None,
    *,
    case_limit: int,
    warmups: int,
) -> dict[str, Any]:
    import cupy as cp

    from peec_fastopt.cuda_pypeec import CudaPyPeecExecutor, clear_cuda_caches
    from plane_opt.current_field_backend import solve_current_field_case

    grid, cases, base_config = _load_board(extract, config_path, candidate_path)
    cases = cases[: max(1, case_limit)]

    experiments: list[dict[str, Any]] = []

    variants = [
        {
            "name": "cache_on_reserve10_release",
            "settings": {
                "cache_voxel": True,
                "voxel_cache_entries": 8,
                "memory_reserve_fraction": 0.10,
                "release_pool_after_solve": True,
            },
        },
        {
            "name": "cache_on_reserve10_keep_pool",
            "settings": {
                "cache_voxel": True,
                "voxel_cache_entries": 8,
                "memory_reserve_fraction": 0.10,
                "release_pool_after_solve": False,
            },
        },
        {
            "name": "cache_off_reserve10_release",
            "settings": {
                "cache_voxel": False,
                "voxel_cache_entries": 1,
                "memory_reserve_fraction": 0.10,
                "release_pool_after_solve": True,
            },
        },
        {
            "name": "cache_on_reserve25_release",
            "settings": {
                "cache_voxel": True,
                "voxel_cache_entries": 8,
                "memory_reserve_fraction": 0.25,
                "release_pool_after_solve": True,
            },
        },
        {
            "name": "cache_on_reserve05_release",
            "settings": {
                "cache_voxel": True,
                "voxel_cache_entries": 8,
                "memory_reserve_fraction": 0.05,
                "release_pool_after_solve": True,
            },
        },
    ]

    for variant in variants:
        clear_cuda_caches()
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.cuda.Device(0).synchronize()

        config = copy.deepcopy(base_config)
        cfs = config.setdefault("current_field_solver", {})
        cfs["backend"] = "cuda_peec"
        cuda = cfs.setdefault("cuda_peec", {})
        cuda.update(variant["settings"])
        cuda["fallback_backend"] = None

        snaps: list[dict[str, Any]] = []
        case_rows: list[dict[str, Any]] = []
        snaps.append(_mem_snapshot(cp, "start"))

        # Optional warmups on first case only
        if warmups > 0 and cases:
            for _ in range(warmups):
                solve_current_field_case(grid, cases[0], config)
            snaps.append(_mem_snapshot(cp, "after_warmup"))

        times_ms: list[float] = []
        peaks: list[int] = []
        cache_hits = 0
        for case in cases:
            before = _mem_snapshot(cp, f"before_{case['name']}")
            t0 = time.perf_counter()
            result = solve_current_field_case(grid, case, config)
            elapsed = (time.perf_counter() - t0) * 1000.0
            after = _mem_snapshot(cp, f"after_{case['name']}")
            metrics = result.metrics
            hit = bool(metrics.get("voxel_cache_hit"))
            cache_hits += int(hit)
            peak = int(metrics.get("peak_device_bytes") or 0)
            peaks.append(peak)
            times_ms.append(elapsed)
            case_rows.append(
                {
                    "case": case["name"],
                    "elapsed_ms": round(elapsed, 3),
                    "voxel_cache_hit": hit,
                    "peak_device_mib": round(_mib(peak), 3),
                    "free_before_mib": round(_mib(before["free_bytes"]), 3),
                    "free_after_mib": round(_mib(after["free_bytes"]), 3),
                    "pool_used_after_mib": round(_mib(after["pool_used_bytes"]), 3),
                    "pool_total_after_mib": round(_mib(after["pool_total_bytes"]), 3),
                    "device_used_after_mib": round(_mib(after["used_device_bytes"]), 3),
                    "current_closure_error_a": metrics.get("current_closure_error_a"),
                }
            )

        snaps.append(_mem_snapshot(cp, "end"))
        free_start = snaps[0]["free_bytes"]
        free_end = snaps[-1]["free_bytes"]
        experiments.append(
            {
                "variant": variant["name"],
                "settings": variant["settings"],
                "case_count": len(cases),
                "cache_hits": cache_hits,
                "median_ms": statistics.median(times_ms) if times_ms else None,
                "mean_ms": statistics.mean(times_ms) if times_ms else None,
                "max_reported_peak_mib": round(_mib(max(peaks)), 3) if peaks else None,
                "median_reported_peak_mib": (
                    round(_mib(statistics.median(peaks)), 3) if peaks else None
                ),
                "free_start_mib": round(_mib(free_start), 3),
                "free_end_mib": round(_mib(free_end), 3),
                "free_delta_mib": round(_mib(free_start - free_end), 3),
                "pool_total_end_mib": round(_mib(snaps[-1]["pool_total_bytes"]), 3),
                "pool_used_end_mib": round(_mib(snaps[-1]["pool_used_bytes"]), 3),
                "cases": case_rows,
            }
        )

    # Tight limit OOM recovery probe: force a tiny pool limit via executor
    # and verify explicit error (not silent CPU fallback).
    clear_cuda_caches()
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    oom_probe: dict[str, Any]
    try:
        from peec_fastopt.cuda_pypeec import CudaPeecSolveError, CudaPyPeecExecutor
        from plane_opt.pypeec_current_solver import build_pypeec_inputs

        config = copy.deepcopy(base_config)
        # Use an absurd reserve so usable becomes tiny relative to solve needs.
        executor = CudaPyPeecExecutor(
            {
                "memory_reserve_fraction": 0.999,
                "cache_voxel": False,
            }
        )
        geometry, problem, tolerance, _mapping = build_pypeec_inputs(
            grid, cases[0], config
        )
        try:
            executor.execute(geometry, problem, tolerance)
            oom_probe = {
                "triggered_error": False,
                "note": "solve succeeded despite 99.9% reserve; device may be large",
            }
        except (CudaPeecSolveError, RuntimeError) as exc:
            oom_probe = {
                "triggered_error": True,
                "error_type": type(exc).__name__,
                "message": str(exc)[:400],
            }
    except Exception as exc:  # pragma: no cover - diagnostic path
        oom_probe = {
            "triggered_error": False,
            "setup_error": f"{type(exc).__name__}: {exc}",
        }

    clear_cuda_caches()
    pool.free_all_blocks()

    return {
        "cases_used": [c["name"] for c in cases],
        "experiments": experiments,
        "oom_high_reserve_probe": oom_probe,
    }


def run_delta_batch_memory(grid: int = 256, candidates: int = 2000) -> dict[str, Any]:
    import cupy as cp
    import numpy as np

    from peec_fastopt.cuda_delta import CudaDeltaQuadraticScorer
    from peec_fastopt.delta_peec import SparseDelta

    rng = np.random.default_rng(7)
    base = rng.normal(size=(grid, grid)).astype(np.float32)
    items = []
    for _ in range(candidates):
        n = int(rng.integers(4, 40))
        rows = rng.integers(0, grid, size=n)
        cols = rng.integers(0, grid, size=n)
        vals = rng.normal(size=n)
        items.append(
            SparseDelta.from_changes(zip(rows.tolist(), cols.tolist(), vals.tolist()))
        )

    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    before = _mem_snapshot(cp, "before_delta")
    t0 = time.perf_counter()
    scorer = CudaDeltaQuadraticScorer(base)
    build_ms = (time.perf_counter() - t0) * 1000.0
    mid = _mem_snapshot(cp, "after_base_field")
    t1 = time.perf_counter()
    scores = scorer.energy_many(items)
    score_ms = (time.perf_counter() - t1) * 1000.0
    after = _mem_snapshot(cp, "after_score")
    del scorer
    del scores
    pool.free_all_blocks()
    freed = _mem_snapshot(cp, "after_free")
    return {
        "grid": grid,
        "candidates": candidates,
        "build_ms": round(build_ms, 3),
        "score_ms": round(score_ms, 3),
        "before_mib": {
            "free": round(_mib(before["free_bytes"]), 3),
            "pool_total": round(_mib(before["pool_total_bytes"]), 3),
        },
        "after_base_field_mib": {
            "free": round(_mib(mid["free_bytes"]), 3),
            "pool_total": round(_mib(mid["pool_total_bytes"]), 3),
            "pool_used": round(_mib(mid["pool_used_bytes"]), 3),
        },
        "after_score_mib": {
            "free": round(_mib(after["free_bytes"]), 3),
            "pool_total": round(_mib(after["pool_total_bytes"]), 3),
            "pool_used": round(_mib(after["pool_used_bytes"]), 3),
        },
        "after_free_mib": {
            "free": round(_mib(freed["free_bytes"]), 3),
            "pool_total": round(_mib(freed["pool_total_bytes"]), 3),
        },
        "released_to_driver": freed["pool_total_bytes"] < after["pool_total_bytes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extract",
        type=Path,
        default=Path(
            "../plane_opt/topology_variants/pgnd_board_left_expanded_0p4/analysis/extract.json"
        ),
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path(
            "../plane_opt/topology_variants/pgnd_board_left_expanded_0p4/analysis/search.json"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("../plane_opt/topology_refinement_config.json"),
    )
    parser.add_argument("--case-limit", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--skip-physical", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/memory-ops.json"))
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema_version": 1,
        "controller_live": run_controller_live(),
        "delta_batch": run_delta_batch_memory(),
    }
    if not args.skip_physical:
        report["physical"] = run_physical_memory_matrix(
            args.extract,
            args.config,
            args.candidate,
            case_limit=args.case_limit,
            warmups=args.warmups,
        )
    # Summary recommendations derived from numbers
    recs: list[str] = []
    if "physical" in report:
        by_name = {e["variant"]: e for e in report["physical"]["experiments"]}
        release = by_name.get("cache_on_reserve10_release")
        keep = by_name.get("cache_on_reserve10_keep_pool")
        cache_off = by_name.get("cache_off_reserve10_release")
        if release and keep:
            free_gain = release["free_end_mib"] - keep["free_end_mib"]
            recs.append(
                f"release_pool_after_solve free-end delta vs keep: {free_gain:+.1f} MiB "
                f"(release free_end={release['free_end_mib']:.1f}, "
                f"keep free_end={keep['free_end_mib']:.1f})."
            )
            if release["median_ms"] and keep["median_ms"]:
                ratio = release["median_ms"] / max(keep["median_ms"], 1e-9)
                recs.append(
                    f"release_pool_after_solve median time ratio vs keep: {ratio:.3f}x "
                    f"({release['median_ms']:.1f} ms / {keep['median_ms']:.1f} ms)."
                )
        if release and cache_off:
            if cache_off["mean_ms"] and release["mean_ms"]:
                ratio = cache_off["mean_ms"] / max(release["mean_ms"], 1e-9)
                recs.append(
                    f"Voxel cache off is {ratio:.2f}x mean wall time vs cache on "
                    f"(hits={release['cache_hits']}/{release['case_count']})."
                )
            recs.append(
                f"Host voxel cache pool_total_end: cache-on "
                f"{release['pool_total_end_mib']:.1f} MiB vs cache-off "
                f"{cache_off['pool_total_end_mib']:.1f} MiB."
            )
        growth = [e["free_delta_mib"] for e in report["physical"]["experiments"]]
        if growth and max(growth) > 100:
            recs.append(
                "Free VRAM dropped by >100 MiB across sequential solves in at least "
                "one variant; check for unreclaimed solver temporaries."
            )
        elif growth and max(abs(g) for g in growth) < 50:
            recs.append(
                "Sequential free-VRAM drift stayed within ~50 MiB across variants; "
                "no large leak observed at this case count."
            )
    report["recommendations"] = recs

    text = json.dumps(report, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
