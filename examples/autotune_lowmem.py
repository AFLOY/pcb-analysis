"""Sweep low-memory near/far parameters against exact interaction energies."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.stats import rankdata

from peec_fastopt.lowmem_peec import approximate_energy, exact_energy

# benchmark.py is co-located in examples/; add examples/ to path if needed.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import make_base, make_candidates  # noqa: E402


def candidate_points(base: np.ndarray, delta) -> tuple[np.ndarray, np.ndarray]:
    vector = base + delta.dense(base.shape)
    points = np.argwhere(vector != 0.0).astype(np.float64)
    values = vector[vector != 0.0].astype(np.float64)
    return points, values


def ranking_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    ref_rank = rankdata(reference, method="average")
    est_rank = rankdata(estimate, method="average")
    correlation = float(np.corrcoef(ref_rank, est_rank)[0, 1])
    top_count = max(1, len(reference) // 10)
    ref_top = set(np.argpartition(reference, top_count - 1)[:top_count])
    est_top = set(np.argpartition(estimate, top_count - 1)[:top_count])
    shortlist_count = max(top_count, len(reference) // 5)
    est_shortlist = set(
        np.argpartition(estimate, shortlist_count - 1)[:shortlist_count]
    )
    return {
        "spearman": correlation,
        "top_10pct_overlap": len(ref_top & est_top) / top_count,
        "true_top_10pct_recall_in_20pct_shortlist": (
            len(ref_top & est_shortlist) / top_count
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--blocks", default="2,4,8,16")
    parser.add_argument("--radii", default="0,2,4,8,16")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    base = make_base(args.grid)
    candidates = make_candidates(args.grid, args.candidates, args.seed)
    geometries = [candidate_points(base, delta) for delta in candidates]

    start = time.perf_counter()
    reference = np.asarray([exact_energy(*geometry) for geometry in geometries])
    reference_seconds = time.perf_counter() - start
    results = []

    for block_size in [int(v) for v in args.blocks.split(",")]:
        for near_radius in [float(v) for v in args.radii.split(",")]:
            for order in (0, 1):
                for dtype in (np.float32, np.float64):
                    start = time.perf_counter()
                    estimate = np.asarray(
                        [
                            approximate_energy(
                                *geometry,
                                block_size=block_size,
                                near_radius=near_radius,
                                order=order,
                                storage_dtype=dtype,
                            )
                            for geometry in geometries
                        ]
                    )
                    seconds = time.perf_counter() - start
                    relative = np.abs(estimate - reference) / np.maximum(
                        np.abs(reference), 1e-30
                    )
                    metrics = ranking_metrics(reference, estimate)
                    results.append(
                        {
                            "block": block_size,
                            "near_radius": near_radius,
                            "order": order,
                            "storage": np.dtype(dtype).name,
                            "median_relative_error": float(np.median(relative)),
                            "max_relative_error": float(np.max(relative)),
                            "seconds": seconds,
                            **metrics,
                        }
                    )

    feasible = [
        item
        for item in results
        if item["max_relative_error"] <= 1e-3
        and item["top_10pct_overlap"] >= 0.99
    ]
    feasible.sort(key=lambda item: (item["near_radius"], item["block"], item["seconds"]))
    payload = {
        "grid": args.grid,
        "candidates": args.candidates,
        "reference_seconds": reference_seconds,
        "recommended_under_0.1pct": feasible[:10],
    }
    if args.summary:
        payload["dipole_float32"] = [
            item
            for item in results
            if item["order"] == 1 and item["storage"] == "float32"
        ]
    else:
        payload["all_results"] = results
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
