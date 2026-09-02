"""Benchmark full FFT rescoring against exact sparse-delta rescoring."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.stats import rankdata

from electrical.dice_peec.delta_peec import (
    DeltaQuadraticScorer,
    FFTInteraction2D,
    SparseDelta,
)


def make_base(grid: int) -> np.ndarray:
    base = np.zeros((grid, grid), dtype=np.float64)
    base[grid // 2, grid // 8 : 7 * grid // 8] = 1.0
    return base


def make_candidates(grid: int, count: int, seed: int) -> list[SparseDelta]:
    rng = np.random.default_rng(seed)
    row = grid // 2
    left, right = grid // 8, 7 * grid // 8
    candidates: list[SparseDelta] = []
    for _ in range(count):
        length = int(rng.integers(4, max(5, grid // 16)))
        start = int(rng.integers(left + 1, right - length - 1))
        offset = int(rng.choice([-3, -2, -1, 1, 2, 3]))
        detour_row = row + offset
        changes: list[tuple[int, int, float]] = []
        # Remove a straight segment and add a rectangular detour.
        for col in range(start, start + length):
            changes.append((row, col, -1.0))
            changes.append((detour_row, col, 1.0))
        step = 1 if offset > 0 else -1
        for rr in range(row + step, detour_row, step):
            changes.append((rr, start, 1.0))
            changes.append((rr, start + length - 1, 1.0))
        candidates.append(SparseDelta.from_changes(changes))
    return candidates


def timed_scores(fn, candidates: list[SparseDelta]) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    values = np.asarray([fn(candidate) for candidate in candidates])
    return values, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--validate", type=int, default=100)
    args = parser.parse_args()

    operator = FFTInteraction2D((args.grid, args.grid))
    base = make_base(args.grid)
    candidates = make_candidates(args.grid, args.candidates, args.seed)
    scorer = DeltaQuadraticScorer(operator, base)

    delta_scores, delta_seconds = timed_scores(scorer.energy, candidates)
    validate_count = min(args.validate, args.candidates)
    full_scores, full_seconds = timed_scores(
        scorer.full_energy, candidates[:validate_count]
    )
    estimated_full_seconds = full_seconds * args.candidates / validate_count
    error = np.abs(full_scores - delta_scores[:validate_count])
    delta_ranks = rankdata(delta_scores[:validate_count], method="average")
    full_ranks = rankdata(full_scores, method="average")
    rank_correlation = float(np.corrcoef(delta_ranks, full_ranks)[0, 1])
    top_count = max(1, validate_count // 10)
    top_delta = set(np.argpartition(delta_scores[:validate_count], top_count - 1)[:top_count])
    top_full = set(np.argpartition(full_scores, top_count - 1)[:top_count])

    report = {
        "grid": [args.grid, args.grid],
        "candidates": args.candidates,
        "changed_cells_mean": float(np.mean([c.size for c in candidates])),
        "delta_seconds": delta_seconds,
        "full_fft_seconds_measured": full_seconds,
        "full_fft_candidates_measured": validate_count,
        "full_fft_seconds_estimated_all": estimated_full_seconds,
        "estimated_speedup": estimated_full_seconds / delta_seconds,
        "max_absolute_error": float(error.max(initial=0.0)),
        "max_relative_error": float(
            np.max(error / np.maximum(np.abs(full_scores), 1e-30), initial=0.0)
        ),
        "spearman_rank_correlation": rank_correlation,
        "top_10pct_overlap": len(top_delta & top_full) / top_count,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
