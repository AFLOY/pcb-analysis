"""The block preconditioner's near block in strips, and the batched flux FFT, on the preconditioner fixture.

The fixture is the one of ``sheet_preconditioner_benchmark.py``: a two-layer
plane of the power_module footprint with a via field and two full-width
terminals.  For each pitch the ``block`` solve runs with ``near_tiles`` from
``--tiles`` at each thread budget of ``--threads``; every solution is compared
with ``near_tiles=1`` at the same budget.  The flux apply of the operator is
timed against the per-layer NumPy FFT loop it replaces, and the two are
compared.  Writes a JSON with ``environment`` and ``decision``.

    .venv/bin/python experiments/sheet_near_tiles_benchmark.py --pitches-mm 0.5,0.25,0.2
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dc_native_benchmark import _cpu_model  # noqa: E402
from sheet_preconditioner_benchmark import build  # noqa: E402

SOLUTION_TOLERANCE = 1e-6   # strips must reach the whole block's solution to this
FLUX_TOLERANCE = 1e-12      # batched FFT against the per-layer loop, relative to the largest flux


def _blas_threads() -> list[int]:
    from threadpoolctl import threadpool_info

    return sorted({pool["num_threads"] for pool in threadpool_info() if pool["user_api"] == "blas"})


def _per_layer_flux(operator, currents_x, currents_y, currents_z):
    """The flux apply before batching: one NumPy FFT per layer and per target."""
    rows, cols = operator.shape
    layers = len(operator.stackup)

    def family(currents, kernel, count):
        spectra = [np.fft.rfft2(currents[k], s=operator.padded) for k in range(count)]
        out = np.zeros((count, rows, cols))
        for target in range(count):
            accumulated = np.zeros_like(spectra[0])
            for source in range(count):
                accumulated += kernel(target, source) * spectra[source]
            out[target] = np.fft.irfft2(accumulated, s=operator.padded)[:rows, :cols]
        return out

    flux_x = family(currents_x, lambda t, s: operator._kernel("x", t, s), layers)
    flux_y = family(currents_y, lambda t, s: operator._kernel("y", t, s), layers)
    flux_z = family(currents_z, operator._kernel_z, len(operator.vertical_levels))
    return flux_x, flux_y, flux_z


def flux_case(pitch_mm: float, threads: list[int], repeats: int) -> dict:
    from electrical.threads import thread_budget_scope

    mesh, operator, _ = build(pitch_mm)
    rows, cols = mesh.shape
    rng = np.random.default_rng(0)
    currents_x = rng.standard_normal((len(mesh.stackup), rows, cols))
    currents_y = rng.standard_normal((len(mesh.stackup), rows, cols))
    currents_z = rng.standard_normal((len(operator.vertical_levels), rows, cols))
    started = time.perf_counter()
    for _ in range(repeats):
        reference = _per_layer_flux(operator, currents_x, currents_y, currents_z)
    record = {"pitch_mm": pitch_mm, "per_layer_numpy_s": round((time.perf_counter() - started) / repeats, 4)}
    scale = max(float(np.abs(part).max()) for part in reference)
    for budget in threads:
        with thread_budget_scope(budget):
            started = time.perf_counter()
            for _ in range(repeats):
                batched = operator.apply(currents_x, currents_y, currents_z)
            record[f"batched_threads_{budget}_s"] = round((time.perf_counter() - started) / repeats, 4)
        record[f"batched_threads_{budget}_max_relative_difference"] = max(
            float(np.abs(a - b).max()) / scale for a, b in zip(batched, reference)
        )
    return record


def solve_case(pitch_mm: float, frequency_hz: float, tiles: list[int], threads: list[int]) -> dict:
    from electrical.sheet_peec.sheet_peec import solve_sheet_case
    from electrical.threads import thread_budget_scope

    mesh, operator, terminals = build(pitch_mm)
    case = {"pitch_mm": pitch_mm, "branches": int(mesh.branch_count), "nodes": int(mesh.node_count), "runs": []}
    for budget in threads:
        reference = None
        for count in tiles:
            with thread_budget_scope(budget):
                started = time.perf_counter()
                solution = solve_sheet_case(
                    mesh, operator, terminals, frequency_hz=frequency_hz, preconditioner="block",
                    tolerance=1e-9, restart=120, max_iterations=400, near_tiles=count,
                )
                wall = time.perf_counter() - started
            if reference is None:
                reference = solution
            difference = float(
                np.abs(solution.node_voltage - reference.node_voltage).max()
                / max(float(np.abs(reference.node_voltage).max()), 1e-30)
            )
            run = {
                "threads": budget, "near_tiles": count, "wall_s": round(wall, 3), "iterations": int(solution.iterations),
                "converged": bool(solution.converged), "residual": float(solution.residual),
                "max_relative_difference_vs_tiles_1": difference,
            }
            case["runs"].append(run)
            print(f"{pitch_mm:5.2f} mm threads {budget} tiles {count:2d}: {wall:7.2f} s it {solution.iterations:4d}"
                  f" conv {solution.converged!s:5s} diff {difference:.1e}", flush=True)
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitches-mm", default="0.5,0.25,0.2")
    parser.add_argument("--frequency-hz", type=float, default=800e3)
    parser.add_argument("--tiles", default="1,2,4,8,16")
    parser.add_argument("--threads", default=None, help="budgets to run at; default 1 and the physical cores")
    parser.add_argument("--flux-repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "benchmark-results" / "sheet_near_tiles_benchmark.json")
    args = parser.parse_args()

    from electrical.threads import available_threads

    pitches = [float(p) for p in args.pitches_mm.split(",") if p]
    tiles = [int(t) for t in args.tiles.split(",") if t]
    if tiles[0] != 1:
        raise SystemExit("--tiles must start with 1, the reference")
    threads = [int(t) for t in args.threads.split(",")] if args.threads else sorted({1, available_threads()})

    flux = []
    for pitch in pitches:
        flux.append(flux_case(pitch, threads, args.flux_repeats))
        print("flux", json.dumps(flux[-1]), flush=True)
    cases = [solve_case(pitch, args.frequency_hz, tiles, threads) for pitch in pitches]

    # A strip count is adopted as the default when, at every pitch and budget,
    # it converges to the whole block's solution and is no slower than it,
    # and it is the fastest such count on the largest problem at the largest budget.
    candidates = []
    for count in tiles[1:]:
        ok = True
        for case in cases:
            for budget in threads:
                runs = {r["near_tiles"]: r for r in case["runs"] if r["threads"] == budget}
                run, whole = runs[count], runs[1]
                ok &= run["converged"] and run["max_relative_difference_vs_tiles_1"] <= SOLUTION_TOLERANCE
                ok &= run["wall_s"] <= whole["wall_s"]
        if ok:
            candidates.append(count)
    largest = {r["near_tiles"]: r["wall_s"] for r in cases[-1]["runs"] if r["threads"] == max(threads)}
    chosen = min(candidates, key=lambda count: largest[count]) if candidates else 1
    flux_ok = all(
        record[f"batched_threads_{budget}_max_relative_difference"] <= FLUX_TOLERANCE for record in flux for budget in threads
    )
    report = {
        "benchmark": "sheet_near_tiles_benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "cpu": _cpu_model(), "cpu_count": os.cpu_count(), "physical_cores": available_threads(),
            "platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
            "scipy": __import__("scipy").__version__, "blas_threads": _blas_threads(),
        },
        "settings": {"frequency_hz": args.frequency_hz, "pitches_mm": pitches, "tiles": tiles, "threads": threads,
                     "tolerance": 1e-9, "restart": 120, "max_iterations": 400, "flux_repeats": args.flux_repeats},
        "flux_apply": flux,
        "cases": cases,
        "decision": {
            "default_near_tiles": chosen,
            "near_tiles_candidates": candidates,
            "near_tiles_criteria": (
                f"at every pitch and budget: converged, within {SOLUTION_TOLERANCE:g} of near_tiles=1, no slower than it; "
                "then the fastest on the largest pitch at the largest budget"
            ),
            "batched_flux_adopted": flux_ok,
            "batched_flux_criteria": f"batched FFT within {FLUX_TOLERANCE:g} of the per-layer NumPy loop at every budget",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("decision:", json.dumps(report["decision"]))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
