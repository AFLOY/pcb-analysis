"""Near, block and diagonal preconditioners of the sheet-PEEC solve: iterations, time, memory.

A two-layer plane of the power_module footprint (33 x 30 mm, 35 um copper,
1.6 mm apart) with a via field and two full-width terminals, solved at
``--frequency-hz`` on each pitch of ``--pitches-mm``.  Every variant runs in
its own subprocess so its peak RSS is its own; the solutions are compared
with the near variant where it fits (``--near-up-to-mm``), else with block.
Writes a JSON with ``environment`` and ``decision``.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/sheet_preconditioner_benchmark.py \\
        --pitches-mm 0.5,0.25,0.2 --near-up-to-mm 0.25
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dc_native_benchmark import _cpu_model  # noqa: E402

FOOTPRINT_MM = (33.0, 30.3)
WALL_LIMIT_RATIO = 1.5          # block may take at most this many times the near variant's wall time
MEMORY_GAIN_REQUIRED = 4.0      # and must need at most 1/4 of the near variant's peak RSS
SOLUTION_TOLERANCE = 1e-6
# ``auto`` picks near up to the size whose near peak RSS, at the largest
# bytes per saddle-point unknown measured, stays within this budget.
NEAR_MEMORY_BUDGET_MB = 4096


def build(pitch_mm: float):
    from electrical.sheet_peec.sheet_operator import SheetInductanceOperator, SheetLayer, SheetStackup
    from electrical.sheet_peec.sheet_peec import SheetMesh, Terminal, ViaBranch

    rows, cols = int(round(FOOTPRINT_MM[1] / pitch_mm)), int(round(FOOTPRINT_MM[0] / pitch_mm))
    stackup = SheetStackup(
        (SheetLayer("F.Cu", 0.0, 35e-6, 1.724e-8), SheetLayer("B.Cu", -1.6e-3, 35e-6, 1.724e-8))
    )
    step = max(1, rows // 8)
    vias = tuple(
        ViaBranch(r, c, 0, 1, resistance_ohm=1e-3) for r in range(step, rows, step) for c in range(cols // 2, cols, step)
    )
    mesh = SheetMesh((rows, cols), pitch_mm * 1e-3, stackup, np.ones((2, rows, cols), dtype=bool), vias=vias)
    operator = SheetInductanceOperator((rows, cols), pitch_mm * 1e-3, stackup, vertical_levels=mesh.vertical_levels)
    terminals = (
        Terminal("in", 0, tuple((r, 0) for r in range(rows)), 10.0),
        Terminal("out", 1, tuple((r, cols - 1) for r in range(rows)), -10.0),
    )
    return mesh, operator, terminals


def run_variant(variant: str, pitch_mm: float, frequency_hz: float, out: Path) -> None:
    from electrical.sheet_peec.sheet_peec import solve_sheet_case

    mesh, operator, terminals = build(pitch_mm)
    started = time.perf_counter()
    solution = solve_sheet_case(
        mesh, operator, terminals, frequency_hz=frequency_hz, preconditioner=variant, tolerance=1e-9, restart=120, max_iterations=400
    )
    wall = time.perf_counter() - started
    np.savez(out, node_voltage=solution.node_voltage, branch_current=solution.branch_current)
    print(json.dumps({
        "variant": variant,
        "pitch_mm": pitch_mm,
        "cells": int(mesh.shape[0] * mesh.shape[1] * len(mesh.stackup)),
        "branches": int(mesh.branch_count),
        "nodes": int(mesh.node_count),
        "iterations": int(solution.iterations),
        "converged": bool(solution.converged),
        "residual": float(solution.residual),
        "wall_s": round(wall, 2),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitches-mm", default="0.5,0.25,0.2")
    parser.add_argument("--frequency-hz", type=float, default=800e3)
    parser.add_argument("--near-up-to-mm", type=float, default=0.25, help="run the near (saddle LU) variant only at pitches at or above this")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "benchmark-results" / "sheet_preconditioner_benchmark.json")
    parser.add_argument("--variants", default="diagonal,block,near", help="which variants to run (near still only down to --near-up-to-mm)")
    parser.add_argument("--variant", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pitch", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.variant:
        run_variant(args.variant, args.pitch, args.frequency_hz, args.out)
        return

    cases = []
    scratch = args.output.parent / "sheet_preconditioner_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    for pitch in [float(p) for p in args.pitches_mm.split(",") if p]:
        chosen = [v for v in args.variants.split(",") if v]
        variants = [v for v in ("diagonal", "block") if v in chosen] + (
            ["near"] if "near" in chosen and pitch >= args.near_up_to_mm else [])
        results: dict[str, dict] = {}
        for variant in variants:
            out = scratch / f"{variant}_{pitch:g}.npz"
            done = subprocess.run(
                [sys.executable, __file__, "--variant", variant, "--pitch", str(pitch), "--frequency-hz", str(args.frequency_hz), "--out", str(out)],
                capture_output=True, text=True, timeout=3600,
            )
            if done.returncode != 0:
                results[variant] = {"variant": variant, "failed": (done.stderr.strip().splitlines() or ["killed"])[-1][:200]}
                continue
            results[variant] = json.loads(done.stdout.strip().splitlines()[-1])
        reference = "near" if "near" in results and "failed" not in results["near"] else "block"
        ref = np.load(scratch / f"{reference}_{pitch:g}.npz") if "failed" not in results.get(reference, {"failed": 1}) else None
        for variant, record in results.items():
            if "failed" in record or ref is None:
                continue
            sol = np.load(scratch / f"{variant}_{pitch:g}.npz")
            record["max_relative_difference_vs_" + reference] = float(
                np.abs(sol["node_voltage"] - ref["node_voltage"]).max() / max(np.abs(ref["node_voltage"]).max(), 1e-30)
            )
        cases.append({"pitch_mm": pitch, "results": results})
        for variant, record in results.items():
            if "failed" in record:
                print(f"{pitch:5.2f} mm {variant:9s} FAILED {record['failed']}")
            else:
                print(f"{pitch:5.2f} mm {variant:9s} it {record['iterations']:4d} conv {record['converged']!s:5s} {record['wall_s']:7.1f} s  peak {record['peak_rss_mb']:6d} MB  diff {record.get('max_relative_difference_vs_' + reference, float('nan')):.1e}")

    verdicts = []
    for case in cases:
        near, block = case["results"].get("near"), case["results"].get("block")
        if near and block and "failed" not in near and "failed" not in block:
            verdicts.append(
                block["converged"]
                and block["wall_s"] <= WALL_LIMIT_RATIO * near["wall_s"]
                and near["peak_rss_mb"] >= MEMORY_GAIN_REQUIRED * block["peak_rss_mb"]
                and block.get("max_relative_difference_vs_near", 1.0) <= SOLUTION_TOLERANCE
            )
    report = {
        "benchmark": "sheet_preconditioner_benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "cpu": _cpu_model(), "cpu_count": os.cpu_count(), "platform": platform.platform(),
            "python": platform.python_version(), "numpy": np.__version__, "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "settings": {"footprint_mm": FOOTPRINT_MM, "frequency_hz": args.frequency_hz, "pitches_mm": args.pitches_mm, "near_up_to_mm": args.near_up_to_mm, "variants": args.variants,
                     "tolerance": 1e-9, "restart": 120, "max_iterations": 400},
        "cases": cases,
        "decision": {
            "adopted": bool(verdicts) and all(verdicts),
            "criteria": (
                f"where the near variant fits: block converges, within {WALL_LIMIT_RATIO:g}x its wall time, "
                f"at most 1/{MEMORY_GAIN_REQUIRED:g} of its peak RSS, same solution to {SOLUTION_TOLERANCE:g}"
            ),
            "criteria_note": (
                "the first run of this benchmark judged block by its GMRES iteration count (3x near); that criterion was "
                "replaced by wall time before adoption, because an iteration of the near variant includes a saddle-point "
                "LU solve and an iteration of block two small ones, so iterations of the two are not the same unit; "
                "the iteration counts are recorded above unchanged"
            ),
            "verdicts_per_pitch_with_near": verdicts,
            "auto_threshold": _auto_threshold(cases),
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("decision:", json.dumps(report["decision"]))
    print("wrote", args.output)


def _auto_threshold(cases: list[dict]) -> dict:
    """Where ``auto`` should switch from near to block, from the near rows' memory and both variants' time."""
    near_rows = [c["results"]["near"] for c in cases if "near" in c["results"] and "failed" not in c["results"]["near"]]
    pairs = [(c["results"]["near"], c["results"]["block"]) for c in cases
             if all(v in c["results"] and "failed" not in c["results"][v] for v in ("near", "block"))]
    if not near_rows:
        return {"near_up_to_unknowns": None, "note": "no near row ran"}
    per_unknown = [row["peak_rss_mb"] / (row["branches"] + row["nodes"]) for row in near_rows]
    worst = max(per_unknown)
    return {
        "near_memory_budget_mb": NEAR_MEMORY_BUDGET_MB,
        "near_peak_mb_per_unknown": per_unknown,
        "near_peak_mb_per_unknown_max": worst,
        "near_up_to_unknowns": int(NEAR_MEMORY_BUDGET_MB / worst),
        "near_faster_than_block_at_every_size": all(n["wall_s"] < b["wall_s"] for n, b in pairs),
        "near_over_block_wall": [n["wall_s"] / b["wall_s"] for n, b in pairs],
        "criteria": ("near is chosen up to the saddle-point size at which its peak RSS, at the largest MB per unknown "
                     f"measured, reaches {NEAR_MEMORY_BUDGET_MB} MB, when it is faster than block at every size measured"),
    }


if __name__ == "__main__":
    main()
