"""Several excitations of one mesh solved together (solve_sheet_cases) against one by one.

The fixture is the plane of ``sheet_preconditioner_benchmark.py``.  Each
excitation feeds the left edge over a different span of rows and takes the
current out of the whole right edge on the other layer, so all of them drive
the same conductor and share one system.  For each pitch and excitation count
the ``block`` solve runs one excitation at a time with ``solve_sheet_case``
and all together with ``solve_sheet_cases``; the solutions must be the same to
the bit.  Writes a JSON with ``environment`` and ``decision``.

    .venv/bin/python experiments/sheet_cases_benchmark.py --pitches-mm 0.5,0.25,0.2 --counts 2,4
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


def excitations(mesh, count: int):
    from electrical.sheet_peec.sheet_peec import Terminal

    rows, cols = mesh.shape
    sets = []
    for k in range(count):
        low, high = (k * rows) // (2 * count), rows - (k * rows) // (2 * count)
        sets.append([
            Terminal("in", 0, tuple((r, 0) for r in range(low, high)), 10.0),
            Terminal("out", 1, tuple((r, cols - 1) for r in range(rows)), -10.0),
        ])
    return sets


def case(pitch_mm: float, count: int, frequency_hz: float) -> dict:
    from electrical.sheet_peec.sheet_peec import solve_sheet_case, solve_sheet_cases

    mesh, operator, _ = build(pitch_mm)
    sets = excitations(mesh, count)
    options = dict(frequency_hz=frequency_hz, preconditioner="block", tolerance=1e-9, restart=120, max_iterations=400)
    started = time.perf_counter()
    alone = [solve_sheet_case(mesh, operator, terminals, **options) for terminals in sets]
    one_by_one = time.perf_counter() - started
    started = time.perf_counter()
    together = solve_sheet_cases(mesh, operator, sets, **options)
    lockstep = time.perf_counter() - started
    identical = all(
        a.branch_current.tobytes() == b.branch_current.tobytes() and a.node_voltage.tobytes() == b.node_voltage.tobytes()
        for a, b in zip(alone, together)
    )
    record = {
        "pitch_mm": pitch_mm, "excitations": count, "branches": int(mesh.branch_count),
        "one_by_one_s": round(one_by_one, 3), "together_s": round(lockstep, 3),
        "speedup": round(one_by_one / lockstep, 3), "identical_to_the_bit": identical,
        "iterations": [s.iterations for s in alone], "converged": all(s.converged for s in together),
    }
    print(json.dumps(record), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitches-mm", default="0.5,0.25,0.2")
    parser.add_argument("--counts", default="2,4")
    parser.add_argument("--frequency-hz", type=float, default=800e3)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "benchmark-results" / "sheet_cases_benchmark.json")
    args = parser.parse_args()

    from electrical.threads import available_threads, thread_budget

    cases = [
        case(float(pitch), int(count), args.frequency_hz)
        for pitch in args.pitches_mm.split(",") if pitch
        for count in args.counts.split(",") if count
    ]
    adopted = all(c["identical_to_the_bit"] and c["converged"] and c["together_s"] <= c["one_by_one_s"] for c in cases)
    report = {
        "benchmark": "sheet_cases_benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "cpu": _cpu_model(), "cpu_count": os.cpu_count(), "physical_cores": available_threads(),
            "thread_budget": thread_budget(), "platform": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "scipy": __import__("scipy").__version__,
        },
        "settings": {"frequency_hz": args.frequency_hz, "preconditioner": "block", "tolerance": 1e-9, "restart": 120,
                     "max_iterations": 400},
        "cases": cases,
        "decision": {
            "adopted": adopted,
            "criteria": "every pitch and count: solutions identical to the bit, converged, and together no slower than one by one",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("decision:", json.dumps(report["decision"]))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
