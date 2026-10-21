"""Memory an excitation adds to a lockstep group, measured against lockstep_column_bytes.

solve_sheet_cases cuts a group of excitations into lockstep columns that fit
in LOCKSTEP_MEMORY_FRACTION of the memory available, counting each column by
``lockstep_column_bytes``.  This runs a group of 2 and of 4 excitations of
one mesh, each in its own subprocess with the cut disabled, through one full
restart cycle (tolerance 1e-14, one cycle, so every basis vector is
written), and takes the peak-RSS difference over the two extra columns as
the measured cost of a column.  The estimate must not be below it (a cut
sized by it must fit) and not more than ESTIMATE_MAX_RATIO times above it
(or the cut throws away parallelism it could have had).  Writes a JSON with
``environment`` and ``decision``.

    .venv/bin/python experiments/sheet_lockstep_memory_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dc_native_benchmark import _cpu_model  # noqa: E402
from sheet_preconditioner_benchmark import FOOTPRINT_MM, build as build_plane  # noqa: E402

RESTART = 120
ESTIMATE_MAX_RATIO = 2.5


def build_four_layer(pitch_mm: float):
    from electrical.sheet_peec.sheet_operator import SheetInductanceOperator, SheetLayer, SheetStackup
    from electrical.sheet_peec.sheet_peec import SheetMesh, Terminal, ViaBranch

    rows, cols = int(round(FOOTPRINT_MM[1] / pitch_mm)), int(round(FOOTPRINT_MM[0] / pitch_mm))
    stackup = SheetStackup(tuple(
        SheetLayer(name, z, 35e-6, 1.724e-8)
        for name, z in (("F.Cu", 0.0), ("In1.Cu", -0.2e-3), ("In2.Cu", -1.4e-3), ("B.Cu", -1.6e-3))
    ))
    occupancy = np.ones((4, rows, cols), dtype=bool)
    occupancy[0, rows // 3 : rows // 2, cols // 4 : 3 * cols // 4] = False
    occupancy[1, rows // 2 : 2 * rows // 3, : cols // 2] = False
    occupancy[3, rows // 5 : 4 * rows // 5, cols // 3 : cols // 3 + 3] = False
    step = max(2, rows // 10)
    pairs = ((3, 0), (1, 0), (3, 2), (2, 1))
    vias = []
    for k, row in enumerate(range(step, rows - 1, step)):
        for m, col in enumerate(range(step, cols - 1, step)):
            upper, lower = pairs[(k + m) % len(pairs)]
            occupancy[lower : upper + 1, row, col] = True
            vias.append(ViaBranch(row, col, upper, lower, resistance_ohm=1e-3))
    mesh = SheetMesh((rows, cols), pitch_mm * 1e-3, stackup, occupancy, vias=tuple(vias))
    operator = SheetInductanceOperator((rows, cols), pitch_mm * 1e-3, stackup, vertical_levels=mesh.vertical_levels)
    terminals = (
        Terminal("in", 0, tuple((r, 0) for r in range(rows)), 10.0),
        Terminal("out", 3, tuple((r, cols - 1) for r in range(rows)), -10.0),
    )
    return mesh, operator, terminals


FIXTURES = {"plane": build_plane, "four_layer": build_four_layer}


def excitations(mesh, terminals, count: int):
    from electrical.sheet_peec.sheet_peec import Terminal

    rows, cols = mesh.shape
    first, last = terminals
    sets = []
    for k in range(count):
        low, high = (k * rows) // (2 * count), rows - (k * rows) // (2 * count)
        sets.append([
            Terminal("in", first.layer, tuple((r, 0) for r in range(low, high)), 10.0),
            Terminal("out", last.layer, tuple((r, cols - 1) for r in range(rows)), -10.0),
        ])
    return sets


def run_group(fixture: str, pitch_mm: float, count: int, frequency_hz: float) -> None:
    import electrical.sheet_peec.sheet_peec as sheet

    mesh, operator, terminals = FIXTURES[fixture](pitch_mm)
    sets = excitations(mesh, terminals, count)
    sheet._available_memory_bytes = lambda: None  # the whole group in one lockstep
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    solutions = sheet.solve_sheet_cases(
        mesh, operator, sets, frequency_hz=frequency_hz, preconditioner="block",
        tolerance=1e-14, restart=RESTART, max_iterations=1,
    )
    print(json.dumps({
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "before_solve_rss_bytes": before * 1024,
        "lockstep_columns": [s.lockstep_columns for s in solutions],
        "size": int(mesh.branch_count + mesh.node_count),
    }))


def case(fixture: str, pitch_mm: float, frequency_hz: float) -> dict:
    from electrical.sheet_peec.sheet_peec import lockstep_column_bytes

    peaks = {}
    for count in (2, 4):
        done = subprocess.run(
            [sys.executable, __file__, "--group", fixture, str(pitch_mm), str(count), str(frequency_hz)],
            capture_output=True, text=True, timeout=7200, check=True,
        )
        peaks[count] = json.loads(done.stdout.strip().splitlines()[-1])
    mesh, operator, _ = FIXTURES[fixture](pitch_mm)
    size = int(mesh.branch_count + mesh.node_count)
    estimate = lockstep_column_bytes(size, RESTART, operator)
    measured = (peaks[4]["peak_rss_bytes"] - peaks[2]["peak_rss_bytes"]) / 2
    record = {
        "fixture": fixture, "pitch_mm": pitch_mm, "frequency_hz": frequency_hz,
        "unknowns": size, "grid": list(operator.shape),
        "layers": len(operator.stackup), "vertical_levels": len(operator.vertical_levels),
        "peak_rss_bytes": {str(k): v["peak_rss_bytes"] for k, v in peaks.items()},
        "measured_column_bytes": int(measured),
        "estimated_column_bytes": int(estimate),
        "estimate_over_measured": round(estimate / measured, 3) if measured > 0 else None,
    }
    print(json.dumps(record), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="plane:0.25,plane:0.2,four_layer:0.25,four_layer:0.2")
    parser.add_argument("--frequency-hz", type=float, default=2.16e5)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "benchmark-results" / "sheet_lockstep_memory_benchmark.json")
    parser.add_argument("--group", nargs=4, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.group:
        fixture, pitch, count, frequency = args.group
        run_group(fixture, float(pitch), int(count), float(frequency))
        return

    from electrical.sheet_peec.sheet_peec import (
        LOCKSTEP_FFT_BYTES_PER_CELL,
        LOCKSTEP_MEMORY_FRACTION,
        LOCKSTEP_WORK_VECTORS,
    )
    from electrical.threads import available_threads, thread_budget

    cases = []
    for item in args.cases.split(","):
        fixture, pitch = item.split(":")
        cases.append(case(fixture, float(pitch), args.frequency_hz))
    ratios = [c["estimate_over_measured"] for c in cases]
    adopted = all(r is not None and 1.0 <= r <= ESTIMATE_MAX_RATIO for r in ratios)
    report = {
        "benchmark": "sheet_lockstep_memory_benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "cpu": _cpu_model(), "cpu_count": os.cpu_count(), "physical_cores": available_threads(),
            "thread_budget": thread_budget(), "platform": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "scipy": __import__("scipy").__version__,
        },
        "settings": {
            "restart": RESTART, "tolerance": 1e-14, "max_iterations": 1, "preconditioner": "block",
            "frequency_hz": args.frequency_hz, "LOCKSTEP_WORK_VECTORS": LOCKSTEP_WORK_VECTORS,
            "LOCKSTEP_FFT_BYTES_PER_CELL": LOCKSTEP_FFT_BYTES_PER_CELL,
            "LOCKSTEP_MEMORY_FRACTION": LOCKSTEP_MEMORY_FRACTION,
        },
        "cases": cases,
        "decision": {
            "adopted": adopted,
            "criteria": f"on every case the estimate is at least the measured column and at most {ESTIMATE_MAX_RATIO}x it",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("decision:", json.dumps(report["decision"]))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
