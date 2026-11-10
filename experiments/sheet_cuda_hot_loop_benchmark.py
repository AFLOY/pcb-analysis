"""Solve the LM5145EVM-HD-20A-2L hot loop on the sheet PEEC, CPU against CUDA.

The fixtures under ``experiments/fixtures/evm2l_hot_loop`` are the current-field
problems plane_opt hands pcb-analysis for that board's commutation loop: Q1's
switch node driven against C3, the two bridges of the shorted switches as extra
sheets, one filament per layer, at 0.4, 0.2 and 0.1 mm.  The loop impedance is
the entry pad's mean voltage less the exits' and less the capacitor's, per
ampere.  The same problems were solved by Radia's HACApK PEEC on Flow2, so the
times here sit beside those.

    python experiments/sheet_cuda_hot_loop_benchmark.py --backend cuda \\
        --resolution 0p2mm --frequency 1e6 --frequency 3e7 --output benchmark-results/x.json

The JSON carries the environment and a ``decision`` field (left ``null``: a
single run decides nothing; comparisons fill it).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "evm2l_hot_loop"


def load_problem(resolution: str) -> dict[str, Any]:
    with gzip.open(FIXTURES / f"evm2l_loop_{resolution}_point.problem.json.gz", "rt", encoding="utf-8") as handle:
        return json.load(handle)


def environment(backend: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
    }
    try:
        import scipy

        record["scipy"] = scipy.__version__
    except ImportError:
        pass
    if backend == "cuda":
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        record.update({
            "cupy": cp.__version__,
            "device": props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"]),
            "compute_capability": f"{props['major']}.{props['minor']}",
            "device_memory_bytes": int(props["totalGlobalMem"]),
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver": int(cp.cuda.runtime.driverGetVersion()),
        })
    return record


def loop_impedance(doc: dict[str, Any], voltage: dict[Any, complex]) -> complex:
    mesh = doc["mesh"]

    def mean(pad: str) -> complex:
        cells = next(t for t in doc["terminals"] if t["pad"] == pad)["cells"]
        return complex(np.mean([voltage[(c["layer"], c["x"], c["y"])] for c in cells]))

    exits = np.mean([mean(pad) for pad in mesh["exits"]])
    return (mean(mesh["entry"]) - exits) - (mean(mesh["positive"]) - mean(mesh["negative"]))


def main(argv: list[str] | None = None) -> int:
    from electrical.sheet_peec.current_field_contract import CurrentFieldProblem, solve_current_field_problem
    from electrical.threads import set_thread_budget

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--resolution", default="0p2mm", choices=("0p4mm", "0p2mm", "0p1mm"))
    parser.add_argument("--frequency", type=float, action="append", required=True)
    parser.add_argument("--preconditioner", help="override the fixture's preconditioner")
    parser.add_argument("--setting", action="append", default=[], help="extra solver setting KEY=JSON")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    set_thread_budget(args.threads)
    doc = load_problem(args.resolution)
    settings = dict(doc["solver_settings"])
    settings["execution_backend"] = args.backend
    if args.preconditioner:
        settings["preconditioner"] = args.preconditioner
    for item in args.setting:
        key, value = item.split("=", 1)
        settings[key] = json.loads(value)
    rows = []
    for frequency in args.frequency:
        started = time.perf_counter()
        problem = CurrentFieldProblem.from_mapping(dict(doc, frequency_hz=frequency))
        parsed = time.perf_counter()
        result = solve_current_field_problem(problem, settings)
        solved = time.perf_counter()
        impedance = loop_impedance(doc, result.voltage_phasor)
        metrics = {key: value for key, value in result.metrics.items() if isinstance(value, (int, float, str, bool))}
        row = {
            "frequency_hz": frequency,
            "R_mohm": impedance.real * 1e3,
            "L_nh": impedance.imag / (2 * np.pi * frequency) * 1e9,
            "parse_s": parsed - started,
            "solve_s": solved - parsed,
            "metrics": metrics,
        }
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "metrics"}), flush=True)
    record = {
        "schema": "pcb-analysis.sheet-cuda-hot-loop/v1",
        "fixture": f"evm2l_loop_{args.resolution}_point",
        "backend": args.backend,
        "settings": settings,
        "threads": args.threads,
        "environment": environment(args.backend),
        "rows": rows,
        "decision": None,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
