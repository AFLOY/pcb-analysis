"""The C++ sheet operator applied to several sets at once on kept workspaces, against one set at a time.

Run once per build of ``electrical._pcbcore`` and compare:

    .venv/bin/python experiments/sheet_operator_batching_benchmark.py --label current --out benchmark-results/op_current.json
    .venv/bin/python experiments/sheet_operator_batching_benchmark.py --label candidate --out benchmark-results/op_candidate.json
    .venv/bin/python experiments/sheet_operator_batching_benchmark.py --compare benchmark-results/op_current.json \\
        benchmark-results/op_candidate.json [more CURRENT CANDIDATE pairs] --out docs/SHEET_OPERATOR_BATCHING_RESULTS.json

Each pair is one compiler's build of the current and the candidate core (the
label of the candidate names it); every build must give the same bits.

The fixture is shaped like the hot loop of a four-layer board whose switches
are bridged by sheets of their own above the top layer (Kicad_PowerOpt's
LM5156 hot loop: 225 x 320 cells at 0.25 mm, eight layers of which four are
bridge sheets, seven vertical levels).

``operator``: the operator alone on that grid, one set of currents at thread
budgets 1, 2, 4 and 8, and ten sets in one ``apply_many`` at 8 (a lockstep
step of five excitations, each a real and an imaginary set).

``solve``: ``solve_sheet_cases`` on a two-layer plane (the plane of
``sheet_preconditioner_benchmark.py`` at 0.25 mm) with four bridge sheets
above it, eight excitations in lockstep, ``near``, tolerance 1e-6, restart 120.

Both record SHA-256 digests of their outputs: the candidate must give the
current build's bits.  Writes a JSON with ``environment`` and ``decision``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dc_native_benchmark import _cpu_model  # noqa: E402

PITCH_MM = 0.25
HOT_LOOP_SHAPE = (225, 320)
# Top to bottom: four bridge sheets 0.135 mm apart above F.Cu, then the board.
HOT_LOOP_LAYERS = (
    ("bridge 4", 0.0, 35e-6), ("bridge 3", -0.135e-3, 35e-6), ("bridge 2", -0.27e-3, 35e-6),
    ("bridge 1", -0.405e-3, 35e-6), ("F.Cu", -0.568e-3, 71.1e-6), ("In1.Cu", -0.8756e-3, 36e-6),
    ("In2.Cu", -1.7244e-3, 36e-6), ("B.Cu", -2.032e-3, 71.1e-6),
)
HOT_LOOP_LEVELS = ((5, 4), (6, 5), (7, 6), (4, 1), (4, 0), (4, 2), (4, 3))
SETS_TOGETHER = 10
EXCITATIONS = 8


def _digest(arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _median_seconds(call, repeats: int) -> float:
    call()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        times.append(time.perf_counter() - started)
    return float(np.median(times))


def operator_case() -> dict:
    from electrical.sheet_peec.sheet_operator import SheetInductanceOperator, SheetLayer, SheetStackup
    from electrical.threads import thread_budget_scope

    stackup = SheetStackup(tuple(SheetLayer(name, z, t, 1.724e-8) for name, z, t in HOT_LOOP_LAYERS))
    rows, cols = HOT_LOOP_SHAPE
    with thread_budget_scope(8):
        operator = SheetInductanceOperator(HOT_LOOP_SHAPE, PITCH_MM * 1e-3, stackup, vertical_levels=HOT_LOOP_LEVELS)
    rng = np.random.default_rng(2026)
    layers, levels = len(HOT_LOOP_LAYERS), len(HOT_LOOP_LEVELS)
    sets = [
        (rng.standard_normal((layers, rows, cols)), rng.standard_normal((layers, rows, cols)),
         rng.standard_normal((levels, rows, cols)))
        for _ in range(SETS_TOGETHER)
    ]
    single = {}
    for threads in (1, 2, 4, 8):
        with thread_budget_scope(threads):
            single[str(threads)] = round(1e3 * _median_seconds(lambda: operator.apply(*sets[0]), 7), 2)
    with thread_budget_scope(8):
        together = round(1e3 * _median_seconds(lambda: operator.apply_many(sets), 3) / SETS_TOGETHER, 2)
        outputs = [part for result in operator.apply_many(sets) for part in result]
    return {
        "shape": list(HOT_LOOP_SHAPE), "layers": layers, "levels": levels,
        "one_set_ms_by_threads": single,
        "ms_per_set_together_at_8": together, "sets_together": SETS_TOGETHER,
        "digest": _digest(outputs),
    }


def solve_case() -> dict:
    from electrical.sheet_peec.sheet_operator import SheetInductanceOperator, SheetLayer, SheetStackup
    from electrical.sheet_peec.sheet_peec import SheetMesh, Terminal, ViaBranch, solve_sheet_cases
    from electrical.threads import thread_budget_scope

    rows, cols = int(round(30.0 / PITCH_MM)), int(round(33.0 / PITCH_MM))
    bridges = 4
    layers = tuple(SheetLayer(f"bridge {k}", 0.135e-3 * (bridges - k), 35e-6, 1.724e-8) for k in range(bridges))
    layers += (SheetLayer("F.Cu", 0.0, 35e-6, 1.724e-8), SheetLayer("B.Cu", -1.6e-3, 35e-6, 1.724e-8))
    stackup = SheetStackup(layers)
    top, bottom = bridges, bridges + 1
    occupancy = np.zeros((len(layers), rows, cols), dtype=bool)
    occupancy[top:] = True
    vias = []
    step = max(1, rows // 8)
    vias += [ViaBranch(r, c, top, bottom, resistance_ohm=1e-3) for r in range(step, rows, step) for c in range(cols // 2, cols, step)]
    for k in range(bridges):
        # A part's package: a 6 x 15 cell sheet joined to F.Cu at its two ends.
        r0, c0 = 20 + 20 * k, 30 + 18 * k
        occupancy[k, r0:r0 + 6, c0:c0 + 15] = True
        vias += [ViaBranch(r, c, top, k, resistance_ohm=1e-4) for r in range(r0, r0 + 6) for c in (c0, c0 + 14)]
    mesh = SheetMesh((rows, cols), PITCH_MM * 1e-3, stackup, occupancy, vias=tuple(vias))
    with thread_budget_scope(8):
        operator = SheetInductanceOperator((rows, cols), PITCH_MM * 1e-3, stackup, vertical_levels=mesh.vertical_levels)
    sets = []
    for k in range(EXCITATIONS):
        low, high = (k * rows) // (2 * EXCITATIONS), rows - (k * rows) // (2 * EXCITATIONS)
        sets.append([
            Terminal("in", top, tuple((r, 0) for r in range(low, high)), 10.0),
            Terminal("out", bottom, tuple((r, cols - 1) for r in range(rows)), -10.0),
        ])
    options = dict(frequency_hz=30e6, preconditioner="near", tolerance=1e-6, restart=120, max_iterations=200)
    with thread_budget_scope(8):
        started = time.perf_counter()
        solutions = solve_sheet_cases(mesh, operator, sets, **options)
        wall = time.perf_counter() - started
    return {
        "shape": [rows, cols], "layers": len(layers), "levels": len(mesh.vertical_levels),
        "branches": int(mesh.branch_count), "nodes": int(mesh.node_count), "excitations": EXCITATIONS,
        "settings": {key: value for key, value in options.items()},
        "iterations": [int(s.iterations) for s in solutions],
        "lockstep_columns": [int(s.lockstep_columns or 1) for s in solutions],
        "converged": all(bool(s.converged) for s in solutions),
        "wall_s": round(wall, 2),
        "digest": _digest([a for s in solutions for a in (s.branch_current, s.node_voltage)]),
    }


def environment() -> dict:
    from electrical.threads import available_threads

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    except OSError:
        commit = None
    return {
        "cpu": _cpu_model(), "cpu_count": os.cpu_count(), "physical_cores": available_threads(),
        "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__, "scipy": __import__("scipy").__version__, "commit": commit,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def measure(label: str, compiler: str | None, out: Path) -> None:
    from electrical import backend

    if not backend.native_available():
        raise SystemExit("electrical._pcbcore is not built")
    env = environment()
    env["compiler"] = compiler
    result = {"label": label, "environment": env, "operator": operator_case(), "solve": solve_case()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


def _pair(current: dict, candidate: dict) -> dict:
    same_operator = current["operator"]["digest"] == candidate["operator"]["digest"]
    same_solve = current["solve"]["digest"] == candidate["solve"]["digest"]
    threads = current["operator"]["one_set_ms_by_threads"]
    no_slower = all(candidate["operator"]["one_set_ms_by_threads"][t] <= threads[t] for t in threads)
    faster_solve = candidate["solve"]["wall_s"] < current["solve"]["wall_s"]
    return {
        "current": current, "candidate": candidate,
        "operator_speedup_together_at_8": round(
            current["operator"]["ms_per_set_together_at_8"] / candidate["operator"]["ms_per_set_together_at_8"], 2),
        "solve_speedup": round(current["solve"]["wall_s"] / candidate["solve"]["wall_s"], 2),
        "same_bits": {"operator": same_operator, "solve": same_solve},
        "passes": same_operator and same_solve and no_slower and faster_solve and candidate["solve"]["converged"],
    }


def compare(paths: list[Path], out: Path) -> None:
    if len(paths) % 2:
        raise SystemExit("--compare takes CURRENT CANDIDATE pairs")
    loaded = [json.loads(path.read_text()) for path in paths]
    pairs = {loaded[k + 1]["label"]: _pair(loaded[k], loaded[k + 1]) for k in range(0, len(loaded), 2)}
    digests = {(entry["operator"]["digest"], entry["solve"]["digest"]) for entry in loaded}
    result = {
        "benchmark": "sheet_operator_batching",
        "environment": loaded[1]["environment"],
        "pairs": pairs,
        "same_bits_across_all_builds": len(digests) == 1,
        "decision": {
            "adopted": all(pair["passes"] for pair in pairs.values()),
            "criteria": "for every compiler: operator and solve outputs identical to the bit, the operator no "
                        "slower at any thread budget, the solve faster and converged",
        },
    }
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({label: {k: pair[k] for k in ("operator_speedup_together_at_8", "solve_speedup", "same_bits", "passes")}
                      for label, pair in pairs.items()}, indent=1))
    print(json.dumps({k: result[k] for k in ("same_bits_across_all_builds", "decision")}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label")
    parser.add_argument("--compiler", help="the compiler the core was built with, recorded as given")
    parser.add_argument("--compare", nargs="+", type=Path, metavar="JSON",
                        help="CURRENT CANDIDATE pairs, one per compiler")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.compare:
        compare(args.compare, args.out)
    else:
        measure(args.label or "build", args.compiler, args.out)


if __name__ == "__main__":
    main()
