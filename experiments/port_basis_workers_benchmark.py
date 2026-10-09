"""Threaded unit solves of the N-port basis: wall time, memory and identity of the result.

The two-layer board of ``dc_native_benchmark.py`` (60 mm square, a slot, a via
bank) with the two edge pads plus interior taps as ports.  For every size and
port count, ``dc_port_basis`` runs with 1, 2, 4 and 8 worker threads
(OpenMP team 1), and once more with one worker and an OpenMP team
of 4 on the native path, so the two kinds of CPU parallelism are compared at
the same core count, and with both combined on 8 cores (2 x 4 and 4 x 2).  Every variant runs in its own subprocess so that its
peak RSS and its thread environment are its own; the fields of every variant
are compared with the one-worker result.  Writes a JSON with ``environment``
and ``decision``.

    PCB_NATIVE_Q1=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \\
        experiments/port_basis_workers_benchmark.py --sizes 100,200,300 --ports 5,9 --repeats 2
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dc_native_benchmark import _compiler, _cpu_model, board  # noqa: E402

WORKERS = (1, 2, 4, 8)
OPENMP_TEAM = 4
COMBINED = ((2, 4), (4, 2))   # (workers, OpenMP team): both kinds together on 8 cores
SPEEDUP_REQUIRED = 1.5      # 4 workers against 1, at the largest size
MEMORY_LIMIT_RATIO = 2.0    # peak RSS of 4 workers at most this many times the serial one
IDENTITY_TOLERANCE = 0.0    # the unit solves are independent; the fields must agree bit for bit
SOLVER = dict(max_inner_iterations=400, max_outer_iterations=16, relative_tolerance=1.0e-8)


def ports(problem: Any, count: int) -> Any:
    """The two edge pads plus ``count - 2`` three-node taps along one interior row."""

    from electrical.matrix_free_mpir_fem import PortSet

    source, sink = problem.terminals
    elements = problem.mesh.element_active.shape[1]
    row = 2 * elements // 3
    taps = [
        tuple((0, r, elements * (i + 1) // (count - 1)) for r in range(row, row + 3))
        for i in range(count - 2)
    ]
    pads = (source.nodes, *taps, sink.nodes)
    names = ("source", *(f"tap{i}" for i in range(len(taps))), "sink")
    return PortSet(pads=pads, names=names, reference=len(pads) - 1)


def run_variant(elements: int, count: int, workers: int, team: int, repeats: int, out: Path) -> None:
    from electrical.matrix_free_mpir_fem import MPIRConfig, dc_port_basis
    from electrical.matrix_free_mpir_fem import ports as port_module
    from electrical.matrix_free_mpir_fem.native_dc import native_requested
    from electrical.threads import set_thread_budget

    # The pool width and the OpenMP team are internal to pcb-analysis; this
    # measurement pins its split (``ports._split_budget``) to the variant and
    # gives the process exactly their product as its budget.
    set_thread_budget(workers * team)
    port_module._split_budget = lambda budget, tasks: (workers, team)

    problem = board(elements)
    port_set = ports(problem, count)
    config = MPIRConfig(**SOLVER)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        basis = dc_port_basis(problem.mesh, port_set, vias=problem.vias, config=config)
        samples.append(time.perf_counter() - started)
    np.save(out, basis.unit_voltage_potential_v)
    print(json.dumps({
        "elements": elements, "nodes": int(problem.mesh.size), "ports": count,
        "workers": workers, "workers_used": int(basis.workers),
        "native_threads": team, "thread_budget": workers * team,
        "native": bool(native_requested()),
        "low_operator_backend": basis.operator.low_operator_backend,
        "converged": bool(basis.converged), "inner_iterations": int(basis.inner_iterations),
        "wall_s": round(statistics.median(samples), 3), "samples_s": [round(s, 3) for s in samples],
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="100,200,300")
    parser.add_argument("--ports", default="5,9")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "benchmark-results" / "port_basis_workers_benchmark.json")
    parser.add_argument("--variant", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.variant:
        elements, count, workers, team = (int(v) for v in args.variant.split(":"))
        run_variant(elements, count, workers, team, args.repeats, args.out)
        return

    scratch = args.output.parent / "port_basis_workers_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for elements in (int(v) for v in args.sizes.split(",") if v):
        for count in (int(v) for v in args.ports.split(",") if v):
            variants = [(w, 1) for w in WORKERS] + [(1, OPENMP_TEAM)] + list(COMBINED)
            results: list[dict[str, Any]] = []
            for workers, team in variants:
                out = scratch / f"{elements}_{count}_{workers}_{team}.npy"
                done = subprocess.run(
                    [sys.executable, __file__, "--variant", f"{elements}:{count}:{workers}:{team}", "--repeats", str(args.repeats), "--out", str(out)],
                    capture_output=True, text=True, timeout=3600,
                )
                if done.returncode != 0:
                    results.append({"workers": workers, "native_threads": team, "failed": (done.stderr.strip().splitlines() or ["killed"])[-1][:200]})
                    continue
                results.append(json.loads(done.stdout.strip().splitlines()[-1]))
            reference = next((r for r in results if r.get("workers") == 1 and r.get("native_threads") == 1 and "failed" not in r), None)
            if reference is not None:
                ref = np.load(scratch / f"{elements}_{count}_1_1.npy")
                for record in results:
                    if "failed" in record:
                        continue
                    sol = np.load(scratch / f"{elements}_{count}_{record['workers']}_{record['native_threads']}.npy")
                    record["max_abs_difference_vs_serial_v"] = float(np.abs(sol - ref).max())
                    record["speedup_vs_serial"] = round(reference["wall_s"] / record["wall_s"], 2) if record["wall_s"] > 0 else None
            cases.append({"elements": elements, "ports": count, "results": results})
            for record in results:
                if "failed" in record:
                    print(f"{elements}^2 {count} ports workers {record['workers']} omp {record['native_threads']} FAILED {record['failed']}")
                else:
                    print(
                        f"{elements}^2 {count} ports workers {record['workers']} omp {record['native_threads']}: "
                        f"{record['wall_s']:7.2f} s ({record.get('speedup_vs_serial', 1.0):.2f}x) inner {record['inner_iterations']:5d} "
                        f"peak {record['peak_rss_mb']:5d} MB diff {record.get('max_abs_difference_vs_serial_v', float('nan')):.1e}"
                    )

    largest = [c for c in cases if c["elements"] == max(c["elements"] for c in cases)]
    verdicts = []
    for case in largest:
        serial = next((r for r in case["results"] if r.get("workers") == 1 and r.get("native_threads") == 1 and "failed" not in r), None)
        four = next((r for r in case["results"] if r.get("workers") == 4 and "failed" not in r), None)
        identical = all(r.get("max_abs_difference_vs_serial_v", 1.0) <= IDENTITY_TOLERANCE for r in case["results"] if "failed" not in r and r["native_threads"] == 1)
        if serial and four:
            verdicts.append(
                bool(four["converged"]) and identical
                and four["speedup_vs_serial"] >= SPEEDUP_REQUIRED
                and four["peak_rss_mb"] <= MEMORY_LIMIT_RATIO * serial["peak_rss_mb"]
            )
    report = {
        "benchmark": "port_basis_workers_benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "cpu": _cpu_model(), "cpu_count": os.cpu_count(), "platform": platform.platform(),
            "python": platform.python_version(), "numpy": np.__version__, "compiler": _compiler(),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"), "pcb_native_q1": os.environ.get("PCB_NATIVE_Q1"),
        },
        "settings": {"sizes": args.sizes, "ports": args.ports, "repeats": args.repeats, "workers": WORKERS, "openmp_team": OPENMP_TEAM, "solver": SOLVER},
        "cases": cases,
        "decision": {
            "adopted": bool(verdicts) and all(verdicts),
            "criteria": (
                f"at the largest size, 4 workers converge, give the serial fields bit for bit, run at least "
                f"{SPEEDUP_REQUIRED:g}x faster than one worker, and need at most {MEMORY_LIMIT_RATIO:g}x its peak RSS; "
                "the pcb-analysis default stays one worker either way (the consumer sets the thread budget)"
            ),
            "verdicts_per_port_count_at_largest_size": verdicts,
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("decision:", json.dumps(report["decision"]))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
