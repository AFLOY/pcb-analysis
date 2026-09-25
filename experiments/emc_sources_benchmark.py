"""Legacy Python loop, shipped NumPy path, and C++ construction of sheet-PEEC current elements.

A fully occupied three-layer sheet mesh with a via bank and random complex
branch currents; ``dipoles_from_sheet_peec`` on the current Python loop, on a
NumPy-vectorised rewrite kept here as the measured alternative, and on the
C++ kernel at each thread count.  Writes a JSON with ``environment`` and
``decision``.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/emc_sources_benchmark.py --shapes 200x200,500x500 --threads 1,4,16
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from electrical.sheet_peec.sheet_operator import SheetLayer, SheetStackup  # noqa: E402
from electrical.sheet_peec.sheet_peec import SheetMesh, ViaBranch  # noqa: E402
from emc.tiled_dipole_superposition import CurrentDipoles, dipoles_from_sheet_peec  # noqa: E402
from emc.tiled_dipole_superposition.native_dipole import native_available  # noqa: E402


def mesh_of(rows: int, cols: int) -> SheetMesh:
    stackup = SheetStackup(
        (
            SheetLayer("F.Cu", 0.0, 35e-6, 1.724e-8),
            SheetLayer("In1.Cu", -0.4e-3, 35e-6, 1.724e-8),
            SheetLayer("B.Cu", -1.6e-3, 35e-6, 1.724e-8),
        )
    )
    occupancy = np.ones((3, rows, cols), dtype=bool)
    step = max(1, rows // 20)
    vias = tuple(
        ViaBranch(r, c, 0, 2, resistance_ohm=1e-3)
        for r in range(step, rows, step)
        for c in range(step, cols, step)
    )
    return SheetMesh((rows, cols), 2e-4, stackup, occupancy, vias=vias)


def legacy_loop(mesh: SheetMesh, solution: Any) -> CurrentDipoles:
    """The Python loop the shipped path replaced (kept here as the reference it was measured against)."""

    pitch = float(mesh.pitch_m)
    heights = np.asarray([layer.z_m for layer in mesh.stackup.layers], dtype=np.float64)
    current = np.asarray(solution.branch_current, dtype=np.complex128)
    positions: list[tuple[float, float, float]] = []
    moments: list[tuple[complex, complex, complex]] = []
    index = 0
    for layer, row, col in mesh.branch_x:
        positions.append(((col + 1.0) * pitch, (row + 0.5) * pitch, heights[layer]))
        moments.append((current[index] * pitch, 0.0, 0.0))
        index += 1
    for layer, row, col in mesh.branch_y:
        positions.append(((col + 0.5) * pitch, (row + 1.0) * pitch, heights[layer]))
        moments.append((0.0, current[index] * pitch, 0.0))
        index += 1
    for via in mesh.via_branches:
        lower = heights[via.lower_layer]
        upper = heights[via.upper_layer]
        positions.append(((via.col + 0.5) * pitch, (via.row + 0.5) * pitch, 0.5 * (lower + upper)))
        moments.append((0.0, 0.0, current[index] * (upper - lower)))
        index += 1
    return CurrentDipoles(
        np.asarray(positions, dtype=np.float64).reshape(-1, 3),
        np.asarray(moments, dtype=np.complex128).reshape(-1, 3),
    )


def vectorised(mesh: SheetMesh, solution: Any) -> CurrentDipoles:
    """The shipped portable path: NumPy indexing (dipoles_from_sheet_peec with native=False)."""

    return dipoles_from_sheet_peec(mesh, solution, native=False)


def _timed(fn: Callable[[], Any], repeats: int, warmups: int) -> tuple[Any, dict[str, Any]]:
    for _ in range(warmups):
        result = fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - start) * 1.0e3)
    return result, {"median_ms": statistics.median(samples), "minimum_ms": min(samples), "maximum_ms": max(samples),
                    "samples_ms": samples, "repeats": repeats}


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _compiler() -> str:
    try:
        return subprocess.run(["g++", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _same(a: CurrentDipoles, b: CurrentDipoles) -> bool:
    return bool(np.array_equal(a.position_m, b.position_m) and np.array_equal(a.moment_a_m, b.moment_a_m))


def _case(rows: int, cols: int, repeats: int, threads: list[int]) -> dict[str, Any]:
    mesh = mesh_of(rows, cols)
    rng = np.random.default_rng(rows * 1000 + cols)
    solution = SimpleNamespace(branch_current=rng.standard_normal(mesh.branch_count) + 1j * rng.standard_normal(mesh.branch_count))
    loop, loop_timing = _timed(lambda: legacy_loop(mesh, solution), repeats, 1)
    vector, vector_timing = _timed(lambda: vectorised(mesh, solution), repeats, 1)
    by_threads: dict[str, Any] = {}
    for count in threads:
        native, timing = _timed(lambda: dipoles_from_sheet_peec(mesh, solution, native=True, native_threads=count), repeats, 1)
        by_threads[str(count)] = {
            "threads": count,
            "native_ms": timing,
            "speedup_vs_loop": loop_timing["median_ms"] / timing["median_ms"],
            "speedup_vs_vectorised_numpy": vector_timing["median_ms"] / timing["median_ms"],
            "identical_to_loop": _same(native, loop),
        }
    return {
        "shape": [rows, cols],
        "branches": mesh.branch_count,
        "vias": len(mesh.via_branches),
        "loop_ms": loop_timing,
        "vectorised_numpy_ms": vector_timing,
        "vectorised_numpy_identical_to_loop": _same(vector, loop),
        "vectorised_numpy_speedup_vs_loop": loop_timing["median_ms"] / vector_timing["median_ms"],
        "native_by_threads": by_threads,
    }


def run(shapes: list[tuple[int, int]], repeats: int, threads: list[int]) -> dict[str, Any]:
    if not native_available():
        raise SystemExit("emc native extension not built; run python -m emc.tiled_dipole_superposition.native.build")
    cases = [_case(r, c, repeats, threads) for r, c in shapes]
    first = str(threads[0])
    speedups = [c["native_by_threads"][first]["speedup_vs_loop"] for c in cases]
    vs_numpy = [c["native_by_threads"][first]["speedup_vs_vectorised_numpy"] for c in cases]
    identical = all(c["native_by_threads"][t]["identical_to_loop"] for c in cases for t in c["native_by_threads"])
    return {
        "schema": "emc-sources-benchmark/v1",
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
                        "cpu": _cpu_model(), "cpu_count": os.cpu_count(), "compiler": _compiler(), "thread_sweep": threads,
                        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"), "device": "cpu"},
        "definitions": {"timing": "wall time of dipoles_from_sheet_peec (branch order, positions and complex moments)",
                        "loop": "the Python loop over branches that the shipped path replaced (legacy_loop in this script)",
                        "vectorised_numpy": "the shipped portable path: dipoles_from_sheet_peec(native=False), NumPy indexing",
                        "native": "C++ kernel sheet_branch_dipoles, OpenMP over branches",
                        "decision_threads": "the first thread count in the sweep"},
        "cases": cases,
        "decision": {
            "adopted": min(vs_numpy) >= 2.0 and identical,
            "criteria": "C++ at least 2x faster than the shipped NumPy path on every case at the decision thread count and bit-identical output",
            "minimum_speedup_vs_loop": min(speedups),
            "minimum_speedup_vs_vectorised_numpy": min(vs_numpy),
            "all_identical": identical,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--shapes", default="100x100,300x300,600x600", help="comma separated rows x cols")
    parser.add_argument("--threads", default="1")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results") / "emc_sources_benchmark.json")
    args = parser.parse_args()
    shapes = [tuple(int(v) for v in item.split("x")) for item in args.shapes.split(",")]
    report = run(shapes, args.repeats, [int(v) for v in args.threads.split(",")])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for case in report["cases"]:
        print(f"branches={case['branches']:8d} loop={case['loop_ms']['median_ms']:8.1f} ms numpy={case['vectorised_numpy_ms']['median_ms']:7.1f} ms (x{case['vectorised_numpy_speedup_vs_loop']:.1f})")
        for item in case["native_by_threads"].values():
            print(f"    threads={item['threads']:2d} native={item['native_ms']['median_ms']:7.2f} ms x{item['speedup_vs_loop']:6.1f} vs loop, x{item['speedup_vs_vectorised_numpy']:5.2f} vs numpy, identical={item['identical_to_loop']}")
    print("decision:", json.dumps(report["decision"]))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
