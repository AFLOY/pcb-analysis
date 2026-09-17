"""Array path versus C++ direct summation for the dipole near and far fields.

Random current elements in a 4 cm cube with complex moments, a near-field scan
plane 5 cm above them, and the default Gauss-Legendre sphere (2,048
directions).  Both paths are complex128 on the host.  Writes a JSON with
``environment`` and ``decision`` to ``benchmark-results/``.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/emc_native_benchmark.py --threads 1,4,16
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
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emc.tiled_dipole_superposition import (  # noqa: E402
    CurrentDipoles,
    evaluate_fields,
    far_field_pattern,
    scan_plane,
)
from emc.tiled_dipole_superposition.native_dipole import native_available  # noqa: E402

FREQUENCY_HZ = 300.0e6


def _timed(fn: Callable[[], Any], repeats: int, warmups: int) -> tuple[Any, dict[str, Any]]:
    for _ in range(warmups):
        result = fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - start) * 1.0e3)
    return result, {"median_ms": statistics.median(samples), "minimum_ms": min(samples),
                    "maximum_ms": max(samples), "samples_ms": samples, "repeats": repeats}


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


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _case(sources: int, points_side: int, repeats: int, threads: list[int]) -> dict[str, Any]:
    rng = np.random.default_rng(sources)
    moment = (rng.normal(size=(sources, 3)) + 1j * rng.normal(size=(sources, 3))) * 1e-3
    dipoles = CurrentDipoles(rng.normal(scale=0.02, size=(sources, 3)), moment)
    points = scan_plane(np.linspace(-0.03, 0.03, points_side), np.linspace(-0.03, 0.03, points_side), 0.05)
    near_ref, near_timing = _timed(lambda: evaluate_fields(dipoles, points, FREQUENCY_HZ, native=False), repeats, 1)
    far_ref, far_timing = _timed(lambda: far_field_pattern(dipoles, FREQUENCY_HZ, native=False), repeats, 1)
    by_threads: dict[str, Any] = {}
    for count in threads:
        near, nt = _timed(lambda: evaluate_fields(dipoles, points, FREQUENCY_HZ, native=True, native_threads=count), repeats, 1)
        far, ft = _timed(lambda: far_field_pattern(dipoles, FREQUENCY_HZ, native=True, native_threads=count), repeats, 1)
        by_threads[str(count)] = {
            "threads": count,
            "near_field_ms": nt,
            "far_field_ms": ft,
            "near_speedup_vs_array": near_timing["median_ms"] / nt["median_ms"],
            "far_speedup_vs_array": far_timing["median_ms"] / ft["median_ms"],
            "relative_error_magnetic": _relative(near.magnetic_a_per_m, near_ref.magnetic_a_per_m),
            "relative_error_electric": _relative(near.electric_v_per_m, near_ref.electric_v_per_m),
            "relative_error_far_field": _relative(far.electric_v_per_m, far_ref.electric_v_per_m),
            "radiated_power_w": far.radiated_power_w,
        }
    return {
        "sources": sources,
        "points": int(points.shape[0]),
        "directions": far_ref.sampling.count,
        "pairs_near": sources * int(points.shape[0]),
        "array_near_field_ms": near_timing,
        "array_far_field_ms": far_timing,
        "array_radiated_power_w": far_ref.radiated_power_w,
        "native_by_threads": by_threads,
    }


def run(shapes: list[tuple[int, int]], repeats: int, threads: list[int]) -> dict[str, Any]:
    if not native_available():
        raise SystemExit("emc native extension not built; run python -m emc.tiled_dipole_superposition.native.build")
    cases = [_case(s, p, repeats, threads) for s, p in shapes]
    first = str(threads[0])
    near = [c["native_by_threads"][first]["near_speedup_vs_array"] for c in cases]
    far = [c["native_by_threads"][first]["far_speedup_vs_array"] for c in cases]
    err = max(max(c["native_by_threads"][t][k] for t in c["native_by_threads"] for k in
                  ("relative_error_magnetic", "relative_error_electric", "relative_error_far_field")) for c in cases)
    return {
        "schema": "emc-native-benchmark/v1",
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
                        "cpu": _cpu_model(), "cpu_count": os.cpu_count(), "compiler": _compiler(), "thread_sweep": threads,
                        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"), "device": "cpu"},
        "definitions": {"timing": "wall time of evaluate_fields (electric and magnetic) and far_field_pattern; complex128 on both paths",
                        "speedup": "array-path median divided by native median", "frequency_hz": FREQUENCY_HZ,
                        "native_path": "direct pairwise summation in C++, threads own disjoint observation points",
                        "decision_threads": "the first thread count in the sweep"},
        "cases": cases,
        "decision": {
            "adopted": min(near) >= 2.0 and min(far) >= 1.0 and err <= 1.0e-10,
            "criteria": "near field at least 2x on every case and far field not slower at the decision thread count; all results within 1e-10 of the array path",
            "minimum_near_speedup": min(near), "minimum_far_speedup": min(far), "maximum_relative_error": err,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--shapes", default="2000x42,8000x64", help="comma separated sources x points-per-side")
    parser.add_argument("--threads", default="1")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results") / "emc_native_benchmark.json")
    args = parser.parse_args()
    shapes = [tuple(int(v) for v in item.split("x")) for item in args.shapes.split(",")]
    report = run(shapes, args.repeats, [int(v) for v in args.threads.split(",")])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for case in report["cases"]:
        print(f"sources={case['sources']} points={case['points']} array near={case['array_near_field_ms']['median_ms']:9.1f} ms far={case['array_far_field_ms']['median_ms']:8.1f} ms")
        for item in case["native_by_threads"].values():
            print(f"    threads={item['threads']:2d} near={item['near_field_ms']['median_ms']:8.1f} ms x{item['near_speedup_vs_array']:6.1f} far={item['far_field_ms']['median_ms']:7.1f} ms x{item['far_speedup_vs_array']:5.1f} err={max(item['relative_error_magnetic'], item['relative_error_electric'], item['relative_error_far_field']):.1e}")
    print("decision:", json.dumps(report["decision"]))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
