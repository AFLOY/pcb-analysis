"""Point-in-solid classification: OpenCASCADE per point vs winding number (NumPy, C++).

Samples the copper layer of a synthetic board (plane plus pads plus vias) and
a voxel grid over a heat sink, as the front end does, and times the three
paths on the same points.  Writes a JSON with ``environment`` and
``decision`` to ``benchmark-results/``; the adopted copy lives at
``docs/GEOMETRY_CLASSIFY_RESULTS.json``.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/geometry_classify_benchmark.py --threads 1,4
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geometry.cad_import import (  # noqa: E402
    box_solid,
    cylinder_solid,
    native_classify_available,
    sample_plane_fill,
    sample_volume_fill,
)

MM = 1.0e-3


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


def _timed(fn: Callable[[], Any], repeats: int) -> tuple[Any, dict[str, float]]:
    times, result = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - start) * 1.0e3)
    return result, {"median_ms": statistics.median(times), "min_ms": min(times), "repeats": repeats}


def fixture() -> dict[str, list[Any]]:
    copper = [box_solid("plane", (1 * MM, 1 * MM, 0.0), (38 * MM, 28 * MM, 35e-6))]
    for i in range(6):
        for j in range(4):
            copper.append(box_solid(f"pad{i}{j}", ((4 + 6 * i) * MM, (4 + 7 * j) * MM, 0.0), (2 * MM, 1.5 * MM, 35e-6)))
            copper.append(cylinder_solid(f"via{i}{j}", ((7 + 6 * i) * MM, (5 + 7 * j) * MM), -1.6 * MM, 1.7 * MM, 0.25 * MM))
    sink = [box_solid("base", (10 * MM, 8 * MM, 1.7 * MM), (20 * MM, 16 * MM, 3 * MM))]
    for f in range(8):
        sink.append(box_solid(f"fin{f}", ((10 + 2.5 * f) * MM, 8 * MM, 4.7 * MM), (1.0 * MM, 16 * MM, 12 * MM)))
    return {"copper": copper, "sink": sink}


def run(repeats: int, threads: list[int], pitch_mm: float, supersample: int) -> dict[str, Any]:
    solids = fixture()
    cases = []
    plane_kwargs = dict(z_m=17.5e-6, origin_m=(0.0, 0.0), pitch_m=pitch_mm * MM, shape=(int(30 / pitch_mm), int(40 / pitch_mm)), supersample=supersample)
    voxel_kwargs = dict(origin_m=(10 * MM, 8 * MM, 1.7 * MM), pitch_m=(0.5 * MM, 0.5 * MM, 1.0 * MM), shape=(15, 32, 40), supersample=2)
    for name, fn, kwargs, group in (
        ("copper layer at %.2f mm, %d samples per axis" % (pitch_mm, supersample), sample_plane_fill, plane_kwargs, "copper"),
        ("heat sink voxels (0.5, 0.5, 1.0) mm, 2 samples per axis", sample_volume_fill, voxel_kwargs, "sink"),
    ):
        points = int(np.prod(kwargs["shape"])) * supersample ** (2 if fn is sample_plane_fill else 3) if fn is sample_plane_fill else int(np.prod(kwargs["shape"])) * 8
        for solid in solids[group]:
            solid.tessellate()  # warm the cache so timings cover classification only
        reference, occ_timing = _timed(lambda: fn(solids[group], method="occ", **kwargs), repeats)
        paths: dict[str, Any] = {"occ": {"timing": occ_timing}}
        numpy_fill, numpy_timing = _timed(lambda: fn(solids[group], method="numpy", **kwargs), repeats)
        paths["numpy"] = {"timing": numpy_timing, "max_fill_difference_vs_occ": float(np.max(np.abs(numpy_fill - reference))), "speedup_vs_occ": occ_timing["median_ms"] / numpy_timing["median_ms"]}
        if native_classify_available():
            for count in threads:
                os.environ["PCB_NATIVE_THREADS"] = str(count)
                native_fill, native_timing = _timed(lambda: fn(solids[group], method="native", **kwargs), repeats)
                paths[f"native_threads{count}"] = {
                    "threads": count,
                    "timing": native_timing,
                    "max_fill_difference_vs_occ": float(np.max(np.abs(native_fill - reference))),
                    "max_fill_difference_vs_numpy": float(np.max(np.abs(native_fill - numpy_fill))),
                    "speedup_vs_occ": occ_timing["median_ms"] / native_timing["median_ms"],
                    "speedup_vs_numpy": numpy_timing["median_ms"] / native_timing["median_ms"],
                }
        cases.append({
            "case": name,
            "solids": len(solids[group]),
            "triangles": int(sum(s.tessellate().size for s in solids[group])),
            "sample_points": points,
            "covered_area_or_volume_fraction": float(reference.mean()),
            "paths": paths,
        })
    native_keys = [k for k in cases[0]["paths"] if k.startswith("native")]
    decision_key = native_keys[0] if native_keys else None
    agree = all(c["paths"][k]["max_fill_difference_vs_occ"] <= 1.0 / supersample**2 + 1e-12 for c in cases for k in c["paths"] if k != "occ")
    speedups = [c["paths"][decision_key]["speedup_vs_occ"] for c in cases] if decision_key else []
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
            "ocp": __import__("importlib.metadata").metadata.version("cadquery-ocp"),
            "cpu": _cpu_model(), "cpu_count": os.cpu_count(), "compiler": _compiler(),
            "native_extension_built": native_classify_available(), "thread_sweep": threads,
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"), "device": "cpu",
        },
        "definitions": {
            "occ": "BRepClass3d_SolidClassifier per point after a bounding-box prefilter (exact on the B-rep)",
            "numpy": "generalized winding number over the 5 um tessellation, chunked NumPy",
            "native": "the same winding number in C++ with OpenMP over points",
            "max_fill_difference": "largest per-cell difference of the sampled fill fraction against the occ path; one sample flipping changes a cell by 1/samples^d",
            "decision_threads": "the first thread count in the sweep",
        },
        "cases": cases,
        "decision": {
            "adopted": bool(decision_key and min(speedups) >= 2.0 and agree),
            "criteria": "native at the decision thread count at least 2x faster than occ on every case, and every winding-number path within one flipped sample per cell of occ",
            "minimum_speedup_vs_occ": min(speedups) if speedups else None,
            "within_one_sample_of_occ": agree,
            "decision_path": decision_key,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", default="1,4")
    parser.add_argument("--pitch-mm", type=float, default=0.25)
    parser.add_argument("--supersample", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results") / "geometry_classify_benchmark.json")
    args = parser.parse_args()
    report = run(args.repeats, [int(v) for v in args.threads.split(",")], args.pitch_mm, args.supersample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for case in report["cases"]:
        print(f"{case['case']}: {case['solids']} solids, {case['triangles']} triangles, {case['sample_points']} points")
        for name, path in case["paths"].items():
            extra = "" if name == "occ" else f" x{path['speedup_vs_occ']:.1f} vs occ, fill diff {path['max_fill_difference_vs_occ']:.3f}"
            print(f"  {name:16s} {path['timing']['median_ms']:9.1f} ms{extra}")
    print("decision:", json.dumps(report["decision"]))


if __name__ == "__main__":
    main()
