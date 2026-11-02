"""The port-impedance solve at the size of a converter's placement candidate: time, iterations, accuracy.

N2 of Kicad_PowerOpt's requirements: about 24 000 cells at 0.25 mm, 20 to 40
ports, 3 to 5 frequencies, at most 5 minutes a candidate on 8 threads.

The fixture is a synthetic four-layer boost stage, stated as
``port-impedance-problem/v1`` (no board is read): pours of VIN, SW, VOUT and
PGND on F.Cu with one-cell clearances, a PGND plane on In1.Cu with
anti-pads, VIN and VOUT pours on In2.Cu, PGND on B.Cu, plated vias with
barrels between them, and the pads of an input connector, input and output
capacitors, the inductor, the switch, the diode, the controller and the
output connector.  Each net's connector or switch pad is its reference;
every other pad is a port.

Each variant runs in its own process, stopped after ``--time-limit``.

    .venv/bin/python experiments/port_impedance_scale_benchmark.py --out docs/PORT_IMPEDANCE_SCALE_RESULTS.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PITCH_MM = 0.25
COLUMNS, ROWS = 128, 96  # 32 x 24 mm
LAYERS = [("F.Cu", 0.0), ("In1.Cu", -0.2), ("In2.Cu", -1.4), ("B.Cu", -1.6)]
COPPER_MM, RHO = 0.035, 1.724e-8
BARREL = {"outer_diameter_mm": 0.6, "drill_diameter_mm": 0.3}


def _box(x0, y0, x1, y1):
    return {(x, y) for x in range(x0, x1) for y in range(y0, y1)}


def board(frequencies=(1e6, 10e6, 30e6, 100e6)) -> dict:
    """The fixture as a ``port-impedance-problem/v1`` mapping."""
    # F.Cu pours, one-cell clearance between nets
    vin_f = _box(2, 8, 40, 60)
    sw_f = _box(41, 30, 70, 60)
    vout_f = _box(71, 8, 126, 60)
    pgnd_f = _box(2, 61, 126, 92) | _box(41, 8, 70, 29)
    # In1.Cu: PGND plane with anti-pads where VIN and VOUT vias pass
    pgnd_1 = _box(1, 1, 127, 95)
    # In2.Cu: VIN and VOUT pours
    vin_2 = _box(4, 10, 38, 40)
    vout_2 = _box(74, 10, 124, 40)
    # B.Cu: PGND under the output and input
    pgnd_b = _box(4, 62, 124, 90) | _box(4, 10, 30, 40)

    vias = []

    def via(name, x, y, upper, lower):
        vias.append({"name": name, "cell": {"x": x, "y": y}, "barrel": dict(BARREL),
                     "segments": [{"upper_layer": upper, "lower_layer": lower, "resistance_ohm": 5e-4,
                                   "length_mm": LAYERS_Z[upper] - LAYERS_Z[lower]}]})

    antipads = set()
    for k, (x, y) in enumerate((x, y) for x in range(8, 36, 6) for y in range(14, 38, 6)):  # VIN F <-> In2
        via(f"VIN_v{k}", x, y, "F.Cu", "In2.Cu")
        antipads |= _box(x - 1, y - 1, x + 2, y + 2)
    for k, (x, y) in enumerate((x, y) for x in range(78, 122, 6) for y in range(14, 38, 6)):  # VOUT F <-> In2
        via(f"VOUT_v{k}", x, y, "F.Cu", "In2.Cu")
        antipads |= _box(x - 1, y - 1, x + 2, y + 2)
    pgnd_1 -= antipads
    for k, (x, y) in enumerate((x, y) for x in range(6, 124, 8) for y in range(64, 90, 8)):  # PGND stitching
        via(f"PGND_s{k}", x, y, "F.Cu", "In1.Cu")
        if (x, y) in pgnd_b:
            via(f"PGND_b{k}", x, y, "In1.Cu", "B.Cu")
    for k, (x, y) in enumerate((x, y) for x in range(44, 68, 4) for y in range(11, 27, 4)):  # switch's PGND island
        via(f"PGND_q{k}", x, y, "F.Cu", "In1.Cu")

    copper = {"F.Cu": vin_f | sw_f | vout_f | pgnd_f, "In1.Cu": pgnd_1, "In2.Cu": vin_2 | vout_2, "B.Cu": pgnd_b}

    pads = []

    def pad(name, part, net, x0, y0, x1, y1, layer="F.Cu"):
        cells = sorted(_box(x0, y0, x1, y1))
        assert all(c in copper[layer] for c in cells), name
        pads.append({"name": name, "part": part, "net": net,
                     "cells": [{"layer": layer, "x": x, "y": y} for x, y in cells]})

    pad("J1.1", "J1", "VIN", 3, 40, 7, 48)
    pad("J1.2", "J1", "PGND", 3, 70, 7, 78)
    for k, x in enumerate((12, 20, 28, 36)):  # input capacitors across the VIN/PGND edge
        pad(f"CIN{k + 1}.1", f"CIN{k + 1}", "VIN", x, 56, x + 3, 60)
        pad(f"CIN{k + 1}.2", f"CIN{k + 1}", "PGND", x, 61, x + 3, 65)
    pad("L1.1", "L1", "VIN", 30, 20, 38, 30)
    pad("L1.2", "L1", "SW", 42, 32, 50, 42)
    pad("Q1.1", "Q1", "SW", 52, 31, 56, 34)  # drain, the SW reference
    pad("Q1.2", "Q1", "PGND", 50, 24, 53, 28)  # source pins
    pad("Q1.3", "Q1", "PGND", 55, 24, 58, 28)
    pad("D1.1", "D1", "SW", 62, 44, 68, 50)
    pad("D1.2", "D1", "VOUT", 72, 44, 78, 50)
    for k, x in enumerate((84, 92, 100, 108, 116, 120)):  # output capacitors
        pad(f"COUT{k + 1}.1", f"COUT{k + 1}", "VOUT", x, 55, x + 3, 60)
        pad(f"COUT{k + 1}.2", f"COUT{k + 1}", "PGND", x, 61, x + 3, 65)
    pad("U1.VIN", "U1", "VIN", 14, 30, 16, 32)
    pad("U1.PGND", "U1", "PGND", 60, 20, 62, 22)
    pad("U1.SW", "U1", "SW", 60, 52, 62, 54)
    pad("J2.1", "J2", "VOUT", 121, 40, 125, 48)
    pad("J2.2", "J2", "PGND", 121, 70, 125, 78)

    references = {"VIN": "J1.1", "PGND": "J1.2", "SW": "Q1.1", "VOUT": "J2.1"}
    ports = [{"pad": p["name"], "reference": references[p["net"]]} for p in pads
             if p["name"] not in references.values()]
    return {
        "schema": "port-impedance-problem/v1",
        "name": "synthetic_boost_4layer",
        "grid": {"rows": ROWS, "columns": COLUMNS, "pitch_mm": PITCH_MM},
        "layers": [{"name": n, "order": i, "center_z_mm": z, "thickness_mm": COPPER_MM, "resistivity_ohm_m": RHO}
                   for i, (n, z) in enumerate(LAYERS)],
        "copper_by_layer": {n: [{"x": x, "y": y} for x, y in sorted(cells)] for n, cells in copper.items()},
        "vertical_connections": vias,
        "frequencies_hz": list(frequencies),
        "pads": pads,
        "ports": ports,
    }


LAYERS_Z = {n: z for n, z in LAYERS}


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


VARIANTS = {
    # the defaults before warm starts: preconditioner "auto" (block at this size), every frequency from zero
    "auto-cold-1e-9": {"preconditioner": "auto", "relative_tolerance": 1e-9, "warm_start": False},
    "near-cold-1e-9": {"preconditioner": "near", "relative_tolerance": 1e-9, "warm_start": False},
    "near-warm-1e-9": {"preconditioner": "near", "relative_tolerance": 1e-9, "warm_start": True},
    "near-cold-1e-6": {"preconditioner": "near", "relative_tolerance": 1e-6, "warm_start": False},
    "near-warm-1e-6": {"preconditioner": "near", "relative_tolerance": 1e-6, "warm_start": True},
    # the device path (not in the default list): ports one after another, warm
    "cuda-warm-1e-6": {"preconditioner": "near", "relative_tolerance": 1e-6, "warm_start": True,
                       "execution_backend": "cuda"},
}
REFERENCE = "near-cold-1e-9"
CANDIDATE, BASELINE = "near-warm-1e-6", "near-cold-1e-6"
TARGET_SECONDS = 300.0


def _run_variant(name: str, frequencies: list[float], threads: int, filaments: int, out: Path) -> None:
    from electrical.sheet_peec import solve_port_impedance_problem
    from electrical.threads import thread_budget_scope

    value = board(tuple(frequencies))
    settings = {"maximum_filaments": filaments, **VARIANTS[name]}
    started = time.perf_counter()
    with thread_budget_scope(threads):
        result = solve_port_impedance_problem(value, settings)
    device = "CPU"
    if settings.get("execution_backend") == "cuda":
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(int(settings.get("device_id", 0)))
        device = f"{props['name'].decode()} (CuPy {cp.__version__})"
    out.write_text(json.dumps({
        "variant": name, "settings": settings, "device": device, "total_seconds": time.perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "frequencies": result.metrics["frequencies"], "ports": list(result.ports),
        "z_real": [z.z_ohm.real.tolist() for z in result.impedances],
        "z_imag": [z.z_ohm.imag.tolist() for z in result.impedances],
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--frequencies", type=float, nargs="+", default=[1e6, 10e6, 30e6, 100e6])
    parser.add_argument("--filaments", type=int, default=1)
    parser.add_argument("--variants", nargs="+", default=[v for v in VARIANTS if not v.startswith("cuda")])
    parser.add_argument("--save-reference-z", help="write the reference variant's Z to this .npz")
    parser.add_argument("--reference-z", help="compare with the Z of an earlier reference run (.npz) instead")
    parser.add_argument("--time-limit", type=float, default=900.0, help="seconds a variant may take before it is stopped")
    parser.add_argument("--variant", help=argparse.SUPPRESS)
    parser.add_argument("--variant-out", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.variant:
        _run_variant(args.variant, args.frequencies, args.threads, args.filaments, Path(args.variant_out))
        return

    import subprocess
    import tempfile

    value = board(tuple(args.frequencies))
    rows = {}
    for name in args.variants:
        with tempfile.TemporaryDirectory() as scratch:
            out = Path(scratch) / "variant.json"
            command = [sys.executable, __file__, "--out", args.out, "--variant", name, "--variant-out", str(out),
                       "--threads", str(args.threads), "--filaments", str(args.filaments),
                       "--frequencies", *map(str, args.frequencies)]
            started = time.perf_counter()
            try:
                subprocess.run(command, check=True, timeout=args.time_limit)
                rows[name] = json.loads(out.read_text())
            except subprocess.TimeoutExpired:
                rows[name] = {"variant": name, "settings": VARIANTS[name], "finished": False,
                              "stopped_after_seconds": time.perf_counter() - started}
        rows[name].setdefault("finished", True)
        summary = {k: v for k, v in rows[name].items() if k not in ("z_real", "z_imag", "ports")}
        print(json.dumps(summary), flush=True)

    reference = rows.get(REFERENCE)
    z_ref = None
    if reference and reference["finished"]:
        z_ref = np.array(reference["z_real"]) + 1j * np.array(reference["z_imag"])
        if args.save_reference_z:
            np.savez(args.save_reference_z, z=z_ref, frequencies_hz=np.array(args.frequencies))
    elif args.reference_z:
        stored = np.load(args.reference_z)
        if list(stored["frequencies_hz"]) != list(args.frequencies):
            raise SystemExit("the stored reference was solved at other frequencies")
        z_ref = stored["z"]
    if z_ref is not None:
        for row in rows.values():
            if row["finished"]:
                z = np.array(row["z_real"]) + 1j * np.array(row["z_imag"])
                row["max_z_difference_vs_reference"] = [
                    float(np.abs(z[k] - z_ref[k]).max() / np.abs(z_ref[k]).max()) for k in range(len(z))]
                row["max_r_difference_vs_reference"] = [
                    float(np.abs(z[k].real - z_ref[k].real).max() / np.abs(z_ref[k].real).max()) for k in range(len(z))]
    for row in rows.values():
        row.pop("z_real", None)
        row.pop("z_imag", None)

    candidate, baseline = rows.get(CANDIDATE), rows.get(BASELINE)
    decision = None
    if candidate and baseline and candidate["finished"] and baseline["finished"]:
        same = max(candidate.get("max_z_difference_vs_reference", [np.inf])) <= 1e-5
        faster = candidate["total_seconds"] < baseline["total_seconds"]
        decision = {
            "warm_start_adopted": bool(same and faster),
            "criteria": ("warm starts are adopted when, at the same preconditioner and tolerance, they solve the "
                         "four frequencies faster than cold starts and their Z stays within 1e-5 (largest entry) "
                         f"of {REFERENCE}"),
            "candidate": CANDIDATE, "baseline": BASELINE,
            "candidate_seconds": candidate["total_seconds"], "baseline_seconds": baseline["total_seconds"],
            "meets_target_seconds": TARGET_SECONDS,
            "candidate_meets_target": candidate["total_seconds"] <= TARGET_SECONDS,
        }
    cells = sum(len(c) for c in value["copper_by_layer"].values())
    result = {
        "benchmark": "port-impedance solve at a placement candidate's size",
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"cpu": _cpu_model(), "cpu_count": os.cpu_count(), "threads": args.threads,
                        "platform": platform.platform(), "python": platform.python_version(),
                        "numpy": np.__version__, "scipy": __import__("scipy").__version__, "device": "CPU",
                        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                        "omp_num_threads": os.environ.get("OMP_NUM_THREADS")},
        "fixture": {"name": value["name"], "pitch_mm": PITCH_MM, "cells": cells,
                    "cells_by_layer": {k: len(v) for k, v in value["copper_by_layer"].items()},
                    "ports": len(value["ports"]), "pads": len(value["pads"]),
                    "vias": len(value["vertical_connections"]), "frequencies_hz": args.frequencies,
                    "maximum_filaments": args.filaments},
        "time_limit_seconds": args.time_limit,
        "reference": REFERENCE if not args.reference_z else f"{REFERENCE} (stored: {Path(args.reference_z).name})",
        "rows": list(rows.values()),
        "decision": decision,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
