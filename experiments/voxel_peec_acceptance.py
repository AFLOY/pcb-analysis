"""Acceptance of the 3D voxel PEEC path for thick conductors.

A copper busbar (20 x 4 x 2 mm) solved with PyPEEC through
electrical.dice_peec.solve_voxel_peec: DC resistance against the analytic
value at several voxel pitches, AC resistance rising with frequency as the
skin depth shrinks below the bar, and the CUDA executor against the CPU
solve.  Writes a JSON with ``environment`` and ``decision``; the adopted copy
lives at ``docs/VOXEL_PEEC_RESULTS.json``.

    .venv/bin/python experiments/voxel_peec_acceptance.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from electrical.dice_peec import VoxelConductorProblem, VoxelTerminal, solve_voxel_peec  # noqa: E402
from electrical.dice_peec.skin_filaments import skin_depth_m  # noqa: E402

RHO = 1.68e-8
LENGTH, WIDTH, THICK = 20.0e-3, 4.0e-3, 2.0e-3


def busbar(pitch: float, frequency_hz: float, current: float = 10.0) -> VoxelConductorProblem:
    nx, ny, nz = round(LENGTH / pitch), round(WIDTH / pitch), round(THICK / pitch)
    conductor = np.ones((nz, ny, nx), dtype=bool)
    src = np.zeros_like(conductor)
    src[:, :, :2] = True
    ref = np.zeros_like(conductor)
    ref[:, :, -2:] = True
    return VoxelConductorProblem(conductor, (pitch,) * 3, (VoxelTerminal("src", src, current), VoxelTerminal("ref", ref)), frequency_hz=frequency_hz, resistivity_ohm_m=RHO, name="busbar")


def timed_solve(problem: VoxelConductorProblem, backend: str) -> tuple[Any, float]:
    start = time.perf_counter()
    solution = solve_voxel_peec(problem, backend=backend)
    return solution, time.perf_counter() - start


def run(pitches: list[float], frequencies: list[float], cuda: bool) -> dict[str, Any]:
    dc_cases = []
    for pitch in pitches:
        problem = busbar(pitch, 0.0)
        solution, seconds = timed_solve(problem, "cpu")
        nx = problem.shape[2]
        analytic = RHO * (nx - 2) * pitch / (WIDTH * THICK)
        r = solution.impedance_ohm["src"].real
        dc_cases.append({
            "pitch_mm": pitch * 1e3, "voxels": int(problem.conductor.sum()), "seconds": seconds,
            "resistance_ohm": r, "analytic_ohm": analytic, "relative_error": abs(r - analytic) / analytic,
            "loss_w": solution.joule_loss_w, "loss_vs_i2r": abs(solution.joule_loss_w - 100.0 * r) / (100.0 * r),
            "converged": solution.converged, "iterations": solution.iterations,
        })
    pitch = pitches[0]
    ac_cases = []
    for frequency in frequencies:
        problem = busbar(pitch, frequency)
        solution, seconds = timed_solve(problem, "cpu")
        z = solution.impedance_ohm["src"]
        delta = skin_depth_m(frequency, RHO)
        ac_cases.append({
            "frequency_hz": frequency, "skin_depth_mm": delta * 1e3, "thickness_over_skin_depth": THICK / delta if delta else 0.0,
            "resistance_ohm": z.real, "reactance_ohm": z.imag, "resistance_over_dc": z.real / dc_cases[0]["resistance_ohm"],
            "loss_w": solution.joule_loss_w, "loss_vs_half_i2r": abs(solution.joule_loss_w - 50.0 * z.real) / (50.0 * z.real),
            "seconds": seconds, "converged": solution.converged, "iterations": solution.iterations,
        })
    cuda_case: dict[str, Any] | None = None
    if cuda:
        problem = busbar(pitch, frequencies[len(frequencies) // 2])
        cpu, cpu_s = timed_solve(problem, "cpu")
        try:
            gpu, gpu_s = timed_solve(problem, "cuda")
            import cupy

            cuda_case = {
                "frequency_hz": problem.frequency_hz, "device": cupy.cuda.runtime.getDeviceProperties(0)["name"].decode(), "cupy": cupy.__version__,
                "cpu_seconds": cpu_s, "cuda_seconds": gpu_s,
                "impedance_relative_difference": abs(gpu.impedance_ohm["src"] - cpu.impedance_ohm["src"]) / abs(cpu.impedance_ohm["src"]),
                "loss_relative_difference": abs(gpu.joule_loss_w - cpu.joule_loss_w) / cpu.joule_loss_w,
                "converged": gpu.converged,
            }
        except Exception as exc:  # pragma: no cover - device dependent
            cuda_case = {"error": f"{type(exc).__name__}: {exc}"}
    import pypeec

    r_ac = [c["resistance_over_dc"] for c in ac_cases]
    decision = {
        "adopted": bool(
            all(c["relative_error"] <= 0.02 and c["converged"] for c in dc_cases)
            and all(c["loss_vs_i2r"] <= 1e-3 for c in dc_cases)
            and all(a <= b for a, b in zip(r_ac, r_ac[1:]))
            and all(c["converged"] and c["loss_vs_half_i2r"] <= 0.02 for c in ac_cases)
            and (cuda_case is None or ("error" not in cuda_case and cuda_case["impedance_relative_difference"] <= 1e-4))
        ),
        "criteria": "DC resistance within 2 % of rho L / A between the terminal midplanes at every pitch with loss = I^2 R to 1e-3; AC resistance non-decreasing with frequency with time-averaged loss = |I|^2 R / 2 within 2 %; CUDA impedance within 1e-4 of the CPU solve when a device is present",
        "max_dc_relative_error": max(c["relative_error"] for c in dc_cases),
        "ac_resistance_over_dc": r_ac,
        "cuda_checked": cuda_case is not None and "error" not in cuda_case,
    }
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__, "pypeec": pypeec.__version__, "cpu_count": os.cpu_count(), "device": cuda_case.get("device", "cpu") if cuda_case else "cpu"},
        "busbar": {"length_mm": LENGTH * 1e3, "width_mm": WIDTH * 1e3, "thickness_mm": THICK * 1e3, "resistivity_ohm_m": RHO, "terminals": "two voxel slabs at the bar ends, 10 A into src, ref at 0 V"},
        "dc": dc_cases, "ac": ac_cases, "cuda": cuda_case, "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitches-mm", default="0.5,0.25")
    parser.add_argument("--frequencies", default="1e3,1e4,1e5,1e6")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results") / "voxel_peec_acceptance.json")
    args = parser.parse_args()
    report = run([float(v) * 1e-3 for v in args.pitches_mm.split(",")], [float(v) for v in args.frequencies.split(",")], not args.no_cuda)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for c in report["dc"]:
        print(f"DC pitch {c['pitch_mm']:.2f} mm: {c['voxels']} voxels, R={c['resistance_ohm']:.4e} (analytic {c['analytic_ohm']:.4e}, err {c['relative_error']:.2e}), {c['seconds']:.1f} s")
    for c in report["ac"]:
        print(f"AC {c['frequency_hz']:g} Hz: t/delta={c['thickness_over_skin_depth']:.1f} R/Rdc={c['resistance_over_dc']:.3f} X={c['reactance_ohm']:.3e} loss err {c['loss_vs_half_i2r']:.2e} {c['seconds']:.1f} s")
    print("CUDA:", report["cuda"])
    print("decision:", json.dumps(report["decision"]))


if __name__ == "__main__":
    main()
