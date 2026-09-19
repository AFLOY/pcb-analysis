"""Acceptance of backward-Euler transient conduction on the steady operator.

1. Lumped cooling of a nearly isothermal copper block: the marched maximum
   against the exact discrete backward-Euler law (checks the lumping) and
   against the analytic exponential for several uniform steps (checks the
   first-order time error halves with the step).
2. A locally heated three-slab board marched from ambient to steady state
   with a geometric schedule and with uniform steps: steps, wall time and the
   final difference from the steady solve, on the array and C++ paths.

Writes a JSON with ``environment`` and ``decision``; the adopted copy lives at
``docs/THERMAL_TRANSIENT_RESULTS.json``.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/thermal_transient_acceptance.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermal.matrix_free_mpir_fem import (  # noqa: E402
    ConvectionBoundary,
    LayeredThermalMesh,
    ThermalConductionProblem,
    TimeSchedule,
    solve_thermal_conduction,
    solve_thermal_transient,
)
from thermal.matrix_free_mpir_fem.native_hex import native_available  # noqa: E402

AMBIENT = 300.0
COPPER_RHO_C = 3.45e6
FR4_RHO_C = 1.8e6


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
        return "unavailable"


def _environment() -> dict[str, Any]:
    try:
        import cupy

        cupy_version = cupy.__version__
    except Exception:  # noqa: BLE001
        cupy_version = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cupy": cupy_version,
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "compiler": _compiler(),
        "native_extension_built": native_available(),
        "PCB_NATIVE_THREADS": os.environ.get("PCB_NATIVE_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def lumped_cooling() -> dict[str, Any]:
    thickness, h, start, end = 2.0e-3, 20.0, 350.0, 400.0
    mesh = LayeredThermalMesh(
        (thickness,), 1e-3, 1e-3, 4000.0, element_shape=(4, 5), volumetric_heat_capacity_j_per_m3_k=COPPER_RHO_C
    )
    problem = ThermalConductionProblem(mesh, convection=(ConvectionBoundary("top", h, AMBIENT),))
    tau = COPPER_RHO_C * thickness / h
    analytic = AMBIENT + (start - AMBIENT) * np.exp(-end / tau)
    cases = []
    for step_s in (40.0, 20.0, 10.0, 5.0, 2.5):
        schedule = TimeSchedule.uniform(step_s, end)
        wall = time.perf_counter()
        solution = solve_thermal_transient(problem, schedule, initial_temperature_k=start, store="final")
        wall = time.perf_counter() - wall
        discrete = start
        worst_lumping = 0.0
        for record in solution.history:
            discrete = AMBIENT + (discrete - AMBIENT) / (1.0 + record.step_s / tau)
            worst_lumping = max(worst_lumping, abs(record.max_temperature_k - discrete))
        cases.append(
            {
                "step_s": step_s,
                "steps": len(solution.history),
                "final_k": solution.history[-1].max_temperature_k,
                "analytic_k": analytic,
                "time_error_k": solution.history[-1].max_temperature_k - analytic,
                "max_deviation_from_discrete_backward_euler_k": worst_lumping,
                "max_heat_balance_error_w": max(abs(r.heat_balance_error_w) for r in solution.history),
                "wall_ms": wall * 1e3,
            }
        )
    ratios = [cases[i]["time_error_k"] / cases[i + 1]["time_error_k"] for i in range(len(cases) - 1)]
    return {"tau_s": tau, "end_s": end, "start_k": start, "cases": cases, "error_ratio_per_halving": ratios}


def board_to_steady() -> dict[str, Any]:
    rows, cols = 40, 60
    mesh = LayeredThermalMesh(
        (35e-6, 1.5e-3, 35e-6), 0.5e-3, 0.5e-3, (385.0, 0.8, 385.0), element_shape=(rows, cols),
        through_plane_conductivity_w_per_m_k=(385.0, 0.3, 385.0),
        volumetric_heat_capacity_j_per_m3_k=(COPPER_RHO_C, FR4_RHO_C, COPPER_RHO_C),
    )
    heat = np.zeros((3, rows, cols))
    heat[2, 18:22, 28:32] = 0.1  # 1.6 W in a 2 x 2 mm patch of the top copper
    problem = ThermalConductionProblem(
        mesh,
        convection=(ConvectionBoundary("top", 10.0, AMBIENT), ConvectionBoundary("bottom", 10.0, AMBIENT)),
        element_heat_w=heat,
    )
    wall = time.perf_counter()
    steady = solve_thermal_conduction(problem)
    steady_wall = time.perf_counter() - wall
    rise = steady.max_temperature_k - AMBIENT
    runs = []
    for label, schedule, native in (
        ("geometric 0.01 s x1.6, array", TimeSchedule.geometric(0.01, 20000.0, growth=1.6), False),
        ("geometric 0.01 s x1.6, native", TimeSchedule.geometric(0.01, 20000.0, growth=1.6), True),
        ("uniform 10 s, array", TimeSchedule.uniform(10.0, 2000.0), False),
    ):
        if native and not native_available():
            continue
        wall = time.perf_counter()
        transient = solve_thermal_transient(
            problem, schedule, until_steady=True, steady_tolerance_k_per_s=2e-5, store="final", native=native
        )
        wall = time.perf_counter() - wall
        runs.append(
            {
                "schedule": label,
                "steps_taken": len(transient.history),
                "steps_in_schedule": len(schedule.steps_s),
                "end_time_s": transient.history[-1].time_s,
                "reached_steady": transient.reached_steady,
                "wall_ms": wall * 1e3,
                "wall_per_step_ms": wall * 1e3 / len(transient.history),
                "inner_iterations_total": sum(r.inner_iterations for r in transient.history),
                "final_max_diff_from_steady_k": float(np.nanmax(np.abs(transient.final_temperature_k - steady.temperature_k))),
                "max_heat_balance_error_w": max(abs(r.heat_balance_error_w) for r in transient.history),
                "time_to_half_rise_s": next(
                    (r.time_s for r in transient.history if r.max_temperature_k - AMBIENT >= 0.5 * rise), None
                ),
            }
        )
    return {
        "nodes": int(steady.temperature_k.size),
        "steady": {"wall_ms": steady_wall * 1e3, "max_temperature_k": steady.max_temperature_k, "rise_k": rise,
                   "inner_iterations": steady.solve.inner_iterations},
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/thermal_transient_acceptance.json"))
    args = parser.parse_args()
    lumped = lumped_cooling()
    board = board_to_steady()
    lumped_ok = all(c["max_deviation_from_discrete_backward_euler_k"] < 5e-4 for c in lumped["cases"]) and all(
        1.8 < r < 2.2 for r in lumped["error_ratio_per_halving"]
    )
    board_ok = all(
        r["reached_steady"] and r["final_max_diff_from_steady_k"] < 2e-3 * board["steady"]["rise_k"] for r in board["runs"]
    )
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": _environment(),
        "lumped_cooling": lumped,
        "board_to_steady": board,
        "decision": {
            "backward_euler_on_steady_operator": "adopted" if (lumped_ok and board_ok) else "not adopted",
            "criteria": {
                "lumping": "marched maximum within 5e-4 K of the exact discrete backward-Euler law",
                "time_order": "analytic error ratio per step halving in (1.8, 2.2)",
                "steady_state": "geometric and uniform marches end within 2e-3 of the rise from the steady solve",
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for c in lumped["cases"]:
        print(f"lumped dt={c['step_s']:5.1f} s steps={c['steps']:3d} time_err={c['time_error_k']:+.4f} K lumping_dev={c['max_deviation_from_discrete_backward_euler_k']:.1e} K wall={c['wall_ms']:.0f} ms")
    print("error ratios per halving:", [round(r, 3) for r in lumped["error_ratio_per_halving"]])
    print(f"board steady: rise={board['steady']['rise_k']:.2f} K wall={board['steady']['wall_ms']:.0f} ms nodes={board['nodes']}")
    for r in board["runs"]:
        print(f"  {r['schedule']:32s} steps={r['steps_taken']:3d}/{r['steps_in_schedule']} end={r['end_time_s']:.0f} s steady={r['reached_steady']} diff={r['final_max_diff_from_steady_k']:.2e} K wall={r['wall_ms']:.0f} ms ({r['wall_per_step_ms']:.0f}/step) t_half={r['time_to_half_rise_s']}")
    print("decision:", json.dumps(report["decision"]["backward_euler_on_steady_operator"]))


if __name__ == "__main__":
    main()
