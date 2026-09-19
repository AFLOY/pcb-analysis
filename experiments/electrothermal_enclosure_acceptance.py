"""Acceptance of radiation boundaries and of σ(T) around the board/body interface.

Two items on synthetic arrays (no CAD, no GPU needed):

1. Radiation: a uniformly heated plate radiating from its top face to the
   ambient, against the analytic surface temperature
   ``(T_amb⁴ + P / (ε σ A))^¼``; records the Newton iteration count, the
   error and the heat budget for several powers.
2. σ(T) enclosure: a two-layer current loop with an aluminium block on a TIM,
   solved as one masked electro-thermal mesh (reference) and as board and
   body through the contact map inside the resistivity loop, with and without
   radiation. Records outer iterations, interface iterations per outer step,
   wall time, the temperature and loss-ratio differences.

Writes a JSON with ``environment`` and ``decision``; the adopted copy lives at
``docs/ELECTROTHERMAL_ENCLOSURE_RESULTS.json``.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/electrothermal_enclosure_acceptance.py
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

from electrical.matrix_free_mpir_fem import (  # noqa: E402
    CurrentTerminal,
    LayeredPCBMesh,
    PCBConductionProblem,
    ViaConnection,
)
from multiphysics.staggered_coupling import (  # noqa: E402
    BodyContact,
    CouplingConfig,
    ElectroThermalEnclosureScenario,
    ElectroThermalScenario,
    InterfaceCouplingConfig,
    run_electro_thermal,
    run_electro_thermal_enclosure,
)
from thermal.matrix_free_mpir_fem import (  # noqa: E402
    STEFAN_BOLTZMANN_W_PER_M2_K4 as SIGMA,
    ConvectionBoundary,
    ExposedFaceConvection,
    ExposedFaceRadiation,
    LayeredThermalMesh,
    RadiationBoundary,
    ThermalConductionProblem,
    planar_contact_map,
    solve_thermal_conduction,
)
from thermal.matrix_free_mpir_fem.native_hex import native_available  # noqa: E402

AMBIENT = 298.15
PITCH = 0.5e-3
ROWS, COLS = 6, 24
BOARD_SLABS = (35e-6, 1.5e-3, 35e-6)
SINK_ROWS, SINK_COLS = slice(1, 5), slice(6, 18)
SINK_SLABS = (1.0e-3,) * 3
SINK_K = 200.0
TIM_K, TIM_T = 1.0e-2, 1.0e-6
H = 12.0
CURRENT = 6.0
EMISSIVITY = 0.9


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


# ---------------------------------------------------------------- radiation
def radiation_plate() -> dict[str, Any]:
    rows, cols, pitch, thickness, k = 8, 10, 1.0e-3, 1.0e-3, 200.0
    area = rows * cols * pitch * pitch
    mesh = LayeredThermalMesh((thickness,), pitch, pitch, k, element_shape=(rows, cols))
    cases = []
    for power_w in (0.05, 0.6, 3.0):
        heat = np.full((1, rows, cols), power_w / (rows * cols))
        exact = (AMBIENT**4 + power_w / (EMISSIVITY * SIGMA * area)) ** 0.25
        for kind, boundary in (
            ("RadiationBoundary top", RadiationBoundary("top", EMISSIVITY, AMBIENT)),
            ("ExposedFaceRadiation +z", ExposedFaceRadiation(EMISSIVITY, AMBIENT, directions=("+z",))),
        ):
            problem = ThermalConductionProblem(mesh, radiation=(boundary,), element_heat_w=heat)
            start = time.perf_counter()
            solution = solve_thermal_conduction(problem)
            wall = time.perf_counter() - start
            surface = float(np.mean(solution.temperature_k[1]))
            cases.append(
                {
                    "boundary": kind,
                    "power_w": power_w,
                    "exact_surface_k": exact,
                    "surface_rise_k": exact - AMBIENT,
                    "solved_surface_k": surface,
                    "surface_error_k": surface - exact,
                    "surface_error_rel": (surface - exact) / exact,
                    "radiative_heat_w": float(solution.radiative_heat_w[0]),
                    "heat_balance_error_w": solution.heat_balance_error_w,
                    "newton_iterations": solution.radiation_iterations,
                    "newton_converged": solution.radiation_converged,
                    "wall_ms": wall * 1e3,
                }
            )
    return {"plate": {"rows": rows, "cols": cols, "area_m2": area, "emissivity": EMISSIVITY, "ambient_k": AMBIENT}, "cases": cases}


# --------------------------------------------------------------- σ(T) loop
def _loop_board() -> PCBConductionProblem:
    active = np.zeros((2, ROWS, COLS), dtype=bool)
    active[:, 1:5, :] = True
    mesh = LayeredPCBMesh(active, (35e-6, 35e-6), PITCH, PITCH)
    node_rows = tuple(range(1, 6))
    vias = tuple(ViaConnection((0, r, c), (1, r, c), 1.5e-3) for r in node_rows[1:-1] for c in (COLS - 1, COLS))
    return PCBConductionProblem(
        mesh,
        (
            CurrentTerminal(tuple((1, r, 0) for r in node_rows), CURRENT, "source"),
            CurrentTerminal(tuple((0, r, 0) for r in node_rows), -CURRENT, "return"),
        ),
        reference_node=(0, 1, 0),
        vias=vias,
    )


def _board_conductivity(problem: PCBConductionProblem) -> tuple[np.ndarray, np.ndarray]:
    active = problem.mesh.element_active
    k = np.full((3, ROWS, COLS), 0.8)
    k[0] = np.where(active[0], 385.0, 0.8)
    k[2] = np.where(active[1], 385.0, 0.8)
    kz = k.copy()
    kz[1] = 0.3
    return k, kz


def _partitioned(radiating: bool) -> ElectroThermalEnclosureScenario:
    problem = _loop_board()
    k, kz = _board_conductivity(problem)
    board_mesh = LayeredThermalMesh(BOARD_SLABS, PITCH, PITCH, k, through_plane_conductivity_w_per_m_k=kz)
    covered = np.zeros((ROWS, COLS), dtype=bool)
    covered[SINK_ROWS, SINK_COLS] = True
    options: dict[str, Any] = dict(
        convection=(
            ExposedFaceConvection(H, AMBIENT, directions=("-z", "-x", "+x", "-y", "+y")),
            ConvectionBoundary("top", np.where(covered, 0.0, H), AMBIENT),
        ),
        conductivity_reference_temperature_k=293.15,
    )
    if radiating:
        emissivity = np.full((3, ROWS, COLS), EMISSIVITY)
        emissivity[2, covered] = 0.0
        options["radiation"] = (ExposedFaceRadiation(emissivity, AMBIENT, directions=("+z", "-x", "+x", "-y", "+y")),)
    electro_thermal = ElectroThermalScenario(problem, board_mesh, (0, 2), **options)
    sink_mesh = LayeredThermalMesh(SINK_SLABS, PITCH, PITCH, SINK_K, element_shape=(4, 12))
    sink_options: dict[str, Any] = dict(
        convection=(ExposedFaceConvection(H, AMBIENT, directions=("+z", "-x", "+x", "-y", "+y")),)
    )
    if radiating:
        sink_options["radiation"] = (ExposedFaceRadiation(EMISSIVITY, AMBIENT, directions=("+z", "-x", "+x", "-y", "+y")),)
    sink = ThermalConductionProblem(sink_mesh, **sink_options)
    contact = planar_contact_map(
        board_mesh, sink_mesh, board_side="top", board_origin_m=(0.0, 0.0),
        body_origin_m=(SINK_COLS.start * PITCH, SINK_ROWS.start * PITCH),
        conductance_per_area_w_per_m2_k=TIM_K / TIM_T,
    )
    return ElectroThermalEnclosureScenario(electro_thermal, (BodyContact(sink, contact, name="sink"),))


def _monolithic(radiating: bool) -> ElectroThermalScenario:
    problem = _loop_board()
    k_board, kz_board = _board_conductivity(problem)
    slabs = BOARD_SLABS + (TIM_T,) + SINK_SLABS
    active = np.ones((len(slabs), ROWS, COLS), dtype=bool)
    active[3:] = False
    active[3:, SINK_ROWS, SINK_COLS] = True
    k = np.ones(active.shape)
    kz = np.ones(active.shape)
    k[:3], kz[:3] = k_board, kz_board
    k[3] = kz[3] = TIM_K
    k[4:] = kz[4:] = SINK_K
    mesh = LayeredThermalMesh(slabs, PITCH, PITCH, k, through_plane_conductivity_w_per_m_k=kz, active=active)
    options: dict[str, Any] = dict(convection=(ExposedFaceConvection(H, AMBIENT),), conductivity_reference_temperature_k=293.15)
    if radiating:
        options["radiation"] = (ExposedFaceRadiation(EMISSIVITY, AMBIENT, directions=("+z", "-x", "+x", "-y", "+y")),)
    return ElectroThermalScenario(problem, mesh, (0, 2), **options)


def sigma_t_enclosure(radiating: bool) -> dict[str, Any]:
    start = time.perf_counter()
    reference = run_electro_thermal(_monolithic(radiating), config=CouplingConfig(temperature_tolerance_k=1e-4))
    reference_wall = time.perf_counter() - start
    start = time.perf_counter()
    result = run_electro_thermal_enclosure(
        _partitioned(radiating),
        config=CouplingConfig(temperature_tolerance_k=1e-4),
        interface=InterfaceCouplingConfig(temperature_tolerance_k=1e-5),
    )
    wall = time.perf_counter() - start
    rise = reference.thermal.max_temperature_k - AMBIENT
    board_error = float(np.nanmax(np.abs(result.thermal.board.temperature_k - reference.thermal.temperature_k[:4])))
    sink_reference = reference.thermal.temperature_k[4:, SINK_ROWS.start : SINK_ROWS.stop + 1, SINK_COLS.start : SINK_COLS.stop + 1]
    sink_error = float(np.nanmax(np.abs(result.thermal.bodies[0].temperature_k - sink_reference)))
    return {
        "radiating": radiating,
        "current_a": CURRENT,
        "reference_monolithic": {
            "nodes": int(reference.thermal.temperature_k.size),
            "converged": reference.converged,
            "outer_iterations": reference.iterations,
            "wall_ms": reference_wall * 1e3,
            "max_temperature_k": reference.thermal.max_temperature_k,
            "rise_k": rise,
            "cold_joule_loss_w": reference.cold_joule_loss_w,
            "joule_loss_w": reference.electrical.joule_loss_w,
            "loss_increase_ratio": reference.loss_increase_ratio,
            "radiative_heat_w": float(np.sum(reference.thermal.radiative_heat_w)),
        },
        "partitioned": {
            "board_nodes": int(result.thermal.board.temperature_k.size),
            "body_nodes": int(result.thermal.bodies[0].temperature_k.size),
            "converged": result.converged,
            "outer_iterations": result.iterations,
            "interface_iterations_per_outer": [step.interface_iterations for step in result.history],
            "relaxation_per_outer": [step.relaxation for step in result.history],
            "wall_ms": wall * 1e3,
            "max_temperature_k": result.thermal.board.max_temperature_k,
            "cold_joule_loss_w": result.cold_joule_loss_w,
            "joule_loss_w": result.electrical.joule_loss_w,
            "loss_increase_ratio": result.loss_increase_ratio,
            "interface_heat_w": result.thermal.interface_heat_w,
            "board_radiative_heat_w": float(np.sum(result.thermal.board.radiative_heat_w)),
            "body_radiative_heat_w": float(np.sum(result.thermal.bodies[0].radiative_heat_w)),
        },
        "board_max_diff_k": board_error,
        "sink_max_diff_k": sink_error,
        "board_diff_over_rise": board_error / rise,
        "sink_diff_over_rise": sink_error / rise,
        "loss_ratio_rel_diff": abs(result.loss_increase_ratio - reference.loss_increase_ratio) / reference.loss_increase_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/electrothermal_enclosure_acceptance.json"))
    args = parser.parse_args()

    radiation = radiation_plate()
    sigma = [sigma_t_enclosure(False), sigma_t_enclosure(True)]

    radiation_ok = all(abs(c["surface_error_rel"]) < 1e-6 and c["newton_converged"] for c in radiation["cases"])
    sigma_ok = all(
        s["reference_monolithic"]["converged"] and s["partitioned"]["converged"]
        and s["board_diff_over_rise"] < 3e-3 and s["sink_diff_over_rise"] < 3e-3 and s["loss_ratio_rel_diff"] < 2e-3
        for s in sigma
    )
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": _environment(),
        "radiation_plate": radiation,
        "sigma_t_enclosure": sigma,
        "decision": {
            "radiation_newton_robin": "adopted" if radiation_ok else "not adopted",
            "sigma_t_enclosure_loop": "adopted" if sigma_ok else "not adopted",
            "criteria": {
                "radiation_surface_error_rel": "< 1e-6 against the analytic plate for every power",
                "board_and_sink_diff_over_rise": "< 3e-3 against the monolithic σ(T) mesh",
                "loss_ratio_rel_diff": "< 2e-3 against the monolithic σ(T) mesh",
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for case in radiation["cases"]:
        print(f"radiation {case['boundary']:26s} P={case['power_w']:.2f} W rise={case['surface_rise_k']:.1f} K err={case['surface_error_rel']:.1e} newton={case['newton_iterations']}")
    for s in sigma:
        print(
            f"sigma(T) radiating={s['radiating']} ref_iters={s['reference_monolithic']['outer_iterations']} part_iters={s['partitioned']['outer_iterations']} "
            f"if={s['partitioned']['interface_iterations_per_outer']} rise={s['reference_monolithic']['rise_k']:.1f} K "
            f"board_diff={s['board_max_diff_k']:.2e} sink_diff={s['sink_max_diff_k']:.2e} loss_ratio={s['partitioned']['loss_increase_ratio']:.4f}/{s['reference_monolithic']['loss_increase_ratio']:.4f} "
            f"wall={s['partitioned']['wall_ms']:.0f}/{s['reference_monolithic']['wall_ms']:.0f} ms"
        )
    print("decision:", json.dumps(report["decision"]))


if __name__ == "__main__":
    main()
