"""Cost of the N-port reduction and of the circuit-coupled electro-thermal loop.

The two-layer board of ``dc_native_benchmark.py`` (60 mm square, a slot in
the top layer, a via bank).  For every size this measures:

- one voltage-driven ``solve_pcb_dc`` between the two edge pads (the unit of
  cost the N-port is paid in);
- ``dc_port_basis`` for 2, 3 and 5 ports (the edge pads plus interior taps),
  against ``(n - 1)`` times the single solve;
- ``mean_loss_w`` for a random PSD correlation on the 5-port basis;
- one whole ``run_electro_thermal`` with two voltage terminals against one
  whole ``run_circuit_coupled`` with the same pads as ideal Thevenin ports,
  which must give the same state at a bounded overhead.

Writes a JSON with ``environment`` and ``decision``.

    PCB_NATIVE_Q1=1 OPENBLAS_NUM_THREADS=1 \\
        .venv/bin/python experiments/port_basis_benchmark.py --sizes 100,200 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dc_native_benchmark import _compiler, _cpu_model, _timed, board  # noqa: E402
from electrical.matrix_free_mpir_fem import (  # noqa: E402
    MPIRConfig,
    PCBConductionProblem,
    PortSet,
    VoltageTerminal,
    dc_port_basis,
    solve_pcb_dc,
)
from electrical.matrix_free_mpir_fem.native_dc import native_available, native_requested  # noqa: E402
from multiphysics.staggered_coupling import (  # noqa: E402
    CircuitCoupledScenario,
    CoupledConductor,
    CouplingConfig,
    ElectroThermalScenario,
    LinearTheveninCircuit,
    TheveninPort,
    run_circuit_coupled,
    run_electro_thermal,
)
from thermal.matrix_free_mpir_fem import ConvectionBoundary, LayeredThermalMesh  # noqa: E402
from thermal.matrix_free_mpir_fem.native_hex import native_available as thermal_native_available  # noqa: E402

AMBIENT_K = 298.15
COLD_CURRENT_A = 10.0
SOLVER = MPIRConfig(max_inner_iterations=400, max_outer_iterations=16, relative_tolerance=1.0e-8)
COUPLING = CouplingConfig(
    max_iterations=20,
    temperature_tolerance_k=1.0e-3,
    relative_loss_tolerance=1.0e-6,
    aitken=True,
    electrical=SOLVER,
    thermal=SOLVER,
)
BASIS_OVERHEAD_LIMIT = 1.3
COUPLED_OVERHEAD_LIMIT = 1.5
LOSS_TOLERANCE = 1.0e-6
TEMPERATURE_TOLERANCE_K = 1.0e-3


def ports(problem: PCBConductionProblem, count: int) -> PortSet:
    """The two edge pads plus ``count - 2`` interior taps on the top layer."""

    source, sink = problem.terminals
    elements = problem.mesh.element_active.shape[1]
    row = 2 * elements // 3
    taps = [
        tuple((0, r, c) for r in range(row, row + 3))
        for c in (elements // 4, elements // 2, 3 * elements // 4)
    ][: count - 2]
    pads = (source.nodes, *taps, sink.nodes)
    names = ("source", *(f"tap{i}" for i in range(len(taps))), "sink")
    return PortSet(pads=pads, names=names, reference=len(pads) - 1)


def thermal_mesh(problem: PCBConductionProblem) -> LayeredThermalMesh:
    active = problem.mesh.element_active
    elements = active.shape[1]
    conductivity = np.full((3, elements, elements), 0.8)
    conductivity[0] = np.where(active[0], 385.0, 0.8)
    conductivity[2] = np.where(active[1], 385.0, 0.8)
    through = conductivity.copy()
    through[1] = 0.3
    pitch = 60.0e-3 / elements
    return LayeredThermalMesh(
        (35.0e-6, 1.5e-3, 35.0e-6), pitch, pitch, conductivity, through_plane_conductivity_w_per_m_k=through
    )


def psd_correlation(count: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(count - 1, 64)) * 3.0 + 10.0
    waveform = np.vstack((samples, -samples.sum(axis=0, keepdims=True)))
    return waveform @ waveform.T / waveform.shape[1]


def measure(elements: int, repeats: int) -> dict[str, Any]:
    problem = board(elements)
    two = ports(problem, 2)
    single_problem = PCBConductionProblem(
        problem.mesh, voltage_terminals=two.voltage_terminals((1.0, 0.0)), vias=problem.vias
    )
    single, single_time = _timed(lambda: solve_pcb_dc(single_problem, config=SOLVER), repeats, 1)
    record: dict[str, Any] = {
        "elements": elements,
        "nodes": int(problem.mesh.size),
        "vias": len(problem.vias),
        "single_voltage_solve": {
            "converged": bool(single.solve.converged),
            "inner_iterations": int(single.solve.inner_iterations),
            **single_time,
        },
        "bases": [],
    }
    for count in (2, 3, 5):
        port_set = ports(problem, count)
        basis, basis_time = _timed(
            lambda: dc_port_basis(problem.mesh, port_set, vias=problem.vias, config=SOLVER), repeats, 1
        )
        ratio = basis_time["median_ms"] / ((count - 1) * single_time["median_ms"])
        entry = {
            "ports": count,
            "converged": bool(basis.converged),
            "inner_iterations": int(basis.inner_iterations),
            "over_unit_solves": ratio,
            **basis_time,
        }
        if count == 5:
            correlation = psd_correlation(count)
            _, loss_time = _timed(lambda: basis.mean_loss_w(correlation), repeats, 1)
            entry["mean_loss_w_ms"] = loss_time["median_ms"]
            _, rms_time = _timed(lambda: basis.rms_current_density_a_per_m2(correlation), repeats, 1)
            entry["rms_current_density_ms"] = rms_time["median_ms"]
        record["bases"].append(entry)

    # The voltage that drives COLD_CURRENT_A through the cold copper.
    voltage = COLD_CURRENT_A / float(single.voltage_terminal_current_a[0])
    mesh = thermal_mesh(problem)
    convection = (ConvectionBoundary("top", 10.0, AMBIENT_K), ConvectionBoundary("bottom", 10.0, AMBIENT_K))
    reference_scenario = ElectroThermalScenario(
        PCBConductionProblem(
            problem.mesh,
            voltage_terminals=(
                VoltageTerminal(two.pads[0], voltage, "source"),
                VoltageTerminal(two.pads[1], 0.0, "sink"),
            ),
            vias=problem.vias,
        ),
        mesh,
        (0, 2),
        convection=convection,
        conductivity_reference_temperature_k=293.15,
    )
    coupled_scenario = CircuitCoupledScenario(
        (CoupledConductor("board", problem.mesh, two, (0, 2), problem.vias),),
        mesh,
        LinearTheveninCircuit({"board": (TheveninPort(voltage), TheveninPort(0.0))}),
        convection=convection,
        conductivity_reference_temperature_k=293.15,
    )
    start = time.perf_counter()
    reference = run_electro_thermal(reference_scenario, config=COUPLING)
    reference_s = time.perf_counter() - start
    start = time.perf_counter()
    coupled = run_circuit_coupled(coupled_scenario, config=COUPLING)
    coupled_s = time.perf_counter() - start
    rise = reference.thermal.max_temperature_k - AMBIENT_K
    record["coupled"] = {
        "source_voltage_v": voltage,
        "voltage_driven": {
            "wall_s": reference_s,
            "iterations": reference.iterations,
            "converged": reference.converged,
            "joule_loss_w": reference.electrical.joule_loss_w,
            "loss_increase_ratio": reference.loss_increase_ratio,
            "max_temperature_rise_k": rise,
            "hot_current_a": float(reference.electrical.voltage_terminal_current_a[0]),
        },
        "circuit_coupled": {
            "wall_s": coupled_s,
            "iterations": coupled.iterations,
            "converged": coupled.converged,
            "joule_loss_w": coupled.joule_loss_w,
            "loss_increase_ratio": coupled.loss_increase_ratio,
            "max_temperature_rise_k": coupled.thermal.max_temperature_k - AMBIENT_K,
            "hot_current_a": float(coupled.conductors["board"].excitation.mean_current_a[0]),
        },
        "wall_ratio": coupled_s / reference_s,
        "relative_loss_difference": abs(coupled.joule_loss_w - reference.electrical.joule_loss_w)
        / reference.electrical.joule_loss_w,
        "max_temperature_difference_k": float(
            np.nanmax(np.abs(coupled.thermal.temperature_k - reference.thermal.temperature_k))
        ),
    }
    return record


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(sizes: list[int], repeats: int) -> dict[str, Any]:
    cases = [measure(elements, repeats) for elements in sizes]
    basis_ok = all(entry["over_unit_solves"] <= BASIS_OVERHEAD_LIMIT and entry["converged"]
                   for case in cases for entry in case["bases"])
    parity_ok = all(
        case["coupled"]["relative_loss_difference"] <= LOSS_TOLERANCE
        and case["coupled"]["max_temperature_difference_k"] <= TEMPERATURE_TOLERANCE_K
        and case["coupled"]["circuit_coupled"]["converged"]
        and case["coupled"]["voltage_driven"]["converged"]
        for case in cases
    )
    overhead_ok = all(case["coupled"]["wall_ratio"] <= COUPLED_OVERHEAD_LIMIT for case in cases)
    return {
        "benchmark": "port_basis_benchmark",
        "source": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpu": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "compiler": _compiler(),
            "device": "cpu",
            "dc_native_available": native_available(),
            "dc_native_requested": native_requested(),
            "thermal_native_available": thermal_native_available(),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "pcb_native_threads": os.environ.get("PCB_NATIVE_THREADS"),
        },
        "settings": {
            "sizes": sizes,
            "repeats": repeats,
            "cold_current_a": COLD_CURRENT_A,
            "solver": {
                "relative_tolerance": SOLVER.relative_tolerance,
                "max_inner_iterations": SOLVER.max_inner_iterations,
                "max_outer_iterations": SOLVER.max_outer_iterations,
            },
            "coupling": {
                "max_iterations": COUPLING.max_iterations,
                "temperature_tolerance_k": COUPLING.temperature_tolerance_k,
                "relative_loss_tolerance": COUPLING.relative_loss_tolerance,
            },
        },
        "cases": cases,
        "decision": {
            "adopted": basis_ok and parity_ok and overhead_ok,
            "criteria": (
                f"every N-port basis within {BASIS_OVERHEAD_LIMIT}x of (n-1) unit solves and converged; "
                f"circuit-coupled loop equals the voltage-driven loop (loss {LOSS_TOLERANCE:g} relative, "
                f"temperature {TEMPERATURE_TOLERANCE_K:g} K) within {COUPLED_OVERHEAD_LIMIT}x of its wall time"
            ),
            "basis_within_limit": basis_ok,
            "coupled_parity": parity_ok,
            "coupled_within_limit": overhead_ok,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="100,200", help="comma separated element counts per side")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "benchmark-results" / "port_basis_benchmark.json",
    )
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",") if value]
    report = run(sizes, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for case in report["cases"]:
        single = case["single_voltage_solve"]["median_ms"]
        print(f"{case['elements']}^2 elements, {case['nodes']} nodes: single voltage solve {single:.0f} ms")
        for entry in case["bases"]:
            print(f"  {entry['ports']} ports: basis {entry['median_ms']:.0f} ms = {entry['over_unit_solves']:.2f}x (n-1) solves"
                  + (f", mean_loss_w {entry['mean_loss_w_ms']:.1f} ms" if "mean_loss_w_ms" in entry else ""))
        c = case["coupled"]
        print(f"  coupled: voltage-driven {c['voltage_driven']['wall_s']:.1f} s / {c['voltage_driven']['iterations']} it, "
              f"circuit {c['circuit_coupled']['wall_s']:.1f} s / {c['circuit_coupled']['iterations']} it, "
              f"ratio {c['wall_ratio']:.2f}, loss diff {c['relative_loss_difference']:.1e}, "
              f"T diff {c['max_temperature_difference_k']:.1e} K, rise {c['voltage_driven']['max_temperature_rise_k']:.2f} K")
    print("decision:", json.dumps(report["decision"]))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
