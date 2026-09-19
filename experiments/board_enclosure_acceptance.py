"""Acceptance of the masked thermal mesh and the board/body interface coupling.

Runs the acceptance items 2, 3 and 4 of ``docs/GEOMETRY_STEP_VOXELIZE.md`` on
synthetic arrays (no CAD needed) and records solve time, inner/outer
iterations, residuals and solution differences for the array, C++ and CUDA
paths.  Writes a JSON with ``environment`` and ``decision``; the adopted copy
lives at ``docs/BOARD_ENCLOSURE_ACCEPTANCE_RESULTS.json``.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/board_enclosure_acceptance.py
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

from multiphysics.staggered_coupling import (  # noqa: E402
    BoardEnclosureThermalScenario,
    BodyContact,
    InterfaceCouplingConfig,
    run_board_enclosure_thermal,
)
from thermal.matrix_free_mpir_fem import (  # noqa: E402
    ConvectionBoundary,
    ExposedFaceConvection,
    LayeredThermalMesh,
    ThermalConductionProblem,
    planar_contact_map,
    solve_thermal_conduction,
)
from thermal.matrix_free_mpir_fem.native_hex import native_available  # noqa: E402


def _cpu_model() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text()
        for line in text.splitlines():
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


def _cuda_runtime() -> Any:
    try:
        from electrical.matrix_free_mpir_fem import CupyFloat32Runtime

        return CupyFloat32Runtime()
    except Exception:  # pragma: no cover - no device or no CuPy
        return None


def _timed(fn: Callable[[], Any], repeats: int) -> tuple[Any, dict[str, float]]:
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - start) * 1.0e3)
    return result, {"median_ms": statistics.median(times), "min_ms": min(times), "repeats": repeats}


def _solve_record(solution: Any) -> dict[str, Any]:
    return {
        "converged": bool(solution.solve.converged),
        "outer_iterations": solution.solve.outer_iterations,
        "inner_iterations": solution.solve.inner_iterations,
        "relative_residual": solution.solve.relative_residual,
        "low_operator": solution.solve.low_runtime if isinstance(solution.solve.low_runtime, str) else str(type(solution.solve.low_runtime).__name__),
        "max_temperature_k": solution.max_temperature_k,
        "heat_balance_error_w": solution.heat_balance_error_w,
    }


# ------------------------------------------------------------- acceptance 2
def block_in_void(size: int, repeats: int, runtime: Any) -> dict[str, Any]:
    """A cooled block with an internal cavity, as a plain mesh and inside a void grid."""

    n = size
    active_block = np.ones((n, n, n), dtype=bool)
    active_block[n // 3 : 2 * n // 3, n // 3 : 2 * n // 3, n // 3 : 2 * n // 3] = False  # cavity
    heat_block = np.zeros((n, n, n))
    heat_block[1, 1:-1, 1:-1] = 2.0 / ((n - 2) ** 2)
    pitch = 1.0e-3
    plain = LayeredThermalMesh((pitch,) * n, pitch, pitch, 200.0, element_shape=(n, n), active=active_block)
    plain_problem = ThermalConductionProblem(plain, convection=(ExposedFaceConvection(30.0, 300.0),), element_heat_w=heat_block)

    pad = 3
    grid = (n + 2 * pad,) * 3
    active = np.zeros(grid, dtype=bool)
    active[pad:-pad, pad:-pad, pad:-pad] = active_block
    heat = np.zeros(grid)
    heat[pad:-pad, pad:-pad, pad:-pad] = heat_block
    masked = LayeredThermalMesh((pitch,) * grid[0], pitch, pitch, 200.0, element_shape=grid[1:], active=active)
    masked_problem = ThermalConductionProblem(masked, convection=(ExposedFaceConvection(30.0, 300.0),), element_heat_w=heat)

    reference, plain_timing = _timed(lambda: solve_thermal_conduction(plain_problem), repeats)
    paths: dict[str, Any] = {}
    candidates: list[tuple[str, Callable[[], Any]]] = [("array", lambda: solve_thermal_conduction(masked_problem))]
    if native_available():
        threads = int(os.environ.get("PCB_NATIVE_THREADS", "1"))
        candidates.append((f"cpp-native-threads{threads}", lambda: solve_thermal_conduction(masked_problem, native=True, native_threads=threads)))
    if runtime is not None:
        candidates.append(("cuda", lambda: solve_thermal_conduction(masked_problem, runtime=runtime)))
    inside = (slice(pad, pad + n + 1),) * 3
    for name, fn in candidates:
        solution, timing = _timed(fn, repeats)
        difference = np.nanmax(np.abs(solution.temperature_k[inside] - reference.temperature_k))
        rise = reference.max_temperature_k - 300.0
        paths[name] = {
            "timing": timing,
            "solve": _solve_record(solution),
            "max_abs_difference_k": float(difference),
            "relative_to_rise": float(difference / rise),
            "nan_outside_body": bool(np.all(np.isnan(solution.temperature_k[~masked.active_nodes]))),
        }
    return {
        "block_elements": int(active_block.sum()),
        "grid_elements": int(np.prod(grid)),
        "nodes_plain": plain.size,
        "nodes_masked": masked.size,
        "plain": {"timing": plain_timing, "solve": _solve_record(reference)},
        "masked_paths": paths,
    }


# ------------------------------------------------------------- acceptance 3
def fin(cols: int, repeats: int) -> dict[str, Any]:
    k, h, thickness, pitch = 200.0, 10.0, 1.0e-3, 20.0e-3 / cols
    rows = max(2, cols // 5)
    grid = (3, rows + 4, cols + 4)
    active = np.zeros(grid, dtype=bool)
    active[1, 2 : 2 + rows, 0:cols] = True
    mesh = LayeredThermalMesh((thickness,) * 3, pitch, pitch, k, element_shape=grid[1:], active=active)
    fixed = np.zeros(mesh.node_shape, dtype=bool)
    fixed[1:3, 2 : 3 + rows, 0] = True
    problem = ThermalConductionProblem(
        mesh,
        convection=(ExposedFaceConvection(h, 300.0, directions=("-z", "+z")),),
        fixed_temperature_mask=fixed,
        fixed_temperature_k=350.0,
    )
    solution, timing = _timed(lambda: solve_thermal_conduction(problem), repeats)
    m = np.sqrt(2.0 * h / (k * thickness))
    length = cols * pitch
    x = pitch * np.arange(cols + 1)
    analytic = 300.0 + 50.0 * np.cosh(m * (length - x)) / np.cosh(m * length)
    centre = solution.temperature_k[1, 2 + rows // 2, : cols + 1]
    q_base = k * rows * pitch * thickness * m * 50.0 * np.tanh(m * length)
    return {
        "elements_along_fin": cols,
        "pitch_m": pitch,
        "mL": float(m * length),
        "timing": timing,
        "solve": _solve_record(solution),
        "max_relative_temperature_error": float(np.max(np.abs(centre - analytic) / (analytic - 300.0))),
        "base_heat_relative_error": float(abs(-solution.fixed_temperature_heat_w - q_base) / q_base),
    }


# ------------------------------------------------------------- acceptance 4
def board_and_sink(repeats: int) -> dict[str, Any]:
    pitch, rows, cols = 1.0e-3, 12, 16
    board_slabs, board_k, board_kz = (35.0e-6, 1.5e-3, 35.0e-6), (385.0, 0.8, 385.0), (385.0, 0.3, 385.0)
    sink_rows, sink_cols, sink_slabs, sink_k = slice(3, 9), slice(5, 11), (1.0e-3,) * 4, 200.0
    tim_k, tim_t, h, ambient = 1.0e-2, 1.0e-6, 10.0, 300.0
    heat = np.zeros((3, rows, cols))
    heat[0, 5:7, 6:10] = 0.05
    heat[2, 1, 13] = 0.1

    slabs = board_slabs + (tim_t,) + sink_slabs
    active = np.ones((len(slabs), rows, cols), dtype=bool)
    active[3:] = False
    active[3:, sink_rows, sink_cols] = True
    k = np.zeros(active.shape)
    kz = np.zeros(active.shape)
    for i, (a, b) in enumerate(zip(board_k, board_kz)):
        k[i], kz[i] = a, b
    k[3] = kz[3] = tim_k
    k[4:] = kz[4:] = sink_k
    mono_mesh = LayeredThermalMesh(slabs, pitch, pitch, np.where(active, k, 1.0), through_plane_conductivity_w_per_m_k=np.where(active, kz, 1.0), active=active)
    mono_heat = np.zeros(active.shape)
    mono_heat[:3] = heat
    mono = ThermalConductionProblem(mono_mesh, convection=(ExposedFaceConvection(h, ambient),), element_heat_w=mono_heat)
    reference, mono_timing = _timed(lambda: solve_thermal_conduction(mono), repeats)

    covered = np.zeros((rows, cols), dtype=bool)
    covered[sink_rows, sink_cols] = True
    board_mesh = LayeredThermalMesh(board_slabs, pitch, pitch, board_k, through_plane_conductivity_w_per_m_k=board_kz, element_shape=(rows, cols))
    board = ThermalConductionProblem(
        board_mesh,
        convection=(
            ExposedFaceConvection(h, ambient, directions=("-z", "-x", "+x", "-y", "+y")),
            ConvectionBoundary("top", np.where(covered, 0.0, h), ambient),
        ),
        element_heat_w=heat,
    )
    sink_mesh = LayeredThermalMesh(sink_slabs, pitch, pitch, sink_k, element_shape=(6, 6))
    sink = ThermalConductionProblem(sink_mesh, convection=(ExposedFaceConvection(h, ambient, directions=("+z", "-x", "+x", "-y", "+y")),))
    contact = planar_contact_map(board_mesh, sink_mesh, board_side="top", body_origin_m=(sink_cols.start * pitch, sink_rows.start * pitch), conductance_per_area_w_per_m2_k=tim_k / tim_t)
    scenario = BoardEnclosureThermalScenario(board, (BodyContact(sink, contact, "sink"),))

    rise = reference.max_temperature_k - ambient
    runs: dict[str, Any] = {}
    for name, config in (
        ("aitken", InterfaceCouplingConfig()),
        ("fixed_relaxation_0.2", InterfaceCouplingConfig(aitken=False, relaxation=0.2, max_iterations=400)),
        ("unrelaxed", InterfaceCouplingConfig(aitken=False, relaxation=1.0, max_iterations=400)),
    ):
        result, timing = _timed(lambda: run_board_enclosure_thermal(scenario, config=config), repeats)
        record: dict[str, Any] = {
            "timing": timing,
            "converged": result.converged,
            "interface_iterations": result.iterations,
            "interface_heat_w": result.interface_heat_w,
            "board_inner_iterations_total": sum(s.board_inner_iterations for s in result.history),
            "body_inner_iterations_total": sum(s.body_inner_iterations for s in result.history),
        }
        if result.converged:
            board_error = float(np.nanmax(np.abs(result.board.temperature_k - reference.temperature_k[:4])))
            sink_error = float(np.nanmax(np.abs(result.bodies[0].temperature_k - reference.temperature_k[4:, 3:10, 5:12])))
            record.update(
                {
                    "board_max_abs_difference_k": board_error,
                    "sink_max_abs_difference_k": sink_error,
                    "board_relative_to_rise": board_error / rise,
                    "sink_relative_to_rise": sink_error / rise,
                    "interface_heat_vs_board_contact_convection": float(abs(result.interface_heat_w - result.board.convective_heat_w[-1]) / result.interface_heat_w),
                }
            )
        runs[name] = record
    return {
        "rise_k": float(rise),
        "monolithic": {"timing": mono_timing, "solve": _solve_record(reference), "nodes": mono_mesh.size},
        "contact_pairs": contact.size,
        "conductance_per_area_w_per_m2_k": tim_k / tim_t,
        "partitioned": runs,
    }


def run(repeats: int, block_size: int, fin_cols: list[int]) -> dict[str, Any]:
    runtime = _cuda_runtime()
    device = "cpu"
    cupy_version = None
    if runtime is not None:
        import cupy

        cupy_version = cupy.__version__
        device = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    block = block_in_void(block_size, repeats, runtime)
    fins = [fin(cols, repeats) for cols in fin_cols]
    coupled = board_and_sink(repeats)

    block_ok = all(p["relative_to_rise"] <= 1.0e-6 and p["solve"]["converged"] and p["nan_outside_body"] for p in block["masked_paths"].values())
    fin_ok = fins[-1]["max_relative_temperature_error"] <= 5.0e-3 and fins[-1]["base_heat_relative_error"] <= 2.0e-2
    fin_converging = all(a["max_relative_temperature_error"] >= b["max_relative_temperature_error"] for a, b in zip(fins, fins[1:]))
    aitken = coupled["partitioned"]["aitken"]
    coupled_ok = aitken["converged"] and aitken["board_relative_to_rise"] <= 2.0e-3 and aitken["sink_relative_to_rise"] <= 2.0e-3
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cupy": cupy_version,
            "device": device,
            "cpu": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "compiler": _compiler(),
            "native_extension_built": native_available(),
            "PCB_NATIVE_THREADS": os.environ.get("PCB_NATIVE_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "acceptance_2_block_in_void": block,
        "acceptance_3_fin": fins,
        "acceptance_4_board_and_sink": coupled,
        "decision": {
            "adopted": bool(block_ok and fin_ok and fin_converging and coupled_ok),
            "criteria": (
                "masked mesh matches the plain mesh of the same body within 1e-6 of the rise on every available path with nan outside the body; "
                "fin within 5e-3 of the 1D solution and base heat within 2 % at the finest pitch, error non-increasing with refinement; "
                "partitioned board and sink within 2e-3 of the rise of the monolithic mesh with Aitken"
            ),
            "block_paths_within_1e-6": block_ok,
            "fin_within_tolerance": fin_ok,
            "fin_error_non_increasing": fin_converging,
            "partitioned_within_2e-3": coupled_ok,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=24)
    parser.add_argument("--fin-cols", default="20,40,80")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results") / "board_enclosure_acceptance.json")
    args = parser.parse_args()
    report = run(args.repeats, args.block_size, [int(v) for v in args.fin_cols.split(",")])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    block = report["acceptance_2_block_in_void"]
    print(f"block: plain {block['plain']['timing']['median_ms']:.1f} ms ({block['nodes_plain']} nodes)")
    for name, path in block["masked_paths"].items():
        print(f"  masked {name:22s} {path['timing']['median_ms']:8.1f} ms inner={path['solve']['inner_iterations']:4d} diff={path['max_abs_difference_k']:.2e} K")
    for item in report["acceptance_3_fin"]:
        print(f"fin cols={item['elements_along_fin']:3d} T err={item['max_relative_temperature_error']:.2e} q err={item['base_heat_relative_error']:.2e} {item['timing']['median_ms']:.1f} ms")
    coupled = report["acceptance_4_board_and_sink"]
    print(f"monolithic {coupled['monolithic']['timing']['median_ms']:.1f} ms rise={coupled['rise_k']:.2f} K")
    for name, item in coupled["partitioned"].items():
        extra = f" board diff={item['board_max_abs_difference_k']:.2e} K sink diff={item['sink_max_abs_difference_k']:.2e} K" if item["converged"] else ""
        print(f"  {name:22s} converged={item['converged']} iterations={item['interface_iterations']:3d} {item['timing']['median_ms']:8.1f} ms{extra}")
    print("decision:", json.dumps(report["decision"]))


if __name__ == "__main__":
    main()
