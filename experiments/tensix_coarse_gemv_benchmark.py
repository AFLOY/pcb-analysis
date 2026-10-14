"""Two-level coarse correction on a Blackhole Tensix FPU, inside the FP32 PCG.

The CPU emulation (tensix_precision_emulation.py) found that the dense coarse
inverse of the two-level preconditioner survives BF16 and BFP8_B only when it
is kept as its lower Cholesky factor ``R`` and applied as ``R (r^T R)^T`` from
one stored matrix.  This run puts that factor on a Tenstorrent device and asks
three questions on the same fixtures:

1. Does ttnn's host conversion of ``R`` match the emulated rounding bit for
   bit?  (``conversion``)
2. How close is the device's two-GEMV result to the emulation and to the FP64
   product, per format and math fidelity, and how long does it take with the
   vectors resident on the device and with one PCIe round trip per call?
   (``kernels``; FP32 accumulation in the destination register throughout.)
3. Does the inner PCG converge as emulated when its coarse GEMVs run on the
   device and everything else stays FP32 on the host?  (``solves``)

Vectors are rows of a 32-row tile (row 0 holds the data): ``t = r R`` then
``y = t R^T`` via ``transpose_b``, so both products read the one stored ``R``.
``latency`` repeats the BFP8_B/HiFi4 pair captured in a ttnn trace (no host
dispatch) and with ``R`` and the vectors in L1 instead of DRAM, to tell host
dispatch, DRAM traffic and the matmul program's own run time apart.
Solve wall times include one host-device round trip per inner iteration and
are not a Tensix estimate of a fully resident PCG.

Run with the ttnn environment (NumPy 1.26, torch, ttnn):

    ~/tt_test/.venv/bin/python experiments/tensix_coarse_gemv_benchmark.py
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import ttnn  # noqa: E402

from dc_native_benchmark import CONFIG as DC_CONFIG, board  # noqa: E402
from electrical.matrix_free_mpir_fem import MatrixFreePCBOperator, MPIRConfig, solve_mpir  # noqa: E402
from tensix_formats import MATRIX_FORMATS  # noqa: E402
from tensix_precision_emulation import EmulatedSystem, Variant, verdict  # noqa: E402
from thermal.matrix_free_mpir_fem import MatrixFreeThermalOperator  # noqa: E402
from thermal_native_benchmark import CONFIG as THERMAL_CONFIG, stack  # noqa: E402

TILE = 32
TRACE_REGION_BYTES = 8 << 20


@dataclass(frozen=True)
class DeviceFormat:
    """One coarse factor format on the device and its emulated twin."""

    name: str
    matrix: Any  # ttnn dtype of R
    activation: Any  # ttnn dtype of r and t
    emulated: str  # key of MATRIX_FORMATS


def _formats() -> dict[str, DeviceFormat]:
    return {
        "fp32": DeviceFormat("fp32", ttnn.float32, ttnn.float32, "fp32"),
        "bf16": DeviceFormat("bf16", ttnn.bfloat16, ttnn.bfloat16, "bf16"),
        "bfp8b": DeviceFormat("bfp8b", ttnn.bfloat8_b, ttnn.bfloat16, "bfp8b"),
        "bfp8b-fp32act": DeviceFormat("bfp8b-fp32act", ttnn.bfloat8_b, ttnn.float32, "bfp8b"),
        "bfp4b": DeviceFormat("bfp4b", ttnn.bfloat4_b, ttnn.bfloat16, "bfp4b"),
    }


FIDELITIES = ("LoFi", "HiFi2", "HiFi4")
SOLVE_CONFIGS = (("fp32", "HiFi4"), ("bf16", "HiFi4"), ("bfp8b", "HiFi4"), ("bfp8b", "LoFi"), ("bfp8b-fp32act", "HiFi4"))


def _padded(size: int) -> int:
    return -(-size // TILE) * TILE


def _to_device(device: Any, host: np.ndarray, dtype: Any, rows: int, cols: int, memory: Any = None) -> Any:
    staged = torch.zeros(rows, cols, dtype=torch.float32)
    staged[: host.shape[0], : host.shape[1]] = torch.from_numpy(np.ascontiguousarray(host, dtype=np.float32))
    return ttnn.from_torch(staged, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=memory or ttnn.DRAM_MEMORY_CONFIG)


class DeviceCoarse:
    """``R (r^T R)^T`` on the device from one stored Cholesky factor."""

    def __init__(self, device: Any, factor: np.ndarray, fmt: DeviceFormat, fidelity: str, memory: Any = None) -> None:
        self.device = device
        self.size = factor.shape[0]
        self.padded = _padded(self.size)
        self.format = fmt
        self.memory = memory or ttnn.DRAM_MEMORY_CONFIG
        self.factor = _to_device(device, factor, fmt.matrix, self.padded, self.padded, self.memory)
        self.config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, fidelity),
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            math_approx_mode=False,
        )

    def factor_on_host(self) -> np.ndarray:
        return ttnn.to_torch(self.factor).float().numpy()[: self.size, : self.size]

    def upload(self, vector: np.ndarray) -> Any:
        return _to_device(self.device, np.asarray(vector, dtype=np.float32).reshape(1, -1), self.format.activation, TILE, self.padded,
                          self.memory)

    def apply(self, row: Any) -> Any:
        inner = ttnn.matmul(row, self.factor, compute_kernel_config=self.config, dtype=self.format.activation,
                            memory_config=self.memory)
        return ttnn.matmul(inner, self.factor, transpose_b=True, compute_kernel_config=self.config, dtype=ttnn.float32,
                           memory_config=self.memory)

    def download(self, row: Any) -> np.ndarray:
        return ttnn.to_torch(row).float().numpy()[0, : self.size].astype(np.float32)

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        return self.download(self.apply(self.upload(vector)))


class DeviceSystem(EmulatedSystem):
    """The emulated factored system with its coarse GEMVs on the device."""

    def __init__(self, operator: Any, variant: Variant, coarse: DeviceCoarse) -> None:
        super().__init__(operator, variant)
        self.device_coarse = coarse

    def precondition_low(self, vector: np.ndarray) -> np.ndarray:
        flat = np.asarray(vector, dtype=np.float32).reshape(-1)
        smoothed = flat / self.diagonal
        grid = np.where(self.free, flat.reshape(self.node_shape), np.float32(0.0))
        coarse = self.device_coarse(self._restrict(grid).reshape(-1))
        correction = np.where(self.free, self._prolong(coarse), np.float32(0.0))
        return (smoothed + correction.reshape(-1)).astype(np.float32)


# ----------------------------------------------------------------- helpers
def _relative(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, np.float64) - reference) / np.linalg.norm(reference))


def _emulated_apply(system: EmulatedSystem, vector: np.ndarray) -> np.ndarray:
    fmt = system.format
    restricted = fmt.vector(np.asarray(vector, dtype=np.float32))
    inner = fmt.vector(np.matmul(restricted, system.coarse_factor).astype(np.float32))
    return np.matmul(system.coarse_factor, inner).astype(np.float32)


def _per_call_us(fn: Any, repeats: int, warmups: int, sync: Any = None) -> dict[str, Any]:
    for _ in range(warmups):
        fn()
    if sync is not None:
        sync()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        if sync is not None:
            sync()
        samples.append((time.perf_counter() - start) / repeats * 1.0e6)
    return {"median_us": statistics.median(samples), "minimum_us": min(samples), "samples_us": samples, "calls_per_sample": repeats}


def _solve(system: Any, rhs: np.ndarray, config: MPIRConfig, reference: np.ndarray | None) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        result = solve_mpir(system, rhs, config=config)
    except RuntimeError as exc:
        return {"failed": str(exc), "solve_ms": (time.perf_counter() - start) * 1e3}
    run = {
        "solve_ms": (time.perf_counter() - start) * 1e3,
        "converged": bool(result.converged),
        "relative_residual": float(result.relative_residual),
        "outer_iterations": int(result.outer_iterations),
        "inner_iterations": int(result.inner_iterations),
        "_solution": result.solution,
    }
    if reference is not None:
        difference = result.solution - reference
        run["relative_solution_error_vs_stock"] = float(np.linalg.norm(difference) / np.linalg.norm(reference))
        run["relative_max_error_vs_stock"] = float(np.abs(difference).max() / np.abs(reference).max())
    return run


def _strip(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if not key.startswith("_")}


def _latency(device: Any, factor: np.ndarray, fmt: DeviceFormat, vector: np.ndarray) -> dict[str, Any]:
    """The BFP8_B/HiFi4 pair resident in DRAM, replayed from a trace, and resident in L1."""

    sync = lambda: ttnn.synchronize_device(device)  # noqa: E731
    dram = DeviceCoarse(device, factor, fmt, "HiFi4")
    row = dram.upload(vector)
    result = {"dram_resident": _per_call_us(lambda: dram.apply(row), repeats=100, warmups=5, sync=sync)}
    dram.apply(row)
    sync()
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    dram.apply(row)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    result["dram_traced"] = _per_call_us(lambda: ttnn.execute_trace(device, trace, cq_id=0, blocking=False), repeats=100,
                                         warmups=5, sync=sync)
    ttnn.release_trace(device, trace)
    del dram, row
    l1 = DeviceCoarse(device, factor, fmt, "HiFi4", ttnn.L1_MEMORY_CONFIG)
    row = l1.upload(vector)
    result["l1_resident"] = _per_call_us(lambda: l1.apply(row), repeats=100, warmups=5, sync=sync)
    return result


# ------------------------------------------------------------------- cases
def _case(device: Any, physics: str, elements: int, operator: Any, rhs: np.ndarray, config: MPIRConfig, solves: bool) -> dict[str, Any]:
    formats = _formats()
    correction = operator.coarse_correction
    stock = _solve(operator, rhs, config, None)
    reference = stock.pop("_solution")
    print(f"{physics:7s} {elements:4d} stock outer={stock['outer_iterations']} inner={stock['inner_iterations']}", flush=True)

    probe = EmulatedSystem(operator, Variant("probe", "fp32", None, True))
    inverse64 = np.linalg.inv(correction.coarse_matrix)
    inverse64 = 0.5 * (inverse64 + inverse64.T)
    factor64 = np.linalg.cholesky(inverse64)
    free = probe.free
    vectors = {
        "restricted_rhs": probe._restrict(np.where(free, rhs.reshape(probe.node_shape), 0.0)).reshape(-1).astype(np.float32),
        "random": np.random.default_rng(0).standard_normal(correction.coarse_size).astype(np.float32),
    }
    dense = inverse64.astype(np.float32)
    cpu_sgemv = _per_call_us(lambda: dense @ vectors["random"], repeats=200, warmups=20)

    conversion: dict[str, Any] = {}
    kernels: dict[str, Any] = {}
    for fmt in formats.values():
        emulated = EmulatedSystem(operator, Variant(fmt.name, fmt.emulated, None, True))
        for fidelity in FIDELITIES:
            coarse = DeviceCoarse(device, factor64, fmt, fidelity)
            if fmt.name not in conversion:
                on_device = coarse.factor_on_host()
                expected = emulated.coarse_factor
                differs = on_device != expected
                conversion[fmt.name] = {
                    "values": int(on_device.size),
                    "mismatched_values": int(differs.sum()),
                    "maximum_relative_difference": float(np.abs(on_device - expected).max() / np.abs(expected).max()),
                    "relative_error_vs_fp64_factor": _relative(on_device, factor64),
                    "emulated_relative_error_vs_fp64_factor": _relative(expected, factor64),
                }
            accuracy = {}
            for label, vector in vectors.items():
                exact = inverse64 @ vector.astype(np.float64)
                on_device = coarse(vector)
                emulated_result = _emulated_apply(emulated, vector)
                accuracy[label] = {
                    "device_vs_fp64": _relative(on_device, exact),
                    "emulated_vs_fp64": _relative(emulated_result, exact),
                    "device_vs_emulated": _relative(on_device, emulated_result.astype(np.float64)),
                }
            row = coarse.upload(vectors["random"])
            resident = _per_call_us(lambda: coarse.apply(row), repeats=100, warmups=5, sync=lambda: ttnn.synchronize_device(device))
            round_trip = _per_call_us(lambda: coarse(vectors["random"]), repeats=20, warmups=3)
            key = f"{fmt.name}/{fidelity}"
            kernels[key] = {
                "matrix_dtype": str(fmt.matrix),
                "activation_dtype": str(fmt.activation),
                "fidelity": fidelity,
                "accuracy": accuracy,
                "device_resident_two_gemv": resident,
                "round_trip_two_gemv": round_trip,
            }
            print(f"{physics:7s} {elements:4d} {key:22s} dev/fp64={accuracy['random']['device_vs_fp64']:.2e} "
                  f"emu/fp64={accuracy['random']['emulated_vs_fp64']:.2e} dev/emu={accuracy['random']['device_vs_emulated']:.2e} "
                  f"resident={resident['median_us']:7.1f} us round-trip={round_trip['median_us']:7.1f} us", flush=True)
            del coarse

    latency = _latency(device, factor64, formats["bfp8b"], vectors["random"])
    print(f"{physics:7s} {elements:4d} latency bfp8b/HiFi4 dram={latency['dram_resident']['median_us']:.1f} us "
          f"traced={latency['dram_traced']['median_us']:.1f} us l1={latency['l1_resident']['median_us']:.1f} us", flush=True)

    solve_runs: dict[str, Any] = {}
    if solves:
        for name, fidelity in SOLVE_CONFIGS:
            fmt = formats[name]
            variant = Variant(f"coarse-factor-{fmt.emulated}", fmt.emulated, None, True)
            emulated_key = f"emulated/{fmt.emulated}"
            if emulated_key not in solve_runs:
                solve_runs[emulated_key] = _strip(_solve(EmulatedSystem(operator, variant), rhs, config, reference))
            coarse = DeviceCoarse(device, factor64, fmt, fidelity)
            run = _strip(_solve(DeviceSystem(operator, variant, coarse), rhs, config, reference))
            run["viable"] = verdict(run, stock, config)
            solve_runs[f"device/{name}/{fidelity}"] = run
            del coarse
            emulated_run = solve_runs[emulated_key]
            print(f"{physics:7s} {elements:4d} solve {name}/{fidelity:6s} "
                  + (f"failed: {run['failed']}" if "failed" in run else
                     f"conv={run['converged']} outer={run['outer_iterations']} inner={run['inner_iterations']} "
                     f"(emulated {emulated_run.get('outer_iterations')}/{emulated_run.get('inner_iterations')}) "
                     f"err={run['relative_solution_error_vs_stock']:.1e} {run['solve_ms'] / 1e3:.2f} s"), flush=True)

    return {
        "physics": physics,
        "elements": [elements, elements],
        "nodes": int(operator.size),
        "coarse_size": int(correction.coarse_size),
        "padded_coarse_size": _padded(correction.coarse_size),
        "config": {"relative_tolerance": config.relative_tolerance, "inner_relative_tolerance": config.inner_relative_tolerance,
                   "max_outer_iterations": config.max_outer_iterations, "max_inner_iterations": config.max_inner_iterations},
        "stock": stock,
        "cpu_fp32_dense_inverse_gemv": cpu_sgemv,
        "conversion": conversion,
        "kernels": kernels,
        "latency": latency,
        "solves": solve_runs,
    }


def _environment(device: Any, device_id: int) -> dict[str, Any]:
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    grid = device.compute_with_storage_grid_size()
    try:
        driver = Path("/sys/module/tenstorrent/version").read_text().strip()
    except OSError:
        driver = None
    try:
        boards = sorted(entry.name for entry in Path("/dev/tenstorrent/by-id").iterdir())
    except OSError:
        boards = []
    cpu = platform.processor()
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "ttnn": version("ttnn"),
        "tenstorrent_driver": driver,
        "device": f"Tenstorrent {device.arch()}",
        "device_id": device_id,
        "boards": boards,
        "compute_grid": [grid.x, grid.y],
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
    }


def run(device_id: int, dc_sizes: list[int], thermal_sizes: list[int], solves: bool) -> dict[str, Any]:
    device = ttnn.open_device(device_id=device_id, trace_region_size=TRACE_REGION_BYTES)
    try:
        environment = _environment(device, device_id)
        cases = []
        for elements in dc_sizes:
            problem = board(elements)
            operator = MatrixFreePCBOperator(problem.mesh, reference_node=problem.reference_node, vias=problem.vias, native=False)
            cases.append(_case(device, "dc", elements, operator, operator.build_rhs(problem.terminals), DC_CONFIG, solves))
        for elements in thermal_sizes:
            operator = MatrixFreeThermalOperator(stack(elements), native=False)
            rhs = operator.build_rhs(operator.default_reference_temperature())
            cases.append(_case(device, "thermal", elements, operator, rhs, THERMAL_CONFIG, solves))
    finally:
        ttnn.close_device(device)

    solve_keys = sorted({key for case in cases for key in case["solves"] if key.startswith("device/")})
    viable = {key: all(case["solves"][key].get("viable", False) for case in cases) for key in solve_keys}
    round_trip = {key: max(case["kernels"][key]["round_trip_two_gemv"]["median_us"] / case["cpu_fp32_dense_inverse_gemv"]["median_us"]
                           for case in cases) for key in cases[0]["kernels"]}
    resident = {key: max(case["kernels"][key]["device_resident_two_gemv"]["median_us"] / case["cpu_fp32_dense_inverse_gemv"]["median_us"]
                         for case in cases) for key in cases[0]["kernels"]}
    conversion_exact = {name: all(case["conversion"][name]["mismatched_values"] == 0 for case in cases) for name in cases[0]["conversion"]}
    return {
        "schema": "tensix-coarse-gemv-benchmark/v1",
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "environment": environment,
        "definitions": {
            "factor": "lower Cholesky factor R of the FP64 coarse inverse; the device stores it once and applies R (r^T R)^T",
            "conversion": "ttnn host conversion of R read back and compared with tensix_formats rounding (round-to-nearest-even, 16-value block exponents)",
            "accuracy": "relative L2 errors of the two-GEMV result against the FP64 dense inverse product and against the emulation",
            "device_resident_two_gemv": "wall time per call of the two ttnn.matmul with the input already on the device, synchronised every sample",
            "round_trip_two_gemv": "wall time per call including ttnn.from_torch of the restricted residual and ttnn.to_torch of the result",
            "latency": "bfp8b/HiFi4 pair per call: dram_resident as above, dram_traced replayed with ttnn.execute_trace, l1_resident with R and vectors in L1",
            "cpu_fp32_dense_inverse_gemv": "NumPy float32 dense-inverse GEMV on the host, the stock two-level preconditioner's product",
            "solves": "solve_mpir with the portable FP32 operator and Jacobi on the host, the coarse GEMVs on the device; emulated/* is the CPU emulation of the same format",
            "viable": "the CPU emulation's criterion: converged, at most one extra outer step, at most 10% more inner iterations than stock, solution within 100 x relative_tolerance",
        },
        "cases": cases,
        "decision": {
            "adopted": False,
            "scope": "measurement only; nothing in src/ changes",
            "host_conversion_matches_emulation": conversion_exact,
            "solve_viable_by_config": viable,
            "worst_round_trip_over_cpu_gemv": round_trip,
            "worst_resident_over_cpu_gemv": resident,
            "coarse_only_offload_pays": any(ratio < 1.0 for ratio in round_trip.values()),
            "worst_traced_over_cpu_gemv": max(case["latency"]["dram_traced"]["median_us"] / case["cpu_fp32_dense_inverse_gemv"]["median_us"]
                                              for case in cases),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--dc-sizes", default="100,200,320")
    parser.add_argument("--thermal-sizes", default="50,100,200")
    parser.add_argument("--no-solves", action="store_true", help="kernel and conversion checks only")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results") / "tensix_coarse_gemv_benchmark.json")
    args = parser.parse_args()
    sizes = lambda text: [int(item) for item in text.split(",") if item]  # noqa: E731
    report = run(args.device_id, sizes(args.dc_sizes), sizes(args.thermal_sizes), not args.no_solves)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
