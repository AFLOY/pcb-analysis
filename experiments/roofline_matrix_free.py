"""Roofline of the matrix-free Q1 kernels against the host's bandwidth and FMA peak.

Measures, on the real cores:

1. machine: FP32/FP64 AVX-512 FMA peak and STREAM-like read/copy/triad
   bandwidth per working-set level (L1, L2, L3, DRAM) and thread count;
2. kernels: the shipped matrix-free actions (scalar Maxwell 2D Q1 complex64,
   thermal hex Q1 float32, compiled in from their sources) against the same
   operators assembled by probing the public ``apply_low`` with 9 / 27
   coloured vectors, stored as a structured stencil (one coefficient array per
   neighbour offset, no indices) and as CSR with int32 indices;
3. solver share: the native MPIR solves of the documented fixtures, with the
   operator share estimated as applications x measured action time.

Flop counts are the arithmetic the kernels execute per interior node (a
multiply or an add is one flop, an FMA two); "useful" flops are the assembled
stencil's.  Byte counts are the compulsory traffic of one pass (each array
read once, the output written once, no write-allocate).  The matrix-free
timings read the operator arrays through the native wrappers' attributes
because the timed kernel needs a preallocated output.

    .venv/bin/python experiments/roofline_matrix_free.py --threads 1,4,16
"""

from __future__ import annotations

import os

# libgomp reads these once when the first OpenMP module loads.
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("OMP_PLACES", "cores")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from electrical.matrix_free_mpir_fem import (  # noqa: E402
    MPIRConfig,
    MatrixFreeScalarMaxwellOperator,
    NumpyComplex64Runtime,
    solve_mpir,
)
from electrical.matrix_free_mpir_fem.native.build import compile_extension  # noqa: E402
from maxwell_cuda_benchmark import _problem as maxwell_problem  # noqa: E402
from thermal.matrix_free_mpir_fem import MatrixFreeThermalOperator  # noqa: E402
from thermal_native_benchmark import stack as thermal_problem  # noqa: E402

MAXWELL_CONFIG = MPIRConfig(
    relative_tolerance=1.0e-10,
    inner_relative_tolerance=2.0e-3,
    max_outer_iterations=12,
    max_inner_iterations=400,
    gmres_restart=32,
)
THERMAL_CONFIG = MPIRConfig(max_outer_iterations=16)

# Per interior node: executed flops and compulsory bytes (see module doc).
MODEL = {
    "maxwell": {
        # 16 (element, local column) terms: coefficient imu*K+rea*M 14,
        # Dirichlet mask 2, complex multiply-add 8; Dirichlet blend 7.
        "matrix_free": {"flops": 16 * 24 + 7, "bytes": 8 + 8 + 4 + 8 + 8},
        "stencil": {"flops": 9 * 8, "bytes": 9 * 8 + 8 + 8},
        "csr": {"flops": 9 * 8, "bytes": 9 * (8 + 4) + 4 + 8 + 8},
    },
    "thermal": {
        # 64 (element, local column) terms: a*Ux+b*Uy+d*Uz 5, mask 1,
        # multiply-add 2; four partial sums 4; Robin and Dirichlet blend 6.
        "matrix_free": {"flops": 64 * 8 + 4 + 6, "bytes": 4 + 4 + 4 + 4 + 12},
        "stencil": {"flops": 27 * 2, "bytes": 27 * 4 + 4 + 4},
        "csr": {"flops": 27 * 2, "bytes": 27 * (4 + 4) + 4 + 4 + 4},
    },
}
USEFUL_FLOPS = {"maxwell": 9 * 8, "thermal": 27 * 2}


def _module() -> Any:
    out = ROOT / "benchmark-results" / "native"
    out.mkdir(parents=True, exist_ok=True)
    compile_extension(ROOT / "experiments" / "roofline_matrix_free.cpp", "_roofline_matrix_free", out, verbose=False)
    sys.path.insert(0, str(out))
    import _roofline_matrix_free

    return _roofline_matrix_free


def _stats(samples_s: list[float]) -> dict[str, Any]:
    ms = [s * 1e3 for s in samples_s]
    return {"median_ms": statistics.median(ms), "minimum_ms": min(ms), "maximum_ms": max(ms), "repeats": len(ms)}


# ---------------------------------------------------------------- machine ---

LEVELS = {  # working set per thread (bytes) or total for L3/DRAM
    "L1": ("per_thread", 24 * 1024),
    "L2": ("per_thread", 1024 * 1024),
    "L3": ("total", 96 * 1024 * 1024),
    "DRAM": ("total", 3 * 1024**3),
}
STREAMS = {"read": (0, 1), "copy": (1, 2), "triad": (2, 3)}  # kind, arrays


def machine(mod: Any, threads: list[int]) -> dict[str, Any]:
    peak: dict[str, Any] = {}
    for fp64 in (False, True):
        lanes = 8 if fp64 else 16
        rows = {}
        for t in threads:
            iterations = 20_000_000
            seconds = min(mod.fma_peak(iterations, t, fp64) for _ in range(3))
            rows[str(t)] = t * iterations * 16 * lanes * 2 / seconds / 1e9
        peak["fp64" if fp64 else "fp32"] = {"gflops_by_threads": rows}
    bandwidth: dict[str, Any] = {}
    for level, (scope, size) in LEVELS.items():
        bandwidth[level] = {}
        for name, (kind, arrays) in STREAMS.items():
            rows = {}
            for t in threads:
                total = size * t if scope == "per_thread" else size
                n = total // (4 * arrays)
                inner = max(1, int(4e8 // total))
                s = statistics.median(mod.stream(n, kind, t, 7, inner)) / inner
                rows[str(t)] = arrays * 4 * n / s / 1e9
            bandwidth[level][name] = {"gbytes_per_s_by_threads": rows}
    return {
        "fma_peak": peak,
        "bandwidth": bandwidth,
        "notes": "bandwidth counts bytes read plus bytes written, no write-allocate; each timed call runs repeated sweeps inside one parallel region; "
        "L1/L2 sizes are per thread, L3/DRAM totals; triad a = b + s c in float32",
    }


# ---------------------------------------------------------------- kernels ---


def _assemble(apply_low: Any, node_shape: tuple[int, ...], dtype: Any) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Stencil coefficients (K, nodes) of a nearest-neighbour operator by coloured probes."""

    dims = len(node_shape)
    offsets = [tuple(o) for o in np.ndindex(*(3,) * dims)]
    grids = np.indices(node_shape)
    colour = sum((grids[d] % 3) * 3 ** (dims - 1 - d) for d in range(dims))
    n = int(np.prod(node_shape))
    coef = np.zeros((len(offsets), n), dtype=dtype)
    probes = {}
    for c in range(3**dims):
        probes[c] = np.asarray(apply_low((colour == c).reshape(-1).astype(dtype))).reshape(node_shape)
    for k, off in enumerate(offsets):
        delta = [o - 1 for o in off]
        neighbour = [grids[d] + delta[d] for d in range(dims)]
        inside = np.ones(node_shape, dtype=bool)
        for d in range(dims):
            inside &= (neighbour[d] >= 0) & (neighbour[d] < node_shape[d])
        ncolour = sum((neighbour[d] % 3) * 3 ** (dims - 1 - d) for d in range(dims))
        values = np.zeros(node_shape, dtype=dtype)
        for c in range(3**dims):
            sel = inside & (ncolour == c)
            values[sel] = probes[c][sel]
        coef[k] = values.reshape(-1)
    return coef, [tuple(o - 1 for o in off) for off in offsets]


def _csr(coef: np.ndarray, node_shape: tuple[int, ...], offsets: list[tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dims = len(node_shape)
    grids = np.indices(node_shape)
    n = int(np.prod(node_shape))
    strides = [int(np.prod(node_shape[d + 1 :])) for d in range(dims)]
    valid = np.zeros((n, len(offsets)), dtype=bool)
    cols = np.zeros((n, len(offsets)), dtype=np.int32)
    node = np.arange(n, dtype=np.int64)
    for k, delta in enumerate(offsets):
        inside = np.ones(node_shape, dtype=bool)
        for d in range(dims):
            inside &= (grids[d] + delta[d] >= 0) & (grids[d] + delta[d] < node_shape[d])
        valid[:, k] = inside.reshape(-1)
        cols[:, k] = np.where(valid[:, k], node + sum(delta[d] * strides[d] for d in range(dims)), 0)
    indptr = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(valid.sum(axis=1), out=indptr[1:])
    data = coef.T[valid]
    indices = cols[valid]
    return indptr, indices.astype(np.int32), np.ascontiguousarray(data)


def _repeats(seconds: float) -> int:
    return max(7, min(400, int(1.0 / max(seconds, 1e-6))))


def _row(physics: str, kernel: str, nodes: int, timing: dict[str, Any], machine_data: dict[str, Any], threads: int, level: str) -> dict[str, Any]:
    model = MODEL[physics][kernel]
    seconds = timing["median_ms"] / 1e3
    peak = machine_data["fma_peak"]["fp32"]["gflops_by_threads"][str(threads)]
    bw = machine_data["bandwidth"][level]["read"]["gbytes_per_s_by_threads"][str(threads)]
    flops = model["flops"] * nodes
    nbytes = model["bytes"] * nodes
    bound = max(flops / (peak * 1e9), nbytes / (bw * 1e9))
    return {
        "timing": timing,
        "ns_per_node": seconds / nodes * 1e9,
        "executed_flops_per_node": model["flops"],
        "useful_flops_per_node": USEFUL_FLOPS[physics],
        "compulsory_bytes_per_node": model["bytes"],
        "arithmetic_intensity_flop_per_byte": model["flops"] / model["bytes"],
        "executed_gflops": flops / seconds / 1e9,
        "useful_gflops": USEFUL_FLOPS[physics] * nodes / seconds / 1e9,
        "compulsory_gbytes_per_s": nbytes / seconds / 1e9,
        "fraction_of_fp32_peak": flops / seconds / 1e9 / peak,
        "fraction_of_read_bandwidth": nbytes / seconds / 1e9 / bw,
        "roofline_bound_ms": bound * 1e3,
        "roofline_limiter": "compute" if flops / (peak * 1e9) > nbytes / (bw * 1e9) else "bandwidth",
        "achieved_fraction_of_roofline": bound / seconds,
        "bandwidth_level": level,
    }


def _level(nbytes: int, threads: int) -> str:
    if nbytes / threads <= 1.5 * 1024**2:
        return "L2"
    if nbytes <= 200 * 1024**2:
        return "L3"
    return "DRAM"


def _time(fn: Any, *args: Any) -> tuple[np.ndarray, dict[str, Any]]:
    y, samples = fn(*args, 1)
    y, samples = fn(*args, _repeats(samples[0]))
    return np.asarray(y), _stats(samples)


def maxwell_case(mod: Any, rows: int, columns: int, threads: list[int], machine_data: dict[str, Any]) -> dict[str, Any]:
    os.environ["PCB_NATIVE_THREADS"] = str(max(threads))
    op = MatrixFreeScalarMaxwellOperator(maxwell_problem(rows, columns), runtime=NumpyComplex64Runtime(), native=True)
    nat = op._native
    node_shape = (rows + 1, columns + 1)
    n = nat.size
    start = time.perf_counter()
    coef, offsets = _assemble(op.apply_low, node_shape, np.complex64)
    indptr, indices, data = _csr(coef, node_shape, offsets)
    assembly_s = time.perf_counter() - start
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    arrays = (nat._inverse_mu, nat._reaction, nat._stiffness, nat._mass, nat._free, rows, columns)
    by_threads: dict[str, Any] = {}
    for t in threads:
        y_mf, t_mf = _time(mod.maxwell_matrix_free, x, *arrays, t)
        y_st, t_st = _time(mod.maxwell_stencil, x, coef.reshape(-1), node_shape[0], node_shape[1], t)
        y_csr, t_csr = _time(mod.csr_c64, x, indptr, indices, data, t)
        ref = np.linalg.norm(y_mf)
        entry = {}
        for name, y, timing in (("matrix_free", y_mf, t_mf), ("stencil", y_st, t_st), ("csr", y_csr, t_csr)):
            level = _level(MODEL["maxwell"][name]["bytes"] * n, t)
            entry[name] = _row("maxwell", name, n, timing, machine_data, t, level)
            entry[name]["relative_difference_vs_matrix_free"] = float(np.linalg.norm(y - y_mf) / ref)
        entry["stencil_over_matrix_free_time"] = t_st["median_ms"] / t_mf["median_ms"]
        entry["csr_over_matrix_free_time"] = t_csr["median_ms"] / t_mf["median_ms"]
        by_threads[str(t)] = entry
    return {
        "physics": "maxwell",
        "element_shape": [rows, columns],
        "nodes": n,
        "storage_bytes": {
            "matrix_free": int(nat._inverse_mu.nbytes + nat._reaction.nbytes + nat._free.nbytes + 4 * n),
            "stencil": int(coef.nbytes),
            "csr": int(indptr.nbytes + indices.nbytes + data.nbytes),
        },
        "assembly_by_probing_s": assembly_s,
        "by_threads": by_threads,
    }


def thermal_case(mod: Any, elements: int, threads: list[int], machine_data: dict[str, Any]) -> dict[str, Any]:
    problem = thermal_problem(elements)
    op = MatrixFreeThermalOperator(problem, native=True, native_threads=max(threads))
    nat = op._native
    node_shape = (nat.slabs + 1, nat.rows + 1, nat.cols + 1)
    n = nat.size
    start = time.perf_counter()
    coef, offsets = _assemble(op.apply_low, node_shape, np.float32)
    indptr, indices, data = _csr(coef, node_shape, offsets)
    assembly_s = time.perf_counter() - start
    x = np.random.default_rng(0).standard_normal(n).astype(np.float32)
    arrays = (nat._coefficients, nat._unit, nat._robin, nat._free, nat.slabs, nat.rows, nat.cols)
    by_threads: dict[str, Any] = {}
    for t in threads:
        y_mf, t_mf = _time(mod.thermal_matrix_free, x, *arrays, t)
        y_st, t_st = _time(mod.thermal_stencil, x, coef.reshape(-1), *node_shape, t)
        y_csr, t_csr = _time(mod.csr_f32, x, indptr, indices, data, t)
        ref = np.linalg.norm(y_mf)
        entry = {}
        for name, y, timing in (("matrix_free", y_mf, t_mf), ("stencil", y_st, t_st), ("csr", y_csr, t_csr)):
            level = _level(MODEL["thermal"][name]["bytes"] * n, t)
            entry[name] = _row("thermal", name, n, timing, machine_data, t, level)
            entry[name]["relative_difference_vs_matrix_free"] = float(np.linalg.norm(y - y_mf) / ref)
        entry["stencil_over_matrix_free_time"] = t_st["median_ms"] / t_mf["median_ms"]
        entry["csr_over_matrix_free_time"] = t_csr["median_ms"] / t_mf["median_ms"]
        by_threads[str(t)] = entry
    return {
        "physics": "thermal",
        "element_grid_shape": [nat.slabs, nat.rows, nat.cols],
        "nodes": n,
        "storage_bytes": {
            "matrix_free": int(nat._coefficients.nbytes + nat._robin.nbytes + nat._free.nbytes + 4 * n),
            "stencil": int(coef.nbytes),
            "csr": int(indptr.nbytes + indices.nbytes + data.nbytes),
        },
        "assembly_by_probing_s": assembly_s,
        "by_threads": by_threads,
    }


# ----------------------------------------------------------- solver share ---


def solver_share(kernels: list[dict[str, Any]], threads: list[int]) -> list[dict[str, Any]]:
    """Native MPIR solve of the documented fixtures and the operator's share of it."""

    out = []
    for case in kernels:
        if case["physics"] == "maxwell" and case["element_shape"] != [256, 256]:
            continue
        if case["physics"] == "thermal" and case["element_grid_shape"][1] != 200:
            continue
        for t in threads:
            if case["physics"] == "maxwell":
                os.environ["PCB_NATIVE_THREADS"] = str(t)
                op = MatrixFreeScalarMaxwellOperator(maxwell_problem(256, 256), runtime=NumpyComplex64Runtime(), native=True)
                rhs, config = op.build_rhs(), MAXWELL_CONFIG
            else:
                op = MatrixFreeThermalOperator(thermal_problem(200), native=True, native_threads=t)
                rhs, config = op.build_rhs(op.default_reference_temperature()), THERMAL_CONFIG
            solve_mpir(op, rhs, config=config)
            samples = []
            for _ in range(5):
                start = time.perf_counter()
                result = solve_mpir(op, rhs, config=config)
                samples.append(time.perf_counter() - start)
            solve_ms = statistics.median(samples) * 1e3
            probe = np.asarray(rhs).copy()
            op.apply_high(probe)
            high_samples = []
            for _ in range(20):
                start = time.perf_counter()
                op.apply_high(probe)
                high_samples.append(time.perf_counter() - start)
            high_ms = statistics.median(high_samples) * 1e3
            high_apps = int(result.high_operator_applications)
            row = case["by_threads"][str(t)]
            apps = int(result.low_operator_applications)
            mf_ms = row["matrix_free"]["timing"]["median_ms"]
            st_ms = row["stencil"]["timing"]["median_ms"]
            out.append(
                {
                    "physics": case["physics"],
                    "nodes": case["nodes"],
                    "threads": t,
                    "solve_ms": solve_ms,
                    "converged": bool(result.converged),
                    "outer_iterations": int(result.outer_iterations),
                    "inner_iterations": int(result.inner_iterations),
                    "low_operator_applications": apps,
                    "estimated_operator_ms": apps * mf_ms,
                    "estimated_operator_share": apps * mf_ms / solve_ms,
                    "estimated_solve_ms_with_stencil_operator": solve_ms - apps * (mf_ms - st_ms),
                    "high_operator_applications": high_apps,
                    "high_operator_apply_ms": high_ms,
                    "estimated_high_operator_share": high_apps * high_ms / solve_ms,
                    "estimated_other_share": 1.0 - (apps * mf_ms + high_apps * high_ms) / solve_ms,
                    "note": "operator times = applications x isolated action medians (the FP64 high operator is the NumPy path, single threaded); other = inner vector updates, Gram-Schmidt or preconditioner, and outer bookkeeping; the stencil solve is an estimate, not a measured solve",
                }
            )
    return out


# ------------------------------------------------------------------- main ---


def _environment() -> dict[str, Any]:
    def cpu() -> str:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
        return platform.processor()

    compiler = subprocess.run(["g++", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cpu": cpu(),
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cupy": None,
        "device": "cpu",
        "compiler": compiler,
        "flags": "-O3 -march=native -mprefer-vector-width=512 -fcx-limited-range -fopenmp",
        "omp": {k: os.environ.get(k) for k in ("OMP_PROC_BIND", "OMP_PLACES", "OPENBLAS_NUM_THREADS")},
        "hardware_counters": "unavailable in this KVM guest; flops and bytes are analytic",
    }


def _decision(kernels: list[dict[str, Any]], threads: list[int]) -> dict[str, Any]:
    faster = []
    for case in kernels:
        for t in threads:
            ratio = case["by_threads"][str(t)]["stencil_over_matrix_free_time"]
            if ratio < 0.9:
                faster.append({"physics": case["physics"], "nodes": case["nodes"], "threads": t, "stencil_over_matrix_free_time": ratio})
    return {
        "adopted": "none; this is a measurement of the current kernels",
        "assembled_stencil": "measured, not adopted",
        "cases_where_assembled_stencil_is_faster_by_10_percent": faster,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", default="1,4,16")
    parser.add_argument("--machine-threads", default="1,2,4,8,16,32")
    parser.add_argument("--maxwell", default="256x256,1024x1024,4096x4096")
    parser.add_argument("--thermal", default="200,500,1500")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark-results" / "roofline_matrix_free.json")
    args = parser.parse_args()
    threads = [int(v) for v in args.threads.split(",")]
    machine_threads = sorted(set(int(v) for v in args.machine_threads.split(",")) | set(threads))
    mod = _module()
    report: dict[str, Any] = {"benchmark": "roofline_matrix_free", "environment": _environment(), "model": MODEL}
    print("machine ...", flush=True)
    report["machine"] = machine(mod, machine_threads)
    print(json.dumps(report["machine"]["fma_peak"]), flush=True)
    kernels = []
    for item in args.maxwell.split(","):
        r, c = (int(v) for v in item.split("x"))
        print(f"maxwell {r}x{c} ...", flush=True)
        kernels.append(maxwell_case(mod, r, c, threads, report["machine"]))
    for item in args.thermal.split(","):
        print(f"thermal {item} ...", flush=True)
        kernels.append(thermal_case(mod, int(item), threads, report["machine"]))
    report["kernels"] = kernels
    print("solver share ...", flush=True)
    report["solver_share"] = solver_share(kernels, threads)
    report["decision"] = _decision(kernels, threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
