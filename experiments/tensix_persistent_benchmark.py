"""K resident stencil sweeps in one Blackhole program (dataflow execution).

``experiments/tensix_stencil/persistent_*.cpp`` run K Richardson sweeps
``p <- p + (-omega) A p`` of the layered DC Q1 FP32 action (the
dc_native_benchmark board without vias, cyclic tile layout of
tensix_stencil_benchmark.py's v3) inside one ``ttnn.generic_op``:

* both vector buffers live in L1 and are swapped every sweep; each core's
  operator tiles are loaded once into a circular buffer and stay there;
* neighbour tiles come straight from the owners' L1 over the NoC; only
  tiles whose index wraps are shifted on the reader RISC-V;
* after writing its tiles a core increments the coordinator's semaphore;
  the coordinator releases every core's reader for the next sweep.  No DRAM
  traffic, no host and no program launch between sweeps.

omega = 2^-m with 2^m >= 4 max(diag A), so the scaling is exact and the
device result is compared value for value with NumPy's
``p + float32(-omega) * apply_low(p)`` after K sweeps.  The time per sweep is
the slope between K = 10 and K = 50 (launch and the one-off operator load
excluded); the single-launch v3 stencil and the C++ action of the same
operator are the references.

    ~/tt_test/.venv/bin/python experiments/tensix_persistent_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import ttnn  # noqa: E402

from dc_native_benchmark import CONFIG as DC_CONFIG, board  # noqa: E402
from electrical.matrix_free_mpir_fem import MatrixFreePCBOperator, solve_mpir  # noqa: E402
from electrical.matrix_free_mpir_fem.pcb import unit_sheet_matrices  # noqa: E402
from electrical.threads import set_thread_budget  # noqa: E402
from tensix_stencil_benchmark import (  # noqa: E402
    CB_OUT, CB_ST, CB_STAGE, CB_X, CB_XM, KERNELS, TILE, TILE_BYTES, TensixStencil3, _environment, _no_vias, _per_call,
)

SWEEPS = (1, 10, 50)


class TensixPersistent(TensixStencil3):
    """K sweeps resident in L1, synchronised by semaphores through a coordinator core."""

    COMPUTE = "persistent_compute.cpp"
    FULL_SYNC = False  # eight instead of four FP32 DEST tiles

    def compute_compile_time_args(self) -> list[int]:
        return self.unit_bits + [self.minus_omega_bits]

    def __init__(self, device: Any, operator: MatrixFreePCBOperator, cores: tuple[int, int], minus_omega: np.float32,
                 skip: tuple[str, ...] = ()) -> None:
        self.sweeps = 1
        self.parts_skipped = skip
        self.buffers = None
        self.minus_omega_bits = int(np.asarray(minus_omega, dtype=np.float32).view(np.uint32))
        super().__init__(device, operator, cores)

    def _descriptor(self) -> Any:
        if self.operator_tiles is None:
            self.operator_tiles = ttnn.from_torch(torch.from_numpy(self._static()), dtype=ttnn.float32,
                                                  layout=ttnn.TILE_LAYOUT, device=self.device,
                                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.buffers is None:
            self.buffers = [ttnn.from_torch(torch.zeros(self.tile_count * TILE, TILE), dtype=ttnn.float32,
                                            layout=ttnn.TILE_LAYOUT, device=self.device,
                                            memory_config=ttnn.L1_MEMORY_CONFIG) for _ in range(2)]
        grid_x, grid_y = self.cores
        cores = grid_x * grid_y
        ranges = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
        lists = self.assignment()
        noc = [self.device.worker_core_from_logical_core(ttnn.CoreCoord(c % grid_x, c // grid_x)) for c in range(cores)]
        all_coords = [value for coord in noc for value in (coord.x, coord.y)]
        b0, b1 = (buffer.buffer_address() for buffer in self.buffers)
        reader_args, writer_args, compute_args = [], [], []
        for core, tiles in enumerate(lists):
            coord = (core % grid_x, core // grid_x)
            reader_args.append((coord, [b0, b1, self.sweeps, len(tiles)] + tiles))
            writer_args.append((coord, [self.operator_tiles.buffer_address(), b0, b1, self.sweeps, noc[0].x, noc[0].y,
                                        1 if core == 0 else 0, cores, len(tiles)] + tiles
                                + (all_coords if core == 0 else [])))
            compute_args.append((coord, [self.sweeps, len(tiles)]))

        def cb(index: int, size: int, page: int) -> Any:
            return ttnn.CBDescriptor(total_size=size, core_ranges=ranges, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32, page_size=page)])

        def accessor(tensor: Any) -> list[int]:
            return list(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())

        compute_config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                      fp32_dest_acc_en=True)
        compute_config.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
        compute_config.dst_full_sync_en = self.FULL_SYNC
        source = ttnn.KernelDescriptor.SourceType.FILE_PATH
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "persistent_reader.cpp"), source_type=source, core_ranges=ranges,
                compile_time_args=[self.tiles[0], self.tiles[1]] + accessor(self.buffers[0]) + accessor(self.buffers[1]),
                defines=[("STENCIL_SKIP_MOVES", "1")] if "moves" in self.parts_skipped else [],
                runtime_args=reader_args, config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "persistent_writer.cpp"), source_type=source, core_ranges=ranges,
                compile_time_args=accessor(self.operator_tiles) + accessor(self.buffers[0]) + accessor(self.buffers[1]),
                runtime_args=writer_args, config=ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / self.COMPUTE), source_type=source, core_ranges=ranges,
                compile_time_args=self.compute_compile_time_args(), runtime_args=compute_args,
                defines=[("STENCIL_SKIP_MATH", "1")] if "math" in self.parts_skipped else [], config=compute_config),
        ]
        resident = max(1, max(len(tiles) for tiles in lists)) * self.STATIC_TILES
        cbs = [
            cb(CB_X, 2 * 9 * TILE_BYTES, TILE_BYTES),
            cb(CB_ST, resident * TILE_BYTES, TILE_BYTES),
            cb(CB_XM, 9 * TILE_BYTES, TILE_BYTES),
            cb(CB_OUT, 2 * TILE_BYTES, TILE_BYTES),
            cb(CB_STAGE, 8 * TILE_BYTES, 8 * TILE_BYTES),
        ]
        semaphores = [ttnn.SemaphoreDescriptor(id=0, core_type=ttnn.CoreType.WORKER, core_ranges=ranges, initial_value=0),
                      ttnn.SemaphoreDescriptor(id=1, core_type=ttnn.CoreType.WORKER, core_ranges=ranges, initial_value=0)]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)

    def with_sweeps(self, sweeps: int) -> None:
        self.sweeps = int(sweeps)
        self.descriptor = self._descriptor()

    def set_input(self, vector: np.ndarray) -> None:
        staged = ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(self._cyclic(self._padded(vector))).reshape(-1, TILE)),
                                 dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(staged, self.buffers[0])

    def result(self) -> np.ndarray:
        layers, rows, cols = self.node_shape
        tiles = ttnn.to_torch(self.buffers[self.sweeps % 2]).numpy().reshape(layers, self.tiles[0], self.tiles[1], TILE, TILE)
        grid = tiles.transpose(0, 3, 1, 4, 2).reshape(layers, TILE * self.tiles[0], TILE * self.tiles[1])
        return grid[:, :rows, :cols].reshape(-1)

    def apply(self) -> None:
        ttnn.generic_op([self.operator_tiles, self.buffers[0], self.buffers[1]], self.descriptor)


class TensixPersistentWeights(TensixPersistent):
    """Masked element weights precomputed in FP32: 16 multiply-adds per node and sweep, no masking pass.

    Each node keeps the 16 entries W_e[r, col] = c0 U0[r, col] + c1 U1[r, col] of its four
    elements, times the free mask of the coupled node: element-matrix entries per node (not
    summed over elements, so no assembled matrix), 16 floats per node instead of 8 shifted
    coefficients and 9 shifted masks.
    """

    COMPUTE = "persistent_weights_compute.cpp"

    def compute_compile_time_args(self) -> list[int]:
        return [self.minus_omega_bits]

    def _planes(self) -> np.ndarray:
        layers, rows, cols = self.node_shape
        padded_rows, padded_cols = TILE * self.tiles[0], TILE * self.tiles[1]
        free = np.zeros((layers, padded_rows + 2, padded_cols + 2), dtype=np.float32)
        free[:, 1 : 1 + rows, 1 : 1 + cols] = np.asarray(self._operator.free_nodes, dtype=np.float32).reshape(self.node_shape)
        elements = np.stack(self._operator.mesh.element_coefficients()).astype(np.float32)
        coefficients = np.zeros((2, layers, padded_rows + 1, padded_cols + 1), dtype=np.float32)
        coefficients[:, :, 1:rows, 1:cols] = elements
        unit = np.stack(unit_sheet_matrices()).astype(np.float32)
        planes = []
        for r in range(4):
            a, b = r >> 1, r & 1
            c0 = coefficients[0, :, 1 - a : 1 - a + padded_rows, 1 - b : 1 - b + padded_cols]
            c1 = coefficients[1, :, 1 - a : 1 - a + padded_rows, 1 - b : 1 - b + padded_cols]
            for col in range(4):
                di, dj = (col >> 1) - a, (col & 1) - b
                weight = c0 * unit[0, r, col] + c1 * unit[1, r, col]  # FP32, as in the NumPy action
                mask = free[:, 1 + di : 1 + di + padded_rows, 1 + dj : 1 + dj + padded_cols]
                planes.append(weight * mask)
        planes.append(free[:, 1 : 1 + padded_rows, 1 : 1 + padded_cols])
        planes.append(1.0 - free[:, 1 : 1 + padded_rows, 1 : 1 + padded_cols])
        return np.stack(planes, axis=1)


class TensixPersistentBatched(TensixPersistentWeights):
    """The masked weights with eight FP32 DEST tiles, one init per batch of like operations."""

    COMPUTE = "persistent_batched_compute.cpp"
    FULL_SYNC = True


def _sweep_runs(device: Any, kernel: TensixPersistent, portable: MatrixFreePCBOperator, vector: np.ndarray,
                minus_omega: np.float32, label: str, elements: int) -> dict[str, Any]:
    sync = lambda: ttnn.synchronize_device(device)  # noqa: E731
    runs: dict[str, Any] = {}
    for sweeps in SWEEPS:
        kernel.with_sweeps(sweeps)
        kernel.set_input(vector)
        kernel.apply()
        sync()
        on_device = kernel.result()
        reference = vector.copy()
        for _ in range(sweeps):
            reference = (reference + minus_omega * np.asarray(portable.apply_low(reference), dtype=np.float32)).astype(np.float32)
        timing = _per_call(kernel.apply, repeats=5, warmups=2, sync=sync)
        runs[str(sweeps)] = {
            "sweeps": sweeps,
            "program": timing,
            "values_differing_from_numpy_fp32": int(np.count_nonzero(on_device != reference)),
            "relative_change_of_p": float(np.linalg.norm(reference - vector) / np.linalg.norm(vector)),
        }
        print(f"dc {elements:4d} {label} K={sweeps:3d} program={timing['median_us']:9.1f} us "
              f"differ={runs[str(sweeps)]['values_differing_from_numpy_fp32']}", flush=True)
    return runs


def _case(device: Any, elements: int, cores: tuple[int, int], threads: int) -> dict[str, Any]:
    problem = board(elements)
    portable = _no_vias(problem, native=False)
    diagonal = np.asarray(portable.diagonal_low(), dtype=np.float64)
    shift = math.ceil(math.log2(4.0 * float(diagonal.max())))
    minus_omega = np.float32(-(2.0 ** -shift))
    rng = np.random.default_rng(0)
    vector = rng.standard_normal(portable.size).astype(np.float32)
    sync = lambda: ttnn.synchronize_device(device)  # noqa: E731

    persistent = TensixPersistent(device, portable, cores, minus_omega)
    runs = _sweep_runs(device, persistent, portable, vector, minus_omega, "coefficients", elements)
    per_sweep = (runs["50"]["program"]["median_us"] - runs["10"]["program"]["median_us"]) / 40.0
    tiles = persistent.tile_count
    busiest = max(len(assigned) for assigned in persistent.assignment())
    del persistent

    breakdown = {}
    for skip in (("moves",), ("math",), ("moves", "math")):
        partial = TensixPersistent(device, portable, cores, minus_omega, skip)
        programs = {}
        for sweeps in (10, 50):
            partial.with_sweeps(sweeps)
            partial.set_input(vector)
            programs[sweeps] = _per_call(partial.apply, repeats=5, warmups=2, sync=sync)["median_us"]
        breakdown["skip_" + "+".join(skip)] = (programs[50] - programs[10]) / 40.0
        del partial

    weights = TensixPersistentWeights(device, portable, cores, minus_omega)
    weight_runs = _sweep_runs(device, weights, portable, vector, minus_omega, "weights", elements)
    weight_per_sweep = (weight_runs["50"]["program"]["median_us"] - weight_runs["10"]["program"]["median_us"]) / 40.0
    del weights
    batched = TensixPersistentBatched(device, portable, cores, minus_omega)
    batched_runs = _sweep_runs(device, batched, portable, vector, minus_omega, "batched", elements)
    batched_per_sweep = (batched_runs["50"]["program"]["median_us"] - batched_runs["10"]["program"]["median_us"]) / 40.0
    batched_breakdown = {}
    for skip in (("moves",), ("math",), ("moves", "math")):
        partial = TensixPersistentBatched(device, portable, cores, minus_omega, skip)
        programs = {}
        for sweeps in (10, 50):
            partial.with_sweeps(sweeps)
            partial.set_input(vector)
            programs[sweeps] = _per_call(partial.apply, repeats=5, warmups=2, sync=sync)["median_us"]
        batched_breakdown["skip_" + "+".join(skip)] = (programs[50] - programs[10]) / 40.0
        del partial
    del batched

    single = TensixStencil3(device, portable, cores)
    single.set_input(vector)
    single.apply()
    sync()
    single_time = _per_call(single.apply, repeats=50, warmups=3, sync=sync)
    del single

    set_thread_budget(threads)
    native = _no_vias(problem, native=True)
    native_apply = _per_call(lambda: native.apply_low(vector), repeats=50, warmups=5)
    full = MatrixFreePCBOperator(problem.mesh, reference_node=problem.reference_node, vias=problem.vias, native=True)
    rhs = full.build_rhs(problem.terminals)
    solve_mpir(full, rhs, config=DC_CONFIG)
    start = time.perf_counter()
    solved = solve_mpir(full, rhs, config=DC_CONFIG)
    per_iteration = (time.perf_counter() - start) * 1.0e6 / max(1, int(solved.inner_iterations))
    set_thread_budget(None)
    print(f"dc {elements:4d} per-sweep={per_sweep:.1f} us weights per-sweep={weight_per_sweep:.1f} us "
          f"batched per-sweep={batched_per_sweep:.1f} us ("
          + " ".join(f"{name}={value:.1f}" for name, value in batched_breakdown.items())
          + f") single-launch v3={single_time['median_us']:.1f} us "
          f"native apply={native_apply['median_us']:.1f} us native inner iteration={per_iteration:.1f} us "
          + " ".join(f"{name}={value:.1f}" for name, value in breakdown.items()), flush=True)
    return {
        "elements": [elements, elements],
        "nodes": int(portable.size),
        "output_tiles": tiles,
        "cores": list(cores),
        "most_tiles_per_core": busiest,
        "omega": float(-minus_omega),
        "runs": runs,
        "device_us_per_sweep": per_sweep,
        "device_us_per_sweep_breakdown": breakdown,
        "weights_runs": weight_runs,
        "weights_device_us_per_sweep": weight_per_sweep,
        "weights_sweep_over_native_apply": weight_per_sweep / native_apply["median_us"],
        "batched_runs": batched_runs,
        "batched_device_us_per_sweep": batched_per_sweep,
        "batched_device_us_per_sweep_breakdown": batched_breakdown,
        "batched_sweep_over_native_apply": batched_per_sweep / native_apply["median_us"],
        "single_launch_v3_apply": single_time,
        "native_threads": threads,
        "native_apply_low": native_apply,
        "native_full_board_us_per_inner_iteration": per_iteration,
        "sweep_over_native_apply": per_sweep / native_apply["median_us"],
    }


def run(device_id: int, sizes: list[int], threads: int) -> dict[str, Any]:
    device = ttnn.open_device(device_id=device_id)
    try:
        environment = _environment(device, device_id)
        grid = device.compute_with_storage_grid_size()
        cases = [_case(device, size, (int(grid.x), int(grid.y)), threads) for size in sizes]
    finally:
        ttnn.close_device(device)
    return {
        "schema": "tensix-persistent-benchmark/v1",
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "environment": environment,
        "definitions": {
            "operator": "MatrixFreePCBOperator FP32 action of the dc_native_benchmark board without vias (two layers)",
            "sweep": "p <- p + (-omega) A p with omega = 2^-m, 2^m >= 4 max(diag A); A p as in tensix_stencil_benchmark v3",
            "program": "wall time per ttnn.generic_op call running K sweeps, synchronised per sample",
            "device_us_per_sweep": "(program(K=50) - program(K=10)) / 40",
            "device_us_per_sweep_breakdown": "the same slope with parts compiled out (wrong results): moves = the reader's wrap shifts, math = the SFPU masking and stencil",
            "weights_device_us_per_sweep": "the same slope with the masked element weights precomputed in FP32 (16 per node) instead of shifted coefficients and masks",
            "batched_device_us_per_sweep": "the masked weights with eight FP32 DEST tiles (dst_full_sync_en) and one init per batch of like SFPU operations",
            "values_differing_from_numpy_fp32": "nodes whose device value after K sweeps differs (as a float) from NumPy's K sweeps with apply_low(native=False)",
            "native_apply_low": "C++ apply_low of the same operator at native_threads",
            "native_full_board_us_per_inner_iteration": "C++ MPIR solve of the board with vias divided by its inner iterations",
        },
        "cases": cases,
        "decision": {
            "adopted": False,
            "scope": "measurement only; nothing in src/ changes",
            "bitwise_equal_to_numpy_fp32": all(run["values_differing_from_numpy_fp32"] == 0
                                               for case in cases for key in ("runs", "weights_runs", "batched_runs")
                                               for run in case[key].values()),
            "weights_sweep_over_native_apply": {str(case["elements"][0]): case["weights_sweep_over_native_apply"]
                                                for case in cases},
            "batched_sweep_over_native_apply": {str(case["elements"][0]): case["batched_sweep_over_native_apply"]
                                                for case in cases},
            "sweep_over_native_apply": {str(case["elements"][0]): case["sweep_over_native_apply"] for case in cases},
            "sweep_faster_than_native_apply": all(min(case["sweep_over_native_apply"], case["weights_sweep_over_native_apply"],
                                                      case["batched_sweep_over_native_apply"]) < 1.0 for case in cases),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--sizes", default="100,200,320,480,640")
    parser.add_argument("--threads", type=int, default=8, help="OpenMP thread budget of the C++ reference")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results") / "tensix_persistent_benchmark.json")
    args = parser.parse_args()
    report = run(args.device_id, [int(item) for item in args.sizes.split(",") if item], args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
