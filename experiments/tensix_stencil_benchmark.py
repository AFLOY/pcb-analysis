"""One FP32 layered DC Q1 stencil application on Blackhole Tensix cores.

The kernels in experiments/tensix_stencil/ apply ``MatrixFreePCBOperator``'s FP32
action (two-layer DC board of dc_native_benchmark.py, without vias) to one vector.
v1 (``stencil_*.cpp``) builds everything per call:

* reader: per output tile of 32 x 32 nodes, aligned 128-byte page reads of the
  34 x 34 node patch (input and free mask) and the 33 x 33 element patch (c0, c1),
  then integer word moves on the RISC-V into nine shifted, masked input tiles and
  eight shifted coefficient tiles;
* compute: the SFPU with FP32 DEST, in the operation order of the NumPy action,
  ``out = sum_r sum_col (c0 U0[r, col] + c1 U1[r, col]) x_col`` and
  ``where(free, out, x)`` as ``out * free + x_fixed``;
* writer: one FP32 tile per output tile.

v2 (``stencil2_*.cpp``) keeps the operator data, which does not change during an
inner solve, as 18 precomputed tiles per output tile in L1 (8 shifted element
coefficients, 9 shifted free masks, the fixed-node mask; 4x the element
coefficients' storage, no element matrix and no assembled stencil): the reader
shifts only the input (plain word moves), the writer streams the operator tiles,
and the compute kernel masks the nine input tiles on the SFPU before the same
operation sequence.

The device result is compared value for value with the portable NumPy FP32
action and against the FP64 action.  Device times are for the program alone
(inputs resident in DRAM in the kernels' layout, synchronised per sample), with
the host's NumPy and C++ (OpenMP) actions of the same operator, and the C++
inner-iteration time of the full board with vias, as references.

    ~/tt_test/.venv/bin/python experiments/tensix_stencil_benchmark.py
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import ttnn  # noqa: E402

from dc_native_benchmark import CONFIG as DC_CONFIG, board  # noqa: E402
from electrical.matrix_free_mpir_fem import MatrixFreePCBOperator, solve_mpir  # noqa: E402
from electrical.matrix_free_mpir_fem.pcb import unit_sheet_matrices  # noqa: E402
from electrical.threads import set_thread_budget  # noqa: E402

KERNELS = Path(__file__).resolve().parent / "tensix_stencil"
TILE = 32
TILE_BYTES = 4096
CB_X, CB_C, CB_S, CB_ST, CB_XM, CB_STAGE, CB_OUT = 0, 1, 2, 3, 4, 8, 16
STAGE_BYTES = 2 * 34 * 96 * 4 + 2 * 33 * 64 * 4
TRACE_REGION_BYTES = 4 << 20


def _per_call(fn: Callable[[], Any], repeats: int, warmups: int, sync: Callable[[], None] | None = None) -> dict[str, Any]:
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


class TensixStencil:
    """Device layout of one layered DC operator and its stencil program."""

    def __init__(self, device: Any, operator: MatrixFreePCBOperator, cores: tuple[int, int],
                 skip: tuple[str, ...] = ()) -> None:
        mesh = operator.mesh
        self.device = device
        self.node_shape = tuple(int(axis) for axis in mesh.node_shape)
        layers, rows, cols = self.node_shape
        self.tiles = (-(-rows // TILE), -(-cols // TILE))
        self.storage_rows = TILE * self.tiles[0] + 2
        self.chunks = self.tiles[1] + 2
        self.tile_count = layers * self.tiles[0] * self.tiles[1]
        coefficients = np.stack(mesh.element_coefficients()).astype(np.float32)
        self.c0 = self._upload(coefficients[0])
        self.c1 = self._upload(coefficients[1])
        self.mask = self._upload(np.asarray(operator.free_nodes, dtype=np.float32).reshape(self.node_shape))
        self.x = self._upload(np.zeros(self.node_shape, dtype=np.float32))
        self.output = ttnn.allocate_tensor_on_device(
            ttnn.Shape((layers * self.tiles[0] * TILE, self.tiles[1] * TILE)), ttnn.float32, ttnn.TILE_LAYOUT, device,
            ttnn.DRAM_MEMORY_CONFIG)
        unit = np.stack(unit_sheet_matrices()).astype(np.float32)
        self.unit_bits = [int(bits) for bits in unit.reshape(-1).view(np.uint32)]
        self.cores = cores
        self.skip = skip
        self.descriptor = self._descriptor()

    def _storage(self, grid: np.ndarray) -> np.ndarray:
        layers = self.node_shape[0]
        storage = np.zeros((layers, self.storage_rows, self.chunks * TILE), dtype=np.float32)
        storage[:, 1 : 1 + grid.shape[1], TILE : TILE + grid.shape[2]] = grid
        return storage.reshape(-1, TILE)

    def _upload(self, grid: np.ndarray) -> Any:
        return ttnn.from_torch(torch.from_numpy(self._storage(grid)), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def set_input(self, vector: np.ndarray) -> None:
        staged = ttnn.from_torch(torch.from_numpy(self._storage(np.asarray(vector, np.float32).reshape(self.node_shape))),
                                 dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(staged, self.x)

    def result(self) -> np.ndarray:
        layers, rows, cols = self.node_shape
        grid = ttnn.to_torch(self.output).numpy().reshape(layers, self.tiles[0] * TILE, self.tiles[1] * TILE)
        return grid[:, :rows, :cols].reshape(-1)

    def _descriptor(self) -> Any:
        grid_x, grid_y = self.cores
        cores = grid_x * grid_y
        ranges = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
        base, extra = divmod(self.tile_count, cores)
        reader_args, writer_args, compute_args = [], [], []
        first = 0
        for core in range(cores):
            count = base + (1 if core < extra else 0)
            coord = (core % grid_x, core // grid_x)
            reader_args.append((coord, [self.x.buffer_address(), self.mask.buffer_address(), self.c0.buffer_address(),
                                        self.c1.buffer_address(), first, count]))
            writer_args.append((coord, [self.output.buffer_address(), first, count]))
            compute_args.append((coord, [count]))
            first += count

        def cb(index: int, size: int, page: int) -> Any:
            return ttnn.CBDescriptor(total_size=size, core_ranges=ranges, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32, page_size=page)])

        def accessor(tensor: Any) -> list[int]:
            return list(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())

        compute_config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                      fp32_dest_acc_en=True)
        compute_config.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
        source = ttnn.KernelDescriptor.SourceType.FILE_PATH
        reader_defines = [(f"STENCIL_SKIP_{part.upper()}", "1") for part in self.skip if part in ("reads", "moves")]
        compute_defines = [("STENCIL_SKIP_MATH", "1")] if "math" in self.skip else []
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "stencil_reader.cpp"), source_type=source, core_ranges=ranges,
                compile_time_args=[self.tiles[0], self.tiles[1], self.storage_rows, self.chunks]
                + accessor(self.x) + accessor(self.mask) + accessor(self.c0) + accessor(self.c1),
                defines=reader_defines, runtime_args=reader_args, config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "stencil_writer.cpp"), source_type=source, core_ranges=ranges,
                compile_time_args=accessor(self.output), runtime_args=writer_args, config=ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "stencil_compute.cpp"), source_type=source, core_ranges=ranges,
                compile_time_args=self.unit_bits, defines=compute_defines, runtime_args=compute_args, config=compute_config),
        ]
        cbs = [
            cb(CB_X, 2 * 9 * TILE_BYTES, TILE_BYTES),
            cb(CB_C, 2 * 8 * TILE_BYTES, TILE_BYTES),
            cb(CB_S, 2 * 2 * TILE_BYTES, TILE_BYTES),
            cb(CB_OUT, 2 * TILE_BYTES, TILE_BYTES),
            cb(CB_STAGE, STAGE_BYTES, STAGE_BYTES),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    def apply(self) -> None:
        ttnn.generic_op([self.x, self.mask, self.c0, self.c1, self.output], self.descriptor)


class TensixStencil2(TensixStencil):
    """v2: operator tiles precomputed once and resident in L1; only the input is shifted."""

    STATIC_TILES = 18

    def __init__(self, device: Any, operator: MatrixFreePCBOperator, cores: tuple[int, int],
                 skip: tuple[str, ...] = ()) -> None:
        self.operator_tiles = None
        self._operator = operator
        super().__init__(device, operator, cores, skip)

    def _planes(self) -> np.ndarray:
        """The 18 operator fields on the padded node grid, (layers, 18, 32 T_r, 32 T_c)."""

        layers, rows, cols = self.node_shape
        padded_rows, padded_cols = TILE * self.tiles[0], TILE * self.tiles[1]
        free = np.zeros((layers, padded_rows + 2, padded_cols + 2), dtype=np.float32)
        free[:, 1 : 1 + rows, 1 : 1 + cols] = np.asarray(self._operator.free_nodes, dtype=np.float32).reshape(self.node_shape)
        elements = np.stack(self._operator.mesh.element_coefficients()).astype(np.float32)
        coefficients = np.zeros((2, layers, padded_rows + 1, padded_cols + 1), dtype=np.float32)
        coefficients[:, :, 1:rows, 1:cols] = elements
        planes = []
        for r in range(4):  # element (i - a, j - b) of the node's local position r = 2 a + b
            a, b = r >> 1, r & 1
            for which in range(2):
                planes.append(coefficients[which, :, 1 - a : 1 - a + padded_rows, 1 - b : 1 - b + padded_cols])
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                planes.append(free[:, 1 + di : 1 + di + padded_rows, 1 + dj : 1 + dj + padded_cols])
        planes.append(1.0 - free[:, 1 : 1 + padded_rows, 1 : 1 + padded_cols])
        return np.stack(planes, axis=1)

    def _static(self) -> np.ndarray:
        layers = self.node_shape[0]
        tiles = self._planes().reshape(layers, self.STATIC_TILES, self.tiles[0], TILE, self.tiles[1], TILE)
        tiles = tiles.transpose(0, 2, 4, 1, 3, 5)  # (layers, T_r, T_c, 18, 32, 32)
        return np.ascontiguousarray(tiles).reshape(-1, TILE)

    def _descriptor(self) -> Any:
        if self.operator_tiles is None:
            self.operator_tiles = ttnn.from_torch(torch.from_numpy(self._static()), dtype=ttnn.float32,
                                                  layout=ttnn.TILE_LAYOUT, device=self.device,
                                                  memory_config=ttnn.L1_MEMORY_CONFIG)
        grid_x, grid_y = self.cores
        cores = grid_x * grid_y
        ranges = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
        base, extra = divmod(self.tile_count, cores)
        reader_args, writer_args, compute_args = [], [], []
        first = 0
        for core in range(cores):
            count = base + (1 if core < extra else 0)
            coord = (core % grid_x, core // grid_x)
            reader_args.append((coord, [self.x.buffer_address(), first, count]))
            writer_args.append((coord, [self.operator_tiles.buffer_address(), self.output.buffer_address(), first, count]))
            compute_args.append((coord, [count]))
            first += count

        def cb(index: int, size: int, page: int) -> Any:
            return ttnn.CBDescriptor(total_size=size, core_ranges=ranges, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32, page_size=page)])

        def accessor(tensor: Any) -> list[int]:
            return list(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())

        compute_config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                      fp32_dest_acc_en=True)
        compute_config.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
        source = ttnn.KernelDescriptor.SourceType.FILE_PATH

        def defines(*parts: str) -> list[tuple[str, str]]:
            return [(f"STENCIL_SKIP_{part.upper()}", "1") for part in self.skip if part in parts]

        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "stencil2_reader.cpp"), source_type=source, core_ranges=ranges,
                compile_time_args=[self.tiles[0], self.tiles[1], self.storage_rows, self.chunks] + accessor(self.x),
                defines=defines("reads", "moves"), runtime_args=reader_args, config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "stencil2_writer.cpp"), source_type=source, core_ranges=ranges,
                compile_time_args=accessor(self.operator_tiles) + accessor(self.output), defines=defines("static"),
                runtime_args=writer_args, config=ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "stencil2_compute.cpp"), source_type=source, core_ranges=ranges,
                compile_time_args=self.unit_bits, defines=defines("mask", "math"), runtime_args=compute_args,
                config=compute_config),
        ]
        cbs = [
            cb(CB_X, 2 * 9 * TILE_BYTES, TILE_BYTES),
            cb(CB_ST, 2 * self.STATIC_TILES * TILE_BYTES, TILE_BYTES),
            cb(CB_XM, 9 * TILE_BYTES, TILE_BYTES),
            cb(CB_OUT, 2 * TILE_BYTES, TILE_BYTES),
            cb(CB_STAGE, 34 * 96 * 4, 34 * 96 * 4),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    def apply(self) -> None:
        ttnn.generic_op([self.x, self.operator_tiles, self.output], self.descriptor)


class TensixStencil3(TensixStencil2):
    """v3: cyclic tile layout; whole neighbour tiles, shifts only where the tile index wraps."""

    def __init__(self, device: Any, operator: MatrixFreePCBOperator, cores: tuple[int, int]) -> None:
        self.x_tiles = None
        super().__init__(device, operator, cores)

    def _cyclic(self, fields: np.ndarray) -> np.ndarray:
        """(..., 32 T_r, 32 T_c) -> (..., T_r, T_c, 32, 32) with node (r T_r + p, c T_c + q) at tile (p, q), (r, c)."""

        lead = fields.shape[:-2]
        split = fields.reshape(*lead, TILE, self.tiles[0], TILE, self.tiles[1])
        n = len(lead)
        return split.transpose(*range(n), n + 1, n + 3, n, n + 2)

    def _padded(self, grid: np.ndarray) -> np.ndarray:
        layers, rows, cols = self.node_shape
        padded = np.zeros((layers, TILE * self.tiles[0], TILE * self.tiles[1]), dtype=np.float32)
        padded[:, :rows, :cols] = np.asarray(grid, dtype=np.float32).reshape(self.node_shape)
        return padded

    def _static(self) -> np.ndarray:
        tiles = self._cyclic(self._planes())  # (layers, 18, T_r, T_c, 32, 32)
        return np.ascontiguousarray(tiles.transpose(0, 2, 3, 1, 4, 5)).reshape(-1, TILE)

    def set_input(self, vector: np.ndarray) -> None:
        staged = ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(self._cyclic(self._padded(vector))).reshape(-1, TILE)),
                                 dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(staged, self.x_tiles)

    def result(self) -> np.ndarray:
        layers, rows, cols = self.node_shape
        flat = ttnn.to_torch(self.output).numpy().reshape(layers, self.tiles[0], TILE, self.tiles[1], TILE)
        tiles = flat.transpose(0, 1, 3, 2, 4)  # (layers, T_r, T_c, 32, 32) by tile id
        grid = tiles.transpose(0, 3, 1, 4, 2).reshape(layers, TILE * self.tiles[0], TILE * self.tiles[1])
        return grid[:, :rows, :cols].reshape(-1)

    def wrapped_neighbours(self, tile: int) -> int:
        p = (tile // self.tiles[1]) % self.tiles[0]
        q = tile % self.tiles[1]
        return sum(1 for di in (-1, 0, 1) for dj in (-1, 0, 1)
                   if not (0 <= p + di < self.tiles[0]) or not (0 <= q + dj < self.tiles[1]))

    def assignment(self) -> list[list[int]]:
        """Longest-processing-time assignment; a wrapped neighbour costs half a tile of compute."""

        cores = self.cores[0] * self.cores[1]
        costs = [(1.0 + 0.5 * self.wrapped_neighbours(t), t) for t in range(self.tile_count)]
        load = [0.0] * cores
        lists: list[list[int]] = [[] for _ in range(cores)]
        for cost, tile in sorted(costs, reverse=True):
            core = min(range(cores), key=lambda c: (load[c], c))
            load[core] += cost
            lists[core].append(tile)
        return [sorted(tiles) for tiles in lists]

    def _descriptor(self) -> Any:
        if self.operator_tiles is None:
            self.operator_tiles = ttnn.from_torch(torch.from_numpy(self._static()), dtype=ttnn.float32,
                                                  layout=ttnn.TILE_LAYOUT, device=self.device,
                                                  memory_config=ttnn.L1_MEMORY_CONFIG)
        if self.x_tiles is None:
            self.x_tiles = ttnn.from_torch(torch.zeros(self.tile_count * TILE, TILE), dtype=ttnn.float32,
                                           layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        grid_x, grid_y = self.cores
        ranges = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
        reader_args, writer_args, compute_args = [], [], []
        for core, tiles in enumerate(self.assignment()):
            coord = (core % grid_x, core // grid_x)
            reader_args.append((coord, [self.x_tiles.buffer_address(), len(tiles)] + tiles))
            writer_args.append((coord, [self.operator_tiles.buffer_address(), self.output.buffer_address(), len(tiles)] + tiles))
            compute_args.append((coord, [len(tiles)]))

        def cb(index: int, size: int, page: int) -> Any:
            return ttnn.CBDescriptor(total_size=size, core_ranges=ranges, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32, page_size=page)])

        def accessor(tensor: Any) -> list[int]:
            return list(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())

        compute_config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                      fp32_dest_acc_en=True)
        compute_config.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
        source = ttnn.KernelDescriptor.SourceType.FILE_PATH
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "stencil3_reader.cpp"), source_type=source, core_ranges=ranges,
                compile_time_args=[self.tiles[0], self.tiles[1]] + accessor(self.x_tiles),
                runtime_args=reader_args, config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "stencil2_writer.cpp"), source_type=source, core_ranges=ranges,
                compile_time_args=accessor(self.operator_tiles) + accessor(self.output),
                defines=[("STENCIL_TILE_LIST", "1")], runtime_args=writer_args, config=ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "stencil2_compute.cpp"), source_type=source, core_ranges=ranges,
                compile_time_args=self.unit_bits, runtime_args=compute_args, config=compute_config),
        ]
        cbs = [
            cb(CB_X, 2 * 9 * TILE_BYTES, TILE_BYTES),
            cb(CB_ST, 2 * self.STATIC_TILES * TILE_BYTES, TILE_BYTES),
            cb(CB_XM, 9 * TILE_BYTES, TILE_BYTES),
            cb(CB_OUT, 2 * TILE_BYTES, TILE_BYTES),
            cb(CB_STAGE, 8 * TILE_BYTES, 8 * TILE_BYTES),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    def apply(self) -> None:
        ttnn.generic_op([self.x_tiles, self.operator_tiles, self.output], self.descriptor)


def _no_vias(problem: Any, **options: Any) -> MatrixFreePCBOperator:
    # Without vias the two layers are not connected; the stencil needs no coarse space.
    return MatrixFreePCBOperator(problem.mesh, reference_node=problem.reference_node, vias=(), preconditioner="jacobi",
                                 **options)


def _case(device: Any, elements: int, cores: tuple[int, int], threads: list[int]) -> dict[str, Any]:
    problem = board(elements)
    portable = _no_vias(problem, native=False)
    rng = np.random.default_rng(0)
    vector = rng.standard_normal(portable.size).astype(np.float32)
    reference_low = np.asarray(portable.apply_low(vector), dtype=np.float32)
    reference_high = portable.apply_high(vector.astype(np.float64))

    stencil = TensixStencil(device, portable, cores)
    stencil.set_input(vector)
    stencil.apply()
    ttnn.synchronize_device(device)
    on_device = stencil.result()
    differs = on_device != reference_low
    sync = lambda: ttnn.synchronize_device(device)  # noqa: E731
    device_time = _per_call(stencil.apply, repeats=50, warmups=3, sync=sync)
    upload = _per_call(lambda: stencil.set_input(vector), repeats=10, warmups=1, sync=sync)

    second = TensixStencil2(device, portable, cores)
    second.set_input(vector)
    second.apply()
    sync()
    on_device_v2 = second.result()
    differs_v2 = on_device_v2 != reference_low
    device_time_v2 = _per_call(second.apply, repeats=50, warmups=3, sync=sync)
    del second
    third = TensixStencil3(device, portable, cores)
    third.set_input(vector)
    third.apply()
    sync()
    on_device_v3 = third.result()
    differs_v3 = on_device_v3 != reference_low
    device_time_v3 = _per_call(third.apply, repeats=50, warmups=3, sync=sync)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    third.apply()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    traced_v3 = _per_call(lambda: ttnn.execute_trace(device, trace, cq_id=0, blocking=False), repeats=50, warmups=3,
                          sync=sync)
    ttnn.release_trace(device, trace)
    empty = TensixStencil2(device, portable, cores, ("reads", "moves", "static", "mask", "math"))
    empty.apply()  # a trace can only replay programs already in the program cache
    sync()
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    empty.apply()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    traced_empty = _per_call(lambda: ttnn.execute_trace(device, trace, cq_id=0, blocking=False), repeats=50, warmups=3,
                             sync=sync)
    ttnn.release_trace(device, trace)
    del third, empty

    breakdown_v2 = {}
    for skip in (("moves",), ("static",), ("mask", "math"), ("reads", "moves", "static"),
                 ("reads", "moves", "static", "mask", "math")):
        partial = TensixStencil2(device, portable, cores, skip)
        breakdown_v2["skip_" + "+".join(skip)] = _per_call(partial.apply, repeats=50, warmups=3, sync=sync)
        del partial

    breakdown = {}
    for skip in (("math",), ("moves",), ("reads", "moves"), ("reads", "moves", "math")):
        partial = TensixStencil(device, portable, cores, skip)
        breakdown["skip_" + "+".join(skip)] = _per_call(partial.apply, repeats=50, warmups=3, sync=sync)
        del partial

    portable_time = _per_call(lambda: portable.apply_low(vector), repeats=10, warmups=2)
    native: dict[str, Any] = {}
    for count in threads:
        set_thread_budget(count)
        native_operator = _no_vias(problem, native=True)
        native_low = np.asarray(native_operator.apply_low(vector), dtype=np.float32)
        full = MatrixFreePCBOperator(problem.mesh, reference_node=problem.reference_node, vias=problem.vias, native=True)
        rhs = full.build_rhs(problem.terminals)
        solve_mpir(full, rhs, config=DC_CONFIG)
        start = time.perf_counter()
        solved = solve_mpir(full, rhs, config=DC_CONFIG)
        solve_us = (time.perf_counter() - start) * 1.0e6
        native[str(count)] = {
            "threads": count,
            "apply_low": _per_call(lambda: native_operator.apply_low(vector), repeats=50, warmups=5),
            "values_differing_from_numpy": int(np.count_nonzero(native_low != reference_low)),
            "full_board_solve_us": solve_us,
            "full_board_inner_iterations": int(solved.inner_iterations),
            "full_board_us_per_inner_iteration": solve_us / max(1, int(solved.inner_iterations)),
        }
    set_thread_budget(None)
    best_native = min(entry["apply_low"]["median_us"] for entry in native.values())
    print(f"dc {elements:4d} nodes={portable.size} tiles={stencil.tile_count} device={device_time['median_us']:.1f} us "
          f"v2={device_time_v2['median_us']:.1f} us (differ {int(differs_v2.sum())}) "
          f"v3={device_time_v3['median_us']:.1f} us (differ {int(differs_v3.sum())}) "
          f"v3-traced={traced_v3['median_us']:.1f} us empty-traced={traced_empty['median_us']:.1f} us "
          f"numpy={portable_time['median_us']:.1f} us native-best={best_native:.1f} us "
          f"differ={int(differs.sum())}/{differs.size} "
          + " ".join(f"{name}={entry['median_us']:.1f}" for name, entry in breakdown.items())
          + " | v2 " + " ".join(f"{name}={entry['median_us']:.1f}" for name, entry in breakdown_v2.items()), flush=True)
    return {
        "elements": [elements, elements],
        "nodes": int(portable.size),
        "node_shape": list(stencil.node_shape),
        "output_tiles": stencil.tile_count,
        "cores": list(cores),
        "device_apply": device_time,
        "device_breakdown": breakdown,
        "v2_device_apply": device_time_v2,
        "v2_device_breakdown": breakdown_v2,
        "v3_device_apply": device_time_v3,
        "v3_values_differing_from_numpy_fp32": int(differs_v3.sum()),
        "v3_over_best_native_apply": device_time_v3["median_us"] / best_native,
        "v3_traced_apply": traced_v3,
        "v3_traced_over_best_native_apply": traced_v3["median_us"] / best_native,
        "traced_empty_program": traced_empty,
        "v2_values_differing_from_numpy_fp32": int(differs_v2.sum()),
        "v2_over_best_native_apply": device_time_v2["median_us"] / best_native,
        "device_input_upload": upload,
        "values_differing_from_numpy_fp32": int(differs.sum()),
        "maximum_abs_difference_from_numpy_fp32": float(np.abs(on_device - reference_low).max()),
        "device_relative_error_vs_fp64": float(np.linalg.norm(on_device - reference_high) / np.linalg.norm(reference_high)),
        "numpy_fp32_relative_error_vs_fp64": float(np.linalg.norm(reference_low - reference_high) / np.linalg.norm(reference_high)),
        "numpy_apply_low": portable_time,
        "native": native,
        "device_over_best_native_apply": device_time["median_us"] / best_native,
    }


def _environment(device: Any, device_id: int) -> dict[str, Any]:
    grid = device.compute_with_storage_grid_size()
    try:
        driver = Path("/sys/module/tenstorrent/version").read_text().strip()
    except OSError:
        driver = None
    cpu = platform.processor()
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "ttnn": importlib.metadata.version("ttnn"),
        "tenstorrent_driver": driver,
        "device": f"Tenstorrent {device.arch()}",
        "device_id": device_id,
        "compute_grid": [grid.x, grid.y],
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
    }


def run(device_id: int, sizes: list[int], threads: list[int]) -> dict[str, Any]:
    device = ttnn.open_device(device_id=device_id, trace_region_size=TRACE_REGION_BYTES)
    try:
        environment = _environment(device, device_id)
        grid = device.compute_with_storage_grid_size()
        cores = (int(grid.x), int(grid.y))
        cases = [_case(device, size, cores, threads) for size in sizes]
    finally:
        ttnn.close_device(device)
    return {
        "schema": "tensix-stencil-benchmark/v1",
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "environment": environment,
        "definitions": {
            "operator": "MatrixFreePCBOperator FP32 action of the dc_native_benchmark board without vias (two layers)",
            "device_apply": "wall time per ttnn.generic_op call of the v1 stencil program, inputs resident in DRAM, synchronised per sample",
            "v2_device_apply": "the same for v2: operator tiles resident in L1, input in DRAM",
            "v3_device_apply": "the same for v3: cyclic tile layout (node (r T_r + p, c T_c + q) at tile (p, q), position (r, c)), whole neighbour tiles, shifts only where the tile index wraps, tiles balanced over cores",
            "v3_traced_apply": "v3 replayed from a ttnn trace; traced_empty_program is the v2 program with every part compiled out, replayed the same way",
            "v2_device_breakdown": "v2 with parts compiled out (wrong results): moves (input tile building), reads (input page reads), static (operator tile reads), mask (SFPU masking), math (SFPU stencil)",
            "device_input_upload": "host-to-device copy of the input in the kernels' padded row-major layout",
            "device_breakdown": "the same program with parts compiled out (wrong results): skip_math packs DEST without SFPU work, skip_moves pushes CBs without building tiles, skip_reads drops the DRAM page reads",
            "values_differing_from_numpy_fp32": "nodes whose device value differs (as a float, so -0 == +0) from MatrixFreePCBOperator.apply_low with native=False",
            "native": "C++ apply_low of the same operator per OpenMP thread budget; full_board_* is the C++ MPIR solve of the board with vias",
        },
        "cases": cases,
        "decision": {
            "adopted": False,
            "scope": "measurement only; nothing in src/ changes",
            "bitwise_equal_to_numpy_fp32": all(case["values_differing_from_numpy_fp32"] == 0
                                               and case["v2_values_differing_from_numpy_fp32"] == 0
                                               and case["v3_values_differing_from_numpy_fp32"] == 0 for case in cases),
            "v2_over_best_native_apply": {str(case["elements"][0]): case["v2_over_best_native_apply"] for case in cases},
            "device_over_best_native_apply": {str(case["elements"][0]): case["device_over_best_native_apply"] for case in cases},
            "v3_over_best_native_apply": {str(case["elements"][0]): case["v3_over_best_native_apply"] for case in cases},
            "v3_traced_over_best_native_apply": {str(case["elements"][0]): case["v3_traced_over_best_native_apply"]
                                                 for case in cases},
            "device_faster_than_best_native": all(min(case["v3_over_best_native_apply"], case["v3_traced_over_best_native_apply"]) < 1.0
                                                  for case in cases),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--sizes", default="100,200,320", help="comma separated in-plane element counts of the DC board")
    parser.add_argument("--threads", default="1,8,16", help="comma separated OpenMP thread budgets for the C++ reference")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results") / "tensix_stencil_benchmark.json")
    args = parser.parse_args()
    numbers = lambda text: [int(item) for item in text.split(",") if item]  # noqa: E731
    report = run(args.device_id, numbers(args.sizes), numbers(args.threads))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
