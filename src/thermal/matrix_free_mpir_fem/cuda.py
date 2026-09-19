"""Fused CUDA kernel for the matrix-free hexahedral heat-conduction action.

The NumPy/CuPy array formulation of the trilinear Q1 action issues 64 corner
products per application; on a GPU that becomes a few hundred small kernel
launches and the action is launch-bound below roughly a million nodes.  This
kernel follows the electrical package's node-owned gather: one thread owns one
output node, visits its at most eight adjacent elements, and writes exactly one
value.  There are no global atomics and the summation order is fixed, so
repeated applications are bitwise identical.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_THERMAL_Q1_SOURCE = r"""
extern "C" __global__
void thermal_hex_q1_apply(
    const float* vector,
    float* output,
    const float* coefficients,    // (3, slabs, rows, cols) a_x, a_y, a_z per element
    const float* unit,            // (3, 8, 8) unit stiffness U_x, U_y, U_z
    const float* robin,           // (nodes,) lumped convective conductance
    const unsigned char* free_nodes,
    const int slabs,
    const int element_rows,
    const int element_cols)
{
    const int node_rows = element_rows + 1;
    const int node_cols = element_cols + 1;
    const int node_layers = slabs + 1;
    const int node_count = node_layers * node_rows * node_cols;
    const int node = blockDim.x * blockIdx.x + threadIdx.x;
    if (node >= node_count) {
        return;
    }
    if (free_nodes[node] == 0) {
        output[node] = vector[node];
        return;
    }

    const int plane = node_rows * node_cols;
    const int node_z = node / plane;
    const int remainder = node - node_z * plane;
    const int node_y = remainder / node_cols;
    const int node_x = remainder - node_y * node_cols;

    const int z_first = node_z > 0 ? node_z - 1 : 0;
    const int z_last = node_z < slabs ? node_z : slabs - 1;
    const int y_first = node_y > 0 ? node_y - 1 : 0;
    const int y_last = node_y < element_rows ? node_y : element_rows - 1;
    const int x_first = node_x > 0 ? node_x - 1 : 0;
    const int x_last = node_x < element_cols ? node_x : element_cols - 1;

    const int element_count = slabs * element_rows * element_cols;
    const float* unit_x = unit;
    const float* unit_y = unit + 64;
    const float* unit_z = unit + 128;
    float accumulated = 0.0f;
    for (int ez = z_first; ez <= z_last; ++ez) {
        const int local_z = node_z - ez;
        for (int ey = y_first; ey <= y_last; ++ey) {
            const int local_y = node_y - ey;
            for (int ex = x_first; ex <= x_last; ++ex) {
                const int local_x = node_x - ex;
                const int local_row = 4 * local_z + 2 * local_y + local_x;
                const int element = (ez * element_rows + ey) * element_cols + ex;
                const float a_x = coefficients[element];
                const float a_y = coefficients[element_count + element];
                const float a_z = coefficients[2 * element_count + element];
                const int corner = (ez * node_rows + ey) * node_cols + ex;
                for (int column = 0; column < 8; ++column) {
                    const int column_node = corner
                        + (column >> 2) * plane
                        + ((column >> 1) & 1) * node_cols
                        + (column & 1);
                    if (free_nodes[column_node] == 0) {
                        continue;
                    }
                    const int local_index = 8 * local_row + column;
                    const float weight = a_x * unit_x[local_index]
                        + a_y * unit_y[local_index]
                        + a_z * unit_z[local_index];
                    accumulated += weight * vector[column_node];
                }
            }
        }
    }
    output[node] = accumulated + robin[node] * vector[node];
}
"""


class CudaThermalHexQ1Apply:
    """One-launch float32 trilinear Q1 conduction action for a layered mesh."""

    kernel_name = "cuda-fused-node-gather-hex-q1"

    def __init__(
        self,
        runtime: Any,
        element_grid_shape: tuple[int, int, int],
    ) -> None:
        if not getattr(runtime, "is_cuda", False):
            raise TypeError("CudaThermalHexQ1Apply requires a CUDA runtime")
        self._cp = runtime.namespace
        self._slabs, self._element_rows, self._element_cols = (
            int(axis) for axis in element_grid_shape
        )
        self._node_count = (
            (self._slabs + 1) * (self._element_rows + 1) * (self._element_cols + 1)
        )
        self._threads = 256
        self._kernel = self._cp.RawKernel(
            _THERMAL_Q1_SOURCE,
            "thermal_hex_q1_apply",
            options=("--std=c++11",),
        )

    def __call__(
        self,
        vector: Any,
        coefficients: Any,
        unit: Any,
        robin: Any,
        free_nodes: Any,
    ) -> Any:
        cp = self._cp
        if vector.dtype != cp.float32 or int(vector.size) != self._node_count:
            raise ValueError("CUDA thermal input must be a node-sized float32 vector")
        flat = cp.ascontiguousarray(vector.reshape(-1))
        output = cp.empty_like(flat)
        blocks = (self._node_count + self._threads - 1) // self._threads
        self._kernel(
            (blocks,),
            (self._threads,),
            (
                flat,
                output,
                coefficients,
                unit,
                robin,
                free_nodes,
                np.int32(self._slabs),
                np.int32(self._element_rows),
                np.int32(self._element_cols),
            ),
        )
        return output


def thermal_cuda_source() -> str:
    """Return CUDA C++ source for offline syntax checks and audit tooling."""

    return _THERMAL_Q1_SOURCE
