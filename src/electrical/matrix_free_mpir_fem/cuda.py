"""Fused CUDA kernels for the matrix-free FEM low-precision path.

The kernels use node-owned gather accumulation.  A thread owns one output
node, visits at most four adjacent Q1 elements, and writes exactly one value.
That removes global atomics and makes the numerical order deterministic while
keeping the assembled Maxwell matrix out of device memory.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .runtime import LowPrecisionRuntime


_SCALAR_MAXWELL_Q1_SOURCE = r"""
#include <cuComplex.h>

extern "C" __global__
void scalar_maxwell_q1_apply(
    const cuFloatComplex* vector,
    cuFloatComplex* output,
    const cuFloatComplex* inverse_mu,
    const cuFloatComplex* reaction,
    const cuFloatComplex* stiffness,
    const cuFloatComplex* mass,
    const unsigned char* free_nodes,
    const int element_rows,
    const int element_columns)
{
    const int node_columns = element_columns + 1;
    const int node_rows = element_rows + 1;
    const int node = blockDim.x * blockIdx.x + threadIdx.x;
    const int node_count = node_rows * node_columns;
    if (node >= node_count) {
        return;
    }
    if (free_nodes[node] == 0) {
        output[node] = vector[node];
        return;
    }

    const int node_y = node / node_columns;
    const int node_x = node - node_y * node_columns;
    const int element_y_first = node_y > 0 ? node_y - 1 : 0;
    const int element_y_last =
        node_y < element_rows ? node_y : element_rows - 1;
    const int element_x_first = node_x > 0 ? node_x - 1 : 0;
    const int element_x_last =
        node_x < element_columns ? node_x : element_columns - 1;
    cuFloatComplex accumulated = make_cuFloatComplex(0.0f, 0.0f);

    for (int element_y = element_y_first;
         element_y <= element_y_last;
         ++element_y) {
        for (int element_x = element_x_first;
             element_x <= element_x_last;
             ++element_x) {
            const int element = element_y * element_columns + element_x;
            const int local_y = node_y - element_y;
            const int local_x = node_x - element_x;
            const int local_row = 2 * local_y + local_x;
            const int top_left = element_y * node_columns + element_x;
            const int element_nodes[4] = {
                top_left,
                top_left + 1,
                top_left + node_columns,
                top_left + node_columns + 1
            };

            for (int local_column = 0; local_column < 4; ++local_column) {
                const int column_node = element_nodes[local_column];
                if (free_nodes[column_node] == 0) {
                    continue;
                }
                const int local_index = 4 * local_row + local_column;
                const cuFloatComplex coefficient = cuCaddf(
                    cuCmulf(inverse_mu[element], stiffness[local_index]),
                    cuCmulf(reaction[element], mass[local_index]));
                accumulated = cuCaddf(
                    accumulated,
                    cuCmulf(coefficient, vector[column_node]));
            }
        }
    }
    output[node] = accumulated;
}
"""


class CudaScalarMaxwellQ1Apply:
    """One-launch complex64 Q1 Maxwell action for a structured mesh."""

    kernel_name = "cuda-fused-node-gather-q1"

    def __init__(
        self,
        runtime: LowPrecisionRuntime,
        element_shape: tuple[int, int],
    ) -> None:
        if not getattr(runtime, "is_cuda", False):
            raise TypeError("CudaScalarMaxwellQ1Apply requires a CUDA runtime")
        self._runtime = runtime
        self._cp = runtime.namespace
        self._element_rows = int(element_shape[0])
        self._element_columns = int(element_shape[1])
        self._node_count = (self._element_rows + 1) * (
            self._element_columns + 1
        )
        self._threads = 256
        self._kernel = self._cp.RawKernel(
            _SCALAR_MAXWELL_Q1_SOURCE,
            "scalar_maxwell_q1_apply",
            options=("--std=c++11",),
        )

    def __call__(
        self,
        vector: Any,
        inverse_mu: Any,
        reaction: Any,
        stiffness: Any,
        mass: Any,
        free_nodes: Any,
    ) -> Any:
        cp = self._cp
        if vector.dtype != cp.complex64 or int(vector.size) != self._node_count:
            raise ValueError(
                "CUDA Maxwell input must be a node-sized complex64 vector"
            )
        output = cp.empty_like(vector)
        blocks = (self._node_count + self._threads - 1) // self._threads
        self._kernel(
            (blocks,),
            (self._threads,),
            (
                vector,
                output,
                inverse_mu,
                reaction,
                stiffness,
                mass,
                free_nodes,
                np.int32(self._element_rows),
                np.int32(self._element_columns),
            ),
        )
        return output


def scalar_maxwell_cuda_source() -> str:
    """Return CUDA C++ source for offline syntax checks and audit tooling."""

    return _SCALAR_MAXWELL_Q1_SOURCE
