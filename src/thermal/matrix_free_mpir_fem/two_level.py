"""Two-level Jacobi plus aggregation coarse correction for layered node grids.

A cooled copper plate is thermally stiff in-plane and weakly coupled to its
surroundings, so the slowest mode of the conduction operator is an almost
uniform temperature of each plate.  Jacobi scaling cannot see that mode and
PCG needs hundreds of iterations per decade on a realistic stack.  This
preconditioner adds a coarse correction in the space of functions that are
constant over ``block`` by ``block`` node patches of one node layer:

```text
M^-1 r = D^-1 r + Z (Z^T A Z)^-1 Z^T r
```

``Z`` is never stored.  Its action is a reshape-and-sum over patches, and
``Z^T A Z`` is assembled exactly with 27 operator applications by colouring
the patches so that same-coloured patches never share an element.  The small
coarse matrix is inverted once in FP64; its inverse is applied on the
low-precision runtime as one small matrix-vector product per iteration.

The construction depends only on the node grid, the free-node mask, and the
high-precision action, so it is physics-neutral within this repository.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


DEFAULT_MAX_COARSE_SIZE = 2048


def choose_block_size(
    node_shape: tuple[int, int, int],
    *,
    max_coarse_size: int = DEFAULT_MAX_COARSE_SIZE,
    minimum: int = 4,
) -> int:
    """Smallest in-plane patch width whose coarse space fits the size cap."""

    layers, rows, cols = node_shape
    block = max(1, int(minimum))
    while True:
        coarse = layers * -(-rows // block) * -(-cols // block)
        if coarse <= max_coarse_size or block >= max(rows, cols):
            return block
        block += 1


class AggregationCoarseCorrection:
    """Jacobi plus patch-constant coarse correction on a layered node grid."""

    def __init__(
        self,
        *,
        node_shape: tuple[int, int, int],
        free_nodes: np.ndarray,
        diagonal_high: np.ndarray,
        apply_high: Callable[[np.ndarray], np.ndarray],
        runtime: Any,
        block: int | None = None,
        max_coarse_size: int = DEFAULT_MAX_COARSE_SIZE,
    ) -> None:
        layers, rows, cols = (int(axis) for axis in node_shape)
        self.node_shape = (layers, rows, cols)
        if block is None:
            block = choose_block_size(
                self.node_shape, max_coarse_size=max_coarse_size
            )
        if block < 1:
            raise ValueError("block must be a positive node count")
        self.block = int(block)
        self.runtime = runtime
        blocks_rows = -(-rows // self.block)
        blocks_cols = -(-cols // self.block)
        self.coarse_shape = (layers, blocks_rows, blocks_cols)
        self.coarse_size = int(np.prod(self.coarse_shape))
        self._padded_shape = (
            layers,
            blocks_rows * self.block,
            blocks_cols * self.block,
        )

        free = np.asarray(free_nodes, dtype=bool).reshape(self.node_shape)
        self._free_high = free
        diagonal = np.asarray(diagonal_high, dtype=np.float64).reshape(-1)
        if diagonal.size != free.size:
            raise ValueError("diagonal_high must hold one value per node")

        coarse_matrix = self._assemble_coarse(apply_high, free)
        counts = self._restrict(free.astype(np.float64), np)
        empty = counts.reshape(-1) <= 0.0
        # Patches made only of fixed nodes have no unknowns; give them a unit
        # diagonal so the matrix stays SPD.  Their restricted residual is zero.
        coarse_matrix[empty, :] = 0.0
        coarse_matrix[:, empty] = 0.0
        coarse_matrix[empty, empty] = 1.0
        try:
            factor = np.linalg.cholesky(coarse_matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "the coarse operator is not positive definite; the fine operator "
                "must be SPD for the two-level preconditioner"
            ) from exc
        identity = np.eye(self.coarse_size)
        inverse = np.linalg.solve(factor.T, np.linalg.solve(factor, identity))
        self.coarse_matrix = coarse_matrix
        self._coarse_inverse_high = 0.5 * (inverse + inverse.T)

        xp = runtime.namespace
        self._diagonal_high = diagonal
        self._diagonal_low = runtime.from_host(diagonal)
        self._free_low = xp.asarray(free, dtype=bool)
        self._coarse_inverse_low = runtime.from_host(self._coarse_inverse_high)

    # ------------------------------------------------------------ transfers
    def _restrict(self, grid: Any, xp: Any) -> Any:
        """``Z^T`` action: sum a node grid over each patch of one node layer."""

        layers, rows, cols = self.node_shape
        padded = xp.zeros(self._padded_shape, dtype=grid.dtype)
        padded[:, :rows, :cols] = grid
        return padded.reshape(
            layers,
            self.coarse_shape[1],
            self.block,
            self.coarse_shape[2],
            self.block,
        ).sum(axis=(2, 4))

    def _prolong(self, coarse: Any, xp: Any) -> Any:
        """``Z`` action: broadcast one value per patch back to its nodes."""

        layers, rows, cols = self.node_shape
        expanded = xp.repeat(
            xp.repeat(coarse.reshape(self.coarse_shape), self.block, axis=1),
            self.block,
            axis=2,
        )
        return expanded[:, :rows, :cols]

    # --------------------------------------------------------- construction
    @staticmethod
    def _colour(indices: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        layer, row, col = indices
        return (layer % 3) * 9 + (row % 3) * 3 + (col % 3)

    def _assemble_coarse(
        self,
        apply_high: Callable[[np.ndarray], np.ndarray],
        free: np.ndarray,
    ) -> np.ndarray:
        """Exact ``Z^T A Z`` from 27 coloured operator applications.

        Same-coloured patches are at least three patches apart along some
        axis, so no Q1 element touches two of them.  ``Z^T A (Z 1_colour)``
        therefore lands in row ``I`` only from the unique same-coloured patch
        adjacent to (or equal to) ``I``.
        """

        coarse_indices = np.indices(self.coarse_shape)
        colours = self._colour(coarse_indices)
        matrix = np.zeros((self.coarse_size, self.coarse_size))
        flat_rows = np.arange(self.coarse_size)
        for colour in range(27):
            selected = (colours == colour).astype(np.float64)
            if not selected.any():
                continue
            fine = np.where(free, self._prolong(selected, np), 0.0)
            action = apply_high(fine.reshape(-1)).reshape(self.node_shape)
            action = np.where(free, action, 0.0)
            restricted = self._restrict(action, np).reshape(-1)
            for offset in np.ndindex(3, 3, 3):
                shifted = tuple(
                    axis + delta - 1 for axis, delta in zip(coarse_indices, offset)
                )
                inside = np.ones(self.coarse_shape, dtype=bool)
                for axis, extent in zip(shifted, self.coarse_shape):
                    inside &= (axis >= 0) & (axis < extent)
                if not inside.any():
                    continue
                neighbour_colour = self._colour(
                    tuple(np.where(inside, axis, 0) for axis in shifted)
                )
                mask = (inside & (neighbour_colour == colour)).reshape(-1)
                if not mask.any():
                    continue
                columns = np.ravel_multi_index(
                    tuple(axis.reshape(-1)[mask] for axis in shifted),
                    self.coarse_shape,
                )
                matrix[flat_rows[mask], columns] = restricted[mask]
        return matrix

    # ----------------------------------------------------------------- apply
    def __call__(self, vector: Any) -> Any:
        xp = self.runtime.namespace
        flat = vector.reshape(-1)
        smoothed = flat / self._diagonal_low
        grid = xp.where(self._free_low, flat.reshape(self.node_shape), 0)
        restricted = self._restrict(grid, xp).reshape(-1)
        coarse = xp.matmul(self._coarse_inverse_low, restricted)
        correction = xp.where(self._free_low, self._prolong(coarse, xp), 0)
        return xp.asarray(smoothed + correction.reshape(-1), dtype=vector.dtype)

    def apply_high(self, vector: np.ndarray) -> np.ndarray:
        """FP64 host action of the same preconditioner, for verification."""

        flat = np.asarray(vector, dtype=np.float64).reshape(-1)
        smoothed = flat / self._diagonal_high
        grid = np.where(self._free_high, flat.reshape(self.node_shape), 0.0)
        restricted = self._restrict(grid, np).reshape(-1)
        coarse = self._coarse_inverse_high @ restricted
        correction = np.where(self._free_high, self._prolong(coarse, np), 0.0)
        return smoothed + correction.reshape(-1)
