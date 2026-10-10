"""NumPy reference for :mod:`.dc_network`: the implementation the C++ core replaced.

It answers when ``electrical._pcbcore`` is not built, and the tests compare
the core with it.  Both return the same fields; :mod:`.dc_network` wraps them
in :class:`~.dc_network.DCNetworkSolution`.  Kept numerically as it was.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse.linalg as spla

if TYPE_CHECKING:  # pragma: no cover
    from .dc_network import ConductanceNetwork


def solve_conductance_network(
    network: ConductanceNetwork,
    reference: int,
    injection: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    """SuperLU (``splu``) when there are objectives, ``spsolve`` otherwise."""
    unknown = network.unknown_index(reference)
    kept = unknown >= 0
    matrix = network.reduced_laplacian(reference)
    rhs = np.zeros(network.node_count - 1, dtype=np.float64)
    rhs[unknown[kept]] = injection[kept]
    count = network.node_count - 1
    factorized = bool(weights.shape[0] and count)
    if factorized:
        columns = np.empty((count, 1 + weights.shape[0]), dtype=np.float64)
        columns[:, 0] = rhs
        for offset, row in enumerate(weights):
            column = np.zeros(count, dtype=np.float64)
            column[unknown[kept]] = row[kept]
            columns[:, offset + 1] = column
        solutions = np.asarray(
            spla.splu(matrix.tocsc()).solve(columns), dtype=np.float64
        )
        voltage_unknowns = solutions[:, 0]
        adjoint_unknowns = [
            solutions[:, offset + 1] for offset in range(weights.shape[0])
        ]
    else:
        voltage_unknowns = np.asarray(spla.spsolve(matrix, rhs), dtype=np.float64)
        adjoint_unknowns = [
            np.zeros(count, dtype=np.float64) for _ in range(weights.shape[0])
        ]
    residual = matrix @ voltage_unknowns - rhs
    rhs_norm = float(np.linalg.norm(rhs))
    relative_residual = float(np.linalg.norm(residual)) / max(rhs_norm, 1e-30)

    def full(values: np.ndarray) -> np.ndarray:
        out = np.zeros(network.node_count, dtype=np.float64)
        out[kept] = values[unknown[kept]]
        return out

    node_voltage = full(voltage_unknowns)
    edge_current = network.conductance * (
        node_voltage[network.left] - node_voltage[network.right]
    )
    node_current = np.zeros(network.node_count, dtype=np.float64)
    signed_nodes = np.empty(2 * network.left.size, dtype=np.int64)
    signed_nodes[0::2] = network.left
    signed_nodes[1::2] = network.right
    signed_current = np.empty(2 * network.left.size, dtype=np.float64)
    signed_current[0::2] = edge_current
    signed_current[1::2] = -edge_current
    np.add.at(node_current, signed_nodes, signed_current)
    # A running sum, not a pairwise one, so the loss is reproducible branch
    # order for branch order.
    loss_w = (
        float(
            np.cumsum(edge_current * edge_current / network.conductance)[-1]
        )
        if network.left.size
        else 0.0
    )
    return {
        "voltage_unknowns": voltage_unknowns,
        "node_voltage": node_voltage,
        "edge_current": edge_current,
        "node_current": node_current,
        "adjoint_voltage": np.array(
            [full(values) for values in adjoint_unknowns], dtype=np.float64
        ).reshape(weights.shape[0], network.node_count),
        "loss_w": loss_w,
        "relative_residual": relative_residual,
        "singular": False,
    }


def split_branch_sensitivity(
    network: ConductanceNetwork,
    branch_product: np.ndarray,
    in_plane: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """``(node_sensitivity, branch_sensitivity, vertical_total)``; see the facade."""
    in_plane = np.asarray(in_plane, dtype=bool)
    halves = np.where(
        in_plane, network.conductance * branch_product * 0.5, 0.0
    )
    node_sensitivity = np.zeros(network.node_count, dtype=np.float64)
    np.add.at(node_sensitivity, network.left, halves)
    np.add.at(node_sensitivity, network.right, halves)
    branch_sensitivity = network.conductance * branch_product
    vertical_total = float(branch_sensitivity[~in_plane].sum())
    return node_sensitivity, branch_sensitivity, vertical_total
