"""Direct DC solve of a conductance network: the zero-frequency sheet mesh.

At zero frequency a sheet mesh is a resistor network: one node per cell (and
per layer), one conductance per branch between neighbouring cells, and the
vertical connections between layers as further branches.  A router that
screens thousands of candidate shapes wants that network solved exactly and
cheaply, and often wants the sensitivity of a voltage to every branch, which
the symmetric Laplacian gives from one factorisation and one more
back substitution per objective.

The caller numbers the nodes and lists the branches; this module knows nothing
about cells, layers or pads.  Node ``reference`` is held at 0 V and removed
from the system.  The order of every floating-point operation here is part of
the contract: the reduced Laplacian is assembled from the branches in the
order given, duplicates accumulate in sequence (``np.add.at``), and the loss
is a running sum, so the same network and injection give the same bits.

``solve_conductance_network`` runs in the C++ core (``electrical._pcbcore``)
when it is built: one SuperLU factorisation solves the forward and every
adjoint right-hand side.  Without the build, the NumPy reference in
:mod:`._dc_network_reference` answers (``splu`` with objectives, ``spsolve``
without).  The two agree to roundoff, not bit for bit; each gives the same
bits every time.  ``split_branch_sensitivity`` turns an adjoint branch product
into a per-node sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .. import _backend
from . import _dc_network_reference as _reference


@dataclass(frozen=True)
class ConductanceNetwork:
    """``node_count`` nodes and the branches ``left[k] -- right[k]``.

    ``conductance`` is in siemens.  Branches are kept in the order given;
    that order fixes the floating-point result.
    """

    node_count: int
    left: np.ndarray
    right: np.ndarray
    conductance: np.ndarray

    def __post_init__(self) -> None:
        left = np.ascontiguousarray(self.left, dtype=np.int64)
        right = np.ascontiguousarray(self.right, dtype=np.int64)
        conductance = np.ascontiguousarray(self.conductance, dtype=np.float64)
        if not (left.shape == right.shape == conductance.shape) or left.ndim != 1:
            raise ValueError("left, right and conductance must be 1-D and equally long")
        if left.size and (
            min(left.min(), right.min()) < 0
            or max(left.max(), right.max()) >= self.node_count
        ):
            raise ValueError("branch endpoint outside 0..node_count-1")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "conductance", conductance)

    def unknown_index(self, reference: int) -> np.ndarray:
        """Row of each node in the reduced system; ``-1`` for the reference.

        Unknowns keep the node order with the reference left out.
        """
        if not 0 <= reference < self.node_count:
            raise ValueError("reference node outside the network")
        nodes = np.arange(self.node_count, dtype=np.int64)
        index = nodes - (nodes > reference)
        index[reference] = -1
        return index

    def reduced_laplacian(self, reference: int) -> sp.csr_matrix:
        """The Laplacian with the reference row and column removed (CSR)."""
        unknown = self.unknown_index(reference)
        count = self.node_count - 1
        left_unknown = unknown[self.left]
        right_unknown = unknown[self.right]
        conductances = self.conductance
        endpoints = np.empty(2 * self.left.size, dtype=np.int64)
        endpoints[0::2] = left_unknown
        endpoints[1::2] = right_unknown
        endpoint_conductances = np.repeat(conductances, 2)
        known = endpoints >= 0
        diagonal = np.zeros(count, dtype=np.float64)
        np.add.at(diagonal, endpoints[known], endpoint_conductances[known])
        interior = (left_unknown >= 0) & (right_unknown >= 0)
        interior_left = left_unknown[interior]
        interior_right = right_unknown[interior]
        interior_conductances = conductances[interior]
        off_diagonal = 2 * len(interior_left)
        rows = np.empty(off_diagonal + count, dtype=np.int64)
        columns = np.empty(off_diagonal + count, dtype=np.int64)
        values = np.empty(off_diagonal + count, dtype=np.float64)
        rows[0:off_diagonal:2] = interior_left
        rows[1:off_diagonal:2] = interior_right
        columns[0:off_diagonal:2] = interior_right
        columns[1:off_diagonal:2] = interior_left
        values[0:off_diagonal:2] = -interior_conductances
        values[1:off_diagonal:2] = -interior_conductances
        rows[off_diagonal:] = np.arange(count)
        columns[off_diagonal:] = np.arange(count)
        values[off_diagonal:] = diagonal
        return sp.coo_matrix(
            (values, (rows, columns)), shape=(count, count)
        ).tocsr()


@dataclass(frozen=True)
class DCNetworkSolution:
    """Potentials, branch currents and the adjoint states of one solve.

    ``node_voltage`` and ``adjoint_voltage[j]`` are per node with the
    reference at 0 V; ``voltage_unknowns`` is the raw reduced solution.
    ``node_current`` is the net current the branches carry out of each node.
    """

    reference: int
    voltage_unknowns: np.ndarray
    node_voltage: np.ndarray
    edge_current: np.ndarray
    node_current: np.ndarray
    loss_w: float
    relative_residual: float
    adjoint_voltage: tuple[np.ndarray, ...]
    factorized: bool

    def adjoint_branch_product(self, network: ConductanceNetwork, objective: int) -> np.ndarray:
        """``(V_l - V_r)(λ_l - λ_r)`` per branch for one objective."""
        voltage = self.node_voltage
        adjoint = self.adjoint_voltage[objective]
        return (voltage[network.left] - voltage[network.right]) * (
            adjoint[network.left] - adjoint[network.right]
        )


def solve_conductance_network(
    network: ConductanceNetwork,
    reference: int,
    injection: np.ndarray,
    objective_weights: np.ndarray | None = None,
) -> DCNetworkSolution:
    """Solve ``L v = injection`` with ``v[reference] = 0``.

    ``injection`` is the current entering each node (A).  ``objective_weights``
    (``objectives × node_count``) are the linear functionals ``cᵀv`` whose
    adjoint states ``L λ = c`` are wanted.  The C++ core factors the reduced
    Laplacian once (SuperLU, COLAMD ordering) and solves the injection and
    every adjoint right-hand side together; the NumPy reference does the same
    with ``splu`` when there are objectives and ``spsolve`` otherwise.  A
    singular network gives non-finite potentials, not an exception: the
    caller names the case it was solving.
    """
    injection = np.ascontiguousarray(injection, dtype=np.float64)
    if injection.shape != (network.node_count,):
        raise ValueError("injection must have one entry per node")
    weights = (
        np.zeros((0, network.node_count), dtype=np.float64)
        if objective_weights is None
        else np.ascontiguousarray(objective_weights, dtype=np.float64)
    )
    if weights.ndim != 2 or weights.shape[1] != network.node_count:
        raise ValueError("objective_weights must be objectives x node_count")
    network.unknown_index(reference)  # validates the reference
    native = _backend.core()
    if native is not None:
        fields = native.network.solve_conductance_network(
            network.node_count, network.left, network.right, network.conductance,
            int(reference), injection, weights,
        )
    else:
        fields = _reference.solve_conductance_network(network, reference, injection, weights)
    adjoint = np.asarray(fields["adjoint_voltage"], dtype=np.float64)
    return DCNetworkSolution(
        reference=reference,
        voltage_unknowns=np.asarray(fields["voltage_unknowns"], dtype=np.float64),
        node_voltage=np.asarray(fields["node_voltage"], dtype=np.float64),
        edge_current=np.asarray(fields["edge_current"], dtype=np.float64),
        node_current=np.asarray(fields["node_current"], dtype=np.float64),
        loss_w=float(fields["loss_w"]),
        relative_residual=float(fields["relative_residual"]),
        adjoint_voltage=tuple(adjoint[j] for j in range(adjoint.shape[0])),
        factorized=bool(weights.shape[0] and network.node_count - 1),
    )


def split_branch_sensitivity(
    network: ConductanceNetwork,
    branch_product: np.ndarray,
    in_plane: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-node sensitivity of an objective to the branch conductances.

    ``G·(V_l−V_r)(λ_l−λ_r)`` is the objective's sensitivity to scaling one
    branch.  An in-plane branch is two half-cells in series, so each endpoint
    owns half of it; a vertical branch belongs to neither node and is returned
    separately.  Returns ``(node_sensitivity, branch_sensitivity, vertical
    total)``; with uniform scaling the node sum plus the vertical total adds
    back to the objective.
    """
    branch_product = np.ascontiguousarray(branch_product, dtype=np.float64)
    in_plane = np.ascontiguousarray(in_plane, dtype=bool)
    native = _backend.core()
    if native is not None:
        node, branch, vertical = native.network.split_branch_sensitivity(
            network.node_count, network.left, network.right, network.conductance,
            branch_product, in_plane,
        )
        return np.asarray(node), np.asarray(branch), float(vertical)
    return _reference.split_branch_sensitivity(network, branch_product, in_plane)
