"""Frequency-domain nodal analysis of a network of series R-L-C branches.

The network is given as nodes ``0 .. node_count-1`` and branches between
them; one node is the reference.  At each frequency the admittance matrix
is assembled from the branch impedances, the injected currents are applied
and the node voltages solved directly.  A branch whose impedance is zero
at that frequency is taken as ``1e-9`` ohm so ties and shorts stay in the
matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .lumped import SeriesRLC

SHORT_OHM = 1e-9


@dataclass(frozen=True)
class Branch:
    """A series element between nodes ``a`` and ``b``."""

    a: int
    b: int
    element: SeriesRLC


def nodal_voltages(
    node_count: int,
    branches: Sequence[Branch],
    injection_a: Sequence[complex],
    reference: int,
    frequency_hz: float,
) -> np.ndarray:
    """Node voltages (reference at 0 V) for currents ``injection_a`` into the nodes.

    Raises ``numpy.linalg.LinAlgError`` when the network is singular (a node
    without a path to the reference).
    """
    omega = 2.0 * math.pi * frequency_hz
    keep = [n for n in range(node_count) if n != reference]
    index = {n: k for k, n in enumerate(keep)}
    y = np.zeros((len(keep), len(keep)), dtype=complex)
    for branch in branches:
        z = branch.element.impedance(omega)
        if z == 0:
            z = SHORT_OHM
        adm = 1.0 / z
        a, b = branch.a, branch.b
        for m, n, sign in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
            if m in index and n in index:
                y[index[m], index[n]] += sign * adm
    injection = np.asarray(injection_a, dtype=complex)
    rhs = np.array([injection[n] for n in keep], dtype=complex)
    v = np.linalg.solve(y, rhs)
    voltages = np.zeros(node_count, dtype=complex)
    voltages[keep] = v
    return voltages


__all__ = ["Branch", "SHORT_OHM", "nodal_voltages"]
