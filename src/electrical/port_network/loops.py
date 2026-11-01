"""Parallel loops through shared copper, closed by lumped parts.

A commutation loop with several capacitors is a set of meshes: mesh ``k``
runs from the switch through capacitor ``k`` and back, and the copper's
mesh impedance matrix ``Zm[k][j]`` is the drop around mesh ``k`` for 1 A in
mesh ``j``.  With each capacitor's impedance added on the diagonal, the
mesh currents for 1 V across the switch are ``(Zm + Zc)^-1 1``; their sum is
the switch current and its inverse the loop impedance.

The copper is taken as ``R + jωL`` from one solve, which holds over the band
where one filament through the foil describes it.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .lumped import SeriesRLC


def loop_impedance(
    resistance_ohm: np.ndarray,
    inductance_h: np.ndarray,
    parts: Sequence[SeriesRLC | None],
    frequency_hz: float,
) -> tuple[complex, np.ndarray]:
    """Loop impedance and mesh currents (for 1 V) at ``frequency_hz``.

    ``resistance_ohm`` and ``inductance_h`` are the copper's mesh matrices; a
    part of ``None`` closes its mesh with the copper alone.
    """
    omega = 2.0 * math.pi * frequency_hz
    zc = np.diag([part.impedance(omega) if part is not None else 0j for part in parts])
    ones = np.ones(len(parts))
    currents = np.linalg.solve(resistance_ohm + 1j * omega * inductance_h + zc, ones)
    return 1.0 / currents.sum(), currents


def mesh_matrices(z_ohm: np.ndarray, frequency_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """``R`` and ``L`` of a mesh impedance matrix solved at ``frequency_hz``,
    symmetrised (the copper is reciprocal; a pad-mean readout is not exactly so)."""
    z = 0.5 * (np.asarray(z_ohm) + np.asarray(z_ohm).T)
    return z.real, z.imag / (2.0 * math.pi * frequency_hz)


__all__ = ["loop_impedance", "mesh_matrices"]
