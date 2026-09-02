"""Electromagnetic-compatibility analysis algorithms.

The second package level names the numerical method and its acceleration
strategy, mirroring ``electrical`` and ``thermal``.  The EMC front ends read
the current distributions those solvers produce; they own no field solve of
their own.
"""

from . import tiled_dipole_superposition

__all__ = ["tiled_dipole_superposition"]
