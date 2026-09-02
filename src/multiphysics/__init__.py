"""Coupled analysis across the electrical, thermal, and EMC packages.

The second package level names the coupling method and its acceleration
strategy, mirroring the single-physics packages.  Nothing here solves a
field problem; the modules orchestrate the solvers the other packages own
and carry the fields between them.
"""

from . import staggered_coupling

__all__ = ["staggered_coupling"]
