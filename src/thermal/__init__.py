"""Thermal analysis algorithms.

The second package level names the numerical method and its acceleration
strategy, mirroring the ``electrical`` package.  The two physics share the
mixed-precision iterative-refinement solver and the low-precision runtimes,
but keep their discretisations and front ends apart.
"""

from . import matrix_free_mpir_fem

__all__ = ["matrix_free_mpir_fem"]
