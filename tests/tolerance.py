"""Tolerance classes for comparing a solver's answer with its reference.

A direct or closed-form result (geometry, kernels, a sparse LU solve) is
held to ``DIRECT_RTOL`` whichever implementation computed it.  An iterative
solve (GMRES, PCG, mixed-precision refinement) stops when its residual meets
the solver tolerance, so two correct implementations agree only to that
tolerance times the problem's conditioning: ``iterative_rtol`` scales the
solver tolerance by a factor measured on the test problems.
"""

from __future__ import annotations

DIRECT_RTOL = 1.0e-9

# Measured: the FP32-inner MPIR on the conduction fixtures leaves solution
# errors up to ~60x the relative residual it stops at; 100 covers that.
ITERATIVE_FACTOR = 100.0


def iterative_rtol(solver_relative_tolerance: float, factor: float = ITERATIVE_FACTOR) -> float:
    """Comparison tolerance for an answer of an iterative solve stopped at ``solver_relative_tolerance``."""

    return factor * solver_relative_tolerance
