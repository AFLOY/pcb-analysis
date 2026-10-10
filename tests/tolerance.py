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

# Hoer-Love closed form: the 64 corner terms cancel more the further apart the
# bars are, so one ulp of a logarithm becomes eps / retained in the answer.  At
# the near radius (24 cells) about seven digits are left (docs/SHEET_PEEC.md);
# two correct implementations, or two libm's, agree only to that.
CLOSED_FORM_RTOL = 1.0e-7


def closed_form_rtol(retained: float) -> float:
    """Comparison tolerance for one closed-form value whose sum kept ``retained`` of its terms."""

    return max(DIRECT_RTOL, 100.0 * 2.2e-16 / max(retained, 1e-300))

# Measured: the FP32-inner MPIR on the conduction fixtures leaves solution
# errors up to ~60x the relative residual it stops at; 100 covers that.
ITERATIVE_FACTOR = 100.0


def iterative_rtol(solver_relative_tolerance: float, factor: float = ITERATIVE_FACTOR) -> float:
    """Comparison tolerance for an answer of an iterative solve stopped at ``solver_relative_tolerance``."""

    return factor * solver_relative_tolerance
