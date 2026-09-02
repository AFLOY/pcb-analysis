"""Electrical analysis algorithms.

The second package level names the numerical method and its acceleration
strategy.  Thermal and other physics can therefore be added beside this
package without mixing their discretisations or solver policies.
"""

from . import dice_peec, matrix_free_mpir_fem

__all__ = ["dice_peec", "matrix_free_mpir_fem"]
