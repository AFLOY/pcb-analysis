"""Electrical analysis algorithms.

The second package level names the numerical method and its acceleration
strategy.  Thermal and other physics can therefore be added beside this
package without mixing their discretisations or solver policies.

The thread count is the one process-wide setting of :mod:`.threads`
(:func:`set_thread_budget`); no solver takes a thread argument.
"""

from . import dice_peec, matrix_free_mpir_fem, threads
from .threads import available_threads, set_thread_budget, thread_budget, thread_budget_scope

__all__ = [
    "available_threads",
    "dice_peec",
    "matrix_free_mpir_fem",
    "set_thread_budget",
    "thread_budget",
    "thread_budget_scope",
    "threads",
]
