"""The two-level preconditioner lives with the solver in ``electrical``; re-exported here."""

from electrical.matrix_free_mpir_fem.two_level import (
    DEFAULT_MAX_COARSE_SIZE,
    AggregationCoarseCorrection,
    choose_block_size,
)

__all__ = ["AggregationCoarseCorrection", "DEFAULT_MAX_COARSE_SIZE", "choose_block_size"]
