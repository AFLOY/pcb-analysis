"""Grid refinement from the parts on the board: fine cells under components, coarse elsewhere.

The tensor grid of ``electrical.matrix_free_mpir_fem.grid`` refines a whole
row and column strip per refinement box.  The boxes come from the parts that
need resolution: the footprint of every component solid of a KiCad STEP
export (``kicad_component_solids``), widened by a margin, or any explicit
rectangles (a narrow trace, a hot via field).  Everything is in the STEP
frame in metres; ``rasterize_board`` takes the resulting grid directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from electrical.matrix_free_mpir_fem.grid import TensorGrid, refined_grid

from .reader import StepSolid

MM = 1.0e-3


@dataclass(frozen=True)
class RefinementBox:
    """One rectangle to refine, ``(x0, x1, y0, y1)`` in metres, with its source name."""

    name: str
    x0_m: float
    x1_m: float
    y0_m: float
    y1_m: float

    def __post_init__(self) -> None:
        if not (self.x1_m > self.x0_m and self.y1_m > self.y0_m):
            raise ValueError(f"refinement box {self.name!r} must have positive extent")

    @property
    def bounds_m(self) -> tuple[float, float, float, float]:
        return (self.x0_m, self.x1_m, self.y0_m, self.y1_m)


def component_boxes(
    components: Mapping[str, Sequence[StepSolid]],
    *,
    references: Iterable[str] | None = None,
    min_size_m: float = 0.0,
) -> tuple[RefinementBox, ...]:
    """The in-plane bounding box of each component's solids.

    ``references`` restricts the parts (default: all); ``min_size_m`` widens a
    box smaller than that (a 0402 that should still get a few fine cells).
    """

    chosen = set(references) if references is not None else None
    boxes = []
    for reference, solids in sorted(components.items()):
        if chosen is not None and reference not in chosen:
            continue
        lo = np.min([solid.bounds_m[0] for solid in solids], axis=0)
        hi = np.max([solid.bounds_m[1] for solid in solids], axis=0)
        x0, x1, y0, y1 = float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])
        if x1 - x0 < min_size_m:
            centre = 0.5 * (x0 + x1)
            x0, x1 = centre - 0.5 * min_size_m, centre + 0.5 * min_size_m
        if y1 - y0 < min_size_m:
            centre = 0.5 * (y0 + y1)
            y0, y1 = centre - 0.5 * min_size_m, centre + 0.5 * min_size_m
        boxes.append(RefinementBox(reference, x0, x1, y0, y1))
    return tuple(boxes)


def board_refined_grid(
    board: StepSolid,
    *,
    coarse_pitch_mm: float,
    fine_pitch_mm: float,
    boxes: Sequence[RefinementBox] = (),
    margin_mm: float = 1.0,
    growth: float = 1.4,
    origin_mm: tuple[float, float] | None = None,
    extent_mm: tuple[float, float] | None = None,
) -> TensorGrid:
    """A graded grid over the board's footprint, fine over the boxes plus the margin.

    The grid spans the board's bounding box (or ``origin_mm`` / ``extent_mm``
    when another tool fixed the frame) in the STEP frame; boxes outside it
    are clipped by the edge generator.
    """

    lo, hi = board.bounds_m
    x0, y0 = (lo[0], lo[1]) if origin_mm is None else (origin_mm[0] * MM, origin_mm[1] * MM)
    x1, y1 = (hi[0], hi[1]) if extent_mm is None else (x0 + extent_mm[0] * MM, y0 + extent_mm[1] * MM)
    return refined_grid(
        (x0, x1),
        (y0, y1),
        coarse_pitch_m=coarse_pitch_mm * MM,
        fine_pitch_m=fine_pitch_mm * MM,
        refine_boxes_m=[box.bounds_m for box in boxes],
        margin_m=margin_mm * MM,
        growth=growth,
    )


def refinement_summary(grid: TensorGrid, boxes: Sequence[RefinementBox]) -> dict[str, float | int]:
    """Cell counts of the grid against the uniform alternatives, for reports."""

    rows, cols = grid.shape
    fine = float(min(grid.pitch_x_m.min(), grid.pitch_y_m.min()))
    coarse = float(max(grid.pitch_x_m.max(), grid.pitch_y_m.max()))
    width = float(grid.x_edges_m[-1] - grid.x_edges_m[0])
    height = float(grid.y_edges_m[-1] - grid.y_edges_m[0])
    return {
        "rows": rows,
        "cols": cols,
        "cells": rows * cols,
        "boxes": len(boxes),
        "fine_pitch_mm": fine / MM,
        "coarse_pitch_mm": coarse / MM,
        "uniform_fine_cells": int(round(width / fine) * round(height / fine)),
        "uniform_coarse_cells": int(round(width / coarse) * round(height / coarse)),
    }


__all__ = ["RefinementBox", "board_refined_grid", "component_boxes", "refinement_summary"]
