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

from .bodymap import ResolvedBodies
from .reader import StepSolid
from .section import sample_plane_fill

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


def narrow_copper_boxes(
    resolved: ResolvedBodies,
    layer_z_mm: Sequence[float],
    *,
    width_threshold_mm: float = 0.3,
    sample_pitch_mm: float = 0.05,
    tile_mm: float = 2.0,
    min_cells: int = 8,
) -> tuple[RefinementBox, ...]:
    """Boxes around the copper narrower than ``width_threshold_mm`` on any layer.

    Every layer's copper (tracks, pads, zones and barrels) is sectioned onto a
    uniform grid of ``sample_pitch_mm``; a morphological opening with a disc
    of the threshold diameter removes every feature thinner than that, and
    what the opening removed is the narrow copper (the edges of wide copper
    survive the opening and are not flagged).
    Their connected components are cut into ``tile_mm`` tiles so a long or
    diagonal trace yields a chain of tight boxes instead of one bounding box
    that would refine a whole strip of the board. Components below
    ``min_cells`` sample cells are noise from copper corners and are dropped.
    """

    from scipy import ndimage

    if width_threshold_mm <= 0.0 or sample_pitch_mm <= 0.0 or tile_mm <= 0.0:
        raise ValueError("width_threshold_mm, sample_pitch_mm and tile_mm must be positive")
    board = resolved.board
    lo, hi = (np.asarray(b) for b in board.bounds_m)
    pitch = sample_pitch_mm * MM
    cols = int(np.ceil((hi[0] - lo[0]) / pitch - 1e-9))
    rows = int(np.ceil((hi[1] - lo[1]) / pitch - 1e-9))
    grid = TensorGrid.uniform(pitch, (rows, cols), (float(lo[0]), float(lo[1])))
    copper = [solid for _, solids in resolved.copper for solid in solids] + [s for _, solids in resolved.vias for s in solids]
    narrow = np.zeros((rows, cols), dtype=bool)
    radius = max(1, int(round(0.5 * width_threshold_mm / sample_pitch_mm)))
    offsets = np.arange(-radius, radius + 1)
    disc = (offsets[:, None] ** 2 + offsets[None, :] ** 2) <= radius**2
    square = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    for z_mm in layer_z_mm:
        mask = sample_plane_fill(copper, z_m=float(z_mm) * MM, grid=grid, method="section") >= 0.5
        if not mask.any():
            continue
        wide = ndimage.binary_opening(mask, structure=disc) | ndimage.binary_opening(mask, structure=square)
        narrow |= mask & ~wide
    labels, count = ndimage.label(narrow)
    boxes: list[RefinementBox] = []
    tile = max(1, int(round(tile_mm / sample_pitch_mm)))
    for index, slc in enumerate(ndimage.find_objects(labels), start=1):
        if slc is None:
            continue
        component = labels[slc] == index
        if int(component.sum()) < min_cells:
            continue
        r0, c0 = slc[0].start, slc[1].start
        for tr in range(0, component.shape[0], tile):
            for tc in range(0, component.shape[1], tile):
                part = component[tr : tr + tile, tc : tc + tile]
                if not part.any():
                    continue
                pr, pc = np.nonzero(part)
                rows_lo, rows_hi = r0 + tr + pr.min(), r0 + tr + pr.max() + 1
                cols_lo, cols_hi = c0 + tc + pc.min(), c0 + tc + pc.max() + 1
                boxes.append(
                    RefinementBox(
                        f"trace{index}:{tr // tile},{tc // tile}",
                        float(grid.x_edges_m[cols_lo]),
                        float(grid.x_edges_m[cols_hi]),
                        float(grid.y_edges_m[rows_lo]),
                        float(grid.y_edges_m[rows_hi]),
                    )
                )
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


__all__ = ["RefinementBox", "board_refined_grid", "component_boxes", "narrow_copper_boxes", "refinement_summary"]
