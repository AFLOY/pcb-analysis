"""Tensor-product (graded Cartesian) in-plane grids shared by the structured solvers.

Every structured mesh in this repository is a tensor product of one set of
x grid lines and one set of y grid lines.  Letting the spacing vary along
each axis keeps the element topology, the corner views, the node-owned
gather kernels and the two-level preconditioner exactly as they are for a
uniform grid; only the per-element coefficients of the unit element matrices
change.  Refinement is therefore a choice of grid lines: fine where a
component, a narrow trace or a hot spot needs it, coarse elsewhere, with a
graded transition between the two so that neighbouring cells differ by at
most a chosen ratio.

Refining a line refines a whole strip of the board (the usual tensor-grid
over-refinement); that is the price of keeping every kernel structured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _check_edges(edges: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(edges, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError(f"{name} needs at least two grid lines")
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return values.copy()


@dataclass(frozen=True)
class TensorGrid:
    """Grid lines of a structured in-plane mesh, in metres.

    ``x_edges_m`` has ``cols + 1`` entries and ``y_edges_m`` ``rows + 1``.
    ``pitch_x_m`` / ``pitch_y_m`` are the per-column / per-row cell widths,
    which is what ``LayeredPCBMesh`` and ``LayeredThermalMesh`` take.
    """

    x_edges_m: np.ndarray
    y_edges_m: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_edges_m", _check_edges(self.x_edges_m, "x_edges_m"))
        object.__setattr__(self, "y_edges_m", _check_edges(self.y_edges_m, "y_edges_m"))

    @classmethod
    def uniform(
        cls, pitch_m: float | tuple[float, float], shape: tuple[int, int], origin_m: tuple[float, float] = (0.0, 0.0)
    ) -> "TensorGrid":
        """The ordinary uniform grid of ``shape = (rows, cols)`` cells."""

        px, py = (float(pitch_m), float(pitch_m)) if np.ndim(pitch_m) == 0 else (float(pitch_m[0]), float(pitch_m[1]))
        rows, cols = (int(value) for value in shape)
        return cls(origin_m[0] + px * np.arange(cols + 1), origin_m[1] + py * np.arange(rows + 1))

    @property
    def shape(self) -> tuple[int, int]:
        return self.y_edges_m.size - 1, self.x_edges_m.size - 1

    @property
    def pitch_x_m(self) -> np.ndarray:
        return np.diff(self.x_edges_m)

    @property
    def pitch_y_m(self) -> np.ndarray:
        return np.diff(self.y_edges_m)

    @property
    def origin_m(self) -> tuple[float, float]:
        return float(self.x_edges_m[0]), float(self.y_edges_m[0])

    @property
    def is_uniform(self) -> bool:
        return bool(np.allclose(self.pitch_x_m, self.pitch_x_m[0]) and np.allclose(self.pitch_y_m, self.pitch_y_m[0]))

    @property
    def cell_area_m2(self) -> np.ndarray:
        return self.pitch_y_m[:, None] * self.pitch_x_m[None, :]

    @property
    def x_centres_m(self) -> np.ndarray:
        return 0.5 * (self.x_edges_m[:-1] + self.x_edges_m[1:])

    @property
    def y_centres_m(self) -> np.ndarray:
        return 0.5 * (self.y_edges_m[:-1] + self.y_edges_m[1:])

    @property
    def size(self) -> int:
        rows, cols = self.shape
        return rows * cols

    def cell_of(self, x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Row and column of the cells containing the points (clipped to the grid)."""

        rows, cols = self.shape
        col = np.clip(np.searchsorted(self.x_edges_m, np.asarray(x_m), side="right") - 1, 0, cols - 1)
        row = np.clip(np.searchsorted(self.y_edges_m, np.asarray(y_m), side="right") - 1, 0, rows - 1)
        return row, col

    def cell_mask(self, x0_m: float, x1_m: float, y0_m: float, y1_m: float) -> np.ndarray:
        """Boolean ``(rows, cols)`` of the cells whose centre lies in the rectangle."""

        xc, yc = self.x_centres_m, self.y_centres_m
        return ((yc >= y0_m) & (yc <= y1_m))[:, None] & ((xc >= x0_m) & (xc <= x1_m))[None, :]

    def cell_overlap_fraction(self, x0_m: float, x1_m: float, y0_m: float, y1_m: float) -> np.ndarray:
        """Fraction of every cell's area inside the rectangle, ``(rows, cols)``.

        Summed against ``cell_area_m2`` this gives the rectangle's area exactly
        on any grid, so a power spread over a footprint lands on the same area
        whether or not the grid lines follow the footprint.
        """

        x = np.clip(np.minimum(self.x_edges_m[1:], x1_m) - np.maximum(self.x_edges_m[:-1], x0_m), 0.0, None)
        y = np.clip(np.minimum(self.y_edges_m[1:], y1_m) - np.maximum(self.y_edges_m[:-1], y0_m), 0.0, None)
        return (y[:, None] * x[None, :]) / self.cell_area_m2


def merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sort and merge overlapping or touching closed intervals."""

    ordered = sorted((min(a, b), max(a, b)) for a, b in intervals)
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def graded_edges(
    start_m: float,
    end_m: float,
    *,
    coarse_pitch_m: float,
    fine_pitch_m: float,
    refine_m: Sequence[tuple[float, float]] = (),
    margin_m: float = 0.0,
    growth: float = 1.5,
) -> np.ndarray:
    """Grid lines from ``start_m`` to ``end_m`` with fine cells over the refined intervals.

    Cells are ``fine_pitch_m`` wide inside every interval of ``refine_m``
    widened by ``margin_m`` on both sides, ``coarse_pitch_m`` far away, and
    grow geometrically by at most ``growth`` from one cell to the next in
    between.  Lines are placed where the integral of the reciprocal target
    size crosses whole numbers, so the last cell ends exactly at ``end_m``
    and the cell count is the rounded integral; the refined intervals need
    not start on a grid line, their cells are fine wherever they fall.
    """

    if not (np.isfinite(start_m) and np.isfinite(end_m)) or end_m <= start_m:
        raise ValueError("end_m must exceed start_m")
    if coarse_pitch_m <= 0.0 or fine_pitch_m <= 0.0 or fine_pitch_m > coarse_pitch_m:
        raise ValueError("pitches must be positive with fine_pitch_m <= coarse_pitch_m")
    if growth <= 1.0:
        raise ValueError("growth must exceed one")
    if margin_m < 0.0:
        raise ValueError("margin_m must be non-negative")
    fine = [(max(start_m, a - margin_m), min(end_m, b + margin_m)) for a, b in refine_m]
    fine = [(a, b) for a, b in merge_intervals(fine) if b > a]
    if not fine:
        count = max(1, int(round((end_m - start_m) / coarse_pitch_m)))
        return np.linspace(start_m, end_m, count + 1)

    # Target cell size: fine inside the intervals, growing geometrically away
    # from them (a progression of ratio r starting at h has size h + (r - 1) d
    # after covering the distance d), capped at the coarse pitch.
    samples = int(np.ceil((end_m - start_m) / fine_pitch_m * 8)) + 1
    x = np.linspace(start_m, end_m, samples)
    distance = np.full(samples, np.inf)
    for a, b in fine:
        distance = np.minimum(distance, np.maximum(0.0, np.maximum(x - b, a - x)))
    size = np.minimum(coarse_pitch_m, fine_pitch_m + (growth - 1.0) * distance)
    # Grid lines sit where the integral of 1 / size crosses whole numbers, so
    # every cell spans about one local target size and the last one ends at
    # end_m exactly (the count is rounded, which scales every cell alike).
    density = 1.0 / size
    cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(x))))
    count = max(1, int(round(cumulative[-1])))
    edges = np.interp(np.linspace(0.0, cumulative[-1], count + 1), cumulative, x)
    edges[0], edges[-1] = start_m, end_m
    return edges


def refined_grid(
    x_range_m: tuple[float, float],
    y_range_m: tuple[float, float],
    *,
    coarse_pitch_m: float,
    fine_pitch_m: float,
    refine_boxes_m: Sequence[tuple[float, float, float, float]] = (),
    margin_m: float = 0.0,
    growth: float = 1.5,
) -> TensorGrid:
    """A tensor grid fine over the given ``(x0, x1, y0, y1)`` boxes, coarse elsewhere."""

    x_intervals = [(x0, x1) for x0, x1, _, _ in refine_boxes_m]
    y_intervals = [(y0, y1) for _, _, y0, y1 in refine_boxes_m]
    options = dict(coarse_pitch_m=coarse_pitch_m, fine_pitch_m=fine_pitch_m, margin_m=margin_m, growth=growth)
    return TensorGrid(
        graded_edges(x_range_m[0], x_range_m[1], refine_m=x_intervals, **options),
        graded_edges(y_range_m[0], y_range_m[1], refine_m=y_intervals, **options),
    )


def check_pitch_axis(value: object, count: int, name: str) -> np.ndarray:
    """Normalise a scalar or per-cell pitch to a ``(count,)`` array of positive lengths."""

    pitch = np.asarray(value, dtype=np.float64)
    if pitch.ndim == 0:
        pitch = np.full(count, float(pitch))
    elif pitch.shape != (count,):
        raise ValueError(f"{name} must be a scalar or hold one value per cell ({count})")
    if not np.all(np.isfinite(pitch)) or np.any(pitch <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return pitch.copy()


__all__ = ["TensorGrid", "check_pitch_axis", "graded_edges", "merge_intervals", "refined_grid"]
