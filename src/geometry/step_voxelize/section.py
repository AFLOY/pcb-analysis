"""Planar rasterisation of the board and its copper onto the routing grid (2.5D).

Each conductor layer is cut at its centre ``z``.  With the native extension
the cut is exact: the copper solids' tessellations are sectioned into
oriented loops and rasterised to the exact fraction of every cell they cover
(``method="section"``, the default when built).  Without it, ``supersample²``
points per cell are classified against the solids and the inside fraction is
the fill (``"numpy"``, ``"native"`` or the per-point OpenCASCADE ``"occ"``).
The board solid treated the same way gives the outline, so a board that is
not a rectangle becomes an active-element mask on the thermal mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .bodymap import BoardSpec, CopperSpec, LayerSpec, ResolvedBodies
from .mesh import ClassifyMethod, default_plane_method, plane_section_coverage
from .reader import MM, StepSolid


def _cell_sample_offsets(supersample: int) -> np.ndarray:
    if supersample < 1:
        raise ValueError("supersample must be at least one")
    return (np.arange(supersample) + 0.5) / supersample


def sample_plane_fill(
    solids: Sequence[StepSolid],
    *,
    z_m: float,
    origin_m: tuple[float, float],
    pitch_m: float,
    shape: tuple[int, int],
    supersample: int = 3,
    method: ClassifyMethod = "auto",
) -> np.ndarray:
    """Fraction of every ``(row, col)`` cell covered by the solids at ``z``.

    ``method="section"`` returns the exact area fraction and ignores
    ``supersample``; the point methods return the sampled fraction.
    """

    rows, cols = shape
    chosen = default_plane_method() if method == "auto" else method
    if chosen == "section":
        return plane_section_coverage(
            [solid.tessellate() for solid in solids if solid.bounds_m[0][2] - 1.0e-12 <= z_m <= solid.bounds_m[1][2] + 1.0e-12],
            z_m,
            origin_m=origin_m,
            pitch_m=pitch_m,
            shape=(rows, cols),
        )
    method = chosen
    offsets = _cell_sample_offsets(supersample)
    x = origin_m[0] + pitch_m * (np.arange(cols)[:, None] + offsets[None, :]).reshape(-1)
    y = origin_m[1] + pitch_m * (np.arange(rows)[:, None] + offsets[None, :]).reshape(-1)
    grid_x, grid_y = np.meshgrid(x, y)  # (rows*s, cols*s)
    points = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1), np.full(grid_x.size, z_m)))
    inside = np.zeros(points.shape[0], dtype=bool)
    for solid in solids:
        lo, hi = solid.bounds_m
        if not (lo[2] - 1.0e-12 <= z_m <= hi[2] + 1.0e-12):
            continue
        inside |= solid.contains(points, method=method)
    fine = inside.reshape(rows, supersample, cols, supersample)
    return fine.mean(axis=(1, 3))


@dataclass(frozen=True)
class BoardRaster:
    """The board on the routing grid: outline and per-layer copper fill."""

    spec: BoardSpec
    pitch_mm: float
    origin_mm: tuple[float, float]
    outline: np.ndarray
    fill: np.ndarray
    threshold: float = 0.5
    y_down: bool = False

    def __post_init__(self) -> None:
        outline = np.asarray(self.outline, dtype=bool)
        fill = np.asarray(self.fill, dtype=np.float64)
        if outline.ndim != 2 or fill.shape != (len(self.spec.layers),) + outline.shape:
            raise ValueError("fill must be (layers, rows, cols) over the outline grid")
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must lie in (0, 1]")
        object.__setattr__(self, "outline", outline)
        object.__setattr__(self, "fill", np.where(outline[None], fill, 0.0))

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(axis) for axis in self.outline.shape)  # type: ignore[return-value]

    @property
    def layers(self) -> tuple[LayerSpec, ...]:
        return self.spec.layers

    @property
    def occupancy(self) -> np.ndarray:
        """Copper occupancy ``(layers, rows, cols)`` as 0/1 floats."""

        return (self.fill >= self.threshold).astype(np.float64)

    @property
    def pitch_m(self) -> float:
        return self.pitch_mm * MM

    @property
    def origin_m(self) -> tuple[float, float]:
        return (self.origin_mm[0] * MM, self.origin_mm[1] * MM)

    def copper_area_m2(self, layer: int) -> float:
        return float(np.sum(self.fill[layer])) * self.pitch_m**2

    def cell_of(self, x_m: float, y_m: float) -> tuple[int, int]:
        """Cell of a point in the STEP frame; row 0 is at the origin unless ``y_down``."""

        col = int(np.floor((x_m - self.origin_m[0]) / self.pitch_m))
        row = int(np.floor((y_m - self.origin_m[1]) / self.pitch_m))
        rows, cols = self.shape
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError(f"point ({x_m}, {y_m}) lies outside the board grid")
        if self.y_down:
            row = rows - 1 - row
        return row, col

    def row_y_m(self, row: int) -> float:
        """Centre ``y`` (STEP frame) of a row."""

        rows = self.shape[0]
        index = rows - 1 - row if self.y_down else row
        return self.origin_m[1] + (index + 0.5) * self.pitch_m


def rasterize_board(
    resolved: ResolvedBodies,
    spec: BoardSpec,
    *,
    pitch_mm: float,
    supersample: int = 3,
    origin_mm: tuple[float, float] | None = None,
    threshold: float = 0.5,
    method: ClassifyMethod = "auto",
    shape: tuple[int, int] | None = None,
    y_down: bool = False,
) -> BoardRaster:
    """Sample the board outline and each layer's copper onto the grid.

    The grid origin defaults to the board's minimum corner and the grid
    covers the board's bounding box with whole cells; ``shape`` fixes the
    ``(rows, cols)`` instead, for a grid another tool has already chosen.
    With ``y_down`` row 0 is the row of largest STEP ``y``, which is KiCad's
    and plane-opt's y-down convention (KiCad exports STEP with ``y`` negated,
    so a KiCad grid whose rows count downwards maps onto STEP rows counted
    from the top).
    """

    if pitch_mm <= 0.0 or not np.isfinite(pitch_mm):
        raise ValueError("pitch_mm must be positive")
    board = resolved.board
    lo, hi = board.bounds_m
    origin = (lo[0] / MM, lo[1] / MM) if origin_mm is None else origin_mm
    if shape is None:
        cols = int(np.ceil((hi[0] / MM - origin[0]) / pitch_mm - 1.0e-6))
        rows = int(np.ceil((hi[1] / MM - origin[1]) / pitch_mm - 1.0e-6))
    else:
        rows, cols = (int(axis) for axis in shape)
    if rows < 1 or cols < 1:
        raise ValueError("the grid origin lies beyond the board")
    pitch_m = pitch_mm * MM
    origin_m = (origin[0] * MM, origin[1] * MM)
    board_mid_z = (lo[2] + hi[2]) / 2.0
    outline = (
        sample_plane_fill(
            [board], z_m=board_mid_z, origin_m=origin_m, pitch_m=pitch_m, shape=(rows, cols), supersample=supersample, method=method
        )
        >= threshold
    )
    by_layer: dict[str, list[StepSolid]] = {layer.name: [] for layer in spec.layers}
    for copper_spec, solids in resolved.copper:
        by_layer[copper_spec.layer].extend(solids)
    # Via barrels are copper on every layer they pass through (their annular
    # rings), so they join each layer's sampling; ``sample_plane_fill`` skips
    # any solid whose z extent misses the layer plane.
    via_solids = [solid for _, solids in resolved.vias for solid in solids]
    for name in by_layer:
        by_layer[name].extend(via_solids)
    fill = np.zeros((len(spec.layers), rows, cols))
    for index, layer in enumerate(spec.layers):
        solids = by_layer[layer.name]
        if not solids:
            continue
        fill[index] = sample_plane_fill(
            solids,
            z_m=layer.center_z_mm * MM,
            origin_m=origin_m,
            pitch_m=pitch_m,
            shape=(rows, cols),
            supersample=supersample,
            method=method,
        )
    if y_down:
        outline = outline[::-1].copy()
        fill = fill[:, ::-1].copy()
    return BoardRaster(spec, pitch_mm, (float(origin[0]), float(origin[1])), outline, fill, threshold, y_down)


__all__ = ["BoardRaster", "rasterize_board", "sample_plane_fill"]
