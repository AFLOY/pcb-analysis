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

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .bodymap import BoardSpec, CopperSpec, LayerSpec, ResolvedBodies
from electrical.matrix_free_mpir_fem.grid import TensorGrid

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
    origin_m: tuple[float, float] | None = None,
    pitch_m: float | None = None,
    shape: tuple[int, int] | None = None,
    grid: TensorGrid | None = None,
    supersample: int = 3,
    method: ClassifyMethod = "auto",
) -> np.ndarray:
    """Fraction of every ``(row, col)`` cell covered by the solids at ``z``.

    The grid is uniform (``origin_m``, ``pitch_m``, ``shape``) or a
    ``TensorGrid`` in the STEP frame (``grid``).  ``method="section"``
    returns the exact area fraction and ignores ``supersample``; the point
    methods return the sampled fraction.
    """

    if grid is None:
        if origin_m is None or pitch_m is None or shape is None:
            raise ValueError("pass grid, or origin_m, pitch_m and shape")
        grid = TensorGrid.uniform(pitch_m, shape, origin_m)
    elif origin_m is not None or pitch_m is not None or shape is not None:
        raise ValueError("pass either grid or origin_m, pitch_m and shape")
    rows, cols = grid.shape
    chosen = default_plane_method() if method == "auto" else method
    crossing = [solid for solid in solids if solid.bounds_m[0][2] - 1.0e-12 <= z_m <= solid.bounds_m[1][2] + 1.0e-12]
    if chosen == "section":
        meshes = [solid.tessellate() for solid in crossing]
        if grid.is_uniform:
            return plane_section_coverage(
                meshes, z_m, origin_m=grid.origin_m, pitch_m=float(grid.pitch_x_m[0]), shape=(rows, cols)
            )
        return plane_section_coverage(meshes, z_m, x_edges_m=grid.x_edges_m, y_edges_m=grid.y_edges_m)
    method = chosen
    offsets = _cell_sample_offsets(supersample)
    x = (grid.x_edges_m[:-1, None] + grid.pitch_x_m[:, None] * offsets[None, :]).reshape(-1)
    y = (grid.y_edges_m[:-1, None] + grid.pitch_y_m[:, None] * offsets[None, :]).reshape(-1)
    grid_x, grid_y = np.meshgrid(x, y)  # (rows*s, cols*s)
    points = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1), np.full(grid_x.size, z_m)))
    inside = np.zeros(points.shape[0], dtype=bool)
    for solid in crossing:
        inside |= solid.contains(points, method=method)
    fine = inside.reshape(rows, supersample, cols, supersample)
    return fine.mean(axis=(1, 3))


@dataclass(frozen=True)
class BoardRaster:
    """The board on the routing grid: outline and per-layer copper fill.

    ``grid`` holds the grid lines in the STEP frame (metres, y up); it is
    built from ``pitch_mm`` and ``origin_mm`` when not given, and those two
    describe it when it is uniform (``pitch_mm`` is ``None`` on a graded
    grid).  Row 0 of the arrays is the row of smallest ``y`` unless
    ``y_down``, in which case the rows are stored top-down and ``pitch_y_m``
    / ``row_y_m`` follow that order.
    """

    spec: BoardSpec
    pitch_mm: float | None
    origin_mm: tuple[float, float]
    outline: np.ndarray
    fill: np.ndarray
    threshold: float = 0.5
    y_down: bool = False
    measured_thickness_mm: tuple[float | None, ...] = ()
    grid: TensorGrid | None = None

    def __post_init__(self) -> None:
        outline = np.asarray(self.outline, dtype=bool)
        fill = np.asarray(self.fill, dtype=np.float64)
        if outline.ndim != 2 or fill.shape != (len(self.spec.layers),) + outline.shape:
            raise ValueError("fill must be (layers, rows, cols) over the outline grid")
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must lie in (0, 1]")
        if self.grid is None:
            if self.pitch_mm is None:
                raise ValueError("a raster without a grid needs pitch_mm")
            grid = TensorGrid.uniform(float(self.pitch_mm) * MM, outline.shape, (self.origin_mm[0] * MM, self.origin_mm[1] * MM))
        else:
            grid = self.grid
            if grid.shape != outline.shape:
                raise ValueError("grid shape must match the outline (rows, cols)")
            origin = (grid.origin_m[0] / MM, grid.origin_m[1] / MM)
            object.__setattr__(self, "origin_mm", (float(origin[0]), float(origin[1])))
            pitch = float(grid.pitch_x_m[0]) / MM if grid.is_uniform else None
            object.__setattr__(self, "pitch_mm", pitch)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "outline", outline)
        object.__setattr__(self, "fill", np.where(outline[None], fill, 0.0))
        measured = tuple(self.measured_thickness_mm)
        if not measured:
            measured = (None,) * len(self.spec.layers)
        if len(measured) != len(self.spec.layers):
            raise ValueError("measured_thickness_mm needs one entry per layer")
        object.__setattr__(self, "measured_thickness_mm", measured)

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
    def is_uniform(self) -> bool:
        return self.grid.is_uniform  # type: ignore[union-attr]

    @property
    def pitch_m(self) -> float:
        """The uniform pitch in metres; raises on a graded grid."""

        if self.pitch_mm is None:
            raise ValueError("the raster grid is graded; use pitch_x_m / pitch_y_m")
        return self.pitch_mm * MM

    @property
    def pitch_x_m(self) -> np.ndarray:
        """Column widths in metres, ``(cols,)``."""

        return self.grid.pitch_x_m  # type: ignore[union-attr]

    @property
    def pitch_y_m(self) -> np.ndarray:
        """Row heights in metres in array row order (top-down when ``y_down``), ``(rows,)``."""

        pitch = self.grid.pitch_y_m  # type: ignore[union-attr]
        return pitch[::-1].copy() if self.y_down else pitch

    @property
    def cell_area_m2(self) -> np.ndarray:
        """Cell areas ``(rows, cols)`` in array row order."""

        return self.pitch_y_m[:, None] * self.pitch_x_m[None, :]

    @property
    def origin_m(self) -> tuple[float, float]:
        return (self.origin_mm[0] * MM, self.origin_mm[1] * MM)

    def layer_thickness_mm(self, index: int, *, source: str = "stackup") -> float:
        """Thickness of one layer from the stackup or from the copper solids.

        ``source="measured"`` falls back to the stackup for a layer without
        copper solids.
        """

        if source not in ("stackup", "measured"):
            raise ValueError("source must be 'stackup' or 'measured'")
        measured = self.measured_thickness_mm[index]
        if source == "measured" and measured is not None:
            return float(measured)
        return float(self.spec.layers[index].thickness_mm)

    def thickness_mismatches(self, *, relative_tolerance: float = 0.05) -> tuple[tuple[str, float, float], ...]:
        """Layers whose copper solids are thicker or thinner than the stackup says.

        Each entry is ``(layer name, stackup mm, measured mm)``.
        """

        out = []
        for layer, measured in zip(self.spec.layers, self.measured_thickness_mm):
            if measured is None:
                continue
            if abs(measured - layer.thickness_mm) > relative_tolerance * layer.thickness_mm:
                out.append((layer.name, float(layer.thickness_mm), float(measured)))
        return tuple(out)

    def copper_area_m2(self, layer: int) -> float:
        return float(np.sum(self.fill[layer] * self.cell_area_m2))

    def cell_of(self, x_m: float, y_m: float) -> tuple[int, int]:
        """Cell of a point in the STEP frame; row 0 is at the origin unless ``y_down``."""

        grid = self.grid
        assert grid is not None
        rows, cols = self.shape
        if not (grid.x_edges_m[0] <= x_m < grid.x_edges_m[-1] and grid.y_edges_m[0] <= y_m < grid.y_edges_m[-1]):
            raise ValueError(f"point ({x_m}, {y_m}) lies outside the board grid")
        col = int(np.searchsorted(grid.x_edges_m, x_m, side="right") - 1)
        row = int(np.searchsorted(grid.y_edges_m, y_m, side="right") - 1)
        if self.y_down:
            row = rows - 1 - row
        return row, col

    def row_y_m(self, row: int) -> float:
        """Centre ``y`` (STEP frame) of a row."""

        rows = self.shape[0]
        index = rows - 1 - row if self.y_down else row
        return float(self.grid.y_centres_m[index])  # type: ignore[union-attr]


def rasterize_board(
    resolved: ResolvedBodies,
    spec: BoardSpec,
    *,
    pitch_mm: float | None = None,
    supersample: int = 3,
    origin_mm: tuple[float, float] | None = None,
    threshold: float = 0.5,
    method: ClassifyMethod = "auto",
    shape: tuple[int, int] | None = None,
    y_down: bool = False,
    thickness_tolerance: float = 0.05,
    grid: TensorGrid | None = None,
) -> BoardRaster:
    """Sample the board outline and each layer's copper onto the grid.

    The grid is uniform with ``pitch_mm``: its origin defaults to the board's
    minimum corner and it covers the board's bounding box with whole cells,
    or ``shape`` fixes the ``(rows, cols)`` for a grid another tool has
    already chosen.  Or it is a ``TensorGrid`` in the STEP frame (metres),
    typically from :func:`geometry.cad_import.refinement.board_refined_grid`.
    With ``y_down`` row 0 is the row of largest STEP ``y``, which is KiCad's
    and plane-opt's y-down convention (KiCad exports STEP with ``y`` negated,
    so a KiCad grid whose rows count downwards maps onto STEP rows counted
    from the top).
    """

    board = resolved.board
    lo, hi = board.bounds_m
    if grid is None:
        if pitch_mm is None or pitch_mm <= 0.0 or not np.isfinite(pitch_mm):
            raise ValueError("pitch_mm must be positive unless a grid is given")
        origin = (lo[0] / MM, lo[1] / MM) if origin_mm is None else origin_mm
        if shape is None:
            cols = int(np.ceil((hi[0] / MM - origin[0]) / pitch_mm - 1.0e-6))
            rows = int(np.ceil((hi[1] / MM - origin[1]) / pitch_mm - 1.0e-6))
        else:
            rows, cols = (int(axis) for axis in shape)
        if rows < 1 or cols < 1:
            raise ValueError("the grid origin lies beyond the board")
        grid = TensorGrid.uniform(pitch_mm * MM, (rows, cols), (origin[0] * MM, origin[1] * MM))
    elif pitch_mm is not None or origin_mm is not None or shape is not None:
        raise ValueError("pass either grid or pitch_mm / origin_mm / shape")
    rows, cols = grid.shape
    origin = (grid.origin_m[0] / MM, grid.origin_m[1] / MM)
    board_mid_z = (lo[2] + hi[2]) / 2.0
    outline = (
        sample_plane_fill([board], z_m=board_mid_z, grid=grid, supersample=supersample, method=method)
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
    measured: list[float | None] = []
    for index, layer in enumerate(spec.layers):
        solids = by_layer[layer.name]
        # The copper solids' own z extent, weighted by volume so a stray
        # sliver does not outvote the plane; barrels span layers and are left out.
        copper_only = [solid for copper_spec, group in resolved.copper if copper_spec.layer == layer.name for solid in group]
        if copper_only:
            heights = np.array([(solid.bounds_m[1][2] - solid.bounds_m[0][2]) / MM for solid in copper_only])
            weights = np.array([solid.volume_m3 for solid in copper_only])
            measured.append(float(np.sum(heights * weights) / np.sum(weights)))
        else:
            measured.append(None)
        if not solids:
            continue
        fill[index] = sample_plane_fill(
            solids, z_m=layer.center_z_mm * MM, grid=grid, supersample=supersample, method=method
        )
    if y_down:
        outline = outline[::-1].copy()
        fill = fill[:, ::-1].copy()
    raster = BoardRaster(
        spec, None, (float(origin[0]), float(origin[1])), outline, fill, threshold, y_down, tuple(measured), grid=grid
    )
    for name, stackup_mm, measured_mm in raster.thickness_mismatches(relative_tolerance=thickness_tolerance):
        warnings.warn(
            f"layer {name}: the copper solids are {measured_mm:.4f} mm thick but the stackup says "
            f"{stackup_mm:.4f} mm; pass source='measured' to use the solids' thickness",
            stacklevel=2,
        )
    return raster


__all__ = ["BoardRaster", "rasterize_board", "sample_plane_fill"]
