"""Partial inductance between the rectangular cells of a layered PEEC mesh.

A PCB is a few thin sheets of copper at known heights.  Meshing it as a uniform
three-dimensional voxel grid pays for the board's whole height at the copper's
thickness, which is a factor of tens of empty cells.  Treating each layer as a
sheet pays for the copper only, and the price is that the vertical structure --
the barrels -- has to be carried explicitly instead of falling out of voxel
adjacency.

This module supplies the piece that makes the sheet form worth having: the
partial inductance between two rectangular cells, as a function of their
in-plane offset and their vertical separation alone.  That is what lets the
operator be a two-dimensional convolution per layer pair rather than a
three-dimensional one over the whole board.

Two properties of an axis-aligned mesh are used throughout.

* A cell carrying current along x and a cell carrying current along y have zero
  mutual partial inductance, because the defining integral carries the factor
  ``dl_i . dl_j``.  Only the like-directed operators exist.
* The y operator is the x operator with the two in-plane offsets exchanged, so
  one kernel serves both.

The quantity is Ruehli's partial mutual inductance between two parallel
rectangular bars,

.. math::

    L_{ij} = \\frac{\\mu_0}{4\\pi a_i a_j}
             \\int_{V_i}\\int_{V_j} \\frac{1}{|r - r'|} \\, dV' \\, dV

with :math:`a` the cross-section normal to the current.  It is evaluated in
closed form rather than by quadrature.  Quadrature fails exactly where this
mesh needs the answer most: the self term places the two bars on top of each
other, where a product rule puts nodes at zero separation, and the nearest
neighbours of a 0.2mm cell are close enough that no affordable order resolves
the peak of ``1/r``.  The closed form has neither problem and is exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

VACUUM_PERMEABILITY = 4.0e-7 * math.pi
_MU0_OVER_4PI = VACUUM_PERMEABILITY / (4.0 * math.pi)

# Signed corner values of one axis: the double integral over two intervals of a
# function of their difference is the second antiderivative evaluated at the
# four combinations of the half-extents, with the sign of their product.
_AXIS_SIGNS = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))


@dataclass(frozen=True)
class CellGeometry:
    """One mesh cell's extent, in metres.

    ``length`` runs along the current, ``width`` across it in plane, and
    ``thickness`` is the copper.  A cell of the x operator has ``length`` along
    x; the y operator uses the same numbers with the in-plane offsets swapped.
    """

    length_m: float
    width_m: float
    thickness_m: float

    def __post_init__(self) -> None:
        for name in ("length_m", "width_m", "thickness_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)

    @property
    def cross_section_m2(self) -> float:
        return self.width_m * self.thickness_m

    @property
    def volume_m3(self) -> float:
        return self.length_m * self.width_m * self.thickness_m


def _safe_log(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Take ``log(numerator / denominator)`` where both are usable, else zero.

    Every logarithm in the primitive is multiplied by a coefficient that
    vanishes wherever its argument degenerates, so replacing the degenerate
    value with zero yields the correct limit of the product.
    """
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    ratio = np.divide(
        numerator,
        denominator,
        out=np.ones(numerator.shape, dtype=np.float64),
        where=(denominator > 0.0) & (numerator > 0.0),
    )
    return np.log(np.where(ratio > 0.0, ratio, 1.0))


def _safe_atan(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Take ``arctan(numerator / denominator)``, zero where undefined.

    As with the logarithms, the arctangent terms carry coefficients that vanish
    wherever the denominator does.
    """
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.arctan(
        np.divide(
            numerator,
            denominator,
            out=np.zeros(numerator.shape, dtype=np.float64),
            where=denominator != 0.0,
        )
    )


def _primitive(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate the sixfold antiderivative of ``1/r`` used by Hoer and Love.

    This is the function whose mixed fourth-and-second derivatives return
    ``1 / sqrt(x^2 + y^2 + z^2)``, so that a double volume integral of the
    Green function over two boxes becomes a signed sum of its values at the
    boxes' corner differences.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    x2, y2, z2 = x * x, y * y, z * z
    rho = np.sqrt(x2 + y2 + z2)

    term = (
        (y2 * z2 / 4.0 - y2 * y2 / 24.0 - z2 * z2 / 24.0)
        * x
        * _safe_log(x + rho, np.sqrt(y2 + z2))
    )
    term += (
        (x2 * z2 / 4.0 - x2 * x2 / 24.0 - z2 * z2 / 24.0)
        * y
        * _safe_log(y + rho, np.sqrt(x2 + z2))
    )
    term += (
        (x2 * y2 / 4.0 - x2 * x2 / 24.0 - y2 * y2 / 24.0)
        * z
        * _safe_log(z + rho, np.sqrt(x2 + y2))
    )
    term += (
        (x2 * x2 + y2 * y2 + z2 * z2 - 3.0 * x2 * y2 - 3.0 * y2 * z2 - 3.0 * z2 * x2)
        * rho
        / 60.0
    )
    term -= (x * y * z * z2 / 6.0) * _safe_atan(x * y, z * rho)
    term -= (x * y * y2 * z / 6.0) * _safe_atan(x * z, y * rho)
    term -= (x2 * x * y * z / 6.0) * _safe_atan(y * z, x * rho)
    return term


# Beyond this separation, in units of the larger in-plane cell extent, the
# closed form is used only as a reference: see `_closed_form_precision` for why
# it stops being usable, and `NEAR_RADIUS_CELLS` for where the switch is made.
# The two regimes overlap widely -- at 24 cells the closed form still retains
# seven digits while the centre-to-centre limit is already within 1.4e-4 -- so
# the seam is far from either one's difficulty.
NEAR_RADIUS_CELLS = 24


def closed_form_mutual_inductance(
    cell: CellGeometry,
    other: CellGeometry,
    offset_m: tuple[float, float, float] | np.ndarray,
    *,
    check_precision: bool = True,
) -> float | np.ndarray:
    """Give the exact partial mutual inductance of two parallel bars, in henries.

    ``offset_m`` is the displacement from the first cell's centre to the
    second's, as (along current, across current in plane, vertical).  It may be
    a triple of scalars or a triple of broadcastable arrays, in which case the
    result has the broadcast shape.  The self inductance is the zero offset of
    a cell with itself and is finite: the singularity of ``1/r`` is integrable,
    and this takes the integral rather than sampling it.

    Exact in exact arithmetic, and progressively less so in floating point as
    the bars separate.  :func:`mutual_partial_inductance` is the entry point
    that keeps to the range where this is trustworthy.
    """
    du, dv, dw = (np.asarray(value, dtype=np.float64) for value in offset_m)
    extents = (
        (du, cell.length_m, other.length_m),
        (dv, cell.width_m, other.width_m),
        (dw, cell.thickness_m, other.thickness_m),
    )

    shape = np.broadcast(du, dv, dw).shape
    total = np.zeros(shape, dtype=np.float64)
    largest = np.zeros(shape, dtype=np.float64)
    for sx, tx in _AXIS_SIGNS:
        x = extents[0][0] + sx * extents[0][1] / 2.0 + tx * extents[0][2] / 2.0
        for sy, ty in _AXIS_SIGNS:
            y = extents[1][0] + sy * extents[1][1] / 2.0 + ty * extents[1][2] / 2.0
            for sz, tz in _AXIS_SIGNS:
                z = extents[2][0] + sz * extents[2][1] / 2.0 + tz * extents[2][2] / 2.0
                value = _primitive(x, y, z)
                total = total + (sx * tx) * (sy * ty) * (sz * tz) * value
                largest = np.maximum(largest, np.abs(value))

    # The 64 terms individually grow like the fifth power of the corner
    # coordinates while their signed sum stays of order the answer, so the
    # further apart the bars are, the more of the sum cancels.  Carry the
    # fraction that survives so the caller can see how many digits are left.
    with np.errstate(divide="ignore", invalid="ignore"):
        retained = np.where(largest > 0.0, np.abs(total) / largest, 1.0)
    # A retained fraction r leaves a relative error of about eps/r, so 1e-11
    # still carries five digits.  The near radius is chosen well inside this.
    if check_precision and float(np.min(retained)) < 1e-11:
        raise ValueError(
            "partial inductance lost its precision to cancellation: only "
            f"{float(np.min(retained)):.1e} of the summed magnitude survives. "
            "The closed form is for cells near each other; use "
            "mutual_partial_inductance, which switches to the centre-to-centre "
            "limit past NEAR_RADIUS_CELLS"
        )

    scale = _MU0_OVER_4PI / (cell.cross_section_m2 * other.cross_section_m2)
    result = scale * total
    return float(result) if result.ndim == 0 else result


def closed_form_precision(
    cell: CellGeometry,
    other: CellGeometry,
    offset_m: tuple[float, float, float] | np.ndarray,
) -> float | np.ndarray:
    """Report the fraction of the signed sum that survives cancellation.

    One means nothing cancelled; ``1e-16`` means nothing is left.  Used to fix
    :data:`NEAR_RADIUS_CELLS` against a measurement rather than a guess.
    """
    du, dv, dw = (np.asarray(value, dtype=np.float64) for value in offset_m)
    extents = (
        (du, cell.length_m, other.length_m),
        (dv, cell.width_m, other.width_m),
        (dw, cell.thickness_m, other.thickness_m),
    )
    shape = np.broadcast(du, dv, dw).shape
    total = np.zeros(shape, dtype=np.float64)
    largest = np.zeros(shape, dtype=np.float64)
    for sx, tx in _AXIS_SIGNS:
        x = extents[0][0] + sx * extents[0][1] / 2.0 + tx * extents[0][2] / 2.0
        for sy, ty in _AXIS_SIGNS:
            y = extents[1][0] + sy * extents[1][1] / 2.0 + ty * extents[1][2] / 2.0
            for sz, tz in _AXIS_SIGNS:
                z = extents[2][0] + sz * extents[2][1] / 2.0 + tz * extents[2][2] / 2.0
                value = _primitive(x, y, z)
                total = total + (sx * tx) * (sy * ty) * (sz * tz) * value
                largest = np.maximum(largest, np.abs(value))
    with np.errstate(divide="ignore", invalid="ignore"):
        retained = np.where(largest > 0.0, np.abs(total) / largest, 1.0)
    return float(retained) if retained.ndim == 0 else retained


def mutual_partial_inductance(
    cell: CellGeometry,
    other: CellGeometry,
    offset_m: tuple[float, float, float] | np.ndarray,
    *,
    near_radius_cells: int = NEAR_RADIUS_CELLS,
    scale_m: float | None = None,
) -> float | np.ndarray:
    """Give the partial mutual inductance of two parallel bars, in henries.

    Near cells take the closed form, where the bars' extent matters and the
    formula is well conditioned.  Distant cells take the centre-to-centre
    limit, where the extent no longer matters and the closed form has cancelled
    away its own precision.  ``near_radius_cells`` is the crossover, measured
    in multiples of ``scale_m``.

    ``scale_m`` defaults to the largest extent of either bar, which is what one
    cell means for a branch running in plane.  A branch running through the
    board is a different shape: it can be far longer than the grid it stands on,
    and taking its own length as the unit would push the crossover out to where
    the closed form has no precision left.  Such a caller passes the in-plane
    pitch instead.
    """
    du, dv, dw = np.broadcast_arrays(
        *(np.asarray(value, dtype=np.float64) for value in offset_m)
    )
    if scale_m is None:
        pitch = max(cell.length_m, cell.width_m, other.length_m, other.width_m)
    else:
        pitch = float(scale_m)
        if pitch <= 0.0:
            raise ValueError("scale_m must be positive")
    radius = np.sqrt(du * du + dv * dv + dw * dw)
    near = radius <= near_radius_cells * pitch

    result = np.asarray(
        far_field_mutual_inductance(cell, other, (du, dv, dw)), dtype=np.float64
    ).copy()
    if near.any():
        result[near] = np.asarray(
            closed_form_mutual_inductance(
                cell, other, (du[near], dv[near], dw[near])
            ),
            dtype=np.float64,
        )
    return float(result) if result.ndim == 0 else result


def self_partial_inductance(cell: CellGeometry) -> float:
    """Give a cell's own partial inductance."""
    return float(closed_form_mutual_inductance(cell, cell, (0.0, 0.0, 0.0)))


def filament_mutual_inductance(length_m: float, separation_m: float) -> float:
    """Give the exact mutual inductance of two equal parallel filaments.

    The closed form for two parallel line filaments of length ``l`` whose ends
    are aligned, separated by ``d``:

    .. math::

        M = \\frac{\\mu_0 l}{2\\pi}\\left[
            \\ln\\!\\left(\\frac{l}{d} + \\sqrt{1 + \\frac{l^2}{d^2}}\\right)
            - \\sqrt{1 + \\frac{d^2}{l^2}} + \\frac{d}{l}\\right]

    An independent check on the closed form above, in the limit where the bars
    are thin enough that their cross-section stops mattering.
    """
    length = float(length_m)
    separation = float(separation_m)
    if length <= 0.0 or separation <= 0.0:
        raise ValueError("length and separation must be positive")
    ratio = length / separation
    return (VACUUM_PERMEABILITY * length / (2.0 * math.pi)) * (
        math.log(ratio + math.sqrt(1.0 + ratio * ratio))
        - math.sqrt(1.0 + 1.0 / (ratio * ratio))
        + 1.0 / ratio
    )


def bar_self_inductance(length_m: float, width_m: float, thickness_m: float) -> float:
    """Give Grover's self partial inductance of a rectangular bar.

    .. math::

        L = \\frac{\\mu_0 l}{2\\pi}\\left[
            \\ln\\frac{2l}{w + t} + \\frac{1}{2}
            + 0.2235\\,\\frac{w + t}{l}\\right]

    A separate check on the self term, accurate where the bar is long against
    its cross-section.
    """
    length = float(length_m)
    girth = float(width_m) + float(thickness_m)
    if length <= 0.0 or girth <= 0.0:
        raise ValueError("length and cross-section must be positive")
    return (VACUUM_PERMEABILITY * length / (2.0 * math.pi)) * (
        math.log(2.0 * length / girth) + 0.5 + 0.2235 * girth / length
    )


def far_field_mutual_inductance(
    cell: CellGeometry,
    other: CellGeometry,
    offset_m: tuple[float, float, float] | np.ndarray,
) -> float | np.ndarray:
    """Take the centre-to-centre limit of the mutual partial inductance.

    Where the separation is large against the cells, the double volume integral
    tends to the product of the lengths over the centre distance.  Kept as a
    reference for :func:`kernel_accuracy`; the kernel itself uses the closed
    form everywhere, because at this mesh size the closed form is affordable.
    """
    du, dv, dw = (np.asarray(value, dtype=np.float64) for value in offset_m)
    radius = np.sqrt(du * du + dv * dv + dw * dw)
    result = np.divide(
        _MU0_OVER_4PI * cell.length_m * other.length_m,
        radius,
        out=np.zeros_like(radius),
        where=radius > 0.0,
    )
    return float(result) if result.ndim == 0 else result


def kernel_accuracy(
    cell: CellGeometry,
    layer_separation_m: float,
    near_radius: int,
) -> dict[str, float]:
    """Measure how far the centre-to-centre limit is from the exact value.

    Reports the worst relative departure over the ring of cells at
    ``near_radius + 1``.  Nothing in the kernel depends on this any more; it
    exists so that a caller who wants the cheaper form -- on a much larger mesh
    than a PCB, say -- can see what radius it would need.
    """
    if near_radius < 1:
        raise ValueError("near_radius must be at least one cell")
    ring = near_radius + 1
    rows = np.arange(-ring, ring + 1)
    cols = np.arange(-ring, ring + 1)
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
    on_ring = np.maximum(np.abs(row_grid), np.abs(col_grid)) == ring
    offset = (
        col_grid * cell.length_m,
        row_grid * cell.width_m,
        np.full(row_grid.shape, float(layer_separation_m)),
    )
    exact = np.asarray(
        closed_form_mutual_inductance(cell, cell, offset, check_precision=False)
    )
    approximate = np.asarray(far_field_mutual_inductance(cell, cell, offset))
    usable = on_ring & (exact != 0.0)
    if not usable.any():
        return {"near_radius": float(near_radius), "worst_relative_error": 0.0}
    error = np.abs(approximate[usable] - exact[usable]) / np.abs(exact[usable])
    return {
        "near_radius": float(near_radius),
        "worst_relative_error": float(error.max()),
        "mean_relative_error": float(error.mean()),
    }


def build_kernel(
    shape: tuple[int, int],
    cell: CellGeometry,
    layer_separation_m: float,
    *,
    other: CellGeometry | None = None,
    axis: str = "x",
    near_radius_cells: int = NEAR_RADIUS_CELLS,
) -> np.ndarray:
    """Tabulate the partial inductance over every in-plane offset of a grid.

    The result is indexed by signed offset wrapped into ``shape``, the layout a
    circular convolution consumes: entry ``[r, c]`` holds the coupling to the
    cell ``r`` rows and ``c`` columns away, with negative offsets stored at the
    end of each axis.  Because the coupling depends on the offset alone, one
    such table serves every cell of a layer pair.

    ``axis`` says which way the current runs.  For ``"x"`` the cell's length
    lies along the columns; for ``"y"`` it lies along the rows.  The two tables
    are transposes of one another only on a grid whose two extents match and
    whose shape is square, so the axis is built rather than assumed.

    ``other`` is the cell of the second layer, which need not match the first.
    A conductor thick against the skin depth is carried as a stack of filaments
    of graded thickness, so a pair of layers with different thicknesses is the
    normal case rather than the exception.  The two must agree on their in-plane
    extents, since those are what the offsets step by.
    """
    rows, cols = (int(value) for value in shape)
    if rows < 1 or cols < 1:
        raise ValueError("kernel shape must be positive")
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    other = cell if other is None else other
    if (other.length_m, other.width_m) != (cell.length_m, cell.width_m):
        raise ValueError(
            "the two layers must share their in-plane extents; the offsets of "
            "a convolution step by those"
        )

    row_offsets = np.fft.fftfreq(rows, d=1.0 / rows).astype(np.int64)
    col_offsets = np.fft.fftfreq(cols, d=1.0 / cols).astype(np.int64)
    pitch_along_row = cell.length_m if axis == "y" else cell.width_m
    pitch_along_col = cell.width_m if axis == "y" else cell.length_m
    along = (col_offsets[None, :] if axis == "x" else row_offsets[:, None])
    across = (row_offsets[:, None] if axis == "x" else col_offsets[None, :])
    offset = (
        along * (pitch_along_col if axis == "x" else pitch_along_row),
        across * (pitch_along_row if axis == "x" else pitch_along_col),
        np.full((1, 1), float(layer_separation_m)),
    )
    return np.asarray(
        mutual_partial_inductance(
            cell, other, offset, near_radius_cells=near_radius_cells
        ),
        dtype=np.float64,
    )


def vertical_cell(span_m: float, pitch_m: float) -> CellGeometry:
    """Describe a branch running through the board, for the same closed form.

    The closed form takes the current along the cell's ``length``, so a vertical
    branch is a bar whose length is the height it spans and whose two transverse
    extents are the in-plane cell.  Both transverse extents being equal makes the
    resulting kernel symmetric in the two in-plane offsets, which is why there is
    one vertical operator where there are two in-plane ones.
    """
    return CellGeometry(length_m=span_m, width_m=pitch_m, thickness_m=pitch_m)


def build_vertical_kernel(
    shape: tuple[int, int],
    pitch_m: float,
    span_a_m: float,
    span_b_m: float,
    center_separation_m: float,
    *,
    near_radius_cells: int = NEAR_RADIUS_CELLS,
) -> np.ndarray:
    """Tabulate the coupling between two levels of vertical branches.

    ``span_*`` are the heights the two levels span and ``center_separation_m``
    the distance between their mid-planes; both are properties of the level pair
    rather than of a cell, because every vertical branch joining the same two
    layers spans the same height.  The table is indexed by wrapped signed
    in-plane offset, as :func:`build_kernel` is.

    Vertical branches couple to no in-plane branch -- the two are perpendicular
    -- but they are parallel to one another, so this operator is not optional
    once vertical current flows.  Its self term alone would be worse than
    nothing: redistribution currents in neighbouring columns oppose each other,
    and the mutual terms cancel much of the loop inductance the self terms claim.
    """
    rows, cols = (int(value) for value in shape)
    if rows < 1 or cols < 1:
        raise ValueError("kernel shape must be positive")
    pitch = float(pitch_m)
    if pitch <= 0.0:
        raise ValueError("pitch must be positive")

    row_offsets = np.fft.fftfreq(rows, d=1.0 / rows).astype(np.int64)
    col_offsets = np.fft.fftfreq(cols, d=1.0 / cols).astype(np.int64)
    cell = vertical_cell(span_a_m, pitch)
    other = vertical_cell(span_b_m, pitch)
    offset = (
        np.full((1, 1), float(center_separation_m)),   # along the current, z
        col_offsets[None, :] * pitch,
        row_offsets[:, None] * pitch,
    )
    return np.asarray(
        mutual_partial_inductance(
            cell,
            other,
            offset,
            near_radius_cells=near_radius_cells,
            scale_m=pitch,
        ),
        dtype=np.float64,
    )
