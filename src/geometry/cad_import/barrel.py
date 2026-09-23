"""A plated hole as the CAD wrote it, not as a guess about it.

A mechanical export says nothing about vias.  What it gives is a solid: for a
plated hole, KiCad and every tool like it write the plating itself -- a tube
standing in the drill, of the order of 25 um thick, spanning the whole stack.
That tube is the vertical conductor, and everything the electrical problem
wants to know about it is in its shape: the drill it stands in, its outer
diameter, the thickness of the plating, the cross-section that carries the
current, and the layers it reaches.

None of that survives a raster.  At a plane optimizer's pitch the wall is a
quarter of a cell and the bore falls between the samples, so sampling the
tube's material finds a broken ring and an empty axis cell -- the very cell a
vertical connection attaches to.  So the barrel is read from the solid rather
than sampled out of it, and its geometry is carried through to the problem
instead of being reduced to one resistance.

Recognition is deliberately narrow.  A barrel is a vertical prism of circular
section: its footprint is square within tolerance, its cross-sectional area
(volume over height) is no larger than the disc that footprint circumscribes,
and its material really does surround the axis at the radius the area implies.
Anything else -- a fused net, a rectangular pin, a heat sink -- is not a barrel
and is left to the ordinary sampling, because inventing a disc for it would
invent copper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from electrical.matrix_free_mpir_fem.grid import TensorGrid

from .reader import MM, StepSolid

# How many points are probed around the wall, and at how many heights.
WALL_SAMPLES = 24
HEIGHT_SAMPLES = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class Barrel:
    """One plated hole: where it is, how big it is, and what carries current."""

    name: str
    centre_xy_m: tuple[float, float]
    outer_radius_m: float
    drill_radius_m: float
    z_range_m: tuple[float, float]
    wall_area_m2: float

    @property
    def plating_thickness_m(self) -> float:
        return self.outer_radius_m - self.drill_radius_m

    @property
    def height_m(self) -> float:
        return self.z_range_m[1] - self.z_range_m[0]

    @property
    def solid_pin(self) -> bool:
        """True when the section is a disc rather than a ring."""

        return self.drill_radius_m <= 0.0

    def crosses(self, z_m: float, *, tolerance_m: float = 1.0e-9) -> bool:
        return self.z_range_m[0] - tolerance_m <= z_m <= self.z_range_m[1] + tolerance_m

    def footprint_fill(
        self, grid: TensorGrid, *, supersample: int = 3
    ) -> np.ndarray:
        """Fraction of each cell inside the barrel's outer circle.

        The bore is not a gap in the conductor: a plated hole carries current
        over its whole footprint, and on a routing grid that footprint is what
        a cell has to hold for the vertical connection to attach to it.
        """

        offsets = (np.arange(supersample) + 0.5) / supersample
        x = (grid.x_edges_m[:-1, None] + grid.pitch_x_m[:, None] * offsets[None, :]).reshape(-1)
        y = (grid.y_edges_m[:-1, None] + grid.pitch_y_m[:, None] * offsets[None, :]).reshape(-1)
        grid_x, grid_y = np.meshgrid(x, y)
        inside = (grid_x - self.centre_xy_m[0]) ** 2 + (
            grid_y - self.centre_xy_m[1]
        ) ** 2 <= self.outer_radius_m**2
        rows, cols = grid.shape
        return inside.reshape(rows, supersample, cols, supersample).mean(axis=(1, 3))

    def as_dict(self) -> dict[str, Any]:
        """The barrel in millimetres, for a problem record."""

        return {
            "name": self.name,
            "centre_mm": [self.centre_xy_m[0] / MM, self.centre_xy_m[1] / MM],
            "drill_diameter_mm": 2.0 * self.drill_radius_m / MM,
            "outer_diameter_mm": 2.0 * self.outer_radius_m / MM,
            "plating_thickness_mm": self.plating_thickness_m / MM,
            "wall_area_mm2": self.wall_area_m2 / MM**2,
            "z_range_mm": [self.z_range_m[0] / MM, self.z_range_m[1] / MM],
            "height_mm": self.height_m / MM,
            "solid_pin": self.solid_pin,
        }


def barrel_of(
    solid: StepSolid,
    *,
    roundness_tolerance: float = 5.0e-3,
    area_tolerance: float = 5.0e-3,
    bore_tolerance: float = 1.0e-4,
    method: str = "auto",
) -> Barrel | None:
    """The barrel this solid is, or None when it is not one.

    ``roundness_tolerance`` is how far the footprint may be from square,
    ``area_tolerance`` how far the section may exceed the circumscribed disc,
    and ``bore_tolerance`` how small a bore is rounding rather than a hole,
    all relative to the outer radius.  The wall probe is the test that matters: it is what tells
    a tube from a solid that merely happens to sit in a square box.
    """

    lo, hi = solid.bounds_m
    height = hi[2] - lo[2]
    if height <= 0.0:
        return None
    dx, dy = hi[0] - lo[0], hi[1] - lo[1]
    if dx <= 0.0 or dy <= 0.0:
        return None
    if abs(dx - dy) > roundness_tolerance * max(dx, dy):
        return None
    outer = max(dx, dy) / 2.0
    area = solid.volume_m3 / height
    disc = math.pi * outer**2
    if area <= 0.0 or area > disc * (1.0 + area_tolerance):
        return None
    inner_squared = outer**2 - area / math.pi
    inner = math.sqrt(inner_squared) if inner_squared > 0.0 else 0.0
    # A solid pin's area is the whole disc, to within the arithmetic: a bore
    # far below the outer radius is rounding, not a hole.
    if inner < bore_tolerance * outer:
        inner = 0.0
    centre = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0)
    probe = (outer + inner) / 2.0
    angles = np.arange(WALL_SAMPLES) * (2.0 * math.pi / WALL_SAMPLES)
    ring = np.column_stack(
        (centre[0] + probe * np.cos(angles), centre[1] + probe * np.sin(angles))
    )
    points = np.vstack(
        [
            np.column_stack((ring, np.full(len(ring), lo[2] + fraction * height)))
            for fraction in HEIGHT_SAMPLES
        ]
    )
    if not bool(np.all(solid.contains(points, method=method))):
        return None
    return Barrel(
        name=solid.name,
        centre_xy_m=centre,
        outer_radius_m=outer,
        drill_radius_m=inner,
        z_range_m=(lo[2], hi[2]),
        wall_area_m2=area,
    )


def barrels_of(solids: Sequence[StepSolid], **kwargs: Any) -> tuple[Barrel, ...]:
    """Every solid of ``solids`` that is a barrel, in order."""

    found = (barrel_of(solid, **kwargs) for solid in solids)
    return tuple(barrel for barrel in found if barrel is not None)


__all__ = ["Barrel", "barrel_of", "barrels_of"]
