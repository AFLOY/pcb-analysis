"""Divide a conductor through its thickness so that skin effect is resolved.

The sheet formulation treats a copper layer as one filament and assumes the
current is uniform through it.  That holds for foil: 35um of copper against a
121um skin depth at 300kHz is uniform to within a few per cent.

A copper inlay is a different conductor.  At 3.5mm it is uniform through its
thickness only below about 89Hz; at 300kHz its thickness is 29 skin depths, and
the current occupies a few per cent of it at each face.  Treating it as one
filament would understate its resistance by more than an order of magnitude and
put the current where it is not.

The remedy is the standard one for PEEC: cut the conductor into filaments
thinner than the skin depth and let the solve decide how the current divides
between them.  Nothing else changes -- each filament is a layer of the existing
stackup, and the proximity effect between filaments falls out of the mutual
inductance that is already there rather than being modelled separately.

This is where placing conductors at stated heights pays against a uniform voxel
grid.  A voxel mesh has one ``dz`` for the whole board, so resolving a 60um
filament in the inlay forces every millimetre of the board's height to be cut
at 60um.  Filaments are placed only in the copper, graded so they are thin at
the faces where the current is and coarse in the middle where it is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .sheet_operator import COPPER_RESISTIVITY_OHM_M, SheetLayer

VACUUM_PERMEABILITY = 4.0e-7 * math.pi


def skin_depth_m(
    frequency_hz: float, resistivity_ohm_m: float = COPPER_RESISTIVITY_OHM_M
) -> float:
    """Give the depth at which the current has fallen to ``1/e``, in metres."""
    frequency_hz = float(frequency_hz)
    if frequency_hz <= 0.0:
        return math.inf
    return math.sqrt(
        resistivity_ohm_m / (math.pi * frequency_hz * VACUUM_PERMEABILITY)
    )


def uniform_through_thickness(
    thickness_m: float,
    frequency_hz: float,
    resistivity_ohm_m: float = COPPER_RESISTIVITY_OHM_M,
    *,
    ratio: float = 0.5,
) -> bool:
    """Say whether one filament is enough for this conductor at this frequency.

    ``ratio`` is how much of a skin depth the conductor may span before it stops
    being uniform.  A half is the usual working limit; at that point the error
    in the resistance is a few per cent.
    """
    depth = skin_depth_m(frequency_hz, resistivity_ohm_m)
    return not math.isfinite(depth) or float(thickness_m) <= ratio * depth


@dataclass(frozen=True)
class FilamentStack:
    """The filaments one conductor was cut into, and where they sit."""

    thicknesses_m: tuple[float, ...]
    centers_m: tuple[float, ...]
    surface_thickness_m: float
    skin_depth_m: float

    def __len__(self) -> int:
        return len(self.thicknesses_m)

    @property
    def total_thickness_m(self) -> float:
        return float(sum(self.thicknesses_m))

    def layers(
        self,
        name: str,
        resistivity_ohm_m: float = COPPER_RESISTIVITY_OHM_M,
    ) -> tuple[SheetLayer, ...]:
        """Present the filaments as stackup layers, front to back."""
        return tuple(
            SheetLayer(
                name=f"{name}#{index}",
                z_m=center,
                thickness_m=thickness,
                resistivity_ohm_m=resistivity_ohm_m,
            )
            for index, (center, thickness) in enumerate(
                zip(self.centers_m, self.thicknesses_m)
            )
        )


def graded_filaments(
    thickness_m: float,
    z_center_m: float,
    frequency_hz: float,
    *,
    resistivity_ohm_m: float = COPPER_RESISTIVITY_OHM_M,
    cells_per_skin_depth: float = 4.0,
    growth: float = 2.0,
    maximum_filaments: int = 64,
) -> FilamentStack:
    """Cut a conductor into filaments, fine at the faces and coarse inside.

    The current lives within a skin depth of each face, so that is where the
    resolution is spent.  Filament thickness starts at
    ``skin depth / cells_per_skin_depth`` at each face and grows by ``growth``
    toward the middle, which is the standard grading for this problem: a
    uniform cut fine enough for the faces would spend most of its unknowns on
    an interior that carries almost nothing.

    Four filaments per skin depth is the default because two was measured to be
    too few.  Against a field solve of the same bar's cross-section, the loss
    the mesh can hold given the exact current profile -- its ceiling, whatever
    the solver does -- came out as:

    ======================  ==========  ==========
    filaments / skin depth  pitch 200um pitch 100um
    ======================  ==========  ==========
    2                       77%         88%
    4                       84%         95%
    8                       90%         --
    ======================  ==========  ==========

    :func:`discretisation_note` reports the same trade-off for a given cut.

    A conductor already thin against the skin depth comes back as one filament,
    so this can be applied to every layer of a board without special-casing the
    foil.
    """
    thickness = float(thickness_m)
    if thickness <= 0.0:
        raise ValueError("thickness must be positive")
    if cells_per_skin_depth <= 0.0:
        raise ValueError("cells_per_skin_depth must be positive")
    if growth < 1.0:
        raise ValueError("growth must be at least one")
    if maximum_filaments < 1:
        raise ValueError("maximum_filaments must be positive")

    depth = skin_depth_m(frequency_hz, resistivity_ohm_m)
    if uniform_through_thickness(thickness, frequency_hz, resistivity_ohm_m):
        return FilamentStack(
            thicknesses_m=(thickness,),
            centers_m=(float(z_center_m),),
            surface_thickness_m=thickness,
            skin_depth_m=depth,
        )

    surface = depth / cells_per_skin_depth
    # Grow inward from both faces at once, so the cut stays symmetric however
    # many filaments the thickness turns out to take.
    lower: list[float] = []
    upper: list[float] = []
    remaining = thickness
    step = surface
    # Each turn adds one filament at each face, and one more may be left over
    # for the middle, so the budget is checked against all three.
    while remaining > 0.0 and len(lower) + len(upper) + 3 <= maximum_filaments:
        if 2.0 * step >= remaining:
            break
        lower.append(step)
        upper.append(step)
        remaining -= 2.0 * step
        step *= growth
    thicknesses = lower + ([remaining] if remaining > 0.0 else []) + upper[::-1]

    # Place each filament's mid-plane, measuring from the conductor's top face
    # downward, so the stack reads front to back like the rest of the tree.
    top = float(z_center_m) + thickness / 2.0
    centers: list[float] = []
    offset = 0.0
    for value in thicknesses:
        centers.append(top - offset - value / 2.0)
        offset += value
    return FilamentStack(
        thicknesses_m=tuple(thicknesses),
        centers_m=tuple(centers),
        surface_thickness_m=surface,
        skin_depth_m=depth,
    )


def slab_surface_impedance(
    thickness_m: float,
    frequency_hz: float,
    resistivity_ohm_m: float = COPPER_RESISTIVITY_OHM_M,
) -> complex:
    """Give the internal impedance of one square of an infinite slab, in ohms.

    For a slab of thickness ``t`` carrying current along its plane with equal
    and opposite tangential field at its two faces,

    .. math::

        Z = \\frac{\\rho k}{2} \\coth\\!\\left(\\frac{k t}{2}\\right),
        \\qquad k = \\frac{1 + j}{\\delta}

    which tends to ``rho / t`` as the frequency goes to zero -- the sheet
    resistance -- and to ``(1 + j) rho / (2 delta)`` when the slab is many skin
    depths thick, the two faces each conducting a skin depth deep.

    This is the reference the filament cut is checked against.  It describes a
    slab of unbounded extent, so a finite conductor departs from it near its
    edges; the check uses a conductor wide enough that the middle is slab-like.
    """
    thickness = float(thickness_m)
    if thickness <= 0.0:
        raise ValueError("thickness must be positive")
    if frequency_hz <= 0.0:
        return complex(resistivity_ohm_m / thickness, 0.0)
    depth = skin_depth_m(frequency_hz, resistivity_ohm_m)
    wavenumber = complex(1.0, 1.0) / depth
    argument = wavenumber * thickness / 2.0
    return resistivity_ohm_m * wavenumber / 2.0 / np.tanh(argument)


def resistance_ratio(
    thickness_m: float,
    frequency_hz: float,
    resistivity_ohm_m: float = COPPER_RESISTIVITY_OHM_M,
) -> float:
    """Give the slab's AC resistance as a multiple of its DC resistance."""
    impedance = slab_surface_impedance(thickness_m, frequency_hz, resistivity_ohm_m)
    return float(impedance.real / (resistivity_ohm_m / float(thickness_m)))


def filament_links(
    stack: FilamentStack,
    shape: tuple[int, int],
    pitch_m: float,
    *,
    first_layer: int = 0,
    resistivity_ohm_m: float = COPPER_RESISTIVITY_OHM_M,
    occupancy: "np.ndarray | None" = None,
) -> tuple["ViaBranch", ...]:
    """Join the filaments of one conductor, because they are one piece of copper.

    Cutting a conductor into filaments does not cut the conductor.  Left
    unjoined, each filament is an isolated sheet from end to end and the current
    put into it has nowhere else to go: the solve then returns whatever division
    the terminals imposed, which is uniform, which is the very thing the cut was
    made to avoid.  The measured symptom is unmistakable -- the filament
    currents come back in exact proportion to their thicknesses, and the
    resistance ratio stays near one where it should be fifteen.

    So every filament is joined to the one below it at every cell, with the
    resistance of the copper between their mid-planes.  These are ordinary
    vertical branches: a vertical conductor couples to no in-plane branch, so
    they add nothing to the transform.
    """
    from .sheet_peec import ViaBranch  # imported here to avoid a cycle

    pitch = float(pitch_m)
    if pitch <= 0.0:
        raise ValueError("pitch must be positive")
    rows, cols = (int(value) for value in shape)
    area = pitch * pitch
    links: list[ViaBranch] = []
    for index in range(len(stack) - 1):
        upper = first_layer + index
        lower = upper + 1
        span = (stack.thicknesses_m[index] + stack.thicknesses_m[index + 1]) / 2.0
        resistance = resistivity_ohm_m * span / area
        for row in range(rows):
            for col in range(cols):
                if occupancy is not None and not (
                    occupancy[upper, row, col] and occupancy[lower, row, col]
                ):
                    continue
                links.append(
                    ViaBranch(
                        row=row,
                        col=col,
                        lower_layer=lower,
                        upper_layer=upper,
                        resistance_ohm=resistance,
                    )
                )
    return tuple(links)


# Measured against a finite-difference field solve of the same bar's
# cross-section: the loss the sheet mesh can hold given the exact current
# profile, as a fraction of the field solve's own.  This is a ceiling, reached
# only by a converged solve, and it does not depend on the solver at all.  The
# bar was 1.6 x 3.5 mm of copper at 300 kHz, skin depth 120.7 um, against a
# field solve at 25 um.
#
# The two entries that came out above one are not the mesh outperforming the
# reference.  There the sheet cells are finer than the field solve's own 25 um,
# so the comparison has saturated: the projection is no longer what limits the
# answer, and the excess is the field solve's own remaining discretisation error
# plus the misalignment of filament edges against its grid.  They are recorded
# as measured and clamped at one where reported.
#
#   (in-plane pitch / skin depth, filaments per skin depth) -> fraction
_MEASURED_CEILING = {
    (1.66, 2.0): 0.7715,
    (1.66, 4.0): 0.8396,
    (1.66, 8.0): 0.8993,
    (0.83, 2.0): 0.8792,
    (0.83, 4.0): 0.9479,
    (0.83, 8.0): 1.0086,
    (0.41, 2.0): 0.9154,
    (0.41, 4.0): 0.9843,
    (0.41, 8.0): 1.0452,
}


def discretisation_note(
    pitch_m: float,
    thickness_m: float,
    frequency_hz: float,
    *,
    resistivity_ohm_m: float = COPPER_RESISTIVITY_OHM_M,
    cells_per_skin_depth: float = 4.0,
) -> dict[str, Any]:
    """Say how much of a thick conductor's AC loss this mesh can hold.

    Resolving skin effect costs resolution in two directions and they cost about
    the same.  Through the thickness, :func:`graded_filaments` spends filaments;
    in plane, nothing here can help -- the current crowds within a skin depth of
    the conductor's side walls, and if the grid is coarser than that, the
    crowding is smeared over a cell whatever the filaments do.

    The returned ``ceiling_fraction`` is interpolated from measurements against
    an independent field solve, so it is an estimate for one bar at one
    frequency rather than a bound.  It is reported so that a caller choosing a
    grid can see what the choice costs, instead of discovering it later as an
    unexplained shortfall.
    """
    depth = skin_depth_m(frequency_hz, resistivity_ohm_m)
    if not math.isfinite(depth):
        return {
            "skin_depth_m": depth,
            "pitch_per_skin_depth": 0.0,
            "filaments": 1,
            "ceiling_fraction": 1.0,
            "note": "direct current: no skin effect to resolve",
        }
    stack = graded_filaments(
        thickness_m,
        0.0,
        frequency_hz,
        resistivity_ohm_m=resistivity_ohm_m,
        cells_per_skin_depth=cells_per_skin_depth,
    )
    ratio = float(pitch_m) / depth
    if len(stack) == 1:
        fraction = 1.0
        note = "one filament suffices: the conductor is thin against the skin depth"
    else:
        # Nearest measured point in the two-dimensional table, in log spacing.
        best = min(
            _MEASURED_CEILING,
            key=lambda key: (
                math.log(key[0] / ratio) ** 2
                + math.log(key[1] / cells_per_skin_depth) ** 2
            ),
        )
        fraction = min(1.0, _MEASURED_CEILING[best])
        note = (
            f"nearest measurement: pitch {best[0]:.2f} skin depths, "
            f"{best[1]:.0f} filaments per skin depth"
        )
    return {
        "skin_depth_m": depth,
        "pitch_per_skin_depth": ratio,
        "filaments": len(stack),
        "surface_filament_m": stack.surface_thickness_m,
        "resistance_ratio": resistance_ratio(
            thickness_m, frequency_hz, resistivity_ohm_m
        ),
        "ceiling_fraction": fraction,
        "note": note,
    }
