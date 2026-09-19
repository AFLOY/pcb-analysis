"""Skin-effect screening of the copper layers before a 2.5D solve.

The 2.5D sheet models take one conductor layer per sheet and let the sheet
PEEC cut its thickness into graded filaments, so the geometry's job is to
hand over the right thickness and to say when a layer is too thick for a
sheet at all.  ``skin_report`` compares each layer's thickness with the skin
depth at the analysis frequency: below ``filament_ratio`` skin depths the
solver's filaments resolve it; above ``sheet_limit_ratio`` (or above
``sheet_limit_mm`` outright) the conductor is a 3D body and belongs on the
voxel path with a 3D PEEC solve, not in the stackup.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

from electrical.dice_peec.skin_filaments import skin_depth_m

from .section import BoardRaster

SkinClass = Literal["uniform", "filaments", "3d"]


@dataclass(frozen=True)
class LayerSkin:
    layer: str
    thickness_mm: float
    skin_depth_mm: float
    thickness_over_skin_depth: float
    classification: SkinClass


@dataclass(frozen=True)
class SkinReport:
    frequency_hz: float
    layers: tuple[LayerSkin, ...]

    @property
    def needs_3d(self) -> tuple[str, ...]:
        return tuple(item.layer for item in self.layers if item.classification == "3d")

    @property
    def needs_filaments(self) -> tuple[str, ...]:
        return tuple(item.layer for item in self.layers if item.classification == "filaments")


def skin_report(
    raster: BoardRaster,
    frequency_hz: float,
    *,
    thickness_source: str = "stackup",
    uniform_ratio: float = 1.0,
    sheet_limit_ratio: float = 20.0,
    sheet_limit_mm: float = 1.0,
) -> SkinReport:
    """Classify every layer by thickness over skin depth at ``frequency_hz``.

    ``uniform``: thinner than one skin depth, the current is uniform to a few
    per cent and a single filament is enough.  ``filaments``: the sheet PEEC's graded
    filaments resolve the profile.  ``3d``: thicker than ``sheet_limit_ratio``
    skin depths or than ``sheet_limit_mm``; a sheet cannot represent it.
    """

    if frequency_hz < 0.0:
        raise ValueError("frequency_hz must be non-negative")
    items = []
    for index, layer in enumerate(raster.layers):
        thickness_mm = raster.layer_thickness_mm(index, source=thickness_source)
        depth_mm = skin_depth_m(frequency_hz, layer.resistivity_ohm_m) * 1.0e3
        ratio = thickness_mm / depth_mm if depth_mm != float("inf") else 0.0
        if thickness_mm > sheet_limit_mm or ratio > sheet_limit_ratio:
            classification: SkinClass = "3d"
        elif ratio > uniform_ratio:
            classification = "filaments"
        else:
            classification = "uniform"
        items.append(LayerSkin(layer.name, thickness_mm, depth_mm, ratio, classification))
    return SkinReport(float(frequency_hz), tuple(items))


def warn_if_not_sheet(report: SkinReport) -> None:
    for item in report.layers:
        if item.classification == "3d":
            warnings.warn(
                f"layer {item.layer}: {item.thickness_mm:.3f} mm of copper is {item.thickness_over_skin_depth:.1f} skin depths "
                f"at {report.frequency_hz:g} Hz; a 2.5D sheet cannot represent it, voxelise it as a 3D body instead",
                stacklevel=3,
            )


__all__ = ["LayerSkin", "SkinClass", "SkinReport", "skin_report", "warn_if_not_sheet"]
