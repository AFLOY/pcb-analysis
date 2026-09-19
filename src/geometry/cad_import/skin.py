"""Skin-effect screening of a board raster's layers.

The classification itself is :func:`electrical.sheet_peec.skin_screen.classify_layer_thickness`;
this module only feeds it the layers' thicknesses (stackup or measured) and
collects the report the plane-opt mapping warns from.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from electrical.sheet_peec.skin_screen import LayerSkin, SkinClass, classify_layer_thickness

from .section import BoardRaster


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
    """Classify every layer of the raster (see ``classify_layer_thickness``)."""

    items = tuple(
        classify_layer_thickness(
            layer.name,
            raster.layer_thickness_mm(index, source=thickness_source),
            frequency_hz,
            resistivity_ohm_m=layer.resistivity_ohm_m,
            uniform_ratio=uniform_ratio,
            sheet_limit_ratio=sheet_limit_ratio,
            sheet_limit_mm=sheet_limit_mm,
        )
        for index, layer in enumerate(raster.layers)
    )
    return SkinReport(float(frequency_hz), items)


def warn_if_not_sheet(report: SkinReport) -> None:
    for item in report.layers:
        if item.classification == "3d":
            warnings.warn(
                f"layer {item.layer}: {item.thickness_mm:.3f} mm of copper is {item.thickness_over_skin_depth:.1f} skin depths "
                f"at {report.frequency_hz:g} Hz; a 2.5D sheet cannot represent it, voxelise it as a 3D body instead",
                stacklevel=3,
            )


__all__ = ["LayerSkin", "SkinClass", "SkinReport", "skin_report", "warn_if_not_sheet"]
