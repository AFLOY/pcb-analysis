"""Screen a conductor layer's thickness against the skin depth.

The 2.5D sheet models hand each layer to the sheet PEEC, which cuts the
thickness into graded filaments (:mod:`.skin_filaments`).  Below one skin
depth the current is uniform and a single filament is enough; up to
``sheet_limit_ratio`` skin depths the filaments resolve the profile; beyond
that, or above ``sheet_limit_mm`` outright, the conductor is a 3D body and
belongs to the voxel PEEC (:mod:`electrical.voxel_peec`).  The classification is
physics, so it lives here; the geometry front end only supplies thicknesses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .skin_filaments import COPPER_RESISTIVITY_OHM_M, skin_depth_m

SkinClass = Literal["uniform", "filaments", "3d"]


@dataclass(frozen=True)
class LayerSkin:
    layer: str
    thickness_mm: float
    skin_depth_mm: float
    thickness_over_skin_depth: float
    classification: SkinClass


def classify_layer_thickness(
    layer: str,
    thickness_mm: float,
    frequency_hz: float,
    *,
    resistivity_ohm_m: float = COPPER_RESISTIVITY_OHM_M,
    uniform_ratio: float = 1.0,
    sheet_limit_ratio: float = 20.0,
    sheet_limit_mm: float = 1.0,
) -> LayerSkin:
    """Classify one layer by thickness over skin depth at ``frequency_hz``."""

    if frequency_hz < 0.0:
        raise ValueError("frequency_hz must be non-negative")
    if thickness_mm <= 0.0:
        raise ValueError("thickness_mm must be positive")
    depth_mm = skin_depth_m(frequency_hz, resistivity_ohm_m) * 1.0e3
    ratio = thickness_mm / depth_mm if depth_mm != float("inf") else 0.0
    if thickness_mm > sheet_limit_mm or ratio > sheet_limit_ratio:
        classification: SkinClass = "3d"
    elif ratio > uniform_ratio:
        classification = "filaments"
    else:
        classification = "uniform"
    return LayerSkin(layer, float(thickness_mm), depth_mm, ratio, classification)


__all__ = ["LayerSkin", "SkinClass", "classify_layer_thickness"]
