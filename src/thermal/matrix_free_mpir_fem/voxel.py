"""Voxelised 3D bodies (heat sinks, enclosures) on the hexahedral Q1 mesh.

A ``VoxelSolidModel`` is what the geometry front end produces from CAD: one
material id per voxel on a Cartesian grid, ``0`` for void, and optionally the
fraction of each voxel the body fills.  ``VoxelThermalMesh`` turns it into the
layered mesh of :mod:`.mesh` with an active-element mask, so the same
operator, preconditioner, CUDA kernel and C++ path solve it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .mesh import FaceDirection, LayeredThermalMesh, exposed_element_faces


@dataclass(frozen=True)
class VoxelMaterial:
    """Isotropic thermal conductivity of one body material."""

    name: str
    conductivity_w_per_m_k: float
    role: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("material name must not be empty")
        value = float(self.conductivity_w_per_m_k)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{self.name}: conductivity must be finite and positive")
        object.__setattr__(self, "conductivity_w_per_m_k", value)


@dataclass(frozen=True)
class VoxelSolidModel:
    """Material ids on a Cartesian voxel grid, ``0`` meaning void.

    ``material_id`` has shape ``(nz, ny, nx)`` in the same ``(slab, row, col)``
    order as the thermal mesh.  ``pitch_m`` is ``(hx, hy, hz)`` and ``origin_m``
    the position of the corner node ``(0, 0, 0)``.  ``fill`` is the occupied
    fraction of each voxel; a body sampled at coarse pitch keeps its thin
    features through a reduced conductivity rather than by vanishing.
    """

    material_id: np.ndarray
    materials: Mapping[int, VoxelMaterial]
    pitch_m: tuple[float, float, float]
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fill: np.ndarray | None = None

    def __post_init__(self) -> None:
        ids = np.asarray(self.material_id)
        if ids.ndim != 3:
            raise ValueError("material_id must have shape (nz, ny, nx)")
        if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
            raise ValueError("material_id must hold non-negative integers")
        ids = ids.astype(np.uint16)
        used = set(np.unique(ids).tolist()) - {0}
        materials = {int(key): value for key, value in self.materials.items()}
        if 0 in materials:
            raise ValueError("material id 0 is reserved for void")
        missing = used - set(materials)
        if missing:
            raise ValueError(f"material ids without a material: {sorted(missing)}")
        if not used:
            raise ValueError("the model has no solid voxel")
        pitch = tuple(float(value) for value in self.pitch_m)
        if len(pitch) != 3 or any(not np.isfinite(p) or p <= 0.0 for p in pitch):
            raise ValueError("pitch_m must be three positive lengths (hx, hy, hz)")
        origin = tuple(float(value) for value in self.origin_m)
        if len(origin) != 3 or any(not np.isfinite(o) for o in origin):
            raise ValueError("origin_m must be three finite coordinates")
        if self.fill is None:
            fill = (ids > 0).astype(np.float64)
        else:
            fill = np.asarray(self.fill, dtype=np.float64)
            if fill.shape != ids.shape:
                raise ValueError("fill must match material_id")
            if not np.all(np.isfinite(fill)) or np.any((fill < 0.0) | (fill > 1.0)):
                raise ValueError("fill must lie in [0, 1]")
            fill = np.where(ids > 0, fill, 0.0)
        object.__setattr__(self, "material_id", ids)
        object.__setattr__(self, "materials", materials)
        object.__setattr__(self, "pitch_m", pitch)
        object.__setattr__(self, "origin_m", origin)
        object.__setattr__(self, "fill", fill)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(axis) for axis in self.material_id.shape)  # type: ignore[return-value]

    @property
    def solid(self) -> np.ndarray:
        return self.material_id > 0

    def conductivity_w_per_m_k(self) -> np.ndarray:
        """Nominal (unscaled) conductivity per voxel, zero in void."""

        table = np.zeros(max(self.materials) + 1, dtype=np.float64)
        for key, material in self.materials.items():
            table[key] = material.conductivity_w_per_m_k
        return table[self.material_id]

    def voxel_centres_m(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Centre coordinates ``(z, y, x)`` as 1D arrays along each axis."""

        nz, ny, nx = self.shape
        hx, hy, hz = self.pitch_m
        ox, oy, oz = self.origin_m
        return (
            oz + hz * (np.arange(nz) + 0.5),
            oy + hy * (np.arange(ny) + 0.5),
            ox + hx * (np.arange(nx) + 0.5),
        )

    def node_coordinates_m(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Node coordinates ``(z, y, x)`` as 1D arrays along each axis."""

        nz, ny, nx = self.shape
        hx, hy, hz = self.pitch_m
        ox, oy, oz = self.origin_m
        return (
            oz + hz * np.arange(nz + 1),
            oy + hy * np.arange(ny + 1),
            ox + hx * np.arange(nx + 1),
        )

    def exposed_faces(self) -> dict[FaceDirection, np.ndarray]:
        return exposed_element_faces(self.solid)

    def volume_m3(self) -> float:
        hx, hy, hz = self.pitch_m
        return float(np.sum(self.fill) * hx * hy * hz)


@dataclass(frozen=True)
class VoxelThermalMesh(LayeredThermalMesh):
    """The layered Q1 mesh of a voxelised body.

    Every voxel is one element; voxels of material ``0`` are inactive.  The
    conductivity of a partially filled voxel is scaled by its fill, and voxels
    below ``min_fill`` are dropped so that a sliver does not become a stiff
    but almost massless element.
    """

    @classmethod
    def from_solid_model(
        cls,
        model: VoxelSolidModel,
        *,
        min_fill: float = 0.05,
        scale_by_fill: bool = True,
    ) -> "VoxelThermalMesh":
        if not 0.0 < min_fill <= 1.0:
            raise ValueError("min_fill must lie in (0, 1]")
        active = model.solid & (model.fill >= min_fill)
        if not np.any(active):
            raise ValueError("no voxel survives the fill threshold")
        conductivity = model.conductivity_w_per_m_k()
        if scale_by_fill:
            conductivity = conductivity * model.fill
        conductivity = np.where(active, conductivity, 1.0)
        hx, hy, hz = model.pitch_m
        return cls(
            slab_thickness_m=(hz,) * model.shape[0],
            pitch_x_m=hx,
            pitch_y_m=hy,
            conductivity_w_per_m_k=conductivity,
            active=active,
        )
