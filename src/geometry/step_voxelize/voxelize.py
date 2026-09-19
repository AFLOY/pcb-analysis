"""3D bodies sampled onto a Cartesian voxel grid (heat sinks, enclosures)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from thermal.matrix_free_mpir_fem import VoxelMaterial, VoxelSolidModel

from .bodymap import BodySpec, ResolvedBodies
from .reader import StepSolid


def voxel_grid(
    lo_m: np.ndarray,
    hi_m: np.ndarray,
    pitch_m: tuple[float, float, float],
    *,
    margin_voxels: int = 0,
    origin_m: tuple[float, float, float] | None = None,
) -> tuple[tuple[float, float, float], tuple[int, int, int]]:
    """Origin and ``(nz, ny, nx)`` covering ``[lo, hi]`` with whole voxels."""

    pitch = np.asarray(pitch_m, dtype=np.float64)
    if origin_m is None:
        origin = np.asarray(lo_m, dtype=np.float64) - margin_voxels * pitch
    else:
        origin = np.asarray(origin_m, dtype=np.float64)
    count = np.ceil((np.asarray(hi_m) - origin) / pitch - 1.0e-6).astype(int) + margin_voxels
    if np.any(count < 1):
        raise ValueError("the voxel grid does not cover the bodies")
    nx, ny, nz = (int(value) for value in count)
    return (float(origin[0]), float(origin[1]), float(origin[2])), (nz, ny, nx)


def sample_volume_fill(
    solids: Sequence[StepSolid],
    *,
    origin_m: tuple[float, float, float],
    pitch_m: tuple[float, float, float],
    shape: tuple[int, int, int],
    supersample: int = 2,
) -> np.ndarray:
    """Fraction of every voxel ``(k, j, i)`` inside the solids."""

    if supersample < 1:
        raise ValueError("supersample must be at least one")
    nz, ny, nx = shape
    hx, hy, hz = pitch_m
    offsets = (np.arange(supersample) + 0.5) / supersample
    x = origin_m[0] + hx * (np.arange(nx)[:, None] + offsets[None, :]).reshape(-1)
    y = origin_m[1] + hy * (np.arange(ny)[:, None] + offsets[None, :]).reshape(-1)
    z = origin_m[2] + hz * (np.arange(nz)[:, None] + offsets[None, :]).reshape(-1)
    grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing="ij")
    points = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)))
    inside = np.zeros(points.shape[0], dtype=bool)
    for solid in solids:
        inside |= solid.contains(points)
    fine = inside.reshape(nz, supersample, ny, supersample, nx, supersample)
    return fine.mean(axis=(1, 3, 5))


def voxelize_bodies(
    resolved: ResolvedBodies,
    *,
    pitch_m: tuple[float, float, float],
    supersample: int = 2,
    margin_voxels: int = 0,
    origin_m: tuple[float, float, float] | None = None,
    bodies: Sequence[BodySpec] | None = None,
) -> tuple[VoxelSolidModel, dict[int, BodySpec]]:
    """Voxelise the map's bodies; returns the model and ``material id → spec``.

    Each body gets its own material id in map order.  Where bodies overlap
    in a voxel the one with the larger fill wins, ties going to the earlier
    entry, so an interface material listed before the parts it joins keeps
    its voxels.  ``fill`` is the total solid fraction of the voxel.
    """

    chosen = [(spec, solids) for spec, solids in resolved.bodies if bodies is None or spec in bodies]
    if not chosen:
        raise ValueError("no body to voxelise")
    all_solids = [solid for _, solids in chosen for solid in solids]
    lo = np.min([solid.bounds_m[0] for solid in all_solids], axis=0)
    hi = np.max([solid.bounds_m[1] for solid in all_solids], axis=0)
    origin, shape = voxel_grid(lo, hi, pitch_m, margin_voxels=margin_voxels, origin_m=origin_m)
    material_id = np.zeros(shape, dtype=np.uint16)
    best = np.zeros(shape)
    total = np.zeros(shape)
    materials: dict[int, VoxelMaterial] = {}
    specs: dict[int, BodySpec] = {}
    for index, (spec, solids) in enumerate(chosen, start=1):
        fill = sample_volume_fill(solids, origin_m=origin, pitch_m=pitch_m, shape=shape, supersample=supersample)
        wins = fill > best
        material_id[wins] = index
        best = np.where(wins, fill, best)
        total += fill
        materials[index] = spec.material
        specs[index] = spec
    model = VoxelSolidModel(
        material_id=material_id,
        materials=materials,
        pitch_m=pitch_m,
        origin_m=origin,
        fill=np.clip(total, 0.0, 1.0),
    )
    return model, specs


__all__ = ["sample_volume_fill", "voxel_grid", "voxelize_bodies"]
