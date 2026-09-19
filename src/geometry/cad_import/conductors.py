"""Thick conductors from CAD to the 3D voxel PEEC problem.

A layer that ``skin_report`` classifies as ``3d``, a busbar or a terminal
block is not a sheet.  Its solids are voxelised on their own grid, the port
regions are named by solids (or boxes) of the same model, and the result is
a ``VoxelConductorProblem`` for ``electrical.voxel_peec.solve_voxel_peec``.
The Joule loss of that solve maps back onto a ``VoxelThermalMesh`` of the
same grid as ``element_heat_w``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from electrical.voxel_peec import VoxelConductorProblem, VoxelPeecSolution, VoxelTerminal
from electrical.sheet_peec.skin_filaments import skin_depth_m
from thermal.matrix_free_mpir_fem import VoxelMaterial, VoxelSolidModel

from .reader import StepSolid
from .voxelize import sample_volume_fill, voxel_grid


@dataclass(frozen=True)
class TerminalRegion:
    """A port: the conductor voxels inside these solids (or this box) form the terminal."""

    name: str
    solids: tuple[StepSolid, ...] = ()
    box_m: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    current_a: complex | None = None

    def __post_init__(self) -> None:
        if not self.solids and self.box_m is None:
            raise ValueError(f"terminal {self.name!r} needs solids or a box")


def conductor_problem_from_solids(
    solids: Sequence[StepSolid],
    terminals: Sequence[TerminalRegion],
    *,
    pitch_m: tuple[float, float, float],
    resistivity_ohm_m: float = 1.68e-8,
    frequency_hz: float = 0.0,
    supersample: int = 2,
    min_fill: float = 0.5,
    margin_voxels: int = 0,
    name: str = "conductor",
) -> tuple[VoxelConductorProblem, VoxelSolidModel]:
    """Voxelise conductor solids and mark the terminal voxels.

    A voxel is conductor when at least ``min_fill`` of it lies inside the
    solids (PyPEEC voxels are full or empty).  Terminal voxels are the
    conductor voxels whose centres lie inside the terminal's solids or box.
    Returns the PEEC problem and the matching ``VoxelSolidModel`` so the same
    grid can carry the thermal solve.
    """

    if not solids:
        raise ValueError("at least one conductor solid is required")
    lo = np.min([solid.bounds_m[0] for solid in solids], axis=0)
    hi = np.max([solid.bounds_m[1] for solid in solids], axis=0)
    origin, shape = voxel_grid(lo, hi, pitch_m, margin_voxels=margin_voxels)
    fill = sample_volume_fill(solids, origin_m=origin, pitch_m=pitch_m, shape=shape, supersample=supersample)
    conductor = fill >= min_fill
    if not np.any(conductor):
        raise ValueError("no voxel is filled to min_fill")
    nz, ny, nx = shape
    hx, hy, hz = pitch_m
    zc = origin[2] + hz * (np.arange(nz) + 0.5)
    yc = origin[1] + hy * (np.arange(ny) + 0.5)
    xc = origin[0] + hx * (np.arange(nx) + 0.5)
    grid_z, grid_y, grid_x = np.meshgrid(zc, yc, xc, indexing="ij")
    centres = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)))
    ports = []
    for region in terminals:
        inside = np.zeros(centres.shape[0], dtype=bool)
        for solid in region.solids:
            inside |= solid.contains(centres)
        if region.box_m is not None:
            lo_b, hi_b = np.asarray(region.box_m[0]), np.asarray(region.box_m[1])
            inside |= np.all((centres >= lo_b) & (centres <= hi_b), axis=1)
        mask = inside.reshape(shape) & conductor
        if not np.any(mask):
            raise ValueError(f"terminal {region.name!r} covers no conductor voxel")
        ports.append(VoxelTerminal(region.name, mask, region.current_a))
    if frequency_hz > 0.0:
        # A voxel PEEC resolves the skin effect only when the voxels are no
        # larger than the skin depth; coarser voxels underestimate the AC
        # resistance (the busbar acceptance shows R/R_dc saturating).
        depth = skin_depth_m(frequency_hz, resistivity_ohm_m)
        if max(pitch_m) > depth:
            warnings.warn(
                f"{name}: voxel pitch {max(pitch_m) * 1e3:.3f} mm exceeds the skin depth {depth * 1e3:.3f} mm at "
                f"{frequency_hz:g} Hz; the AC resistance and loss will be underestimated",
                stacklevel=2,
            )
    problem = VoxelConductorProblem(
        conductor=conductor,
        pitch_m=pitch_m,
        terminals=tuple(ports),
        frequency_hz=frequency_hz,
        resistivity_ohm_m=resistivity_ohm_m,
        origin_m=origin,
        name=name,
    )
    model = VoxelSolidModel(
        material_id=conductor.astype(np.int64),
        materials={1: VoxelMaterial(name, 385.0, role="conductor")},
        pitch_m=pitch_m,
        origin_m=origin,
        fill=np.where(conductor, fill, 0.0),
    )
    return problem, model


def conductor_heat_w(solution: VoxelPeecSolution, model: VoxelSolidModel) -> np.ndarray:
    """The solve's Joule heat per voxel on the model's grid, for ``element_heat_w``."""

    heat = solution.element_heat_w()
    if heat.shape != model.shape:
        raise ValueError("the solution and the voxel model are on different grids")
    return heat


__all__ = ["TerminalRegion", "conductor_heat_w", "conductor_problem_from_solids"]
