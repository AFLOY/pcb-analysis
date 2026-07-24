"""Near/far low-memory cascade helpers for multilayer 2.5D occupancy.

Far interactions stay global (no electromagnetic board cuts).  Sources are
lifted to 3D cell coordinates using the stackup z positions so interlayer
coupling participates in both the exact near sphere and the block multipole
far field.  Reuses the scalar near/far machinery in ``lowmem_peec``.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .lowmem_peec import approximate_energy, exact_energy
from .multilayer_peec import SparseDeltaML
from .stackup import Stackup


def volume_to_points(
    volume: np.ndarray,
    stackup: Stackup,
    *,
    cell_size_m: float,
    abs_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract nonzero occupancy as ``(N, 3)`` cell-unit coordinates."""
    volume = np.asarray(volume, dtype=np.float64)
    if volume.ndim != 3:
        raise ValueError("volume must be (n_layers, ny, nx)")
    if volume.shape[0] != stackup.n_layers:
        raise ValueError("volume layer count must match stackup")
    if not np.all(np.isfinite(volume)):
        raise ValueError("volume must contain only finite values")
    cell_size_m = float(cell_size_m)
    if not math.isfinite(cell_size_m) or cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be finite and positive")
    abs_threshold = float(abs_threshold)
    if not math.isfinite(abs_threshold) or abs_threshold < 0.0:
        raise ValueError("abs_threshold must be finite and non-negative")
    layers, rows, cols = np.nonzero(np.abs(volume) > abs_threshold)
    if layers.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64)
    values = volume[layers, rows, cols]
    z_cells = stackup.z_m[layers] / cell_size_m
    points = np.column_stack(
        [
            rows.astype(np.float64),
            cols.astype(np.float64),
            z_cells.astype(np.float64),
        ]
    )
    return points, values


def delta_to_points(
    base: np.ndarray,
    delta: SparseDeltaML,
    stackup: Stackup,
    *,
    cell_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    occupied = base + delta.dense(base.shape[0], base.shape[1:])
    return volume_to_points(occupied, stackup, cell_size_m=cell_size_m)


def exact_multilayer_energy(
    volume: np.ndarray,
    stackup: Stackup,
    *,
    cell_size_m: float,
    softening: float = 0.75,
) -> float:
    points, values = volume_to_points(
        volume, stackup, cell_size_m=cell_size_m
    )
    return exact_energy(points, values, softening=softening)


def approximate_multilayer_energy(
    volume: np.ndarray,
    stackup: Stackup,
    *,
    cell_size_m: float,
    block_size: int,
    near_radius: float,
    order: Literal[0, 1] = 1,
    softening: float = 0.75,
    storage_dtype=np.float32,
) -> float:
    points, values = volume_to_points(
        volume, stackup, cell_size_m=cell_size_m
    )
    return approximate_energy(
        points,
        values,
        block_size=block_size,
        near_radius=near_radius,
        order=order,
        softening=softening,
        storage_dtype=storage_dtype,
    )


def fidelity_cascade_scores(
    volume: np.ndarray,
    stackup: Stackup,
    *,
    cell_size_m: float,
    softening: float = 0.75,
) -> dict[str, float]:
    """Default two-pass cascade from DESIGN: radius 8 then 16, block width 2."""
    coarse = approximate_multilayer_energy(
        volume,
        stackup,
        cell_size_m=cell_size_m,
        block_size=2,
        near_radius=8.0,
        order=1,
        softening=softening,
        storage_dtype=np.float32,
    )
    fine = approximate_multilayer_energy(
        volume,
        stackup,
        cell_size_m=cell_size_m,
        block_size=2,
        near_radius=16.0,
        order=1,
        softening=softening,
        storage_dtype=np.float32,
    )
    exact = exact_multilayer_energy(
        volume, stackup, cell_size_m=cell_size_m, softening=softening
    )
    return {
        "coarse_radius8": float(coarse),
        "fine_radius16": float(fine),
        "exact": float(exact),
        "split_error": float(abs(fine - coarse)),
        "fine_exact_error": float(abs(exact - fine)),
    }
