"""Low-memory near/far interaction approximation for PCB-like current paths.

This module is a scalar validation model for the proposed streamed 2.5D PEEC
operator. Near interactions are evaluated exactly. Far interactions are grouped
into spatial blocks using a monopole or monopole-plus-dipole expansion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FarFieldBlocks:
    centers: np.ndarray
    charge: np.ndarray
    dipole: np.ndarray
    source_block: np.ndarray


def interaction_kernel(delta: np.ndarray, softening: float) -> np.ndarray:
    return 1.0 / np.sqrt(np.sum(delta * delta, axis=-1) + softening**2)


def exact_energy(
    points: np.ndarray, values: np.ndarray, softening: float = 0.75
) -> float:
    delta = points[:, None, :] - points[None, :, :]
    kernel = interaction_kernel(delta, softening)
    return float(values @ kernel @ values)


def build_blocks(
    points: np.ndarray, values: np.ndarray, block_size: int
) -> FarFieldBlocks:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points must have shape (n_sources, n_dim)")
    n_dim = points.shape[1]
    block_index = np.floor_divide(points, block_size).astype(np.int32)
    unique, source_block = np.unique(block_index, axis=0, return_inverse=True)
    centers = (unique.astype(np.float64) + 0.5) * block_size - 0.5
    charge = np.bincount(source_block, weights=values, minlength=len(unique))
    offsets = points - centers[source_block]
    dipole = np.zeros((len(unique), n_dim), dtype=np.float64)
    np.add.at(dipole, source_block, values[:, None] * offsets)
    return FarFieldBlocks(centers, charge, dipole, source_block)


def approximate_energy(
    points: np.ndarray,
    values: np.ndarray,
    block_size: int,
    near_radius: float,
    order: int = 1,
    softening: float = 0.75,
    storage_dtype=np.float32,
) -> float:
    """Calculate energy with exact near field and grouped far field.

    The full block expansion is evaluated first. For sources within the near
    radius, their individual expansion contribution is removed and replaced by
    the exact kernel value. This gives a clean, non-overlapping split.
    """
    if order not in (0, 1):
        raise ValueError("order must be 0 (monopole) or 1 (dipole)")
    points64 = np.asarray(points, dtype=np.float64)
    values64 = np.asarray(values, dtype=np.float64)
    blocks = build_blocks(points64, values64, block_size)

    calc_dtype = np.dtype(storage_dtype).type
    points_calc = points64.astype(calc_dtype)
    values_calc = values64.astype(calc_dtype)
    centers = blocks.centers.astype(calc_dtype)
    charge = blocks.charge.astype(calc_dtype)
    dipole = blocks.dipole.astype(calc_dtype)
    softening_calc = calc_dtype(softening)
    radius2 = float(near_radius) ** 2
    energy = 0.0

    for target_index, target in enumerate(points_calc):
        s = target - centers
        den2 = np.sum(s * s, axis=1, dtype=calc_dtype) + softening_calc**2
        block_phi = charge / np.sqrt(den2)
        if order == 1:
            block_phi += (
                np.sum(s * dipole, axis=1, dtype=calc_dtype)
                / np.power(den2, calc_dtype(1.5))
            )
        # Mixed-precision policy: field terms are stored/computed in the
        # selected dtype, while global reductions use float64.
        phi = float(np.sum(block_phi, dtype=np.float64))

        source_delta = target - points_calc
        near = np.sum(source_delta * source_delta, axis=1, dtype=calc_dtype) <= radius2
        near_indices = np.flatnonzero(near)
        if near_indices.size:
            source_blocks = blocks.source_block[near_indices]
            source_s = target - centers[source_blocks]
            source_den2 = (
                np.sum(source_s * source_s, axis=1, dtype=calc_dtype)
                + softening_calc**2
            )
            approximate_source = values_calc[near_indices] / np.sqrt(source_den2)
            if order == 1:
                source_offset = points_calc[near_indices] - centers[source_blocks]
                approximate_source += (
                    values_calc[near_indices]
                    * np.sum(source_s * source_offset, axis=1, dtype=calc_dtype)
                    / np.power(source_den2, calc_dtype(1.5))
                )
            exact_den2 = (
                np.sum(
                    source_delta[near_indices] * source_delta[near_indices],
                    axis=1,
                    dtype=calc_dtype,
                )
                + softening_calc**2
            )
            exact_source = values_calc[near_indices] / np.sqrt(exact_den2)
            phi += float(
                np.sum(exact_source - approximate_source, dtype=np.float64)
            )
        energy += float(values_calc[target_index]) * phi
    return float(energy)
