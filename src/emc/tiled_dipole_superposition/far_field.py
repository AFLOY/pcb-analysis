"""Far-field radiation pattern and radiated power of a current distribution.

In the far zone the dipole sum collapses to one phase-weighted vector sum per
direction ``n``:

```text
F(n) = Σ_i [(n × p_i) × n] e^{+jk n·r_i}
E(n, r) = (jηk / 4π) e^{-jkr} F(n) / r
P_rad = (1/2η) ∮ |E|² r² dΩ = (η k² / 32π²) ∮ |F|² dΩ
```

The sphere is sampled with Gauss-Legendre nodes in ``cos θ`` and uniform
``φ``, which integrates the pattern of a small source exactly to machine
precision with a few dozen nodes.  The same samples give the maximum field a
test site would read at its measurement distance, per polarisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fields import (
    FREE_SPACE_IMPEDANCE_OHM,
    Backend,
    CurrentDipoles,
    array_namespace,
    to_host,
    wavenumber_per_m,
)


def db_microvolt_per_m(field_v_per_m: np.ndarray | float) -> np.ndarray | float:
    """Convert a field magnitude in V/m to dBµV/m."""

    magnitude = np.asarray(field_v_per_m, dtype=np.float64)
    with np.errstate(divide="ignore"):
        value = 20.0 * np.log10(magnitude / 1.0e-6)
    return float(value) if value.ndim == 0 else value


@dataclass(frozen=True)
class SphereSampling:
    """Directions and quadrature weights covering the full sphere."""

    theta_rad: np.ndarray
    phi_rad: np.ndarray
    direction: np.ndarray
    weight_sr: np.ndarray

    @classmethod
    def gauss_legendre(cls, polar_nodes: int = 32, azimuth_nodes: int = 64) -> "SphereSampling":
        if polar_nodes < 2 or azimuth_nodes < 2:
            raise ValueError("at least two nodes per angle are required")
        cos_theta, cos_weight = np.polynomial.legendre.leggauss(int(polar_nodes))
        theta = np.arccos(cos_theta)
        phi = np.linspace(0.0, 2.0 * np.pi, int(azimuth_nodes), endpoint=False)
        phi_weight = 2.0 * np.pi / int(azimuth_nodes)
        grid_theta, grid_phi = np.meshgrid(theta, phi, indexing="ij")
        weight = np.outer(cos_weight, np.full(int(azimuth_nodes), phi_weight))
        direction = np.stack(
            (
                np.sin(grid_theta) * np.cos(grid_phi),
                np.sin(grid_theta) * np.sin(grid_phi),
                np.cos(grid_theta),
            ),
            axis=-1,
        )
        return cls(
            theta_rad=grid_theta.reshape(-1),
            phi_rad=grid_phi.reshape(-1),
            direction=direction.reshape(-1, 3),
            weight_sr=weight.reshape(-1),
        )

    @property
    def count(self) -> int:
        return int(self.direction.shape[0])


@dataclass(frozen=True)
class FarFieldPattern:
    """Radiation pattern at ``distance_m`` and its integrated power."""

    frequency_hz: float
    distance_m: float
    sampling: SphereSampling
    electric_v_per_m: np.ndarray          # (directions, 3) complex, transverse
    e_theta_v_per_m: np.ndarray           # (directions,) complex
    e_phi_v_per_m: np.ndarray             # (directions,) complex
    radiated_power_w: float

    @property
    def magnitude_v_per_m(self) -> np.ndarray:
        return np.sqrt(np.abs(self.e_theta_v_per_m) ** 2 + np.abs(self.e_phi_v_per_m) ** 2)

    @property
    def max_field_v_per_m(self) -> float:
        return float(np.max(self.magnitude_v_per_m))

    @property
    def max_field_dbuv_per_m(self) -> float:
        return float(db_microvolt_per_m(self.max_field_v_per_m))

    @property
    def max_polarised_field_v_per_m(self) -> float:
        """Largest single linear polarisation, as a receive antenna sees it."""

        return float(max(np.max(np.abs(self.e_theta_v_per_m)), np.max(np.abs(self.e_phi_v_per_m))))

    @property
    def max_direction(self) -> np.ndarray:
        return self.sampling.direction[int(np.argmax(self.magnitude_v_per_m))]

    @property
    def directivity_dbi(self) -> float:
        intensity = self.magnitude_v_per_m**2 * self.distance_m**2 / (2.0 * FREE_SPACE_IMPEDANCE_OHM)
        if self.radiated_power_w <= 0.0:
            return float("nan")
        return float(10.0 * np.log10(4.0 * np.pi * np.max(intensity) / self.radiated_power_w))

    def scaled_to_distance(self, distance_m: float) -> "FarFieldPattern":
        """Inverse-distance rescaling, valid while the new distance is far-zone."""

        factor = self.distance_m / float(distance_m)
        return FarFieldPattern(
            frequency_hz=self.frequency_hz,
            distance_m=float(distance_m),
            sampling=self.sampling,
            electric_v_per_m=self.electric_v_per_m * factor,
            e_theta_v_per_m=self.e_theta_v_per_m * factor,
            e_phi_v_per_m=self.e_phi_v_per_m * factor,
            radiated_power_w=self.radiated_power_w,
        )


def far_field_pattern(
    sources: CurrentDipoles,
    frequency_hz: float,
    *,
    distance_m: float = 10.0,
    sampling: SphereSampling | None = None,
    backend: Backend = "cpu",
    tile_directions: int = 1024,
    dtype: Any = np.complex128,
) -> FarFieldPattern:
    """Evaluate the far-zone field on a sphere and integrate the radiated power.

    The phase reference is the coordinate origin.  Directions are processed in
    tiles against all sources.  ``distance_m`` only scales the reported field
    by ``1/r``; it has to be in the far zone of the source for the numbers to
    mean what a test site measures.
    """

    k = wavenumber_per_m(frequency_hz)
    if k == 0.0:
        raise ValueError("a far field needs a positive frequency")
    if distance_m <= 0.0:
        raise ValueError("distance_m must be positive")
    sampling = sampling or SphereSampling.gauss_legendre()
    xp = array_namespace(backend)
    real_dtype = xp.dtype(dtype).type(0).real.dtype
    src_pos = xp.asarray(sources.position_m, dtype=real_dtype)
    src_mom = xp.asarray(sources.moment_a_m, dtype=dtype)

    pattern = np.empty((sampling.count, 3), dtype=np.complex128)
    for start in range(0, sampling.count, tile_directions):
        stop = min(start + tile_directions, sampling.count)
        n = xp.asarray(sampling.direction[start:stop], dtype=real_dtype)   # (D, 3)
        phase = xp.exp(1j * k * xp.matmul(n, src_pos.T)).astype(dtype)      # (D, S)
        weighted = xp.matmul(phase, src_mom)                                 # (D, 3) Σ p e^{jk n·r}
        radial = xp.sum(n * weighted, axis=-1, keepdims=True)
        transverse = weighted - n * radial                                   # (n×p)×n = p - n(n·p)
        pattern[start:stop] = to_host(transverse)

    eta = FREE_SPACE_IMPEDANCE_OHM
    field = (1j * eta * k / (4.0 * np.pi)) * np.exp(-1j * k * distance_m) / distance_m * pattern
    theta_hat = np.stack(
        (
            np.cos(sampling.theta_rad) * np.cos(sampling.phi_rad),
            np.cos(sampling.theta_rad) * np.sin(sampling.phi_rad),
            -np.sin(sampling.theta_rad),
        ),
        axis=-1,
    )
    phi_hat = np.stack(
        (-np.sin(sampling.phi_rad), np.cos(sampling.phi_rad), np.zeros_like(sampling.phi_rad)),
        axis=-1,
    )
    e_theta = np.sum(field * theta_hat, axis=-1)
    e_phi = np.sum(field * phi_hat, axis=-1)
    intensity = np.sum(np.abs(pattern) ** 2, axis=-1)
    power = float(eta * k * k / (32.0 * np.pi**2) * np.sum(intensity * sampling.weight_sr))
    return FarFieldPattern(
        frequency_hz=float(frequency_hz),
        distance_m=float(distance_m),
        sampling=sampling,
        electric_v_per_m=field,
        e_theta_v_per_m=e_theta,
        e_phi_v_per_m=e_phi,
        radiated_power_w=power,
    )
