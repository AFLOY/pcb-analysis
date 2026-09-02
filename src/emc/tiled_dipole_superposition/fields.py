"""Exact Hertzian-dipole fields of a current distribution, evaluated in tiles.

Every current element of a solved board is an electric dipole with moment
``p = I dl`` (A·m).  Its fields at a point ``r`` away, with ``n = r / |r|``,
``k = ω / c`` and the ``e^{jωt}`` phasor convention, are

```text
H = (-jk / 4π) (n × p) (1 + 1/(jkr)) e^{-jkr} / r
E = (η / (4π jk)) { k² (n × p) × n / r + [3n(n·p) - p] (1/r³ + jk/r²) } e^{-jkr}
```

These hold at every distance, so one evaluation serves the near-field scan a
probe would measure a few millimetres above the board and the far-field
estimate a test site would read at 3 m or 10 m.  At ``k = 0`` the magnetic
expression reduces to Biot-Savart; the electric field of a pure current
distribution is undefined there, because the static field belongs to charges
the current alone does not determine.

The pairwise work is ``O(points × sources)``.  It is evaluated in tiles of
observation points against all sources so memory stays bounded, on NumPy or
CuPy through one array namespace.  No pair matrix is retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
MU_0_H_PER_M = 1.25663706212e-6
EPSILON_0_F_PER_M = 1.0 / (MU_0_H_PER_M * SPEED_OF_LIGHT_M_PER_S**2)
FREE_SPACE_IMPEDANCE_OHM = MU_0_H_PER_M * SPEED_OF_LIGHT_M_PER_S

Backend = Literal["cpu", "cuda", "auto"]


def array_namespace(backend: Backend = "cpu") -> Any:
    """Return NumPy or CuPy for the requested backend."""

    if backend == "cpu":
        return np
    if backend in ("cuda", "auto"):
        try:
            import cupy as cp

            if int(cp.cuda.runtime.getDeviceCount()) > 0:
                return cp
        except Exception as exc:  # pragma: no cover - depends on local CUDA
            if backend == "cuda":
                raise RuntimeError(
                    "CUDA backend requested, but CuPy could not see a device"
                ) from exc
        if backend == "cuda":
            raise RuntimeError("CUDA backend requested, but no CUDA device is visible")
        return np
    raise ValueError("backend must be 'cpu', 'cuda', or 'auto'")


def to_host(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    return np.asarray(array.get())


def wavenumber_per_m(frequency_hz: float) -> float:
    frequency = float(frequency_hz)
    if not np.isfinite(frequency) or frequency < 0.0:
        raise ValueError("frequency must be finite and non-negative")
    return 2.0 * np.pi * frequency / SPEED_OF_LIGHT_M_PER_S


@dataclass(frozen=True)
class CurrentDipoles:
    """Electric current elements: positions in m and complex moments in A·m."""

    position_m: np.ndarray
    moment_a_m: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position_m, dtype=np.float64)
        moment = np.asarray(self.moment_a_m, dtype=np.complex128)
        if position.ndim != 2 or position.shape[1] != 3:
            raise ValueError("position_m must have shape (sources, 3)")
        if moment.shape != position.shape:
            raise ValueError("moment_a_m must match position_m in shape")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(moment)):
            raise ValueError("dipole positions and moments must be finite")
        object.__setattr__(self, "position_m", position.copy())
        object.__setattr__(self, "moment_a_m", moment.copy())

    @property
    def count(self) -> int:
        return int(self.position_m.shape[0])

    def concatenate(self, other: "CurrentDipoles") -> "CurrentDipoles":
        return CurrentDipoles(
            np.vstack((self.position_m, other.position_m)),
            np.vstack((self.moment_a_m, other.moment_a_m)),
        )

    def scaled(self, factor: complex) -> "CurrentDipoles":
        return CurrentDipoles(self.position_m, self.moment_a_m * complex(factor))

    def with_ground_plane_images(self, plane_z_m: float) -> "CurrentDipoles":
        """Add the images in an infinite perfect conductor at ``z = plane_z_m``.

        A horizontal current element sees an inverted image, a vertical one an
        upright image.  Fields are then valid only on the source side of the
        plane.  Every source must lie on one side of it.
        """

        z0 = float(plane_z_m)
        offsets = self.position_m[:, 2] - z0
        if np.any(offsets == 0.0) or (np.any(offsets > 0.0) and np.any(offsets < 0.0)):
            raise ValueError("all sources must lie strictly on one side of the plane")
        image_position = self.position_m.copy()
        image_position[:, 2] = 2.0 * z0 - self.position_m[:, 2]
        image_moment = self.moment_a_m * np.array([-1.0, -1.0, 1.0])
        return self.concatenate(CurrentDipoles(image_position, image_moment))


@dataclass(frozen=True)
class FieldSamples:
    """Complex field phasors at observation points (V/m and A/m)."""

    point_m: np.ndarray
    frequency_hz: float
    electric_v_per_m: np.ndarray | None
    magnetic_a_per_m: np.ndarray

    @property
    def magnetic_magnitude_a_per_m(self) -> np.ndarray:
        return np.linalg.norm(self.magnetic_a_per_m, axis=-1)

    @property
    def electric_magnitude_v_per_m(self) -> np.ndarray:
        if self.electric_v_per_m is None:
            raise ValueError("the electric field was not evaluated (zero frequency)")
        return np.linalg.norm(self.electric_v_per_m, axis=-1)


def _cross(xp: Any, a: Any, b: Any) -> Any:
    return xp.stack(
        (
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ),
        axis=-1,
    )


def evaluate_fields(
    sources: CurrentDipoles,
    points_m: np.ndarray,
    frequency_hz: float,
    *,
    electric: bool = True,
    backend: Backend = "cpu",
    tile_points: int = 2048,
    dtype: Any = np.complex128,
) -> FieldSamples:
    """Sum the exact dipole fields of every source at every observation point.

    ``tile_points`` observation points are processed against all sources at
    once, so peak memory is ``tile_points × sources × 3`` complex values.  Use
    ``dtype=np.complex64`` on a GPU whose FP64 throughput is limited when the
    near field, which does not rely on cancellation, is what is wanted.
    """

    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_m must have shape (points, 3)")
    if tile_points < 1:
        raise ValueError("tile_points must be positive")
    k = wavenumber_per_m(frequency_hz)
    if electric and k == 0.0:
        raise ValueError(
            "the electric field of a current distribution is undefined at zero "
            "frequency; pass electric=False for the Biot-Savart magnetic field"
        )

    xp = array_namespace(backend)
    real_dtype = xp.dtype(dtype).type(0).real.dtype
    src_pos = xp.asarray(sources.position_m, dtype=real_dtype)
    src_mom = xp.asarray(sources.moment_a_m, dtype=dtype)
    magnetic = np.empty(points.shape, dtype=np.complex128)
    electric_field = np.empty(points.shape, dtype=np.complex128) if electric else None
    eta = FREE_SPACE_IMPEDANCE_OHM

    for start in range(0, points.shape[0], tile_points):
        stop = min(start + tile_points, points.shape[0])
        obs = xp.asarray(points[start:stop], dtype=real_dtype)
        separation = obs[:, None, :] - src_pos[None, :, :]        # (P, S, 3)
        distance = xp.sqrt(xp.sum(separation * separation, axis=-1))
        if bool(xp.any(distance == 0.0)):
            raise ValueError("an observation point coincides with a source")
        unit = separation / distance[..., None]
        phase = xp.exp(-1j * k * distance).astype(dtype)
        n_cross_p = _cross(xp, unit, src_mom[None, :, :])
        inv_r = 1.0 / distance
        if k == 0.0:
            h_scale = -(inv_r * inv_r) / (4.0 * np.pi)
        else:
            h_scale = (-1j * k / (4.0 * np.pi)) * (1.0 + 1.0 / (1j * k * distance)) * phase * inv_r
        magnetic[start:stop] = to_host(
            xp.sum(n_cross_p * h_scale[..., None], axis=1)
        )
        if electric_field is not None:
            radial = xp.sum(unit * src_mom[None, :, :], axis=-1)
            static_like = 3.0 * unit * radial[..., None] - src_mom[None, :, :]
            far_part = _cross(xp, n_cross_p, unit) * (k * k * inv_r)[..., None]
            near_part = static_like * (inv_r**3 + 1j * k * inv_r**2)[..., None]
            e_scale = (eta / (4.0 * np.pi * 1j * k)) * phase
            electric_field[start:stop] = to_host(
                xp.sum((far_part + near_part) * e_scale[..., None], axis=1)
            )
    return FieldSamples(
        point_m=points,
        frequency_hz=float(frequency_hz),
        electric_v_per_m=electric_field,
        magnetic_a_per_m=magnetic,
    )


def scan_plane(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: float,
) -> np.ndarray:
    """Observation points of a horizontal near-field scan, row-major in (y, x)."""

    xs = np.asarray(x_m, dtype=np.float64).reshape(-1)
    ys = np.asarray(y_m, dtype=np.float64).reshape(-1)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return np.column_stack(
        (grid_x.reshape(-1), grid_y.reshape(-1), np.full(grid_x.size, float(z_m)))
    )
