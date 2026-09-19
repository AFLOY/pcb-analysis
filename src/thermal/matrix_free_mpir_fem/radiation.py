"""Surface-to-ambient radiation as a Newton-linearised Robin boundary.

A surface at ``T`` facing an environment at ``T_amb`` with emissivity ``ε``
loses

```text
q = ε σ (T⁴ - T_amb⁴)                       [W/m²].
```

Linearised at the current iterate ``T_k`` this is Newton's step on the
radiation term,

```text
q ≈ h_k (T - T_eff,k),   h_k = 4 ε σ T_k³,
T_eff,k = T_k - (T_k⁴ - T_amb⁴) / (4 T_k³),
```

which is an ordinary convection boundary with a per-face film coefficient and
a per-face ambient.  Conduction is linear, so iterating the linearised solve
to a fixed point is Newton's method on the whole problem: it converges
quadratically and, unlike the secant form ``h = ε σ (T² + T_amb²)(T + T_amb)``
with the true ambient, it does not oscillate when the rise exceeds a third
of the absolute temperature.

The model is grey and diffuse and sees only the ambient: no view factors
between surfaces, so a board inside a case radiates to the case's *given*
inner temperature, not to its computed field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundaries import ConvectionBoundary, ExposedFaceConvection
from .mesh import FACE_DIRECTIONS, FaceDirection, LayeredThermalMesh, Side, _corner_views


STEFAN_BOLTZMANN_W_PER_M2_K4 = 5.670374419e-8


def _check_emissivity(value: float | np.ndarray, ndim: tuple[int, ...]) -> float | np.ndarray:
    emissivity = np.asarray(value, dtype=np.float64)
    if emissivity.ndim not in ndim:
        raise ValueError("emissivity must be a scalar or an array matching the faces")
    if not np.all(np.isfinite(emissivity)) or np.any(emissivity < 0.0) or np.any(emissivity > 1.0):
        raise ValueError("emissivity must lie in [0, 1]")
    return float(emissivity) if emissivity.ndim == 0 else emissivity.copy()


def _check_ambient(value: float | np.ndarray, ndim: tuple[int, ...]) -> float | np.ndarray:
    ambient = np.asarray(value, dtype=np.float64)
    if ambient.ndim not in ndim:
        raise ValueError("ambient must be a scalar or an array matching the faces")
    if not np.all(np.isfinite(ambient)) or np.any(ambient <= 0.0):
        raise ValueError("radiative ambient temperature must be finite and positive (kelvin)")
    return float(ambient) if ambient.ndim == 0 else ambient.copy()


def newton_linearisation(
    emissivity: float | np.ndarray,
    surface_temperature_k: np.ndarray,
    ambient_temperature_k: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``(h_k, T_eff,k)`` of the Newton step at the given surface temperature."""

    surface = np.asarray(surface_temperature_k, dtype=np.float64)
    ambient = np.asarray(ambient_temperature_k, dtype=np.float64)
    surface = np.where(np.isfinite(surface) & (surface > 0.0), surface, ambient)
    cubic = surface**3
    coefficient = 4.0 * np.asarray(emissivity) * STEFAN_BOLTZMANN_W_PER_M2_K4 * cubic
    effective = surface - (surface**4 - ambient**4) / (4.0 * cubic)
    return coefficient, np.broadcast_to(effective, surface.shape).copy()


def face_temperature_k(mesh: LayeredThermalMesh, temperature_k: np.ndarray, side: Side) -> np.ndarray:
    """Mean corner temperature of every top or bottom element face, ``(rows, cols)``."""

    grid = np.asarray(temperature_k, dtype=np.float64).reshape(mesh.node_shape)
    layer = grid[-1] if side == "top" else grid[0]
    return 0.25 * (layer[:-1, :-1] + layer[:-1, 1:] + layer[1:, :-1] + layer[1:, 1:])


def element_temperature_k(mesh: LayeredThermalMesh, temperature_k: np.ndarray) -> np.ndarray:
    """Mean corner temperature of every element, ``(slabs, rows, cols)``."""

    grid = np.asarray(temperature_k, dtype=np.float64).reshape(mesh.node_shape)
    total = np.zeros(mesh.element_grid_shape, dtype=np.float64)
    for view in _corner_views(grid):
        total += view
    return total / 8.0


@dataclass(frozen=True)
class RadiationBoundary:
    """Radiation from the top or bottom face of the stack to an ambient.

    ``emissivity`` and ``ambient_temperature_k`` are scalars or ``(rows, cols)``
    arrays; a zero emissivity switches a face off.  Faces of inactive
    elements radiate nothing.
    """

    side: Side
    emissivity: float | np.ndarray
    ambient_temperature_k: float | np.ndarray

    def __post_init__(self) -> None:
        if self.side not in ("top", "bottom"):
            raise ValueError("side must be 'top' or 'bottom'")
        object.__setattr__(self, "emissivity", _check_emissivity(self.emissivity, (0, 2)))
        object.__setattr__(self, "ambient_temperature_k", _check_ambient(self.ambient_temperature_k, (0, 2)))

    def check_shape(self, mesh: LayeredThermalMesh) -> None:
        shape = mesh.element_grid_shape[1:]
        for name in ("emissivity", "ambient_temperature_k"):
            value = np.asarray(getattr(self, name))
            if value.ndim == 2 and value.shape != shape:
                raise ValueError(f"{name} array must match (rows, cols) of the mesh")

    @property
    def radiates(self) -> bool:
        return bool(np.any(np.asarray(self.emissivity) > 0.0))

    def mean_ambient_k(self) -> float:
        return float(np.mean(self.ambient_temperature_k))

    def linearize(self, mesh: LayeredThermalMesh, temperature_k: np.ndarray) -> ConvectionBoundary:
        """The Newton-linearised Robin boundary at the given nodal temperatures."""

        shape = mesh.element_grid_shape[1:]
        surface = face_temperature_k(mesh, temperature_k, self.side)
        ambient = np.broadcast_to(self.ambient_temperature_k, shape)
        coefficient, effective = newton_linearisation(self.emissivity, surface, ambient)
        return ConvectionBoundary(self.side, coefficient, effective)


@dataclass(frozen=True)
class ExposedFaceRadiation:
    """Radiation from every exposed face of the active elements to an ambient.

    ``emissivity`` and ``ambient_temperature_k`` are scalars or one value per
    element, ``(slabs, rows, cols)``, applied to that element's exposed faces
    in ``directions``.
    """

    emissivity: float | np.ndarray
    ambient_temperature_k: float | np.ndarray
    directions: tuple[FaceDirection, ...] = FACE_DIRECTIONS

    def __post_init__(self) -> None:
        directions = tuple(self.directions)
        if not directions or any(d not in FACE_DIRECTIONS for d in directions):
            raise ValueError(f"directions must be drawn from {FACE_DIRECTIONS}")
        if len(set(directions)) != len(directions):
            raise ValueError("directions must be unique")
        object.__setattr__(self, "emissivity", _check_emissivity(self.emissivity, (0, 3)))
        object.__setattr__(self, "ambient_temperature_k", _check_ambient(self.ambient_temperature_k, (0, 3)))
        object.__setattr__(self, "directions", directions)

    def check_shape(self, mesh: LayeredThermalMesh) -> None:
        shape = mesh.element_grid_shape
        for name in ("emissivity", "ambient_temperature_k"):
            value = np.asarray(getattr(self, name))
            if value.ndim == 3 and value.shape != shape:
                raise ValueError(f"{name} array must match (slabs, rows, cols) of the mesh")

    @property
    def radiates(self) -> bool:
        return bool(np.any(np.asarray(self.emissivity) > 0.0))

    def mean_ambient_k(self) -> float:
        return float(np.mean(self.ambient_temperature_k))

    def linearize(self, mesh: LayeredThermalMesh, temperature_k: np.ndarray) -> ExposedFaceConvection:
        """The Newton-linearised Robin boundary at the given nodal temperatures."""

        shape = mesh.element_grid_shape
        surface = element_temperature_k(mesh, temperature_k)
        ambient = np.broadcast_to(self.ambient_temperature_k, shape)
        coefficient, effective = newton_linearisation(self.emissivity, surface, ambient)
        return ExposedFaceConvection(coefficient, effective, directions=self.directions)


Radiation = RadiationBoundary | ExposedFaceRadiation


__all__ = [
    "ExposedFaceRadiation",
    "Radiation",
    "RadiationBoundary",
    "STEFAN_BOLTZMANN_W_PER_M2_K4",
    "element_temperature_k",
    "face_temperature_k",
    "newton_linearisation",
]
