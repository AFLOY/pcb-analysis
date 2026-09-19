"""Boundary conditions and loads of the thermal conduction problem.

Newton cooling on the top or bottom face (per-face coefficient and ambient),
Newton cooling on every exposed face of the active elements, and power
spread over named nodes.  Each boundary lumps its own nodal conductance and
load so the operator stays boundary-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import (
    FACE_DIRECTIONS,
    FaceDirection,
    LayeredThermalMesh,
    Node,
    Side,
    _face_area_m2,
    _lump_faces_onto_nodes,
)


@dataclass(frozen=True)
class ConvectionBoundary:
    """Newton cooling on the top or bottom face of the stack.

    The film coefficient is a scalar or one value per element face with shape
    ``(rows, cols)``.  Zero switches a face off, so a board clamped on one
    side and cooled on the other is expressed with one boundary.  The ambient
    temperature is likewise a scalar or one value per element face; the
    per-face form is how a separately meshed heat sink presents its contact
    temperature to the board.  Faces of inactive elements carry no cooling.
    """

    side: Side
    coefficient_w_per_m2_k: float | np.ndarray
    ambient_temperature_k: float | np.ndarray

    def __post_init__(self) -> None:
        if self.side not in ("top", "bottom"):
            raise ValueError("side must be 'top' or 'bottom'")
        coefficient = np.asarray(self.coefficient_w_per_m2_k, dtype=np.float64)
        if coefficient.ndim not in (0, 2):
            raise ValueError("coefficient must be a scalar or a (rows, cols) array")
        if not np.all(np.isfinite(coefficient)) or np.any(coefficient < 0.0):
            raise ValueError("film coefficients must be finite and non-negative")
        ambient = np.asarray(self.ambient_temperature_k, dtype=np.float64)
        if ambient.ndim not in (0, 2):
            raise ValueError("ambient must be a scalar or a (rows, cols) array")
        if not np.all(np.isfinite(ambient)):
            raise ValueError("ambient temperature must be finite")
        if ambient.ndim == 0:
            ambient = float(ambient)
        else:
            ambient = ambient.copy()
        object.__setattr__(self, "coefficient_w_per_m2_k", coefficient.copy())
        object.__setattr__(self, "ambient_temperature_k", ambient)

    def check_shape(self, mesh: "LayeredThermalMesh") -> None:
        shape = mesh.element_grid_shape[1:]
        coefficient = self.coefficient_w_per_m2_k
        if coefficient.ndim == 2 and coefficient.shape != shape:
            raise ValueError(
                "convection coefficient array must match (rows, cols) of the mesh"
            )
        ambient = np.asarray(self.ambient_temperature_k)
        if ambient.ndim == 2 and ambient.shape != shape:
            raise ValueError(
                "convection ambient array must match (rows, cols) of the mesh"
            )

    @property
    def cools(self) -> bool:
        return bool(np.any(self.coefficient_w_per_m2_k > 0.0))

    def mean_ambient_k(self) -> float:
        return float(np.mean(self.ambient_temperature_k))

    def lumped_nodal_weights(
        self, mesh: "LayeredThermalMesh"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Nodal conductance ``R`` and load ``R T_amb`` of this face, flat."""

        slab = -1 if self.side == "top" else 0
        direction = "+z" if self.side == "top" else "-z"
        coefficient = np.broadcast_to(
            self.coefficient_w_per_m2_k, mesh.element_grid_shape[1:]
        )
        ambient = np.broadcast_to(
            self.ambient_temperature_k, mesh.element_grid_shape[1:]
        )
        face = np.zeros(mesh.element_grid_shape, dtype=np.float64)
        face[slab] = coefficient * mesh.active[slab]  # type: ignore[index]
        face *= _face_area_m2(mesh, direction)
        load = face.copy()
        load[slab] *= ambient
        weights = _lump_faces_onto_nodes(face, direction, mesh.node_shape)
        rhs = _lump_faces_onto_nodes(load, direction, mesh.node_shape)
        return weights.reshape(-1), rhs.reshape(-1)


@dataclass(frozen=True)
class ExposedFaceConvection:
    """Newton cooling on every exposed face of the active elements.

    A face is exposed when it borders a void element or the grid boundary.
    ``directions`` restricts the cooled faces (a fin whose edges are
    adiabatic, a body standing on an insulating floor).  With a full mesh and
    all directions this cools the six outer surfaces of the block.
    """

    coefficient_w_per_m2_k: float
    ambient_temperature_k: float
    directions: tuple[FaceDirection, ...] = FACE_DIRECTIONS

    def __post_init__(self) -> None:
        coefficient = float(self.coefficient_w_per_m2_k)
        if not np.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("film coefficient must be finite and non-negative")
        ambient = float(self.ambient_temperature_k)
        if not np.isfinite(ambient):
            raise ValueError("ambient temperature must be finite")
        directions = tuple(self.directions)
        if not directions or any(d not in FACE_DIRECTIONS for d in directions):
            raise ValueError(f"directions must be drawn from {FACE_DIRECTIONS}")
        if len(set(directions)) != len(directions):
            raise ValueError("directions must be unique")
        object.__setattr__(self, "coefficient_w_per_m2_k", coefficient)
        object.__setattr__(self, "ambient_temperature_k", ambient)
        object.__setattr__(self, "directions", directions)

    def check_shape(self, mesh: "LayeredThermalMesh") -> None:
        return None

    @property
    def cools(self) -> bool:
        return self.coefficient_w_per_m2_k > 0.0

    def mean_ambient_k(self) -> float:
        return self.ambient_temperature_k

    def exposed_area_m2(self, mesh: "LayeredThermalMesh") -> float:
        faces = mesh.exposed_faces()
        return float(
            sum(
                np.sum(_face_area_m2(mesh, direction) * faces[direction])
                for direction in self.directions
            )
        )

    def lumped_nodal_weights(
        self, mesh: "LayeredThermalMesh"
    ) -> tuple[np.ndarray, np.ndarray]:
        faces = mesh.exposed_faces()
        weights = np.zeros(mesh.node_shape, dtype=np.float64)
        for direction in self.directions:
            face = (
                self.coefficient_w_per_m2_k
                * _face_area_m2(mesh, direction)
                * faces[direction]
            )
            weights += _lump_faces_onto_nodes(face, direction, mesh.node_shape)
        flat = weights.reshape(-1)
        return flat, flat * self.ambient_temperature_k


Convection = ConvectionBoundary | ExposedFaceConvection


@dataclass(frozen=True)
class HeatSource:
    """A total power spread uniformly over a set of mesh nodes."""

    nodes: tuple[Node, ...]
    power_w: float
    name: str = "source"

    def __post_init__(self) -> None:
        nodes = tuple(tuple(int(index) for index in node) for node in self.nodes)
        if not nodes:
            raise ValueError("a heat source needs at least one node")
        if any(len(node) != 3 for node in nodes):
            raise ValueError("heat source nodes must be (slab_face, row, column)")
        if len(set(nodes)) != len(nodes):
            raise ValueError("heat source nodes must be unique")
        if not np.isfinite(self.power_w):
            raise ValueError("heat source power must be finite")
        object.__setattr__(self, "nodes", nodes)
