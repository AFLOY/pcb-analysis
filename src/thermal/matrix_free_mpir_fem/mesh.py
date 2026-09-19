"""Structured hexahedral Q1 mesh of a layered board or a voxelised body.

Every slab is one layer of hexahedral elements on a shared in-plane pitch;
``active`` carves void elements out (see :mod:`.voxel`).  Also the corner
views, face bookkeeping and unit element matrices the operator and the
boundaries share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np


COPPER_THERMAL_CONDUCTIVITY_W_PER_M_K = 385.0
FR4_IN_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K = 0.8
FR4_THROUGH_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K = 0.3

Node = tuple[int, int, int]
Side = Literal["top", "bottom"]
Preconditioner = Literal["two-level", "jacobi"]


def _corner_views(grid: Any) -> tuple[Any, ...]:
    """Eight corner views of a node grid, ordered ``4 dz + 2 dy + dx``."""

    slabs, rows, cols = (axis - 1 for axis in grid.shape)
    return tuple(
        grid[dz : dz + slabs, dy : dy + rows, dx : dx + cols]
        for dz in (0, 1)
        for dy in (0, 1)
        for dx in (0, 1)
    )


FaceDirection = Literal["-x", "+x", "-y", "+y", "-z", "+z"]
FACE_DIRECTIONS: tuple[FaceDirection, ...] = ("-x", "+x", "-y", "+y", "-z", "+z")
# Local corner indices (``4 dz + 2 dy + dx``) of each element face.
_FACE_CORNERS: dict[str, tuple[int, int, int, int]] = {
    "-z": (0, 1, 2, 3),
    "+z": (4, 5, 6, 7),
    "-y": (0, 1, 4, 5),
    "+y": (2, 3, 6, 7),
    "-x": (0, 2, 4, 6),
    "+x": (1, 3, 5, 7),
}


def exposed_element_faces(active: np.ndarray) -> dict[FaceDirection, np.ndarray]:
    """Faces of active elements that border a void element or the grid edge.

    Returns one boolean element-grid array per direction.  A fully active
    grid exposes only its six outer surfaces.
    """

    mask = np.asarray(active, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("active must have shape (slabs, rows, cols)")
    faces: dict[FaceDirection, np.ndarray] = {}
    for direction in FACE_DIRECTIONS:
        axis = {"x": 2, "y": 1, "z": 0}[direction[1]]
        neighbour = np.zeros_like(mask)
        source = [slice(None)] * 3
        target = [slice(None)] * 3
        if direction[0] == "+":
            target[axis] = slice(None, -1)
            source[axis] = slice(1, None)
        else:
            target[axis] = slice(1, None)
            source[axis] = slice(None, -1)
        neighbour[tuple(target)] = mask[tuple(source)]
        faces[direction] = mask & ~neighbour
    return faces


def _lump_faces_onto_nodes(
    face_weight: np.ndarray,
    direction: str,
    node_shape: tuple[int, int, int],
) -> np.ndarray:
    """Spread one weight per element face equally onto its four nodes."""

    weights = np.zeros(node_shape, dtype=np.float64)
    targets = _corner_views(weights)
    share = face_weight / 4.0
    for corner in _FACE_CORNERS[direction]:
        targets[corner][...] += share
    return weights


def _face_area_m2(mesh: "LayeredThermalMesh", direction: str) -> np.ndarray:
    """Area of every element face pointing in ``direction``, element grid shape."""

    shape = mesh.element_grid_shape
    thickness = np.asarray(mesh.slab_thickness_m, dtype=np.float64)[:, None, None]
    if direction[1] == "z":
        return np.full(shape, mesh.pitch_x_m * mesh.pitch_y_m)
    if direction[1] == "y":
        return np.broadcast_to(mesh.pitch_x_m * thickness, shape).copy()
    return np.broadcast_to(mesh.pitch_y_m * thickness, shape).copy()


def _broadcast_element_field(
    value: Any,
    shape: tuple[int, int, int],
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(shape, float(array), dtype=np.float64)
    if array.shape == (shape[0],):
        return np.broadcast_to(array[:, None, None], shape).copy()
    if array.shape == shape:
        return array.copy()
    raise ValueError(
        f"{name} must be a scalar, one value per slab, or match shape {shape}"
    )


@dataclass(frozen=True)
class LayeredThermalMesh:
    """Structured hexahedral Q1 mesh of a PCB stack.

    ``slab_thickness_m`` lists element slabs from the bottom of the board to
    the top.  Conductivity is a scalar, one value per slab, or one value per
    element with shape ``(slabs, rows, cols)``; ``element_shape`` gives the
    in-plane element count when the conductivity does not.  Through-plane
    conductivity defaults to the in-plane value; laminates are usually
    anisotropic, so both can be given.

    ``active`` marks the elements that exist.  Inactive elements are void:
    their conductivity is zeroed, they carry no heat, and their faces towards
    active elements are exposed surfaces.  A plain board leaves it ``None``;
    a voxelised heat sink or enclosure uses it to carve the body out of its
    bounding box.
    """

    slab_thickness_m: Sequence[float]
    pitch_x_m: float
    pitch_y_m: float
    conductivity_w_per_m_k: float | Sequence[float] | np.ndarray
    element_shape: tuple[int, int] | None = None
    through_plane_conductivity_w_per_m_k: (
        float | Sequence[float] | np.ndarray | None
    ) = None
    active: np.ndarray | None = None

    def __post_init__(self) -> None:
        thickness = np.asarray(self.slab_thickness_m, dtype=np.float64)
        if thickness.ndim != 1 or thickness.size < 1:
            raise ValueError("slab_thickness_m must list at least one slab")
        if not np.all(np.isfinite(thickness)) or np.any(thickness <= 0.0):
            raise ValueError("slab thicknesses must be finite and positive")
        for name in ("pitch_x_m", "pitch_y_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)

        in_plane = np.asarray(self.conductivity_w_per_m_k, dtype=np.float64)
        if self.element_shape is None:
            if in_plane.ndim != 3:
                raise ValueError(
                    "element_shape is required unless conductivity has shape "
                    "(slabs, rows, cols)"
                )
            rows, cols = int(in_plane.shape[1]), int(in_plane.shape[2])
        else:
            rows, cols = (int(value) for value in self.element_shape)
        if rows < 1 or cols < 1:
            raise ValueError("element_shape must have positive axes")
        shape = (int(thickness.size), rows, cols)

        in_plane = _broadcast_element_field(in_plane, shape, "conductivity_w_per_m_k")
        through = (
            in_plane.copy()
            if self.through_plane_conductivity_w_per_m_k is None
            else _broadcast_element_field(
                self.through_plane_conductivity_w_per_m_k,
                shape,
                "through_plane_conductivity_w_per_m_k",
            )
        )
        if self.active is None:
            active = np.ones(shape, dtype=bool)
        else:
            active = np.asarray(self.active, dtype=bool)
            if active.shape != shape:
                raise ValueError("active must match (slabs, rows, cols)")
            if not np.any(active):
                raise ValueError("at least one element must be active")
            active = active.copy()
        for name, array in (
            ("conductivity_w_per_m_k", in_plane),
            ("through_plane_conductivity_w_per_m_k", through),
        ):
            if not np.all(np.isfinite(array)) or np.any(array[active] <= 0.0):
                raise ValueError(
                    f"{name} must be finite and positive on active elements"
                )
        in_plane = np.where(active, in_plane, 0.0)
        through = np.where(active, through, 0.0)

        object.__setattr__(self, "slab_thickness_m", tuple(thickness.tolist()))
        object.__setattr__(self, "element_shape", (rows, cols))
        object.__setattr__(self, "conductivity_w_per_m_k", in_plane)
        object.__setattr__(self, "through_plane_conductivity_w_per_m_k", through)
        object.__setattr__(self, "active", active)

    @property
    def element_grid_shape(self) -> tuple[int, int, int]:
        rows, cols = self.element_shape  # type: ignore[misc]
        return len(self.slab_thickness_m), rows, cols

    @property
    def node_shape(self) -> tuple[int, int, int]:
        slabs, rows, cols = self.element_grid_shape
        return slabs + 1, rows + 1, cols + 1

    @property
    def size(self) -> int:
        return int(np.prod(self.node_shape))

    @property
    def element_volume_m3(self) -> np.ndarray:
        thickness = np.asarray(self.slab_thickness_m, dtype=np.float64)
        return np.broadcast_to(
            (thickness * self.pitch_x_m * self.pitch_y_m)[:, None, None],
            self.element_grid_shape,
        ).copy()

    @property
    def is_full(self) -> bool:
        """True when every element is active (a plain layered board)."""

        return bool(np.all(self.active))

    @property
    def active_nodes(self) -> np.ndarray:
        """Nodes that belong to at least one active element."""

        nodes = np.zeros(self.node_shape, dtype=bool)
        for view in _corner_views(nodes):
            view[...] |= self.active  # type: ignore[operator]
        return nodes

    def exposed_faces(self) -> dict[FaceDirection, np.ndarray]:
        """Active element faces bordering a void or the grid boundary."""

        return exposed_element_faces(self.active)  # type: ignore[arg-type]


def _flat_index(node: Node, shape: tuple[int, int, int]) -> int:
    if len(node) != 3 or any(index < 0 for index in node):
        raise ValueError(f"invalid node {node!r}")
    try:
        return int(np.ravel_multi_index(node, shape))
    except ValueError as exc:
        raise ValueError(f"node {node!r} lies outside mesh shape {shape}") from exc


_STIFFNESS_1D = np.array([[1.0, -1.0], [-1.0, 1.0]])
_MASS_1D = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0


def _local_hexahedron_matrices(
    pitch_x_m: float,
    pitch_y_m: float,
    slab_thickness_m: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Unit-conductivity trilinear stiffness split into in-plane and z parts.

    Local node ordering is ``4 * dz + 2 * dy + dx``, matching the corner
    views below.  Returns arrays of shape ``(slabs, 8, 8)``.
    """

    in_plane = []
    through = []
    for hz in slab_thickness_m:
        hx, hy = pitch_x_m, pitch_y_m
        x_part = (hy * hz / hx) * np.kron(
            _MASS_1D, np.kron(_MASS_1D, _STIFFNESS_1D)
        )
        y_part = (hx * hz / hy) * np.kron(
            _MASS_1D, np.kron(_STIFFNESS_1D, _MASS_1D)
        )
        z_part = (hx * hy / hz) * np.kron(
            _STIFFNESS_1D, np.kron(_MASS_1D, _MASS_1D)
        )
        in_plane.append(x_part + y_part)
        through.append(z_part)
    return np.asarray(in_plane), np.asarray(through)
