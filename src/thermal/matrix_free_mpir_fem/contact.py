"""Thermal contact between a board face and a separately meshed body.

The board is a ``LayeredThermalMesh`` on the routing grid; a heat sink,
enclosure or component body is another ``LayeredThermalMesh`` (usually a
``VoxelThermalMesh``) with its own pitch and origin.  A ``ContactMap`` lists
which board element face touches which body element face, the shared area,
and the conductance of the joint.  It is pure geometry plus a contact
resistance; the interface iteration that uses it lives in
``multiphysics.staggered_coupling``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import (
    _FACE_CORNERS,
    FaceDirection,
    LayeredThermalMesh,
    Side,
    _corner_views,
)


@dataclass(frozen=True)
class ContactMap:
    """Element-face pairs across one board/body interface.

    ``board_cells`` are ``(row, col)`` faces on ``board_side`` of the board;
    ``body_cells`` are ``(slab, row, col)`` elements of the body whose
    ``body_face`` touches them.  ``area_m2`` is the overlap of the two
    footprints and ``conductance_w_per_k`` the joint conductance ``G`` of the
    pair, so ``q = G (T_board - T_body)`` flows across it.
    """

    board_side: Side
    body_face: FaceDirection
    board_cells: np.ndarray
    body_cells: np.ndarray
    area_m2: np.ndarray
    conductance_w_per_k: np.ndarray

    def __post_init__(self) -> None:
        if self.board_side not in ("top", "bottom"):
            raise ValueError("board_side must be 'top' or 'bottom'")
        if self.body_face not in _FACE_CORNERS:
            raise ValueError(f"body_face must be one of {tuple(_FACE_CORNERS)}")
        board = np.asarray(self.board_cells, dtype=np.int64)
        body = np.asarray(self.body_cells, dtype=np.int64)
        area = np.asarray(self.area_m2, dtype=np.float64).reshape(-1)
        conductance = np.asarray(self.conductance_w_per_k, dtype=np.float64).reshape(-1)
        count = area.size
        if count == 0:
            raise ValueError("a contact map needs at least one pair")
        if board.shape != (count, 2) or body.shape != (count, 3):
            raise ValueError(
                "board_cells must be (m, 2), body_cells (m, 3), with m pairs"
            )
        if np.any(board < 0) or np.any(body < 0):
            raise ValueError("cell indices must be non-negative")
        if not np.all(np.isfinite(area)) or np.any(area <= 0.0):
            raise ValueError("areas must be finite and positive")
        if not np.all(np.isfinite(conductance)) or np.any(conductance <= 0.0):
            raise ValueError("conductances must be finite and positive")
        object.__setattr__(self, "board_cells", board)
        object.__setattr__(self, "body_cells", body)
        object.__setattr__(self, "area_m2", area)
        object.__setattr__(self, "conductance_w_per_k", conductance)

    @property
    def size(self) -> int:
        return int(self.area_m2.size)

    @property
    def board_face(self) -> FaceDirection:
        return "+z" if self.board_side == "top" else "-z"

    @property
    def board_slab_index(self) -> int:
        return -1 if self.board_side == "top" else 0

    def check(self, board: LayeredThermalMesh, body: LayeredThermalMesh) -> None:
        """Every pair must name an active face on both meshes."""

        slabs, rows, cols = board.element_grid_shape
        if np.any(self.board_cells[:, 0] >= rows) or np.any(self.board_cells[:, 1] >= cols):
            raise ValueError("board_cells lie outside the board grid")
        board_slab = slabs - 1 if self.board_side == "top" else 0
        if not np.all(board.active[board_slab, self.board_cells[:, 0], self.board_cells[:, 1]]):  # type: ignore[index]
            raise ValueError("board_cells name inactive board elements")
        shape = np.asarray(body.element_grid_shape)
        if np.any(self.body_cells >= shape[None, :]):
            raise ValueError("body_cells lie outside the body grid")
        if not np.all(body.active[tuple(self.body_cells.T)]):  # type: ignore[index]
            raise ValueError("body_cells name inactive body elements")

    # ------------------------------------------------------------ transfers
    def _face_nodes(
        self, cells: np.ndarray, face: FaceDirection, node_shape: tuple[int, int, int]
    ) -> np.ndarray:
        """Flat node indices ``(m, 4)`` of the given face of each element."""

        corners = np.asarray(_FACE_CORNERS[face], dtype=np.int64)
        dz, dy, dx = corners // 4, (corners // 2) % 2, corners % 2
        z = cells[:, 0][:, None] + dz[None, :]
        y = cells[:, 1][:, None] + dy[None, :]
        x = cells[:, 2][:, None] + dx[None, :]
        return np.ravel_multi_index((z, y, x), node_shape)

    def board_face_nodes(self, board: LayeredThermalMesh) -> np.ndarray:
        slabs = board.element_grid_shape[0]
        slab = np.full(self.size, slabs - 1 if self.board_side == "top" else 0)
        cells = np.column_stack((slab, self.board_cells))
        return self._face_nodes(cells, self.board_face, board.node_shape)

    def body_face_nodes(self, body: LayeredThermalMesh) -> np.ndarray:
        return self._face_nodes(self.body_cells, self.body_face, body.node_shape)

    def board_face_temperature_k(
        self, board: LayeredThermalMesh, temperature_k: np.ndarray
    ) -> np.ndarray:
        flat = np.asarray(temperature_k, dtype=np.float64).reshape(-1)
        return np.mean(flat[self.board_face_nodes(board)], axis=1)

    def body_face_temperature_k(
        self, body: LayeredThermalMesh, temperature_k: np.ndarray
    ) -> np.ndarray:
        flat = np.asarray(temperature_k, dtype=np.float64).reshape(-1)
        return np.mean(flat[self.body_face_nodes(body)], axis=1)

    def board_robin(
        self, board: LayeredThermalMesh, body_temperature_k: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-face film coefficient and ambient the board sees, ``(rows, cols)``.

        Several body faces may share one board face; their conductances add
        and their temperatures average with conductance weights.
        """

        body_temperature = np.asarray(body_temperature_k, dtype=np.float64).reshape(-1)
        if body_temperature.size != self.size:
            raise ValueError("one body temperature per contact pair is required")
        shape = board.element_grid_shape[1:]
        conductance = np.zeros(shape)
        weighted = np.zeros(shape)
        np.add.at(conductance, (self.board_cells[:, 0], self.board_cells[:, 1]), self.conductance_w_per_k)
        np.add.at(
            weighted,
            (self.board_cells[:, 0], self.board_cells[:, 1]),
            self.conductance_w_per_k * body_temperature,
        )
        cell_area = board.cell_area_m2
        touched = conductance > 0.0
        ambient = np.where(touched, weighted / np.where(touched, conductance, 1.0), 0.0)
        return conductance / cell_area, ambient

    def pair_heat_w(
        self, board_face_temperature_k: np.ndarray, body_face_temperature_k: np.ndarray
    ) -> np.ndarray:
        """Heat from board to body across each pair, ``G (T_board - T_body)``."""

        return self.conductance_w_per_k * (
            np.asarray(board_face_temperature_k, dtype=np.float64)
            - np.asarray(body_face_temperature_k, dtype=np.float64)
        )

    def body_nodal_heat_w(self, body: LayeredThermalMesh, pair_heat_w: np.ndarray) -> np.ndarray:
        """Lump the pair heat onto the four body face nodes of each pair."""

        load = np.zeros(body.size, dtype=np.float64)
        nodes = self.body_face_nodes(body)
        share = np.asarray(pair_heat_w, dtype=np.float64) / 4.0
        np.add.at(load, nodes.reshape(-1), np.repeat(share, 4))
        return load.reshape(body.node_shape)


def planar_contact_map(
    board: LayeredThermalMesh,
    body: LayeredThermalMesh,
    *,
    board_side: Side,
    board_origin_m: tuple[float, float] = (0.0, 0.0),
    body_origin_m: tuple[float, float] = (0.0, 0.0),
    conductance_per_area_w_per_m2_k: float,
    minimum_overlap: float = 1.0e-6,
) -> ContactMap:
    """Contact map between a board face and the facing plane of a body.

    The body touches the board with its ``-z`` face when it sits on the
    board's top, and with its ``+z`` face when it hangs below the bottom.  The
    body elements considered are those of the facing slab whose face is
    exposed (the body's own bottom or top surface).  Footprints are
    intersected in the common in-plane frame given by the two origins (the
    position of node ``(row 0, col 0)`` of each grid).  Pairs whose overlap is
    below ``minimum_overlap`` of the smaller cell are dropped.
    ``conductance_per_area_w_per_m2_k`` is the joint conductance per unit
    area, ``k / t`` for an interface material of conductivity ``k`` and
    thickness ``t``.
    """

    if conductance_per_area_w_per_m2_k <= 0.0 or not np.isfinite(conductance_per_area_w_per_m2_k):
        raise ValueError("conductance per area must be finite and positive")
    body_face: FaceDirection = "-z" if board_side == "top" else "+z"
    body_slab = 0 if board_side == "top" else body.element_grid_shape[0] - 1
    board_slab = board.element_grid_shape[0] - 1 if board_side == "top" else 0
    board_active = board.active[board_slab]  # type: ignore[index]
    body_exposed = body.exposed_faces()[body_face][body_slab]

    # Overlap of two 1D cell ranges along one axis, for every pair; the cells
    # of either grid may be graded.
    def overlaps(edges_a: np.ndarray, origin_a: float, edges_b: np.ndarray, origin_b: float) -> np.ndarray:
        a0 = (origin_a + edges_a[:-1])[:, None]
        a1 = (origin_a + edges_a[1:])[:, None]
        b0 = (origin_b + edges_b[:-1])[None, :]
        b1 = (origin_b + edges_b[1:])[None, :]
        return np.clip(np.minimum(a1, b1) - np.maximum(a0, b0), 0.0, None)

    overlap_y = overlaps(board.y_edges_m, board_origin_m[1], body.y_edges_m, body_origin_m[1])
    overlap_x = overlaps(board.x_edges_m, board_origin_m[0], body.x_edges_m, body_origin_m[0])
    smaller = min(float(np.min(board.cell_area_m2)), float(np.min(body.cell_area_m2)))

    board_cells: list[tuple[int, int]] = []
    body_cells: list[tuple[int, int, int]] = []
    areas: list[float] = []
    rows_y, rows_body = np.nonzero(overlap_y > 0.0)
    cols_x, cols_body = np.nonzero(overlap_x > 0.0)
    for by, bby in zip(rows_y.tolist(), rows_body.tolist()):
        for bx, bbx in zip(cols_x.tolist(), cols_body.tolist()):
            if not board_active[by, bx] or not body_exposed[bby, bbx]:
                continue
            area = float(overlap_y[by, bby] * overlap_x[bx, bbx])
            if area < minimum_overlap * smaller:
                continue
            board_cells.append((by, bx))
            body_cells.append((body_slab, bby, bbx))
            areas.append(area)
    if not areas:
        raise ValueError("the board face and the body do not overlap")
    area_array = np.asarray(areas)
    contact = ContactMap(
        board_side=board_side,
        body_face=body_face,
        board_cells=np.asarray(board_cells),
        body_cells=np.asarray(body_cells),
        area_m2=area_array,
        conductance_w_per_k=area_array * conductance_per_area_w_per_m2_k,
    )
    contact.check(board, body)
    return contact


__all__ = ["ContactMap", "planar_contact_map"]
