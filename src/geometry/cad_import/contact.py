"""Place a voxelised body against the board and build the contact map."""

from __future__ import annotations

from thermal.matrix_free_mpir_fem import ContactMap, LayeredThermalMesh, VoxelSolidModel, planar_contact_map

from .bodymap import ContactSpec
from .section import BoardRaster


def board_body_contact(
    raster: BoardRaster,
    board_mesh: LayeredThermalMesh,
    body_model: VoxelSolidModel,
    body_mesh: LayeredThermalMesh,
    spec: ContactSpec,
) -> ContactMap:
    """Intersect the board's cells with the body's facing voxels in the CAD frame."""

    return planar_contact_map(
        board_mesh,
        body_mesh,
        board_side=spec.board_side,  # type: ignore[arg-type]
        board_origin_m=raster.origin_m,
        body_origin_m=(body_model.origin_m[0], body_model.origin_m[1]),
        conductance_per_area_w_per_m2_k=spec.conductance_per_area_w_per_m2_k,
    )


__all__ = ["board_body_contact"]
