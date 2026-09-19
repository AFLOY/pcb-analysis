"""STEP geometry front end: 2.5D board raster, 3D voxel bodies, contact maps.

``load_step`` needs the optional ``cad`` extra (``OCP``); everything that
works on arrays and dataclasses imports without it.
"""

from .adapters import (
    BoardThermalModel,
    board_occupancy,
    board_stackup,
    board_thermal_mesh,
    board_vias,
    body_heat_sources,
    body_thermal_mesh,
    plane_opt_problem_mapping,
)
from .bodymap import (
    BoardSpec,
    BodyMap,
    BodySpec,
    ContactSpec,
    CopperSpec,
    LayerSpec,
    ResolvedBodies,
    ViaSpec,
    resolve_bodies,
)
from .contact import board_body_contact
from .reader import (
    StepModel,
    StepSolid,
    box_solid,
    cylinder_solid,
    load_step,
    ocp_available,
    synthetic_model,
    write_step,
)
from .section import BoardRaster, rasterize_board, sample_plane_fill
from .voxelize import sample_volume_fill, voxel_grid, voxelize_bodies

__all__ = [
    "BoardRaster",
    "BoardSpec",
    "BoardThermalModel",
    "BodyMap",
    "BodySpec",
    "ContactSpec",
    "CopperSpec",
    "LayerSpec",
    "ResolvedBodies",
    "StepModel",
    "StepSolid",
    "ViaSpec",
    "board_body_contact",
    "board_occupancy",
    "board_stackup",
    "board_thermal_mesh",
    "board_vias",
    "body_heat_sources",
    "body_thermal_mesh",
    "box_solid",
    "cylinder_solid",
    "load_step",
    "ocp_available",
    "plane_opt_problem_mapping",
    "rasterize_board",
    "resolve_bodies",
    "sample_plane_fill",
    "sample_volume_fill",
    "synthetic_model",
    "voxel_grid",
    "voxelize_bodies",
    "write_step",
]
