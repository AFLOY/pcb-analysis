"""Matrix-free thermal FEM accelerated by mixed-precision iterative refinement.

The front end covers steady heat conduction in a layered PCB stack with
convective faces and fixed-temperature nodes, and in voxelised 3D bodies
(heat sinks, enclosures) through an active-element mask on the same mesh.  The MPIR solver and the
low-precision runtimes are shared with ``electrical.matrix_free_mpir_fem``.
"""

from .conduction import (
    COPPER_THERMAL_CONDUCTIVITY_W_PER_M_K,
    FR4_IN_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K,
    FR4_THROUGH_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K,
    FACE_DIRECTIONS,
    ConvectionBoundary,
    ExposedFaceConvection,
    HeatSource,
    LayeredThermalMesh,
    MatrixFreeThermalOperator,
    ThermalConductionProblem,
    ThermalConductionSolution,
    exposed_element_faces,
    solve_thermal_conduction,
)
from .contact import ContactMap, planar_contact_map
from .coupling import element_joule_heat_w, via_joule_heat_sources
from .two_level import AggregationCoarseCorrection, choose_block_size
from .voxel import VoxelMaterial, VoxelSolidModel, VoxelThermalMesh

__all__ = [
    "AggregationCoarseCorrection",
    "COPPER_THERMAL_CONDUCTIVITY_W_PER_M_K",
    "ContactMap",
    "ConvectionBoundary",
    "ExposedFaceConvection",
    "FACE_DIRECTIONS",
    "FR4_IN_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K",
    "FR4_THROUGH_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K",
    "HeatSource",
    "LayeredThermalMesh",
    "MatrixFreeThermalOperator",
    "ThermalConductionProblem",
    "ThermalConductionSolution",
    "VoxelMaterial",
    "VoxelSolidModel",
    "VoxelThermalMesh",
    "choose_block_size",
    "element_joule_heat_w",
    "exposed_element_faces",
    "planar_contact_map",
    "solve_thermal_conduction",
    "via_joule_heat_sources",
]
