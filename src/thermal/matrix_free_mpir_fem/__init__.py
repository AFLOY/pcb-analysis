"""Matrix-free thermal FEM accelerated by mixed-precision iterative refinement.

The front end covers steady heat conduction in a layered PCB stack with
convective faces and fixed-temperature nodes, and in voxelised 3D bodies
(heat sinks, enclosures) through an active-element mask on the same mesh.  The MPIR solver and the
low-precision runtimes are shared with ``electrical.matrix_free_mpir_fem``.
"""

from .boundaries import ConvectionBoundary, ExposedFaceConvection, HeatSource
from .mesh import (
    COPPER_THERMAL_CONDUCTIVITY_W_PER_M_K,
    FACE_DIRECTIONS,
    FR4_IN_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K,
    FR4_THROUGH_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K,
    LayeredThermalMesh,
    exposed_element_faces,
)
from .operator import MatrixFreeThermalOperator
from .problem import ThermalConductionProblem
from .radiation import (
    STEFAN_BOLTZMANN_W_PER_M2_K4,
    ExposedFaceRadiation,
    RadiationBoundary,
    element_temperature_k,
    face_temperature_k,
    newton_linearisation,
)
from .solve import ThermalConductionSolution, solve_thermal_conduction
from .transient import TimeSchedule, TransientStep, TransientThermalSolution, solve_thermal_transient
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
    "ExposedFaceRadiation",
    "FACE_DIRECTIONS",
    "FR4_IN_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K",
    "FR4_THROUGH_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K",
    "HeatSource",
    "LayeredThermalMesh",
    "MatrixFreeThermalOperator",
    "RadiationBoundary",
    "STEFAN_BOLTZMANN_W_PER_M2_K4",
    "ThermalConductionProblem",
    "ThermalConductionSolution",
    "TimeSchedule",
    "TransientStep",
    "TransientThermalSolution",
    "VoxelMaterial",
    "VoxelSolidModel",
    "VoxelThermalMesh",
    "choose_block_size",
    "element_joule_heat_w",
    "element_temperature_k",
    "face_temperature_k",
    "newton_linearisation",
    "exposed_element_faces",
    "planar_contact_map",
    "solve_thermal_conduction",
    "solve_thermal_transient",
    "via_joule_heat_sources",
]
