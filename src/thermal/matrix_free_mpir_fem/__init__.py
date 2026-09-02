"""Matrix-free thermal FEM accelerated by mixed-precision iterative refinement.

The front end covers steady heat conduction in a layered PCB stack with
convective faces and fixed-temperature nodes.  The MPIR solver and the
low-precision runtimes are shared with ``electrical.matrix_free_mpir_fem``.
"""

from .conduction import (
    COPPER_THERMAL_CONDUCTIVITY_W_PER_M_K,
    FR4_IN_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K,
    FR4_THROUGH_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K,
    ConvectionBoundary,
    HeatSource,
    LayeredThermalMesh,
    MatrixFreeThermalOperator,
    ThermalConductionProblem,
    ThermalConductionSolution,
    solve_thermal_conduction,
)
from .coupling import element_joule_heat_w, via_joule_heat_sources
from .two_level import AggregationCoarseCorrection, choose_block_size

__all__ = [
    "AggregationCoarseCorrection",
    "COPPER_THERMAL_CONDUCTIVITY_W_PER_M_K",
    "ConvectionBoundary",
    "FR4_IN_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K",
    "FR4_THROUGH_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K",
    "HeatSource",
    "LayeredThermalMesh",
    "MatrixFreeThermalOperator",
    "ThermalConductionProblem",
    "ThermalConductionSolution",
    "choose_block_size",
    "element_joule_heat_w",
    "solve_thermal_conduction",
    "via_joule_heat_sources",
]
