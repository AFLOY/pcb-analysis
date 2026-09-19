"""Matrix-free FEM accelerated by mixed-precision iterative refinement.

The physical front ends cover layered-PCB DC conduction and scalar-polarised
2D frequency-domain Maxwell fields.  The MPIR solver and low-precision runtime
contracts remain independent of either front end.
"""

from .grid import TensorGrid, graded_edges, refined_grid
from .pcb import (
    COPPER_CONDUCTIVITY_S_PER_M,
    CurrentTerminal,
    LayeredPCBMesh,
    MatrixFreePCBOperator,
    PCBConductionProblem,
    PCBConductionSolution,
    ViaConnection,
    solve_pcb_dc,
)
from .frequency_domain import (
    EPSILON_0_F_PER_M,
    MU_0_H_PER_M,
    MatrixFreeScalarMaxwellOperator,
    ScalarMaxwellMesh2D,
    ScalarMaxwellProblem,
    ScalarMaxwellSolution,
    propagation_constant_per_m,
    skin_depth_m,
    solve_scalar_maxwell,
)
from .runtime import (
    CupyComplex64Runtime,
    CupyFloat32Runtime,
    LowPrecisionRuntime,
    NumpyComplex64Runtime,
    NumpyFloat32Runtime,
    RuntimeBackend,
    cuda_available,
    make_complex64_runtime,
    make_float32_runtime,
)
from .solver import (
    MPIRConfig,
    MPIRResult,
    MPIRStep,
    MatrixFreeMPIRSystem,
    solve_mpir,
)

__all__ = [
    "COPPER_CONDUCTIVITY_S_PER_M",
    "CupyComplex64Runtime",
    "CupyFloat32Runtime",
    "CurrentTerminal",
    "EPSILON_0_F_PER_M",
    "LayeredPCBMesh",
    "LowPrecisionRuntime",
    "MPIRConfig",
    "MPIRResult",
    "MPIRStep",
    "MatrixFreeMPIRSystem",
    "MatrixFreePCBOperator",
    "MatrixFreeScalarMaxwellOperator",
    "MU_0_H_PER_M",
    "NumpyComplex64Runtime",
    "NumpyFloat32Runtime",
    "PCBConductionProblem",
    "PCBConductionSolution",
    "ScalarMaxwellMesh2D",
    "ScalarMaxwellProblem",
    "ScalarMaxwellSolution",
    "RuntimeBackend",
    "TensorGrid",
    "ViaConnection",
    "propagation_constant_per_m",
    "refined_grid",
    "cuda_available",
    "graded_edges",
    "make_complex64_runtime",
    "make_float32_runtime",
    "skin_depth_m",
    "solve_mpir",
    "solve_pcb_dc",
    "solve_scalar_maxwell",
]
