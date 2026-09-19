"""3D voxel PEEC through PyPEEC.

PyPEEC owns the voxel assembly and the coupled solve; this package owns the
array contract (``contract``: conductor mask, resistivity, lumped terminals,
fields and losses back), the CUDA execution policy and telemetry
(``cuda_pypeec``) and the memory prediction that decides whether a model fits
a device (``pypeec_memory``).  It is the high-fidelity path for conductors a
2.5D sheet cannot represent and for the promoted candidates of the DICE
search.
"""

from .contract import (
    VoxelConductorProblem,
    VoxelPeecSolution,
    VoxelTerminal,
    VoxelTerminalResult,
    build_pypeec_inputs,
    default_tolerance,
    solve_voxel_peec,
)
from .cuda_pypeec import (
    CudaPeecConfig,
    CudaPeecMemoryError,
    CudaPeecResult,
    CudaPeecSolveError,
    CudaPyPeecExecutor,
    CudaUnavailableError,
)
from .pypeec_memory import MemoryEstimate, describe_estimate, estimate_pypeec_memory

__all__ = [
    "CudaPeecConfig",
    "CudaPeecMemoryError",
    "CudaPeecResult",
    "CudaPeecSolveError",
    "CudaPyPeecExecutor",
    "CudaUnavailableError",
    "MemoryEstimate",
    "VoxelConductorProblem",
    "VoxelPeecSolution",
    "VoxelTerminal",
    "VoxelTerminalResult",
    "build_pypeec_inputs",
    "default_tolerance",
    "describe_estimate",
    "estimate_pypeec_memory",
    "solve_voxel_peec",
]
