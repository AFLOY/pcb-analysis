"""DICE-PEEC acceleration for single- and multilayer electrical PCB analysis.

There are two solvers here and they answer different questions.

``cuda_pypeec`` runs PyPEEC's own three-dimensional voxel solve on CUDA.  It is
the high-fidelity path: PyPEEC owns the physics, this owns the device policy,
and ``pypeec_memory`` predicts what a model will cost before it is attempted --
which matters once a model spans a board's height rather than one copper layer.

``sheet_peec`` solves the same physics on a mesh built for what a PCB actually
is: a few thin sheets at known heights.  Its inductance operator is a
two-dimensional transform per layer pair instead of a three-dimensional one
over the board's height, which is a large saving on a board of few layers.  It
is exact in the sense that matters -- at zero frequency it reproduces a
resistor network to machine precision, and above it, the transform path matches
a dense assembly of the same operator.

Separately, ``multilayer_peec`` holds a scalar interaction proxy for ranking
many candidate shapes cheaply.  It does not solve for current or potential and
is not a substitute for either solver above.
"""

from .layout_ops import (
    CandidateEdit,
    CompiledCandidate,
    SegmentOp,
    ViaOp,
    compile_candidate,
    compile_many,
)
from .multilayer_peec import (
    FFTInteraction25D,
    MultilayerDeltaScorer,
    SparseDeltaML,
    ViaSet,
    ViaSpec,
)
from .plane_opt_contract import (
    PLANE_OPT_PROBLEM_SCHEMA,
    PLANE_OPT_RESULT_SCHEMA,
    PlaneOptProblem,
    PlaneOptSolveResult,
    build_plane_opt_sheet_inputs,
    solve_plane_opt_problem,
)
from .sheet_inductance import (
    CellGeometry,
    build_kernel,
    mutual_partial_inductance,
    self_partial_inductance,
)
from .sheet_cuda import (
    CudaSheetSolveError,
    CudaSheetTelemetry,
    CudaSheetUnavailableError,
    solve_sheet_case_cuda,
)
from .sheet_operator import SheetInductanceOperator, SheetLayer, SheetStackup
from .sheet_results import (
    SheetFields,
    cell_current_density,
    cell_current_density_phasor,
    sheet_fields,
    vertical_currents,
)
from .sheet_peec import (
    SheetMesh,
    SheetSolution,
    Terminal,
    ViaBranch,
    solve_sheet_case,
    via_resistance,
)
from .stackup import Stackup
from .voxel_peec import (
    VoxelConductorProblem,
    VoxelPeecSolution,
    VoxelTerminal,
    VoxelTerminalResult,
    build_pypeec_inputs,
    solve_voxel_peec,
)

__all__ = [
    "CandidateEdit",
    "CellGeometry",
    "CompiledCandidate",
    "CudaSheetSolveError",
    "CudaSheetTelemetry",
    "CudaSheetUnavailableError",
    "FFTInteraction25D",
    "MultilayerDeltaScorer",
    "PLANE_OPT_PROBLEM_SCHEMA",
    "PLANE_OPT_RESULT_SCHEMA",
    "PlaneOptProblem",
    "PlaneOptSolveResult",
    "SegmentOp",
    "SheetFields",
    "SheetInductanceOperator",
    "SheetLayer",
    "SheetMesh",
    "SheetSolution",
    "SheetStackup",
    "SparseDeltaML",
    "Stackup",
    "VoxelConductorProblem",
    "VoxelPeecSolution",
    "VoxelTerminal",
    "VoxelTerminalResult",
    "build_pypeec_inputs",
    "solve_voxel_peec",
    "Terminal",
    "ViaBranch",
    "ViaOp",
    "ViaSet",
    "ViaSpec",
    "build_kernel",
    "cell_current_density",
    "cell_current_density_phasor",
    "compile_candidate",
    "compile_many",
    "build_plane_opt_sheet_inputs",
    "mutual_partial_inductance",
    "self_partial_inductance",
    "sheet_fields",
    "solve_sheet_case",
    "solve_sheet_case_cuda",
    "solve_plane_opt_problem",
    "via_resistance",
    "vertical_currents",
]
