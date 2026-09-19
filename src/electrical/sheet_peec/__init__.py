"""Sheet PEEC: the 2.5D electrical solver of this repository.

A PCB is a few thin conductive sheets at known heights.  The inductance
operator is a two-dimensional transform per layer pair rather than a
three-dimensional one over the board's height; at zero frequency the solve
reproduces a resistor network to machine precision and above it the transform
path matches a dense assembly of the same operator.  The thickness of each
sheet is cut into graded filaments for the skin effect, and
``plane_opt_contract`` is the serialised problem/result schema the router
speaks.  A CUDA path (``sheet_cuda``) mirrors the CPU solve.
"""

from .plane_opt_contract import (
    PLANE_OPT_PROBLEM_SCHEMA,
    PLANE_OPT_RESULT_SCHEMA,
    PlaneOptProblem,
    PlaneOptSolveResult,
    build_plane_opt_sheet_inputs,
    solve_plane_opt_problem,
)
from .sheet_cuda import (
    CudaSheetSolveError,
    CudaSheetTelemetry,
    CudaSheetUnavailableError,
    solve_sheet_case_cuda,
)
from .sheet_inductance import (
    CellGeometry,
    build_kernel,
    mutual_partial_inductance,
    self_partial_inductance,
)
from .sheet_operator import SheetInductanceOperator, SheetLayer, SheetStackup
from .sheet_peec import (
    SheetMesh,
    SheetSolution,
    Terminal,
    ViaBranch,
    solve_sheet_case,
    via_resistance,
)
from .sheet_results import (
    SheetFields,
    cell_current_density,
    cell_current_density_phasor,
    sheet_fields,
    vertical_currents,
)
from .skin_filaments import skin_depth_m
from .skin_screen import LayerSkin, classify_layer_thickness

__all__ = [
    "CellGeometry",
    "CudaSheetSolveError",
    "CudaSheetTelemetry",
    "CudaSheetUnavailableError",
    "LayerSkin",
    "PLANE_OPT_PROBLEM_SCHEMA",
    "PLANE_OPT_RESULT_SCHEMA",
    "PlaneOptProblem",
    "PlaneOptSolveResult",
    "SheetFields",
    "SheetInductanceOperator",
    "SheetLayer",
    "SheetMesh",
    "SheetSolution",
    "SheetStackup",
    "Terminal",
    "ViaBranch",
    "build_kernel",
    "build_plane_opt_sheet_inputs",
    "cell_current_density",
    "cell_current_density_phasor",
    "classify_layer_thickness",
    "mutual_partial_inductance",
    "self_partial_inductance",
    "sheet_fields",
    "skin_depth_m",
    "solve_plane_opt_problem",
    "solve_sheet_case",
    "solve_sheet_case_cuda",
    "via_resistance",
    "vertical_currents",
]
