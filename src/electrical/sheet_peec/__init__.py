"""Sheet PEEC: the 2.5D electrical solver of this repository.

A PCB is a few thin conductive sheets at known heights.  The inductance
operator is a two-dimensional transform per layer pair rather than a
three-dimensional one over the board's height; at zero frequency the solve
reproduces a resistor network to machine precision and above it the transform
path matches a dense assembly of the same operator.  The thickness of each
sheet is cut into graded filaments for the skin effect, and
``current_field_contract`` is the serialised problem/result schema a router
speaks.  A CUDA path (``sheet_cuda``) mirrors the CPU solve.
"""

from .current_field_contract import (
    CURRENT_FIELD_PROBLEM_SCHEMA,
    CURRENT_FIELD_RESULT_SCHEMA,
    CurrentFieldProblem,
    CurrentFieldSolveResult,
    build_current_field_sheet_inputs,
    solve_current_field_problem,
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
    "CURRENT_FIELD_PROBLEM_SCHEMA",
    "CURRENT_FIELD_RESULT_SCHEMA",
    "CellGeometry",
    "CudaSheetSolveError",
    "CudaSheetTelemetry",
    "CudaSheetUnavailableError",
    "CurrentFieldProblem",
    "CurrentFieldSolveResult",
    "LayerSkin",
    "SheetFields",
    "SheetInductanceOperator",
    "SheetLayer",
    "SheetMesh",
    "SheetSolution",
    "SheetStackup",
    "Terminal",
    "ViaBranch",
    "build_kernel",
    "build_current_field_sheet_inputs",
    "cell_current_density",
    "cell_current_density_phasor",
    "classify_layer_thickness",
    "mutual_partial_inductance",
    "self_partial_inductance",
    "sheet_fields",
    "skin_depth_m",
    "solve_current_field_problem",
    "solve_sheet_case",
    "solve_sheet_case_cuda",
    "via_resistance",
    "vertical_currents",
]
