"""The sheet solve's residual is weighted so that ``tolerance`` bounds the currents' relative error."""

from __future__ import annotations

import numpy as np
import pytest

from electrical.sheet_peec.current_field_contract import build_current_field_sheet_inputs
from electrical.sheet_peec.sheet_peec import solve_sheet_case

from test_sheet_pfft import _graded_problem


@pytest.mark.parametrize("preconditioner", ["near", "block", "diagonal"])
def test_the_residual_bounds_the_currents_error(preconditioner) -> None:
    # The near preconditioner leaves a loop current, which keeps KCL and shows in
    # an unweighted residual only as Z di (Z about 1e-3 ohm here): its currents
    # were 14 times the residual off on the CPU, 66 to 150 times with cuDSS.
    mesh, operator, terminals, _context = build_current_field_sheet_inputs(_graded_problem(1.0e6))
    reference = solve_sheet_case(mesh, operator, terminals, frequency_hz=1.0e6, tolerance=1e-13)
    result = solve_sheet_case(
        mesh, operator, terminals, frequency_hz=1.0e6, tolerance=1e-9, preconditioner=preconditioner
    )
    assert reference.converged and result.converged and result.residual <= 1e-9
    peak = np.abs(reference.branch_current).max()
    error = np.abs(result.branch_current - reference.branch_current).max()
    assert error <= 2.0 * max(result.residual, 1e-13) * peak
