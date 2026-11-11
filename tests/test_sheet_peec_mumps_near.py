"""The near preconditioner factored by MUMPS (``near_factorization='mumps_quasi_definite'``).

Runs where the C++ core is built with ``-DPCB_NATIVE_MUMPS=ON``; the refusal
without it is checked everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from electrical.sheet_peec import near_factor
from electrical.sheet_peec.near_factor import MumpsQuasiDefiniteNear, mumps_available

needs_mumps = pytest.mark.skipif(not mumps_available(), reason="the C++ core is not linked against MUMPS")


def _near_system(seed: int = 0):
    rng = np.random.default_rng(seed)
    branches, nodes = 40, 15
    coupling = sp.random(branches, branches, density=0.1, random_state=seed) * 1e-5
    near = sp.diags(1e-3 + 1e-4 * rng.random(branches)) + 1j * 2e-3 * (coupling + coupling.T + sp.identity(branches))
    reduced = sp.random(branches, nodes, density=0.2, random_state=seed + 1, data_rvs=lambda n: rng.choice([-1.0, 1.0], n))
    reduced = sp.csr_matrix(reduced + sp.eye(branches, nodes))
    return sp.csr_matrix(near), reduced


@needs_mumps
def test_the_mumps_factor_solves_the_near_saddle_point_matrix() -> None:
    near, reduced = _near_system()
    saddle = sp.bmat([[near, -reduced], [reduced.T, None]], format="csc")
    rng = np.random.default_rng(3)
    vector = rng.standard_normal(saddle.shape[0]) + 1j * rng.standard_normal(saddle.shape[0])
    block = rng.standard_normal((saddle.shape[0], 3)) + 1j * rng.standard_normal((saddle.shape[0], 3))
    factor = MumpsQuasiDefiniteNear(near, reduced, 1e-12, threads=2)
    try:
        assert np.allclose(factor.solve(vector), spl.spsolve(saddle, vector), rtol=1e-7, atol=0)
        solved = factor.solve(block)
        assert solved.shape == block.shape
        for column in range(block.shape[1]):
            assert np.allclose(solved[:, column], spl.spsolve(saddle, block[:, column]), rtol=1e-7, atol=0)
    finally:
        factor.close()


@needs_mumps
def test_a_mesh_s_next_frequency_reuses_the_analysis() -> None:
    near_factor.clear_near_factor_cache()
    near, reduced = _near_system()
    first = MumpsQuasiDefiniteNear(near, reduced)
    assert not first.reused_analysis
    first.close()
    second = MumpsQuasiDefiniteNear(near.real + 2j * near.imag, reduced)
    assert second.reused_analysis
    # A factor handed out once is not handed out again while it is in use.
    third = MumpsQuasiDefiniteNear(near, reduced)
    assert not third.reused_analysis
    second.close()
    third.close()
    with pytest.raises(RuntimeError):
        second.solve(np.zeros(second.size, dtype=complex))
    near_factor.clear_near_factor_cache()


@needs_mumps
@pytest.mark.parametrize("cases", [1, 3])
def test_solving_with_the_mumps_factor_gives_the_superlu_solution(cases: int) -> None:
    from electrical.sheet_peec import current_field_contract as cfc

    from test_current_field_contract import _problem

    value = _problem(frequency_hz=3e7)
    settings = {"maximum_filaments": 64, "preconditioner": "near", "relative_tolerance": 1e-11}
    problems = [value] * cases
    reference = cfc.solve_current_field_problems(problems, settings)
    mumps = cfc.solve_current_field_problems(problems, dict(settings, near_factorization="mumps_quasi_definite"))
    for left, right in zip(reference, mumps):
        assert right.metrics["near_factorization"] == "mumps_quasi_definite"
        assert right.metrics["converged"]
        keys = list(left.voltage_phasor)
        a = np.array([left.voltage_phasor[k] for k in keys])
        b = np.array([right.voltage_phasor[k] for k in keys])
        assert np.max(np.abs(a - b)) <= 1e-8 * np.max(np.abs(a))


def test_mumps_is_refused_where_the_core_lacks_it(monkeypatch: pytest.MonkeyPatch) -> None:
    from electrical.sheet_peec import sheet_peec

    monkeypatch.setattr(sheet_peec, "mumps_available", lambda: False)
    with pytest.raises(ValueError, match="PCB_NATIVE_MUMPS"):
        sheet_peec.solve_sheet_cases(None, None, [], near_factorization="mumps_quasi_definite")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="near_factorization"):
        sheet_peec.solve_sheet_cases(None, None, [], near_factorization="pardiso")  # type: ignore[arg-type]
