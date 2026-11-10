"""The quasi-definite form of the near preconditioner, checked on the CPU.

The device test (``test_sheet_cuda_device.py``) solves with cuDSS; this one
checks the matrix it is handed: complex symmetric, the saddle-point system
with the node rows negated, and close to it once the regularization is small.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from electrical.sheet_peec.sheet_cuda import quasi_definite_near_matrix


def _near_system(seed: int = 0):
    rng = np.random.default_rng(seed)
    branches, nodes = 40, 15
    # a symmetric near impedance with a positive real diagonal
    coupling = sp.random(branches, branches, density=0.1, random_state=seed) * 1e-5
    near = sp.diags(1e-3 + 1e-4 * rng.random(branches)) + 1j * 2e-3 * (coupling + coupling.T + sp.identity(branches))
    reduced = sp.random(branches, nodes, density=0.2, random_state=seed + 1, data_rvs=lambda n: rng.choice([-1.0, 1.0], n))
    reduced = sp.csr_matrix(reduced + sp.eye(branches, nodes))
    return sp.csr_matrix(near), reduced


def test_the_quasi_definite_near_matrix_is_the_saddle_point_one_signed_and_regularized() -> None:
    near, reduced = _near_system()
    matrix, sign, delta = quasi_definite_near_matrix(near, reduced, 1e-8)
    assert abs(matrix - matrix.T).max() == 0.0  # complex symmetric
    assert delta == pytest.approx(1e-8 / np.median(np.abs(near.diagonal())))
    saddle = sp.bmat([[near, -reduced], [reduced.T, None]], format="csc")
    # its solve, signs undone, is the saddle-point solve to about the regularization
    rng = np.random.default_rng(3)
    rhs = rng.standard_normal(matrix.shape[0]) + 1j * rng.standard_normal(matrix.shape[0])
    exact = spl.spsolve(saddle, rhs)
    regularized = spl.spsolve(sp.csc_matrix(matrix), rhs * sign)
    assert np.linalg.norm(regularized - exact) / np.linalg.norm(exact) < 1e-5
    # and a large regularization is a different system
    loose, sign_loose, _ = quasi_definite_near_matrix(near, reduced, 1.0)
    assert np.linalg.norm(spl.spsolve(sp.csc_matrix(loose), rhs * sign_loose) - exact) / np.linalg.norm(exact) > 1e-3
