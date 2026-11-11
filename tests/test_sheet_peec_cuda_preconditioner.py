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


def test_the_batched_layer_pair_product_is_the_pairwise_convolution() -> None:
    """The stacked (F, L, L) product gives what the loop over layer pairs gives.

    CuPy and NumPy share the calls the device operator makes, so NumPy stands
    in for the device here and the two paths run on the same arrays.
    """
    from electrical.sheet_peec.current_field_contract import CurrentFieldProblem, _sheet_geometry
    from electrical.sheet_peec.sheet_cuda import CudaSheetInductanceOperator

    from test_current_field_contract import _problem

    problem = CurrentFieldProblem.from_mapping(_problem(frequency_hz=3e7))
    _mesh, operator, _geometry = _sheet_geometry(problem, {"maximum_filaments": 8})
    device = CudaSheetInductanceOperator(operator, np)
    assert device._stack is not None and device.layer_count > 3  # the thick layer is cut
    rng = np.random.default_rng(7)
    rows, cols = device.shape
    x = rng.standard_normal((2, device.layer_count, rows, cols))
    y = rng.standard_normal((2, device.layer_count, rows, cols))
    z = rng.standard_normal((2, len(device.vertical_levels), rows, cols))
    batched = device.apply_batch(x, y, z)
    for b in range(2):
        single = _pairwise(device, x[b], y[b], z[b])
        for got, want in zip(batched, single):
            assert np.allclose(got[b], want, rtol=1e-12, atol=1e-12 * np.abs(want).max())


def _pairwise(device, x, y, z):
    """The operator by its loop over layer pairs (the path the stack replaces)."""
    spectra = lambda a: np.fft.rfft2(a, s=device.padded, axes=(-2, -1))  # noqa: E731
    return device._convolve(spectra(x), "x"), device._convolve(spectra(y), "y"), device._convolve_z(spectra(z))


@pytest.mark.parametrize("filaments, barrels", [(1, False), (64, False), (64, True)])
def test_the_device_near_matrix_is_the_host_one(filaments: int, barrels: bool) -> None:
    """device_quasi_definite_near, run on NumPy and SciPy, gives the host path's matrix bit for bit."""
    from electrical.sheet_peec import current_field_contract as cfc
    from electrical.sheet_peec import sheet_peec as sheet
    from electrical.sheet_peec.sheet_cuda import device_quasi_definite_near

    from test_current_field_contract import _problem

    value = _problem(frequency_hz=3e7)
    if barrels:
        for connection in value["vertical_connections"]:
            connection["barrel"] = {"outer_diameter_mm": 0.3, "drill_diameter_mm": 0.25}
    problem = cfc.CurrentFieldProblem.from_mapping(value)
    mesh, operator, geometry = cfc._sheet_geometry(problem, {"maximum_filaments": filaments})
    operator = sheet._with_barrels(mesh, operator)
    assert hasattr(operator, "_delta_branch") == barrels
    terminals = cfc._sheet_terminals(problem, geometry)
    reduction = sheet._reduce(mesh, mesh.incidence(), sheet._source_vector(mesh, terminals), None)
    omega, resistance = 2 * np.pi * 3e7, mesh.resistances()
    near = sheet._near_impedance(mesh, operator, omega, resistance, reduction.active_branches)
    host, host_sign, host_delta = quasi_definite_near_matrix(near, reduction.reduced, 1e-8)
    device, device_sign, device_delta = device_quasi_definite_near(
        mesh, operator, omega, resistance, reduction.active_branches, reduction.reduced, 1e-8, np, sp
    )
    assert np.array_equal(host.indptr, device.indptr) and np.array_equal(host.indices, device.indices)
    assert np.array_equal(host.data, device.data)
    assert host_delta == device_delta and np.array_equal(host_sign, device_sign)
