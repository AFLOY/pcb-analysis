"""The CPU solve's path when the preconditioner is ``auto``: MUMPS+near, SuperLU+near, SuperLU+block, SciPy."""

from __future__ import annotations

import warnings

import pytest

from electrical import backend as _backend
from electrical.sheet_peec import current_field_contract as cfc
from electrical.sheet_peec import sheet_peec as sp_module
from electrical.sheet_peec.near_factor import mumps_available

from test_sheet_peec_edge_strands import _strip_pair


def _geometry(settings: dict, frequency: float = 3e7, faces: bool = False):
    value = _strip_pair(1.2, 4.0, 0.2, frequency)
    if faces:
        value["copper_faces_by_layer"] = {"F.Cu": {"x": [[3, 2, 0.5, 0.1]], "y": []}}
    problem = cfc.CurrentFieldProblem.from_mapping(value)
    return cfc._sheet_geometry(problem, {"maximum_filaments": 1, **settings})


def _core_path(name: str) -> str:
    return "scipy" if _backend.core() is None else name


def test_superlu_takes_near_with_strands_below_the_size_limit() -> None:
    _mesh, _operator, geometry = _geometry({"near_factorization": "superlu"})
    assert geometry.preconditioner == "near"
    assert geometry.solver_path == _core_path("superlu+near")
    assert geometry.edge_strands


def test_superlu_takes_block_without_strands_or_faces_from_the_size_limit(monkeypatch) -> None:
    monkeypatch.setattr(sp_module, "AUTO_BLOCK_FROM_UNKNOWNS", 10)
    with pytest.warns(RuntimeWarning, match="block preconditioner"):
        _mesh, _operator, geometry = _geometry({"near_factorization": "superlu"}, faces=True)
    assert geometry.preconditioner == "block"
    assert geometry.solver_path == _core_path("superlu+block")
    assert not geometry.edge_strands and not geometry.face_coverage
    # and the solve takes that path (a stranded mesh would refuse block)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = cfc.solve_current_field_problem(
            _strip_pair(1.2, 4.0, 0.2, 3e7), {"maximum_filaments": 1, "near_factorization": "superlu"}
        )
    assert result.metrics["solver_path"] == _core_path("superlu+block")
    assert result.metrics["edge_strands"] is False


@pytest.mark.skipif(not mumps_available(), reason="the C++ core is not linked against MUMPS")
def test_mumps_takes_near_at_every_size(monkeypatch) -> None:
    monkeypatch.setattr(sp_module, "AUTO_BLOCK_FROM_UNKNOWNS", 10)
    _mesh, _operator, geometry = _geometry({"near_factorization": "mumps_quasi_definite"})
    assert geometry.preconditioner == "near"
    assert geometry.solver_path == "mumps+near"
    assert geometry.edge_strands


def test_the_device_and_an_explicit_preconditioner_choose_for_themselves() -> None:
    assert cfc.solver_path(_geometry({})[0], {"execution_backend": "cuda"}) == (None, None)
    assert cfc.solver_path(_geometry({})[0], {"preconditioner": "near"}) == (None, None)


def test_an_explicit_block_preconditioner_leaves_face_coverage_out() -> None:
    with pytest.warns(RuntimeWarning, match="face coverage left out"):
        _mesh, _operator, geometry = _geometry({"preconditioner": "block"}, faces=True)
    assert not geometry.face_coverage and not geometry.edge_strands
