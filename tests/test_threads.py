"""The process-wide thread budget: the one threading control of pcb-analysis."""

from __future__ import annotations

import os

import numpy as np
import pytest
from threadpoolctl import threadpool_info

import electrical
from electrical.threads import (
    available_threads,
    set_thread_budget,
    thread_budget,
    serial_blas,
    thread_budget_scope,
)
from electrical.voxel_peec import default_tolerance


@pytest.fixture(autouse=True)
def _default_budget():
    set_thread_budget(None)
    yield
    set_thread_budget(None)


def _blas_threads() -> list[int]:
    return [pool["num_threads"] for pool in threadpool_info() if pool["user_api"] == "blas"]


def test_default_budget_is_the_physical_cores() -> None:
    """Hyper-threads are not part of the default: one thread per physical core."""
    cpus = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else set()
    cores = set()
    for cpu in cpus:
        topology = f"/sys/devices/system/cpu/cpu{cpu}/topology/"
        try:
            cores.add((open(topology + "physical_package_id").read(), open(topology + "core_id").read()))
        except OSError:
            cores = set()
            break
    expected = len(cores) or len(cpus) or (os.cpu_count() or 1)
    assert available_threads() == expected
    assert available_threads() <= (len(cpus) or expected)
    assert thread_budget() == available_threads()


def test_the_default_budget_also_limits_the_blas_pools() -> None:
    np.dot(np.ones((4, 4)), np.ones((4, 4)))
    pools = _blas_threads()
    if not pools:
        pytest.skip("no BLAS library visible to threadpoolctl")
    set_thread_budget(None)
    assert _blas_threads() == [available_threads()] * len(pools)


def test_the_package_exports_the_budget_and_no_other_thread_control() -> None:
    for name in ("available_threads", "set_thread_budget", "thread_budget", "thread_budget_scope"):
        assert name in electrical.__all__
    assert electrical.thread_budget is thread_budget
    from electrical import matrix_free_mpir_fem

    assert "port_basis_workers" not in matrix_free_mpir_fem.__all__


def test_set_and_restore() -> None:
    set_thread_budget(3)
    assert thread_budget() == 3
    set_thread_budget(1)
    assert thread_budget() == 1
    set_thread_budget(None)
    assert thread_budget() == available_threads()


def test_scope_restores_the_previous_budget_also_on_exception() -> None:
    set_thread_budget(5)
    with thread_budget_scope(2) as inside:
        assert inside == 2 and thread_budget() == 2
        with thread_budget_scope(1):
            assert thread_budget() == 1
        assert thread_budget() == 2
    assert thread_budget() == 5
    with pytest.raises(RuntimeError, match="boom"):
        with thread_budget_scope(3):
            assert thread_budget() == 3
            raise RuntimeError("boom")
    assert thread_budget() == 5
    set_thread_budget(None)
    with thread_budget_scope(2):
        pass
    assert thread_budget() == available_threads()


@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_budgets_are_rejected(value: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        set_thread_budget(value)
    with pytest.raises(ValueError, match="at least 1"):
        with thread_budget_scope(value):
            pass
    assert thread_budget() == available_threads()


@pytest.mark.parametrize("value", [2.0, "2", True])
def test_non_integer_budgets_are_rejected(value: object) -> None:
    with pytest.raises(TypeError):
        set_thread_budget(value)  # type: ignore[arg-type]
    assert thread_budget() == available_threads()


def test_budget_limits_the_blas_pools_and_restores_them() -> None:
    np.dot(np.ones((4, 4)), np.ones((4, 4)))  # make sure NumPy's BLAS is loaded
    original = _blas_threads()
    if not original:
        pytest.skip("no BLAS library visible to threadpoolctl")
    set_thread_budget(1)
    assert _blas_threads() == [1] * len(original)
    set_thread_budget(2)  # a new call replaces the limit
    assert _blas_threads() == [2] * len(original)
    set_thread_budget(None)
    assert _blas_threads() == original
    with thread_budget_scope(1):
        assert _blas_threads() == [1] * len(original)
    assert _blas_threads() == original


def test_pypeec_fft_workers_follow_the_budget() -> None:
    with thread_budget_scope(3):
        tolerance = default_tolerance()
    assert tolerance["dense_options"]["fft_options"]["scipy_worker"] == 3
    with pytest.raises(ValueError, match="set_thread_budget"):
        default_tolerance({"scipy_workers": 4})


def _openmp_threads() -> list[int]:
    return [pool["num_threads"] for pool in threadpool_info() if pool["user_api"] == "openmp"]


def test_serial_blas_holds_blas_at_one_and_keeps_openmp_at_the_budget() -> None:
    np.dot(np.ones((4, 4)), np.ones((4, 4)))
    pools = _blas_threads()
    if not pools:
        pytest.skip("no BLAS library visible to threadpoolctl")
    set_thread_budget(2)
    with serial_blas():
        assert _blas_threads() == [1] * len(pools)
        assert all(threads == 2 for threads in _openmp_threads())
        with serial_blas():
            assert _blas_threads() == [1] * len(pools)
        assert _blas_threads() == [1] * len(pools)  # the outer block still holds it
        set_thread_budget(3)  # a budget change inside keeps BLAS serial
        assert _blas_threads() == [1] * len(pools) and thread_budget() == 3
    assert _blas_threads() == [3] * len(pools)
    with pytest.raises(RuntimeError, match="boom"):
        with serial_blas():
            raise RuntimeError("boom")
    assert _blas_threads() == [3] * len(pools)


def test_the_sheet_solve_runs_its_krylov_iterations_on_serial_blas(monkeypatch) -> None:
    import electrical.sheet_peec.sheet_peec as sheet_peec
    from electrical.sheet_peec.sheet_operator import SheetInductanceOperator, SheetLayer, SheetStackup

    np.dot(np.ones((4, 4)), np.ones((4, 4)))
    if not _blas_threads():
        pytest.skip("no BLAS library visible to threadpoolctl")
    set_thread_budget(2)
    seen = []
    original = sheet_peec.spla.gmres

    def recording_gmres(*args, **kwargs):
        seen.append(_blas_threads())
        return original(*args, **kwargs)

    monkeypatch.setattr(sheet_peec.spla, "gmres", recording_gmres)
    rows, cols, pitch = 5, 7, 2e-4
    stackup = SheetStackup((SheetLayer("F.Cu", 0.0, 3.5e-5, 1.724e-8),))
    mesh = sheet_peec.SheetMesh((rows, cols), pitch, stackup, np.ones((1, rows, cols), dtype=bool))
    operator = SheetInductanceOperator((rows, cols), pitch, stackup)
    terminals = [
        sheet_peec.Terminal("in", 0, tuple((row, 0) for row in range(rows)), 1.0),
        sheet_peec.Terminal("out", 0, tuple((row, cols - 1) for row in range(rows)), -1.0),
    ]
    # The NumPy reference is the path that calls SciPy's GMRES (and BLAS);
    # the C++ core runs no BLAS at all.
    from electrical import _backend

    with _backend.use_reference():
        sheet_peec.solve_sheet_case(mesh, operator, terminals, frequency_hz=1e6)
    assert seen and all(threads == [1] * len(threads) for threads in seen)
    assert _blas_threads() == [2] * len(seen[0])
