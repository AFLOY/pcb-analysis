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
