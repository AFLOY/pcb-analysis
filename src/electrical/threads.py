"""The process-wide thread budget: the only threading control pcb-analysis offers.

An application states how many threads pcb-analysis may use in total, once,
with :func:`set_thread_budget` (or for a block with
:func:`thread_budget_scope`).  Every parallel path reads that one number:

* the OpenMP team of every fused C++ kernel (layered DC and scalar Maxwell
  Q1, thermal hexahedral Q1, dipole fields, point-in-solid tests, sheet pFFT
  near-field corrections) is the budget when the kernel runs alone;
* PyPEEC's SciPy FFT workers are the budget;
* the BLAS and OpenMP pools of NumPy and SciPy are limited to the budget
  (through :mod:`threadpoolctl`), except that the BLAS pools run on one
  thread while a solve that gains nothing from threaded BLAS is running
  (:func:`serial_blas`).

Where pcb-analysis nests parallelism (the thread pool over the unit solves
of an N-port basis, each solve running an OpenMP team) it splits the budget
between the levels itself, so that their product never exceeds it.  There is
no per-call, per-kernel or per-level knob and no environment variable for a
thread count; the split is an internal decision that follows measurements.

The budget is process-wide state guarded by a lock.  Set it from the main
thread before starting analyses; changing it while a solve runs on another
thread affects only operators built (and kernel calls made) after the change.  Each pcb-analysis call
may use the whole budget, so an application that runs several calls at once
on its own threads sets the budget to its total divided by that concurrency.
CUDA paths do not read it.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

__all__ = [
    "available_threads",
    "set_thread_budget",
    "thread_budget",
    "thread_budget_scope",
]

_lock = threading.RLock()
_budget: int | None = None
_pool_limit: Any = None
_serial_blas_depth = 0


def _physical_cores(cpus: set[int]) -> int | None:
    """Distinct (package, core) pairs among ``cpus``; ``None`` without sysfs topology."""

    cores = set()
    for cpu in cpus:
        topology = f"/sys/devices/system/cpu/cpu{cpu}/topology/"
        try:
            with open(topology + "physical_package_id") as package, open(topology + "core_id") as core:
                cores.add((package.read().strip(), core.read().strip()))
        except OSError:
            return None
    return len(cores) or None


def available_threads() -> int:
    """Physical cores this process may run on: the default budget.

    Hyper-threads (SMT siblings) are not counted.  The kernels here are
    floating-point and memory bound, where a sibling adds contention rather
    than throughput, and a thread count measured on physical cores is the one
    that transfers between machines.  The cores are those of the affinity
    mask, read from the sysfs topology; where that is not available the
    logical CPU count is the fallback.
    """

    try:
        cpus = os.sched_getaffinity(0)
    except (AttributeError, OSError):  # pragma: no cover - non-Linux hosts
        return os.cpu_count() or 1
    return max(1, _physical_cores(set(cpus)) or len(cpus))


def thread_budget() -> int:
    """Threads pcb-analysis may use in total; :func:`available_threads` unless set."""

    with _lock:
        return _budget if _budget is not None else available_threads()


def _validated(threads: int | None) -> int | None:
    if threads is None:
        return None
    if isinstance(threads, bool) or not isinstance(threads, int):
        raise TypeError(f"the thread budget must be an int or None, got {threads!r}")
    if threads < 1:
        raise ValueError(f"the thread budget must be at least 1, got {threads}")
    return threads


def set_thread_budget(threads: int | None) -> None:
    """Set the total thread count of pcb-analysis for the whole process.

    ``threads`` must be at least 1; ``None`` sets the default
    (:func:`available_threads`, the physical cores).  Either way the BLAS and
    OpenMP pools of the NumPy and SciPy libraries loaded so far are limited
    to the budget, so the application's total is the budget whatever library
    a solve happens to use; a later call replaces the limit.  An application
    that never calls this leaves those pools at their own defaults (usually
    every logical CPU) until a solve under :func:`serial_blas` sets them to
    the default budget, so call it once at start-up.
    """

    global _budget
    value = _validated(threads)
    with _lock:
        _budget = value
        _apply_pool_limits()


def _apply_pool_limits() -> None:
    """Limit the loaded BLAS and OpenMP pools to the budget, BLAS to one inside :func:`serial_blas`."""

    global _pool_limit
    with _lock:
        if _pool_limit is not None:
            _pool_limit.restore_original_limits()
            _pool_limit = None
        from threadpoolctl import threadpool_limits

        budget = _budget if _budget is not None else available_threads()
        _pool_limit = threadpool_limits(
            limits={"blas": 1 if _serial_blas_depth else budget, "openmp": budget}
        )


@contextmanager
def serial_blas() -> Iterator[None]:
    """Run a block with the NumPy and SciPy BLAS pools on one thread.

    For solves whose BLAS calls are too small to share: the sheet-PEEC solve
    calls BLAS only for Krylov vector work, and with the BLAS pools at the
    budget it ran no faster than on one thread (power_module radiation
    cases, budget 5: 25.5 s on 8.6 busy cores against 24.9 s on one) while
    the idle pool threads spun on every core -- the extra load that froze
    the measuring host at a budget of 8 (docs/SHEET_PEEC.md).  The OpenMP
    kernels keep the budget.  The limit is process-wide and reference
    counted, so concurrent and nested blocks keep it until the last one
    leaves; a budget change inside the block keeps it as well.
    """

    global _serial_blas_depth
    with _lock:
        _serial_blas_depth += 1
        _apply_pool_limits()
    try:
        yield
    finally:
        with _lock:
            _serial_blas_depth -= 1
            _apply_pool_limits()


@contextmanager
def thread_budget_scope(threads: int) -> Iterator[int]:
    """Run a block with the thread budget set to ``threads``, then restore the previous one.

    The previous budget (explicit or default) and its pool limits come back
    on exit, also when the block raises.  Scopes nest; they are process-wide,
    not per thread.
    """

    value = _validated(threads)
    if value is None:
        raise TypeError("thread_budget_scope needs an int; use set_thread_budget(None) to restore the default")
    with _lock:
        previous = _budget
        set_thread_budget(value)
    try:
        yield value
    finally:
        set_thread_budget(previous)
