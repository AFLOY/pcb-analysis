"""The process-wide thread budget: the only threading control pcb-analysis offers.

An application states how many threads pcb-analysis may use in total, once,
with :func:`set_thread_budget` (or for a block with
:func:`thread_budget_scope`).  Every parallel path reads that one number:

* the OpenMP team of every fused C++ kernel (layered DC and scalar Maxwell
  Q1, thermal hexahedral Q1, dipole fields, point-in-solid tests, sheet pFFT
  near-field corrections) is the budget when the kernel runs alone;
* PyPEEC's SciPy FFT workers are the budget;
* the BLAS and OpenMP pools of NumPy and SciPy are limited to the budget
  (through :mod:`threadpoolctl`) while an explicit budget is set.

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


def available_threads() -> int:
    """Logical CPUs this process may run on (its affinity mask where the OS reports one)."""

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):  # pragma: no cover - non-Linux hosts
        return os.cpu_count() or 1


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

    ``threads`` must be at least 1; ``None`` restores the default
    (:func:`available_threads`).  An explicit budget also limits the BLAS and
    OpenMP pools of the NumPy and SciPy libraries loaded so far to that many
    threads; a later call replaces the limit and ``None`` restores the
    limits those libraries had before the first call.
    """

    global _budget, _pool_limit
    value = _validated(threads)
    with _lock:
        if _pool_limit is not None:
            _pool_limit.restore_original_limits()
            _pool_limit = None
        _budget = value
        if value is not None:
            from threadpoolctl import threadpool_limits

            _pool_limit = threadpool_limits(limits=value)


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
