"""Which implementation answers a solver call: the C++ core or the NumPy reference.

The C++ core, ``electrical._pcbcore``, answers whenever it is built.  The
NumPy implementations it replaced stay beside their facades as
``_*_reference`` modules: they answer on machines without the build, and
tests compare the two.  Nothing outside the tests chooses between them; there
is no environment variable or per-call switch.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType

# A core built with Clang or AOCC runs its teams on LLVM's libomp, whose
# threads spin for 200 ms after every parallel region by default.  A sheet
# solve alternates the core's operator with SuperLU solves and Gram-Schmidt
# steps on Python's own thread pools, which those spinning threads starved:
# on a lockstep solve of eight excitations (AOCC 5.2.0, 8 threads) the solve
# took 41.7 s spinning and 32.8 s with the threads put to sleep at once.
# libomp reads this when its runtime starts, so it is set before the core
# loads; a value already in the environment is kept.  GCC's libgomp does not
# read it.
os.environ.setdefault("KMP_BLOCKTIME", "0")

try:  # pragma: no cover - depends on the local build
    from . import _pcbcore as _core
except ImportError:  # pragma: no cover
    _core = None

try:  # pragma: no cover - built only with -DPCB_NATIVE_CUDA=ON
    from . import _pcbcore_cuda as _cuda
except ImportError:  # pragma: no cover
    _cuda = None

_reference_forced = False


def native_available() -> bool:
    """True when the C++ core is built for this interpreter."""

    return _core is not None


def native_core() -> ModuleType | None:
    """The built extension whatever :func:`use_reference` says, for kernels
    that keep their own ``native`` switch (dipole sums, point-in-solid tests)."""

    return _core


def core() -> ModuleType | None:
    """The C++ core when it should answer, else ``None`` (use the reference)."""

    return None if _reference_forced else _core


def cuda_core() -> ModuleType | None:
    """The CUDA module when it is built and a device is present, else ``None``.

    It runs the FP32 inner solves of prepared systems on a device
    (``operator.device_system()``); it is opt-in until it has been checked
    on a CUDA machine, so no solver selects it on its own.
    """

    if _reference_forced or _cuda is None or _core is None:
        return None
    return _cuda if _cuda.device_count() > 0 else None


@contextmanager
def use_reference() -> Iterator[None]:
    """For tests: run the block on the NumPy references even if the core is built.

    Process-wide and not re-entrant across threads; tests use it to compute
    the reference answer the C++ result is compared with.
    """

    global _reference_forced
    previous = _reference_forced
    _reference_forced = True
    try:
        yield
    finally:
        _reference_forced = previous
