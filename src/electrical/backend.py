"""Which implementation answers a solver call: the C++ core or the NumPy reference.

The C++ core, ``electrical._pcbcore``, answers whenever it is built.  The
NumPy implementations it replaced stay beside their facades as
``_*_reference`` modules: they answer on machines without the build, and
tests compare the two.  Nothing outside the tests chooses between them; there
is no environment variable or per-call switch.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType

try:  # pragma: no cover - depends on the local build
    from . import _pcbcore as _core
except ImportError:  # pragma: no cover
    _core = None

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
