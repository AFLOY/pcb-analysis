"""Optional C++ direct summation for the dipole fields.

Built in place with ``python -m emc.tiled_dipole_superposition.native.build``.
It serves the CPU, complex128 evaluation only; CuPy and complex64 requests keep
the array path.  When built it is the default for those calls; ``native=False``
keeps the array path.  Every call runs an OpenMP team of
:func:`electrical.threads.thread_budget` threads.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from electrical.threads import thread_budget

try:  # pragma: no cover - depends on the local build
    from electrical._pcbcore import dipole as _native
except ImportError:  # pragma: no cover
    try:  # a module built on its own by native/build.py
        from . import _dipole_native as _native
    except ImportError:
        _native = None


def native_available() -> bool:
    return _native is not None


def native_requested() -> bool:
    """The built extension is the default for eligible calls; ``native=False`` opts out."""

    return native_available()


def use_native(native: bool | None, backend: str, dtype: Any) -> bool:
    """Decide the path; ``native=True`` raises when it cannot be honoured."""

    if native is False:
        return False
    eligible = backend == "cpu" and np.dtype(dtype) == np.dtype(np.complex128)
    if native is True:
        if _native is None:
            raise ImportError(
                "the emc native extension is not built; run "
                "python -m emc.tiled_dipole_superposition.native.build"
            )
        if not eligible:
            raise ValueError("native=True requires backend='cpu' and complex128")
        return True
    return eligible and native_requested()


def evaluate_fields_native(
    points: np.ndarray,
    source_position: np.ndarray,
    source_moment: np.ndarray,
    wavenumber: float,
    *,
    electric: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    magnetic, electric_field = _native.evaluate_fields(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(source_position, dtype=np.float64),
        np.ascontiguousarray(source_moment, dtype=np.complex128),
        float(wavenumber),
        bool(electric),
        thread_budget(),
    )
    return np.asarray(magnetic), (np.asarray(electric_field) if electric else None)


def far_field_pattern_native(
    directions: np.ndarray,
    source_position: np.ndarray,
    source_moment: np.ndarray,
    wavenumber: float,
) -> np.ndarray:
    return np.asarray(
        _native.far_field_pattern(
            np.ascontiguousarray(directions, dtype=np.float64),
            np.ascontiguousarray(source_position, dtype=np.float64),
            np.ascontiguousarray(source_moment, dtype=np.complex128),
            float(wavenumber),
            thread_budget(),
        )
    )
