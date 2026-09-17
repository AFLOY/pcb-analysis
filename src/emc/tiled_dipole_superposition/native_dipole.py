"""Optional C++ direct summation for the dipole fields.

Built in place with ``python -m emc.tiled_dipole_superposition.native.build``.
It serves the CPU, complex128 evaluation only; CuPy and complex64 requests keep
the array path.  Opt in per call with ``native=True`` or for the process with
``PCB_NATIVE_EMC=1``; ``PCB_NATIVE_THREADS`` sets the thread count.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

try:  # pragma: no cover - depends on the local build
    from . import _dipole_native as _native
except ImportError:  # pragma: no cover
    _native = None


def native_available() -> bool:
    return _native is not None


def native_requested() -> bool:
    flag = os.environ.get("PCB_NATIVE_EMC", "").strip().lower()
    return flag in {"1", "true", "yes", "on"} and native_available()


def native_threads() -> int:
    value = os.environ.get("PCB_NATIVE_THREADS")
    if value:
        return max(1, int(value))
    return 1


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
    threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    magnetic, electric_field = _native.evaluate_fields(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(source_position, dtype=np.float64),
        np.ascontiguousarray(source_moment, dtype=np.complex128),
        float(wavenumber),
        bool(electric),
        threads if threads is not None else native_threads(),
    )
    return np.asarray(magnetic), (np.asarray(electric_field) if electric else None)


def far_field_pattern_native(
    directions: np.ndarray,
    source_position: np.ndarray,
    source_moment: np.ndarray,
    wavenumber: float,
    *,
    threads: int | None = None,
) -> np.ndarray:
    return np.asarray(
        _native.far_field_pattern(
            np.ascontiguousarray(directions, dtype=np.float64),
            np.ascontiguousarray(source_position, dtype=np.float64),
            np.ascontiguousarray(source_moment, dtype=np.complex128),
            float(wavenumber),
            threads if threads is not None else native_threads(),
        )
    )
