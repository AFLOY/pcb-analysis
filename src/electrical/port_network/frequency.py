"""Finding a frequency among those something was solved or stated at.

A frequency reaches the network along several paths -- a problem's list, a
refinement's geometric midpoint ``sqrt(low * high)``, a source's harmonic
``k * f0``, a JSON round trip -- and the same frequency can arrive with its
last bits apart.  Looked up as an exact ``float`` key it is then missing,
and a source spectrum that misses its frequency carries nothing without a
word.  Frequencies are therefore matched within a relative
``FREQUENCY_RTOL``: far below any two frequencies a board is solved at, far
above the rounding of a product or a square root.
"""

from __future__ import annotations

from typing import Iterable

FREQUENCY_RTOL = 1e-9
"""Two frequencies are the same when they differ by at most this fraction of the larger."""


def match_frequency(frequency_hz: float, frequencies_hz: Iterable[float]) -> float | None:
    """The one of ``frequencies_hz`` that is ``frequency_hz`` within ``FREQUENCY_RTOL`` (the nearest), or ``None``."""
    f = float(frequency_hz)
    best: float | None = None
    for candidate in frequencies_hz:
        g = float(candidate)
        gap = abs(g - f)
        if gap <= FREQUENCY_RTOL * max(abs(f), abs(g)) and (best is None or gap < abs(best - f)):
            best = g
    return best


def distinct_frequencies(frequencies_hz: Iterable[float]) -> list[float]:
    """Ascending, with frequencies that :func:`match_frequency` would take as one kept once (the first seen)."""
    kept: list[float] = []
    for f in frequencies_hz:
        if match_frequency(f, kept) is None:
            kept.append(float(f))
    return sorted(kept)


__all__ = ["FREQUENCY_RTOL", "distinct_frequencies", "match_frequency"]
