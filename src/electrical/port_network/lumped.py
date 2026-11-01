"""Lumped parts a copper network is closed with.

A capacitor on a board is, over the band a layout cares about, a series
C-ESR-ESL: its capacitance at the operating bias, its loss, and the
inductance of the part and its mounting.  The same series element with no
capacitance is a resistor or an inductor.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeriesRLC:
    """Series R, L, C of one part.  ``capacitance_f`` of ``0`` means no capacitor
    (the branch is R + jωL alone)."""

    resistance_ohm: float = 0.0
    inductance_h: float = 0.0
    capacitance_f: float = 0.0

    def impedance(self, omega: float) -> complex:
        """``R + jωL + 1/(jωC)`` at angular frequency ``omega`` (> 0 when C is set)."""
        z = complex(float(self.resistance_ohm), omega * float(self.inductance_h))
        c = float(self.capacitance_f)
        return z + (1.0 / (1j * omega * c) if c > 0.0 else 0j)


def capacitor(capacitance_f: float, esr_ohm: float, esl_h: float) -> SeriesRLC:
    """A capacitor's series model from the figures a datasheet gives."""
    return SeriesRLC(resistance_ohm=esr_ohm, inductance_h=esl_h, capacitance_f=capacitance_f)


__all__ = ["SeriesRLC", "capacitor"]
