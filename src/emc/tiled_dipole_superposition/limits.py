"""Radiated-emission limit lines and the margin of a predicted field to them.

The tables hold the published radiated limits for information-technology
equipment.  Values are quasi-peak unless the band says otherwise, at the
distance the standard states.  A limit read at another distance is rescaled
by ``20 log10(d_limit / d)``, the inverse-distance rule the standards
themselves use for far-zone measurements; the caller has to decide whether a
3 m reading of a small board is far-zone at the frequency in question.

Sources: CISPR 32 Ed. 2 (identical to CISPR 22 below 1 GHz); 47 CFR 15.109
for FCC Part 15 Subpart B.  These are the tabulated numbers, not a
certification: a test site adds antenna factors, height scans, and
measurement uncertainty that a field prediction does not carry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .far_field import db_microvolt_per_m


Detector = Literal["quasi-peak", "average", "peak"]


@dataclass(frozen=True)
class LimitBand:
    lower_hz: float
    upper_hz: float
    limit_dbuv_per_m: float
    distance_m: float
    detector: Detector = "quasi-peak"


@dataclass(frozen=True)
class EmissionLimit:
    """A named limit line built from contiguous or overlapping bands."""

    name: str
    bands: tuple[LimitBand, ...]

    def limit_dbuv_per_m(
        self,
        frequency_hz: float,
        *,
        distance_m: float | None = None,
        detector: Detector = "quasi-peak",
    ) -> float:
        """Limit at ``frequency_hz`` rescaled to ``distance_m`` (default: as published)."""

        frequency = float(frequency_hz)
        for band in self.bands:
            if band.detector != detector:
                continue
            if band.lower_hz <= frequency < band.upper_hz:
                if distance_m is None:
                    return float(band.limit_dbuv_per_m)
                return float(band.limit_dbuv_per_m + 20.0 * np.log10(band.distance_m / float(distance_m)))
        raise ValueError(
            f"{self.name} has no {detector} limit at {frequency / 1e6:.3f} MHz"
        )

    def published_distance_m(self, frequency_hz: float, detector: Detector = "quasi-peak") -> float:
        frequency = float(frequency_hz)
        for band in self.bands:
            if band.detector == detector and band.lower_hz <= frequency < band.upper_hz:
                return band.distance_m
        raise ValueError(f"{self.name} has no {detector} limit at {frequency / 1e6:.3f} MHz")


_MHZ = 1.0e6
_GHZ = 1.0e9

CISPR32_CLASS_A = EmissionLimit(
    "CISPR 32 Class A",
    (
        LimitBand(30 * _MHZ, 230 * _MHZ, 40.0, 10.0),
        LimitBand(230 * _MHZ, 1 * _GHZ, 47.0, 10.0),
        LimitBand(1 * _GHZ, 3 * _GHZ, 56.0, 3.0, "average"),
        LimitBand(3 * _GHZ, 6 * _GHZ, 60.0, 3.0, "average"),
        LimitBand(1 * _GHZ, 3 * _GHZ, 76.0, 3.0, "peak"),
        LimitBand(3 * _GHZ, 6 * _GHZ, 80.0, 3.0, "peak"),
    ),
)

CISPR32_CLASS_B = EmissionLimit(
    "CISPR 32 Class B",
    (
        LimitBand(30 * _MHZ, 230 * _MHZ, 30.0, 10.0),
        LimitBand(230 * _MHZ, 1 * _GHZ, 37.0, 10.0),
        LimitBand(1 * _GHZ, 3 * _GHZ, 50.0, 3.0, "average"),
        LimitBand(3 * _GHZ, 6 * _GHZ, 54.0, 3.0, "average"),
        LimitBand(1 * _GHZ, 3 * _GHZ, 70.0, 3.0, "peak"),
        LimitBand(3 * _GHZ, 6 * _GHZ, 74.0, 3.0, "peak"),
    ),
)

# 47 CFR 15.109(a): Class B at 3 m in µV/m: 100, 150, 200, 500.
FCC_PART15_CLASS_B = EmissionLimit(
    "FCC Part 15 Class B",
    (
        LimitBand(30 * _MHZ, 88 * _MHZ, float(db_microvolt_per_m(100e-6)), 3.0),
        LimitBand(88 * _MHZ, 216 * _MHZ, float(db_microvolt_per_m(150e-6)), 3.0),
        LimitBand(216 * _MHZ, 960 * _MHZ, float(db_microvolt_per_m(200e-6)), 3.0),
        LimitBand(960 * _MHZ, 40 * _GHZ, float(db_microvolt_per_m(500e-6)), 3.0),
    ),
)

# 47 CFR 15.109(b): Class A at 10 m in µV/m: 90, 150, 210, 300.
FCC_PART15_CLASS_A = EmissionLimit(
    "FCC Part 15 Class A",
    (
        LimitBand(30 * _MHZ, 88 * _MHZ, float(db_microvolt_per_m(90e-6)), 10.0),
        LimitBand(88 * _MHZ, 216 * _MHZ, float(db_microvolt_per_m(150e-6)), 10.0),
        LimitBand(216 * _MHZ, 960 * _MHZ, float(db_microvolt_per_m(210e-6)), 10.0),
        LimitBand(960 * _MHZ, 40 * _GHZ, float(db_microvolt_per_m(300e-6)), 10.0),
    ),
)

STANDARD_LIMITS: dict[str, EmissionLimit] = {
    limit.name: limit
    for limit in (CISPR32_CLASS_A, CISPR32_CLASS_B, FCC_PART15_CLASS_A, FCC_PART15_CLASS_B)
}


@dataclass(frozen=True)
class EmissionMargin:
    """Predicted field against a limit; positive margin means below the limit."""

    limit_name: str
    frequency_hz: float
    distance_m: float
    detector: Detector
    predicted_dbuv_per_m: float
    limit_dbuv_per_m: float

    @property
    def margin_db(self) -> float:
        return self.limit_dbuv_per_m - self.predicted_dbuv_per_m

    @property
    def compliant(self) -> bool:
        return self.margin_db >= 0.0


def emission_margin(
    field_v_per_m: float,
    frequency_hz: float,
    limit: EmissionLimit,
    *,
    distance_m: float,
    detector: Detector = "quasi-peak",
) -> EmissionMargin:
    """Compare a predicted field magnitude at ``distance_m`` with a limit line."""

    predicted = float(db_microvolt_per_m(float(field_v_per_m)))
    allowed = limit.limit_dbuv_per_m(frequency_hz, distance_m=distance_m, detector=detector)
    return EmissionMargin(
        limit_name=limit.name,
        frequency_hz=float(frequency_hz),
        distance_m=float(distance_m),
        detector=detector,
        predicted_dbuv_per_m=predicted,
        limit_dbuv_per_m=allowed,
    )
