"""Radiated-emission evaluation by tiled superposition of current dipoles.

Every current element of a solved board is a Hertzian dipole.  Summing the
exact dipole fields gives the near-field map a scan probe reads; summing the
far-zone terms gives the pattern, radiated power, and the field a test site
reads at 3 m or 10 m, which the limit tables then turn into a margin.
"""

from .far_field import (
    FarFieldPattern,
    SphereSampling,
    db_microvolt_per_m,
    far_field_pattern,
)
from .fields import (
    EPSILON_0_F_PER_M,
    FREE_SPACE_IMPEDANCE_OHM,
    MU_0_H_PER_M,
    SPEED_OF_LIGHT_M_PER_S,
    CurrentDipoles,
    FieldSamples,
    array_namespace,
    evaluate_fields,
    scan_plane,
    wavenumber_per_m,
)
from .limits import (
    CISPR32_CLASS_A,
    CISPR32_CLASS_B,
    FCC_PART15_CLASS_A,
    FCC_PART15_CLASS_B,
    STANDARD_LIMITS,
    EmissionLimit,
    EmissionMargin,
    LimitBand,
    emission_margin,
)
from .moments import DipoleMoments, dipole_moments
from .sources import (
    dipoles_from_pcb_dc,
    dipoles_from_sheet_peec,
    terminal_closure_dipoles,
)

__all__ = [
    "CISPR32_CLASS_A",
    "CISPR32_CLASS_B",
    "CurrentDipoles",
    "DipoleMoments",
    "EPSILON_0_F_PER_M",
    "EmissionLimit",
    "EmissionMargin",
    "FCC_PART15_CLASS_A",
    "FCC_PART15_CLASS_B",
    "FREE_SPACE_IMPEDANCE_OHM",
    "FarFieldPattern",
    "FieldSamples",
    "LimitBand",
    "MU_0_H_PER_M",
    "SPEED_OF_LIGHT_M_PER_S",
    "STANDARD_LIMITS",
    "SphereSampling",
    "array_namespace",
    "db_microvolt_per_m",
    "dipole_moments",
    "dipoles_from_pcb_dc",
    "dipoles_from_sheet_peec",
    "emission_margin",
    "evaluate_fields",
    "far_field_pattern",
    "scan_plane",
    "terminal_closure_dipoles",
    "wavenumber_per_m",
]
