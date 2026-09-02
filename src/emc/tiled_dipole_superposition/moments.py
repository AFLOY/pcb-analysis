"""Low-order multipole moments of a current distribution as EMC proxies.

For a source small compared with a wavelength the far field is set by two
numbers: the net electric dipole moment ``P = Σ p_i`` (the common-mode-like
current that does not return nearby) and the magnetic dipole moment
``M = ½ Σ r_i × p_i`` (the loop the differential current encloses).  Their
radiated powers are

```text
P_e = η k² |P|² / 12π          P_m = η k⁴ |M|² / 12π
```

Both are quadratic in the currents, so they fit the exact incremental scoring
in ``electrical.dice_peec`` and make cheap gates for an optimizer before a
full pattern is evaluated.  The ratio of the pattern's integrated power to
``P_e + P_m`` tells how much of the radiation the two lowest moments miss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fields import FREE_SPACE_IMPEDANCE_OHM, CurrentDipoles, wavenumber_per_m


@dataclass(frozen=True)
class DipoleMoments:
    electric_a_m: np.ndarray       # complex (3,), A·m
    magnetic_a_m2: np.ndarray      # complex (3,), A·m²
    frequency_hz: float

    @property
    def electric_radiated_power_w(self) -> float:
        k = wavenumber_per_m(self.frequency_hz)
        return float(
            FREE_SPACE_IMPEDANCE_OHM * k**2 * np.sum(np.abs(self.electric_a_m) ** 2) / (12.0 * np.pi)
        )

    @property
    def magnetic_radiated_power_w(self) -> float:
        k = wavenumber_per_m(self.frequency_hz)
        return float(
            FREE_SPACE_IMPEDANCE_OHM * k**4 * np.sum(np.abs(self.magnetic_a_m2) ** 2) / (12.0 * np.pi)
        )

    @property
    def total_radiated_power_w(self) -> float:
        return self.electric_radiated_power_w + self.magnetic_radiated_power_w


def dipole_moments(
    sources: CurrentDipoles,
    frequency_hz: float,
    *,
    origin_m: np.ndarray | None = None,
) -> DipoleMoments:
    """Net electric and magnetic dipole moments about ``origin_m``.

    The magnetic moment of a closed current is origin-independent; for an
    unbalanced distribution it is not, and the origin should be the point the
    far field is referred to (the default is the moment-weighted centroid of
    the source positions).
    """

    position = sources.position_m
    moment = sources.moment_a_m
    if origin_m is None:
        weight = np.linalg.norm(moment, axis=1)
        total = float(np.sum(weight))
        origin = (
            np.sum(position * weight[:, None], axis=0) / total
            if total > 0.0
            else np.mean(position, axis=0)
        )
    else:
        origin = np.asarray(origin_m, dtype=np.float64).reshape(3)
    relative = position - origin
    electric = np.sum(moment, axis=0)
    magnetic = 0.5 * np.sum(np.cross(relative, moment), axis=0)
    return DipoleMoments(
        electric_a_m=electric,
        magnetic_a_m2=magnetic,
        frequency_hz=float(frequency_hz),
    )
