"""A copper layer as two surface sheets with the slab's internal impedance (hybrid sheet mesh, experiment).

At high frequency a layer's current lives in skins at its two faces.  Rather
than cutting the thickness into graded filaments (the sheet pairs, and the
near zone, grow with their square), the layer is two thin sheets, one at each
face, whose currents K_t and K_b obey the one-dimensional diffusion through
the thickness t exactly: with k = (1 + j) / delta,

    E_t = (k / sigma) (coth(kt) K_t + csch(kt) K_b)
    E_b = (k / sigma) (csch(kt) K_t + coth(kt) K_b)

(the slab's surface impedance matrix, fields at the faces from the face
currents).  The face currents are the equivalent currents n x H of the
fields outside, radiating in free space: the field they make between the
sheets is not the copper's inside (it is what screens it), so Z_s is used as
it is -- subtracting that field (tried first, as "double counting") left
small loops of the edges' sliver elements with a negative inductance and
inflated R five-fold at 150 MHz.  At low frequency the common current sees
R_dc and the slab's internal inductance mu t / 12; the difference current
sees j omega mu t / 2 more than a slab would, which outside the copper
changes nothing (opposite sheets) and costs no loss.  Lateral variation
(edges, holes) is left to the mesh.
"""
from __future__ import annotations

import cmath
import math

MU0 = 4e-7 * math.pi


def slab_coefficients(thickness: float, resistivity: float, frequency: float) -> tuple[complex, complex]:
    """(coth(kt), csch(kt)) times k / sigma: the slab's surface impedance matrix entries, ohm per square."""
    if frequency <= 0.0:
        raise ValueError("the surface sheets need a frequency above zero (one sheet carries DC)")
    sigma = 1.0 / resistivity
    k = (1.0 + 1.0j) * math.sqrt(math.pi * frequency * MU0 * sigma)
    x = k * thickness
    e2 = cmath.exp(-2.0 * x)
    coth = (1.0 + e2) / (1.0 - e2)
    csch = 2.0 * cmath.exp(-x) / (1.0 - e2)
    return k / sigma * coth, k / sigma * csch


def internal_impedance(thickness: float, resistivity: float, frequency: float, separation: float) -> tuple[complex, complex]:
    """(z_self, z_mutual) per square of a layer's two face sheets: the slab's surface impedance matrix.

    ``separation`` (the sheets' distance) is not used: see the module's note.
    """
    return slab_coefficients(thickness, resistivity, frequency)


def face_sheets(thickness: float, centre: float, frequency: float, resistivity: float):
    """The two face sheets of a layer: (kernel thickness, (z_top, z_bottom), separation).

    Each sheet is a skin's thickness (at most a quarter of the copper),
    centred that far inside its face.
    """
    delta = math.sqrt(resistivity / (math.pi * frequency * MU0))
    eps = min(delta, 0.25 * thickness)
    half = 0.5 * (thickness - eps)
    return eps, (centre + half, centre - half), 2.0 * half
