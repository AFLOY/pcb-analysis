"""A plated barrel as azimuthal strips for the hybrid sheet mesh (experiment).

The wall of a via's hole on each layer is cut into ``strips`` sectors by
angle about the barrel's axis; each sector's edges meet in a node, and the
barrel between two layers is ``strips`` vertical strips joining the
sectors' nodes, each a share of the barrel's wall: resistance ``strips``
times the barrel's, and partial inductances of the strips as parallel
filaments on the barrel's mid-wall circle (self: a flat bar of the strip's
width and the wall's thickness).  Between the two extremes -- the whole wall
one node (current crowds around the hole freely) and the barrel tied to the
elements around the hole by equal links (current forced to spread evenly) --
the strips let the current crowd only as far as the barrel's own impedance
allows.
"""

from __future__ import annotations

import math

import numpy as np

from .sheet_inductance import bar_self_inductance, filament_mutual_inductance


def strip_inductance(length: float, outer_radius: float, inner_radius: float, strips: int) -> np.ndarray:
    """(strips, strips) partial inductances of a barrel cut into equal azimuthal strips."""
    radius = 0.5 * (outer_radius + inner_radius)
    wall = max(outer_radius - inner_radius, 1e-9)
    width = 2.0 * math.pi * radius / strips
    out = np.empty((strips, strips))
    for i in range(strips):
        for j in range(strips):
            if i == j:
                out[i, j] = bar_self_inductance(length, width, wall)
            else:
                d = 2.0 * radius * abs(math.sin(math.pi * (i - j) / strips))
                out[i, j] = filament_mutual_inductance(length, d)
    return out


def sector_of(points: np.ndarray, centre, strips: int) -> np.ndarray:
    """The strip each point (N, 2) faces, by its angle about ``centre``."""
    angle = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0]) % (2.0 * math.pi)
    return np.minimum((angle / (2.0 * math.pi) * strips).astype(int), strips - 1)
