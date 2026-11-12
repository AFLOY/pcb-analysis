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


def barrel_network(top_ports, bottom_ports, next_node: int, length: float, outer_radius: float, inner_radius: float,
                   resistivity: float, segments: int = 4):
    """A barrel of ``len(top_ports)`` strips and ``segments`` levels, conducting around as well as along.

    ``top_ports``/``bottom_ports``: the node ports of the wall sectors on the
    two layers, strip by strip.  Internal levels get new node ids from
    ``next_node`` on.  Returns (new node ids, coupled vertical block, azimuthal
    links): the block holds every strip segment, coupled to every other (its
    partial inductances: a flat bar for itself, parallel filaments with their
    axial offset otherwise); the azimuthal links join neighbouring strips at
    each internal level with the wall's resistance around (no inductance).
    """
    from .sheet_inductance import parallel_filament_mutual_inductance

    strips = len(top_ports)
    radius = 0.5 * (outer_radius + inner_radius)
    wall = max(outer_radius - inner_radius, 1e-9)
    width = 2.0 * math.pi * radius / strips
    seg = length / segments
    levels = [list(top_ports)]
    nodes = []
    for _ in range(segments - 1):
        level = list(range(next_node, next_node + strips))
        next_node += strips
        nodes += level
        levels.append(level)
    levels.append(list(bottom_ports))
    pairs, where = [], []
    for m in range(segments):
        for k in range(strips):
            pairs.append((levels[m][k], levels[m + 1][k]))
            where.append((m, k))
    n = len(pairs)
    inductance = np.empty((n, n))
    for i, (mi, ki) in enumerate(where):
        for j, (mj, kj) in enumerate(where):
            if i == j:
                inductance[i, j] = bar_self_inductance(seg, width, wall)
            else:
                d = 2.0 * radius * abs(math.sin(math.pi * (ki - kj) / strips))
                inductance[i, j] = float(parallel_filament_mutual_inductance(seg, seg, (mi - mj) * seg, d))
    resistance = np.full(n, resistivity * seg / (width * wall))
    around = resistivity * width / (seg * wall)
    azimuthal = []
    for level in levels[1:-1]:
        for k in range(strips):
            azimuthal.append((level[k], level[(k + 1) % strips], around))
    return nodes, (pairs, resistance, inductance), azimuthal, next_node
