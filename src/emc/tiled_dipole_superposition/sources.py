"""Current dipoles from the repository's electrical solutions.

Both electrical front ends deliver a current distribution on a structured
grid.  A current element of volume ``V`` carrying density ``J`` has the moment
``I dl = J V``; a branch carrying ``I`` over length ``l`` has ``I l``.  The
adapters here turn those into :class:`CurrentDipoles` with positions in a
common right-handed board frame: ``x`` along columns, ``y`` along rows, ``z``
the layer height supplied by the caller or the sheet stackup.

The DC conduction solve carries no frequency of its own.  Using its
distribution as a phasor at ``f`` is the quasi-static assumption that the
current pattern does not change with frequency, which holds while the board
is small compared with a wavelength and skin effect is weak.  The sheet-PEEC
solve is frequency-resolved and needs no such assumption.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from electrical.matrix_free_mpir_fem.pcb import (
    PCBConductionProblem,
    PCBConductionSolution,
)

from .fields import CurrentDipoles
from .native_dipole import sheet_branch_dipoles_native, use_native


def terminal_closure_dipoles(
    problem: PCBConductionProblem,
    layer_height_m: Sequence[float],
) -> CurrentDipoles:
    """Current elements that close the terminal currents outside the copper.

    A conduction solve injects current at some pads and removes it at others;
    the component or cable between them carries it back.  Without that path
    the board's current has a net moment ``Σ I_t c_t`` that radiates like an
    open wire, and its end charges dominate the electric near field.  This
    routes every sink terminal's current from its centroid to a star point at
    the current-weighted centroid of the source pads and on to each source
    pad, as one straight element per leg.  The moments are exact, so the net
    electric dipole moment of board plus closure is zero to discretisation;
    the geometry of the actual component is not represented.
    """

    mesh = problem.mesh
    heights = np.asarray(layer_height_m, dtype=np.float64)
    x_nodes = np.concatenate(([0.0], np.cumsum(mesh.pitch_x_m)))
    y_nodes = np.concatenate(([0.0], np.cumsum(mesh.pitch_y_m)))
    centroids = []
    currents = []
    for terminal in problem.terminals:
        nodes = np.asarray(terminal.nodes, dtype=np.int64)
        centroid = np.array(
            [
                np.mean(x_nodes[nodes[:, 2]]),
                np.mean(y_nodes[nodes[:, 1]]),
                np.mean(heights[nodes[:, 0]]),
            ]
        )
        centroids.append(centroid)
        currents.append(float(terminal.current_a))
    centroid_array = np.asarray(centroids)
    current_array = np.asarray(currents)
    sources = current_array > 0.0
    if not np.any(sources):
        return CurrentDipoles(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.complex128))
    star = np.sum(centroid_array[sources] * current_array[sources, None], axis=0) / np.sum(
        current_array[sources]
    )
    positions = []
    moments = []
    for centroid, current in zip(centroid_array, current_array):
        if current == 0.0:
            continue
        # Current flows externally from sink pads to the star and on to source
        # pads; either way the element points from star to pad with weight I.
        positions.append(0.5 * (centroid + star))
        moments.append(current * (centroid - star))
    return CurrentDipoles(np.asarray(positions).reshape(-1, 3), np.asarray(moments, dtype=np.complex128).reshape(-1, 3))


def dipoles_from_pcb_dc(
    problem: PCBConductionProblem,
    solution: PCBConductionSolution,
    layer_height_m: Sequence[float],
    *,
    close_terminals: bool = False,
) -> CurrentDipoles:
    """Element and via current elements of a layered DC conduction solve.

    Each active element contributes ``J · (t · px · py)`` at its centre on its
    layer's height.  Each via contributes ``I · (z_upper - z_lower)`` along
    ``z`` at the via node, with the sign of the current from ``lower`` to
    ``upper``.  ``close_terminals`` appends :func:`terminal_closure_dipoles`
    so the distribution is divergence-free as a whole.
    """

    mesh = problem.mesh
    heights = np.asarray(layer_height_m, dtype=np.float64)
    layers, rows, cols = mesh.element_active.shape
    if heights.shape != (layers,):
        raise ValueError("layer_height_m must hold one height per electrical layer")
    if not np.all(np.isfinite(heights)):
        raise ValueError("layer heights must be finite")

    density = np.asarray(solution.current_density_a_per_m2, dtype=np.float64)
    if density.shape != (layers, rows, cols, 2):
        raise ValueError("current density must have shape (layers, rows, cols, 2)")
    thickness = np.asarray(mesh.layer_thickness_m, dtype=np.float64)
    volume = thickness[:, None, None] * mesh.cell_area_m2[None, :, :]
    x_nodes = np.concatenate(([0.0], np.cumsum(mesh.pitch_x_m)))
    y_nodes = np.concatenate(([0.0], np.cumsum(mesh.pitch_y_m)))
    x_centres = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    y_centres = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    active = mesh.element_active
    layer_index, row_index, col_index = np.nonzero(active)

    positions = np.column_stack((x_centres[col_index], y_centres[row_index], heights[layer_index]))
    moments = np.zeros((positions.shape[0], 3), dtype=np.complex128)
    moments[:, :2] = density[active] * volume[active][:, None]

    if problem.vias:
        via_current = np.asarray(solution.via_current_a, dtype=np.float64)
        via_positions = []
        via_moments = []
        for via, current in zip(problem.vias, via_current):
            lower_layer, row, col = via.lower
            upper_layer = via.upper[0]
            span = heights[upper_layer] - heights[lower_layer]
            via_positions.append(
                (x_nodes[col], y_nodes[row], 0.5 * (heights[upper_layer] + heights[lower_layer]))
            )
            via_moments.append((0.0, 0.0, current * span))
        positions = np.vstack((positions, np.asarray(via_positions)))
        moments = np.vstack((moments, np.asarray(via_moments, dtype=np.complex128)))
    dipoles = CurrentDipoles(positions, moments)
    if close_terminals:
        dipoles = dipoles.concatenate(terminal_closure_dipoles(problem, heights))
    return dipoles


def dipoles_from_sheet_peec(
    mesh, solution, *, native: bool | None = None, native_threads: int | None = None
) -> CurrentDipoles:
    """Branch currents of a sheet-PEEC solve as current elements.

    ``mesh`` is a ``SheetMesh`` and ``solution`` a ``SheetSolution`` from
    ``electrical.sheet_peec.sheet_peec``.  An in-plane branch is one pitch
    long between cell centres; its element sits midway, on the layer's
    ``z_m``.  A via branch spans the two layers' heights.  Branch currents
    are positive from the leaving node to the entering node, which is ``+x``,
    ``+y``, and lower-to-upper respectively.  ``native=True`` builds the
    elements in the optional C++ extension (``native_threads`` OpenMP
    threads); ``None`` follows ``PCB_NATIVE_EMC``.
    """

    pitch = float(mesh.pitch_m)
    heights = np.asarray([layer.z_m for layer in mesh.stackup.layers], dtype=np.float64)
    current = np.asarray(solution.branch_current, dtype=np.complex128)
    if current.shape != (mesh.branch_count,):
        raise ValueError("solution.branch_current must hold one value per branch")

    if use_native(native, "cpu", np.complex128):
        position_array, moment_array = sheet_branch_dipoles_native(
            np.asarray(mesh.branch_x, dtype=np.int64).reshape(-1, 3),
            np.asarray(mesh.branch_y, dtype=np.int64).reshape(-1, 3),
            np.asarray(
                [(via.lower_layer, via.upper_layer, via.row, via.col) for via in mesh.via_branches],
                dtype=np.int64,
            ).reshape(-1, 4),
            pitch,
            heights,
            current,
            threads=native_threads,
        )
        return CurrentDipoles(position_array, moment_array)

    positions: list[tuple[float, float, float]] = []
    moments: list[tuple[complex, complex, complex]] = []
    index = 0
    for layer, row, col in mesh.branch_x:
        positions.append(((col + 1.0) * pitch, (row + 0.5) * pitch, heights[layer]))
        moments.append((current[index] * pitch, 0.0, 0.0))
        index += 1
    for layer, row, col in mesh.branch_y:
        positions.append(((col + 0.5) * pitch, (row + 1.0) * pitch, heights[layer]))
        moments.append((0.0, current[index] * pitch, 0.0))
        index += 1
    for via in mesh.via_branches:
        lower = heights[via.lower_layer]
        upper = heights[via.upper_layer]
        positions.append(((via.col + 0.5) * pitch, (via.row + 0.5) * pitch, 0.5 * (lower + upper)))
        moments.append((0.0, 0.0, current[index] * (upper - lower)))
        index += 1
    return CurrentDipoles(
        np.asarray(positions, dtype=np.float64).reshape(-1, 3),
        np.asarray(moments, dtype=np.complex128).reshape(-1, 3),
    )
