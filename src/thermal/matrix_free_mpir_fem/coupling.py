"""Electrothermal coupling: Joule loss of an electrical solve as thermal load.

The electrical mesh knows copper layers; the thermal mesh knows element slabs
that also contain the dielectric between them.  ``layer_slabs`` maps every
electrical layer index to the thermal slab that holds that copper.  Both
meshes have to share the in-plane element grid; the two solvers use the same
pitch and the same ``(rows, cols)`` footprint, so no interpolation is needed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from electrical.matrix_free_mpir_fem.pcb import (
    PCBConductionProblem,
    PCBConductionSolution,
)

from .conduction import HeatSource, LayeredThermalMesh


def _check_layer_slabs(
    layer_slabs: Sequence[int], layer_count: int, slab_count: int
) -> tuple[int, ...]:
    slabs = tuple(int(index) for index in layer_slabs)
    if len(slabs) != layer_count:
        raise ValueError(
            f"layer_slabs has {len(slabs)} entries for {layer_count} electrical layers"
        )
    if any(index < 0 or index >= slab_count for index in slabs):
        raise ValueError(f"layer_slabs must lie in [0, {slab_count - 1}]")
    return slabs


def element_joule_heat_w(
    solution: PCBConductionSolution,
    thermal_mesh: LayeredThermalMesh,
    layer_slabs: Sequence[int],
) -> np.ndarray:
    """Place per-element copper Joule loss into the thermal element grid.

    Returns an array of shape ``(slabs, rows, cols)`` suitable for
    ``ThermalConductionProblem.element_heat_w``.  Its sum equals the sum of
    the electrical solution's element losses; via losses are handled by
    :func:`via_joule_heat_sources` because vias sit between element slabs.
    """

    element_loss = np.asarray(solution.element_joule_loss_w, dtype=np.float64)
    slabs, rows, cols = thermal_mesh.element_grid_shape
    if element_loss.ndim != 3 or element_loss.shape[1:] != (rows, cols):
        raise ValueError(
            "the electrical element grid must match the thermal (rows, cols) footprint"
        )
    mapping = _check_layer_slabs(layer_slabs, element_loss.shape[0], slabs)
    heat = np.zeros((slabs, rows, cols), dtype=np.float64)
    for layer, slab in enumerate(mapping):
        heat[slab] += element_loss[layer]
    return heat


def via_joule_heat_sources(
    problem: PCBConductionProblem,
    solution: PCBConductionSolution,
    thermal_mesh: LayeredThermalMesh,
    layer_slabs: Sequence[int],
) -> tuple[HeatSource, ...]:
    """Turn each via's Joule loss into nodal heat at its two endpoints.

    Half of a via's loss goes to each endpoint.  An endpoint on electrical
    layer ``L`` lands on the two thermal node faces bounding slab
    ``layer_slabs[L]``, so the heat enters the copper slab and not only one
    of its surfaces.
    """

    slabs, rows, cols = thermal_mesh.element_grid_shape
    layer_count = problem.mesh.node_shape[0]
    mapping = _check_layer_slabs(layer_slabs, layer_count, slabs)
    losses = np.asarray(solution.via_joule_loss_w, dtype=np.float64)
    if losses.shape != (len(problem.vias),):
        raise ValueError("solution.via_joule_loss_w must hold one value per via")

    sources: list[HeatSource] = []
    for index, (via, loss) in enumerate(zip(problem.vias, losses)):
        for end, node in (("lower", via.lower), ("upper", via.upper)):
            layer, row, col = node
            if row > rows or col > cols:
                raise ValueError(f"via node {node!r} lies outside the thermal footprint")
            slab = mapping[layer]
            sources.append(
                HeatSource(
                    nodes=((slab, row, col), (slab + 1, row, col)),
                    power_w=0.5 * float(loss),
                    name=f"via{index}-{end}",
                )
            )
    return tuple(sources)
