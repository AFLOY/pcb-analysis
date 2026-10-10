"""Conduction problems as the C++ core holds them, and its solutions back.

The coupling loops of ``multiphysics.staggered_coupling`` run in
``electrical._pcbcore.coupling`` on a ``DCProblem``: the mesh at the
reference temperature, the vias, the gauge and the terminals as flat node
groups, and the solver settings.  :func:`dc_problem` builds it from a
:class:`.pcb.PCBConductionProblem`; :func:`solution_from` wraps the core's
result in a :class:`.pcb.PCBConductionSolution`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .pcb import PCBConductionProblem, PCBConductionSolution, Preconditioner, _flat_index
from .solver import MPIRConfig, mpir_result_from_core

LOW_RUNTIME_NAME = "numpy-fp32"


def _groups(terminals: Any, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(terminals) + 1, dtype=np.int64)
    np.cumsum([len(terminal.nodes) for terminal in terminals], out=offsets[1:])
    nodes = np.asarray([_flat_index(node, shape) for terminal in terminals for node in terminal.nodes], dtype=np.int64)
    return offsets, nodes


def dc_problem(
    core: Any,
    problem: PCBConductionProblem,
    *,
    config: MPIRConfig | None = None,
    preconditioner: Preconditioner = "two-level",
    coarse_block_nodes: int | None = None,
) -> Any:
    """``problem`` as an ``electrical._pcbcore.coupling.DCProblem``."""

    mesh = problem.mesh
    shape = mesh.node_shape
    config = config or MPIRConfig()
    current_offsets, current_nodes = _groups(problem.terminals, shape)
    voltage_offsets, voltage_nodes = _groups(problem.voltage_terminals, shape)
    return core.coupling.DCProblem(
        np.ascontiguousarray(mesh.element_active, dtype=np.uint8),
        np.asarray(mesh.layer_thickness_m, dtype=np.float64),
        mesh.pitch_x_m,
        mesh.pitch_y_m,
        mesh.conductivity_s_per_m,
        np.asarray([_flat_index(via.lower, shape) for via in problem.vias], dtype=np.int64),
        np.asarray([_flat_index(via.upper, shape) for via in problem.vias], dtype=np.int64),
        np.asarray([via.resistance_ohm for via in problem.vias], dtype=np.float64),
        -1 if problem.reference_node is None else _flat_index(problem.reference_node, shape),
        current_offsets,
        current_nodes,
        np.asarray([terminal.current_a for terminal in problem.terminals], dtype=np.float64),
        voltage_offsets,
        voltage_nodes,
        np.asarray([terminal.voltage_v for terminal in problem.voltage_terminals], dtype=np.float64),
        preconditioner == "two-level",
        0 if coarse_block_nodes is None else int(coarse_block_nodes),
        float(config.relative_tolerance),
        float(config.absolute_tolerance),
        float(config.inner_relative_tolerance),
        int(config.max_outer_iterations),
        int(config.max_inner_iterations),
    )


def solution_from(result: dict) -> PCBConductionSolution:
    return PCBConductionSolution(
        potential_v=result["potential"],
        current_density_a_per_m2=result["current_density"],
        via_current_a=result["via_current"],
        joule_loss_w=float(result["joule_loss"]),
        element_joule_loss_w=result["element_joule_loss"],
        via_joule_loss_w=result["via_joule_loss"],
        max_current_density_a_per_m2=float(result["max_current_density"]),
        solve=mpir_result_from_core(result["solve"], LOW_RUNTIME_NAME),
        voltage_terminal_current_a=result["terminal_current"],
    )
