"""The coupling loops in the C++ core: which calls go there, and their inputs.

With the core built, ``run_electro_thermal`` (and the loops built the same
way) run in ``electrical._pcbcore.coupling``: the electrical solve, the
Joule heat handed to the thermal mesh, the thermal solve, the resistivity
update and the Aitken fixed point, without returning to Python between
iterations.  The thermal operator is prepared once and reused while only its
load changes.  ``native=False`` or a CUDA backend keep the Python loops,
which call the same solvers one at a time.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from electrical import _backend
from thermal.matrix_free_mpir_fem import HeatSource, LayeredThermalMesh, ThermalConductionProblem
from thermal.matrix_free_mpir_fem.mesh import _flat_index
from thermal.matrix_free_mpir_fem.native_system import native_problem, solve_options


def coupling_core(native: bool | None, backend: str | None) -> Any:
    """The C++ core when it runs this loop, else ``None`` (the Python loop)."""

    if native is False or backend not in (None, "cpu"):
        return None
    core = _backend.core()
    if native and core is None:
        raise ImportError(
            "the pcbcore extension is not built; run cmake -S . -B build/native && cmake --build build/native"
        )
    return core


def board_arguments(
    core: Any,
    thermal_mesh: LayeredThermalMesh,
    *,
    convection: Sequence[Any],
    fixed_temperature_mask: np.ndarray | None,
    fixed_temperature_k: float | np.ndarray | None,
    radiation: Sequence[Any],
    extra_heat_sources: Sequence[HeatSource],
    extra_element_heat_w: np.ndarray | None,
) -> dict[str, Any]:
    """The unloaded board and the scenario's extra heat, as the core's loops take them."""

    board = ThermalConductionProblem(
        thermal_mesh,
        convection=tuple(convection),
        fixed_temperature_mask=fixed_temperature_mask,
        fixed_temperature_k=fixed_temperature_k,
        radiation=tuple(radiation),
    )
    offsets = np.zeros(len(extra_heat_sources) + 1, dtype=np.int64)
    np.cumsum([len(source.nodes) for source in extra_heat_sources], out=offsets[1:])
    return dict(
        board=native_problem(core, board),
        extra_element_heat=(
            None if extra_element_heat_w is None else np.ascontiguousarray(extra_element_heat_w, dtype=np.float64)
        ),
        nodal_heat=np.zeros(thermal_mesh.size, dtype=np.float64),
        source_offsets=offsets,
        source_nodes=np.asarray(
            [_flat_index(node, thermal_mesh.node_shape) for source in extra_heat_sources for node in source.nodes],
            dtype=np.int64,
        ),
        source_power=np.asarray([source.power_w for source in extra_heat_sources], dtype=np.float64),
    )


def fixed_point_arguments(config: Any) -> dict[str, Any]:
    return dict(
        max_iterations=int(config.max_iterations),
        temperature_tolerance=float(config.temperature_tolerance_k),
        relative_loss_tolerance=float(config.relative_loss_tolerance),
        relaxation=float(config.relaxation),
        aitken=bool(config.aitken),
        max_relaxation=float(config.max_relaxation),
    )


def thermal_arguments(config: Any) -> dict[str, Any]:
    """The thermal solve settings of a coupling config (the defaults of solve_thermal_conduction)."""

    options = solve_options(config.thermal, "two-level", None, None, 25, 1.0e-4)
    options.pop("reference")
    return options
