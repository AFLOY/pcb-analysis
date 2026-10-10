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

from electrical import backend as _backend
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


def body_start_k(problem: ThermalConductionProblem) -> float:
    """A body's contact temperature before the first exchange: its own reference."""

    if problem.convection:
        return float(problem.convection[0].mean_ambient_k())
    if problem.radiation:
        return float(problem.radiation[0].mean_ambient_k())
    mask = problem.fixed_temperature_mask
    return float(np.mean(problem.fixed_temperature_k[mask]))


def body_arguments(core: Any, bodies: Sequence[Any]) -> list[tuple]:
    """``BodyContact`` entries as the core's interface loop takes them."""

    from thermal.matrix_free_mpir_fem.mesh import FACE_DIRECTIONS

    return [
        (
            native_problem(core, body.body),
            body.contact.board_side == "top",
            FACE_DIRECTIONS.index(body.contact.body_face),
            np.ascontiguousarray(body.contact.board_cells, dtype=np.int64),
            np.ascontiguousarray(body.contact.body_cells, dtype=np.int64),
            np.ascontiguousarray(body.contact.conductance_w_per_k, dtype=np.float64),
            body_start_k(body.body),
        )
        for body in bodies
    ]


def interface_arguments(config: Any) -> dict[str, Any]:
    def options(mpir: Any) -> dict[str, Any]:
        values = solve_options(mpir, "two-level", None, None, 25, 1.0e-4)
        values.pop("reference")
        return values

    return dict(
        max_iterations=int(config.max_iterations),
        temperature_tolerance=float(config.temperature_tolerance_k),
        relative_heat_tolerance=float(config.relative_heat_tolerance),
        relaxation=float(config.relaxation),
        aitken=bool(config.aitken),
        max_relaxation=float(config.max_relaxation),
        divergence_temperature=float(config.divergence_temperature_k),
        board=options(config.board),
        body=options(config.body),
    )


def enclosure_warm_start(result: Any) -> dict[str, Any] | None:
    """The fields of a previous ``BoardEnclosureThermalResult`` the interface loop restarts from."""

    if result is None:
        return None
    return dict(
        board=(np.ascontiguousarray(result.board.temperature_k).reshape(-1), float(result.board.min_temperature_k)),
        bodies=[(np.ascontiguousarray(s.temperature_k).reshape(-1), float(s.min_temperature_k)) for s in result.bodies],
        contact_temperature=[np.ascontiguousarray(t, dtype=np.float64) for t in result.contact_temperature_k],
    )
