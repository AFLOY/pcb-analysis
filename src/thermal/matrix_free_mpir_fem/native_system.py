"""The C++ thermal core behind the conduction facades.

``electrical._pcbcore.thermal`` holds the problem (mesh arrays, fixed nodes,
loads, convective and radiating faces), the prepared system of one linear
solve and the solves themselves: the linear solve with its re-reference
restart and heat budget, the radiation Newton loop and the backward-Euler
march.  This module only converts the dataclasses of the package to arrays
and the results back; :func:`thermal_core` says whether a call goes to the
core (CPU runtime, core built, ``native`` not ``False``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from electrical import _backend
from electrical.matrix_free_mpir_fem.runtime import LowPrecisionRuntime, RuntimeBackend
from electrical.matrix_free_mpir_fem.solver import MPIRConfig, MPIRResult, mpir_result_from_core
from electrical.threads import thread_budget

from .boundaries import ConvectionBoundary, ExposedFaceConvection
from .mesh import FACE_DIRECTIONS, _flat_index
from .problem import ThermalConductionProblem
from .radiation import ExposedFaceRadiation, RadiationBoundary

LOW_RUNTIME_NAME = "numpy-fp32"


def thermal_core(
    runtime: LowPrecisionRuntime | None, backend: RuntimeBackend | None, native: bool | None
) -> Any:
    """The C++ core when it answers this call, else ``None`` (NumPy or CuPy path)."""

    cuda = bool(getattr(runtime, "is_cuda", False)) if runtime is not None else backend == "cuda"
    if cuda:
        if native:
            raise ValueError("native=True requires the CPU runtime")
        return None
    if native is False:
        return None
    core = _backend.core()
    if native and core is None:
        raise ImportError(
            "the pcbcore extension is not built; run cmake -S . -B build/native && cmake --build build/native"
        )
    return core


def _element_array(value: Any, shape: tuple[int, int, int]) -> np.ndarray:
    return np.ascontiguousarray(np.broadcast_to(np.asarray(value, dtype=np.float64), shape))


def _boundary(boundary: Any, shape: tuple[int, int, int]) -> tuple[int, list[int], np.ndarray, np.ndarray, float]:
    if isinstance(boundary, (ConvectionBoundary, RadiationBoundary)):
        top = boundary.side == "top"
        slab = shape[0] - 1 if top else 0
        coefficient = np.zeros(shape, dtype=np.float64)
        ambient = np.zeros(shape, dtype=np.float64)
        value = boundary.coefficient_w_per_m2_k if isinstance(boundary, ConvectionBoundary) else boundary.emissivity
        coefficient[slab] = np.broadcast_to(np.asarray(value, dtype=np.float64), shape[1:])
        ambient[slab] = np.broadcast_to(np.asarray(boundary.ambient_temperature_k, dtype=np.float64), shape[1:])
        return (1 if top else 0, [5 if top else 4], coefficient, ambient, boundary.mean_ambient_k())
    if isinstance(boundary, (ExposedFaceConvection, ExposedFaceRadiation)):
        value = (
            boundary.coefficient_w_per_m2_k if isinstance(boundary, ExposedFaceConvection) else boundary.emissivity
        )
        return (
            2,
            [FACE_DIRECTIONS.index(direction) for direction in boundary.directions],
            _element_array(value, shape),
            _element_array(boundary.ambient_temperature_k, shape),
            boundary.mean_ambient_k(),
        )
    raise TypeError(f"unsupported boundary {type(boundary).__name__}")


def native_problem(core: Any, problem: ThermalConductionProblem) -> Any:
    """The problem as the core holds it (its nodal load is computed there)."""

    mesh = problem.mesh
    shape = mesh.element_grid_shape
    offsets = np.zeros(len(problem.heat_sources) + 1, dtype=np.int64)
    np.cumsum([len(source.nodes) for source in problem.heat_sources], out=offsets[1:])
    nodes = np.asarray(
        [_flat_index(node, mesh.node_shape) for source in problem.heat_sources for node in source.nodes],
        dtype=np.int64,
    )
    return core.thermal.ThermalProblem(
        np.ascontiguousarray(mesh.active, dtype=np.uint8),
        np.asarray(mesh.slab_thickness_m, dtype=np.float64),
        mesh.pitch_x_m,
        mesh.pitch_y_m,
        mesh.conductivity_w_per_m_k,
        mesh.through_plane_conductivity_w_per_m_k,
        np.ascontiguousarray(problem.fixed_temperature_mask, dtype=np.uint8),
        problem.fixed_temperature_k,
        [_boundary(boundary, shape) for boundary in problem.convection],
        [_boundary(boundary, shape) for boundary in problem.radiation],
        problem.element_heat_w,
        problem.nodal_heat_w,
        offsets,
        nodes,
        np.asarray([source.power_w for source in problem.heat_sources], dtype=np.float64),
    )


def mpir_result(result: dict, low_runtime: str = LOW_RUNTIME_NAME) -> MPIRResult:
    return mpir_result_from_core(result, low_runtime)


def solution_from(result: dict) -> Any:
    """A ``ThermalConductionSolution`` from the core's result."""

    from .solve import ThermalConductionSolution  # solve.py imports this module

    return ThermalConductionSolution(
        temperature_k=result["temperature"],
        heat_flux_w_per_m2=result["heat_flux"],
        max_temperature_k=float(result["max_temperature"]),
        min_temperature_k=float(result["min_temperature"]),
        total_heat_input_w=float(result["total_heat_input"]),
        convective_heat_w=result["convective_heat"],
        fixed_temperature_heat_w=float(result["fixed_temperature_heat"]),
        heat_balance_error_w=float(result["heat_balance_error"]),
        solve=mpir_result(result["solve"]),
        radiative_heat_w=result["radiative_heat"],
        radiation_iterations=int(result["radiation_iterations"]),
        radiation_converged=bool(result["radiation_converged"]),
        radiation_change_k=float(result["radiation_change"]),
        stored_heat_w=float(result["stored_heat"]),
    )


def solve_options(
    config: MPIRConfig | None,
    preconditioner: str,
    coarse_block_nodes: int | None,
    reference_temperature_k: float | None,
    radiation_max_iterations: int,
    radiation_tolerance_k: float,
) -> dict[str, Any]:
    """Keyword arguments of the core's solves."""

    # A stiff copper/laminate stack lets the FP32 inner solve buy only about
    # one decade per outer step; give the FP64 refinement room.
    config = config or MPIRConfig(max_outer_iterations=16)
    return dict(
        relative_tolerance=float(config.relative_tolerance),
        absolute_tolerance=float(config.absolute_tolerance),
        inner_relative_tolerance=float(config.inner_relative_tolerance),
        max_outer_iterations=int(config.max_outer_iterations),
        max_inner_iterations=int(config.max_inner_iterations),
        two_level=preconditioner == "two-level",
        block=0 if coarse_block_nodes is None else int(coarse_block_nodes),
        reference=None if reference_temperature_k is None else float(reference_temperature_k),
        radiation_max_iterations=int(radiation_max_iterations),
        radiation_tolerance=float(radiation_tolerance_k),
        threads=thread_budget(),
    )


def nodal_field(value: np.ndarray | float | None, size: int, name: str) -> np.ndarray | None:
    """A nodal temperature argument as a flat float64 array (a scalar fills it)."""

    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(size, float(array))
    flat = np.ascontiguousarray(array, dtype=np.float64).reshape(-1)
    if flat.size != size:
        raise ValueError(f"{name} must hold one value per node")
    return flat
