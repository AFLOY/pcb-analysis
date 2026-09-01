"""Comparable PEEC/FEM audit on DC strips and a skin-effect slab.

The DC case is deliberately one cell wide.  A sheet-PEEC branch and a Q1 FEM
strip then represent the same copper length, width, thickness, terminal
current, and closed-form resistance.

The skin case is solved by both methods against the same closed-form slab.
FEM solves the one-dimensional field through the slab's thickness directly.
Sheet PEEC has no infinite slab, so it solves a copper bar cut into graded
filaments and reads the through-thickness current division in the middle of
the bar, far from its side faces and its terminals, where the bar is slab-like.
The analytical slab impedance that PEEC's filament module carries is still
reported, but only as the shared reference, never as a PEEC result.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from electrical.dice_peec import (
    SheetInductanceOperator,
    SheetLayer,
    SheetMesh,
    SheetStackup,
    Terminal,
    solve_sheet_case,
    solve_sheet_case_cuda,
)
from electrical.dice_peec.skin_filaments import (
    FilamentStack,
    filament_links,
    graded_filaments,
    slab_surface_impedance,
)
from electrical.matrix_free_mpir_fem import (
    CurrentTerminal,
    LayeredPCBMesh,
    MPIRConfig,
    MatrixFreePCBOperator,
    PCBConductionProblem,
    ScalarMaxwellMesh2D,
    ScalarMaxwellProblem,
    propagation_constant_per_m,
    solve_mpir,
    solve_scalar_maxwell,
)


COPPER_RESISTIVITY_OHM_M = 1.724e-8
COPPER_CONDUCTIVITY_S_PER_M = 1.0 / COPPER_RESISTIVITY_OHM_M
PITCH_M = 0.2e-3
COPPER_THICKNESS_M = 35.0e-6

# The skin-effect slab both methods solve: 0.5 mm of copper at 1 MHz, which is
# 7.6 skin depths thick, so the current lives at the two faces.
SKIN_FREQUENCY_HZ = 1.0e6
SKIN_THICKNESS_M = 0.5e-3
# The PEEC bar is this many cells wide.  Its middle row is then 0.8 mm, twelve
# skin depths, from either side face, and the side-face crowding a bar has and
# a slab does not no longer reaches it.  Narrower bars were measured: at four
# cells the middle profile still departed from the slab by 44%, at eight by 7%,
# and sixteen changed nothing further.
SKIN_BAR_WIDTH_CELLS = 8


def _timed(
    function: Callable[[], Any],
    *,
    repeats: int,
    warmups: int = 1,
    synchronize: Callable[[], None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    result: Any = None
    for _ in range(warmups):
        result = function()
        if synchronize is not None:
            synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        if synchronize is not None:
            synchronize()
        started = time.perf_counter_ns()
        result = function()
        if synchronize is not None:
            synchronize()
        samples.append((time.perf_counter_ns() - started) / 1.0e6)
    return result, {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "samples_ms": samples,
        "repeats": repeats,
    }


def _array_bytes(*values: Any) -> int:
    return sum(int(value.nbytes) for value in values)


def _dc_inputs(elements: int) -> tuple[
    SheetMesh,
    tuple[Terminal, ...],
    PCBConductionProblem,
]:
    cells = elements + 1
    stackup = SheetStackup(
        (
            SheetLayer(
                "Cu",
                0.0,
                COPPER_THICKNESS_M,
                COPPER_RESISTIVITY_OHM_M,
            ),
        )
    )
    peec_mesh = SheetMesh(
        (1, cells),
        PITCH_M,
        stackup,
        np.ones((1, 1, cells), dtype=bool),
    )
    peec_terminals = (
        Terminal("source", 0, ((0, 0),), 1.0),
        Terminal("sink", 0, ((0, cells - 1),), -1.0),
    )

    fem_mesh = LayeredPCBMesh(
        element_active=np.ones((1, 1, elements), dtype=bool),
        layer_thickness_m=(COPPER_THICKNESS_M,),
        pitch_x_m=PITCH_M,
        pitch_y_m=PITCH_M,
    )
    fem_problem = PCBConductionProblem(
        mesh=fem_mesh,
        terminals=(
            CurrentTerminal(((0, 0, 0), (0, 1, 0)), 1.0, "source"),
            CurrentTerminal(
                (
                    (0, 0, elements),
                    (0, 1, elements),
                ),
                -1.0,
                "sink",
            ),
        ),
        reference_node=(0, 0, elements),
    )
    return peec_mesh, peec_terminals, fem_problem


def dc_strip_case(
    elements: int,
    repeats: int,
    *,
    include_cuda: bool,
) -> dict[str, Any]:
    peec_mesh, peec_terminals, fem_problem = _dc_inputs(elements)

    peec_operator, peec_setup = _timed(
        lambda: SheetInductanceOperator(
            peec_mesh.shape, PITCH_M, peec_mesh.stackup
        ),
        repeats=1,
        warmups=0,
    )
    peec_solution, peec_cpu = _timed(
        lambda: solve_sheet_case(
            peec_mesh,
            peec_operator,
            peec_terminals,
            frequency_hz=0.0,
        ),
        repeats=repeats,
    )

    fem_operator, fem_setup = _timed(
        lambda: MatrixFreePCBOperator(
            fem_problem.mesh,
            reference_node=fem_problem.reference_node,
        ),
        repeats=1,
        warmups=0,
    )
    fem_rhs = fem_operator.build_rhs(fem_problem.terminals)
    fem_config = MPIRConfig(
        relative_tolerance=1.0e-11,
        inner_relative_tolerance=2.0e-3,
        max_outer_iterations=12,
        max_inner_iterations=max(300, 2 * elements),
    )
    fem_result, fem_cpu = _timed(
        lambda: solve_mpir(fem_operator, fem_rhs, config=fem_config),
        repeats=repeats,
    )

    exact_resistance = (
        COPPER_RESISTIVITY_OHM_M
        * (elements * PITCH_M)
        / (PITCH_M * COPPER_THICKNESS_M)
    )
    peec_resistance = peec_solution.voltage_span_v()
    fem_resistance = fem_operator.joule_loss(fem_result.solution)
    fem_low_bytes = _array_bytes(
        fem_operator._coefficient_low,
        fem_operator._local_low,
        fem_operator._free_low,
        fem_operator._via_a_low,
        fem_operator._via_b_low,
        fem_operator._via_g_low,
        fem_operator._diagonal_low,
    )

    row: dict[str, Any] = {
        "elements_along_length": elements,
        "length_m": elements * PITCH_M,
        "width_m": PITCH_M,
        "thickness_m": COPPER_THICKNESS_M,
        "exact_resistance_ohm": exact_resistance,
        "peec": {
            "nodes": peec_mesh.node_count,
            "branches": peec_mesh.branch_count,
            "resistance_ohm": peec_resistance,
            "relative_error": abs(peec_resistance - exact_resistance)
            / exact_resistance,
            "residual": peec_solution.residual,
            "operator_setup": peec_setup,
            "cpu_solve": peec_cpu,
            "static_operator_bytes": peec_operator.kernel_bytes,
        },
        "fem": {
            "unknowns": fem_operator.size,
            "resistance_ohm": fem_resistance,
            "relative_error": abs(fem_resistance - exact_resistance)
            / exact_resistance,
            "residual": fem_result.relative_residual,
            "outer_iterations": fem_result.outer_iterations,
            "inner_iterations": fem_result.inner_iterations,
            "operator_setup": fem_setup,
            "cpu_solve": fem_cpu,
            "static_low_operator_bytes": fem_low_bytes,
        },
    }

    if include_cuda:
        import cupy as cp

        peec_cuda_samples = []
        peec_cuda_result = None
        peec_cuda_telemetry = None
        for iteration in range(repeats + 1):
            peec_cuda_result, telemetry = solve_sheet_case_cuda(
                peec_mesh,
                peec_operator,
                peec_terminals,
                frequency_hz=0.0,
            )
            if iteration:
                peec_cuda_samples.append(telemetry.total_ms)
            peec_cuda_telemetry = telemetry
        assert peec_cuda_result is not None and peec_cuda_telemetry is not None

        fem_cuda_operator, fem_cuda_setup = _timed(
            lambda: MatrixFreePCBOperator(
                fem_problem.mesh,
                reference_node=fem_problem.reference_node,
                backend="cuda",
            ),
            repeats=1,
            warmups=0,
            synchronize=cp.cuda.get_current_stream().synchronize,
        )
        fem_cuda_rhs = fem_cuda_operator.build_rhs(fem_problem.terminals)
        fem_cuda_result, fem_cuda_solve = _timed(
            lambda: solve_mpir(
                fem_cuda_operator, fem_cuda_rhs, config=fem_config
            ),
            repeats=repeats,
            synchronize=cp.cuda.get_current_stream().synchronize,
        )
        row["peec"]["cuda_total"] = {
            "median_ms": statistics.median(peec_cuda_samples),
            "minimum_ms": min(peec_cuda_samples),
            "maximum_ms": max(peec_cuda_samples),
            "samples_ms": peec_cuda_samples,
            "repeats": repeats,
        }
        row["peec"]["cuda_last_prepare_ms"] = peec_cuda_telemetry.prepare_ms
        row["peec"]["cuda_last_solve_ms"] = peec_cuda_telemetry.solve_ms
        row["peec"]["cuda_memory_pool_peak_bytes"] = (
            peec_cuda_telemetry.memory_pool_peak_bytes
        )
        row["fem"]["cuda_operator_setup"] = fem_cuda_setup
        row["fem"]["cuda_warm_solve"] = fem_cuda_solve
        row["fem"]["cuda_residual"] = fem_cuda_result.relative_residual
        row["fem"]["cuda_relative_solution_error_vs_cpu"] = float(
            np.linalg.norm(fem_cuda_result.solution - fem_result.solution)
            / np.linalg.norm(fem_result.solution)
        )
    return row


def _closed_form_slab() -> tuple[complex, complex, float]:
    """Give the slab's propagation constant, impedance, and AC/DC ratio."""
    gamma = propagation_constant_per_m(
        SKIN_FREQUENCY_HZ,
        conductivity_s_per_m=COPPER_CONDUCTIVITY_S_PER_M,
    )
    impedance = gamma / (
        2.0
        * COPPER_CONDUCTIVITY_S_PER_M
        * np.tanh(gamma * SKIN_THICKNESS_M / 2.0)
    )
    dc_sheet_resistance = COPPER_RESISTIVITY_OHM_M / SKIN_THICKNESS_M
    return gamma, impedance, float(impedance.real / dc_sheet_resistance)


def _filament_averaged_slab_profile(
    cut: FilamentStack, gamma: complex
) -> np.ndarray:
    """Average the slab's ``cosh`` current profile over each filament.

    A filament carries one current, so the finest statement the cut can make
    of the profile is its mean over each filament.  Comparing the solve with
    this, rather than with the profile at the filament centres, separates what
    the solver got wrong from what the cut could never hold.
    """
    centres = np.asarray(cut.centers_m)
    thicknesses = np.asarray(cut.thicknesses_m)
    upper = np.sinh(gamma * (centres + thicknesses / 2.0))
    lower = np.sinh(gamma * (centres - thicknesses / 2.0))
    return (upper - lower) / (gamma * thicknesses)


def _through_thickness_ratio(
    currents: np.ndarray, thicknesses: np.ndarray
) -> float:
    """Give AC/DC resistance from how a column of filaments shares current.

    The loss in a column of filaments is ``sum rho |I_k|^2 / t_k`` per unit
    width and length; the loss the same total current would cause spread
    evenly through the thickness is ``rho |sum I_k|^2 / T``.  Their ratio is
    the slab's ``R_ac / R_dc`` and depends on the current division alone, so
    it can be read at one column of the bar without the bar's own external
    inductance or its side faces entering.
    """
    thicknesses = np.asarray(thicknesses, dtype=np.float64)
    total = float(thicknesses.sum())
    loss = float(np.sum(np.abs(currents) ** 2 / thicknesses))
    return total * loss / float(abs(np.sum(currents)) ** 2)


def _peec_skin_inputs(
    length_cells: int, cells_per_skin_depth: float
) -> tuple[FilamentStack, SheetMesh, tuple[Terminal, ...]]:
    cut = graded_filaments(
        SKIN_THICKNESS_M,
        0.0,
        SKIN_FREQUENCY_HZ,
        resistivity_ohm_m=COPPER_RESISTIVITY_OHM_M,
        cells_per_skin_depth=cells_per_skin_depth,
    )
    rows, cols = SKIN_BAR_WIDTH_CELLS, length_cells
    stackup = SheetStackup(cut.layers("Cu", COPPER_RESISTIVITY_OHM_M))
    mesh = SheetMesh(
        (rows, cols),
        PITCH_M,
        stackup,
        np.ones((len(stackup), rows, cols), dtype=bool),
        vias=filament_links(
            cut,
            (rows, cols),
            PITCH_M,
            resistivity_ohm_m=COPPER_RESISTIVITY_OHM_M,
        ),
    )
    # The terminals divide the current between filaments in proportion to
    # their thickness, which is the DC division and the wrong one at 1 MHz.
    # The solve has to move the current to the faces itself, and the bar has
    # to be long enough for that to have happened by its middle; the length
    # sweep shows how long.
    terminals: list[Terminal] = []
    for layer, thickness in enumerate(cut.thicknesses_m):
        share = thickness / cut.total_thickness_m
        terminals.append(
            Terminal(
                f"in{layer}", layer, tuple((r, 0) for r in range(rows)), share
            )
        )
        terminals.append(
            Terminal(
                f"out{layer}",
                layer,
                tuple((r, cols - 1) for r in range(rows)),
                -share,
            )
        )
    return cut, mesh, tuple(terminals)


def _middle_column_currents(
    mesh: SheetMesh, solution: Any, layers: int
) -> np.ndarray:
    rows, cols = mesh.shape
    row, col = rows // 2, cols // 2
    return np.array(
        [
            solution.branch_current[mesh.branch_x.index((layer, row, col))]
            for layer in range(layers)
        ]
    )


def peec_skin_effect_case(
    length_cells: int,
    repeats: int,
    *,
    include_cuda: bool = False,
    cells_per_skin_depth: float = 4.0,
) -> dict[str, Any]:
    """Solve the skin-effect slab with sheet PEEC's graded filaments."""
    cut, mesh, terminals = _peec_skin_inputs(length_cells, cells_per_skin_depth)
    gamma, exact_impedance, exact_ratio = _closed_form_slab()
    thicknesses = np.asarray(cut.thicknesses_m)
    ideal = _filament_averaged_slab_profile(cut, gamma)
    # The ratio the exact slab profile gives once it is averaged over these
    # filaments.  It says what the cut alone costs against the closed form; the
    # solve is not bound by it, because the discrete filament system has its
    # own solution rather than the averaged continuous one.
    averaged_ratio = _through_thickness_ratio(ideal * thicknesses, thicknesses)

    operator, setup = _timed(
        lambda: SheetInductanceOperator(
            mesh.shape,
            PITCH_M,
            mesh.stackup,
            vertical_levels=mesh.vertical_levels,
        ),
        repeats=1,
        warmups=0,
    )
    solve_kwargs = dict(
        frequency_hz=SKIN_FREQUENCY_HZ,
        tolerance=1.0e-8,
        restart=120,
        # SciPy counts restart cycles here, so this is 60 * 120 inner steps.
        max_iterations=60,
    )
    # No warm-up: a CPU Krylov solve has nothing to warm, and one of these takes
    # a minute and a half at 64 cells.
    solution, cpu = _timed(
        lambda: solve_sheet_case(mesh, operator, terminals, **solve_kwargs),
        repeats=repeats,
        warmups=0,
    )
    currents = _middle_column_currents(mesh, solution, len(cut))
    ratio = _through_thickness_ratio(currents, thicknesses)
    profile = currents / thicknesses
    profile /= np.sum(profile * thicknesses)
    ideal_profile = ideal / np.sum(ideal * thicknesses)
    profile_error = float(
        np.linalg.norm(profile - ideal_profile) / np.linalg.norm(ideal_profile)
    )

    row: dict[str, Any] = {
        "length_cells": length_cells,
        "length_m": length_cells * PITCH_M,
        "width_cells": SKIN_BAR_WIDTH_CELLS,
        "width_m": SKIN_BAR_WIDTH_CELLS * PITCH_M,
        "thickness_m": SKIN_THICKNESS_M,
        "frequency_hz": SKIN_FREQUENCY_HZ,
        "cells_per_skin_depth": cells_per_skin_depth,
        "filaments": len(cut),
        "surface_filament_m": cut.surface_thickness_m,
        "nodes": mesh.node_count,
        "branches": mesh.branch_count,
        "closed_form_ac_to_dc_ratio": exact_ratio,
        "closed_form_impedance_ohm": {
            "real": exact_impedance.real,
            "imag": exact_impedance.imag,
        },
        "filament_averaged_slab_ac_to_dc_ratio": averaged_ratio,
        "peec_ac_to_dc_ratio": ratio,
        "peec_relative_error_vs_closed_form": abs(ratio - exact_ratio)
        / exact_ratio,
        "peec_relative_error_vs_filament_averaged_slab": abs(
            ratio - averaged_ratio
        )
        / averaged_ratio,
        "peec_profile_relative_l2_error_vs_filament_averaged_slab": profile_error,
        "peec_iterations": solution.iterations,
        "peec_residual": solution.residual,
        "peec_converged": solution.converged,
        "operator_setup": setup,
        "cpu_solve": cpu,
        "static_operator_bytes": operator.kernel_bytes,
    }

    if include_cuda:
        samples: list[float] = []
        cuda_solution = None
        telemetry = None
        for iteration in range(repeats + 1):
            cuda_solution, telemetry = solve_sheet_case_cuda(
                mesh, operator, terminals, **solve_kwargs
            )
            if iteration:
                samples.append(telemetry.total_ms)
        assert cuda_solution is not None and telemetry is not None
        cuda_currents = _middle_column_currents(mesh, cuda_solution, len(cut))
        row["cuda_total"] = {
            "median_ms": statistics.median(samples),
            "minimum_ms": min(samples),
            "maximum_ms": max(samples),
            "samples_ms": samples,
            "repeats": repeats,
        }
        row["cuda_last_prepare_ms"] = telemetry.prepare_ms
        row["cuda_last_solve_ms"] = telemetry.solve_ms
        row["cuda_memory_pool_peak_bytes"] = telemetry.memory_pool_peak_bytes
        row["cuda_iterations"] = cuda_solution.iterations
        row["cuda_residual"] = cuda_solution.residual
        row["cuda_ac_to_dc_ratio"] = _through_thickness_ratio(
            cuda_currents, thicknesses
        )
        row["cuda_relative_solution_error_vs_cpu"] = float(
            np.linalg.norm(cuda_solution.node_voltage - solution.node_voltage)
            / np.linalg.norm(solution.node_voltage)
        )
    return row


def _skin_problem(elements: int) -> ScalarMaxwellProblem:
    frequency_hz = SKIN_FREQUENCY_HZ
    thickness_m = SKIN_THICKNESS_M
    mesh = ScalarMaxwellMesh2D(
        (1, elements),
        thickness_m / elements,
        1.0e-3,
        conductivity_s_per_m=COPPER_CONDUCTIVITY_S_PER_M,
    )
    mask = np.zeros(mesh.node_shape, dtype=bool)
    mask[:, 0] = True
    mask[:, -1] = True
    values = np.zeros(mesh.node_shape, dtype=np.complex128)
    values[:, 0] = 1.0
    values[:, -1] = 1.0
    return ScalarMaxwellProblem(mesh, frequency_hz, mask, values)


def skin_effect_case(elements: int, repeats: int) -> dict[str, Any]:
    problem = _skin_problem(elements)
    config = MPIRConfig(
        relative_tolerance=1.0e-11,
        inner_relative_tolerance=2.0e-3,
        max_outer_iterations=12,
        max_inner_iterations=400,
        gmres_restart=32,
    )
    solution, timing = _timed(
        lambda: solve_scalar_maxwell(problem, config=config),
        repeats=repeats,
    )
    thickness_m = problem.mesh.length_x_m
    frequency_hz = problem.frequency_hz
    nodal = solution.electric_field_z_v_per_m[0]
    sheet_current = COPPER_CONDUCTIVITY_S_PER_M * np.sum(
        0.5 * (nodal[:-1] + nodal[1:])
        * (thickness_m / elements)
    )
    numerical_impedance = 1.0 / sheet_current
    peec_reference = slab_surface_impedance(thickness_m, frequency_hz)
    filament_stack = graded_filaments(
        thickness_m,
        0.0,
        frequency_hz,
        cells_per_skin_depth=4.0,
    )
    gamma = propagation_constant_per_m(
        frequency_hz,
        conductivity_s_per_m=COPPER_CONDUCTIVITY_S_PER_M,
    )
    exact_impedance = gamma / (
        2.0
        * COPPER_CONDUCTIVITY_S_PER_M
        * np.tanh(gamma * thickness_m / 2.0)
    )
    dc_sheet_resistance = COPPER_RESISTIVITY_OHM_M / thickness_m
    return {
        "elements_through_thickness": elements,
        "unknowns": problem.mesh.size,
        "frequency_hz": frequency_hz,
        "thickness_m": thickness_m,
        "fem_ac_to_dc_ratio": float(numerical_impedance.real / dc_sheet_resistance),
        "closed_form_ac_to_dc_ratio": float(
            exact_impedance.real / dc_sheet_resistance
        ),
        "fem_ac_to_dc_ratio_relative_error": float(
            abs(numerical_impedance.real - exact_impedance.real)
            / exact_impedance.real
        ),
        "fem_impedance_ohm": {
            "real": numerical_impedance.real,
            "imag": numerical_impedance.imag,
        },
        "peec_slab_reference_ohm": {
            "real": peec_reference.real,
            "imag": peec_reference.imag,
        },
        "closed_form_impedance_ohm": {
            "real": exact_impedance.real,
            "imag": exact_impedance.imag,
        },
        "fem_relative_error_vs_peec_slab_reference": float(
            abs(numerical_impedance - peec_reference) / abs(peec_reference)
        ),
        "peec_reference_relative_error_vs_closed_form": float(
            abs(peec_reference - exact_impedance) / abs(exact_impedance)
        ),
        "peec_filaments_at_four_per_skin_depth": len(filament_stack),
        "peec_surface_filament_m": filament_stack.surface_thickness_m,
        "fem_cpu_timing": timing,
        "fem_residual": solution.solve.relative_residual,
    }


def _cuda_available() -> bool:
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def run_benchmark(
    *,
    dc_elements: tuple[int, ...],
    skin_elements: tuple[int, ...],
    peec_skin_lengths: tuple[int, ...],
    repeats: int,
    include_cuda: bool,
    peec_skin_repeats: int = 1,
) -> dict[str, Any]:
    cuda = include_cuda and _cuda_available()
    return {
        "schema": "electrical-peec-fem-comparison/v2",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cuda_requested": include_cuda,
            "cuda_executed": cuda,
        },
        "definitions": {
            "dc_case": "same one-cell-wide copper strip and 1 A terminal current",
            "dc_exact_resistance": "rho * length / (width * thickness)",
            "setup_timing": "single cold construction; not a statistical estimate",
            "solve_timing": "median wall-clock time after one warmup",
            "static_operator_bytes": "prepared coefficients/tables only; Krylov vectors excluded",
            "skin_reference": "closed-form infinite-slab impedance; the PEEC filament module's copy of it is reported only as a cross-check",
            "skin_fem": "1D field solve through the slab thickness; AC/DC ratio from the sheet current the surface field drives",
            "skin_peec": "sheet-PEEC bar of graded filaments, current injected in DC proportion at both ends; AC/DC ratio from the filament current division at the middle of the middle row",
            "filament_averaged_slab": "AC/DC ratio the exact slab profile gives once averaged over the same filaments; what the cut alone costs, not a bound on the solve",
            "peec_skin_timing": "cold operator construction once; CPU solve timed without warm-up over --peec-skin-repeats runs; CUDA timed after one warm-up",
        },
        "dc_strip": [
            dc_strip_case(elements, repeats, include_cuda=cuda)
            for elements in dc_elements
        ],
        "skin_effect": [
            skin_effect_case(elements, repeats) for elements in skin_elements
        ],
        "skin_effect_peec": [
            peec_skin_effect_case(
                length, peec_skin_repeats, include_cuda=cuda
            )
            for length in peec_skin_lengths
        ],
        "capabilities": [
            {
                "capability": "DC PCB conduction and vias",
                "sheet_peec": "yes",
                "matrix_free_fem": "yes",
                "comparison": "directly comparable",
            },
            {
                "capability": "magneto-quasistatic partial inductance",
                "sheet_peec": "yes",
                "matrix_free_fem": "not in scalar Ez front end",
                "comparison": "different scope",
            },
            {
                "capability": "skin/proximity effect",
                "sheet_peec": "graded conductor filaments, solved",
                "matrix_free_fem": "conductive cross-section field, solved",
                "comparison": "both solved against the same closed-form slab; different discretisation and different problem size",
            },
            {
                "capability": "dielectric loss and wave propagation",
                "sheet_peec": "no displacement-current dielectric field",
                "matrix_free_fem": "yes, 2D scalar full-wave",
                "comparison": "FEM-only",
            },
            {
                "capability": "arbitrary 3D vector full-wave Maxwell",
                "sheet_peec": "no",
                "matrix_free_fem": "no",
                "comparison": "requires a future 3D edge-element solver",
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dc-elements", nargs="+", type=int, default=(32, 64, 128))
    parser.add_argument("--skin-elements", nargs="+", type=int, default=(16, 32, 64, 128))
    parser.add_argument(
        "--peec-skin-lengths",
        nargs="+",
        type=int,
        default=(16, 32, 64),
        help="sheet-PEEC bar lengths in cells for the skin-effect case",
    )
    parser.add_argument(
        "--peec-skin-repeats",
        type=int,
        default=1,
        help="timed solves per PEEC skin bar; one takes ~90 s at 64 cells",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/peec-fem-comparison.json"),
    )
    args = parser.parse_args()
    sizes = (*args.dc_elements, *args.skin_elements, *args.peec_skin_lengths)
    if (
        args.repeats < 1
        or args.peec_skin_repeats < 1
        or any(value < 1 for value in sizes)
    ):
        parser.error("repeats and element counts must be positive")
    result = run_benchmark(
        dc_elements=tuple(args.dc_elements),
        skin_elements=tuple(args.skin_elements),
        peec_skin_lengths=tuple(args.peec_skin_lengths),
        repeats=args.repeats,
        include_cuda=not args.no_cuda,
        peec_skin_repeats=args.peec_skin_repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
