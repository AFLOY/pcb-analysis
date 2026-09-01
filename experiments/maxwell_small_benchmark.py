"""Accuracy and CPU timing audit for scalar full-wave MPIR-FEM.

The benchmark uses one-element-high strips so every case has an independent
closed-form 1D Maxwell solution while still exercising the production 2D Q1
operator.  Timing is a small-problem CPU measurement, not a CUDA or Tenstorrent
performance claim.
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
import scipy
from scipy.sparse.linalg import LinearOperator, gmres

from electrical.matrix_free_mpir_fem import (
    EPSILON_0_F_PER_M,
    MU_0_H_PER_M,
    MPIRConfig,
    MatrixFreeScalarMaxwellOperator,
    ScalarMaxwellMesh2D,
    ScalarMaxwellProblem,
    propagation_constant_per_m,
    skin_depth_m,
    solve_mpir,
    solve_scalar_maxwell,
)


COPPER_CONDUCTIVITY_S_PER_M = 1.0 / 1.724e-8


def _problem(
    mesh: ScalarMaxwellMesh2D,
    frequency_hz: float,
    *,
    right_value: complex = 0.0,
) -> ScalarMaxwellProblem:
    mask = np.zeros(mesh.node_shape, dtype=bool)
    mask[:, 0] = True
    mask[:, -1] = True
    values = np.zeros(mesh.node_shape, dtype=np.complex128)
    values[:, 0] = 1.0
    values[:, -1] = right_value
    return ScalarMaxwellProblem(mesh, frequency_hz, mask, values)


def _config() -> MPIRConfig:
    return MPIRConfig(
        relative_tolerance=1.0e-11,
        inner_relative_tolerance=2.0e-3,
        max_outer_iterations=12,
        max_inner_iterations=400,
        gmres_restart=32,
    )


def _timed(
    function: Callable[[], Any],
    *,
    repeats: int,
    warmups: int = 1,
) -> tuple[Any, dict[str, Any]]:
    result: Any = None
    for _ in range(warmups):
        result = function()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = function()
        samples.append((time.perf_counter_ns() - started) / 1.0e6)
    return result, {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "samples_ms": samples,
        "repeats": repeats,
    }


def dielectric_wave_case(elements: int, repeats: int) -> dict[str, Any]:
    frequency_hz = 1.0e9
    relative_permittivity = 4.0
    length_m = 20.0e-3
    mesh = ScalarMaxwellMesh2D(
        (1, elements),
        length_m / elements,
        1.0e-3,
        relative_permittivity=relative_permittivity,
    )
    problem = _problem(mesh, frequency_hz)
    solution, timing = _timed(
        lambda: solve_scalar_maxwell(problem, config=_config()), repeats=repeats
    )
    x = np.linspace(0.0, length_m, elements + 1)
    wave_number = 2.0 * np.pi * frequency_hz * np.sqrt(
        MU_0_H_PER_M * EPSILON_0_F_PER_M * relative_permittivity
    )
    exact = np.sin(wave_number * (length_m - x)) / np.sin(
        wave_number * length_m
    )
    error = np.linalg.norm(solution.electric_field_z_v_per_m[0] - exact)
    error /= np.linalg.norm(exact)
    return {
        "case": "lossless_dielectric_full_wave",
        "elements": elements,
        "unknowns": mesh.size,
        "relative_l2_field_error": float(error),
        "relative_residual": solution.solve.relative_residual,
        "outer_iterations": solution.solve.outer_iterations,
        "inner_iterations": solution.solve.inner_iterations,
        "high_operator_applications": solution.solve.high_operator_applications,
        "low_operator_applications": solution.solve.low_operator_applications,
        "timing": timing,
    }


def lossy_dielectric_case(elements: int, repeats: int) -> dict[str, Any]:
    frequency_hz = 2.0e9
    relative_permittivity = 4.2
    loss_tangent = 0.02
    length_m = 15.0e-3
    mesh = ScalarMaxwellMesh2D(
        (1, elements),
        length_m / elements,
        1.0e-3,
        relative_permittivity=relative_permittivity,
        dielectric_loss_tangent=loss_tangent,
    )
    problem = _problem(mesh, frequency_hz)
    solution, timing = _timed(
        lambda: solve_scalar_maxwell(problem, config=_config()), repeats=repeats
    )
    gamma = propagation_constant_per_m(
        frequency_hz,
        relative_permittivity=relative_permittivity,
        dielectric_loss_tangent=loss_tangent,
    )
    x = np.linspace(0.0, length_m, elements + 1)
    exact = np.sinh(gamma * (length_m - x)) / np.sinh(gamma * length_m)
    error = np.linalg.norm(solution.electric_field_z_v_per_m[0] - exact)
    error /= np.linalg.norm(exact)
    return {
        "case": "lossy_dielectric_full_wave",
        "elements": elements,
        "unknowns": mesh.size,
        "relative_l2_field_error": float(error),
        "dielectric_loss_w_per_m": solution.dielectric_loss_w_per_m,
        "relative_residual": solution.solve.relative_residual,
        "outer_iterations": solution.solve.outer_iterations,
        "inner_iterations": solution.solve.inner_iterations,
        "high_operator_applications": solution.solve.high_operator_applications,
        "low_operator_applications": solution.solve.low_operator_applications,
        "timing": timing,
    }


def skin_effect_case(elements: int, repeats: int) -> dict[str, Any]:
    frequency_hz = 1.0e6
    thickness_m = 0.5e-3
    mesh = ScalarMaxwellMesh2D(
        (1, elements),
        thickness_m / elements,
        1.0e-3,
        conductivity_s_per_m=COPPER_CONDUCTIVITY_S_PER_M,
    )
    problem = _problem(mesh, frequency_hz, right_value=1.0)
    solution, timing = _timed(
        lambda: solve_scalar_maxwell(problem, config=_config()), repeats=repeats
    )
    gamma = propagation_constant_per_m(
        frequency_hz, conductivity_s_per_m=COPPER_CONDUCTIVITY_S_PER_M
    )
    x = np.linspace(-thickness_m / 2.0, thickness_m / 2.0, elements + 1)
    exact = np.cosh(gamma * x) / np.cosh(gamma * thickness_m / 2.0)
    field_error = np.linalg.norm(solution.electric_field_z_v_per_m[0] - exact)
    field_error /= np.linalg.norm(exact)
    nodal = solution.electric_field_z_v_per_m[0]
    sheet_current = COPPER_CONDUCTIVITY_S_PER_M * np.sum(
        0.5 * (nodal[:-1] + nodal[1:]) * (thickness_m / elements)
    )
    numerical_impedance = 1.0 / sheet_current
    exact_impedance = gamma / (
        2.0
        * COPPER_CONDUCTIVITY_S_PER_M
        * np.tanh(gamma * thickness_m / 2.0)
    )
    impedance_error = abs(numerical_impedance - exact_impedance) / abs(
        exact_impedance
    )
    dc_sheet_resistance = 1.0 / (
        COPPER_CONDUCTIVITY_S_PER_M * thickness_m
    )
    return {
        "case": "copper_eddy_current_skin_effect",
        "elements": elements,
        "unknowns": mesh.size,
        "skin_depth_m": skin_depth_m(
            frequency_hz, COPPER_CONDUCTIVITY_S_PER_M
        ),
        "elements_per_skin_depth": (
            skin_depth_m(frequency_hz, COPPER_CONDUCTIVITY_S_PER_M)
            / (thickness_m / elements)
        ),
        "relative_l2_field_error": float(field_error),
        "relative_surface_impedance_error": float(impedance_error),
        "ac_to_dc_resistance_ratio": float(
            numerical_impedance.real / dc_sheet_resistance
        ),
        "conduction_loss_w_per_m": solution.conduction_loss_w_per_m,
        "relative_residual": solution.solve.relative_residual,
        "outer_iterations": solution.solve.outer_iterations,
        "inner_iterations": solution.solve.inner_iterations,
        "high_operator_applications": solution.solve.high_operator_applications,
        "low_operator_applications": solution.solve.low_operator_applications,
        "timing": timing,
    }


def solver_speed_case(elements: int, repeats: int) -> dict[str, Any]:
    mesh = ScalarMaxwellMesh2D(
        (1, elements),
        20.0e-3 / elements,
        1.0e-3,
        relative_permittivity=4.0,
        dielectric_loss_tangent=0.01,
    )
    problem = _problem(mesh, 1.0e9)
    operator = MatrixFreeScalarMaxwellOperator(problem)
    rhs = operator.build_rhs()
    identity = np.eye(operator.size, dtype=np.complex128)
    dense = np.column_stack(
        [operator.apply_high(identity[:, column]) for column in range(operator.size)]
    )

    mpir_result, mpir_timing = _timed(
        lambda: solve_mpir(operator, rhs, config=_config()), repeats=repeats
    )

    linear = LinearOperator(
        (operator.size, operator.size),
        matvec=operator.apply_high,
        dtype=np.complex128,
    )
    inverse_diagonal = 1.0 / np.diag(dense)
    jacobi = LinearOperator(
        (operator.size, operator.size),
        matvec=lambda vector: inverse_diagonal * vector,
        dtype=np.complex128,
    )
    fp64_iterations = 0

    def solve_fp64_gmres() -> np.ndarray:
        nonlocal fp64_iterations
        fp64_iterations = 0

        def count_iteration(_: float) -> None:
            nonlocal fp64_iterations
            fp64_iterations += 1

        solution, info = gmres(
            linear,
            rhs,
            M=jacobi,
            rtol=1.0e-11,
            atol=0.0,
            restart=32,
            maxiter=50,
            callback=count_iteration,
            callback_type="pr_norm",
        )
        if info != 0:
            raise RuntimeError(f"complex128 GMRES failed with info={info}")
        return solution

    fp64_solution, fp64_timing = _timed(solve_fp64_gmres, repeats=repeats)
    direct_solution, direct_timing = _timed(
        lambda: np.linalg.solve(dense, rhs), repeats=repeats
    )
    vector = np.linspace(0.0, 1.0, operator.size).astype(np.complex128)
    low_vector = operator.runtime.from_host(vector)
    _, high_apply = _timed(
        lambda: operator.apply_high(vector), repeats=max(50, repeats * 10)
    )
    _, low_apply = _timed(
        lambda: operator.apply_low(low_vector), repeats=max(50, repeats * 10)
    )

    return {
        "case": "solver_speed_cpu",
        "elements": elements,
        "unknowns": operator.size,
        "matrix_assembly_excluded_from_direct_timing": True,
        "mpir": {
            "timing": mpir_timing,
            "relative_error_vs_dense_direct": float(
                np.linalg.norm(mpir_result.solution - direct_solution)
                / np.linalg.norm(direct_solution)
            ),
            "relative_residual": mpir_result.relative_residual,
            "high_operator_applications": mpir_result.high_operator_applications,
            "low_operator_applications": mpir_result.low_operator_applications,
        },
        "complex128_matrix_free_gmres": {
            "timing": fp64_timing,
            "iterations": fp64_iterations,
            "preconditioner": "Jacobi, matching MPIR inner solve",
            "relative_error_vs_dense_direct": float(
                np.linalg.norm(fp64_solution - direct_solution)
                / np.linalg.norm(direct_solution)
            ),
        },
        "dense_direct_preassembled": {"timing": direct_timing},
        "operator_apply": {
            "complex128_matrix_free": high_apply,
            "complex64_matrix_free": low_apply,
        },
    }


def run_benchmark(repeats: int = 5) -> dict[str, Any]:
    return {
        "schema": "electrical-maxwell-small-benchmark/v1",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "accelerator": "none; NumPy CPU runtimes",
        },
        "definitions": {
            "field_error": "nodal relative L2 norm against closed-form 1D Maxwell field",
            "impedance_error": "relative complex magnitude error against conducting-slab surface impedance",
            "timing": "median wall-clock milliseconds after one warmup",
            "phasor_convention": "exp(+j omega t), peak-amplitude fields",
        },
        "dielectric_convergence": [
            dielectric_wave_case(elements, repeats) for elements in (8, 16, 32, 64)
        ],
        "lossy_dielectric_convergence": [
            lossy_dielectric_case(elements, repeats) for elements in (12, 24, 48)
        ],
        "skin_effect_convergence": [
            skin_effect_case(elements, repeats) for elements in (16, 32, 64, 128)
        ],
        "speed": solver_speed_case(64, repeats),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/maxwell-small.json"),
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    result = run_benchmark(args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
