"""Real-device acceptance checks for the sheet-PEEC CUDA backend.

This file is intentionally excluded from sandbox CPU test commands.  A skip
means CUDA was not tested; it is not an acceptance result.
"""

from __future__ import annotations

import math

import pytest

from electrical.sheet_peec.current_field_contract import solve_current_field_problem
from test_current_field_contract import _problem


def _require_cuda_device() -> None:
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except (ImportError, OSError, RuntimeError) as error:
        pytest.skip(f"CUDA unavailable: {error}")


def test_multilayer_nonuniform_contract_matches_cpu_on_cuda() -> None:
    _require_cuda_device()
    problem = _problem(frequency_hz=123456.0)
    common = {
        "relative_tolerance": 1e-9,
        "maximum_iterations": 240,
        "restart": 60,
    }
    cpu = solve_current_field_problem(
        problem, {**common, "execution_backend": "cpu"}
    )
    cuda = solve_current_field_problem(
        problem,
        {**common, "execution_backend": "cuda", "device_id": 0},
    )

    assert cpu.metrics["converged"]
    assert cuda.metrics["converged"]
    assert cuda.metrics["backend"] == "cuda_sheet_peec"
    assert cuda.metrics["resolved_backend"] == "cupy-sheet-peec:0"
    assert not cuda.metrics["fallback_used"]
    assert cuda.metrics["solver_relative_residual"] < 1e-9
    assert cuda.metrics["current_closure_error_a"] < 1e-8
    assert cuda.metrics["cuda_memory_pool_peak_bytes"] > 0
    assert cuda.metrics["filament_counts"] == {
        "F.Cu": 1,
        "In1.Cu": 1,
        "B.Cu": 3,
    }

    voltage_difference = max(
        abs(cpu.voltage_phasor[node] - cuda.voltage_phasor[node])
        for node in cpu.voltage_phasor
    )
    density_difference = max(
        math.hypot(
            abs(
                cpu.current_density_phasor[node][0]
                - cuda.current_density_phasor[node][0]
            ),
            abs(
                cpu.current_density_phasor[node][1]
                - cuda.current_density_phasor[node][1]
            ),
        )
        for node in cpu.current_density_phasor
    )
    assert voltage_difference < 1e-9
    assert density_difference < 1e-6


def test_barrel_vias_solve_on_cuda_as_on_the_cpu() -> None:
    _require_cuda_device()
    problem = _problem(frequency_hz=1.0e6)
    for connection in problem["vertical_connections"]:
        connection["barrel"] = {"outer_diameter_mm": 0.3, "drill_diameter_mm": 0.25, "solid_pin": False}
    common = {"relative_tolerance": 1e-10, "maximum_iterations": 240, "restart": 60, "maximum_filaments": 1}
    results = {}
    for model in ("barrel", "column"):
        cpu = solve_current_field_problem(problem, {**common, "via_self_inductance": model, "execution_backend": "cpu"})
        cuda = solve_current_field_problem(
            problem, {**common, "via_self_inductance": model, "execution_backend": "cuda", "device_id": 0}
        )
        assert cuda.metrics["converged"] and cpu.metrics["converged"]
        scale = max(abs(v) for v in cpu.voltage_phasor.values())
        assert max(abs(cpu.voltage_phasor[n] - cuda.voltage_phasor[n]) for n in cpu.voltage_phasor) < 1e-7 * scale
        results[model] = cuda
    # the barrels move the field on the device as they do on the host
    assert results["barrel"].voltage_phasor != results["column"].voltage_phasor
