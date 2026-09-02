"""Real-device acceptance checks for the sheet-PEEC CUDA backend.

This file is intentionally excluded from sandbox CPU test commands.  A skip
means CUDA was not tested; it is not an acceptance result.
"""

from __future__ import annotations

import math

import pytest

from electrical.dice_peec.plane_opt_contract import solve_plane_opt_problem
from test_plane_opt_contract import _problem


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
    cpu = solve_plane_opt_problem(
        problem, {**common, "execution_backend": "cpu"}
    )
    cuda = solve_plane_opt_problem(
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
