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
        "B.Cu": 2,  # the middle sliver joins its neighbours (fix/filament-sliver)
    }
    assert cuda.metrics["filament_counts"] == cpu.metrics["filament_counts"]

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


def test_port_impedance_on_cuda_as_on_the_cpu() -> None:
    """The port matrix needs keep_terminal_components (undriven conductors solved) and warm starts on the device."""
    _require_cuda_device()
    from electrical.sheet_peec import solve_port_impedance_problem
    from test_port_impedance_contract import _problem as _port_problem

    problem = _port_problem(frequencies=(1e6, 1e7, 1e8))  # two strips, two conductors
    common = {"maximum_filaments": 1, "relative_tolerance": 1e-10, "maximum_iterations": 240, "restart": 60}
    cpu = solve_port_impedance_problem(problem, {**common, "warm_start": False})
    cuda = solve_port_impedance_problem(problem, {**common, "execution_backend": "cuda", "device_id": 0})
    assert cuda.metrics["execution_backend"] == "cuda"
    assert [r["started_from_hz"] for r in cuda.metrics["frequencies"]] == [None, 1e6, 1e7]
    for a, b in zip(cpu.impedances, cuda.impedances):
        assert all(a.converged) and all(b.converged)
        # the second conductor carries no net current in the first port's solve: its open voltage is read
        assert abs(b.z_ohm[1, 0]) > 0.0
        assert abs(b.z_ohm - a.z_ohm).max() < 1e-7 * abs(a.z_ohm).max()
    cold = solve_port_impedance_problem(problem, {**common, "execution_backend": "cuda", "warm_start": False})
    assert (sum(r["maximum_iterations"] for r in cuda.metrics["frequencies"][1:])
            < sum(r["maximum_iterations"] for r in cold.metrics["frequencies"][1:]))


def test_the_cudss_quasi_definite_near_preconditioner_solves_as_superlu_does() -> None:
    _require_cuda_device()
    pytest.importorskip("nvmath.sparse.advanced")
    problem = _problem(frequency_hz=3e7)
    common = {"relative_tolerance": 1e-9, "maximum_iterations": 240, "restart": 60,
              "execution_backend": "cuda", "device_id": 0, "preconditioner": "near"}
    superlu = solve_current_field_problem(problem, common)
    cudss = solve_current_field_problem(problem, {**common, "near_factorization": "cudss_quasi_definite"})
    assert cudss.metrics["converged"] and superlu.metrics["converged"]
    assert cudss.metrics["preconditioner_factorization_backend"] == "cudss_quasi_definite_ldlt"
    assert cudss.metrics["solver_relative_residual"] < 1e-9
    assert max(abs(superlu.voltage_phasor[node] - cudss.voltage_phasor[node]) for node in superlu.voltage_phasor) < 1e-9


def test_the_layer_pair_kernel_mixes_as_the_batched_product_does() -> None:
    _require_cuda_device()
    import cupy as cp
    import numpy as np

    from electrical.sheet_peec.current_field_contract import CurrentFieldProblem, _sheet_geometry
    from electrical.sheet_peec.sheet_cuda import CudaSheetInductanceOperator

    problem = CurrentFieldProblem.from_mapping(_problem(frequency_hz=3e7))
    _mesh, operator, _geometry = _sheet_geometry(problem, {"maximum_filaments": 8})
    device = CudaSheetInductanceOperator(operator, cp)
    host = CudaSheetInductanceOperator(operator, np)  # NumPy: the batched matmul path
    rng = np.random.default_rng(11)
    rows, cols = device.shape
    for batch in (1, 2, 3, 5):
        x = rng.standard_normal((batch, device.layer_count, rows, cols))
        y = rng.standard_normal((batch, device.layer_count, rows, cols))
        z = rng.standard_normal((batch, len(device.vertical_levels), rows, cols))
        got = device.apply_batch(cp.asarray(x), cp.asarray(y), cp.asarray(z))
        want = host.apply_batch(x, y, z)
        for a, b in zip(got, want):
            assert np.allclose(cp.asnumpy(a), b, rtol=1e-11, atol=1e-11 * np.abs(b).max())



def test_a_second_frequency_reuses_the_near_plan_and_solves_as_a_fresh_plan_does() -> None:
    _require_cuda_device()
    pytest.importorskip("nvmath.sparse.advanced")
    from electrical.sheet_peec import sheet_cuda

    common = {"relative_tolerance": 1e-9, "maximum_iterations": 240, "restart": 60, "execution_backend": "cuda",
              "device_id": 0, "preconditioner": "near", "near_factorization": "cudss_quasi_definite",
              "maximum_filaments": 1}
    sheet_cuda.clear_cuda_sheet_caches()
    solve_current_field_problem(_problem(frequency_hz=1e6), common)
    planned = dict(sheet_cuda._CUDSS_CACHE)
    reused = solve_current_field_problem(_problem(frequency_hz=3e7), common)
    assert set(sheet_cuda._CUDSS_CACHE) == set(planned)  # one pattern, planned once
    sheet_cuda.clear_cuda_sheet_caches()
    fresh = solve_current_field_problem(_problem(frequency_hz=3e7), common)
    assert reused.metrics["converged"] and fresh.metrics["converged"]
    # both solved to 1e-9; the reused and the fresh factors steer GMRES apart below that
    scale = max(abs(value) for value in fresh.voltage_phasor.values())
    assert max(abs(reused.voltage_phasor[n] - fresh.voltage_phasor[n]) for n in fresh.voltage_phasor) < 1e-8 * scale
    sheet_cuda.clear_cuda_sheet_caches()
