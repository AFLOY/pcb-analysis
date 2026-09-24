from __future__ import annotations

import math

import pytest

from electrical.sheet_peec.current_field_contract import (
    CURRENT_FIELD_PROBLEM_SCHEMA,
    CURRENT_FIELD_RESULT_SCHEMA,
    CurrentFieldProblem,
    build_current_field_sheet_inputs,
    solve_current_field_problem,
)
from electrical.sheet_peec.sheet_cuda import CudaSheetTelemetry


def _problem(*, frequency_hz: float = 0.0) -> dict:
    layers = [
        {
            "name": "F.Cu",
            "order": 0,
            "center_z_mm": 0.0,
            "thickness_mm": 0.035,
            "resistivity_ohm_m": 1.724e-8,
            "thickness_source": "board_stackup",
        },
        {
            "name": "In1.Cu",
            "order": 1,
            "center_z_mm": -0.1525,
            "thickness_mm": 0.07,
            "resistivity_ohm_m": 1.724e-8,
            "thickness_source": "board_stackup",
        },
        {
            "name": "B.Cu",
            "order": 2,
            "center_z_mm": -0.44,
            "thickness_mm": 0.105,
            "resistivity_ohm_m": 1.724e-8,
            "thickness_source": "board_stackup",
        },
    ]
    copper = {
        layer["name"]: [{"x": x, "y": 0} for x in range(3)]
        for layer in layers
    }
    connections = []
    for index, x in enumerate((0, 2)):
        connections.append(
            {
                "name": f"vertical_{index:04d}",
                "cell": {"x": x, "y": 0},
                "layers": ["F.Cu", "In1.Cu", "B.Cu"],
                "segments": [
                    {
                        "upper_layer": "F.Cu",
                        "lower_layer": "In1.Cu",
                        "resistance_ohm": 0.0003448275862068966,
                        "length_mm": 0.1525,
                    },
                    {
                        "upper_layer": "In1.Cu",
                        "lower_layer": "B.Cu",
                        "resistance_ohm": 0.0006551724137931035,
                        "length_mm": 0.2875,
                    },
                ],
            }
        )
    return {
        "schema": CURRENT_FIELD_PROBLEM_SCHEMA,
        "name": "three_layer_return",
        "role": "POWER",
        "state": "steady",
        "weight": 1.0,
        "frequency_hz": frequency_hz,
        "grid": {
            "min_x_mm": 0.0,
            "min_y_mm": 0.0,
            "columns": 3,
            "rows": 1,
            "pitch_mm": 0.2,
        },
        "layers": layers,
        "copper_by_layer": copper,
        "vertical_connections": connections,
        "terminals": [
            {
                "name": "terminal_000",
                "pad": "SRC.1",
                "current_a": {"real": 1.0, "imag": 0.0},
                "cells": [{"layer": "F.Cu", "x": 0, "y": 0}],
            },
            {
                "name": "terminal_001",
                "pad": "SNK.1",
                "current_a": {"real": -1.0, "imag": 0.0},
                "cells": [{"layer": "B.Cu", "x": 2, "y": 0}],
            },
        ],
        "current_balance_tolerance_a": 1e-9,
        "source_board_sha256": "fixture",
    }


def test_parser_preserves_dynamic_layer_geometry() -> None:
    problem = CurrentFieldProblem.from_mapping(_problem())

    assert [layer.thickness_mm for layer in problem.layers] == [
        0.035,
        0.07,
        0.105,
    ]
    assert [layer.center_z_mm for layer in problem.layers] == [
        0.0,
        -0.1525,
        -0.44,
    ]
    assert len(problem.vertical_segments) == 4


def test_parser_accepts_balanced_complex_currents() -> None:
    value = _problem(frequency_hz=1000.0)
    value["terminals"][0]["current_a"] = {"real": 0.0, "imag": 1.0}
    value["terminals"][1]["current_a"] = {"real": 0.0, "imag": -1.0}

    problem = CurrentFieldProblem.from_mapping(value)

    assert problem.terminals[0].current_a == 1.0j
    assert problem.terminals[1].current_a == -1.0j


def test_parser_rejects_terminal_outside_conductor() -> None:
    value = _problem()
    value["terminals"][0]["cells"][0]["y"] = 1

    with pytest.raises(ValueError, match="cell is not conductor"):
        CurrentFieldProblem.from_mapping(value)


def test_parser_rejects_the_pre_0_9_0_schema_string() -> None:
    """0.9.0 renamed plane-opt-current-field-problem/v1 without keeping an alias."""

    value = _problem()
    value["schema"] = "plane-opt-current-field-problem/v1"

    with pytest.raises(ValueError, match="unsupported current-field problem schema"):
        CurrentFieldProblem.from_mapping(value)


def test_sheet_mesh_uses_each_layer_own_thickness() -> None:
    mesh, _, terminals, context = build_current_field_sheet_inputs(_problem())

    for actual, expected in zip(
        [layer.thickness_m for layer in mesh.stackup.layers],
        [35e-6, 70e-6, 105e-6],
    ):
        assert math.isclose(actual, expected)
    for actual, expected in zip(
        [layer.z_m for layer in mesh.stackup.layers],
        [0.0, -0.0001525, -0.00044],
    ):
        assert math.isclose(actual, expected)
    assert context.filament_counts == {
        "F.Cu": 1,
        "In1.Cu": 1,
        "B.Cu": 1,
    }
    assert sum(terminal.current_a for terminal in terminals) == 0.0j


def test_end_to_end_dc_contract_result() -> None:
    result = solve_current_field_problem(_problem())

    assert result.metrics["converged"]
    assert result.metrics["problem_schema"] == CURRENT_FIELD_PROBLEM_SCHEMA
    assert result.metrics["result_schema"] == CURRENT_FIELD_RESULT_SCHEMA
    assert result.metrics["source_board_sha256"] == "fixture"
    assert result.metrics["resolved_backend"] == "numpy-scipy-sheet-peec"
    assert not result.metrics["fallback_used"]
    assert result.metrics["resolved_layers"][1]["thickness_mm"] == 0.07
    assert result.metrics["current_closure_error_a"] < 1e-8
    assert result.metrics["voltage_span_v"] > 0.0
    assert result.metrics["current_density_definition"].startswith(
        "maximum_over_thickness_filaments"
    )
    assert math.isclose(
        abs(result.current_density_phasor[("F.Cu", 1, 0)][0]),
        result.current_density[("F.Cu", 1, 0)],
    )


def test_end_to_end_ac_solution_tracks_terminal_phase() -> None:
    real_value = _problem(frequency_hz=1000.0)
    quadrature_value = _problem(frequency_hz=1000.0)
    quadrature_value["terminals"][0]["current_a"] = {
        "real": 0.0,
        "imag": 1.0,
    }
    quadrature_value["terminals"][1]["current_a"] = {
        "real": 0.0,
        "imag": -1.0,
    }

    real_result = solve_current_field_problem(real_value)
    quadrature_result = solve_current_field_problem(quadrature_value)

    assert real_result.metrics["converged"]
    assert quadrature_result.metrics["converged"]
    for node, voltage in real_result.voltage_phasor.items():
        assert quadrature_result.voltage_phasor[node] == pytest.approx(
            1.0j * voltage,
            rel=1e-7,
            abs=1e-12,
        )


def test_contract_rejects_implicit_backend_fallback() -> None:
    with pytest.raises(ValueError, match="do not allow implicit"):
        solve_current_field_problem(
            _problem(),
            {"execution_backend": "cuda", "fallback_backend": "cpu"},
        )


def test_cuda_telemetry_records_explicit_device_backend() -> None:
    metrics = CudaSheetTelemetry(
        resolved_backend="cupy-sheet-peec:0",
        device_id=0,
        device_name="fixture-device",
        compute_capability="7.5",
        driver_version=13030,
        runtime_version=13020,
        cupy_version="14.1.1",
        prepare_ms=1.0,
        solve_ms=2.0,
        total_ms=3.0,
        krylov_relative_tolerance=1e-12,
        memory_pool_peak_bytes=4096,
        free_memory_before_bytes=8192,
        free_memory_after_bytes=4096,
        preconditioner_factorization_backend="fixture",
    ).as_metrics()

    assert metrics["backend"] == "cuda_sheet_peec"
    assert metrics["resolved_backend"] == "cupy-sheet-peec:0"
    assert not metrics["fallback_used"]
