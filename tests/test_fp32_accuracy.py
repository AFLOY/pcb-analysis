from __future__ import annotations

import numpy as np

from experiments.fp32_accuracy import (
    SolveOutcome,
    _gather_branch_flux,
    build_fixture,
    compare,
    fp64_residuals,
    terminal_impedance,
)
from electrical.sheet_peec.sheet_peec import solve_sheet_case


def test_distributed_terminal_impedance_uses_every_pad_cell() -> None:
    fixture = build_fixture(8, 3e5)
    voltage = np.arange(fixture.mesh.node_count, dtype=np.float64).astype(
        np.complex128
    )
    voltage += 1j * voltage[::-1]

    delivered_power = 0.0j
    for terminal in fixture.terminals:
        indices = [
            fixture.mesh.node_index[(terminal.layer, row, col)]
            for row, col in terminal.cells
        ]
        share = terminal.current_a / len(indices)
        delivered_power += sum(
            voltage[index] * np.conjugate(share) for index in indices
        )
    expected = delivered_power / abs(fixture.current_a) ** 2

    assert terminal_impedance(fixture, voltage) == expected


def test_branch_flux_gather_excludes_fft_embedding_cells() -> None:
    fixture = build_fixture(8, 3e5)
    mesh = fixture.mesh
    rows, cols = mesh.shape
    flux_x = np.arange(2 * rows * cols, dtype=float).reshape(2, rows, cols)
    flux_y = flux_x + 1000.0
    flux_z = np.arange(rows * cols, dtype=float).reshape(1, rows, cols) + 2000.0

    gathered = _gather_branch_flux(mesh, flux_x, flux_y, flux_z)

    assert gathered.shape == (mesh.branch_count,)
    assert gathered[-1] == flux_z[
        0, mesh.via_branches[-1].row, mesh.via_branches[-1].col
    ]


def test_identical_solution_has_small_fp64_block_residuals_and_zero_gate_gap() -> None:
    fixture = build_fixture(8, 3e5)
    reference = solve_sheet_case(
        fixture.mesh,
        fixture.operator,
        fixture.terminals,
        frequency_hz=fixture.frequency_hz,
        tolerance=1e-11,
    )
    replay = fp64_residuals(
        fixture, reference.node_voltage, reference.branch_current
    )
    outcome = SolveOutcome(
        precision="fp64",
        converged=True,
        working_precision_mixed_unit_residual=reference.residual,
        requested_rtol=1e-10,
        iterations=reference.iterations,
        prepare_ms=0.0,
        factor_ms=0.0,
        solve_ms=0.0,
        total_ms=0.0,
        voltage=reference.node_voltage.copy(),
        current=reference.branch_current.copy(),
        preconditioner_dtype="complex128",
    )

    measured = compare(fixture, reference, outcome)

    assert replay["fp64_max_block_relative_residual"] < 1e-9
    assert measured["max_gate_metric_relative_error"] == 0.0
    assert measured["relative_resistance_error"] == 0.0
    assert measured["relative_inductance_error"] == 0.0
