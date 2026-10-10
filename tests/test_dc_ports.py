"""The copper as an N-port: conductance matrix, unit fields, second-moment loss."""

from __future__ import annotations

import numpy as np
import pytest

from electrical.matrix_free_mpir_fem import (
    CurrentTerminal,
    LayeredPCBMesh,
    MPIRConfig,
    PCBConductionProblem,
    PortSet,
    ViaConnection,
    dc_port_basis,
    solve_pcb_dc,
)
from electrical.matrix_free_mpir_fem import ports as ports_module
from electrical.matrix_free_mpir_fem.native_dc import native_available as native_dc_available
from electrical.matrix_free_mpir_fem.ports import _split_budget
from electrical.threads import thread_budget_scope

TIGHT = MPIRConfig(relative_tolerance=1.0e-12)


def _strip(columns: int = 6, rows: int = 2) -> tuple[LayeredPCBMesh, PortSet]:
    mesh = LayeredPCBMesh(
        element_active=np.ones((1, rows, columns), dtype=bool),
        layer_thickness_m=(35.0e-6,),
        pitch_x_m=1.0e-3,
        pitch_y_m=1.0e-3,
    )
    ports = PortSet(
        pads=(
            tuple((0, r, 0) for r in range(rows + 1)),
            tuple((0, r, columns) for r in range(rows + 1)),
        ),
        names=("in", "out"),
        reference=1,
    )
    return mesh, ports


def _three_port_board() -> tuple[LayeredPCBMesh, PortSet, tuple[ViaConnection, ...]]:
    """Two layers, a slot, a graded grid, two vias; three pads on different layers."""

    active = np.ones((2, 4, 7), dtype=bool)
    active[0, 1:3, 2:4] = False
    mesh = LayeredPCBMesh(
        element_active=active,
        layer_thickness_m=(35.0e-6, 18.0e-6),
        pitch_x_m=(0.4e-3, 0.5e-3, 0.6e-3, 0.5e-3, 0.4e-3, 0.5e-3, 0.6e-3),
        pitch_y_m=(0.5e-3, 0.4e-3, 0.5e-3, 0.6e-3),
        conductivity_s_per_m=(5.8e7, 5.2e7),
    )
    ports = PortSet(
        pads=(
            ((0, 0, 0), (0, 1, 0)),            # phase A feed, top layer
            ((0, 4, 0), (0, 3, 0)),            # phase B feed, top layer
            ((1, 2, 7), (1, 1, 7), (1, 3, 7)), # shared load pad, bottom layer
        ),
        names=("A", "B", "load"),
        reference=2,
    )
    vias = (
        ViaConnection((0, 2, 5), (1, 2, 5), 1.0e-3),
        ViaConnection((0, 1, 6), (1, 1, 6), 1.5e-3),
    )
    return mesh, ports, vias


def test_strip_conductance_and_unit_fields_match_the_closed_form() -> None:
    mesh, ports = _strip()
    basis = dc_port_basis(mesh, ports, config=TIGHT)
    sigma = float(mesh.conductivity_s_per_m[0, 0, 0])
    resistance = 6.0e-3 / (sigma * 2.0e-3 * 35.0e-6)

    assert basis.converged
    expected = np.array([[1.0, -1.0], [-1.0, 1.0]]) / resistance
    np.testing.assert_allclose(basis.conductance_s, expected, rtol=1.0e-9)
    np.testing.assert_allclose(basis.reduced_resistance_ohm, [[resistance]], rtol=1.0e-9)
    # 1 A into "in", out of "out": linear potential from R down to 0.
    phi = basis.potential_v((1.0, -1.0))
    np.testing.assert_allclose(phi[0, 0], resistance * np.linspace(1.0, 0.0, 7), rtol=1.0e-9, atol=1.0e-15)
    np.testing.assert_allclose(basis.port_voltage_v((2.0, -2.0)), (2.0 * resistance, 0.0), rtol=1.0e-9)
    # A constant current loses I² R and has density I / (w t) everywhere.
    element, via = basis.mean_loss_w(np.array([[4.0, -4.0], [-4.0, 4.0]]))
    assert float(np.sum(element)) == pytest.approx(4.0 * resistance, rel=1.0e-9)
    assert via.shape == (0,)
    np.testing.assert_allclose(
        basis.rms_current_density_a_per_m2(np.array([[4.0, -4.0], [-4.0, 4.0]])),
        2.0 / (2.0e-3 * 35.0e-6),
        rtol=1.0e-9,
    )


def test_constant_current_reproduces_the_current_driven_solve_on_a_strip() -> None:
    """With two nodes across, an equal split is a uniform current and the pads coincide."""

    mesh, ports = _strip(rows=1)
    basis = dc_port_basis(mesh, ports, config=TIGHT)
    problem = PCBConductionProblem(
        mesh,
        terminals=(
            CurrentTerminal(ports.pads[0], 1.5, "in"),
            CurrentTerminal(ports.pads[1], -1.5, "out"),
        ),
        reference_node=ports.pads[1][0],
    )
    direct = solve_pcb_dc(problem, config=TIGHT)
    element, _ = basis.mean_loss_w(np.outer((1.5, -1.5), (1.5, -1.5)))
    np.testing.assert_allclose(element, direct.element_joule_loss_w, rtol=1.0e-9, atol=1.0e-16)
    np.testing.assert_allclose(
        basis.potential_v((1.5, -1.5)), np.nan_to_num(direct.potential_v), rtol=1.0e-9, atol=1.0e-14
    )


def test_basis_potentials_equal_direct_voltage_driven_solves() -> None:
    mesh, ports, vias = _three_port_board()
    basis = dc_port_basis(mesh, ports, vias=vias, config=TIGHT)
    assert basis.converged
    assert basis.conductance_s.shape == (3, 3)

    g = basis.conductance_s
    np.testing.assert_allclose(g, g.T, rtol=0.0, atol=1.0e-12 * np.max(np.abs(g)))
    np.testing.assert_allclose(g.sum(axis=1), 0.0, atol=1.0e-9 * np.max(np.abs(g)))
    assert np.all(np.linalg.eigvalsh(basis.reduced_conductance_s) > 0.0)

    # An arbitrary port voltage pattern: the superposition of unit fields is the
    # direct Dirichlet solve, and G V is what that solve reports as terminal current.
    voltages = np.array([3.0e-3, -1.0e-3, 0.0])
    direct = solve_pcb_dc(
        PCBConductionProblem(mesh, voltage_terminals=ports.voltage_terminals(voltages), vias=vias),
        config=TIGHT,
    )
    superposed = np.tensordot(voltages[list(ports.driven)], basis.unit_voltage_potential_v, axes=1)
    np.testing.assert_allclose(superposed, np.nan_to_num(direct.potential_v), rtol=1.0e-9, atol=1.0e-15)
    np.testing.assert_allclose(g @ voltages, direct.voltage_terminal_current_a, rtol=1.0e-8)

    # Currents round-trip through the resistance matrix.
    currents = np.array([0.7, 0.5, -1.2])
    np.testing.assert_allclose(g @ basis.port_voltage_v(currents), currents, rtol=1.0e-8, atol=1.0e-12)
    np.testing.assert_allclose(
        basis.potential_v(currents),
        np.tensordot(basis.port_voltage_v(currents)[list(ports.driven)], basis.unit_voltage_potential_v, axes=1),
        rtol=1.0e-9,
        atol=1.0e-15,
    )


def test_second_moments_reproduce_the_time_average_of_the_instantaneous_loss() -> None:
    """Two interleaved phases into one shared pad: the brute-force average over
    the waveform equals the loss from the correlation matrix, cross terms included."""

    mesh, ports, vias = _three_port_board()
    basis = dc_port_basis(mesh, ports, vias=vias, config=TIGHT)
    t = np.linspace(0.0, 1.0, 240, endpoint=False)
    ripple = lambda phase: 1.0 + 0.6 * (2.0 * np.abs(((t + phase) % 1.0) - 0.5) - 0.5)
    i_a = 2.0 * ripple(0.0)
    i_b = 2.0 * ripple(0.5)                       # 180° interleaved
    waveform = np.stack((i_a, i_b, -(i_a + i_b)))  # (ports, samples), KCL at every sample

    element_bf = np.zeros(mesh.element_active.shape)
    via_bf = np.zeros(len(vias))
    for sample in waveform.T:
        phi = basis.potential_v(sample)
        element_bf += basis.operator.element_joule_loss(phi)
        via_bf += basis.operator.via_joule_loss(phi)
    element_bf /= waveform.shape[1]
    via_bf /= waveform.shape[1]

    correlation = waveform @ waveform.T / waveform.shape[1]
    element, via = basis.mean_loss_w(correlation)
    np.testing.assert_allclose(element, element_bf, rtol=1.0e-10, atol=1.0e-18)
    np.testing.assert_allclose(via, via_bf, rtol=1.0e-10, atol=1.0e-18)

    # Averaging the current first drops the ripple: the loss is lower everywhere it flows.
    mean = waveform.mean(axis=1)
    element_mean, _ = basis.mean_loss_w(np.outer(mean, mean))
    assert float(np.sum(element_mean)) < float(np.sum(element))
    # Per-phase path losses miss the shared copper's cross term 2<I_A I_B> > 0 here.
    only_a = np.zeros((3, 3)); only_a[np.ix_((0, 2), (0, 2))] = correlation[0, 0] * np.array([[1, -1], [-1, 1]])
    only_b = np.zeros((3, 3)); only_b[np.ix_((1, 2), (1, 2))] = correlation[1, 1] * np.array([[1, -1], [-1, 1]])
    separate = basis.mean_loss_w(only_a)[0] + basis.mean_loss_w(only_b)[0]
    assert float(np.sum(separate)) < float(np.sum(element))
    # The interleaved ripple cancels on the shared pad: the cross-correlation of the
    # ripples is negative, so the per-phase sum overestimates the ripple loss there.
    ripple_cov = correlation[0, 1] - mean[0] * mean[1]
    assert ripple_cov < 0.0

    rms = basis.rms_current_density_a_per_m2(correlation)
    assert rms.shape == mesh.element_active.shape
    assert np.all(rms[~mesh.element_active] == 0.0)
    assert np.all(rms[mesh.element_active] >= 0.0)


def test_warm_start_from_a_previous_basis_saves_inner_iterations() -> None:
    mesh, ports, vias = _three_port_board()
    cold = dc_port_basis(mesh, ports, vias=vias)
    heated = LayeredPCBMesh(
        mesh.element_active,
        mesh.layer_thickness_m,
        mesh.pitch_x_m,
        mesh.pitch_y_m,
        conductivity_s_per_m=np.asarray(mesh.conductivity_s_per_m) / 1.1,
    )
    warm = dc_port_basis(heated, ports, vias=vias, initial=cold)
    restart = dc_port_basis(heated, ports, vias=vias)

    assert warm.converged
    assert warm.inner_iterations < restart.inner_iterations
    np.testing.assert_allclose(warm.conductance_s, restart.conductance_s, rtol=1.0e-7)


def test_invalid_ports_and_correlations_are_rejected() -> None:
    mesh, ports, vias = _three_port_board()
    with pytest.raises(ValueError, match="at least two ports"):
        PortSet(pads=(ports.pads[0],))
    with pytest.raises(ValueError, match="belongs to ports"):
        PortSet(pads=(ports.pads[0], ports.pads[0]))
    with pytest.raises(ValueError, match="reference"):
        PortSet(pads=ports.pads, reference=3)
    basis = dc_port_basis(mesh, ports, vias=vias)
    with pytest.raises(ValueError, match="sum to zero"):
        basis.potential_v((1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="3×3"):
        basis.mean_loss_w(np.eye(2))
    with pytest.raises(ValueError, match="rows must sum to zero"):
        basis.mean_loss_w(np.eye(3))
    with pytest.raises(ValueError, match="positive semi-definite"):
        basis.mean_loss_w(-np.array([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]]))
    with pytest.raises(ValueError, match="share the ports"):
        dc_port_basis(mesh, PortSet(pads=ports.pads, reference=0), vias=vias, initial=basis)


def test_split_budget_keeps_pool_times_team_inside_the_budget() -> None:
    for budget in (1, 2, 3, 8, 16):
        for tasks in (1, 2, 8):
            width, team = _split_budget(budget, tasks)
            assert 1 <= width <= max(1, tasks)
            assert team >= 1 and width * team <= budget
    # Provisional rule: a serial pool, the whole budget to each solve's team.
    assert _split_budget(8, 4) == (1, 8)


def test_an_odd_budget_is_split_without_idle_threads() -> None:
    from electrical.matrix_free_mpir_fem.ports import _fit_split

    # Five threads as two teams would use four; one solve on five uses them all.
    assert _fit_split(5, 2, 8) == (1, 5)
    assert _fit_split(5, 5, 8) == (5, 1)
    # Six asked for as four side by side run as three teams of two.
    assert _fit_split(6, 4, 8) == (3, 2)
    assert _fit_split(7, 3, 8) == (1, 7)
    assert _fit_split(9, 3, 8) == (3, 3)
    for budget in range(1, 17):
        for preferred in range(1, 9):
            for tasks in (1, 2, 3, 8):
                width, team = _fit_split(budget, preferred, tasks)
                assert 1 <= width <= min(preferred, tasks, budget)
                assert width * team <= budget
                # No smaller-or-equal width would have used more threads.
                assert width * team == max(
                    w * (budget // w) for w in range(1, min(preferred, tasks, budget) + 1)
                )


@pytest.mark.parametrize("native", [False, pytest.param(True, marks=pytest.mark.skipif(not native_dc_available(), reason="layered DC native extension not built"))])
def test_threaded_unit_solves_give_the_serial_basis(monkeypatch: pytest.MonkeyPatch, native: bool) -> None:
    mesh, ports, vias = _three_port_board()
    with thread_budget_scope(4):
        serial = dc_port_basis(mesh, ports, vias=vias, config=TIGHT, native=native)
        monkeypatch.setattr(ports_module, "_split_budget", lambda budget, tasks: (4, 1))
        threaded = dc_port_basis(mesh, ports, vias=vias, config=TIGHT, native=native)
    assert serial.workers == 1
    assert threaded.workers == 2  # capped at n - 1 unit solves
    if native:
        assert serial.operator._team == 4  # one solve at a time: the whole budget
        assert threaded.operator._team == 1  # the team the split asked for
    assert threaded.converged
    np.testing.assert_array_equal(threaded.conductance_s, serial.conductance_s)
    np.testing.assert_array_equal(threaded.unit_voltage_potential_v, serial.unit_voltage_potential_v)
    np.testing.assert_array_equal(threaded.unit_current_potential_v, serial.unit_current_potential_v)
    assert [r.inner_iterations for r in threaded.solves] == [r.inner_iterations for r in serial.solves]

    # The warm start works on the pool too; a pool wider than the budget is cut to it.
    with thread_budget_scope(1):
        narrow = dc_port_basis(mesh, ports, vias=vias, config=TIGHT, initial=serial, native=native)
    assert narrow.workers == 1
    with thread_budget_scope(4):
        warm = dc_port_basis(mesh, ports, vias=vias, config=TIGHT, initial=serial, native=native)
    assert warm.workers == 2
    np.testing.assert_allclose(warm.conductance_s, serial.conductance_s, rtol=1.0e-9)
    np.testing.assert_array_equal(narrow.conductance_s, warm.conductance_s)
