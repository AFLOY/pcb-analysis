"""Electro-thermal coupling with the currents decided by an external circuit."""

from __future__ import annotations

import numpy as np
import pytest

from electrical.matrix_free_mpir_fem import (
    CurrentTerminal,
    LayeredPCBMesh,
    PCBConductionProblem,
    PortSet,
    ViaConnection,
    VoltageTerminal,
)
from multiphysics.staggered_coupling import (
    CircuitCoupledResult,
    CircuitCoupledScenario,
    CoupledConductor,
    CouplingConfig,
    ElectroThermalScenario,
    LinearTheveninCircuit,
    PortExcitation,
    TheveninPort,
    run_circuit_coupled,
    run_electro_thermal,
    run_scenario,
)
from thermal.matrix_free_mpir_fem import ConvectionBoundary, LayeredThermalMesh

AMBIENT = 298.15
PITCH = 0.5e-3
ROWS, COLS = 6, 24


def _loop_mesh() -> tuple[LayeredPCBMesh, PortSet, tuple[ViaConnection, ...]]:
    active = np.zeros((2, ROWS, COLS), dtype=bool)
    active[:, 1:5, :] = True
    mesh = LayeredPCBMesh(active, (35e-6, 35e-6), PITCH, PITCH)
    node_rows = tuple(range(1, 6))
    vias = tuple(ViaConnection((0, r, c), (1, r, c), 1.5e-3) for r in node_rows[1:-1] for c in (COLS - 1, COLS))
    ports = PortSet(
        pads=(tuple((1, r, 0) for r in node_rows), tuple((0, r, 0) for r in node_rows)),
        names=("source", "return"),
        reference=1,
    )
    return mesh, ports, vias


def _thermal_mesh(active: np.ndarray) -> LayeredThermalMesh:
    k = np.full((3, ROWS, COLS), 0.8)
    k[0] = np.where(active[0], 385.0, 0.8)
    k[2] = np.where(active[1], 385.0, 0.8)
    kz = k.copy()
    kz[1] = 0.3
    return LayeredThermalMesh((35e-6, 1.5e-3, 35e-6), PITCH, PITCH, k, through_plane_conductivity_w_per_m_k=kz)


def _convection() -> tuple[ConvectionBoundary, ...]:
    return (ConvectionBoundary("top", 10.0, AMBIENT), ConvectionBoundary("bottom", 10.0, AMBIENT))


def _coupled(circuit, **overrides) -> CircuitCoupledScenario:
    mesh, ports, vias = _loop_mesh()
    options = dict(convection=_convection(), conductivity_reference_temperature_k=293.15)
    options.update(overrides)
    return CircuitCoupledScenario(
        (CoupledConductor("loop", mesh, ports, (0, 2), vias),), _thermal_mesh(mesh.element_active), circuit, **options
    )


def _voltage_for_cold_current(current_a: float) -> float:
    mesh, ports, vias = _loop_mesh()
    probe = run_circuit_coupled(
        _coupled(LinearTheveninCircuit({"loop": (TheveninPort(1.0), TheveninPort(0.0))}), temperature_coefficient_per_k=0.0)
    )
    return current_a / float(probe.conductors["loop"].excitation.mean_current_a[0])


def test_ideal_voltage_ports_reproduce_the_voltage_driven_coupling() -> None:
    mesh, ports, vias = _loop_mesh()
    voltage = _voltage_for_cold_current(2.0)
    reference = run_electro_thermal(
        ElectroThermalScenario(
            PCBConductionProblem(
                mesh,
                voltage_terminals=(VoltageTerminal(ports.pads[0], voltage, "source"), VoltageTerminal(ports.pads[1], 0.0, "return")),
                vias=vias,
            ),
            _thermal_mesh(mesh.element_active),
            (0, 2),
            convection=_convection(),
            conductivity_reference_temperature_k=293.15,
        )
    )
    result = run_circuit_coupled(_coupled(LinearTheveninCircuit({"loop": (TheveninPort(voltage), TheveninPort(0.0))})))

    assert result.converged and reference.converged
    assert result.joule_loss_w == pytest.approx(reference.electrical.joule_loss_w, rel=1e-6)
    assert result.cold_joule_loss_w == pytest.approx(reference.cold_joule_loss_w, rel=1e-8)
    assert result.thermal.max_temperature_k == pytest.approx(reference.thermal.max_temperature_k, abs=1e-4)
    assert result.loss_increase_ratio < 0.9
    state = result.conductors["loop"]
    np.testing.assert_allclose(
        state.excitation.mean_current_a, reference.electrical.voltage_terminal_current_a, rtol=1e-6
    )
    np.testing.assert_allclose(state.element_loss_w, reference.electrical.element_joule_loss_w, rtol=1e-5, atol=1e-15)
    assert result.thermal.total_heat_input_w == pytest.approx(result.joule_loss_w, rel=1e-12)
    np.testing.assert_allclose(state.element_temperature_k, reference.element_temperature_k, atol=1e-3)


def test_a_stiff_source_approaches_the_deprecated_current_drive() -> None:
    mesh, ports, vias = _loop_mesh()
    current = 2.0
    with pytest.warns(DeprecationWarning, match="current-driven"):
        reference = run_electro_thermal(
            ElectroThermalScenario(
                PCBConductionProblem(
                    mesh,
                    (CurrentTerminal(ports.pads[0], current, "source"), CurrentTerminal(ports.pads[1], -current, "return")),
                    reference_node=ports.pads[1][0],
                    vias=vias,
                ),
                _thermal_mesh(mesh.element_active),
                (0, 2),
                convection=_convection(),
                conductivity_reference_temperature_k=293.15,
            )
        )
    stiff = 1.0e3
    result = run_circuit_coupled(
        _coupled(LinearTheveninCircuit({"loop": (TheveninPort(current * stiff, stiff), TheveninPort(0.0))}))
    )
    assert result.converged
    assert result.loss_increase_ratio == pytest.approx(reference.loss_increase_ratio, rel=1e-3)
    assert result.loss_increase_ratio > 1.1
    # The pad model differs (equipotential against uniform current), so the
    # absolute loss agrees only to the pad effect, not to solver precision.
    assert result.joule_loss_w == pytest.approx(reference.electrical.joule_loss_w, rel=2e-2)


def test_thevenin_source_and_load_follow_the_closed_form_at_fixed_temperature() -> None:
    source_r, load_r, v_th = 0.02, 0.05, 1.0
    circuit = LinearTheveninCircuit({"loop": (TheveninPort(v_th, source_r), TheveninPort(0.0, load_r))})
    result = run_circuit_coupled(_coupled(circuit, temperature_coefficient_per_k=0.0))
    state = result.conductors["loop"]
    r_cu = float(state.basis.reduced_resistance_ohm[0, 0])
    expected = v_th / (source_r + r_cu + load_r)

    assert result.converged and result.iterations == 1
    np.testing.assert_allclose(state.excitation.mean_current_a, (expected, -expected), rtol=1e-9)
    assert state.joule_loss_w == pytest.approx(expected**2 * r_cu, rel=1e-9)
    port_voltage = state.excitation.metadata["port_voltage_v"]
    assert port_voltage[0] == pytest.approx(v_th - expected * source_r, rel=1e-9)
    assert port_voltage[1] == pytest.approx(expected * load_r, rel=1e-9)


class _TwoPhaseCircuit:
    """Two phases into one shared pad, with or without 180°-interleaved ripple."""

    def __init__(self, amplitude_a: float, ripple: float) -> None:
        t = np.linspace(0.0, 1.0, 200, endpoint=False)
        tri = lambda phase: 2.0 * np.abs(((t + phase) % 1.0) - 0.5) - 0.5
        i_a = amplitude_a * (1.0 + ripple * tri(0.0))
        i_b = amplitude_a * (1.0 + ripple * tri(0.5))
        self.waveform = np.stack((i_a, i_b, -(i_a + i_b)))
        self.calls = 0

    def excite(self, networks):
        self.calls += 1
        w = self.waveform
        return {
            name: PortExcitation(w @ w.T / w.shape[1], w.mean(axis=1), {"window": "synthetic"})
            for name in networks
        }


def _two_phase_scenario(circuit) -> CircuitCoupledScenario:
    active = np.zeros((2, ROWS, COLS), dtype=bool)
    active[0, 0:2, :COLS // 2 + 1] = True      # phase A feed, top
    active[0, 4:6, :COLS // 2 + 1] = True      # phase B feed, top
    active[0, :, COLS // 2:] = True            # shared neck and load, top
    mesh = LayeredPCBMesh(active, (35e-6, 35e-6), PITCH, PITCH)
    # An unrelated bottom-layer strip is heat-spreading copper only: it has
    # no port, so it stays out of the electrical mesh.
    thermal_active = active.copy()
    thermal_active[1, 2:4, :] = True
    ports = PortSet(
        pads=(((0, 0, 0), (0, 1, 0), (0, 2, 0)), ((0, 4, 0), (0, 5, 0), (0, 6, 0)), tuple((0, r, COLS) for r in range(ROWS + 1))),
        names=("A", "B", "load"),
        reference=2,
    )
    return CircuitCoupledScenario(
        (CoupledConductor("vout", mesh, ports, (0, 2)),),
        _thermal_mesh(thermal_active),
        circuit,
        convection=_convection(),
        conductivity_reference_temperature_k=293.15,
    )


def test_ripple_second_moments_heat_the_shared_neck_more_than_the_mean_current() -> None:
    with_ripple = _TwoPhaseCircuit(3.0, 0.8)
    mean_only = _TwoPhaseCircuit(3.0, 0.0)
    hot = run_circuit_coupled(_two_phase_scenario(with_ripple))
    flat = run_circuit_coupled(_two_phase_scenario(mean_only))

    assert hot.converged and flat.converged
    assert with_ripple.calls == hot.iterations
    np.testing.assert_allclose(
        hot.conductors["vout"].excitation.mean_current_a, flat.conductors["vout"].excitation.mean_current_a, rtol=1e-12
    )
    assert hot.joule_loss_w > flat.joule_loss_w
    assert hot.thermal.max_temperature_k > flat.thermal.max_temperature_k
    assert hot.thermal.total_heat_input_w == pytest.approx(hot.joule_loss_w, rel=1e-12)
    # Mean currents agree, so the extra loss is the ripple; in the shared neck the
    # interleaved ripples cancel partly and the per-phase feeds carry the rest.
    state = hot.conductors["vout"]
    neck = state.element_loss_w[0, :, COLS // 2 + 1 :]
    feeds = state.element_loss_w[0, :, : COLS // 2]
    flat_state = flat.conductors["vout"]
    neck_gain = np.sum(neck) / np.sum(flat_state.element_loss_w[0, :, COLS // 2 + 1 :])
    feed_gain = np.sum(feeds) / np.sum(flat_state.element_loss_w[0, :, : COLS // 2])
    assert feed_gain > neck_gain > 1.0
    assert state.max_rms_current_density_a_per_m2 > flat_state.max_rms_current_density_a_per_m2


def test_run_scenario_dispatches_and_validation_rejects_bad_circuits() -> None:
    scenario = _coupled(LinearTheveninCircuit({"loop": (TheveninPort(1e-3), TheveninPort(0.0))}), temperature_coefficient_per_k=0.0)
    assert isinstance(run_scenario(scenario), CircuitCoupledResult)

    class Silent:
        def excite(self, networks):
            return {}

    with pytest.raises(ValueError, match="no excitation for conductors"):
        run_circuit_coupled(_coupled(Silent()))

    class WrongShape:
        def excite(self, networks):
            return {name: PortExcitation(np.zeros((3, 3))) for name in networks}

    with pytest.raises(ValueError, match="correlation for 2 ports"):
        run_circuit_coupled(_coupled(WrongShape()))

    with pytest.raises(ValueError, match="no Thevenin ports"):
        run_circuit_coupled(_coupled(LinearTheveninCircuit({})))
    with pytest.raises(ValueError, match="Thevenin ports for 2 copper ports"):
        run_circuit_coupled(_coupled(LinearTheveninCircuit({"loop": (TheveninPort(1.0),)})))
    with pytest.raises(ValueError, match="every pad is open"):
        run_circuit_coupled(
            _coupled(LinearTheveninCircuit({"loop": (TheveninPort(1.0, np.inf), TheveninPort(0.0, np.inf))}))
        )
    mesh, ports, vias = _loop_mesh()
    with pytest.raises(ValueError, match="names must be unique"):
        CircuitCoupledScenario(
            (CoupledConductor("a", mesh, ports, (0, 2)), CoupledConductor("a", mesh, ports, (0, 2))),
            _thermal_mesh(mesh.element_active),
            LinearTheveninCircuit({}),
            convection=_convection(),
        )
    with pytest.raises(ValueError, match="one thermal slab per electrical layer"):
        CircuitCoupledScenario(
            (CoupledConductor("a", mesh, ports, (0,)),), _thermal_mesh(mesh.element_active), LinearTheveninCircuit({}), convection=_convection()
        )
    with pytest.raises(ValueError, match="resistance_ohm must be non-negative"):
        TheveninPort(1.0, -1.0)
