"""The coupling loops in the C++ core against the Python loops on the same solvers.

The Python loop with ``native=True`` calls the C++ solvers one at a time, so
the two loops solve identical problems; only the Aitken factor's dot
products are summed differently (BLAS there, lane sums here).  The first two
iterates therefore agree bit for bit, the later ones to the solver tolerance,
and both stop after the same number of iterations.
"""

from __future__ import annotations

import numpy as np
import pytest

from electrical import _backend
from electrical.threads import thread_budget_scope
from multiphysics.staggered_coupling import CouplingConfig, run_electro_thermal
from multiphysics.staggered_coupling.electro_thermal import _python_electro_thermal
from thermal.matrix_free_mpir_fem import ExposedFaceRadiation, HeatSource
from tests.test_staggered_coupling import AMBIENT, _scenario

pytestmark = pytest.mark.skipif(not _backend.native_available(), reason="pcbcore extension not built")


def _extras(scenario):
    import dataclasses

    heat = np.zeros(scenario.thermal_mesh.element_grid_shape)
    heat[1, 2, 3] = 0.05
    return dataclasses.replace(
        scenario,
        extra_element_heat_w=heat,
        extra_heat_sources=(HeatSource(((1, 1, 1), (1, 1, 2)), 0.02),),
    )


@pytest.mark.parametrize("variant", ["plain", "extras", "radiation"])
def test_the_core_loop_follows_the_python_loop(variant: str) -> None:
    scenario = _scenario(3.0)
    if variant == "extras":
        scenario = _extras(scenario)
    if variant == "radiation":
        scenario = _scenario(3.0, radiation=(ExposedFaceRadiation(0.6, AMBIENT, directions=("+z",)),))
    config = CouplingConfig()
    native = run_electro_thermal(scenario, config=config)
    python = _python_electro_thermal(scenario, config, backend=None, device_id=0, native=True)
    assert native.converged and python.converged
    assert native.iterations == python.iterations
    first, second = native.history[0], python.history[0]
    assert first.joule_loss_w == second.joule_loss_w
    assert first.max_temperature_k == second.max_temperature_k
    assert native.cold_joule_loss_w == python.cold_joule_loss_w
    rtol = 1e-6
    assert native.electrical.joule_loss_w == pytest.approx(python.electrical.joule_loss_w, rel=rtol)
    np.testing.assert_allclose(native.element_temperature_k, python.element_temperature_k, rtol=1e-9)
    np.testing.assert_allclose(native.conductivity_s_per_m, python.conductivity_s_per_m, rtol=rtol)
    np.testing.assert_allclose(native.via_resistance_ohm, python.via_resistance_ohm, rtol=rtol)
    active = ~np.isnan(python.thermal.temperature_k)
    np.testing.assert_allclose(native.thermal.temperature_k[active], python.thermal.temperature_k[active], rtol=1e-9)
    np.testing.assert_allclose(
        native.electrical.voltage_terminal_current_a, python.electrical.voltage_terminal_current_a, rtol=rtol
    )
    for a, b in zip(native.history, python.history):
        assert (a.iteration, a.electrical_inner_iterations > 0, a.thermal_inner_iterations > 0) == (
            b.iteration,
            b.electrical_inner_iterations > 0,
            b.thermal_inner_iterations > 0,
        )


def test_the_core_loop_has_the_same_bits_at_every_budget() -> None:
    scenario = _extras(_scenario(3.0))
    outputs = []
    for budget in (1, 2, 3, 8):
        with thread_budget_scope(budget):
            result = run_electro_thermal(scenario)
        outputs.append(
            (
                result.history,
                result.electrical.potential_v.tobytes(),
                result.thermal.temperature_k.tobytes(),
                result.conductivity_s_per_m.tobytes(),
            )
        )
    assert all(output == outputs[0] for output in outputs[1:])


def test_the_core_circuit_loop_follows_the_python_loop() -> None:
    from multiphysics.staggered_coupling import LinearTheveninCircuit, TheveninPort, run_circuit_coupled
    from multiphysics.staggered_coupling.circuit_coupled import _python_circuit_coupled
    from tests.test_circuit_coupled import _two_phase_scenario

    calls = {"count": 0}

    class Counting(LinearTheveninCircuit):
        def excite(self, networks):
            calls["count"] += 1
            return super().excite(networks)

    circuit = Counting({"vout": (TheveninPort(8.0e-3, 1e-3), TheveninPort(6.0e-3, 1e-3), TheveninPort(0.0))})
    scenario = _two_phase_scenario(circuit)
    config = CouplingConfig()
    native = run_circuit_coupled(scenario, config=config)
    native_calls = calls["count"]
    python = _python_circuit_coupled(scenario, config, backend=None, device_id=0, native=True)
    assert native.converged and python.converged
    assert native.iterations == python.iterations == native_calls
    assert native.history[0].joule_loss_w == pytest.approx(python.history[0].joule_loss_w, rel=1e-12)
    assert native.joule_loss_w == pytest.approx(python.joule_loss_w, rel=1e-6)
    for name in native.conductors:
        a, b = native.conductors[name], python.conductors[name]
        np.testing.assert_allclose(a.basis.conductance_s, b.basis.conductance_s, rtol=1e-6)
        np.testing.assert_allclose(a.element_loss_w, b.element_loss_w, rtol=0, atol=1e-6 * b.element_loss_w.max())
        np.testing.assert_allclose(
            a.rms_current_density_a_per_m2, b.rms_current_density_a_per_m2, rtol=0,
            atol=1e-6 * b.rms_current_density_a_per_m2.max(),
        )
        np.testing.assert_allclose(a.element_temperature_k, b.element_temperature_k, rtol=1e-9)
        np.testing.assert_allclose(a.excitation.correlation_a2, b.excitation.correlation_a2, rtol=1e-6)
        # The returned basis is a working one: its operator answers further reductions.
        np.testing.assert_allclose(
            a.basis.mean_loss_w(a.excitation.correlation_a2)[0], a.element_loss_w, rtol=0,
            atol=1e-9 * a.element_loss_w.max(),
        )


def test_the_core_interface_loop_follows_the_python_loop() -> None:
    from multiphysics.staggered_coupling.board_enclosure import (
        InterfaceCouplingConfig,
        _python_board_enclosure,
        run_board_enclosure_thermal,
    )
    from tests.test_board_enclosure_coupling import _partitioned

    scenario = _partitioned()
    config = InterfaceCouplingConfig()
    native = run_board_enclosure_thermal(scenario, config=config)
    python = _python_board_enclosure(scenario, config, backend=None, device_id=0, initial=None)
    assert native.converged and python.converged
    assert native.iterations == python.iterations
    assert native.history[0].interface_heat_w == pytest.approx(python.history[0].interface_heat_w, rel=1e-12)
    assert native.interface_heat_w == pytest.approx(python.interface_heat_w, rel=1e-6)
    np.testing.assert_allclose(native.contact_temperature_k[0], python.contact_temperature_k[0], rtol=1e-9)
    active = ~np.isnan(python.board.temperature_k)
    np.testing.assert_allclose(native.board.temperature_k[active], python.board.temperature_k[active], rtol=1e-9)
    # Warm-started from its own result the loop stops at once.
    again = run_board_enclosure_thermal(scenario, config=config, initial=native)
    assert again.converged and again.iterations <= 2


@pytest.mark.parametrize("radiating", [False, True])
def test_the_core_enclosure_loop_follows_the_python_loop(radiating: bool) -> None:
    from multiphysics.staggered_coupling.board_enclosure import InterfaceCouplingConfig
    from multiphysics.staggered_coupling.electro_thermal_enclosure import (
        _python_electro_thermal_enclosure,
        run_electro_thermal_enclosure,
    )
    from tests.test_electro_thermal_enclosure import _partitioned

    scenario = _partitioned(radiating=radiating)
    config, interface = CouplingConfig(), InterfaceCouplingConfig()
    native = run_electro_thermal_enclosure(scenario, config=config, interface=interface)
    python = _python_electro_thermal_enclosure(scenario, config, interface, backend=None, device_id=0)
    assert native.converged and python.converged
    assert native.iterations == python.iterations
    assert [s.interface_iterations for s in native.history] == [s.interface_iterations for s in python.history]
    assert native.cold_joule_loss_w == python.cold_joule_loss_w
    assert native.electrical.joule_loss_w == pytest.approx(python.electrical.joule_loss_w, rel=1e-6)
    np.testing.assert_allclose(native.element_temperature_k, python.element_temperature_k, rtol=1e-9)
    np.testing.assert_allclose(native.conductivity_s_per_m, python.conductivity_s_per_m, rtol=1e-6)
