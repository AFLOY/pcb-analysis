"""The C++ EMC post-processing against the NumPy reference."""

from __future__ import annotations

import numpy as np
import pytest

from electrical import _backend
from electrical.matrix_free_mpir_fem import PCBConductionProblem, VoltageTerminal, solve_pcb_dc
from emc.tiled_dipole_superposition import (
    CurrentDipoles,
    dipole_moments,
    dipoles_from_pcb_dc,
    far_field_pattern,
    terminal_closure_dipoles,
)
from tests.test_native_dc import _board
from tests.tolerance import DIRECT_RTOL

pytestmark = pytest.mark.skipif(not _backend.native_available(), reason="pcbcore extension not built")


def _close(a: np.ndarray, b: np.ndarray) -> None:
    np.testing.assert_allclose(a, b, rtol=0, atol=DIRECT_RTOL * max(np.abs(b).max(initial=0.0), 1e-300))


def _sources() -> CurrentDipoles:
    rng = np.random.default_rng(11)
    position = rng.uniform(-0.05, 0.05, (300, 3))
    moment = rng.standard_normal((300, 3)) + 1j * rng.standard_normal((300, 3))
    return CurrentDipoles(position, moment)


def test_far_field_and_moments_match_the_reference() -> None:
    sources = _sources()
    native = far_field_pattern(sources, 3.0e8, distance_m=3.0)
    with _backend.use_reference():
        reference = far_field_pattern(sources, 3.0e8, distance_m=3.0, native=False)
    for name in ("electric_v_per_m", "e_theta_v_per_m", "e_phi_v_per_m"):
        _close(getattr(native, name), getattr(reference, name))
    assert native.radiated_power_w == pytest.approx(reference.radiated_power_w, rel=DIRECT_RTOL)
    for origin in (None, np.array([0.01, -0.02, 0.003])):
        a = dipole_moments(sources, 3.0e8, origin_m=origin)
        with _backend.use_reference():
            b = dipole_moments(sources, 3.0e8, origin_m=origin)
        _close(a.electric_a_m, b.electric_a_m)
        _close(a.magnetic_a_m2, b.magnetic_a_m2)


@pytest.mark.parametrize("voltage", [False, True])
def test_dc_current_elements_match_the_reference(voltage: bool) -> None:
    problem = _board(8, 12, graded=True)
    if voltage:
        source, sink = problem.terminals
        problem = PCBConductionProblem(
            mesh=problem.mesh,
            voltage_terminals=(VoltageTerminal(source.nodes, 1.0e-3, "s"), VoltageTerminal(sink.nodes, 0.0, "k")),
            vias=problem.vias,
        )
    solution = solve_pcb_dc(problem)
    heights = (0.0, 1.6e-3)
    native = dipoles_from_pcb_dc(problem, solution, heights, close_terminals=True)
    with _backend.use_reference():
        reference = dipoles_from_pcb_dc(problem, solution, heights, close_terminals=True)
    assert native.count == reference.count
    _close(native.position_m, reference.position_m)
    _close(native.moment_a_m, reference.moment_a_m)
    closure = terminal_closure_dipoles(problem, heights, voltage_terminal_current_a=solution.voltage_terminal_current_a)
    assert closure.count > 0
