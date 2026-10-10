"""The C++ DC network solve against its NumPy reference.

Both solve the same reduced Laplacian with SuperLU (the core factors it once
for the forward and every adjoint right-hand side; the reference uses
``spsolve`` without objectives), so they agree to roundoff.  The core gives
the same bits on every call and at every thread budget, returns non-finite
potentials for a singular network instead of raising, and refuses bad input
with the errors the reference raised.
"""

from __future__ import annotations

import numpy as np
import pytest

from electrical import backend as _backend
from electrical.sheet_peec import ConductanceNetwork, solve_conductance_network, split_branch_sensitivity
from electrical.threads import thread_budget_scope
from tests.test_dc_network import _grid_network
from tests.tolerance import DIRECT_RTOL

pytestmark = pytest.mark.skipif(not _backend.native_available(), reason="electrical._pcbcore is not built")


def _case(seed: int, objectives: int):
    network, in_plane = _grid_network(9, 7, layers=3, seed=seed)
    rng = np.random.default_rng(seed)
    injection = np.zeros(network.node_count)
    injection[0], injection[network.node_count // 2] = 1.25, -1.25
    weights = rng.standard_normal((objectives, network.node_count)) if objectives else None
    return network, in_plane, network.node_count // 2, injection, weights


def _close(actual: np.ndarray, expected: np.ndarray) -> None:
    scale = float(np.max(np.abs(expected))) if expected.size else 1.0
    np.testing.assert_allclose(actual, expected, rtol=DIRECT_RTOL, atol=DIRECT_RTOL * scale)


@pytest.mark.parametrize("objectives", [0, 1, 3])
def test_the_core_matches_the_reference(objectives: int) -> None:
    network, in_plane, reference, injection, weights = _case(5, objectives)
    native = solve_conductance_network(network, reference, injection, weights)
    with _backend.use_reference():
        expected = solve_conductance_network(network, reference, injection, weights)

    assert native.factorized == expected.factorized
    for field in ("voltage_unknowns", "node_voltage", "edge_current", "node_current"):
        _close(getattr(native, field), getattr(expected, field))
    assert len(native.adjoint_voltage) == len(expected.adjoint_voltage) == objectives
    for actual, reference_adjoint in zip(native.adjoint_voltage, expected.adjoint_voltage):
        _close(actual, reference_adjoint)
    assert native.loss_w == pytest.approx(expected.loss_w, rel=DIRECT_RTOL)
    assert native.relative_residual < 1e-12

    if objectives:
        product = native.adjoint_branch_product(network, 0)
        node, branch, vertical = split_branch_sensitivity(network, product, in_plane)
        with _backend.use_reference():
            node_ref, branch_ref, vertical_ref = split_branch_sensitivity(network, product, in_plane)
        _close(node, node_ref)
        _close(branch, branch_ref)
        assert vertical == pytest.approx(vertical_ref, rel=DIRECT_RTOL, abs=DIRECT_RTOL * abs(vertical_ref))


def test_the_core_gives_the_same_bits_every_time_and_at_every_budget() -> None:
    network, _, reference, injection, weights = _case(7, 2)
    runs = []
    for threads in (1, 2, 3, 8, 1):
        with thread_budget_scope(threads):
            solution = solve_conductance_network(network, reference, injection, weights)
        runs.append(
            solution.node_voltage.tobytes()
            + b"".join(adjoint.tobytes() for adjoint in solution.adjoint_voltage)
            + np.float64(solution.loss_w).tobytes()
        )
    assert all(run == runs[0] for run in runs)


def test_a_singular_network_gives_non_finite_potentials() -> None:
    # Two islands: the second has no path to the reference.
    network = ConductanceNetwork(4, np.array([0, 2]), np.array([1, 3]), np.array([1.0, 1.0]))
    injection = np.array([1.0, -1.0, 1.0, -1.0])
    solution = solve_conductance_network(network, 1, injection, np.ones((1, 4)))
    assert not np.all(np.isfinite(solution.voltage_unknowns))


def test_bad_input_raises_the_reference_errors() -> None:
    network, _, _, injection, _ = _case(1, 0)
    with pytest.raises(ValueError):
        solve_conductance_network(network, network.node_count, injection)
    with pytest.raises(ValueError):
        solve_conductance_network(network, 0, injection[:-1])
    with pytest.raises(ValueError):
        solve_conductance_network(network, 0, injection, np.ones((2, network.node_count + 1)))


def test_a_network_of_one_node_has_nothing_to_solve() -> None:
    network = ConductanceNetwork(1, np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0))
    solution = solve_conductance_network(network, 0, np.zeros(1))
    assert solution.voltage_unknowns.size == 0
    assert solution.node_voltage.tolist() == [0.0]
    assert solution.loss_w == 0.0
