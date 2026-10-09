"""The DC conductance network: exact against a dense solve, and reproducible."""

from __future__ import annotations

import numpy as np
import pytest

from electrical.sheet_peec import (
    ConductanceNetwork,
    solve_conductance_network,
    split_branch_sensitivity,
)


def _grid_network(rows: int, cols: int, layers: int = 1, seed: int = 0):
    """A rows x cols grid per layer, with a few vertical branches between layers."""
    rng = np.random.default_rng(seed)
    count = rows * cols * layers

    def node(layer: int, r: int, c: int) -> int:
        return (layer * rows + r) * cols + c

    left, right, conductance, in_plane = [], [], [], []
    for layer in range(layers):
        for r in range(rows):
            for c in range(cols):
                for dr, dc in ((0, 1), (1, 0)):
                    if r + dr < rows and c + dc < cols:
                        left.append(node(layer, r, c))
                        right.append(node(layer, r + dr, c + dc))
                        conductance.append(rng.uniform(0.5, 2.0))
                        in_plane.append(True)
    for layer in range(layers - 1):
        for r, c in ((0, 0), (rows - 1, cols - 1), (rows // 2, cols // 2)):
            left.append(node(layer, r, c))
            right.append(node(layer + 1, r, c))
            conductance.append(rng.uniform(5.0, 10.0))
            in_plane.append(False)
    return (
        ConductanceNetwork(count, np.array(left), np.array(right), np.array(conductance)),
        np.array(in_plane),
    )


def _dense_reference(network: ConductanceNetwork, reference: int, injection: np.ndarray) -> np.ndarray:
    laplacian = np.zeros((network.node_count, network.node_count))
    for a, b, g in zip(network.left, network.right, network.conductance):
        laplacian[a, a] += g
        laplacian[b, b] += g
        laplacian[a, b] -= g
        laplacian[b, a] -= g
    keep = np.arange(network.node_count) != reference
    voltage = np.zeros(network.node_count)
    voltage[keep] = np.linalg.solve(laplacian[np.ix_(keep, keep)], injection[keep])
    return voltage


def test_the_direct_solve_matches_a_dense_one():
    network, _ = _grid_network(7, 9, layers=2)
    injection = np.zeros(network.node_count)
    injection[3] = 2.0
    injection[network.node_count - 5] = -2.0
    reference = network.node_count - 5
    solution = solve_conductance_network(network, reference, injection)

    assert not solution.factorized
    np.testing.assert_allclose(
        solution.node_voltage, _dense_reference(network, reference, injection), rtol=1e-12, atol=1e-12
    )
    assert solution.node_voltage[reference] == 0.0
    # Every non-reference node closes its own current.
    keep = np.arange(network.node_count) != reference
    np.testing.assert_allclose(solution.node_current[keep], injection[keep], atol=1e-10)
    assert solution.loss_w == pytest.approx(float(injection @ solution.node_voltage), rel=1e-12)
    assert solution.relative_residual < 1e-12


def test_the_same_network_gives_the_same_bits():
    network, _ = _grid_network(6, 6, layers=2, seed=3)
    injection = np.zeros(network.node_count)
    injection[0], injection[-1] = 1.0, -1.0
    weights = np.zeros((1, network.node_count))
    weights[0, 0], weights[0, -1] = 1.0, -1.0
    first = solve_conductance_network(network, network.node_count - 1, injection, weights)
    second = solve_conductance_network(network, network.node_count - 1, injection, weights)

    assert first.node_voltage.tobytes() == second.node_voltage.tobytes()
    assert first.adjoint_voltage[0].tobytes() == second.adjoint_voltage[0].tobytes()
    assert first.loss_w == second.loss_w


def test_the_factored_path_gives_the_forward_solution_and_the_adjoint():
    network, _ = _grid_network(8, 5, layers=1, seed=1)
    reference = network.node_count - 1
    injection = np.zeros(network.node_count)
    injection[0], injection[reference] = 1.5, -1.5
    weights = np.zeros((2, network.node_count))
    weights[0, 0] = 1.0
    weights[1, 7], weights[1, 12] = 0.5, 0.5
    factored = solve_conductance_network(network, reference, injection, weights)
    direct = solve_conductance_network(network, reference, injection)

    assert factored.factorized
    np.testing.assert_allclose(factored.node_voltage, direct.node_voltage, rtol=1e-12, atol=1e-14)
    for row, adjoint in zip(weights, factored.adjoint_voltage):
        np.testing.assert_allclose(adjoint, _dense_reference(network, reference, row), rtol=1e-11, atol=1e-13)


def test_the_branch_sensitivities_add_back_to_the_objective():
    """Scaling every conductance by s scales the potentials by 1/s.

    So d(objective)/d(log s) = -objective, and the per-branch terms G·ΔV·Δλ,
    summed, give the objective back (the sign is the caller's convention).
    """
    network, in_plane = _grid_network(6, 7, layers=2, seed=2)
    reference = network.node_count - 1
    injection = np.zeros(network.node_count)
    injection[0], injection[reference] = 1.0, -1.0
    weights = np.zeros((1, network.node_count))
    weights[0, 0] = 1.0
    solution = solve_conductance_network(network, reference, injection, weights)
    node_sensitivity, branch_sensitivity, vertical_total = split_branch_sensitivity(
        network, solution.adjoint_branch_product(network, 0), in_plane
    )
    objective = float(weights[0] @ solution.node_voltage)

    assert float(node_sensitivity.sum()) + vertical_total == pytest.approx(objective, rel=1e-10)
    assert float(branch_sensitivity.sum()) == pytest.approx(objective, rel=1e-10)
    assert vertical_total == pytest.approx(float(branch_sensitivity[~in_plane].sum()), rel=0, abs=0)


def test_a_bad_network_is_refused():
    with pytest.raises(ValueError):
        ConductanceNetwork(3, np.array([0, 1]), np.array([1, 3]), np.array([1.0, 1.0]))
    network, _ = _grid_network(2, 2)
    with pytest.raises(ValueError):
        solve_conductance_network(network, 0, np.zeros(network.node_count + 1))
