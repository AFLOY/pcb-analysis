"""Port reduction of a layered DC conductor: the copper as an N-port.

Seen from its pads, one conductor role is an N-port.  This module measures
that N-port with the Q1 conduction solver of :mod:`.pcb` and keeps the unit
solutions, so that any port excitation an external circuit returns can be
turned back into a potential field and a Joule-loss field without another
field solve.

Every port is a set of pad nodes held at one potential (an equipotential
pad, see :class:`.pcb.VoltageTerminal`).  With ``n`` ports, ``n - 1`` unit
voltage solves give the conductance matrix ``G`` (``n × n``, symmetric,
positive semi-definite, zero row sums) and the unit potential fields.  The
potential for port currents ``I`` (positive into the copper, summing to
zero) is linear in ``I``, so the time average of the loss over any current
waveform follows from the second moments ``C = <I Iᵀ>`` alone:

```text
<P_e> = Σ_kl C_kl  φ_kᵀ K_e φ_l
```

where ``φ_k`` is the potential for unit current at port ``k`` returned at
the reference port.  Averaging the current first and squaring afterwards
would drop the ripple and every cross term between ports that share copper.
The cross terms are what a per-port path resistance cannot see: two phases
feeding one neck lose ``R (I_1 + I_2)²``, not ``R I_1² + R I_2²``.

The reduction is evaluated by diagonalising ``C`` restricted to the
non-reference ports: ``C_rr = Σ_m λ_m u_m u_mᵀ`` gives ``<P_e> = Σ_m λ_m
P_e(ψ_m)`` with ``ψ_m = Σ_k u_mk φ_k``, so the loss of every element and via
comes from the exact element quadratic forms of :class:`.pcb.MatrixFreePCBOperator`.

The ``n - 1`` unit solves are independent and share one read-only operator,
so :func:`dc_port_basis` can run them on a thread pool whose threads each
drive an OpenMP team of the native kernels.  This is the one place where
pcb-analysis nests parallelism.  The caller sets only the process-wide total
(:func:`electrical.threads.set_thread_budget`); :func:`_split_budget` divides
it into pool width × team with a product inside the budget.  A CUDA runtime
solves serially, since its stream is shared.

With the C++ core built, the pool, the unit solves, the KCL closure, the
unit-current fields and the modal loss and RMS sums run in
``electrical._pcbcore.fem`` (one ``std::thread`` per pool slot, each driving
an OpenMP team); this module keeps the split rule, the input checks and the
dataclasses.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..threads import thread_budget
from .pcb import (
    LayeredPCBMesh,
    MatrixFreePCBOperator,
    Node,
    Preconditioner,
    ViaConnection,
    VoltageTerminal,
)
from .runtime import LowPrecisionRuntime, RuntimeBackend
from .solver import MPIRConfig, MPIRResult, MPIRStep, solve_mpir


def _split_budget(budget: int, tasks: int) -> tuple[int, int]:
    """Pool width and per-solve OpenMP team for ``tasks`` unit solves in ``budget`` threads.

    Returns ``(width, team)`` with ``1 <= width <= max(1, tasks)``,
    ``team >= 1`` and ``width * team <= budget``.

    Provisional rule: a serial pool and the whole budget to each solve's
    OpenMP team.  ``docs/PORT_BASIS_WORKERS_RESULTS.json`` (8-core Ryzen
    7 9700X, fused kernels, 5 and 9 ports, 100² to 300² elements) has, at
    four threads, a pool of 4 and a team of 4 within 5 % of each other, and at
    eight threads 2 × 4 and 4 × 2 fastest (up to 2.6× over one thread), but no
    team of 8 to set against them, so it does not yet say where a pool beats
    a wider team.  A forthcoming measurement with
    ``experiments/port_basis_workers_benchmark.py`` sweeps width × team at
    fixed totals per mesh size and port count; its crossover replaces this
    rule.  Only this function changes then: callers neither pass nor see the
    split, and the result does not depend on it.
    """

    budget = max(1, int(budget))
    tasks = max(1, int(tasks))
    return _fit_split(budget, _preferred_width(budget, tasks), tasks)


def _preferred_width(budget: int, tasks: int) -> int:
    """The pool width the measurements favour; provisionally a serial pool."""

    del budget, tasks  # the provisional rule ignores them; the measured one will not
    return 1


def _fit_split(budget: int, preferred: int, tasks: int) -> tuple[int, int]:
    """The width at most ``preferred`` that leaves the fewest budget threads idle.

    A width that does not divide the budget strands its remainder (five
    threads as a pool of two teams of two use four), so among widths up to
    the preferred one the split that uses the most threads wins, and of those
    the one nearest the preference: five threads asked for as two unit solves
    side by side run as one solve on five, six asked for as four run as three
    pools of two.
    """

    limit = max(1, min(int(preferred), tasks, budget))
    width = max(
        range(1, limit + 1),
        key=lambda candidate: (candidate * (budget // candidate), -(limit - candidate)),
    )
    return width, budget // width


@dataclass(frozen=True)
class PortSet:
    """The pads of one conductor role, one node set per port.

    ``reference`` indexes the port whose potential is the zero of every
    reported field; port currents are reported for every port, the
    reference's being minus the sum of the others.
    """

    pads: tuple[tuple[Node, ...], ...]
    names: tuple[str, ...] = ()
    reference: int = 0

    def __post_init__(self) -> None:
        pads = tuple(tuple(tuple(int(i) for i in node) for node in pad) for pad in self.pads)
        if len(pads) < 2:
            raise ValueError("a port set needs at least two ports")
        if any(not pad for pad in pads):
            raise ValueError("every port needs at least one node")
        seen: dict[Node, int] = {}
        for index, pad in enumerate(pads):
            for node in pad:
                if node in seen:
                    raise ValueError(f"node {node!r} belongs to ports {seen[node]} and {index}")
                seen[node] = index
        names = tuple(self.names) or tuple(f"port_{index}" for index in range(len(pads)))
        if len(names) != len(pads) or len(set(names)) != len(names):
            raise ValueError("port names must be unique and one per port")
        if not 0 <= int(self.reference) < len(pads):
            raise ValueError("reference must index a port")
        object.__setattr__(self, "pads", pads)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "reference", int(self.reference))

    @property
    def count(self) -> int:
        return len(self.pads)

    @property
    def driven(self) -> tuple[int, ...]:
        """Port indices other than the reference, in port order."""

        return tuple(index for index in range(self.count) if index != self.reference)

    def voltage_terminals(self, voltage_v: Sequence[float]) -> tuple[VoltageTerminal, ...]:
        values = np.asarray(voltage_v, dtype=np.float64)
        if values.shape != (self.count,):
            raise ValueError("one voltage per port is required")
        return tuple(
            VoltageTerminal(pad, float(value), name)
            for pad, value, name in zip(self.pads, values, self.names)
        )


@dataclass(frozen=True)
class DCPortBasis:
    """The N-port of one conductor and the unit fields that span its solutions.

    ``conductance_s[j, k]`` is the current into the copper at port ``j`` when
    port ``k`` is held at 1 V and every other port at 0 V.
    ``unit_current_potential_v[m]`` is the potential field (``mesh.node_shape``)
    for 1 A into port ``ports.driven[m]`` and out of the reference port.
    ``workers`` is the number of threads the unit solves actually ran on.
    """

    mesh: LayeredPCBMesh
    ports: PortSet
    operator: MatrixFreePCBOperator
    conductance_s: np.ndarray
    unit_voltage_potential_v: np.ndarray
    unit_current_potential_v: np.ndarray
    solves: tuple[MPIRResult, ...]
    workers: int = 1

    @property
    def converged(self) -> bool:
        return all(result.converged for result in self.solves)

    @property
    def inner_iterations(self) -> int:
        return int(sum(result.inner_iterations for result in self.solves))

    @property
    def reduced_conductance_s(self) -> np.ndarray:
        """``G`` restricted to the driven ports, ``(n - 1) × (n - 1)``, SPD."""

        driven = list(self.ports.driven)
        return self.conductance_s[np.ix_(driven, driven)]

    @property
    def reduced_resistance_ohm(self) -> np.ndarray:
        """Pad-to-pad resistance matrix against the reference port: ``G_rr⁻¹``."""

        return np.linalg.inv(self.reduced_conductance_s)

    def _reduce_currents(self, current_a: np.ndarray) -> np.ndarray:
        values = np.asarray(current_a, dtype=np.float64)
        if values.shape != (self.ports.count,):
            raise ValueError("one current per port is required")
        total = float(np.sum(values))
        if abs(total) > 1.0e-9 * max(1.0, float(np.sum(np.abs(values)))):
            raise ValueError(f"port currents must sum to zero, got {total:.6g} A")
        return values[list(self.ports.driven)]

    def potential_v(self, current_a: Sequence[float]) -> np.ndarray:
        """Potential field (``mesh.node_shape``) for the given port currents."""

        reduced = self._reduce_currents(np.asarray(current_a))
        return np.tensordot(reduced, self.unit_current_potential_v, axes=1)

    def port_voltage_v(self, current_a: Sequence[float]) -> np.ndarray:
        """Potential of every port for the given port currents, reference at 0 V."""

        reduced = self._reduce_currents(np.asarray(current_a))
        voltages = np.zeros(self.ports.count, dtype=np.float64)
        voltages[list(self.ports.driven)] = self.reduced_resistance_ohm @ reduced
        return voltages

    def _reduced_correlation(self, correlation_a2: np.ndarray) -> tuple[np.ndarray, float]:
        """``C_rr`` after the checks every reduction needs, and its scale."""

        matrix = np.asarray(correlation_a2, dtype=np.float64)
        n = self.ports.count
        if matrix.shape != (n, n):
            raise ValueError(f"correlation must be {n}×{n}, one row and column per port")
        if not np.allclose(matrix, matrix.T, rtol=1.0e-10, atol=0.0):
            raise ValueError("correlation must be symmetric")
        scale = max(1.0, float(np.max(np.abs(matrix))))
        if np.any(np.abs(matrix.sum(axis=1)) > 1.0e-9 * scale):
            raise ValueError("correlation rows must sum to zero (port currents satisfy KCL)")
        driven = list(self.ports.driven)
        return np.ascontiguousarray(matrix[np.ix_(driven, driven)]), scale

    def _modes(self, correlation_a2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Eigen-decompose ``C_rr`` into weights and potential modes ``ψ_m``."""

        reduced, scale = self._reduced_correlation(correlation_a2)
        weights, vectors = np.linalg.eigh(reduced)
        if float(np.min(weights)) < -1.0e-10 * scale:
            raise ValueError("correlation must be positive semi-definite")
        keep = weights > 1.0e-14 * scale
        modes = np.tensordot(vectors[:, keep].T, self.unit_current_potential_v, axes=1)
        return weights[keep], modes

    def mean_loss_w(self, correlation_a2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Time-averaged Joule loss of every element and every via, in W.

        ``correlation_a2[k, l] = <I_k I_l>`` over the averaging window, with
        currents positive into the copper.  For a constant current ``I`` it
        is ``I Iᵀ`` and the result is the loss of that current.
        """

        if self.operator._system is not None:
            reduced, scale = self._reduced_correlation(correlation_a2)
            return self.operator._core.fem.port_modal_loss(
                self.operator._system, reduced, scale, self.unit_current_potential_v, self.operator._team
            )
        weights, modes = self._modes(correlation_a2)
        element = np.zeros(self.mesh.element_active.shape, dtype=np.float64)
        via = np.zeros(len(self.operator.vias), dtype=np.float64)
        for weight, mode in zip(weights, modes):
            element += weight * self.operator.element_joule_loss(mode)
            via += weight * self.operator.via_joule_loss(mode)
        return element, via

    def rms_current_density_a_per_m2(self, correlation_a2: np.ndarray) -> np.ndarray:
        """Root-mean-square current density magnitude of every element."""

        if self.operator._system is not None:
            reduced, scale = self._reduced_correlation(correlation_a2)
            return self.operator._core.fem.port_rms_current_density(
                self.operator._system, reduced, scale, self.unit_current_potential_v, self.operator._team
            )
        weights, modes = self._modes(correlation_a2)
        conductivity = np.asarray(self.mesh.conductivity_s_per_m, dtype=np.float64)
        square = np.zeros(self.mesh.element_active.shape, dtype=np.float64)
        for weight, mode in zip(weights, modes):
            field = self.operator.element_electric_field(mode)
            square += weight * np.sum((conductivity[..., None] * field) ** 2, axis=-1)
        return np.sqrt(square)


def dc_port_basis(
    mesh: LayeredPCBMesh,
    ports: PortSet,
    *,
    vias: Sequence[ViaConnection] = (),
    config: MPIRConfig | None = None,
    runtime: LowPrecisionRuntime | None = None,
    backend: RuntimeBackend | None = None,
    device_id: int = 0,
    preconditioner: Preconditioner = "two-level",
    coarse_block_nodes: int | None = None,
    native: bool | None = None,
    initial: DCPortBasis | None = None,
) -> DCPortBasis:
    """Measure the N-port of ``mesh`` at ``ports`` with ``n - 1`` unit voltage solves.

    One operator with every port node fixed serves all solves.  ``initial``
    warm-starts each unit solve from the same port's field of an earlier
    basis (for example the previous iterate of a coupled analysis on the
    same mesh with other conductivities).  The unit solves may run
    concurrently: the process-wide thread budget
    (:func:`electrical.threads.set_thread_budget`) is split internally between
    a pool over the solves and the OpenMP team of each solve
    (``DCPortBasis.workers`` reports the pool width used).  The operator is
    read-only once built and every solve owns its vectors, so the result does
    not depend on the split.  A CUDA runtime solves serially.
    """

    operator = MatrixFreePCBOperator(
        mesh,
        dirichlet_nodes=tuple(node for pad in ports.pads for node in pad),
        vias=vias,
        runtime=runtime,
        backend=backend,
        device_id=device_id,
        preconditioner=preconditioner,
        coarse_block_nodes=coarse_block_nodes,
        native=native,
    )
    if initial is not None and (
        initial.ports != ports or initial.mesh.node_shape != mesh.node_shape
    ):
        raise ValueError("initial basis must share the ports and the mesh shape")
    n = ports.count
    conductance = np.zeros((n, n), dtype=np.float64)
    unit_voltage = np.zeros((n - 1,) + mesh.node_shape, dtype=np.float64)

    def unit_solve(column: int, port: int) -> tuple[np.ndarray, np.ndarray, MPIRResult]:
        excitation = np.zeros(n)
        excitation[port] = 1.0
        terminals = ports.voltage_terminals(excitation)
        rhs = operator.build_rhs((), terminals)
        guess = operator.dirichlet_potential(terminals)
        if initial is not None:
            previous = initial.unit_voltage_potential_v[column].reshape(-1)
            guess = np.where(operator.free_nodes.reshape(-1), previous, guess)
        result = solve_mpir(operator, rhs, config=config, initial_guess=guess)
        potential = result.solution.reshape(mesh.node_shape)
        return potential, operator.terminal_currents(potential, terminals), result

    budget = thread_budget()
    width, team = _split_budget(budget, n - 1)
    # Whatever the rule says: no idle pool threads, no oversubscription, and
    # one stream for CUDA (whose solves have no OpenMP team).
    width = max(1, min(int(width), n - 1, budget))
    if getattr(operator.runtime, "is_cuda", False):
        width = 1
    team = max(1, min(int(team), budget // width))
    operator._set_native_team(team)
    if operator._system is not None:
        return _native_port_basis(operator, mesh, ports, config or MPIRConfig(), initial, width, team)
    if width > 1:
        from threadpoolctl import threadpool_limits

        # The pool's own NumPy work (coarse correction, portable path) must
        # stay inside the team as well, or width × BLAS pool overruns the budget.
        with threadpool_limits(limits=team), ThreadPoolExecutor(
            max_workers=width, thread_name_prefix="port-basis"
        ) as pool:
            outcomes = list(pool.map(unit_solve, range(n - 1), ports.driven))
    else:
        outcomes = [unit_solve(column, port) for column, port in enumerate(ports.driven)]
    solves = []
    for column, port in enumerate(ports.driven):
        potential, currents, result = outcomes[column]
        unit_voltage[column] = potential
        conductance[:, port] = currents
        solves.append(result)
    # The driven block is what the solves measured; it is symmetrised against
    # solver round-off.  The reference row and column follow from KCL exactly
    # (every port at 1 V drives no current), so G has zero row and column sums
    # to machine precision instead of to the solver tolerance, and a constant
    # current I = G V satisfies KCL as tightly as the loss reconstruction asks.
    driven = list(ports.driven)
    block = conductance[np.ix_(driven, driven)]
    block = 0.5 * (block + block.T)
    conductance = np.zeros((n, n), dtype=np.float64)
    conductance[np.ix_(driven, driven)] = block
    conductance[driven, ports.reference] = -block.sum(axis=1)
    conductance[ports.reference, driven] = -block.sum(axis=0)
    conductance[ports.reference, ports.reference] = float(block.sum())
    resistance = np.linalg.inv(block)
    # φ_k (1 A into driven port k) = Σ_j R_jk · (unit voltage field of port j).
    unit_current = np.tensordot(resistance.T, unit_voltage, axes=1)
    return DCPortBasis(
        mesh=mesh,
        ports=ports,
        operator=operator,
        conductance_s=conductance,
        unit_voltage_potential_v=unit_voltage,
        unit_current_potential_v=unit_current,
        solves=tuple(solves),
        workers=width,
    )


def _native_port_basis(
    operator: MatrixFreePCBOperator,
    mesh: LayeredPCBMesh,
    ports: PortSet,
    config: MPIRConfig,
    initial: DCPortBasis | None,
    width: int,
    team: int,
) -> DCPortBasis:
    """The unit solves, the KCL closure and the unit-current fields in C++."""

    offsets, nodes = operator._node_groups(ports.voltage_terminals(np.zeros(ports.count)))
    basis = operator._core.fem.dc_port_basis(
        operator._system,
        offsets,
        nodes,
        ports.reference,
        None if initial is None else np.ascontiguousarray(initial.unit_voltage_potential_v, dtype=np.float64),
        float(config.relative_tolerance),
        float(config.absolute_tolerance),
        float(config.inner_relative_tolerance),
        int(config.max_outer_iterations),
        int(config.max_inner_iterations),
        width,
        team,
    )
    unit_voltage = basis["unit_voltage"]
    solves = tuple(
        MPIRResult(
            solution=unit_voltage[column].reshape(-1),
            converged=bool(solve["converged"]),
            outer_iterations=int(solve["outer_iterations"]),
            inner_iterations=int(solve["inner_iterations"]),
            relative_residual=float(solve["relative_residual"]),
            high_operator_applications=int(solve["high_operator_applications"]),
            low_operator_applications=int(solve["low_operator_applications"]),
            low_runtime=operator.runtime.name,
            history=tuple(MPIRStep(*step) for step in solve["history"]),
        )
        for column, solve in enumerate(basis["solves"])
    )
    return DCPortBasis(
        mesh=mesh,
        ports=ports,
        operator=operator,
        conductance_s=basis["conductance"],
        unit_voltage_potential_v=unit_voltage,
        unit_current_potential_v=basis["unit_current"],
        solves=solves,
        workers=width,
    )
