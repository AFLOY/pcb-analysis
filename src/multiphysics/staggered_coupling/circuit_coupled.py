"""Electro-thermal coupling with the currents decided by an external circuit.

The copper of every conductor role is reduced to an N-port at its pads
(:func:`electrical.matrix_free_mpir_fem.dc_port_basis`), the external
circuit is asked what it drives through those ports, and the time-averaged
loss follows from the second moments ``C = <I Iᵀ>`` of that answer.  The
loss heats the board, the copper conductivity follows the temperature, the
N-port changes, and the circuit is asked again.  The loop is the staggered
fixed point of :mod:`.electro_thermal` with the same warm starts and Aitken
relaxation; only the electrical half is different.

Nothing here knows what the circuit is.  :class:`PortCircuit` is the
protocol: given every conductor's conductance matrix at the current
temperature, return every conductor's excitation.  :class:`LinearTheveninCircuit`
is the closed-form implementation (each pad behind a resistance to a fixed
potential) that a board without a circuit model uses; a circuit simulator
adapter returns the second moments of its own waveforms over a switching
window instead, and nothing in this module changes.

The N-port keeps every pad of a conductor in one matrix, so the current
distribution in copper shared by several pads (two phases feeding one
neck) is solved, not assembled from per-pad path resistances.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from electrical.matrix_free_mpir_fem import (
    DCPortBasis,
    LayeredPCBMesh,
    PortSet,
    RuntimeBackend,
    ViaConnection,
    dc_port_basis,
)
from thermal.matrix_free_mpir_fem import (
    ConvectionBoundary,
    ExposedFaceRadiation,
    HeatSource,
    LayeredThermalMesh,
    RadiationBoundary,
    ThermalConductionProblem,
    ThermalConductionSolution,
    element_heat_from_losses,
    solve_thermal_conduction,
    via_heat_sources,
)

from .electro_thermal import (
    COPPER_TEMPERATURE_COEFFICIENT_PER_K,
    CouplingConfig,
    CouplingStep,
    TemperatureFixedPoint,
    conductivity_at_temperature,
    electrical_layer_temperature_k,
    via_temperature_k,
)


@dataclass(frozen=True)
class CoupledConductor:
    """One conductor role: its copper, its pads as ports, and where it sits in the thermal stack."""

    name: str
    mesh: LayeredPCBMesh
    ports: PortSet
    layer_slabs: tuple[int, ...]
    vias: tuple[ViaConnection, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a conductor needs a name")
        object.__setattr__(self, "layer_slabs", tuple(int(index) for index in self.layer_slabs))
        object.__setattr__(self, "vias", tuple(self.vias))


@dataclass(frozen=True)
class PortNetwork:
    """What the circuit receives for one conductor: its N-port at the current temperature.

    ``conductance_s[j, k]`` is the current into the copper at port ``j`` per
    volt at port ``k`` with every other port at 0 V.  Currents are positive
    into the copper, the convention the excitation must follow.
    """

    conductor: str
    names: tuple[str, ...]
    reference: int
    conductance_s: np.ndarray

    @property
    def count(self) -> int:
        return len(self.names)

    @property
    def driven(self) -> tuple[int, ...]:
        return tuple(index for index in range(self.count) if index != self.reference)

    @property
    def reduced_resistance_ohm(self) -> np.ndarray:
        """Pad-to-pad resistance against the reference port, ``(n − 1) × (n − 1)``."""

        driven = list(self.driven)
        return np.linalg.inv(self.conductance_s[np.ix_(driven, driven)])


@dataclass(frozen=True)
class PortExcitation:
    """What the circuit returns for one conductor over its averaging window.

    ``correlation_a2[k, l] = <I_k I_l>`` with currents positive into the
    copper; for a constant current it is ``I Iᵀ``.  ``mean_current_a`` is
    reported, not used by the loss.  ``metadata`` is the circuit's own record
    (port voltages, window, simulator) and is carried into the result.
    """

    correlation_a2: np.ndarray
    mean_current_a: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrix = np.asarray(self.correlation_a2, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("correlation_a2 must be a square matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("correlation_a2 must be finite")
        object.__setattr__(self, "correlation_a2", matrix)
        if self.mean_current_a is not None:
            mean = np.asarray(self.mean_current_a, dtype=np.float64)
            if mean.shape != (matrix.shape[0],):
                raise ValueError("mean_current_a must hold one value per port")
            object.__setattr__(self, "mean_current_a", mean)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def constant(cls, current_a: Sequence[float], **metadata: Any) -> "PortExcitation":
        current = np.asarray(current_a, dtype=np.float64)
        return cls(np.outer(current, current), current, metadata)


class PortCircuit(Protocol):
    """The external circuit: every conductor's N-port in, every conductor's excitation out."""

    def excite(self, networks: Mapping[str, PortNetwork]) -> Mapping[str, PortExcitation]: ...


@dataclass(frozen=True)
class TheveninPort:
    """A pad behind ``resistance_ohm`` to a node at ``voltage_v``.

    ``resistance_ohm = 0`` is an ideal voltage at the pad; ``inf`` leaves the
    pad open.  A load is a pad at 0 V behind the load resistance; a source
    with internal resistance is a pad at its open-circuit voltage behind it.
    """

    voltage_v: float
    resistance_ohm: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.voltage_v):
            raise ValueError("voltage_v must be finite")
        if np.isnan(self.resistance_ohm) or self.resistance_ohm < 0.0:
            raise ValueError("resistance_ohm must be non-negative")


@dataclass(frozen=True)
class LinearTheveninCircuit:
    """Every pad behind its own resistance to a fixed potential; closed form.

    Solves ``(G + diag(1/r)) V = v / r`` for the pad potentials, pads with
    ``r = 0`` held at ``v``, and returns the constant currents ``I = G V``.
    No ripple: the excitation's second moments are ``I Iᵀ``.
    """

    ports_by_conductor: Mapping[str, tuple[TheveninPort, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ports_by_conductor",
            {name: tuple(ports) for name, ports in self.ports_by_conductor.items()},
        )

    def excite(self, networks: Mapping[str, PortNetwork]) -> Mapping[str, PortExcitation]:
        missing = sorted(set(networks) - set(self.ports_by_conductor))
        if missing:
            raise ValueError(f"no Thevenin ports for conductors {missing}")
        result: dict[str, PortExcitation] = {}
        for name, network in networks.items():
            ports = self.ports_by_conductor[name]
            if len(ports) != network.count:
                raise ValueError(
                    f"{name}: {len(ports)} Thevenin ports for {network.count} copper ports"
                )
            g = network.conductance_s
            v = np.array([port.voltage_v for port in ports], dtype=np.float64)
            r = np.array([port.resistance_ohm for port in ports], dtype=np.float64)
            fixed = r == 0.0
            free = ~fixed
            voltage = v.copy()
            if np.any(free):
                with np.errstate(divide="ignore"):
                    admittance = np.where(np.isinf(r), 0.0, 1.0 / np.where(free, r, 1.0))
                system = g[np.ix_(free, free)] + np.diag(admittance[free])
                rhs = admittance[free] * v[free] - g[np.ix_(free, fixed)] @ v[fixed]
                if not np.any(fixed) and not np.any(admittance[free] > 0.0):
                    raise ValueError(f"{name}: every pad is open; nothing fixes the potential")
                voltage[free] = np.linalg.solve(system, rhs)
            current = g @ voltage
            result[name] = PortExcitation.constant(
                current, port_voltage_v=voltage.tolist(), circuit="linear-thevenin"
            )
        return result


@dataclass(frozen=True)
class CircuitCoupledScenario:
    """Several conductor roles on one board, their thermal environment, ρ(T), and the circuit."""

    conductors: tuple[CoupledConductor, ...]
    thermal_mesh: LayeredThermalMesh
    circuit: PortCircuit
    convection: tuple[ConvectionBoundary, ...] = ()
    fixed_temperature_mask: np.ndarray | None = None
    fixed_temperature_k: float | np.ndarray | None = None
    extra_heat_sources: tuple[HeatSource, ...] = ()
    extra_element_heat_w: np.ndarray | None = None
    conductivity_reference_temperature_k: float = 293.15
    temperature_coefficient_per_k: float = COPPER_TEMPERATURE_COEFFICIENT_PER_K
    radiation: tuple[RadiationBoundary | ExposedFaceRadiation, ...] = ()

    def __post_init__(self) -> None:
        conductors = tuple(self.conductors)
        if not conductors:
            raise ValueError("at least one conductor is required")
        names = [conductor.name for conductor in conductors]
        if len(set(names)) != len(names):
            raise ValueError("conductor names must be unique")
        slabs, rows, cols = self.thermal_mesh.element_grid_shape
        for conductor in conductors:
            layers, c_rows, c_cols = conductor.mesh.element_active.shape
            if (c_rows, c_cols) != (rows, cols):
                raise ValueError(f"{conductor.name}: electrical and thermal meshes must share (rows, cols)")
            if not (
                np.allclose(conductor.mesh.pitch_x_m, self.thermal_mesh.pitch_x_m)
                and np.allclose(conductor.mesh.pitch_y_m, self.thermal_mesh.pitch_y_m)
            ):
                raise ValueError(f"{conductor.name}: electrical and thermal meshes must share the same grid lines")
            if len(conductor.layer_slabs) != layers or any(
                not 0 <= index < slabs for index in conductor.layer_slabs
            ):
                raise ValueError(f"{conductor.name}: layer_slabs must name one thermal slab per electrical layer")
        if not np.isfinite(self.temperature_coefficient_per_k) or self.temperature_coefficient_per_k < 0.0:
            raise ValueError("temperature_coefficient_per_k must be finite and non-negative")
        object.__setattr__(self, "conductors", conductors)
        object.__setattr__(self, "convection", tuple(self.convection))
        object.__setattr__(self, "extra_heat_sources", tuple(self.extra_heat_sources))
        object.__setattr__(self, "radiation", tuple(self.radiation))
        # Validate the thermal boundary conditions once, without heat.
        self.thermal_problem(np.zeros((slabs, rows, cols)), ())

    def thermal_problem(
        self, element_heat_w: np.ndarray, heat_sources: Sequence[HeatSource]
    ) -> ThermalConductionProblem:
        heat = np.asarray(element_heat_w, dtype=np.float64)
        if self.extra_element_heat_w is not None:
            heat = heat + np.asarray(self.extra_element_heat_w, dtype=np.float64)
        return ThermalConductionProblem(
            self.thermal_mesh,
            convection=self.convection,
            fixed_temperature_mask=self.fixed_temperature_mask,
            fixed_temperature_k=self.fixed_temperature_k,
            heat_sources=tuple(heat_sources) + self.extra_heat_sources,
            element_heat_w=heat,
            radiation=self.radiation,
        )


@dataclass(frozen=True)
class ConductorState:
    """One conductor at the end of the coupled solve."""

    basis: DCPortBasis
    excitation: PortExcitation
    element_loss_w: np.ndarray
    via_loss_w: np.ndarray
    rms_current_density_a_per_m2: np.ndarray
    conductivity_s_per_m: np.ndarray
    via_resistance_ohm: np.ndarray
    element_temperature_k: np.ndarray

    @property
    def joule_loss_w(self) -> float:
        return float(np.sum(self.element_loss_w) + np.sum(self.via_loss_w))

    @property
    def max_rms_current_density_a_per_m2(self) -> float:
        active = self.rms_current_density_a_per_m2[self.basis.mesh.element_active]
        return float(np.max(active)) if active.size else 0.0


@dataclass(frozen=True)
class CircuitCoupledStep(CouplingStep):
    circuit_calls: int = 1


@dataclass(frozen=True)
class CircuitCoupledResult:
    conductors: Mapping[str, ConductorState]
    thermal: ThermalConductionSolution
    converged: bool
    iterations: int
    history: tuple[CircuitCoupledStep, ...]
    cold_joule_loss_w: float

    @property
    def joule_loss_w(self) -> float:
        return float(sum(state.joule_loss_w for state in self.conductors.values()))

    @property
    def loss_increase_ratio(self) -> float:
        return self.joule_loss_w / self.cold_joule_loss_w if self.cold_joule_loss_w else float("nan")


def heated_conductor(
    scenario: CircuitCoupledScenario, conductor: CoupledConductor, temperature_k: np.ndarray
) -> tuple[LayeredPCBMesh, tuple[ViaConnection, ...]]:
    """The conductor's copper and vias at the board temperature."""

    layer_temperature = electrical_layer_temperature_k(
        temperature_k, scenario.thermal_mesh, conductor.layer_slabs
    )
    conductivity = conductivity_at_temperature(
        np.asarray(conductor.mesh.conductivity_s_per_m, dtype=np.float64),
        layer_temperature,
        reference_temperature_k=scenario.conductivity_reference_temperature_k,
        coefficient_per_k=scenario.temperature_coefficient_per_k,
    )
    via_factor = 1.0 + scenario.temperature_coefficient_per_k * (
        via_temperature_k(temperature_k, scenario.thermal_mesh, conductor.layer_slabs, conductor.vias)
        - scenario.conductivity_reference_temperature_k
    )
    vias = tuple(
        dataclasses.replace(via, resistance_ohm=via.resistance_ohm * float(factor))
        for via, factor in zip(conductor.vias, via_factor)
    )
    return dataclasses.replace(conductor.mesh, conductivity_s_per_m=conductivity), vias


def run_circuit_coupled(
    scenario: CircuitCoupledScenario,
    *,
    config: CouplingConfig | None = None,
    backend: RuntimeBackend | None = None,
    device_id: int = 0,
    native: bool | None = None,
) -> CircuitCoupledResult:
    """Iterate N-port, circuit and thermal solves to a self-consistent ρ(T) state.

    Every solve draws on the process-wide thread budget
    (:func:`electrical.threads.set_thread_budget`); how an N-port basis splits
    it between concurrent unit solves and their OpenMP teams is decided inside
    :func:`electrical.matrix_free_mpir_fem.dc_port_basis`.
    """

    config = config or CouplingConfig()
    fixed_point = TemperatureFixedPoint(scenario, config)
    history: list[CircuitCoupledStep] = []
    bases: dict[str, DCPortBasis | None] = {c.name: None for c in scenario.conductors}
    states: dict[str, ConductorState] = {}
    thermal_solution: ThermalConductionSolution | None = None
    temperature: np.ndarray | None = None
    cold_loss = float("nan")
    previous_loss = float("nan")
    converged = False

    for iteration in range(1, config.max_iterations + 1):
        networks: dict[str, PortNetwork] = {}
        heated: dict[str, tuple[LayeredPCBMesh, tuple[ViaConnection, ...]]] = {}
        electrical_inner = 0
        for conductor in scenario.conductors:
            if temperature is None:
                mesh, vias = conductor.mesh, conductor.vias
            else:
                mesh, vias = heated_conductor(scenario, conductor, temperature)
            basis = dc_port_basis(
                mesh,
                conductor.ports,
                vias=vias,
                config=config.electrical,
                backend=backend,
                device_id=device_id,
                native=native,
                initial=bases[conductor.name],
            )
            bases[conductor.name] = basis
            heated[conductor.name] = (mesh, vias)
            electrical_inner += basis.inner_iterations
            networks[conductor.name] = PortNetwork(
                conductor.name, conductor.ports.names, conductor.ports.reference, basis.conductance_s
            )

        excitations = scenario.circuit.excite(networks)
        missing = sorted(set(networks) - set(excitations))
        if missing:
            raise ValueError(f"the circuit returned no excitation for conductors {missing}")

        slabs, rows, cols = scenario.thermal_mesh.element_grid_shape
        heat = np.zeros((slabs, rows, cols), dtype=np.float64)
        sources: list[HeatSource] = []
        loss = 0.0
        for conductor in scenario.conductors:
            basis = bases[conductor.name]
            assert basis is not None
            excitation = excitations[conductor.name]
            if excitation.correlation_a2.shape != (conductor.ports.count,) * 2:
                raise ValueError(
                    f"{conductor.name}: the circuit returned a {excitation.correlation_a2.shape} "
                    f"correlation for {conductor.ports.count} ports"
                )
            element, via = basis.mean_loss_w(excitation.correlation_a2)
            heat += element_heat_from_losses(element, scenario.thermal_mesh, conductor.layer_slabs)
            mesh, vias = heated[conductor.name]
            sources.extend(
                via_heat_sources(vias, via, mesh.node_shape[0], scenario.thermal_mesh, conductor.layer_slabs)
            )
            loss += float(np.sum(element) + np.sum(via))
            states[conductor.name] = ConductorState(
                basis=basis,
                excitation=excitation,
                element_loss_w=element,
                via_loss_w=via,
                rms_current_density_a_per_m2=basis.rms_current_density_a_per_m2(excitation.correlation_a2),
                conductivity_s_per_m=np.asarray(mesh.conductivity_s_per_m, dtype=np.float64),
                via_resistance_ohm=np.asarray([v.resistance_ohm for v in vias], dtype=np.float64),
                element_temperature_k=np.zeros(conductor.mesh.element_active.shape),
            )
        if iteration == 1:
            cold_loss = loss

        thermal_solution = solve_thermal_conduction(
            scenario.thermal_problem(heat, sources),
            config=config.thermal,
            backend=backend,
            device_id=device_id,
            initial_temperature_k=fixed_point.temperature,
            native=native,
        )
        proposed = np.where(
            np.isfinite(thermal_solution.temperature_k),
            thermal_solution.temperature_k,
            thermal_solution.min_temperature_k,
        )
        change = fixed_point.update(proposed)
        temperature = fixed_point.temperature
        assert temperature is not None

        relative_loss_change = (
            abs(loss - previous_loss) / abs(loss) if np.isfinite(previous_loss) and loss else float("inf")
        )
        previous_loss = loss
        electrical_converged = all(b is not None and b.converged for b in bases.values())
        history.append(
            CircuitCoupledStep(
                iteration=iteration,
                joule_loss_w=loss,
                max_temperature_k=float(np.max(temperature)),
                temperature_change_k=change,
                relative_loss_change=relative_loss_change,
                relaxation=fixed_point.relaxation,
                electrical_inner_iterations=electrical_inner,
                thermal_inner_iterations=thermal_solution.solve.inner_iterations,
            )
        )
        if scenario.temperature_coefficient_per_k == 0.0:
            converged = electrical_converged and thermal_solution.solve.converged
            break
        if (
            change <= config.temperature_tolerance_k
            and relative_loss_change <= config.relative_loss_tolerance
            and electrical_converged
            and thermal_solution.solve.converged
        ):
            converged = True
            break

    assert thermal_solution is not None and temperature is not None
    final = {
        name: dataclasses.replace(
            state,
            element_temperature_k=electrical_layer_temperature_k(
                temperature, scenario.thermal_mesh, conductor.layer_slabs
            ),
        )
        for conductor in scenario.conductors
        for name, state in ((conductor.name, states[conductor.name]),)
    }
    return CircuitCoupledResult(
        conductors=final,
        thermal=thermal_solution,
        converged=converged,
        iterations=len(history),
        history=tuple(history),
        cold_joule_loss_w=cold_loss,
    )
