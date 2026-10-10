"""Matrix-free Q1 FEM for DC conduction in layered PCB copper.

This is an electrical-conduction solver, not a full-wave Maxwell solver.  Each
copper layer is discretised by bilinear quadrilateral elements, and plated
vertical connections are conductance links between layer nodes.  Element
stiffness contributions are evaluated directly; no global sparse matrix is
assembled.

A problem is driven by current terminals (a fixed total current spread over a
pad's nodes), by voltage terminals (a pad's nodes held at a fixed potential),
or by both.  Current terminals alone need a reference node for the gauge and
must balance; once a voltage terminal fixes the potential, the currents
through the voltage terminals follow from the solve and are reported.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from .. import backend as _backend
from ..threads import thread_budget
from .grid import check_pitch_axis
from .native_dc import NATIVE_HIGH_KERNEL_NAME, NATIVE_KERNEL_NAME
from .runtime import LowPrecisionRuntime, RuntimeBackend, make_float32_runtime
from .solver import MPIRConfig, MPIRResult, solve_mpir
from .two_level import AggregationCoarseCorrection

Preconditioner = Literal["two-level", "jacobi"]


COPPER_CONDUCTIVITY_S_PER_M = 1.0 / 1.724e-8
Node = tuple[int, int, int]


@dataclass(frozen=True)
class ViaConnection:
    """One resistive vertical connection between two nodal layer locations."""

    lower: Node
    upper: Node
    resistance_ohm: float

    def __post_init__(self) -> None:
        lower = tuple(int(index) for index in self.lower)
        upper = tuple(int(index) for index in self.upper)
        if len(lower) != 3 or len(upper) != 3:
            raise ValueError("via endpoints must be (layer, row, column) nodes")
        if lower == upper:
            raise ValueError("a via must connect two different nodes")
        if lower[0] == upper[0]:
            raise ValueError("a via must connect different layers")
        if not np.isfinite(self.resistance_ohm) or self.resistance_ohm <= 0.0:
            raise ValueError("via resistance must be finite and positive")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def conductance_s(self) -> float:
        return 1.0 / float(self.resistance_ohm)


@dataclass(frozen=True)
class CurrentTerminal:
    """A total current distributed uniformly over a set of FEM nodes."""

    nodes: tuple[Node, ...]
    current_a: float
    name: str = "terminal"

    def __post_init__(self) -> None:
        nodes = tuple(tuple(int(index) for index in node) for node in self.nodes)
        if not nodes:
            raise ValueError("a current terminal needs at least one node")
        if len(set(nodes)) != len(nodes):
            raise ValueError("terminal nodes must be unique")
        if not np.isfinite(self.current_a):
            raise ValueError("terminal current must be finite")
        object.__setattr__(self, "nodes", nodes)


@dataclass(frozen=True)
class VoltageTerminal:
    """A set of FEM nodes held at one fixed potential (an ideal voltage source)."""

    nodes: tuple[Node, ...]
    voltage_v: float
    name: str = "terminal"

    def __post_init__(self) -> None:
        nodes = tuple(tuple(int(index) for index in node) for node in self.nodes)
        if not nodes:
            raise ValueError("a voltage terminal needs at least one node")
        if len(set(nodes)) != len(nodes):
            raise ValueError("terminal nodes must be unique")
        if not np.isfinite(self.voltage_v):
            raise ValueError("terminal voltage must be finite")
        object.__setattr__(self, "nodes", nodes)


@dataclass(frozen=True)
class LayeredPCBMesh:
    """Structured Q1 element mesh for one electrical conductor role.

    ``element_active`` has shape ``(layer, node_rows - 1, node_cols - 1)``.
    ``pitch_x_m`` is one width for every column or one value per column,
    ``pitch_y_m`` likewise per row (a graded tensor grid, see :mod:`.grid`).
    Conductivity can be a scalar, one value per layer, or one value per element.
    """

    element_active: np.ndarray
    layer_thickness_m: Sequence[float]
    pitch_x_m: float | Sequence[float] | np.ndarray
    pitch_y_m: float | Sequence[float] | np.ndarray
    conductivity_s_per_m: float | Sequence[float] | np.ndarray = (
        COPPER_CONDUCTIVITY_S_PER_M
    )

    def __post_init__(self) -> None:
        active = np.asarray(self.element_active, dtype=bool)
        if active.ndim != 3 or active.shape[1] < 1 or active.shape[2] < 1:
            raise ValueError(
                "element_active must have shape (layers, rows, cols) with positive axes"
            )
        thickness = np.asarray(self.layer_thickness_m, dtype=np.float64)
        if thickness.shape != (active.shape[0],):
            raise ValueError("layer_thickness_m must contain one value per layer")
        if not np.all(np.isfinite(thickness)) or np.any(thickness <= 0.0):
            raise ValueError("layer thicknesses must be finite and positive")
        object.__setattr__(self, "pitch_x_m", check_pitch_axis(self.pitch_x_m, active.shape[2], "pitch_x_m"))
        object.__setattr__(self, "pitch_y_m", check_pitch_axis(self.pitch_y_m, active.shape[1], "pitch_y_m"))

        conductivity = np.asarray(self.conductivity_s_per_m, dtype=np.float64)
        if conductivity.ndim == 0:
            conductivity = np.full(active.shape, conductivity, dtype=np.float64)
        elif conductivity.shape == (active.shape[0],):
            conductivity = np.broadcast_to(
                conductivity[:, None, None], active.shape
            ).copy()
        elif conductivity.shape != active.shape:
            raise ValueError(
                "conductivity must be scalar, per-layer, or match element_active"
            )
        if not np.all(np.isfinite(conductivity)) or np.any(conductivity <= 0.0):
            raise ValueError("conductivity must be finite and positive")

        object.__setattr__(self, "element_active", active.copy())
        object.__setattr__(self, "layer_thickness_m", tuple(thickness.tolist()))
        object.__setattr__(self, "conductivity_s_per_m", conductivity.copy())

    @property
    def node_shape(self) -> tuple[int, int, int]:
        layers, element_rows, element_cols = self.element_active.shape
        return layers, element_rows + 1, element_cols + 1

    @property
    def size(self) -> int:
        return int(np.prod(self.node_shape))

    @property
    def uniform_pitch(self) -> bool:
        return bool(np.all(self.pitch_x_m == self.pitch_x_m[0]) and np.all(self.pitch_y_m == self.pitch_y_m[0]))

    @property
    def cell_area_m2(self) -> np.ndarray:
        """In-plane area of every cell, ``(rows, cols)``."""

        return self.pitch_y_m[:, None] * self.pitch_x_m[None, :]

    def element_coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-element factors of the two unit sheet stiffness matrices.

        ``K_e = c_x U_x + c_y U_y`` with ``c_x = σ t h_y / h_x`` and
        ``c_y = σ t h_x / h_y``; zero on inactive elements.
        """

        conductivity = np.asarray(self.conductivity_s_per_m, dtype=np.float64)
        thickness = np.asarray(self.layer_thickness_m, dtype=np.float64)[:, None, None]
        sheet = conductivity * thickness * self.element_active.astype(np.float64)
        hx = self.pitch_x_m[None, None, :]
        hy = self.pitch_y_m[None, :, None]
        return np.ascontiguousarray(sheet * hy / hx), np.ascontiguousarray(sheet * hx / hy)


@dataclass(frozen=True)
class PCBConductionProblem:
    """One conductor role and its drive.

    Current-driven: ``terminals`` only, balanced, with ``reference_node``
    fixing the gauge.  Voltage-driven: one or more ``voltage_terminals`` fix
    the potential, ``reference_node`` stays ``None``, and any current
    terminals need not balance; the remainder flows through the voltage
    terminals.  A node belongs to at most one terminal of either kind.
    """

    mesh: LayeredPCBMesh
    terminals: tuple[CurrentTerminal, ...] = ()
    reference_node: Node | None = None
    vias: tuple[ViaConnection, ...] = ()
    voltage_terminals: tuple[VoltageTerminal, ...] = ()

    def __post_init__(self) -> None:
        terminals = tuple(self.terminals)
        voltage_terminals = tuple(self.voltage_terminals)
        if voltage_terminals:
            if self.reference_node is not None:
                raise ValueError(
                    "voltage terminals fix the potential; reference_node must be None"
                )
        else:
            if not terminals:
                raise ValueError(
                    "a PCB conduction problem needs current or voltage terminals"
                )
            if self.reference_node is None:
                raise ValueError("a current-driven problem needs a reference_node")
            total = float(sum(terminal.current_a for terminal in terminals))
            magnitude = float(sum(abs(terminal.current_a) for terminal in terminals))
            if abs(total) > 1.0e-12 * max(1.0, magnitude):
                raise ValueError(f"terminal currents must sum to zero, got {total:.6g} A")
        owner: dict[Node, str] = {}
        for terminal in (*terminals, *voltage_terminals):
            for node in terminal.nodes:
                if node in owner:
                    raise ValueError(
                        f"node {node!r} belongs to terminals {owner[node]!r} and {terminal.name!r}"
                    )
                owner[node] = terminal.name
        object.__setattr__(self, "terminals", terminals)
        object.__setattr__(self, "voltage_terminals", voltage_terminals)
        object.__setattr__(self, "vias", tuple(self.vias))
        if self.reference_node is not None:
            object.__setattr__(
                self, "reference_node", tuple(int(value) for value in self.reference_node)
            )

    @property
    def voltage_driven(self) -> bool:
        return bool(self.voltage_terminals)


_STIFFNESS_1D = np.array([[1.0, -1.0], [-1.0, 1.0]])
_MASS_1D = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0


def unit_sheet_matrices() -> tuple[np.ndarray, np.ndarray]:
    """The two ``(4, 4)`` unit Q1 sheet stiffness matrices ``U_x, U_y``.

    Local node ordering is ``2 dy + dx``.  A ``h_x × h_y`` element of sheet
    conductance ``σ t`` has stiffness ``σ t (h_y / h_x U_x + h_x / h_y U_y)``.
    """

    return np.kron(_MASS_1D, _STIFFNESS_1D), np.kron(_STIFFNESS_1D, _MASS_1D)


def _local_stiffness(pitch_x_m: float, pitch_y_m: float) -> np.ndarray:
    """Unit-sheet-conductance Q1 stiffness for one rectangular element (uniform grids)."""

    unit_x, unit_y = unit_sheet_matrices()
    return (pitch_y_m / pitch_x_m) * unit_x + (pitch_x_m / pitch_y_m) * unit_y


def _flat_index(node: Node, shape: tuple[int, int, int]) -> int:
    if len(node) != 3 or any(index < 0 for index in node):
        raise ValueError(f"invalid node {node!r}")
    try:
        return int(np.ravel_multi_index(node, shape))
    except ValueError as exc:
        raise ValueError(f"node {node!r} lies outside mesh shape {shape}") from exc


class MatrixFreePCBOperator:
    """Split FP64/FP32 element-by-element conductivity operator.

    ``preconditioner="two-level"`` (default) adds the patch-constant coarse
    correction of :mod:`.two_level` to Jacobi scaling; a wide copper sheet or
    a graded grid with elongated elements otherwise costs hundreds of inner
    PCG iterations per outer step.  ``"jacobi"`` keeps the plain scaling.

    ``native=True`` hands the whole prepared operator to the C++ system
    ``electrical._pcbcore.fem.LayeredDCSystem`` (CPU runtime only, an OpenMP
    team of :func:`electrical.threads.thread_budget` threads): coefficients,
    masks, diagonal, coarse space, both actions, the solve, the right-hand
    side and the post-processing.  ``None`` uses it when built, ``False``
    keeps NumPy.  The two agree within the solver tolerance.

    Fixed nodes are the inactive nodes, the ``reference_node`` and every
    ``dirichlet_nodes`` entry; at least one of the last two must be given.
    The operator is the identity on fixed rows and ignores fixed columns, so
    it stays symmetric positive definite; :meth:`build_rhs` lifts the fixed
    potentials into the free rows.
    """

    def __init__(
        self,
        mesh: LayeredPCBMesh,
        *,
        reference_node: Node | None = None,
        dirichlet_nodes: Sequence[Node] = (),
        vias: Sequence[ViaConnection] = (),
        runtime: LowPrecisionRuntime | None = None,
        backend: RuntimeBackend | None = None,
        device_id: int = 0,
        preconditioner: Preconditioner = "two-level",
        coarse_block_nodes: int | None = None,
        native: bool | None = None,
    ) -> None:
        if runtime is not None and backend is not None:
            raise ValueError("pass either runtime or backend, not both")
        if preconditioner not in ("two-level", "jacobi"):
            raise ValueError("preconditioner must be 'two-level' or 'jacobi'")
        self.high_dtype = np.float64
        self.inner_solver = "pcg"
        self.mesh = mesh
        self.size = mesh.size
        self.runtime = runtime or make_float32_runtime(
            backend or "cpu", device_id=device_id
        )
        self.vias = tuple(vias)
        self.preconditioner = preconditioner
        self.reference_node = None if reference_node is None else tuple(reference_node)
        if reference_node is None and not dirichlet_nodes:
            raise ValueError("pass a reference_node or dirichlet_nodes to fix the potential")

        is_cuda = bool(getattr(self.runtime, "is_cuda", False))
        if is_cuda and native:
            raise ValueError("native=True requires the CPU runtime")
        core = None if is_cuda or native is False else _backend.core()
        if native and core is None:
            raise ImportError(
                "the pcbcore extension is not built; run cmake -S . -B build/native && cmake --build build/native"
            )
        # The C++ system owns the whole prepared operator; the arrays below
        # are the NumPy (and CuPy) implementation of the same operator.
        self._system: Any = None
        if core is not None:
            self._init_system(core, mesh, reference_node, dirichlet_nodes, preconditioner, coarse_block_nodes)
            return

        # K_e = c_x U_x + c_y U_y; the coefficients carry sheet conductance and
        # the (graded) element dimensions.
        self._unit_high = np.stack(unit_sheet_matrices())  # (2, 4, 4)
        self._coefficients_high = np.stack(mesh.element_coefficients())  # (2, layers, rows, cols)

        active_nodes = np.zeros(mesh.node_shape, dtype=bool)
        active = mesh.element_active
        active_nodes[:, :-1, :-1] |= active
        active_nodes[:, :-1, 1:] |= active
        active_nodes[:, 1:, :-1] |= active
        active_nodes[:, 1:, 1:] |= active

        via_a: list[int] = []
        via_b: list[int] = []
        via_g: list[float] = []
        for via in self.vias:
            first = _flat_index(via.lower, mesh.node_shape)
            second = _flat_index(via.upper, mesh.node_shape)
            via_a.append(first)
            via_b.append(second)
            via_g.append(via.conductance_s)
            active_nodes.flat[first] = True
            active_nodes.flat[second] = True

        fixed = ~active_nodes
        if reference_node is not None:
            reference_index = _flat_index(reference_node, mesh.node_shape)
            if not active_nodes.flat[reference_index]:
                raise ValueError("reference_node must lie on active copper or a via endpoint")
            fixed.flat[reference_index] = True
        dirichlet_index = np.asarray(
            [_flat_index(node, mesh.node_shape) for node in dirichlet_nodes], dtype=np.int64
        )
        if dirichlet_index.size and not np.all(active_nodes.flat[dirichlet_index]):
            raise ValueError("dirichlet_nodes must lie on active copper or via endpoints")
        fixed.flat[dirichlet_index] = True
        self.active_nodes = active_nodes
        self.free_nodes = ~fixed
        self._via_a_high = np.asarray(via_a, dtype=np.int64)
        self._via_b_high = np.asarray(via_b, dtype=np.int64)
        self._via_g_high = np.asarray(via_g, dtype=np.float64)

        self._coefficients_low = self.runtime.from_host(self._coefficients_high)
        self._unit_low = self.runtime.from_host(self._unit_high)
        self._free_low = self.runtime.namespace.asarray(self.free_nodes, dtype=bool)
        self._via_a_low = self.runtime.namespace.asarray(via_a, dtype=np.int64)
        self._via_b_low = self.runtime.namespace.asarray(via_b, dtype=np.int64)
        self._via_g_low = self.runtime.namespace.asarray(via_g, dtype=self.runtime.dtype)

        self.high_operator_backend = "array-element-loops-fp64"
        self.low_operator_backend = "array-element-loops"

        diagonal = self._build_diagonal(
            np,
            self._coefficients_high,
            self._unit_high,
            self.free_nodes,
            self._via_a_high,
            self._via_b_high,
            self._via_g_high,
        )
        if np.any(diagonal[self.free_nodes.reshape(-1)] <= 0.0):
            raise ValueError("every free node must have positive conductivity coupling")
        self._diagonal_high = diagonal
        self._diagonal_low = self.runtime.from_host(diagonal)
        self.coarse_correction: AggregationCoarseCorrection | None = None
        if preconditioner == "two-level":
            self.coarse_correction = AggregationCoarseCorrection(
                node_shape=mesh.node_shape,
                free_nodes=self.free_nodes,
                diagonal_high=diagonal,
                apply_high=self.apply_high,
                runtime=self.runtime,
                block=coarse_block_nodes,
            )

    def _init_system(
        self,
        core: Any,
        mesh: LayeredPCBMesh,
        reference_node: Node | None,
        dirichlet_nodes: Sequence[Node],
        preconditioner: Preconditioner,
        coarse_block_nodes: int | None,
    ) -> None:
        shape = mesh.node_shape
        # OpenMP team of every call: the process budget when the operator is
        # built.  ``ports.dc_port_basis`` lowers it to its share of the budget
        # when it runs several solves on this operator at once.
        self._team = thread_budget()
        system = core.fem.LayeredDCSystem(
            np.ascontiguousarray(mesh.element_active, dtype=np.uint8),
            np.asarray(mesh.layer_thickness_m, dtype=np.float64),
            mesh.pitch_x_m,
            mesh.pitch_y_m,
            mesh.conductivity_s_per_m,
            np.asarray([_flat_index(via.lower, shape) for via in self.vias], dtype=np.int64),
            np.asarray([_flat_index(via.upper, shape) for via in self.vias], dtype=np.int64),
            np.asarray([via.conductance_s for via in self.vias], dtype=np.float64),
            -1 if reference_node is None else _flat_index(reference_node, shape),
            np.asarray([_flat_index(node, shape) for node in dirichlet_nodes], dtype=np.int64),
            preconditioner == "two-level",
            0 if coarse_block_nodes is None else int(coarse_block_nodes),
            self._team,
        )
        self._adopt_system(core, system, preconditioner)

    def _adopt_system(self, core: Any, system: Any, preconditioner: Preconditioner) -> None:
        """Answer every call with the prepared C++ ``system``."""

        shape = self.mesh.node_shape
        self._system = system
        self._core = core
        self.active_nodes = system.active_nodes
        self.free_nodes = system.free_nodes
        self._diagonal_high = system.diagonal
        self._diagonal_low = self.runtime.from_host(self._diagonal_high)
        self.high_operator_backend = NATIVE_HIGH_KERNEL_NAME
        self.low_operator_backend = NATIVE_KERNEL_NAME
        self.coarse_correction = None
        if preconditioner == "two-level":
            self.coarse_correction = AggregationCoarseCorrection(
                node_shape=shape,
                free_nodes=self.free_nodes,
                diagonal_high=self._diagonal_high,
                apply_high=self.apply_high,
                runtime=self.runtime,
                block=system.block,
                assembled=(system.coarse_matrix, system.coarse_inverse),
            )

    @classmethod
    def _from_system(
        cls,
        mesh: LayeredPCBMesh,
        vias: Sequence[ViaConnection],
        system: Any,
        core: Any,
        *,
        preconditioner: Preconditioner = "two-level",
        team: int | None = None,
    ) -> "MatrixFreePCBOperator":
        """The operator of a C++ system the core built (a coupling loop's last basis)."""

        operator = cls.__new__(cls)
        operator.high_dtype = np.float64
        operator.inner_solver = "pcg"
        operator.mesh = mesh
        operator.size = mesh.size
        operator.runtime = make_float32_runtime("cpu")
        operator.vias = tuple(vias)
        operator.preconditioner = preconditioner
        operator.reference_node = None
        operator._team = thread_budget() if team is None else max(1, int(team))
        operator._adopt_system(core, system, preconditioner)
        return operator

    def _set_native_team(self, threads: int) -> None:
        """OpenMP team of the C++ system when this operator shares the budget.

        The system starts with the whole process budget; :func:`.ports.dc_port_basis`
        lowers it when it runs several solves on this operator at once.
        """

        if self._system is not None:
            self._team = max(1, int(threads))

    def _flat(self, values: np.ndarray) -> np.ndarray:
        flat = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
        if flat.size != self.size:
            raise ValueError(f"vector has size {flat.size}, expected {self.size}")
        return flat

    def _node_groups(
        self, terminals: Sequence[CurrentTerminal] | Sequence[VoltageTerminal]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Flat node indices of each terminal, CSR form (offsets, nodes)."""

        offsets = np.zeros(len(terminals) + 1, dtype=np.int64)
        np.cumsum([len(terminal.nodes) for terminal in terminals], out=offsets[1:])
        shape = self.mesh.node_shape
        nodes = np.asarray(
            [_flat_index(node, shape) for terminal in terminals for node in terminal.nodes], dtype=np.int64
        )
        return offsets, nodes

    def _current_groups(self, terminals: Sequence[CurrentTerminal]) -> tuple[np.ndarray, np.ndarray]:
        offsets, nodes = self._node_groups(terminals)
        inactive = ~self.active_nodes.reshape(-1)[nodes]
        if inactive.any():
            at = int(np.argmax(inactive))
            terminal = terminals[int(np.searchsorted(offsets, at, side="right")) - 1]
            node = np.unravel_index(int(nodes[at]), self.mesh.node_shape)
            raise ValueError(
                f"terminal {terminal.name!r} contains inactive node {tuple(int(i) for i in node)!r}"
            )
        return offsets, nodes

    def _voltage_groups(self, voltage_terminals: Sequence[VoltageTerminal]) -> tuple[np.ndarray, np.ndarray]:
        offsets, nodes = self._node_groups(voltage_terminals)
        wrong = self.free_nodes.reshape(-1)[nodes] | ~self.active_nodes.reshape(-1)[nodes]
        if wrong.any():
            at = int(np.argmax(wrong))
            terminal = voltage_terminals[int(np.searchsorted(offsets, at, side="right")) - 1]
            node = np.unravel_index(int(nodes[at]), self.mesh.node_shape)
            raise ValueError(
                f"voltage terminal {terminal.name!r} node {tuple(int(i) for i in node)!r} is not a "
                "Dirichlet node of this operator"
            )
        return offsets, nodes

    @staticmethod
    def _element_views(grid: Any) -> tuple[Any, Any, Any, Any]:
        return (
            grid[:, :-1, :-1],
            grid[:, :-1, 1:],
            grid[:, 1:, :-1],
            grid[:, 1:, 1:],
        )

    def _apply_impl(
        self,
        vector: Any,
        xp: Any,
        coefficients: Any,
        unit: Any,
        free: Any,
        via_a: Any,
        via_b: Any,
        via_g: Any,
    ) -> Any:
        grid = vector.reshape(self.mesh.node_shape)
        working = xp.where(free, grid, xp.asarray(0.0, dtype=vector.dtype))
        values = self._element_views(working)
        output = xp.zeros_like(working)
        targets = self._element_views(output)
        for row in range(4):
            contribution = xp.zeros_like(coefficients[0])
            for column in range(4):
                weight = coefficients[0] * unit[0, row, column] + coefficients[1] * unit[1, row, column]
                contribution = contribution + weight * values[column]
            targets[row][...] += contribution

        flat_working = working.reshape(-1)
        flat_output = output.reshape(-1)
        if via_a.size:
            delta = via_g * (flat_working[via_a] - flat_working[via_b])
            xp.add.at(flat_output, via_a, delta)
            xp.add.at(flat_output, via_b, -delta)
        return xp.where(free.reshape(-1), flat_output, vector.reshape(-1))

    @staticmethod
    def _build_diagonal(
        xp: Any,
        coefficients: Any,
        unit: Any,
        free: Any,
        via_a: Any,
        via_b: Any,
        via_g: Any,
    ) -> Any:
        diagonal = xp.zeros(free.shape, dtype=coefficients.dtype)
        targets = MatrixFreePCBOperator._element_views(diagonal)
        for index in range(4):
            targets[index][...] += coefficients[0] * unit[0, index, index] + coefficients[1] * unit[1, index, index]
        flat = diagonal.reshape(-1)
        if via_a.size:
            xp.add.at(flat, via_a, via_g)
            xp.add.at(flat, via_b, via_g)
        return xp.where(free.reshape(-1), flat, xp.asarray(1.0, dtype=flat.dtype))

    def apply_high(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        if vector.size != self.size:
            raise ValueError(f"vector has size {vector.size}, expected {self.size}")
        if self._system is not None:
            return self._system.apply_high(vector, self._team)
        return self._apply_impl(
            vector,
            np,
            self._coefficients_high,
            self._unit_high,
            self.free_nodes,
            self._via_a_high,
            self._via_b_high,
            self._via_g_high,
        )

    def native_solve_mpir(
        self,
        rhs_high: np.ndarray,
        config: MPIRConfig,
        initial_guess: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool, int, int, float, int, int] | None:
        """End-to-end mixed-precision solve in C++; ``None`` when native is off."""

        if self._system is None:
            return None
        result = self._system.solve(
            np.ascontiguousarray(rhs_high, dtype=np.float64).reshape(-1),
            None if initial_guess is None else self._flat(initial_guess),
            float(config.relative_tolerance),
            float(config.absolute_tolerance),
            float(config.inner_relative_tolerance),
            int(config.max_outer_iterations),
            int(config.max_inner_iterations),
            self._team,
        )
        return (
            result["solution"],
            bool(result["converged"]),
            int(result["outer_iterations"]),
            int(result["inner_iterations"]),
            float(result["relative_residual"]),
            int(result["high_operator_applications"]),
            int(result["low_operator_applications"]),
            list(result["history"]),
        )

    def apply_low(self, vector: Any) -> Any:
        if self._system is not None:
            vector = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
            if vector.size != self.size:
                raise ValueError(f"vector has size {vector.size}, expected {self.size}")
            return self._system.apply_low(vector, self._team)
        return self._apply_impl(
            vector,
            self.runtime.namespace,
            self._coefficients_low,
            self._unit_low,
            self._free_low,
            self._via_a_low,
            self._via_b_low,
            self._via_g_low,
        )

    @property
    def low_precision_bytes(self) -> int:
        """Bytes of the arrays the low-precision inner solve reads."""

        if self._system is not None:
            return int(self._system.low_bytes)
        arrays = (
            self._coefficients_low,
            self._unit_low,
            self._free_low,
            self._via_a_low,
            self._via_b_low,
            self._via_g_low,
            self._diagonal_low,
        )
        coarse = () if self.coarse_correction is None else (self.coarse_correction._coarse_inverse_low,)
        return sum(int(array.nbytes) for array in (*arrays, *coarse))

    def diagonal_low(self) -> Any:
        return self._diagonal_low

    def precondition_low(self, vector: Any) -> Any:
        """Low-precision preconditioner action used by the inner PCG."""

        if self.coarse_correction is None:
            return self.runtime.divide(vector, self._diagonal_low)
        return self.coarse_correction(vector)

    def apply_full_high(self, vector: np.ndarray) -> np.ndarray:
        """FP64 action of the unconstrained conductance matrix, ``K v``.

        Entry ``n`` is the current that leaves node ``n`` into the copper and
        vias for potentials ``v``; fixed nodes are not masked out.
        """

        if self._system is not None:
            return self._system.apply_full(self._flat(vector), self._team)
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        return self._apply_impl(
            vector,
            np,
            self._coefficients_high,
            self._unit_high,
            np.ones(self.mesh.node_shape, dtype=bool),
            self._via_a_high,
            self._via_b_high,
            self._via_g_high,
        )

    def dirichlet_potential(
        self, voltage_terminals: Sequence[VoltageTerminal] = ()
    ) -> np.ndarray:
        """Potential vector holding each voltage terminal's value, zero elsewhere."""

        if self._system is not None:
            offsets, nodes = self._voltage_groups(voltage_terminals)
            voltages = np.asarray([terminal.voltage_v for terminal in voltage_terminals], dtype=np.float64)
            return self._system.dirichlet_potential(offsets, nodes, voltages)
        potential = np.zeros(self.size, dtype=np.float64)
        for terminal in voltage_terminals:
            for node in terminal.nodes:
                index = _flat_index(node, self.mesh.node_shape)
                if self.free_nodes.flat[index] or not self.active_nodes.flat[index]:
                    raise ValueError(
                        f"voltage terminal {terminal.name!r} node {node!r} is not a "
                        "Dirichlet node of this operator"
                    )
                potential[index] = float(terminal.voltage_v)
        return potential

    def terminal_currents(
        self, potential_v: np.ndarray, voltage_terminals: Sequence[VoltageTerminal]
    ) -> np.ndarray:
        """Current each voltage terminal drives into the copper, in A."""

        if self._system is not None:
            offsets, nodes = self._node_groups(voltage_terminals)
            return self._system.group_currents(self._flat(potential_v), offsets, nodes, self._team)
        injected = self.apply_full_high(np.nan_to_num(np.asarray(potential_v, dtype=np.float64), nan=0.0))
        return np.asarray(
            [
                float(
                    np.sum(
                        injected[[_flat_index(node, self.mesh.node_shape) for node in terminal.nodes]]
                    )
                )
                for terminal in voltage_terminals
            ],
            dtype=np.float64,
        )

    def build_rhs(
        self,
        terminals: Sequence[CurrentTerminal],
        voltage_terminals: Sequence[VoltageTerminal] = (),
    ) -> np.ndarray:
        if self._system is not None:
            current_offsets, current_nodes = self._current_groups(terminals)
            voltage_offsets, voltage_nodes = self._voltage_groups(voltage_terminals)
            return self._system.build_rhs(
                current_offsets,
                current_nodes,
                np.asarray([terminal.current_a for terminal in terminals], dtype=np.float64),
                voltage_offsets,
                voltage_nodes,
                np.asarray([terminal.voltage_v for terminal in voltage_terminals], dtype=np.float64),
                self._team,
            )
        rhs = np.zeros(self.mesh.node_shape, dtype=np.float64)
        for terminal in terminals:
            share = float(terminal.current_a) / len(terminal.nodes)
            for node in terminal.nodes:
                index = _flat_index(node, self.mesh.node_shape)
                if not self.active_nodes.flat[index]:
                    raise ValueError(
                        f"terminal {terminal.name!r} contains inactive node {node!r}"
                    )
                rhs.flat[index] += share
        # The reference equation is redundant in the balanced Neumann system.
        # Replacing it by the gauge equation V_ref = 0 preserves the solution.
        rhs[~self.free_nodes] = 0.0
        rhs = rhs.reshape(-1)
        if voltage_terminals:
            # Lifting: free rows see f - K g, fixed rows read V = g directly.
            fixed = self.dirichlet_potential(voltage_terminals)
            lifted = self.apply_full_high(fixed)
            free = self.free_nodes.reshape(-1)
            rhs = np.where(free, rhs - lifted, fixed)
        return rhs

    def element_electric_field(self, potential_v: np.ndarray) -> np.ndarray:
        """Return the electric field at each Q1 element centre, in V/m."""

        if self._system is not None:
            return self._system.element_electric_field(self._flat(potential_v), self._team)
        potential = np.asarray(potential_v, dtype=np.float64).reshape(
            self.mesh.node_shape
        )
        v00, v01, v10, v11 = self._element_views(potential)
        field_x = -((v01 + v11) - (v00 + v10)) / (2.0 * self.mesh.pitch_x_m[None, None, :])
        field_y = -((v10 + v11) - (v00 + v01)) / (2.0 * self.mesh.pitch_y_m[None, :, None])
        field = np.stack((field_x, field_y), axis=-1)
        return np.where(self.mesh.element_active[..., None], field, 0.0)

    def via_currents(self, potential_v: np.ndarray) -> np.ndarray:
        if self._system is not None:
            return self._system.via_current(self._flat(potential_v))
        flat = np.asarray(potential_v, dtype=np.float64).reshape(-1)
        return self._via_g_high * (
            flat[self._via_a_high] - flat[self._via_b_high]
        )

    def element_joule_loss(self, potential_v: np.ndarray) -> np.ndarray:
        """Return the Joule loss of every Q1 element, in W.

        The loss is the exact element quadratic form ``sigma t v^T K_e v``,
        which sums to the operator's total loss.  It is the heat a thermal
        solve receives from this electrical solve, element by element.
        """

        if self._system is not None:
            return self._system.element_joule_loss(self._flat(potential_v), self._team)
        potential = np.asarray(potential_v, dtype=np.float64).reshape(
            self.mesh.node_shape
        )
        element_values = np.stack(self._element_views(potential), axis=-1)
        quadratic = lambda unit: np.einsum("...i,ij,...j->...", element_values, unit, element_values, optimize=True)
        return self._coefficients_high[0] * quadratic(self._unit_high[0]) + self._coefficients_high[1] * quadratic(
            self._unit_high[1]
        )

    def via_joule_loss(self, potential_v: np.ndarray) -> np.ndarray:
        """Return the Joule loss dissipated in each via, in W."""

        if self._system is not None:
            return self._system.via_joule_loss(self._flat(potential_v))
        flat = np.asarray(potential_v, dtype=np.float64).reshape(-1)
        drop = flat[self._via_a_high] - flat[self._via_b_high]
        return self._via_g_high * drop * drop

    def joule_loss(self, potential_v: np.ndarray) -> float:
        if self._system is not None:
            return float(self._system.post_process(self._flat(potential_v), self._team)["joule_loss"])
        return float(
            np.sum(self.element_joule_loss(potential_v))
            + np.sum(self.via_joule_loss(potential_v))
        )


@dataclass(frozen=True)
class PCBConductionSolution:
    potential_v: np.ndarray
    current_density_a_per_m2: np.ndarray
    via_current_a: np.ndarray
    joule_loss_w: float
    element_joule_loss_w: np.ndarray
    via_joule_loss_w: np.ndarray
    max_current_density_a_per_m2: float
    solve: MPIRResult
    voltage_terminal_current_a: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    """Current each of ``problem.voltage_terminals`` drives into the copper."""


def solve_pcb_dc(
    problem: PCBConductionProblem,
    *,
    config: MPIRConfig | None = None,
    runtime: LowPrecisionRuntime | None = None,
    backend: RuntimeBackend | None = None,
    device_id: int = 0,
    initial_potential_v: np.ndarray | None = None,
    preconditioner: Preconditioner = "two-level",
    coarse_block_nodes: int | None = None,
    native: bool | None = None,
) -> PCBConductionSolution:
    """Solve a layered PCB's DC conduction problem with matrix-free MPIR.

    ``initial_potential_v`` warm-starts the outer refinement, for example with
    the potential of the previous iteration of a coupled analysis; NaN entries
    (inactive nodes of a reported solution) are treated as zero.
    ``preconditioner`` selects the two-level (default) or Jacobi inner
    preconditioner.  ``native`` selects the C++ system of
    :class:`MatrixFreePCBOperator`, whose threads come from the process-wide
    budget (:func:`electrical.threads.set_thread_budget`).
    """

    operator = MatrixFreePCBOperator(
        problem.mesh,
        reference_node=problem.reference_node,
        dirichlet_nodes=tuple(
            node for terminal in problem.voltage_terminals for node in terminal.nodes
        ),
        vias=problem.vias,
        runtime=runtime,
        backend=backend,
        device_id=device_id,
        preconditioner=preconditioner,
        coarse_block_nodes=coarse_block_nodes,
        native=native,
    )
    rhs = operator.build_rhs(problem.terminals, problem.voltage_terminals)
    fixed = operator.dirichlet_potential(problem.voltage_terminals)
    initial = None
    if initial_potential_v is not None:
        initial = np.nan_to_num(
            np.asarray(initial_potential_v, dtype=np.float64).reshape(-1), nan=0.0
        )
        if initial.size != operator.size:
            raise ValueError("initial_potential_v must hold one value per node")
        initial = np.where(operator.free_nodes.reshape(-1), initial, fixed)
    elif problem.voltage_terminals:
        initial = fixed
    result = solve_mpir(operator, rhs, config=config, initial_guess=initial)
    potential = result.solution.reshape(problem.mesh.node_shape)
    reported_potential = np.where(operator.active_nodes, potential, np.nan)
    terminal_current = operator.terminal_currents(potential, problem.voltage_terminals)
    if operator._system is not None:
        post = operator._system.post_process(result.solution, operator._team)
        return PCBConductionSolution(
            potential_v=reported_potential,
            current_density_a_per_m2=post["current_density"],
            via_current_a=post["via_current"],
            joule_loss_w=float(post["joule_loss"]),
            element_joule_loss_w=post["element_joule_loss"],
            via_joule_loss_w=post["via_joule_loss"],
            max_current_density_a_per_m2=float(post["max_current_density"]),
            solve=result,
            voltage_terminal_current_a=terminal_current,
        )
    field = operator.element_electric_field(potential)
    conductivity = np.asarray(problem.mesh.conductivity_s_per_m, dtype=np.float64)
    current_density = conductivity[..., None] * field
    magnitudes = np.linalg.norm(current_density, axis=-1)
    active_magnitudes = magnitudes[problem.mesh.element_active]
    maximum = float(np.max(active_magnitudes)) if active_magnitudes.size else 0.0
    return PCBConductionSolution(
        potential_v=reported_potential,
        current_density_a_per_m2=current_density,
        via_current_a=operator.via_currents(potential),
        joule_loss_w=operator.joule_loss(potential),
        element_joule_loss_w=operator.element_joule_loss(potential),
        via_joule_loss_w=operator.via_joule_loss(potential),
        max_current_density_a_per_m2=maximum,
        solve=result,
        voltage_terminal_current_a=terminal_current,
    )
