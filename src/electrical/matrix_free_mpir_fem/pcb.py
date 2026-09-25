"""Matrix-free Q1 FEM for DC conduction in layered PCB copper.

This is an electrical-conduction solver, not a full-wave Maxwell solver.  Each
copper layer is discretised by bilinear quadrilateral elements, and plated
vertical connections are conductance links between layer nodes.  Element
stiffness contributions are evaluated directly; no global sparse matrix is
assembled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from .grid import check_pitch_axis
from .native_dc import NativeLayeredDCQ1, NativeLayeredDCQ1High, native_requested
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
    mesh: LayeredPCBMesh
    terminals: tuple[CurrentTerminal, ...]
    reference_node: Node
    vias: tuple[ViaConnection, ...] = ()

    def __post_init__(self) -> None:
        terminals = tuple(self.terminals)
        if not terminals:
            raise ValueError("a PCB conduction problem needs current terminals")
        total = float(sum(terminal.current_a for terminal in terminals))
        magnitude = float(sum(abs(terminal.current_a) for terminal in terminals))
        if abs(total) > 1.0e-12 * max(1.0, magnitude):
            raise ValueError(f"terminal currents must sum to zero, got {total:.6g} A")
        object.__setattr__(self, "terminals", terminals)
        object.__setattr__(self, "vias", tuple(self.vias))
        object.__setattr__(
            self, "reference_node", tuple(int(value) for value in self.reference_node)
        )


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

    ``native=True`` runs the FP32 action, the whole inner PCG and the FP64
    action in the optional C++ extension (CPU runtime only, ``native_threads``
    OpenMP threads); ``None`` follows ``PCB_NATIVE_Q1``.  The results are the
    same either way; only the speed differs.
    """

    def __init__(
        self,
        mesh: LayeredPCBMesh,
        *,
        reference_node: Node,
        vias: Sequence[ViaConnection] = (),
        runtime: LowPrecisionRuntime | None = None,
        backend: RuntimeBackend | None = None,
        device_id: int = 0,
        preconditioner: Preconditioner = "two-level",
        coarse_block_nodes: int | None = None,
        native: bool | None = None,
        native_threads: int | None = None,
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

        reference_index = _flat_index(reference_node, mesh.node_shape)
        if not active_nodes.flat[reference_index]:
            raise ValueError("reference_node must lie on active copper or a via endpoint")
        fixed = ~active_nodes
        fixed.flat[reference_index] = True
        self.active_nodes = active_nodes
        self.free_nodes = ~fixed
        self.reference_node = tuple(reference_node)
        self._via_a_high = np.asarray(via_a, dtype=np.int64)
        self._via_b_high = np.asarray(via_b, dtype=np.int64)
        self._via_g_high = np.asarray(via_g, dtype=np.float64)

        self._coefficients_low = self.runtime.from_host(self._coefficients_high)
        self._unit_low = self.runtime.from_host(self._unit_high)
        self._free_low = self.runtime.namespace.asarray(self.free_nodes, dtype=bool)
        self._via_a_low = self.runtime.namespace.asarray(via_a, dtype=np.int64)
        self._via_b_low = self.runtime.namespace.asarray(via_b, dtype=np.int64)
        self._via_g_low = self.runtime.namespace.asarray(via_g, dtype=self.runtime.dtype)

        if getattr(self.runtime, "is_cuda", False) and native:
            raise ValueError("native=True requires the CPU runtime")
        use_native = not getattr(self.runtime, "is_cuda", False) and (
            native or (native is None and native_requested())
        )
        # The FP64 action (outer residual and coarse assembly) goes native with
        # the low path; it is built first because the coarse matrix below is
        # assembled from FP64 applications.
        self._native_high: NativeLayeredDCQ1High | None = None
        self.high_operator_backend = "array-element-loops-fp64"
        if use_native:
            self._native_high = NativeLayeredDCQ1High(
                mesh.node_shape,
                self._coefficients_high,
                self._unit_high,
                self.free_nodes,
                self._via_a_high,
                self._via_b_high,
                self._via_g_high,
                threads=native_threads,
            )
            self.high_operator_backend = self._native_high.kernel_name

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
        self.preconditioner = preconditioner
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
        # Opt-in fused C++ host path: operator plus the whole inner PCG with
        # the same preconditioner.
        self._native: NativeLayeredDCQ1 | None = None
        self.low_operator_backend = "array-element-loops"
        if use_native:
            coarse = self.coarse_correction
            self._native = NativeLayeredDCQ1(
                mesh.node_shape,
                self._coefficients_high,
                self._unit_high,
                self.free_nodes,
                self._via_a_high,
                self._via_b_high,
                self._via_g_high,
                diagonal,
                coarse_block=None if coarse is None else coarse.block,
                coarse_inverse=None if coarse is None else coarse._coarse_inverse_high,
                threads=native_threads,
            )
            self.low_operator_backend = self._native.kernel_name

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
        if self._native_high is not None:
            return self._native_high.apply(vector)
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

    def native_inner_pcg(
        self, rhs_high: np.ndarray, config: MPIRConfig
    ) -> tuple[np.ndarray, int, float, int] | None:
        """Whole inner PCG in C++; ``None`` when the native path is off."""

        if self._native is None:
            return None
        return self._native.inner_pcg(
            rhs_high,
            inner_relative_tolerance=config.inner_relative_tolerance,
            max_inner_iterations=config.max_inner_iterations,
        )

    def apply_low(self, vector: Any) -> Any:
        if self._native is not None:
            return self._native.apply(vector)
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

    def diagonal_low(self) -> Any:
        return self._diagonal_low

    def precondition_low(self, vector: Any) -> Any:
        """Low-precision preconditioner action used by the inner PCG."""

        if self.coarse_correction is None:
            return self.runtime.divide(vector, self._diagonal_low)
        return self.coarse_correction(vector)

    def build_rhs(self, terminals: Sequence[CurrentTerminal]) -> np.ndarray:
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
        return rhs.reshape(-1)

    def element_electric_field(self, potential_v: np.ndarray) -> np.ndarray:
        """Return the electric field at each Q1 element centre, in V/m."""

        potential = np.asarray(potential_v, dtype=np.float64).reshape(
            self.mesh.node_shape
        )
        v00, v01, v10, v11 = self._element_views(potential)
        field_x = -((v01 + v11) - (v00 + v10)) / (2.0 * self.mesh.pitch_x_m[None, None, :])
        field_y = -((v10 + v11) - (v00 + v01)) / (2.0 * self.mesh.pitch_y_m[None, :, None])
        field = np.stack((field_x, field_y), axis=-1)
        return np.where(self.mesh.element_active[..., None], field, 0.0)

    def via_currents(self, potential_v: np.ndarray) -> np.ndarray:
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

        flat = np.asarray(potential_v, dtype=np.float64).reshape(-1)
        drop = flat[self._via_a_high] - flat[self._via_b_high]
        return self._via_g_high * drop * drop

    def joule_loss(self, potential_v: np.ndarray) -> float:
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
    native_threads: int | None = None,
) -> PCBConductionSolution:
    """Solve a layered PCB's DC conduction problem with matrix-free MPIR.

    ``initial_potential_v`` warm-starts the outer refinement, for example with
    the potential of the previous iteration of a coupled analysis; NaN entries
    (inactive nodes of a reported solution) are treated as zero.
    ``preconditioner`` selects the two-level (default) or Jacobi inner
    preconditioner.  ``native`` and ``native_threads`` select the fused C++
    host path of :class:`MatrixFreePCBOperator`.
    """

    operator = MatrixFreePCBOperator(
        problem.mesh,
        reference_node=problem.reference_node,
        vias=problem.vias,
        runtime=runtime,
        backend=backend,
        device_id=device_id,
        preconditioner=preconditioner,
        coarse_block_nodes=coarse_block_nodes,
        native=native,
        native_threads=native_threads,
    )
    rhs = operator.build_rhs(problem.terminals)
    initial = None
    if initial_potential_v is not None:
        initial = np.nan_to_num(
            np.asarray(initial_potential_v, dtype=np.float64).reshape(-1), nan=0.0
        )
        if initial.size != operator.size:
            raise ValueError("initial_potential_v must hold one value per node")
        initial = np.where(operator.free_nodes.reshape(-1), initial, 0.0)
    result = solve_mpir(operator, rhs, config=config, initial_guess=initial)
    potential = result.solution.reshape(problem.mesh.node_shape)
    field = operator.element_electric_field(potential)
    conductivity = np.asarray(problem.mesh.conductivity_s_per_m, dtype=np.float64)
    current_density = conductivity[..., None] * field
    magnitudes = np.linalg.norm(current_density, axis=-1)
    active_magnitudes = magnitudes[problem.mesh.element_active]
    maximum = float(np.max(active_magnitudes)) if active_magnitudes.size else 0.0
    reported_potential = np.where(operator.active_nodes, potential, np.nan)
    return PCBConductionSolution(
        potential_v=reported_potential,
        current_density_a_per_m2=current_density,
        via_current_a=operator.via_currents(potential),
        joule_loss_w=operator.joule_loss(potential),
        element_joule_loss_w=operator.element_joule_loss(potential),
        via_joule_loss_w=operator.via_joule_loss(potential),
        max_current_density_a_per_m2=maximum,
        solve=result,
    )
