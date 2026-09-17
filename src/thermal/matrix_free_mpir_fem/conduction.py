"""Matrix-free trilinear Q1 FEM for steady heat conduction in a layered PCB.

The board is a stack of element slabs.  Every slab is one layer of hexahedral
Q1 elements whose in-plane footprint is the shared ``pitch_x_m`` by
``pitch_y_m`` grid and whose height is the slab thickness.  Copper, prepreg,
core, and plated vias are all just element conductivities, so a via is a
column of copper elements inside a dielectric slab.  Element stiffness
contributions are evaluated directly; no global sparse matrix is assembled.

Heat enters as per-element power (typically the Joule loss of an electrical
solve) or as power spread over nodes (a component).  It leaves through
convective faces on the top and bottom of the stack and through fixed-
temperature nodes.  The same split-precision contract as the electrical
front ends drives the MPIR solver, so the low-precision path runs on NumPy or
CuPy without changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from electrical.matrix_free_mpir_fem.runtime import (
    LowPrecisionRuntime,
    RuntimeBackend,
    make_float32_runtime,
)
from electrical.matrix_free_mpir_fem.solver import (
    MPIRConfig,
    MPIRResult,
    solve_mpir,
)

from .two_level import AggregationCoarseCorrection
from .native_hex import NativeThermalHexQ1, native_requested


COPPER_THERMAL_CONDUCTIVITY_W_PER_M_K = 385.0
FR4_IN_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K = 0.8
FR4_THROUGH_PLANE_THERMAL_CONDUCTIVITY_W_PER_M_K = 0.3

Node = tuple[int, int, int]
Side = Literal["top", "bottom"]
Preconditioner = Literal["two-level", "jacobi"]


def _broadcast_element_field(
    value: Any,
    shape: tuple[int, int, int],
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(shape, float(array), dtype=np.float64)
    if array.shape == (shape[0],):
        return np.broadcast_to(array[:, None, None], shape).copy()
    if array.shape == shape:
        return array.copy()
    raise ValueError(
        f"{name} must be a scalar, one value per slab, or match shape {shape}"
    )


@dataclass(frozen=True)
class LayeredThermalMesh:
    """Structured hexahedral Q1 mesh of a PCB stack.

    ``slab_thickness_m`` lists element slabs from the bottom of the board to
    the top.  Conductivity is a scalar, one value per slab, or one value per
    element with shape ``(slabs, rows, cols)``; ``element_shape`` gives the
    in-plane element count when the conductivity does not.  Through-plane
    conductivity defaults to the in-plane value; laminates are usually
    anisotropic, so both can be given.
    """

    slab_thickness_m: Sequence[float]
    pitch_x_m: float
    pitch_y_m: float
    conductivity_w_per_m_k: float | Sequence[float] | np.ndarray
    element_shape: tuple[int, int] | None = None
    through_plane_conductivity_w_per_m_k: (
        float | Sequence[float] | np.ndarray | None
    ) = None

    def __post_init__(self) -> None:
        thickness = np.asarray(self.slab_thickness_m, dtype=np.float64)
        if thickness.ndim != 1 or thickness.size < 1:
            raise ValueError("slab_thickness_m must list at least one slab")
        if not np.all(np.isfinite(thickness)) or np.any(thickness <= 0.0):
            raise ValueError("slab thicknesses must be finite and positive")
        for name in ("pitch_x_m", "pitch_y_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)

        in_plane = np.asarray(self.conductivity_w_per_m_k, dtype=np.float64)
        if self.element_shape is None:
            if in_plane.ndim != 3:
                raise ValueError(
                    "element_shape is required unless conductivity has shape "
                    "(slabs, rows, cols)"
                )
            rows, cols = int(in_plane.shape[1]), int(in_plane.shape[2])
        else:
            rows, cols = (int(value) for value in self.element_shape)
        if rows < 1 or cols < 1:
            raise ValueError("element_shape must have positive axes")
        shape = (int(thickness.size), rows, cols)

        in_plane = _broadcast_element_field(in_plane, shape, "conductivity_w_per_m_k")
        through = (
            in_plane.copy()
            if self.through_plane_conductivity_w_per_m_k is None
            else _broadcast_element_field(
                self.through_plane_conductivity_w_per_m_k,
                shape,
                "through_plane_conductivity_w_per_m_k",
            )
        )
        for name, array in (
            ("conductivity_w_per_m_k", in_plane),
            ("through_plane_conductivity_w_per_m_k", through),
        ):
            if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
                raise ValueError(f"{name} must be finite and positive")

        object.__setattr__(self, "slab_thickness_m", tuple(thickness.tolist()))
        object.__setattr__(self, "element_shape", (rows, cols))
        object.__setattr__(self, "conductivity_w_per_m_k", in_plane)
        object.__setattr__(self, "through_plane_conductivity_w_per_m_k", through)

    @property
    def element_grid_shape(self) -> tuple[int, int, int]:
        rows, cols = self.element_shape  # type: ignore[misc]
        return len(self.slab_thickness_m), rows, cols

    @property
    def node_shape(self) -> tuple[int, int, int]:
        slabs, rows, cols = self.element_grid_shape
        return slabs + 1, rows + 1, cols + 1

    @property
    def size(self) -> int:
        return int(np.prod(self.node_shape))

    @property
    def element_volume_m3(self) -> np.ndarray:
        thickness = np.asarray(self.slab_thickness_m, dtype=np.float64)
        return np.broadcast_to(
            (thickness * self.pitch_x_m * self.pitch_y_m)[:, None, None],
            self.element_grid_shape,
        ).copy()


@dataclass(frozen=True)
class ConvectionBoundary:
    """Newton cooling on the top or bottom face of the stack.

    The film coefficient is a scalar or one value per element face with shape
    ``(rows, cols)``.  Zero switches a face off, so a board clamped on one
    side and cooled on the other is expressed with one boundary.
    """

    side: Side
    coefficient_w_per_m2_k: float | np.ndarray
    ambient_temperature_k: float

    def __post_init__(self) -> None:
        if self.side not in ("top", "bottom"):
            raise ValueError("side must be 'top' or 'bottom'")
        coefficient = np.asarray(self.coefficient_w_per_m2_k, dtype=np.float64)
        if coefficient.ndim not in (0, 2):
            raise ValueError("coefficient must be a scalar or a (rows, cols) array")
        if not np.all(np.isfinite(coefficient)) or np.any(coefficient < 0.0):
            raise ValueError("film coefficients must be finite and non-negative")
        ambient = float(self.ambient_temperature_k)
        if not np.isfinite(ambient):
            raise ValueError("ambient temperature must be finite")
        object.__setattr__(self, "coefficient_w_per_m2_k", coefficient.copy())
        object.__setattr__(self, "ambient_temperature_k", ambient)


@dataclass(frozen=True)
class HeatSource:
    """A total power spread uniformly over a set of mesh nodes."""

    nodes: tuple[Node, ...]
    power_w: float
    name: str = "source"

    def __post_init__(self) -> None:
        nodes = tuple(tuple(int(index) for index in node) for node in self.nodes)
        if not nodes:
            raise ValueError("a heat source needs at least one node")
        if any(len(node) != 3 for node in nodes):
            raise ValueError("heat source nodes must be (slab_face, row, column)")
        if len(set(nodes)) != len(nodes):
            raise ValueError("heat source nodes must be unique")
        if not np.isfinite(self.power_w):
            raise ValueError("heat source power must be finite")
        object.__setattr__(self, "nodes", nodes)


@dataclass(frozen=True)
class ThermalConductionProblem:
    """Steady conduction with convective faces and fixed-temperature nodes.

    ``element_heat_w`` is the total power released in every element; it is
    lumped equally onto the element's eight corner nodes.  At least one
    positive film coefficient or one fixed node is required, otherwise the
    temperature level is undetermined.
    """

    mesh: LayeredThermalMesh
    convection: tuple[ConvectionBoundary, ...] = ()
    fixed_temperature_mask: np.ndarray | None = None
    fixed_temperature_k: float | np.ndarray | None = None
    heat_sources: tuple[HeatSource, ...] = ()
    element_heat_w: np.ndarray | None = None

    def __post_init__(self) -> None:
        node_shape = self.mesh.node_shape
        element_shape = self.mesh.element_grid_shape

        convection = tuple(self.convection)
        for boundary in convection:
            coefficient = boundary.coefficient_w_per_m2_k
            if coefficient.ndim == 2 and coefficient.shape != element_shape[1:]:
                raise ValueError(
                    "convection coefficient array must match (rows, cols) of the mesh"
                )
        has_convection = any(
            np.any(boundary.coefficient_w_per_m2_k > 0.0) for boundary in convection
        )

        if self.fixed_temperature_mask is None:
            mask = np.zeros(node_shape, dtype=bool)
            values = np.zeros(node_shape, dtype=np.float64)
            if self.fixed_temperature_k is not None:
                raise ValueError(
                    "fixed_temperature_k needs a fixed_temperature_mask"
                )
        else:
            mask = np.asarray(self.fixed_temperature_mask, dtype=bool)
            if mask.shape != node_shape:
                raise ValueError("fixed_temperature_mask must match mesh.node_shape")
            if self.fixed_temperature_k is None:
                raise ValueError("fixed_temperature_mask needs fixed_temperature_k")
            values = np.asarray(self.fixed_temperature_k, dtype=np.float64)
            if values.ndim == 0:
                values = np.full(node_shape, float(values), dtype=np.float64)
            elif values.shape != node_shape:
                raise ValueError(
                    "fixed_temperature_k must be a scalar or match mesh.node_shape"
                )
            if not np.all(np.isfinite(values[mask])):
                raise ValueError("fixed temperatures must be finite")
            values = np.where(mask, values, 0.0)
        if not has_convection and not np.any(mask):
            raise ValueError(
                "the problem needs a positive film coefficient or a fixed node"
            )

        if self.element_heat_w is None:
            element_heat = np.zeros(element_shape, dtype=np.float64)
        else:
            element_heat = np.asarray(self.element_heat_w, dtype=np.float64)
            if element_heat.shape != element_shape:
                raise ValueError("element_heat_w must match (slabs, rows, cols)")
            if not np.all(np.isfinite(element_heat)):
                raise ValueError("element heat must be finite")

        sources = tuple(self.heat_sources)
        for source in sources:
            for node in source.nodes:
                _flat_index(node, node_shape)

        object.__setattr__(self, "convection", convection)
        object.__setattr__(self, "fixed_temperature_mask", mask.copy())
        object.__setattr__(self, "fixed_temperature_k", values.copy())
        object.__setattr__(self, "heat_sources", sources)
        object.__setattr__(self, "element_heat_w", element_heat.copy())


def _flat_index(node: Node, shape: tuple[int, int, int]) -> int:
    if len(node) != 3 or any(index < 0 for index in node):
        raise ValueError(f"invalid node {node!r}")
    try:
        return int(np.ravel_multi_index(node, shape))
    except ValueError as exc:
        raise ValueError(f"node {node!r} lies outside mesh shape {shape}") from exc


_STIFFNESS_1D = np.array([[1.0, -1.0], [-1.0, 1.0]])
_MASS_1D = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0


def _local_hexahedron_matrices(
    pitch_x_m: float,
    pitch_y_m: float,
    slab_thickness_m: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Unit-conductivity trilinear stiffness split into in-plane and z parts.

    Local node ordering is ``4 * dz + 2 * dy + dx``, matching the corner
    views below.  Returns arrays of shape ``(slabs, 8, 8)``.
    """

    in_plane = []
    through = []
    for hz in slab_thickness_m:
        hx, hy = pitch_x_m, pitch_y_m
        x_part = (hy * hz / hx) * np.kron(
            _MASS_1D, np.kron(_MASS_1D, _STIFFNESS_1D)
        )
        y_part = (hx * hz / hy) * np.kron(
            _MASS_1D, np.kron(_STIFFNESS_1D, _MASS_1D)
        )
        z_part = (hx * hy / hz) * np.kron(
            _STIFFNESS_1D, np.kron(_MASS_1D, _MASS_1D)
        )
        in_plane.append(x_part + y_part)
        through.append(z_part)
    return np.asarray(in_plane), np.asarray(through)


class MatrixFreeThermalOperator:
    """Split FP64/FP32 element-by-element heat-conduction operator.

    Rows of fixed nodes are replaced by the identity.  The Robin (convection)
    term is a lumped nodal conductance added to the diagonal, so the operator
    stays symmetric positive definite and PCG applies unchanged.

    The default ``"two-level"`` preconditioner adds a patch-constant coarse
    correction to Jacobi scaling (see :mod:`.two_level`); a cooled copper plate
    otherwise costs hundreds of PCG iterations per decade.  ``"jacobi"`` keeps
    the plain diagonal scaling.  ``coarse_block_nodes`` sets the in-plane patch
    width; ``None`` picks the smallest width whose coarse space stays small.
    """

    def __init__(
        self,
        problem: ThermalConductionProblem,
        *,
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
        self.problem = problem
        self.mesh = problem.mesh
        self.size = self.mesh.size
        self.runtime = runtime or make_float32_runtime(
            backend or "cpu", device_id=device_id
        )

        mesh = self.mesh
        self._local_in_plane_high, self._local_through_high = (
            _local_hexahedron_matrices(
                mesh.pitch_x_m, mesh.pitch_y_m, mesh.slab_thickness_m
            )
        )
        self._in_plane_high = np.asarray(
            mesh.conductivity_w_per_m_k, dtype=np.float64
        )
        self._through_high = np.asarray(
            mesh.through_plane_conductivity_w_per_m_k, dtype=np.float64
        )

        self.fixed_nodes = problem.fixed_temperature_mask
        self.free_nodes = ~self.fixed_nodes

        # Lumped convective conductances, kept per boundary so the solution can
        # report how much heat each face removes.
        self._robin_weights_high: list[np.ndarray] = []
        self._robin_ambient_k: list[float] = []
        face_area = mesh.pitch_x_m * mesh.pitch_y_m
        for boundary in problem.convection:
            coefficient = boundary.coefficient_w_per_m2_k
            if coefficient.ndim == 0:
                coefficient = np.full(mesh.element_grid_shape[1:], float(coefficient))
            face_weight = coefficient * face_area / 4.0
            weights = np.zeros(mesh.node_shape, dtype=np.float64)
            face = weights[-1] if boundary.side == "top" else weights[0]
            face[:-1, :-1] += face_weight
            face[:-1, 1:] += face_weight
            face[1:, :-1] += face_weight
            face[1:, 1:] += face_weight
            self._robin_weights_high.append(weights.reshape(-1))
            self._robin_ambient_k.append(boundary.ambient_temperature_k)
        self._robin_total_high = (
            np.sum(self._robin_weights_high, axis=0)
            if self._robin_weights_high
            else np.zeros(self.size, dtype=np.float64)
        )
        self._robin_rhs_high = sum(
            (
                weights * ambient
                for weights, ambient in zip(
                    self._robin_weights_high, self._robin_ambient_k
                )
            ),
            np.zeros(self.size, dtype=np.float64),
        )

        diagonal = self._build_diagonal(
            np,
            self._in_plane_high,
            self._through_high,
            self._local_in_plane_high,
            self._local_through_high,
            self._robin_total_high,
            self.free_nodes,
        )
        if np.any(diagonal[self.free_nodes.reshape(-1)] <= 0.0):
            raise ValueError("every free node must have positive conductance")
        self._diagonal_high = diagonal

        runtime_ns = self.runtime.namespace
        self._in_plane_low = self.runtime.from_host(self._in_plane_high)
        self._through_low = self.runtime.from_host(self._through_high)
        self._local_in_plane_low = self.runtime.from_host(self._local_in_plane_high)
        self._local_through_low = self.runtime.from_host(self._local_through_high)
        self._robin_total_low = self.runtime.from_host(self._robin_total_high)
        self._free_low = runtime_ns.asarray(self.free_nodes, dtype=bool)
        self._diagonal_low = self.runtime.from_host(diagonal)

        self._cuda_apply = None
        self._native: NativeThermalHexQ1 | None = None
        self.low_operator_backend = "array-corner-products"
        if getattr(self.runtime, "is_cuda", False):
            if native:
                raise ValueError("native=True requires the CPU runtime")
            from .cuda import CudaThermalHexQ1Apply

            self._cuda_apply = CudaThermalHexQ1Apply(
                self.runtime, mesh.element_grid_shape
            )
            self.low_operator_backend = self._cuda_apply.kernel_name
            self._free_low_u8 = runtime_ns.ascontiguousarray(
                self._free_low.reshape(-1).astype(runtime_ns.uint8)
            )
            self._in_plane_low = runtime_ns.ascontiguousarray(self._in_plane_low)
            self._through_low = runtime_ns.ascontiguousarray(self._through_low)
            self._local_in_plane_low = runtime_ns.ascontiguousarray(
                self._local_in_plane_low
            )
            self._local_through_low = runtime_ns.ascontiguousarray(
                self._local_through_low
            )
            self._robin_total_low = runtime_ns.ascontiguousarray(self._robin_total_low)

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
        if self._cuda_apply is None and (native or (native is None and native_requested())):
            # Opt-in fused C++ host path: operator plus the whole inner PCG
            # with the same preconditioner.  ``native=True`` demands it and
            # ``PCB_NATIVE_THERMAL=1`` selects it for every CPU operator.
            coarse = self.coarse_correction
            self._native = NativeThermalHexQ1(
                mesh.element_grid_shape,
                self._in_plane_high,
                self._through_high,
                self._local_in_plane_high,
                self._local_through_high,
                self._robin_total_high,
                self.free_nodes,
                diagonal,
                coarse_block=None if coarse is None else coarse.block,
                coarse_inverse=None if coarse is None else coarse._coarse_inverse_high,
                threads=native_threads,
            )
            self.low_operator_backend = self._native.kernel_name

    # ------------------------------------------------------------------ views
    @staticmethod
    def _corner_views(grid: Any) -> tuple[Any, ...]:
        """Eight corner views of a node grid, ordered ``4 dz + 2 dy + dx``."""

        slabs, rows, cols = (axis - 1 for axis in grid.shape)
        return tuple(
            grid[dz : dz + slabs, dy : dy + rows, dx : dx + cols]
            for dz in (0, 1)
            for dy in (0, 1)
            for dx in (0, 1)
        )

    # ---------------------------------------------------------------- actions
    def _stiffness_action(
        self,
        grid: Any,
        xp: Any,
        in_plane: Any,
        through: Any,
        local_in_plane: Any,
        local_through: Any,
    ) -> Any:
        """Unconstrained conduction action ``K T`` on a node grid."""

        values = self._corner_views(grid)
        output = xp.zeros_like(grid)
        targets = self._corner_views(output)
        for row in range(8):
            contribution = xp.zeros_like(in_plane)
            for column in range(8):
                weight = (
                    in_plane * local_in_plane[:, row, column][:, None, None]
                    + through * local_through[:, row, column][:, None, None]
                )
                contribution = contribution + weight * values[column]
            targets[row][...] += contribution
        return output

    def _apply_impl(
        self,
        vector: Any,
        xp: Any,
        in_plane: Any,
        through: Any,
        local_in_plane: Any,
        local_through: Any,
        robin: Any,
        free: Any,
    ) -> Any:
        flat = vector.reshape(-1)
        free_flat = free.reshape(-1)
        working = xp.where(free_flat, flat, xp.asarray(0.0, dtype=vector.dtype))
        conduction = self._stiffness_action(
            working.reshape(self.mesh.node_shape),
            xp,
            in_plane,
            through,
            local_in_plane,
            local_through,
        ).reshape(-1)
        physical = conduction + robin * working
        return xp.where(free_flat, physical, flat)

    @staticmethod
    def _build_diagonal(
        xp: Any,
        in_plane: Any,
        through: Any,
        local_in_plane: Any,
        local_through: Any,
        robin: Any,
        free: Any,
    ) -> Any:
        diagonal = xp.zeros(free.shape, dtype=in_plane.dtype)
        targets = MatrixFreeThermalOperator._corner_views(diagonal)
        for index in range(8):
            targets[index][...] += (
                in_plane * local_in_plane[:, index, index][:, None, None]
                + through * local_through[:, index, index][:, None, None]
            )
        flat = diagonal.reshape(-1) + robin
        return xp.where(free.reshape(-1), flat, xp.asarray(1.0, dtype=flat.dtype))

    def apply_high(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        if vector.size != self.size:
            raise ValueError(f"vector has size {vector.size}, expected {self.size}")
        return self._apply_impl(
            vector,
            np,
            self._in_plane_high,
            self._through_high,
            self._local_in_plane_high,
            self._local_through_high,
            self._robin_total_high,
            self.free_nodes,
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
        if self._cuda_apply is not None:
            return self._cuda_apply(
                vector,
                self._in_plane_low,
                self._through_low,
                self._local_in_plane_low,
                self._local_through_low,
                self._robin_total_low,
                self._free_low_u8,
            )
        return self._apply_impl(
            vector,
            self.runtime.namespace,
            self._in_plane_low,
            self._through_low,
            self._local_in_plane_low,
            self._local_through_low,
            self._robin_total_low,
            self._free_low,
        )

    def diagonal_low(self) -> Any:
        return self._diagonal_low

    def precondition_low(self, vector: Any) -> Any:
        """Low-precision preconditioner action used by the inner PCG."""

        if self.coarse_correction is None:
            return self.runtime.divide(vector, self._diagonal_low)
        return self.coarse_correction(vector)

    # ------------------------------------------------------------------ loads
    def nodal_load(self) -> np.ndarray:
        """Heat input per node in W: lumped element heat plus nodal sources."""

        load = np.zeros(self.mesh.node_shape, dtype=np.float64)
        share = self.problem.element_heat_w / 8.0
        for target in self._corner_views(load):
            target[...] += share
        for source in self.problem.heat_sources:
            per_node = float(source.power_w) / len(source.nodes)
            for node in source.nodes:
                load.flat[_flat_index(node, self.mesh.node_shape)] += per_node
        return load.reshape(-1)

    def default_reference_temperature(self) -> float:
        """Ambient of the first convective face, else the mean fixed value."""

        if self._robin_ambient_k:
            return float(self._robin_ambient_k[0])
        mask = self.fixed_nodes
        return float(np.mean(self.problem.fixed_temperature_k[mask]))

    def build_rhs(self, reference_temperature_k: float = 0.0) -> np.ndarray:
        """Right-hand side for the temperature rise above a reference.

        The unknown is ``theta = T - reference``.  Solving for the rise instead
        of the absolute temperature keeps the FP64 residual free of the
        cancellation ``K (300 K) ~ 0`` that otherwise caps attainable accuracy
        at about ``eps * ||A|| * 300 K / ||q||``.  Fixed temperatures move onto
        the free rows through the unconstrained stiffness action, and their
        own rows carry the shifted value.
        """

        reference = float(reference_temperature_k)
        mask = self.fixed_nodes.reshape(-1)
        shifted_fixed = np.where(
            mask, self.problem.fixed_temperature_k.reshape(-1) - reference, 0.0
        )
        boundary_action = self._stiffness_action(
            shifted_fixed.reshape(self.mesh.node_shape),
            np,
            self._in_plane_high,
            self._through_high,
            self._local_in_plane_high,
            self._local_through_high,
        ).reshape(-1)
        robin_rhs = sum(
            (
                weights * (ambient - reference)
                for weights, ambient in zip(
                    self._robin_weights_high, self._robin_ambient_k
                )
            ),
            np.zeros(self.size, dtype=np.float64),
        )
        rhs = self.nodal_load() + robin_rhs - boundary_action
        rhs[mask] = shifted_fixed[mask]
        return rhs

    # ---------------------------------------------------------- post-process
    def unconstrained_residual(self, temperature_k: np.ndarray) -> np.ndarray:
        """``K T + R (T - T_amb) - q`` at every node, including fixed ones.

        At free nodes this is the solver's residual.  At fixed nodes its
        negative is the heat the fixed-temperature sink absorbs to hold that
        temperature, because the row balance reads ``q = K T + R (T - T_amb)
        + sink``.
        """

        flat = np.asarray(temperature_k, dtype=np.float64).reshape(-1)
        conduction = self._stiffness_action(
            flat.reshape(self.mesh.node_shape),
            np,
            self._in_plane_high,
            self._through_high,
            self._local_in_plane_high,
            self._local_through_high,
        ).reshape(-1)
        return (
            conduction
            + self._robin_total_high * flat
            - self._robin_rhs_high
            - self.nodal_load()
        )

    def convective_heat(self, temperature_k: np.ndarray) -> np.ndarray:
        """Heat removed by each convection boundary, in W, in problem order."""

        flat = np.asarray(temperature_k, dtype=np.float64).reshape(-1)
        return np.asarray(
            [
                float(np.sum(weights * (flat - ambient)))
                for weights, ambient in zip(
                    self._robin_weights_high, self._robin_ambient_k
                )
            ],
            dtype=np.float64,
        )

    def element_heat_flux(self, temperature_k: np.ndarray) -> np.ndarray:
        """Return ``-k grad T`` at each element centre, in W/m², as (x, y, z)."""

        grid = np.asarray(temperature_k, dtype=np.float64).reshape(
            self.mesh.node_shape
        )
        corners = self._corner_views(grid)
        # Corner index 4 dz + 2 dy + dx.
        x_plus = corners[1] + corners[3] + corners[5] + corners[7]
        x_minus = corners[0] + corners[2] + corners[4] + corners[6]
        y_plus = corners[2] + corners[3] + corners[6] + corners[7]
        y_minus = corners[0] + corners[1] + corners[4] + corners[5]
        z_plus = corners[4] + corners[5] + corners[6] + corners[7]
        z_minus = corners[0] + corners[1] + corners[2] + corners[3]
        thickness = np.asarray(self.mesh.slab_thickness_m)[:, None, None]
        gradient_x = (x_plus - x_minus) / (4.0 * self.mesh.pitch_x_m)
        gradient_y = (y_plus - y_minus) / (4.0 * self.mesh.pitch_y_m)
        gradient_z = (z_plus - z_minus) / (4.0 * thickness)
        return np.stack(
            (
                -self._in_plane_high * gradient_x,
                -self._in_plane_high * gradient_y,
                -self._through_high * gradient_z,
            ),
            axis=-1,
        )


@dataclass(frozen=True)
class ThermalConductionSolution:
    """Nodal temperatures and the heat budget of one steady solve."""

    temperature_k: np.ndarray
    heat_flux_w_per_m2: np.ndarray
    max_temperature_k: float
    min_temperature_k: float
    total_heat_input_w: float
    convective_heat_w: np.ndarray
    fixed_temperature_heat_w: float
    heat_balance_error_w: float
    solve: MPIRResult


def solve_thermal_conduction(
    problem: ThermalConductionProblem,
    *,
    config: MPIRConfig | None = None,
    runtime: LowPrecisionRuntime | None = None,
    backend: RuntimeBackend | None = None,
    device_id: int = 0,
    initial_temperature_k: np.ndarray | float | None = None,
    preconditioner: Preconditioner = "two-level",
    coarse_block_nodes: int | None = None,
    reference_temperature_k: float | None = None,
    native: bool | None = None,
    native_threads: int | None = None,
) -> ThermalConductionSolution:
    """Solve steady heat conduction in a layered PCB with matrix-free MPIR.

    The solver works on the rise above ``reference_temperature_k`` (default:
    the first convective ambient, else the mean fixed temperature) so the
    outer FP64 residual is measured against the heat load rather than against
    ``300 K`` of absolute temperature.  ``initial_temperature_k`` warm-starts
    the outer refinement; a previous solution shortens the solve.  Without it
    the solve starts from the reference temperature.
    """

    operator = MatrixFreeThermalOperator(
        problem,
        runtime=runtime,
        backend=backend,
        device_id=device_id,
        preconditioner=preconditioner,
        coarse_block_nodes=coarse_block_nodes,
        native=native,
        native_threads=native_threads,
    )
    reference = (
        operator.default_reference_temperature()
        if reference_temperature_k is None
        else float(reference_temperature_k)
    )
    if not np.isfinite(reference):
        raise ValueError("reference_temperature_k must be finite")
    rhs = operator.build_rhs(reference)
    if initial_temperature_k is None:
        initial = None
    else:
        guess = np.asarray(initial_temperature_k, dtype=np.float64)
        if guess.ndim == 0:
            initial = np.full(operator.size, float(guess))
        else:
            initial = guess.reshape(-1).copy()
        if initial.size != operator.size:
            raise ValueError("initial_temperature_k must hold one value per node")
        initial = np.where(
            operator.fixed_nodes.reshape(-1),
            problem.fixed_temperature_k.reshape(-1),
            initial,
        ) - reference
    if config is None:
        # A stiff copper/laminate stack lets the FP32 inner solve buy only
        # about one decade per outer step; give the FP64 refinement room.
        config = MPIRConfig(max_outer_iterations=16)
    result = solve_mpir(operator, rhs, config=config, initial_guess=initial)
    free = operator.free_nodes.reshape(-1)
    if not result.converged and np.any(free):
        # The FP64 residual of ``K theta`` floors at eps * ||K|| * ||theta||.  A
        # plate that sits far above the reference, almost uniformly, hits that
        # floor before the requested tolerance.  Re-reference to the mean rise
        # and continue from the current iterate; the remaining unknown is the
        # small in-plane variation, whose residual is accurate.
        shift = float(np.mean(result.solution[free]))
        if abs(shift) > 0.0:
            reference += shift
            rhs = operator.build_rhs(reference)
            resumed = solve_mpir(
                operator, rhs, config=config, initial_guess=result.solution - shift
            )
            result = MPIRResult(
                solution=resumed.solution,
                converged=resumed.converged,
                outer_iterations=result.outer_iterations + resumed.outer_iterations,
                inner_iterations=result.inner_iterations + resumed.inner_iterations,
                relative_residual=resumed.relative_residual,
                high_operator_applications=result.high_operator_applications
                + resumed.high_operator_applications,
                low_operator_applications=result.low_operator_applications
                + resumed.low_operator_applications,
                low_runtime=resumed.low_runtime,
                history=result.history + resumed.history,
            )
    temperature = result.solution.reshape(problem.mesh.node_shape) + reference

    load = operator.nodal_load()
    residual = operator.unconstrained_residual(temperature)
    fixed = operator.fixed_nodes.reshape(-1)
    convective = operator.convective_heat(temperature)
    fixed_heat = -float(np.sum(residual[fixed]))
    total_input = float(np.sum(load))
    return ThermalConductionSolution(
        temperature_k=temperature,
        heat_flux_w_per_m2=operator.element_heat_flux(temperature),
        max_temperature_k=float(np.max(temperature)),
        min_temperature_k=float(np.min(temperature)),
        total_heat_input_w=total_input,
        convective_heat_w=convective,
        fixed_temperature_heat_w=fixed_heat,
        heat_balance_error_w=total_input - float(np.sum(convective)) - fixed_heat,
        solve=result,
    )
