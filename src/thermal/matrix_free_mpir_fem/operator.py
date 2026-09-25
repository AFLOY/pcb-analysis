"""Matrix-free split-precision operator of the conduction problem.

Rows of fixed nodes (and of nodes touched by no active element) are the
identity; the lumped Robin conductance sits on the diagonal so the operator
stays symmetric positive definite.  The low-precision action runs on NumPy,
CuPy (fused kernel) or the C++ path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from electrical.matrix_free_mpir_fem.runtime import (
    LowPrecisionRuntime,
    RuntimeBackend,
    make_float32_runtime,
)
from electrical.matrix_free_mpir_fem.solver import MPIRConfig

from .mesh import Preconditioner, _corner_views, _flat_index, unit_hexahedron_matrices
from .native_hex import NativeThermalHexQ1, NativeThermalHexQ1High, native_requested
from .problem import ThermalConductionProblem
from electrical.matrix_free_mpir_fem.two_level import AggregationCoarseCorrection


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
        capacity_per_s: np.ndarray | None = None,
    ) -> None:
        """``capacity_per_s`` is the lumped nodal heat capacity over the time step,
        ``C / Δt`` in W/K, for one backward-Euler step; ``None`` is steady state."""

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
        # K_e = a_x U_x + a_y U_y + a_z U_z with per-element coefficients that
        # carry both the conductivity and the (graded) element dimensions.
        self._unit_high = np.stack(unit_hexahedron_matrices())  # (3, 8, 8)
        self._coefficients_high = np.stack(mesh.element_coefficients())  # (3, slabs, rows, cols)

        # Nodes of void elements only are held at the reference temperature:
        # they are fixed rows with a zero rise, invisible to the active part.
        self.active_nodes = mesh.active_nodes
        self.fixed_nodes = problem.fixed_temperature_mask | ~self.active_nodes
        self.free_nodes = ~self.fixed_nodes
        self.fixed_temperature_k = np.where(
            problem.fixed_temperature_mask & self.active_nodes,
            problem.fixed_temperature_k,
            0.0,
        )

        # Lumped convective conductances ``R`` and loads ``R T_amb``, kept per
        # boundary so the solution can report how much heat each face removes.
        self._robin_weights_high: list[np.ndarray] = []
        self._robin_rhs_weights_high: list[np.ndarray] = []
        self._robin_ambient_k: list[float] = []
        for boundary in problem.convection:
            weights, rhs_weights = boundary.lumped_nodal_weights(mesh)
            self._robin_weights_high.append(weights)
            self._robin_rhs_weights_high.append(rhs_weights)
            self._robin_ambient_k.append(boundary.mean_ambient_k())
        self._robin_only_high = (
            np.sum(self._robin_weights_high, axis=0)
            if self._robin_weights_high
            else np.zeros(self.size, dtype=np.float64)
        )
        if capacity_per_s is None:
            self._capacity_high = np.zeros(self.size, dtype=np.float64)
        else:
            capacity = np.asarray(capacity_per_s, dtype=np.float64).reshape(-1)
            if capacity.size != self.size:
                raise ValueError("capacity_per_s must hold one value per node")
            if not np.all(np.isfinite(capacity)) or np.any(capacity < 0.0):
                raise ValueError("capacity_per_s must be finite and non-negative")
            self._capacity_high = capacity
        # The kernels see one diagonal addition: Robin conductance plus C / Δt.
        self._robin_total_high = self._robin_only_high + self._capacity_high
        self._robin_rhs_high = (
            np.sum(self._robin_rhs_weights_high, axis=0)
            if self._robin_rhs_weights_high
            else np.zeros(self.size, dtype=np.float64)
        )

        diagonal = self._build_diagonal(
            np,
            self._coefficients_high,
            self._unit_high,
            self._robin_total_high,
            self.free_nodes,
        )
        if np.any(diagonal[self.free_nodes.reshape(-1)] <= 0.0):
            raise ValueError("every free node must have positive conductance")
        self._diagonal_high = diagonal

        runtime_ns = self.runtime.namespace
        self._coefficients_low = self.runtime.from_host(self._coefficients_high)
        self._unit_low = self.runtime.from_host(self._unit_high)
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
            self._coefficients_low = runtime_ns.ascontiguousarray(self._coefficients_low)
            self._unit_low = runtime_ns.ascontiguousarray(self._unit_low)
            self._robin_total_low = runtime_ns.ascontiguousarray(self._robin_total_low)

        use_native = self._cuda_apply is None and (native or (native is None and native_requested()))
        # The FP64 action (outer residual and coarse assembly) goes native with
        # the low path; it is built first because the coarse matrix below is
        # assembled from FP64 applications.
        self._native_high: NativeThermalHexQ1High | None = None
        self.high_operator_backend = "array-corner-products-fp64"
        if use_native:
            self._native_high = NativeThermalHexQ1High(
                mesh.element_grid_shape,
                self._coefficients_high,
                self._unit_high,
                self._robin_total_high,
                self.free_nodes,
                threads=native_threads,
            )
            self.high_operator_backend = self._native_high.kernel_name

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
        if use_native:
            # Opt-in fused C++ host path: operator plus the whole inner PCG
            # with the same preconditioner.  ``native=True`` demands it and
            # ``PCB_NATIVE_THERMAL=1`` selects it for every CPU operator.
            coarse = self.coarse_correction
            self._native = NativeThermalHexQ1(
                mesh.element_grid_shape,
                self._coefficients_high,
                self._unit_high,
                self._robin_total_high,
                self.free_nodes,
                diagonal,
                coarse_block=None if coarse is None else coarse.block,
                coarse_inverse=None if coarse is None else coarse._coarse_inverse_high,
                threads=native_threads,
            )
            self.low_operator_backend = self._native.kernel_name

    # ------------------------------------------------------------------ views
    _corner_views = staticmethod(_corner_views)

    # ---------------------------------------------------------------- actions
    def _stiffness_action(
        self,
        grid: Any,
        xp: Any,
        coefficients: Any,
        unit: Any,
    ) -> Any:
        """Unconstrained conduction action ``K T`` on a node grid.

        ``coefficients`` is ``(3, slabs, rows, cols)`` and ``unit`` the three
        ``(8, 8)`` unit matrices; the element stiffness is their contraction.
        """

        values = self._corner_views(grid)
        output = xp.zeros_like(grid)
        targets = self._corner_views(output)
        for row in range(8):
            contribution = xp.zeros_like(coefficients[0])
            for column in range(8):
                weight = (
                    coefficients[0] * unit[0, row, column]
                    + coefficients[1] * unit[1, row, column]
                    + coefficients[2] * unit[2, row, column]
                )
                contribution = contribution + weight * values[column]
            targets[row][...] += contribution
        return output

    def _apply_impl(
        self,
        vector: Any,
        xp: Any,
        coefficients: Any,
        unit: Any,
        robin: Any,
        free: Any,
    ) -> Any:
        flat = vector.reshape(-1)
        free_flat = free.reshape(-1)
        working = xp.where(free_flat, flat, xp.asarray(0.0, dtype=vector.dtype))
        conduction = self._stiffness_action(
            working.reshape(self.mesh.node_shape), xp, coefficients, unit
        ).reshape(-1)
        physical = conduction + robin * working
        return xp.where(free_flat, physical, flat)

    @staticmethod
    def _build_diagonal(
        xp: Any,
        coefficients: Any,
        unit: Any,
        robin: Any,
        free: Any,
    ) -> Any:
        diagonal = xp.zeros(free.shape, dtype=coefficients.dtype)
        targets = MatrixFreeThermalOperator._corner_views(diagonal)
        for index in range(8):
            targets[index][...] += (
                coefficients[0] * unit[0, index, index]
                + coefficients[1] * unit[1, index, index]
                + coefficients[2] * unit[2, index, index]
            )
        flat = diagonal.reshape(-1) + robin
        return xp.where(free.reshape(-1), flat, xp.asarray(1.0, dtype=flat.dtype))

    def apply_high(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        if vector.size != self.size:
            raise ValueError(f"vector has size {vector.size}, expected {self.size}")
        if self._native_high is not None:
            return self._native_high.apply(vector)
        return self._apply_impl(
            vector, np, self._coefficients_high, self._unit_high, self._robin_total_high, self.free_nodes
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
                vector, self._coefficients_low, self._unit_low, self._robin_total_low, self._free_low_u8
            )
        return self._apply_impl(
            vector, self.runtime.namespace, self._coefficients_low, self._unit_low, self._robin_total_low, self._free_low
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

        load = self.problem.nodal_heat_w.copy()
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
        mask = self.problem.fixed_temperature_mask & self.active_nodes
        return float(np.mean(self.problem.fixed_temperature_k[mask]))

    def _capacity_load(self, previous_temperature_k: np.ndarray | None) -> np.ndarray:
        """``C / Δt · T_n`` for a backward-Euler step, zero in steady state."""

        if previous_temperature_k is None:
            if np.any(self._capacity_high > 0.0):
                raise ValueError("a transient operator needs previous_temperature_k")
            return np.zeros(self.size, dtype=np.float64)
        previous = np.asarray(previous_temperature_k, dtype=np.float64).reshape(-1)
        if previous.size != self.size:
            raise ValueError("previous_temperature_k must hold one value per node")
        return self._capacity_high * np.where(np.isfinite(previous), previous, 0.0)

    def build_rhs(
        self, reference_temperature_k: float = 0.0, previous_temperature_k: np.ndarray | None = None
    ) -> np.ndarray:
        """Right-hand side for the temperature rise above a reference.

        The unknown is ``theta = T - reference``.  Solving for the rise instead
        of the absolute temperature keeps the FP64 residual free of the
        cancellation ``K (300 K) ~ 0`` that otherwise caps attainable accuracy
        at about ``eps * ||A|| * 300 K / ||q||``.  Fixed temperatures move onto
        the free rows through the unconstrained stiffness action, and their
        own rows carry the shifted value.  A transient step adds ``C / Δt T_n``
        from ``previous_temperature_k``.
        """

        reference = float(reference_temperature_k)
        mask = self.fixed_nodes.reshape(-1)
        active = self.active_nodes.reshape(-1)
        shifted_fixed = np.where(
            mask & active, self.fixed_temperature_k.reshape(-1) - reference, 0.0
        )
        boundary_action = self._stiffness_action(
            shifted_fixed.reshape(self.mesh.node_shape), np, self._coefficients_high, self._unit_high
        ).reshape(-1)
        robin_rhs = self._robin_rhs_high - self._robin_total_high * reference
        rhs = self.nodal_load() + robin_rhs + self._capacity_load(previous_temperature_k) - boundary_action
        rhs[mask] = shifted_fixed[mask]
        return rhs

    # ---------------------------------------------------------- post-process
    def stored_heat_w(
        self, temperature_k: np.ndarray, previous_temperature_k: np.ndarray | None
    ) -> np.ndarray:
        """``C / Δt (T - T_n)`` per node: the heat going into the thermal mass this step."""

        flat = np.asarray(temperature_k, dtype=np.float64).reshape(-1)
        if previous_temperature_k is None:
            return np.zeros(self.size, dtype=np.float64)
        previous = np.asarray(previous_temperature_k, dtype=np.float64).reshape(-1)
        finite = np.isfinite(flat) & np.isfinite(previous)
        return np.where(finite, self._capacity_high * (flat - previous), 0.0)

    def unconstrained_residual(
        self, temperature_k: np.ndarray, previous_temperature_k: np.ndarray | None = None
    ) -> np.ndarray:
        """``K T + R (T - T_amb) + C/Δt (T - T_n) - q`` at every node, including fixed ones.

        At free nodes this is the solver's residual.  At fixed nodes its
        negative is the heat the fixed-temperature sink absorbs to hold that
        temperature, because the row balance reads ``q = K T + R (T - T_amb)
        + sink``.
        """

        flat = np.asarray(temperature_k, dtype=np.float64).reshape(-1)
        conduction = self._stiffness_action(
            flat.reshape(self.mesh.node_shape), np, self._coefficients_high, self._unit_high
        ).reshape(-1)
        return (
            conduction
            + self._robin_only_high * flat
            - self._robin_rhs_high
            + self.stored_heat_w(flat, previous_temperature_k)
            - self.nodal_load()
        )

    def convective_heat(self, temperature_k: np.ndarray) -> np.ndarray:
        """Heat removed by each convection boundary, in W, in problem order."""

        flat = np.asarray(temperature_k, dtype=np.float64).reshape(-1)
        return np.asarray(
            [
                float(np.sum(weights * flat) - np.sum(rhs_weights))
                for weights, rhs_weights in zip(
                    self._robin_weights_high, self._robin_rhs_weights_high
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
        gradient_x = (x_plus - x_minus) / (4.0 * self.mesh.pitch_x_m[None, None, :])
        gradient_y = (y_plus - y_minus) / (4.0 * self.mesh.pitch_y_m[None, :, None])
        gradient_z = (z_plus - z_minus) / (4.0 * thickness)
        in_plane = np.asarray(self.mesh.conductivity_w_per_m_k, dtype=np.float64)
        through = np.asarray(self.mesh.through_plane_conductivity_w_per_m_k, dtype=np.float64)
        return np.stack(
            (-in_plane * gradient_x, -in_plane * gradient_y, -through * gradient_z),
            axis=-1,
        )
