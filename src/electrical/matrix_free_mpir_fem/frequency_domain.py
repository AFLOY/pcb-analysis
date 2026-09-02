"""2D scalar-polarisation frequency-domain Maxwell FEM.

The solved phasor is ``E_z(x, y)`` with an ``exp(+j omega t)`` convention.  It
is the exact scalar reduction of Maxwell's equations for geometry and material
fields invariant in ``z``.  Conductivity produces eddy currents and skin
effect; complex permittivity produces dielectric loss; the displacement term
retains wave propagation.  Arbitrary 3D vector fields remain outside scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .runtime import (
    LowPrecisionRuntime,
    RuntimeBackend,
    make_complex64_runtime,
)
from .solver import MPIRConfig, MPIRResult, solve_mpir


MU_0_H_PER_M = 4.0e-7 * np.pi
EPSILON_0_F_PER_M = 8.8541878128e-12


def _element_field(
    value: float | Sequence[float] | np.ndarray,
    shape: tuple[int, int],
    name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(shape, array, dtype=np.float64)
    elif array.shape != shape:
        raise ValueError(f"{name} must be scalar or have element shape {shape}")
    else:
        array = array.copy()
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be positive")
    if non_negative and np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


@dataclass(frozen=True)
class ScalarMaxwellMesh2D:
    """Structured bilinear-element material mesh for scalar ``E_z``."""

    element_shape: tuple[int, int]
    pitch_x_m: float
    pitch_y_m: float
    relative_permittivity: float | np.ndarray = 1.0
    relative_permeability: float | np.ndarray = 1.0
    conductivity_s_per_m: float | np.ndarray = 0.0
    dielectric_loss_tangent: float | np.ndarray = 0.0

    def __post_init__(self) -> None:
        rows, columns = (int(value) for value in self.element_shape)
        if rows < 1 or columns < 1:
            raise ValueError("element_shape axes must be positive")
        object.__setattr__(self, "element_shape", (rows, columns))
        for name in ("pitch_x_m", "pitch_y_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        fields = (
            ("relative_permittivity", True, False),
            ("relative_permeability", True, False),
            ("conductivity_s_per_m", False, True),
            ("dielectric_loss_tangent", False, True),
        )
        for name, positive, non_negative in fields:
            object.__setattr__(
                self,
                name,
                _element_field(
                    getattr(self, name),
                    (rows, columns),
                    name,
                    positive=positive,
                    non_negative=non_negative,
                ),
            )

    @property
    def node_shape(self) -> tuple[int, int]:
        return self.element_shape[0] + 1, self.element_shape[1] + 1

    @property
    def size(self) -> int:
        return int(np.prod(self.node_shape))

    @property
    def length_x_m(self) -> float:
        return self.element_shape[1] * self.pitch_x_m

    @property
    def length_y_m(self) -> float:
        return self.element_shape[0] * self.pitch_y_m


@dataclass(frozen=True)
class ScalarMaxwellProblem:
    mesh: ScalarMaxwellMesh2D
    frequency_hz: float
    dirichlet_mask: np.ndarray
    dirichlet_electric_field_v_per_m: np.ndarray
    nodal_source: np.ndarray | None = None

    def __post_init__(self) -> None:
        frequency = float(self.frequency_hz)
        if not np.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("frequency_hz must be finite and positive")
        mask = np.asarray(self.dirichlet_mask, dtype=bool)
        values = np.asarray(
            self.dirichlet_electric_field_v_per_m, dtype=np.complex128
        )
        if mask.shape != self.mesh.node_shape or values.shape != self.mesh.node_shape:
            raise ValueError("Dirichlet mask and values must match mesh.node_shape")
        if not np.any(mask):
            raise ValueError("at least one Dirichlet node is required")
        if not np.all(np.isfinite(values)):
            raise ValueError("Dirichlet values must be finite")
        source = (
            np.zeros(self.mesh.node_shape, dtype=np.complex128)
            if self.nodal_source is None
            else np.asarray(self.nodal_source, dtype=np.complex128)
        )
        if source.shape != self.mesh.node_shape or not np.all(np.isfinite(source)):
            raise ValueError("nodal_source must be finite and match mesh.node_shape")
        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(self, "dirichlet_mask", mask.copy())
        object.__setattr__(self, "dirichlet_electric_field_v_per_m", values.copy())
        object.__setattr__(self, "nodal_source", source.copy())


def _q1_matrices(pitch_x_m: float, pitch_y_m: float) -> tuple[np.ndarray, np.ndarray]:
    stiffness_1d = np.array([[1.0, -1.0], [-1.0, 1.0]])
    mass_1d = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0
    stiffness = (
        (pitch_y_m / pitch_x_m) * np.kron(mass_1d, stiffness_1d)
        + (pitch_x_m / pitch_y_m) * np.kron(stiffness_1d, mass_1d)
    )
    mass = pitch_x_m * pitch_y_m * np.kron(mass_1d, mass_1d)
    return stiffness, mass


class MatrixFreeScalarMaxwellOperator:
    """Complex matrix-free Q1 action for the scalar full-wave equation."""

    high_dtype = np.complex128
    inner_solver = "gmres"

    def __init__(
        self,
        problem: ScalarMaxwellProblem,
        *,
        runtime: LowPrecisionRuntime | None = None,
        backend: RuntimeBackend | None = None,
        device_id: int = 0,
    ) -> None:
        if runtime is not None and backend is not None:
            raise ValueError("pass either runtime or backend, not both")
        self.problem = problem
        self.mesh = problem.mesh
        self.size = self.mesh.size
        self.runtime = runtime or make_complex64_runtime(
            backend or "cpu", device_id=device_id
        )
        self.free_nodes = ~problem.dirichlet_mask
        self._free_low = self.runtime.namespace.asarray(self.free_nodes, dtype=bool)
        stiffness, mass = _q1_matrices(
            self.mesh.pitch_x_m, self.mesh.pitch_y_m
        )
        self._stiffness_high = stiffness.astype(np.complex128)
        self._mass_high = mass.astype(np.complex128)

        omega = 2.0 * np.pi * problem.frequency_hz
        epsilon = (
            EPSILON_0_F_PER_M
            * np.asarray(self.mesh.relative_permittivity)
            * (1.0 - 1j * np.asarray(self.mesh.dielectric_loss_tangent))
        )
        permeability = MU_0_H_PER_M * np.asarray(
            self.mesh.relative_permeability
        )
        conductivity = np.asarray(self.mesh.conductivity_s_per_m)
        self._inverse_mu_high = (1.0 / permeability).astype(np.complex128)
        self._reaction_high = (
            1j * omega * conductivity - omega * omega * epsilon
        ).astype(np.complex128)
        self._inverse_mu_low = self.runtime.from_host(self._inverse_mu_high)
        self._reaction_low = self.runtime.from_host(self._reaction_high)
        self._stiffness_low = self.runtime.from_host(self._stiffness_high)
        self._mass_low = self.runtime.from_host(self._mass_high)

        diagonal = self._diagonal_impl(
            np,
            self._inverse_mu_high,
            self._reaction_high,
            self._stiffness_high,
            self._mass_high,
            self.free_nodes,
        )
        free_diagonal = diagonal[self.free_nodes.reshape(-1)]
        threshold = np.finfo(np.float64).eps * max(1.0, float(np.max(abs(diagonal))))
        if np.any(abs(free_diagonal) <= threshold):
            raise ValueError("Jacobi diagonal is singular at this frequency")
        self._diagonal_low = self.runtime.from_host(diagonal)
        self._cuda_apply = None
        self.low_operator_backend = "portable-array-q1"
        if getattr(self.runtime, "is_cuda", False):
            from .cuda import CudaScalarMaxwellQ1Apply

            self._cuda_apply = CudaScalarMaxwellQ1Apply(
                self.runtime, self.mesh.element_shape
            )
            self.low_operator_backend = self._cuda_apply.kernel_name

    @staticmethod
    def _views(grid: Any) -> tuple[Any, Any, Any, Any]:
        return (
            grid[:-1, :-1],
            grid[:-1, 1:],
            grid[1:, :-1],
            grid[1:, 1:],
        )

    def _physical_action(
        self,
        vector: Any,
        xp: Any,
        inverse_mu: Any,
        reaction: Any,
        stiffness: Any,
        mass: Any,
    ) -> Any:
        grid = vector.reshape(self.mesh.node_shape)
        values = self._views(grid)
        output = xp.zeros_like(grid)
        targets = self._views(output)
        for row in range(4):
            contribution = xp.zeros_like(inverse_mu)
            for column in range(4):
                coefficient = (
                    inverse_mu * stiffness[row, column]
                    + reaction * mass[row, column]
                )
                contribution = contribution + coefficient * values[column]
            targets[row][...] += contribution
        return output.reshape(-1)

    def _apply_impl(
        self,
        vector: Any,
        xp: Any,
        inverse_mu: Any,
        reaction: Any,
        stiffness: Any,
        mass: Any,
        free: Any,
    ) -> Any:
        zero = xp.asarray(0.0, dtype=vector.dtype)
        working = xp.where(free.reshape(-1), vector.reshape(-1), zero)
        physical = self._physical_action(
            working, xp, inverse_mu, reaction, stiffness, mass
        )
        return xp.where(free.reshape(-1), physical, vector.reshape(-1))

    def _diagonal_impl(
        self,
        xp: Any,
        inverse_mu: Any,
        reaction: Any,
        stiffness: Any,
        mass: Any,
        free: Any,
    ) -> Any:
        diagonal = xp.zeros(self.mesh.node_shape, dtype=reaction.dtype)
        targets = self._views(diagonal)
        for index in range(4):
            targets[index][...] += (
                inverse_mu * stiffness[index, index]
                + reaction * mass[index, index]
            )
        flat = diagonal.reshape(-1)
        return xp.where(
            free.reshape(-1), flat, xp.asarray(1.0, dtype=flat.dtype)
        )

    def apply_high(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        if vector.size != self.size:
            raise ValueError(f"vector has size {vector.size}, expected {self.size}")
        return self._apply_impl(
            vector,
            np,
            self._inverse_mu_high,
            self._reaction_high,
            self._stiffness_high,
            self._mass_high,
            self.free_nodes,
        )

    def apply_low(self, vector: Any) -> Any:
        if self._cuda_apply is not None:
            vector = self.runtime.namespace.asarray(
                vector, dtype=self.runtime.dtype
            ).reshape(-1)
            if vector.size != self.size:
                raise ValueError(
                    f"vector has size {vector.size}, expected {self.size}"
                )
            if not vector.flags.c_contiguous:
                vector = self.runtime.namespace.ascontiguousarray(vector)
            return self._cuda_apply(
                vector,
                self._inverse_mu_low,
                self._reaction_low,
                self._stiffness_low,
                self._mass_low,
                self._free_low,
            )
        return self._apply_impl(
            vector,
            self.runtime.namespace,
            self._inverse_mu_low,
            self._reaction_low,
            self._stiffness_low,
            self._mass_low,
            self._free_low,
        )

    def diagonal_low(self) -> Any:
        return self._diagonal_low

    def build_rhs(self) -> np.ndarray:
        boundary = np.where(
            self.problem.dirichlet_mask,
            self.problem.dirichlet_electric_field_v_per_m,
            0.0,
        ).reshape(-1)
        boundary_action = self._physical_action(
            boundary,
            np,
            self._inverse_mu_high,
            self._reaction_high,
            self._stiffness_high,
            self._mass_high,
        )
        source = np.asarray(self.problem.nodal_source).reshape(-1)
        rhs = source - boundary_action
        mask = self.problem.dirichlet_mask.reshape(-1)
        rhs[mask] = boundary[mask]
        return rhs

    def element_fields(
        self, electric_field: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return element-centre ``E_z``, vector ``H_xy``, and ``J_z``."""

        electric = np.asarray(electric_field, dtype=np.complex128).reshape(
            self.mesh.node_shape
        )
        e00, e01, e10, e11 = self._views(electric)
        centre = (e00 + e01 + e10 + e11) / 4.0
        derivative_x = ((e01 + e11) - (e00 + e10)) / (
            2.0 * self.mesh.pitch_x_m
        )
        derivative_y = ((e10 + e11) - (e00 + e01)) / (
            2.0 * self.mesh.pitch_y_m
        )
        omega = 2.0 * np.pi * self.problem.frequency_hz
        permeability = MU_0_H_PER_M * np.asarray(
            self.mesh.relative_permeability
        )
        magnetic_x = 1j * derivative_y / (omega * permeability)
        magnetic_y = -1j * derivative_x / (omega * permeability)
        magnetic = np.stack((magnetic_x, magnetic_y), axis=-1)
        current = np.asarray(self.mesh.conductivity_s_per_m) * centre
        return centre, magnetic, current

    def losses(self, electric_field: np.ndarray) -> tuple[float, float]:
        """Return conduction and dielectric loss in W per metre of z depth."""

        electric = np.asarray(electric_field, dtype=np.complex128).reshape(
            self.mesh.node_shape
        )
        element_values = np.stack(self._views(electric), axis=-1)
        norm_integral = np.einsum(
            "...i,ij,...j->...",
            element_values.conj(),
            self._mass_high.real,
            element_values,
            optimize=True,
        ).real
        omega = 2.0 * np.pi * self.problem.frequency_hz
        conduction = 0.5 * np.sum(
            np.asarray(self.mesh.conductivity_s_per_m) * norm_integral
        )
        dielectric = 0.5 * omega * EPSILON_0_F_PER_M * np.sum(
            np.asarray(self.mesh.relative_permittivity)
            * np.asarray(self.mesh.dielectric_loss_tangent)
            * norm_integral
        )
        return float(conduction), float(dielectric)


@dataclass(frozen=True)
class ScalarMaxwellSolution:
    electric_field_z_v_per_m: np.ndarray
    element_electric_field_z_v_per_m: np.ndarray
    magnetic_field_xy_a_per_m: np.ndarray
    eddy_current_density_z_a_per_m2: np.ndarray
    conduction_loss_w_per_m: float
    dielectric_loss_w_per_m: float
    solve: MPIRResult


def solve_scalar_maxwell(
    problem: ScalarMaxwellProblem,
    *,
    config: MPIRConfig | None = None,
    runtime: LowPrecisionRuntime | None = None,
    backend: RuntimeBackend | None = None,
    device_id: int = 0,
) -> ScalarMaxwellSolution:
    """Solve the scalar 2D full-wave Maxwell boundary-value problem.

    ``backend="cuda"`` keeps the inner complex64 GMRES vectors and fused Q1
    operator on the selected GPU.  The reliable complex128 residual and final
    field post-processing intentionally remain on the host.
    """

    operator = MatrixFreeScalarMaxwellOperator(
        problem,
        runtime=runtime,
        backend=backend,
        device_id=device_id,
    )
    result = solve_mpir(operator, operator.build_rhs(), config=config)
    electric = result.solution.reshape(problem.mesh.node_shape)
    centre, magnetic, current = operator.element_fields(electric)
    conduction, dielectric = operator.losses(electric)
    return ScalarMaxwellSolution(
        electric_field_z_v_per_m=electric,
        element_electric_field_z_v_per_m=centre,
        magnetic_field_xy_a_per_m=magnetic,
        eddy_current_density_z_a_per_m2=current,
        conduction_loss_w_per_m=conduction,
        dielectric_loss_w_per_m=dielectric,
        solve=result,
    )


def propagation_constant_per_m(
    frequency_hz: float,
    *,
    relative_permittivity: float = 1.0,
    relative_permeability: float = 1.0,
    conductivity_s_per_m: float = 0.0,
    dielectric_loss_tangent: float = 0.0,
) -> complex:
    """Return ``gamma`` for the package's ``exp(+j omega t)`` convention."""

    omega = 2.0 * np.pi * float(frequency_hz)
    epsilon = (
        EPSILON_0_F_PER_M
        * float(relative_permittivity)
        * (1.0 - 1j * float(dielectric_loss_tangent))
    )
    permeability = MU_0_H_PER_M * float(relative_permeability)
    value = 1j * omega * permeability * float(conductivity_s_per_m)
    value -= omega * omega * permeability * epsilon
    root = complex(np.sqrt(value + 0j))
    return root if root.real >= 0.0 else -root


def skin_depth_m(
    frequency_hz: float,
    conductivity_s_per_m: float,
    *,
    relative_permeability: float = 1.0,
) -> float:
    """Good-conductor skin depth ``sqrt(2 / (omega mu sigma))``."""

    omega = 2.0 * np.pi * float(frequency_hz)
    denominator = (
        omega
        * MU_0_H_PER_M
        * float(relative_permeability)
        * float(conductivity_s_per_m)
    )
    if denominator <= 0.0:
        raise ValueError("frequency, permeability, and conductivity must be positive")
    return float(np.sqrt(2.0 / denominator))
