"""CUDA execution for the layered sheet-PEEC system.

The physical discretisation is shared with :mod:`electrical.dice_peec.sheet_peec`.
Only the prepared convolution spectra, sparse incidence products, Krylov
vectors, and preconditioner solves move to CuPy.  There is deliberately no CPU
fallback in this module: a requested CUDA solve either reports a CUDA backend
or raises.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Sequence

import numpy as np

from .sheet_operator import SheetInductanceOperator
from .sheet_peec import (
    SheetMesh,
    SheetSolution,
    Terminal,
    _components,
    _components_with_terminals,
    _self_inductance,
    _source_vector,
    _vertical_self_inductance,
)


class CudaSheetUnavailableError(RuntimeError):
    """Raised when the requested CUDA sheet backend cannot initialize."""


# The Krylov call below passes ``rtol``, which CuPy renamed from ``tol`` in 14.
# The distribution requirement already states this, but an environment can
# satisfy the import and not the requirement, and the mismatch then surfaces
# from inside the solve as a bare ``TypeError``.
MINIMUM_CUPY_MAJOR_VERSION = 14


class CudaSheetSolveError(RuntimeError):
    """Raised when CUDA initialized but the sheet solve failed."""


@dataclass(frozen=True)
class CudaSheetTelemetry:
    """Runtime identity, timing, and device allocation for one solve."""

    resolved_backend: str
    device_id: int
    device_name: str
    compute_capability: str
    driver_version: int
    runtime_version: int
    cupy_version: str
    prepare_ms: float
    solve_ms: float
    total_ms: float
    krylov_relative_tolerance: float
    memory_pool_peak_bytes: int
    free_memory_before_bytes: int
    free_memory_after_bytes: int
    preconditioner_factorization_backend: str
    fallback_used: bool = False

    def as_metrics(self) -> dict[str, Any]:
        return {
            "backend": "cuda_sheet_peec",
            "resolved_backend": self.resolved_backend,
            "cuda_device_id": self.device_id,
            "cuda_device_name": self.device_name,
            "cuda_compute_capability": self.compute_capability,
            "cuda_driver_version": self.driver_version,
            "cuda_runtime_version": self.runtime_version,
            "cupy_version": self.cupy_version,
            "cuda_prepare_ms": self.prepare_ms,
            "cuda_solve_ms": self.solve_ms,
            "cuda_total_ms": self.total_ms,
            "cuda_krylov_relative_tolerance": (
                self.krylov_relative_tolerance
            ),
            "cuda_memory_pool_peak_bytes": self.memory_pool_peak_bytes,
            "cuda_free_memory_before_bytes": self.free_memory_before_bytes,
            "cuda_free_memory_after_bytes": self.free_memory_after_bytes,
            "preconditioner_factorization_backend": (
                self.preconditioner_factorization_backend
            ),
            "fallback_used": self.fallback_used,
        }


class CudaSheetInductanceOperator:
    """Device-resident spectra for the sheet convolution operator."""

    def __init__(self, source: SheetInductanceOperator, cp: Any) -> None:
        self.cp = cp
        self.shape = source.shape
        self.padded = source.padded
        self.layer_count = len(source.stackup)
        self.vertical_levels = source.vertical_levels
        self._kernels = {
            key: cp.asarray(value) for key, value in source._kernels.items()
        }
        self._kernels_z = {
            key: cp.asarray(value) for key, value in source._kernels_z.items()
        }

    def _kernel(self, axis: str, first: int, second: int) -> Any:
        key = (
            (axis, first, second)
            if first <= second
            else (axis, second, first)
        )
        return self._kernels[key]

    def _kernel_z(self, first: int, second: int) -> Any:
        key = (
            (first, second)
            if first <= second
            else (second, first)
        )
        return self._kernels_z[key]

    @property
    def kernel_bytes(self) -> int:
        return sum(int(value.nbytes) for value in self._kernels.values()) + sum(
            int(value.nbytes) for value in self._kernels_z.values()
        )

    def _convolve(self, spectra: Any, axis: str) -> Any:
        cp = self.cp
        rows, cols = self.shape
        output = cp.empty(
            (self.layer_count, rows, cols), dtype=cp.float64
        )
        for target in range(self.layer_count):
            accumulated = cp.zeros_like(spectra[0])
            for source in range(self.layer_count):
                accumulated += (
                    self._kernel(axis, target, source) * spectra[source]
                )
            product = cp.fft.irfft2(accumulated, s=self.padded)
            output[target] = product[:rows, :cols]
        return output

    def _convolve_z(self, spectra: Any) -> Any:
        cp = self.cp
        rows, cols = self.shape
        output = cp.empty(
            (len(self.vertical_levels), rows, cols), dtype=cp.float64
        )
        for target in range(len(self.vertical_levels)):
            accumulated = cp.zeros_like(spectra[0])
            for source in range(len(self.vertical_levels)):
                accumulated += self._kernel_z(target, source) * spectra[source]
            product = cp.fft.irfft2(accumulated, s=self.padded)
            output[target] = product[:rows, :cols]
        return output

    def apply(
        self,
        currents_x: Any,
        currents_y: Any,
        currents_z: Any | None = None,
    ) -> tuple[Any, Any] | tuple[Any, Any, Any]:
        cp = self.cp
        rows, cols = self.shape
        expected = (self.layer_count, rows, cols)
        for name, array in (
            ("currents_x", currents_x),
            ("currents_y", currents_y),
        ):
            if array.shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {array.shape}"
                )
        spectra_x = cp.fft.rfft2(
            currents_x, s=self.padded, axes=(-2, -1)
        )
        spectra_y = cp.fft.rfft2(
            currents_y, s=self.padded, axes=(-2, -1)
        )
        flux_x = self._convolve(spectra_x, "x")
        flux_y = self._convolve(spectra_y, "y")
        if currents_z is None:
            return flux_x, flux_y
        vertical_expected = (len(self.vertical_levels), rows, cols)
        if currents_z.shape != vertical_expected:
            raise ValueError(
                f"currents_z must have shape {vertical_expected}, "
                f"got {currents_z.shape}"
            )
        if not self.vertical_levels:
            return flux_x, flux_y, cp.zeros_like(currents_z)
        spectra_z = cp.fft.rfft2(
            currents_z, s=self.padded, axes=(-2, -1)
        )
        return flux_x, flux_y, self._convolve_z(spectra_z)


@dataclass(frozen=True)
class _BranchIndices:
    x_layer: Any
    x_row: Any
    x_col: Any
    y_layer: Any
    y_row: Any
    y_col: Any
    z_level: Any
    z_row: Any
    z_col: Any


def _branch_indices(mesh: SheetMesh, cp: Any) -> _BranchIndices:
    levels = {pair: index for index, pair in enumerate(mesh.vertical_levels)}

    def _axis(
        values: Sequence[tuple[int, int, int]],
    ) -> tuple[Any, Any, Any]:
        return tuple(
            cp.asarray([item[index] for item in values], dtype=cp.int32)
            for index in range(3)
        )

    x_layer, x_row, x_col = _axis(mesh.branch_x)
    y_layer, y_row, y_col = _axis(mesh.branch_y)
    z_level = cp.asarray(
        [
            levels[(via.lower_layer, via.upper_layer)]
            for via in mesh.via_branches
        ],
        dtype=cp.int32,
    )
    z_row = cp.asarray(
        [via.row for via in mesh.via_branches], dtype=cp.int32
    )
    z_col = cp.asarray(
        [via.col for via in mesh.via_branches], dtype=cp.int32
    )
    return _BranchIndices(
        x_layer=x_layer,
        x_row=x_row,
        x_col=x_col,
        y_layer=y_layer,
        y_row=y_row,
        y_col=y_col,
        z_level=z_level,
        z_row=z_row,
        z_col=z_col,
    )


def _cuda_identity(cp: Any, device_id: int) -> dict[str, Any]:
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    raw_name = properties["name"]
    name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
    return {
        "device_name": name,
        "compute_capability": (
            f"{int(properties['major'])}.{int(properties['minor'])}"
        ),
        "driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cupy_version": str(cp.__version__),
    }


def _import_cupy() -> Any:
    try:
        import cupy as cp
    except (ImportError, OSError) as error:
        raise CudaSheetUnavailableError(
            "CuPy with a matching CUDA runtime is required"
        ) from error
    _require_cupy_version(cp)
    return cp


def _require_cupy_version(cp: Any) -> None:
    """Reject an unsupported CuPy before it can fail from inside the solve."""
    version = str(getattr(cp, "__version__", ""))
    try:
        major = int(version.split(".", 1)[0])
    except ValueError:
        raise CudaSheetUnavailableError(
            f"cannot read the installed CuPy version from {version!r}"
        ) from None
    if major < MINIMUM_CUPY_MAJOR_VERSION:
        raise CudaSheetUnavailableError(
            f"CuPy {version} is installed, but the CUDA sheet backend needs "
            f"{MINIMUM_CUPY_MAJOR_VERSION} or newer for the Krylov solver's "
            "keyword arguments"
        )


def solve_sheet_case_cuda(
    mesh: SheetMesh,
    operator: SheetInductanceOperator,
    terminals: Sequence[Terminal],
    *,
    frequency_hz: float = 0.0,
    tolerance: float = 1e-10,
    max_iterations: int = 400,
    restart: int = 60,
    device_id: int = 0,
    cupy_module: Any | None = None,
) -> tuple[SheetSolution, CudaSheetTelemetry]:
    """Solve one sheet case with cuFFT, CuPy sparse algebra, and CUDA GMRES."""
    cp = cupy_module if cupy_module is not None else _import_cupy()
    try:
        import cupyx.scipy.sparse as csp
        import cupyx.scipy.sparse.linalg as csl
    except (ImportError, OSError) as error:
        raise CudaSheetUnavailableError(
            "cupyx.scipy.sparse is required for the CUDA sheet backend"
        ) from error

    if operator.shape != mesh.shape:
        raise ValueError("the operator and the mesh must share a shape")
    if len(operator.stackup) != len(mesh.stackup):
        raise ValueError("the operator and the mesh must share a stackup")
    frequency_hz = float(frequency_hz)
    if not math.isfinite(frequency_hz) or frequency_hz < 0.0:
        raise ValueError("frequency must be finite and non-negative")
    device_id = int(device_id)
    if device_id < 0:
        raise ValueError("device_id must be non-negative")

    incidence = mesh.incidence()
    resistance = mesh.resistances()
    injected = _source_vector(mesh, terminals)
    if frequency_hz == 0.0 and float(np.max(np.abs(injected.imag))) > 1e-15:
        raise ValueError("DC terminal currents must be real")
    components, labels = _components(incidence, mesh.node_count)
    carried = _components_with_terminals(labels, components, injected)
    if not carried:
        raise ValueError(
            "no connected component of the conductor carries a terminal"
        )
    kept_nodes = np.isin(labels, list(carried))
    dropped = int(mesh.node_count - kept_nodes.sum())
    for component in sorted(carried):
        total = complex(injected[labels == component].sum())
        scale = float(np.abs(injected[labels == component]).sum())
        if abs(total) > 1e-9 * max(1.0, scale):
            raise ValueError(
                f"connected component {component} is given a net "
                f"{total:.6g} A; each isolated component has to close its own "
                "current"
            )
    keep = kept_nodes.copy()
    grounded_nodes = []
    for component in sorted(carried):
        first = int(np.flatnonzero(labels == component)[0])
        keep[first] = False
        grounded_nodes.append(first)
    grounded = grounded_nodes[0]
    driven_endpoint_count = np.asarray(
        abs(incidence) @ kept_nodes.astype(np.int8)
    ).reshape(-1)
    active_branches = driven_endpoint_count == 2
    active_incidence = incidence[active_branches]
    active_resistance = resistance[active_branches]
    reduced_cpu = active_incidence[:, keep]

    total_started = time.perf_counter()
    device = None
    pool = None
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_id >= device_count:
            raise CudaSheetUnavailableError(
                f"CUDA device {device_id} is unavailable; "
                f"device count is {device_count}"
            )
        device = cp.cuda.Device(device_id)
        pool = cp.cuda.MemoryPool()
        with device:
            identity = _cuda_identity(cp, device_id)
            free_before, _ = cp.cuda.runtime.memGetInfo()
            with cp.cuda.using_allocator(pool.malloc):
                prepare_started = time.perf_counter()
                reduced = csp.csr_matrix(reduced_cpu)
                injected_device = cp.asarray(injected)
                resistance_device = cp.asarray(active_resistance)
                active_branches_device = cp.asarray(active_branches)
                cuda_operator = CudaSheetInductanceOperator(operator, cp)
                indices = _branch_indices(mesh, cp)
                cp.cuda.get_current_stream().synchronize()
                prepare_ms = (
                    time.perf_counter() - prepare_started
                ) * 1000.0

                solve_started = time.perf_counter()
                if frequency_hz == 0.0:
                    admittance = csp.diags(1.0 / resistance_device)
                    system = (reduced.T @ admittance @ reduced).tocsr()
                    solved_voltage = csl.spsolve(
                        system, injected_device[keep].real
                    )
                    voltage_device = cp.zeros(
                        mesh.node_count, dtype=cp.complex128
                    )
                    voltage_device[keep] = solved_voltage
                    active_current_device = admittance @ (
                        reduced @ solved_voltage
                    )
                    current_device = cp.zeros(
                        mesh.branch_count, dtype=cp.complex128
                    )
                    current_device[active_branches_device] = (
                        active_current_device
                    )
                    residual_device = (
                        cp.linalg.norm(
                            reduced.T @ active_current_device
                            - injected_device[keep].real
                        )
                        / max(
                            float(
                                cp.linalg.norm(
                                    injected_device[keep].real
                                ).get()
                            ),
                            1e-30,
                        )
                    )
                    residual = float(residual_device.get())
                    iterations = 1
                    converged = residual < 1e-8
                    factor_backend = "cusolver_sparse_qr"
                else:
                    levels = mesh.vertical_levels
                    if tuple(levels) != tuple(operator.vertical_levels):
                        raise ValueError(
                            "the mesh and operator vertical levels differ"
                        )
                    has_vertical = bool(levels)
                    inline_count = len(mesh.branch_x) + len(mesh.branch_y)
                    branches = int(active_branches.sum())
                    unknowns = int(keep.sum())
                    size = branches + unknowns
                    omega = 2.0 * math.pi * frequency_hz

                    def _flux(currents: Any) -> Any:
                        rows, cols = mesh.shape
                        full_currents = cp.zeros(
                            mesh.branch_count, dtype=cp.float64
                        )
                        full_currents[active_branches_device] = currents
                        grid_x = cp.zeros(
                            (len(mesh.stackup), rows, cols),
                            dtype=cp.float64,
                        )
                        grid_y = cp.zeros_like(grid_x)
                        x_count = len(mesh.branch_x)
                        y_count = len(mesh.branch_y)
                        grid_x[
                            indices.x_layer,
                            indices.x_row,
                            indices.x_col,
                        ] = full_currents[:x_count]
                        grid_y[
                            indices.y_layer,
                            indices.y_row,
                            indices.y_col,
                        ] = full_currents[x_count : x_count + y_count]
                        if has_vertical:
                            grid_z = cp.zeros(
                                (len(levels), rows, cols),
                                dtype=cp.float64,
                            )
                            grid_z[
                                indices.z_level,
                                indices.z_row,
                                indices.z_col,
                            ] = full_currents[x_count + y_count :]
                            flux_x, flux_y, flux_z = cuda_operator.apply(
                                grid_x, grid_y, grid_z
                            )
                        else:
                            flux_x, flux_y = cuda_operator.apply(
                                grid_x, grid_y
                            )
                            flux_z = None
                        full_output = cp.zeros(
                            mesh.branch_count, dtype=cp.float64
                        )
                        full_output[:x_count] = flux_x[
                            indices.x_layer,
                            indices.x_row,
                            indices.x_col,
                        ]
                        full_output[x_count : x_count + y_count] = flux_y[
                            indices.y_layer,
                            indices.y_row,
                            indices.y_col,
                        ]
                        if flux_z is not None:
                            full_output[x_count + y_count :] = flux_z[
                                indices.z_level,
                                indices.z_row,
                                indices.z_col,
                            ]
                        return full_output[active_branches_device]

                    def impedance(currents: Any) -> Any:
                        flux = _flux(currents.real).astype(cp.complex128)
                        flux += 1j * _flux(currents.imag)
                        return (
                            resistance_device * currents
                            + 1j * omega * flux
                        )

                    def saddle(vector: Any) -> Any:
                        currents = vector[:branches]
                        voltages = vector[branches:]
                        return cp.concatenate(
                            [
                                impedance(currents) - reduced @ voltages,
                                reduced.T @ currents,
                            ]
                        )

                    full_diagonal_cpu = resistance + 1j * omega * np.concatenate(
                        [
                            np.full(
                                inline_count,
                                _self_inductance(operator),
                            ),
                            _vertical_self_inductance(mesh, operator),
                        ]
                    )
                    diagonal = cp.asarray(
                        full_diagonal_cpu[active_branches]
                    )
                    schur = (
                        reduced.T @ csp.diags(1.0 / diagonal) @ reduced
                    ).tocsc()
                    factored = csl.splu(schur)

                    def precondition(vector: Any) -> Any:
                        rhs_current = vector[:branches]
                        rhs_node = vector[branches:]
                        node = factored.solve(
                            rhs_node
                            + reduced.T @ (rhs_current / diagonal)
                        )
                        current = (
                            rhs_current + reduced @ node
                        ) / diagonal
                        return cp.concatenate([current, node])

                    right_hand_side = cp.concatenate(
                        [
                            cp.zeros(branches, dtype=cp.complex128),
                            injected_device[keep],
                        ]
                    )
                    # CuPy's ``M=`` path implements a right-preconditioned
                    # variable, unlike SciPy's left-preconditioned GMRES used
                    # by the CPU reference.  Build M^-1 A and M^-1 b
                    # explicitly so both paths solve the same Krylov system
                    # and the reference-node closure does not accumulate the
                    # residual left at every other node.
                    def left_preconditioned_saddle(vector: Any) -> Any:
                        return precondition(saddle(vector))

                    linear_system = csl.LinearOperator(
                        (size, size),
                        matvec=left_preconditioned_saddle,
                        dtype=cp.complex128,
                    )
                    left_right_hand_side = precondition(right_hand_side)
                    counter = {"restarts": 0}

                    def count(_value: Any) -> None:
                        counter["restarts"] += 1

                    # The norm of the explicitly left-preconditioned residual
                    # is not the norm reported in the public contract.  A
                    # tighter internal target keeps the original saddle-point
                    # residual at or below the requested tolerance.
                    krylov_tolerance = max(
                        tolerance * 1e-3,
                        10.0 * np.finfo(np.float64).eps,
                    )
                    effective_restart = min(restart, size)
                    # SciPy interprets maxiter as restart cycles for the CPU
                    # call above, while CuPy interprets it as total inner
                    # iterations.  Preserve one public setting meaning.
                    maximum_inner_iterations = (
                        max_iterations * effective_restart
                    )
                    result, info = csl.gmres(
                        linear_system,
                        left_right_hand_side,
                        rtol=krylov_tolerance,
                        restart=restart,
                        maxiter=maximum_inner_iterations,
                        callback=count,
                        callback_type="pr_norm",
                    )
                    current_device = cp.zeros(
                        mesh.branch_count, dtype=cp.complex128
                    )
                    current_device[active_branches_device] = result[:branches]
                    voltage_device = cp.zeros(
                        mesh.node_count, dtype=cp.complex128
                    )
                    voltage_device[keep] = result[branches:]
                    residual = float(
                        (
                            cp.linalg.norm(
                                saddle(result) - right_hand_side
                            )
                            / cp.maximum(
                                cp.linalg.norm(right_hand_side),
                                cp.asarray(1e-30),
                            )
                        ).get()
                    )
                    iterations = (
                        counter["restarts"] * effective_restart
                    )
                    converged = info == 0 and residual <= tolerance
                    factor_backend = (
                        "scipy_superlu_factor_cupy_triangular_solve"
                    )

                cp.cuda.get_current_stream().synchronize()
                solve_ms = (time.perf_counter() - solve_started) * 1000.0
                voltage = cp.asnumpy(voltage_device)
                current = cp.asnumpy(current_device).astype(
                    np.complex128, copy=False
                )
                pool_peak = int(pool.total_bytes())
                free_after, _ = cp.cuda.runtime.memGetInfo()
    except CudaSheetUnavailableError:
        raise
    except Exception as error:
        # Name the class.  A CuPy signature change arrives here as a
        # ``TypeError`` and reads as a numerical failure without it.
        raise CudaSheetSolveError(
            f"CUDA sheet-PEEC execution failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    finally:
        if device is not None and pool is not None:
            with device:
                pool.free_all_blocks()

    total_ms = (time.perf_counter() - total_started) * 1000.0
    telemetry = CudaSheetTelemetry(
        resolved_backend=f"cupy-sheet-peec:{device_id}",
        device_id=device_id,
        device_name=identity["device_name"],
        compute_capability=identity["compute_capability"],
        driver_version=identity["driver_version"],
        runtime_version=identity["runtime_version"],
        cupy_version=identity["cupy_version"],
        prepare_ms=prepare_ms,
        solve_ms=solve_ms,
        total_ms=total_ms,
        krylov_relative_tolerance=(
            tolerance if frequency_hz == 0.0 else krylov_tolerance
        ),
        memory_pool_peak_bytes=pool_peak,
        free_memory_before_bytes=int(free_before),
        free_memory_after_bytes=int(free_after),
        preconditioner_factorization_backend=factor_backend,
    )
    return (
        SheetSolution(
            node_voltage=voltage,
            branch_current=current,
            frequency_hz=frequency_hz,
            iterations=iterations,
            residual=residual,
            converged=converged,
            grounded_node=grounded,
            undriven_nodes=dropped,
        ),
        telemetry,
    )
