"""CUDA execution for the layered sheet-PEEC system.

The physical discretisation is shared with :mod:`electrical.sheet_peec.sheet_peec`.
Only the prepared convolution spectra, sparse incidence products, Krylov
vectors, and preconditioner solves move to CuPy.  There is deliberately no CPU
fallback in this module: a requested CUDA solve either reports a CUDA backend
or raises.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math
import time
import weakref
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp

from .sheet_operator import SheetInductanceOperator
from .sheet_peec import (
    _conductor_of_layer,
    SheetMesh,
    SheetSolution,
    Terminal,
    _inline_self_inductance,
    _near_impedance,
    _self_inductance,
    _reduce,
    _source_vector,
    _terminal_nodes,
    _vertical_self_inductance,
    _BarrelSelf,
    _with_barrels,
)


class CudaSheetUnavailableError(RuntimeError):
    """Raised when the requested CUDA sheet backend cannot initialize."""


# The Krylov call below passes ``rtol``, which CuPy renamed from ``tol`` in 14.
# The distribution requirement already states this, but an environment can
# satisfy the import and not the requirement, and the mismatch then surfaces
# from inside the solve as a bare ``TypeError``.
MINIMUM_CUPY_MAJOR_VERSION = 14


# Device objects kept between calls: an operator's spectra and the cuDSS
# plans of near matrices.  A sweep solves one conductor at many frequencies
# and a hot loop the same conductor for many capacitors; neither should move
# the operator again or redo the host-side ordering of a matrix whose
# pattern it has already planned.  Each entry owns the memory pool it was
# allocated from, so the per-call pools can be emptied freely.
CUDA_CACHE_ENTRIES = 2
_OPERATOR_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_CUDSS_CACHE: "OrderedDict[tuple, _CudssEntry]" = OrderedDict()


def clear_cuda_sheet_caches() -> None:
    """Drop the device operators and cuDSS plans kept between calls."""
    _OPERATOR_CACHE.clear()
    while _CUDSS_CACHE:
        _key, entry = _CUDSS_CACHE.popitem(last=False)
        entry.solver.free()


def _cached_device_operator(operator: Any, cp: Any, device_id: int) -> Any:
    """The device form of a host operator, made once while the host operator lives."""
    key = (device_id, id(operator))
    entry = _OPERATOR_CACHE.get(key)
    if entry is not None and entry[0]() is operator:
        _OPERATOR_CACHE.move_to_end(key)
        return entry[2]
    pool = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(pool.malloc):
        device_operator = cuda_inductance_operator(operator, cp)
    try:
        reference = weakref.ref(operator)
    except TypeError:  # an operator that cannot be weakly referenced is not cached
        return device_operator
    _OPERATOR_CACHE[key] = (reference, pool, device_operator)
    while len(_OPERATOR_CACHE) > CUDA_CACHE_ENTRIES:
        _OPERATOR_CACHE.popitem(last=False)
    return device_operator


@dataclass
class _CudssEntry:
    pool: Any
    solver: Any
    rhs: Any
    matrix: Any  # the device CSR the solver was planned on; its values are overwritten in place


# How the ``near`` preconditioner is factored and applied on the device.
NEAR_FACTORIZATIONS = ("superlu", "cudss_quasi_definite")


class NearMatrixTooLarge(RuntimeError):
    """The near matrix would not fit beside the solve on the device; it is formed on the host."""


# Entries of the near matrix formed at once on the device, a block of source
# branches at a time: the block's keys, order and values are the only scratch.
NEAR_BLOCK_ENTRIES = 1 << 24


def device_quasi_definite_near(
    mesh: SheetMesh,
    operator: Any,
    omega: float,
    resistance: np.ndarray,
    active_branches: np.ndarray,
    reduced: Any,
    regularization: float,
    xp: Any,
    xsparse: Any,
    radius_cells: int = 1,
    *,
    block_entries: int = NEAR_BLOCK_ENTRIES,
    memory_budget: int | None = None,
) -> tuple[Any, Any, float]:
    """:func:`quasi_definite_near_matrix` of :func:`_near_impedance`, built where ``xp`` lives.

    The near matrix of the CUDA solve was assembled on the host: the near
    inductance read off the convolution tables, filtered to one copper layer
    (a cut layer's filaments together), cut to the active branches, made
    ``R + j omega L`` and stacked with the incidence.  On a 0.2 mm hot loop
    with its planes cut into filaments that took 9.4 s on a Colab A100 host.
    Here only the few table values the near radius reaches are read on the
    host; the matrix is formed by ``xp`` (CuPy, or NumPy standing in) and
    ``xsparse`` (``cupyx.scipy.sparse`` or ``scipy.sparse``).  The result is
    the host path's matrix: the same pattern and the same values, entry for
    entry.

    It is written straight into its CSR arrays (32-bit indices) in two
    passes over the near-radius slots: the first counts each row's entries,
    the second forms a block of source branches' entries at a time, orders
    them by column within the block and writes them in place.  The scratch is
    one block (``block_entries``), not the whole matrix: assembling it as COO
    and converting took 6.7 GB beside the 9.2 GB solve of that hot loop.
    When the counted matrix and a block exceed ``memory_budget`` bytes,
    :class:`NearMatrixTooLarge` is raised before anything is allocated, and
    the caller forms it on the host instead.

    Only for the uniform convolution operator (``_tables``); returns the
    CSR matrix, the row signs and ``delta``.
    """
    base = getattr(operator, "_base", operator)
    barrel_delta = getattr(operator, "_delta_branch", None)
    if not hasattr(base, "_tables"):
        raise TypeError("the device near matrix needs the uniform convolution operator")
    rows, cols = mesh.shape
    padded_rows, padded_cols = base.padded
    span = list(range(-radius_cells, radius_cells + 1))
    conductors = [_conductor_of_layer(layer.name) for layer in mesh.stackup.layers]
    conductor_ids = {name: k for k, name in enumerate(dict.fromkeys(conductors))}
    sheet = np.asarray([conductor_ids[name] for name in conductors], dtype=np.int64)
    count = mesh.branch_count
    position = np.full(count, -1, dtype=np.int64)
    active_index = np.flatnonzero(active_branches)
    position[active_index] = np.arange(len(active_index))
    branches = len(active_index)
    incidence = sp.csr_matrix(reduced)
    incidence.sort_indices()
    nodes = incidence.shape[1]
    size = branches + nodes

    # The branch groups as the host path walks them: (sources kept, their
    # branch numbers, lookup, per-slot values, the levels each may reach).
    groups = []

    def group(sources: np.ndarray, first: int, levels_count: int, table_of: Any, group_of: np.ndarray | None) -> None:
        lookup = np.full((levels_count, rows, cols), -1, dtype=np.int64)
        lookup[sources[:, 0], sources[:, 1], sources[:, 2]] = np.arange(first, first + len(sources))
        # a branch repeated at one place couples once, as its last copy
        keep = np.flatnonzero(lookup[sources[:, 0], sources[:, 1], sources[:, 2]] == np.arange(first, first + len(sources)))
        sources, number = sources[keep], first + keep
        live = position[number] >= 0
        sources, number = sources[live], number[live]
        active_lookup = np.where(lookup >= 0, position[np.maximum(lookup, 0)], -1)
        slots = []
        for other in range(levels_count):
            reaches = (
                np.ones(levels_count, dtype=bool) if group_of is None else group_of[:levels_count] == group_of[other]
            )
            if not reaches.any():
                continue
            for dr in span:
                for dc in span:
                    values = np.array(
                        [table_of(lvl, other)[dr % padded_rows, dc % padded_cols] for lvl in range(levels_count)]
                    )
                    slots.append((other, dr, dc, reaches, values))
        groups.append((sources, number, active_lookup, slots))

    offset = 0
    for axis, branches_of_axis in (("x", mesh.branch_x), ("y", mesh.branch_y)):
        if branches_of_axis:
            group(
                np.asarray(branches_of_axis, dtype=np.int64).reshape(-1, 3), offset, len(mesh.stackup),
                lambda layer, other, axis=axis: base._tables[(axis, min(layer, other), max(layer, other))], sheet,
            )
        offset += len(branches_of_axis)
    levels = list(base.vertical_levels)
    if mesh.via_branches and levels:
        index_of = {key: k for k, key in enumerate(levels)}
        vias = np.array(
            [(index_of[(via.lower_layer, via.upper_layer)], via.row, via.col) for via in mesh.via_branches],
            dtype=np.int64,
        )
        group(vias, offset, len(levels), lambda a, b: base._tables_z[(min(a, b), max(a, b))], None)

    def blocks(sources: np.ndarray, slot_count: int):
        step = max(1, block_entries // max(1, slot_count))
        for start in range(0, len(sources), step):
            yield start, min(len(sources), start + step)

    def slot_hits(device_lookup: Any, level: Any, row0: Any, col0: Any, slot: tuple) -> tuple[Any, Any]:
        other, dr, dc, reaches, _values = slot
        row, col = row0 + dr, col0 + dc
        inside = (row >= 0) & (row < rows) & (col >= 0) & (col < cols) & xp.asarray(reaches)[level]
        target = xp.where(inside, device_lookup[other, xp.clip(row, 0, rows - 1), xp.clip(col, 0, cols - 1)], -1)
        return target >= 0, target

    # pass 1: each row's near entries
    near_count = xp.zeros(branches, dtype=xp.int64)
    device_groups = []
    for sources, number, active_lookup, slots in groups:
        device_lookup = xp.asarray(active_lookup)
        level_all = xp.asarray(sources[:, 0])
        row_all, col_all = xp.asarray(sources[:, 1]), xp.asarray(sources[:, 2])
        source_row = xp.asarray(position[number])
        device_groups.append((device_lookup, level_all, row_all, col_all, source_row, slots, sources))
        for start, stop in blocks(sources, len(slots)):
            hits = xp.zeros(stop - start, dtype=xp.int64)
            for slot in slots:
                hit, _ = slot_hits(device_lookup, level_all[start:stop], row_all[start:stop], col_all[start:stop], slot)
                hits += hit
            near_count[source_row[start:stop]] += hits
    # an active branch with no near entries (a repeated copy) still has R on its diagonal
    lone = near_count == 0
    incidence_count = xp.asarray(np.diff(incidence.indptr).astype(np.int64))
    transpose = sp.csr_matrix(incidence.T)
    transpose.sort_indices()
    branch_count = near_count + lone + incidence_count
    node_count = xp.asarray(np.diff(transpose.indptr).astype(np.int64)) + 1
    indptr = xp.concatenate([xp.zeros(1, dtype=xp.int64), xp.cumsum(xp.concatenate([branch_count, node_count]))])
    nnz = int(indptr[-1])
    if nnz >= 2**31 or size >= 2**31:
        raise NearMatrixTooLarge("the near matrix needs 64-bit indices")
    if memory_budget is not None:
        needed = nnz * (4 + 16) + (size + 1) * 4 + block_entries * (8 + 8 + 8 + 16 + 4)
        if needed > memory_budget:
            raise NearMatrixTooLarge(f"the near matrix needs {needed / 1e9:.1f} GB, {memory_budget / 1e9:.1f} GB free")
    indices = xp.empty(nnz, dtype=xp.int32)
    data = xp.empty(nnz, dtype=xp.complex128)
    z_diagonal = xp.zeros(branches, dtype=xp.complex128)
    resistance_active = xp.asarray(np.asarray(resistance, dtype=np.float64)[active_branches])
    barrel = None
    if barrel_delta is not None:
        barrel = xp.zeros(count, dtype=xp.float64)
        barrel[len(mesh.branch_x) + len(mesh.branch_y):] = xp.asarray(barrel_delta)
        barrel = barrel[xp.asarray(active_index)]

    # pass 2: a block of sources' entries at a time, ordered by column within each row
    for device_lookup, level_all, row_all, col_all, source_row, slots, sources in device_groups:
        for start, stop in blocks(sources, len(slots)):
            level, row0, col0 = level_all[start:stop], row_all[start:stop], col_all[start:stop]
            parts_row, parts_col, parts_value = [], [], []
            for slot in slots:
                hit, target = slot_hits(device_lookup, level, row0, col0, slot)
                parts_row.append(source_row[start:stop][hit])
                parts_col.append(target[hit])
                parts_value.append(xp.asarray(slot[4])[level[hit]])
            entry_row = xp.concatenate(parts_row)
            entry_col = xp.concatenate(parts_col)
            inductance = xp.concatenate(parts_value).astype(xp.float64)
            del parts_row, parts_col, parts_value
            diagonal = entry_row == entry_col
            if barrel is not None:
                inductance = inductance + xp.where(diagonal, barrel[entry_row], 0.0)
            value = (1j * float(omega)) * inductance + xp.where(diagonal, resistance_active[entry_row], 0.0)
            order = xp.argsort(entry_row * size + entry_col)
            entry_row, entry_col, value, diagonal = entry_row[order], entry_col[order], value[order], diagonal[order]
            first = xp.searchsorted(entry_row, entry_row, side="left")
            destination = indptr[entry_row] + (xp.arange(entry_row.size) - first)
            indices[destination] = entry_col.astype(xp.int32)
            data[destination] = value
            z_diagonal[entry_row[diagonal]] = value[diagonal]
            del entry_row, entry_col, value, order, first, destination, inductance, diagonal
    lone_rows = xp.flatnonzero(lone)
    indices[indptr[lone_rows]] = lone_rows.astype(xp.int32)
    data[indptr[lone_rows]] = resistance_active[lone_rows].astype(xp.complex128)
    z_diagonal[lone_rows] = resistance_active[lone_rows]
    # the incidence after each branch row's near entries; its transpose in the node rows
    inc_rows = np.repeat(np.arange(branches), np.diff(incidence.indptr))
    inc_rank = np.arange(incidence.nnz) - incidence.indptr[inc_rows]
    inc_rows_device = xp.asarray(inc_rows)
    destination = indptr[inc_rows_device] + near_count[inc_rows_device] + lone[inc_rows_device] + xp.asarray(inc_rank)
    indices[destination] = xp.asarray((branches + incidence.indices).astype(np.int32))
    data[destination] = xp.asarray(-incidence.data.astype(np.complex128))
    t_rows = np.repeat(np.arange(nodes), np.diff(transpose.indptr))
    t_rank = np.arange(transpose.nnz) - transpose.indptr[t_rows]
    destination = indptr[branches + xp.asarray(t_rows)] + xp.asarray(t_rank)
    indices[destination] = xp.asarray(transpose.indices.astype(np.int32))
    data[destination] = xp.asarray(-transpose.data.astype(np.complex128))
    magnitude = xp.abs(z_diagonal)
    positive = magnitude[magnitude > 0.0]
    scale = float(xp.median(positive)) if positive.size else 1.0
    delta = float(regularization) / scale
    node_diagonal = indptr[branches + 1:] - 1
    indices[node_diagonal] = xp.arange(branches, size, dtype=xp.int32)
    data[node_diagonal] = -delta
    matrix = xsparse.csr_matrix((data, indices, indptr.astype(xp.int32)), shape=(size, size))
    matrix.has_canonical_format = True
    sign = xp.concatenate([xp.ones(branches), -xp.ones(nodes)]).astype(xp.complex128)
    return matrix, sign, delta


def _pattern_digest(matrix: Any) -> str:
    """A digest of a CSR matrix's sparsity pattern (row pointers and column indices)."""
    digest = hashlib.blake2b(digest_size=16)
    for array in (matrix.indptr, matrix.indices):
        host = array.get() if hasattr(array, "get") else array
        digest.update(np.ascontiguousarray(host).tobytes())
    return digest.hexdigest()


def quasi_definite_near_matrix(near: Any, reduced: Any, regularization: float) -> tuple[Any, np.ndarray, float]:
    """``(matrix, sign, delta)``: the near saddle-point matrix made quasi-definite.

    ``matrix`` is ``[[Z, -A], [-A^T, -delta I]]`` (SciPy CSR, complex128,
    sorted), ``sign`` the row signs that take the saddle-point right-hand side
    to it (``+1`` on branches, ``-1`` on nodes), ``delta`` the regularization
    over the median ``|Z_ii|``.  See :class:`CudssQuasiDefiniteNear`.
    """
    import scipy.sparse as sp

    near = sp.csr_matrix(near)
    diagonal = np.abs(near.diagonal())
    scale = float(np.median(diagonal[diagonal > 0.0])) if np.any(diagonal > 0.0) else 1.0
    delta = float(regularization) / scale
    nodes = reduced.shape[1]
    matrix = sp.bmat(
        [[near, -reduced], [-reduced.T, -delta * sp.identity(nodes, format="csr")]], format="csr"
    ).astype(np.complex128)
    matrix.sum_duplicates()
    matrix.sort_indices()
    sign = np.concatenate([np.ones(near.shape[0]), -np.ones(nodes)]).astype(np.complex128)
    return matrix, sign, delta


class CudssQuasiDefiniteNear:
    """The near-field saddle-point preconditioner as a quasi-definite cuDSS LDL^T.

    The near matrix is ``[[Z, -A], [A^T, 0]]``: ``Z`` the near branch
    impedances (complex symmetric, its real part the branch resistances),
    ``A`` the reduced incidence.  Its zero block defeats a factorization that
    does not pivot across the whole matrix: cuDSS's default static pivoting
    gave residuals above one, and the general COLAMD ordering that does pivot
    is 70 times slower per solve.  Negating the node rows and putting
    ``-delta I`` in the zero block gives

        [[Z, -A], [-A^T, -delta I]],

    complex symmetric and quasi-definite (``Re Z`` positive, the node block
    negative), which factors stably as LDL^T in any symmetric ordering
    without pivoting, so cuDSS's nested dissection and parallel triangular
    solves apply.  ``delta`` is ``regularization`` over the median ``|Z_ii|``;
    at 1e-8 the factors solve the unregularized near matrix to about 1e-6,
    which is a preconditioner, not the solution, so the Krylov iteration
    count is unchanged.

    ``solve(v)`` returns ``M^-1 v`` for the unregularized sign convention.
    """

    def __init__(
        self, near: Any, reduced: Any, regularization: float, cp: Any, prepared: tuple[Any, Any, float] | None = None,
        columns: int = 1,
    ) -> None:
        try:
            import nvmath.bindings.cudss as cudss_bindings
            import nvmath.sparse.advanced as nsa
        except ImportError as error:
            raise CudaSheetUnavailableError(
                "near_factorization='cudss_quasi_definite' needs nvmath-python with cuDSS"
            ) from error
        import cupyx.scipy.sparse as csp

        # Double precision: complex64 factors were measured on a Colab A100
        # (0.4 mm filament hot loop, 30 MHz) and did not precondition at all --
        # GMRES ran to its cap at residuals 1.0 (delta 1e-8, under single
        # precision's resolution), 0.036 (1e-4) and 0.016 (1e-3).
        # ``prepared``: the matrix already formed on the device by
        # device_quasi_definite_near, in place of the host's near matrix
        matrix, sign, delta = (
            prepared if prepared is not None else quasi_definite_near_matrix(near, reduced, regularization)
        )
        self._cp = cp
        self._sign = cp.asarray(sign)
        self.regularization = float(regularization)
        self.delta = delta
        # The pattern of the near matrix depends on the mesh and its cut, not
        # on the frequency: a matrix whose pattern was planned before only
        # has its values replaced and is factored again.
        # the right-hand side's shape is fixed with the plan: one column, or a
        # column per excitation solved in lockstep (solve_many)
        self.columns = int(columns)
        key = (int(cp.cuda.Device().id), matrix.shape, _pattern_digest(matrix), self.columns)
        entry = _CUDSS_CACHE.get(key)
        self.reused_plan = entry is not None
        if entry is not None:
            _CUDSS_CACHE.move_to_end(key)
            # nvmath drops the plan when an operand is reset, so the values
            # are written into the planned matrix itself and only the
            # numerical factorization is repeated.
            stream = cp.cuda.get_current_stream()
            with cp.cuda.using_allocator(entry.pool.malloc):
                entry.matrix.data[...] = cp.asarray(matrix.data)
                stream.synchronize()
                entry.solver.factorize(stream=stream)
                stream.synchronize()
            self._solver, self._rhs, self._pool = entry.solver, entry.rhs, entry.pool
            return
        pool = cp.cuda.MemoryPool()
        with cp.cuda.using_allocator(pool.malloc):
            self._rhs = (
                cp.zeros(matrix.shape[0], dtype=cp.complex128) if self.columns == 1
                else cp.zeros((matrix.shape[0], self.columns), dtype=cp.complex128, order="F")
            )
            # copied into the entry's own pool: a matrix formed on the device
            # lives in the calling solve's pool, which is emptied when it ends
            device_matrix = csp.csr_matrix(
                (cp.array(matrix.data, copy=True), cp.array(matrix.indices, copy=True),
                 cp.array(matrix.indptr, copy=True)),
                shape=matrix.shape,
            )
            self._solver = self._planned(device_matrix, nsa, cudss_bindings)
        self._pool = pool
        _CUDSS_CACHE[key] = _CudssEntry(pool, self._solver, self._rhs, device_matrix)
        while len(_CUDSS_CACHE) > CUDA_CACHE_ENTRIES:
            _old, evicted = _CUDSS_CACHE.popitem(last=False)
            evicted.solver.free()

    def _planned(self, device_matrix: Any, nsa: Any, cudss_bindings: Any) -> Any:
        options = {
            "sparse_system_type": nsa.DirectSolverMatrixType.SYMMETRIC,
            "sparse_system_view": nsa.DirectSolverMatrixViewType.FULL,
        }
        threading = _cudss_threading_library()
        if threading is not None:
            # cuDSS orders and plans on the host; single-threaded that took
            # 5.6 s of a 0.4 mm filament hot loop on a Colab host
            options["multithreading_lib"] = threading
        solver = nsa.DirectSolver(device_matrix, self._rhs, execution="cuda", options=options)
        solver.plan_config.pivot_type = cudss_bindings.PivotType.PIVOT_NONE
        solver.plan()
        solver.factorize()
        return solver

    def solve(self, vector: Any) -> Any:
        stream = self._cp.cuda.get_current_stream()
        self._rhs[...] = vector * self._sign
        stream.synchronize()
        # cuDSS's solve workspace comes from the entry's pool too: drawn from
        # a call's own pool, it was freed under the cached plan at the end of
        # that call and the next frequency's GMRES read NaNs.
        with self._cp.cuda.using_allocator(self._pool.malloc):
            solution = self._solver.solve(stream=stream)
        stream.synchronize()
        return solution.copy()

    def solve_many(self, vectors: Any) -> Any:
        """``M^-1`` of each row of ``vectors`` (k, n), k up to ``columns``, in one solve.

        The factors are read once for all the columns: the bytes a
        triangular solve moves are the factors', whatever the column count.
        """
        cp = self._cp
        count = vectors.shape[0]
        if count > self.columns:
            raise ValueError(f"{count} right-hand sides for factors planned with {self.columns}")
        stream = cp.cuda.get_current_stream()
        self._rhs[:, :count] = (vectors * self._sign[None, :]).T
        if count < self.columns:
            self._rhs[:, count:] = 0.0
        stream.synchronize()
        with cp.cuda.using_allocator(self._pool.malloc):
            solution = self._solver.solve(stream=stream)
        stream.synchronize()
        return cp.ascontiguousarray(solution[:, :count].T)

    def free(self) -> None:
        """Nothing to release: the plan and its factors stay cached for the next call."""
        self._solver = None


def _cudss_threading_library() -> str | None:
    """cuDSS's OpenMP threading layer beside its library, or None when it is not installed."""
    import glob
    import os

    stated = os.environ.get("CUDSS_THREADING_LIB")
    if stated:
        return stated
    try:
        import nvidia.cu13 as cu13_package  # nvidia-cudss-cu13 ships it here
    except ImportError:
        try:
            import nvidia.cudss as cu13_package  # older wheel layout
        except ImportError:
            return None
    for root in getattr(cu13_package, "__path__", []):
        found = sorted(glob.glob(os.path.join(root, "**", "libcudss_mtlayer_gomp.so*"), recursive=True))
        if found:
            return found[0]
    return None


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
        # The host operator keeps spectra only on its NumPy path; with the C++
        # core built they live in the core.  Both keep the real-space tables,
        # whose spectra are the ones the NumPy path would have.
        kernels = source._kernels or {key: np.fft.rfft2(table) for key, table in source._tables.items()}
        kernels_z = source._kernels_z or {key: np.fft.rfft2(table) for key, table in source._tables_z.items()}
        self._kernels = {key: cp.asarray(value) for key, value in kernels.items()}
        self._kernels_z = {key: cp.asarray(value) for key, value in kernels_z.items()}
        # The layer-pair products as one batched matrix product per axis: at
        # every frequency point the coupling of the layers is an L x L matrix
        # and the layers' spectra a vector, so the whole convolution is one
        # (F, L, L) @ (F, L, B) product.  The loop over L^2 pairs it replaces
        # launched a kernel per pair: 841 a product per axis with the two
        # 71 um planes of a 2-layer board cut into filaments, which left the
        # device waiting on launches.  The stack reads every pair's spectrum
        # once per application whatever the batch, so it is bound by memory
        # bandwidth.  It is built only where it fits beside the solve.
        self._stack = self._stacked(kernels, kernels_z, cp)

    def _stacked(self, kernels: dict, kernels_z: dict, cp: Any) -> dict[str, Any] | None:
        layers, levels = self.layer_count, len(self.vertical_levels)
        if not kernels:
            return None
        sample = next(iter(kernels.values()))
        frequencies = int(np.prod(sample.shape))
        # On a device the pairs are kept once, the upper triangle packed per
        # frequency (the coupling is symmetric in the layers); NumPy standing
        # in for CuPy gets the full (F, L, L) stack its matmul path reads.
        packed = _layer_pair_mix_kernel(cp) is not None
        pairs = (lambda n: n * (n + 1) // 2) if packed else (lambda n: n * n)
        needed = 16 * frequencies * (2 * pairs(layers) + pairs(levels))
        budget = _device_memory_budget(cp)
        if budget is not None and needed > budget:
            return None
        self.packed = packed

        def stack(pick: Any, count: int) -> Any:
            if packed:
                out = cp.empty((frequencies, count * (count + 1) // 2), dtype=cp.complex128)
                position = 0
                for first in range(count):
                    for second in range(first, count):
                        out[:, position] = cp.asarray(pick(first, second)).reshape(-1)
                        position += 1
                return out
            out = cp.empty((frequencies, count, count), dtype=cp.complex128)
            for first in range(count):
                for second in range(count):
                    out[:, first, second] = cp.asarray(pick(first, second)).reshape(-1)
            return out

        stacked = {
            "x": stack(lambda a, b: kernels[("x", min(a, b), max(a, b))], layers),
            "y": stack(lambda a, b: kernels[("y", min(a, b), max(a, b))], layers),
        }
        if levels:
            stacked["z"] = stack(lambda a, b: kernels_z[(min(a, b), max(a, b))], levels)
        self.spectrum_shape = tuple(sample.shape)
        if packed:
            # the pairwise spectra are in the stack now; apply goes through it too
            self._kernels, self._kernels_z = {}, {}
        return stacked

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
        ) + (0 if self._stack is None else sum(int(value.nbytes) for value in self._stack.values()))

    def _convolve_batch(self, currents: Any, matrix: Any) -> Any:
        """``currents`` ``(B, L, rows, cols)`` convolved by the stacked pairs, one product for all."""
        cp = self.cp
        rows, cols = self.shape
        batch, count = currents.shape[0], currents.shape[1]
        spectra = cp.fft.rfft2(currents, s=self.padded, axes=(-2, -1))
        frequencies = spectra.shape[-2] * spectra.shape[-1]
        flat = spectra.reshape(batch, count, frequencies)
        if getattr(self, "packed", False):
            mixed = cp.empty_like(flat)
            _mix_packed(cp, matrix, cp.ascontiguousarray(flat), mixed, frequencies, count, batch)
            accumulated = mixed.reshape(spectra.shape)
        else:
            # (B, L, F) -> (F, L, B): each frequency's layers a column per batch member
            columns = cp.ascontiguousarray(flat.transpose(2, 1, 0))
            mixed = cp.matmul(matrix, columns)
            accumulated = mixed.transpose(2, 1, 0).reshape(spectra.shape)
        product = cp.fft.irfft2(accumulated, s=self.padded, axes=(-2, -1))
        return cp.ascontiguousarray(product[..., :rows, :cols])

    def apply_batch(self, currents_x: Any, currents_y: Any, currents_z: Any | None = None) -> Any:
        """:meth:`apply` for a batch: every array carries a leading batch axis."""
        if self._stack is None:
            outputs = [
                self.apply(currents_x[b], currents_y[b], None if currents_z is None else currents_z[b])
                for b in range(currents_x.shape[0])
            ]
            return tuple(self.cp.stack(parts) for parts in zip(*outputs))
        flux_x = self._convolve_batch(currents_x, self._stack["x"])
        flux_y = self._convolve_batch(currents_y, self._stack["y"])
        if currents_z is None:
            return flux_x, flux_y
        if not self.vertical_levels:
            return flux_x, flux_y, self.cp.zeros_like(currents_z)
        return flux_x, flux_y, self._convolve_batch(currents_z, self._stack["z"])

    def _apply_via_stack(self, currents_x: Any, currents_y: Any, currents_z: Any | None) -> Any:
        flux = self.apply_batch(
            currents_x[None], currents_y[None], None if currents_z is None else currents_z[None]
        )
        return tuple(part[0] for part in flux)

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
        if getattr(self, "packed", False):
            return self._apply_via_stack(currents_x, currents_y, currents_z)
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


class CudaPfftSheetInductanceOperator:
    """Device-resident projection, spectra and precorrection of the pFFT operator."""

    def __init__(self, source: Any, cp: Any) -> None:
        import cupyx.scipy.sparse as csp

        self.cp = cp
        self.source = source
        self.shape = source.shape
        self.layer_count = len(source.stackup)
        self.vertical_levels = source.vertical_levels
        self._projection = {axis: csp.csr_matrix(matrix) for axis, matrix in source._projection.items()}
        self._projection_t = {axis: csp.csr_matrix(matrix.T.tocsr()) for axis, matrix in source._projection.items()}
        self._correction = {axis: csp.csr_matrix(matrix) for axis, matrix in source._correction.items()}
        self._length = {axis: cp.asarray(source._geometry[axis][2].reshape(-1)) for axis in ("x", "y")}
        self._spectra = {key: cp.asarray(value) for key, value in source._spectra.items()}
        self._spectra_z = {key: cp.asarray(value) for key, value in source._spectra_z.items()}
        self._projection_z = csp.csr_matrix(source._projection_z) if source._projection_z is not None else None
        self._projection_z_t = (
            csp.csr_matrix(source._projection_z.T.tocsr()) if source._projection_z is not None else None
        )
        self._correction_z = csp.csr_matrix(source._correction_z) if source._correction_z is not None else None

    @property
    def kernel_bytes(self) -> int:
        return int(self.source.kernel_bytes)

    def _apply_axis(self, currents: Any, axis: str) -> Any:
        cp = self.cp
        grid = self.source.grid
        grid_shape = (grid.nodes_y, grid.nodes_x)
        length = self._length[axis]
        projection, projection_t = self._projection[axis], self._projection_t[axis]
        flat = currents.reshape(self.layer_count, -1)
        spectra = [
            cp.fft.rfft2((projection_t @ (flat[l] * length)).reshape(grid_shape), s=grid.padded)
            for l in range(self.layer_count)
        ]
        out = cp.zeros_like(flat)
        for target in range(self.layer_count):
            accumulated = cp.zeros_like(spectra[0])
            for source in range(self.layer_count):
                accumulated += self._spectra[self.source._separation[(target, source)]] * spectra[source]
            potential = cp.fft.irfft2(accumulated, s=grid.padded)[: grid_shape[0], : grid_shape[1]]
            out[target] = length * (projection @ potential.reshape(-1))
        return (out.reshape(-1) + self._correction[axis] @ flat.reshape(-1)).reshape(currents.shape)

    def apply_batch(self, currents_x: Any, currents_y: Any, currents_z: Any | None = None) -> Any:
        """:meth:`apply` over a leading batch axis, one member at a time."""
        outputs = [
            self.apply(currents_x[b], currents_y[b], None if currents_z is None else currents_z[b])
            for b in range(currents_x.shape[0])
        ]
        return tuple(self.cp.stack(parts) for parts in zip(*outputs))

    def apply(self, currents_x: Any, currents_y: Any, currents_z: Any | None = None) -> Any:
        cp = self.cp
        rows, cols = self.shape
        flux_x = cp.zeros_like(currents_x)
        flux_y = cp.zeros_like(currents_y)
        flux_x[:, :, :-1] = self._apply_axis(cp.ascontiguousarray(currents_x[:, :, :-1]), "x")
        flux_y[:, :-1, :] = self._apply_axis(cp.ascontiguousarray(currents_y[:, :-1, :]), "y")
        if currents_z is None:
            return flux_x, flux_y
        levels = len(self.vertical_levels)
        if not levels:
            return flux_x, flux_y, cp.zeros_like(currents_z)
        grid = self.source.grid
        grid_shape = (grid.nodes_y, grid.nodes_x)
        spans = [span for span, _ in self.source._vertical_geometry]
        flat = currents_z.reshape(levels, -1)
        spectra = [
            cp.fft.rfft2((self._projection_z_t @ (flat[l] * spans[l])).reshape(grid_shape), s=grid.padded)
            for l in range(levels)
        ]
        out = cp.zeros_like(flat)
        for target in range(levels):
            accumulated = cp.zeros_like(spectra[0])
            for source in range(levels):
                accumulated += self._spectra_z[self.source._vertical_key(target, source)] * spectra[source]
            potential = cp.fft.irfft2(accumulated, s=grid.padded)[: grid_shape[0], : grid_shape[1]]
            out[target] = spans[target] * (self._projection_z @ potential.reshape(-1))
        flux_z = out.reshape(-1) + self._correction_z @ flat.reshape(-1)
        return flux_x, flux_y, flux_z.reshape(currents_z.shape)


class _CudaBarrelSelf:
    """A device operator whose barrel vias carry ``L_barrel - L_column`` on their own diagonal."""

    def __init__(self, base: Any, delta_grid: Any) -> None:
        self._base = base
        self._delta_grid = delta_grid

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def apply(self, currents_x: Any, currents_y: Any, currents_z: Any | None = None) -> Any:
        if currents_z is None:
            return self._base.apply(currents_x, currents_y)
        flux_x, flux_y, flux_z = self._base.apply(currents_x, currents_y, currents_z)
        return flux_x, flux_y, flux_z + self._delta_grid * currents_z

    def apply_batch(self, currents_x: Any, currents_y: Any, currents_z: Any | None = None) -> Any:
        if currents_z is None:
            return self._base.apply_batch(currents_x, currents_y)
        flux_x, flux_y, flux_z = self._base.apply_batch(currents_x, currents_y, currents_z)
        return flux_x, flux_y, flux_z + self._delta_grid * currents_z


# Members of a batch the layer-pair kernel accumulates in registers at once.
LAYER_PAIR_MIX_MAX_BATCH = 4

_LAYER_PAIR_MIX_SOURCE = r"""
#include <cupy/complex.cuh>
// O[b, t, f] = sum_s K[f, s, t] S[b, s, f]: the layer-pair mixing of a sheet
// convolution in the frequency domain.  One thread per (frequency, target
// layer), the target layer fastest: a warp reads the coupling row K[f, s, :]
// contiguously and every lane of one frequency shares S[b, s, f], so the
// pairs' spectra are read once per application and nothing is transposed.
// K is symmetric in its last two axes, so K[f, s, t] is the (t, s) coupling.
extern "C" __global__ void layer_pair_mix(
    const complex<double>* __restrict__ K,
    const complex<double>* __restrict__ S,
    complex<double>* __restrict__ O,
    const long long F, const int L, const int B)
{
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= F * (long long)L) return;
    const long long f = index / L;
    const int t = (int)(index - f * L);
    const complex<double>* row = K + f * (long long)L * L + t;
    complex<double> a0(0.0, 0.0), a1(0.0, 0.0), a2(0.0, 0.0), a3(0.0, 0.0);
    for (int s = 0; s < L; ++s) {
        const complex<double> k = row[(long long)s * L];
        const complex<double>* column = S + (long long)s * F + f;
        a0 += k * column[0];
        if (B > 1) a1 += k * column[(long long)L * F];
        if (B > 2) a2 += k * column[2LL * L * F];
        if (B > 3) a3 += k * column[3LL * L * F];
    }
    complex<double>* out = O + (long long)t * F + f;
    out[0] = a0;
    if (B > 1) out[(long long)L * F] = a1;
    if (B > 2) out[2LL * L * F] = a2;
    if (B > 3) out[3LL * L * F] = a3;
}
"""
_LAYER_PAIR_MIX_PACKED_SOURCE = r"""
#include <cupy/complex.cuh>
// O[b, t, f] = sum_s K[f, p(t, s)] S[b, s, f] with K the upper triangle of the
// symmetric layer coupling packed per frequency (p(t, s) for t <= s, row by
// row).  A block takes FC consecutive frequencies: it copies their packed
// rows (contiguous) and their source spectra into shared memory with every
// thread loading, then forms its (b, t, f) outputs, f fastest, and writes
// them contiguously along f.  The pairs are read once from global memory,
// half the full stack, in long coalesced runs.
extern "C" __global__ void layer_pair_mix_packed(
    const complex<double>* __restrict__ K,
    const complex<double>* __restrict__ S,
    complex<double>* __restrict__ O,
    const long long F, const int L, const int B, const int FC)
{
    extern __shared__ unsigned char raw[];
    complex<double>* Ks = reinterpret_cast<complex<double>*>(raw);  // [p][fi]
    const int P = L * (L + 1) / 2;
    complex<double>* Ss = Ks + (long long)P * FC;                      // [b*L + s][fi]
    const long long f0 = (long long)blockIdx.x * FC;
    const int fc = (int)(F - f0 < FC ? F - f0 : FC);
    const complex<double>* rows = K + f0 * P;
    for (int i = threadIdx.x; i < fc * P; i += blockDim.x) {
        const int fi = i / P, p = i - fi * P;
        Ks[p * FC + fi] = rows[i];
    }
    for (int i = threadIdx.x; i < B * L * fc; i += blockDim.x) {
        const int bs = i / fc, fi = i - bs * fc;
        Ss[bs * FC + fi] = S[(long long)bs * F + f0 + fi];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < B * L * fc; i += blockDim.x) {
        const int bt = i / fc, fi = i - bt * fc;
        const int b = bt / L, t = bt - b * L;
        complex<double> acc(0.0, 0.0);
        const complex<double>* column = Ss + (b * L) * FC + fi;
        for (int s = 0; s < L; ++s) {
            const int lo = t < s ? t : s, hi = t < s ? s : t;
            const int p = lo * L - (lo * (lo - 1)) / 2 + (hi - lo);
            acc += Ks[p * FC + fi] * column[s * FC];
        }
        O[(long long)bt * F + f0 + fi] = acc;
    }
}
"""
_LAYER_PAIR_MIX_PACKED: dict[int, Any] = {}
# Shared memory a mixing block takes: 48 KiB, the size every device gives a
# block without opting in.
LAYER_PAIR_MIX_SHARED_BYTES = 48 * 1024
LAYER_PAIR_MIX_MAX_FREQUENCIES = 64


def _mix_packed(cp: Any, packed: Any, spectra: Any, out: Any, frequencies: int, count: int, batch: int) -> None:
    """Launch layer_pair_mix_packed over ``spectra`` (B, L, F) into ``out``."""
    key = id(cp)
    if key not in _LAYER_PAIR_MIX_PACKED:
        _LAYER_PAIR_MIX_PACKED[key] = cp.RawKernel(_LAYER_PAIR_MIX_PACKED_SOURCE, "layer_pair_mix_packed")
    kernel = _LAYER_PAIR_MIX_PACKED[key]
    pairs = count * (count + 1) // 2
    per_frequency = 16 * (pairs + batch * count)
    block_frequencies = max(1, min(LAYER_PAIR_MIX_MAX_FREQUENCIES, LAYER_PAIR_MIX_SHARED_BYTES // per_frequency))
    shared = per_frequency * block_frequencies
    if shared > LAYER_PAIR_MIX_SHARED_BYTES:
        raise ValueError(f"{count} layers and a batch of {batch} need {shared} bytes of shared memory a frequency")
    blocks = (frequencies + block_frequencies - 1) // block_frequencies
    kernel((blocks,), (256,), (packed, spectra, out, np.int64(frequencies), np.int32(count), np.int32(batch),
                               np.int32(block_frequencies)), shared_mem=shared)


_LAYER_PAIR_MIX: dict[int, Any] = {}


def _layer_pair_mix_kernel(cp: Any) -> Any | None:
    """The layer-pair mixing kernel of a CuPy module, or None for NumPy standing in for it."""
    raw = getattr(cp, "RawKernel", None)
    if raw is None:
        return None
    key = id(cp)
    if key not in _LAYER_PAIR_MIX:
        _LAYER_PAIR_MIX[key] = raw(_LAYER_PAIR_MIX_SOURCE, "layer_pair_mix")
    return _LAYER_PAIR_MIX[key]


def _device_memory_budget(cp: Any) -> int | None:
    """Half the device's free memory, or None where there is no device (NumPy as ``cp``)."""
    cuda = getattr(cp, "cuda", None)
    if cuda is None:
        return None
    free, _total = cuda.runtime.memGetInfo()
    return int(free) // 2


def cuda_inductance_operator(operator: Any, cp: Any) -> Any:
    """Device counterpart of either sheet inductance operator."""

    if hasattr(operator, "_kernels"):
        return CudaSheetInductanceOperator(operator, cp)
    return CudaPfftSheetInductanceOperator(operator, cp)


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
    preconditioner: str = "near",
    keep_terminal_components: bool = False,
    initial: SheetSolution | None = None,
    near_factorization: str = "superlu",
    near_regularization: float = 1e-8,
) -> tuple[SheetSolution, CudaSheetTelemetry]:
    """Solve one sheet case with cuFFT, CuPy sparse algebra, and CUDA GMRES."""
    return solve_sheet_cases_cuda(
        mesh, operator, [terminals], frequency_hz=frequency_hz, tolerance=tolerance,
        max_iterations=max_iterations, restart=restart, device_id=device_id, cupy_module=cupy_module,
        preconditioner=preconditioner, keep_terminal_components=keep_terminal_components,
        initial=None if initial is None else [initial],
        near_factorization=near_factorization, near_regularization=near_regularization,
    )[0]


def solve_sheet_cases_cuda(
    mesh: SheetMesh,
    operator: SheetInductanceOperator,
    terminal_sets: Sequence[Sequence[Terminal]],
    *,
    frequency_hz: float = 0.0,
    tolerance: float = 1e-10,
    max_iterations: int = 400,
    restart: int = 60,
    device_id: int = 0,
    cupy_module: Any | None = None,
    preconditioner: str = "near",
    keep_terminal_components: bool = False,
    initial: Sequence[SheetSolution | None] | None = None,
    near_factorization: str = "superlu",
    near_regularization: float = 1e-8,
) -> list[tuple[SheetSolution, CudaSheetTelemetry]]:
    """Solve several excitations of one mesh on the device, in the order given.

    The options are :func:`~electrical.sheet_peec.sheet_peec.solve_sheet_cases`'.
    The operator moves to the device once; excitations that drive the same
    components share the reduced system and its factored preconditioner and
    are then solved one after another (no lockstep).  ``keep_terminal_components``
    solves a component a terminal touches even when it carries no current, and
    ``initial`` starts an excitation's Krylov solve from its solution at another
    frequency (the least-residual mix of its parts), as on the CPU.  Each
    excitation gets its own telemetry; ``prepare_ms`` counts the shared
    preparation and its group's factorisation.

    ``near_factorization`` chooses how the ``near`` preconditioner is factored
    and applied: ``superlu`` (SciPy SuperLU factors, CuPy triangular solves)
    or ``cudss_quasi_definite`` (:class:`CudssQuasiDefiniteNear`, cuDSS LDL^T
    of the regularized near matrix, ``near_regularization`` relative).
    """
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
    levels_stated = mesh.vertical_levels
    if not levels_stated or tuple(levels_stated) == tuple(operator.vertical_levels):
        # Barrel vias: the host operator's near matrix and self terms (the
        # preconditioner) and the device products both carry L_barrel - L_column.
        operator = _with_barrels(mesh, operator)
    if preconditioner == "auto":
        # The contract's default.  The CUDA path has no block form, so the
        # choice auto makes on the CPU by size does not arise: it is near.
        preconditioner = "near"
    if preconditioner not in ("near", "diagonal"):
        raise ValueError("preconditioner must be 'auto', 'near' or 'diagonal'")
    if near_factorization not in NEAR_FACTORIZATIONS:
        raise ValueError(f"near_factorization must be one of {NEAR_FACTORIZATIONS}")
    near_regularization = float(near_regularization)
    if not (math.isfinite(near_regularization) and near_regularization > 0.0):
        raise ValueError("near_regularization must be positive")
    frequency_hz = float(frequency_hz)
    if not math.isfinite(frequency_hz) or frequency_hz < 0.0:
        raise ValueError("frequency must be finite and non-negative")
    device_id = int(device_id)
    if device_id < 0:
        raise ValueError("device_id must be non-negative")
    if initial is not None and len(initial) != len(terminal_sets):
        raise ValueError("initial needs one entry (or None) per excitation")
    given_list = list(initial) if initial is not None else [None] * len(terminal_sets)
    for given in given_list:
        if given is not None and (given.branch_current.shape != (mesh.branch_count,)
                                  or given.node_voltage.shape != (mesh.node_count,)):
            raise ValueError("an initial solution is not of this mesh")

    incidence = mesh.incidence()
    resistance = mesh.resistances()
    cases = []
    for terminals in terminal_sets:
        injected = _source_vector(mesh, terminals)
        if frequency_hz == 0.0 and float(np.max(np.abs(injected.imag))) > 1e-15:
            raise ValueError("DC terminal currents must be real")
        touched = _terminal_nodes(mesh, terminals) if keep_terminal_components else None
        cases.append((injected, _reduce(mesh, incidence, injected, touched)))
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, (_, reduction) in enumerate(cases):
        groups.setdefault(reduction.carried, []).append(index)
    solved: list[dict[str, Any] | None] = [None] * len(cases)

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
                if isinstance(operator, _BarrelSelf):
                    cuda_operator = _CudaBarrelSelf(
                        _cached_device_operator(operator._base, cp, device_id), cp.asarray(operator._delta_grid)
                    )
                else:
                    cuda_operator = _cached_device_operator(operator, cp, device_id)
                indices = _branch_indices(mesh, cp)
                cp.cuda.get_current_stream().synchronize()
                shared_prepare_ms = (time.perf_counter() - prepare_started) * 1000.0
                for members in groups.values():
                    _solve_group_cuda(
                        cp, csp, csl, mesh, operator, cuda_operator, indices, resistance, cases, members,
                        given_list, solved, frequency_hz, tolerance, max_iterations, restart, preconditioner,
                        shared_prepare_ms, near_factorization, near_regularization,
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
                # CuPy caches cuFFT plans process-wide, and a plan made here
                # holds a work area from this call's pool.  Left cached past
                # the pool, the next call of the same FFT size ran on freed
                # memory: a 0.2 mm sweep's second frequency solved to zero
                # and the third failed with CUFFT_EXEC_FAILED.
                cp.fft.config.get_plan_cache().clear()
                pool.free_all_blocks()

    total_ms = (time.perf_counter() - total_started) * 1000.0
    out = []
    for record in solved:
        assert record is not None
        telemetry = CudaSheetTelemetry(
            resolved_backend=f"cupy-sheet-peec:{device_id}",
            device_id=device_id,
            device_name=identity["device_name"],
            compute_capability=identity["compute_capability"],
            driver_version=identity["driver_version"],
            runtime_version=identity["runtime_version"],
            cupy_version=identity["cupy_version"],
            prepare_ms=record["prepare_ms"],
            solve_ms=record["solve_ms"],
            total_ms=total_ms,
            krylov_relative_tolerance=tolerance,
            memory_pool_peak_bytes=pool_peak,
            free_memory_before_bytes=int(free_before),
            free_memory_after_bytes=int(free_after),
            preconditioner_factorization_backend=record["factor_backend"],
        )
        out.append((record["solution"], telemetry))
    return out


def _solve_group_cuda(
    cp: Any, csp: Any, csl: Any, mesh: SheetMesh, operator: Any, cuda_operator: Any, indices: "_BranchIndices",
    resistance: np.ndarray, cases: list, members: list[int], given_list: list, solved: list,
    frequency_hz: float, tolerance: float, max_iterations: int, restart: int, preconditioner: str,
    shared_prepare_ms: float, near_factorization: str = "superlu", near_regularization: float = 1e-8,
) -> None:
    """The excitations ``members`` of one reduction: their system prepared once, each solved in turn."""
    group_started = time.perf_counter()
    reduction = cases[members[0]][1]
    keep, grounded, dropped = reduction.keep, reduction.grounded, reduction.dropped
    active_branches = reduction.active_branches
    reduced_cpu = reduction.reduced
    active_resistance = resistance[active_branches]
    reduced = csp.csr_matrix(reduced_cpu)
    resistance_device = cp.asarray(active_resistance)
    # integer positions, not the boolean mask: a masked index makes CuPy count
    # the mask on the host, a device synchronisation in every product
    active_branches_device = cp.asarray(np.flatnonzero(active_branches))

    def finish(index: int, voltage_device: Any, current_device: Any, iterations: int, residual: float,
               converged: bool, factor_backend: str, prepare_ms: float, solve_started: float) -> None:
        cp.cuda.get_current_stream().synchronize()
        solve_ms = (time.perf_counter() - solve_started) * 1000.0
        solved[index] = {
            "solution": SheetSolution(
                node_voltage=cp.asnumpy(voltage_device),
                branch_current=cp.asnumpy(current_device).astype(np.complex128, copy=False),
                frequency_hz=frequency_hz,
                iterations=iterations,
                residual=residual,
                converged=converged,
                grounded_node=grounded,
                undriven_nodes=dropped,
            ),
            "prepare_ms": prepare_ms,
            "solve_ms": solve_ms,
            "factor_backend": factor_backend,
        }

    if frequency_hz == 0.0:
        admittance = csp.diags(1.0 / resistance_device)
        system = (reduced.T @ admittance @ reduced).tocsr()
        prepare_ms = shared_prepare_ms + (time.perf_counter() - group_started) * 1000.0
        for index in members:
            solve_started = time.perf_counter()
            injected_device = cp.asarray(cases[index][0])
            solved_voltage = csl.spsolve(system, injected_device[keep].real)
            voltage_device = cp.zeros(mesh.node_count, dtype=cp.complex128)
            voltage_device[keep] = solved_voltage
            active_current_device = admittance @ (reduced @ solved_voltage)
            current_device = cp.zeros(mesh.branch_count, dtype=cp.complex128)
            current_device[active_branches_device] = active_current_device
            residual = float((
                cp.linalg.norm(reduced.T @ active_current_device - injected_device[keep].real)
                / max(float(cp.linalg.norm(injected_device[keep].real).get()), 1e-30)
            ).get())
            finish(index, voltage_device, current_device, 1, residual, residual < 1e-8, "cusolver_sparse_qr",
                   prepare_ms, solve_started)
        return

    levels = mesh.vertical_levels
    if tuple(levels) != tuple(operator.vertical_levels):
        raise ValueError("the mesh and operator vertical levels differ")
    has_vertical = bool(levels)
    branches = int(active_branches.sum())
    unknowns = int(keep.sum())
    size = branches + unknowns
    omega = 2.0 * math.pi * frequency_hz

    def _flux(currents: Any) -> Any:
        """The operator on ``currents`` ``(B, branches)``, real, all ``B`` in one application."""
        rows, cols = mesh.shape
        batch = currents.shape[0]
        full_currents = cp.zeros((batch, mesh.branch_count), dtype=cp.float64)
        full_currents[:, active_branches_device] = currents
        grid_x = cp.zeros((batch, len(mesh.stackup), rows, cols), dtype=cp.float64)
        grid_y = cp.zeros_like(grid_x)
        x_count = len(mesh.branch_x)
        y_count = len(mesh.branch_y)
        grid_x[:, indices.x_layer, indices.x_row, indices.x_col] = full_currents[:, :x_count]
        grid_y[:, indices.y_layer, indices.y_row, indices.y_col] = full_currents[:, x_count : x_count + y_count]
        if has_vertical:
            grid_z = cp.zeros((batch, len(levels), rows, cols), dtype=cp.float64)
            grid_z[:, indices.z_level, indices.z_row, indices.z_col] = full_currents[:, x_count + y_count :]
            flux_x, flux_y, flux_z = cuda_operator.apply_batch(grid_x, grid_y, grid_z)
        else:
            flux_x, flux_y = cuda_operator.apply_batch(grid_x, grid_y)
            flux_z = None
        full_output = cp.zeros((batch, mesh.branch_count), dtype=cp.float64)
        full_output[:, :x_count] = flux_x[:, indices.x_layer, indices.x_row, indices.x_col]
        full_output[:, x_count : x_count + y_count] = flux_y[:, indices.y_layer, indices.y_row, indices.y_col]
        if flux_z is not None:
            full_output[:, x_count + y_count :] = flux_z[:, indices.z_level, indices.z_row, indices.z_col]
        return full_output[:, active_branches_device]

    def impedance(currents: Any) -> Any:
        # the real and imaginary parts are one batch of two: one read of the operator
        parts = _flux(cp.stack([currents.real, currents.imag]))
        return resistance_device * currents + 1j * omega * (parts[0] + 1j * parts[1])

    def saddle(vector: Any) -> Any:
        currents = vector[:branches]
        voltages = vector[branches:]
        return cp.concatenate([impedance(currents) - reduced @ voltages, reduced.T @ currents])

    def saddle_many(vectors: Any) -> Any:
        """``saddle`` of each row of ``vectors`` (k, size): one operator application for all."""
        currents = vectors[:, :branches]
        voltages = vectors[:, branches:]
        count = vectors.shape[0]
        parts = _flux(cp.concatenate([currents.real, currents.imag]))
        flux = parts[:count] + 1j * parts[count:]
        top = resistance_device[None, :] * currents + 1j * omega * flux - (reduced @ voltages.T).T
        return cp.concatenate([top, (reduced.T @ currents.T).T], axis=1)

    # Excitations of one reduction share the factors; with cuDSS they are also
    # solved together, every product and preconditioner application taking
    # all of them at once (_lockstep_gmres).  One excitation keeps CuPy's GMRES.
    lockstep = (
        len(members)
        if len(members) > 1 and preconditioner == "near" and near_factorization == "cudss_quasi_definite"
        and hasattr(operator, "near_inductance")
        else 1
    )
    if preconditioner == "near" and hasattr(operator, "near_inductance"):
        # Sparse near-field impedance in the saddle-point matrix, factored
        # once (SciPy SuperLU factors, CUDA triangular solves as for the
        # Schur complement).
        import scipy.sparse as sp

        if near_factorization == "cudss_quasi_definite":
            prepared = None
            if hasattr(getattr(operator, "_base", operator), "_tables"):
                # the near matrix formed on the device, unless it would not fit there
                try:
                    prepared = device_quasi_definite_near(
                        mesh, operator, omega, resistance, active_branches, reduced_cpu, near_regularization, cp, csp,
                        memory_budget=_device_memory_budget(cp),
                    )
                except NearMatrixTooLarge:
                    prepared = None
            if prepared is not None:
                factored = CudssQuasiDefiniteNear(None, None, near_regularization, cp, prepared, columns=lockstep)
                factor_backend = "cudss_quasi_definite_ldlt"
            else:
                near_cpu = _near_impedance(mesh, operator, omega, resistance, active_branches)
                factored = CudssQuasiDefiniteNear(near_cpu, reduced_cpu, near_regularization, cp, columns=lockstep)
                factor_backend = "cudss_quasi_definite_ldlt_host_near"
        else:
            near_cpu = _near_impedance(mesh, operator, omega, resistance, active_branches)
            saddle_cpu = sp.bmat([[near_cpu, -reduced_cpu], [reduced_cpu.T, None]], format="csc")
            factored = csl.splu(csp.csc_matrix(saddle_cpu))
            factor_backend = "scipy_superlu_factor_cupy_triangular_solve"

        def precondition(vector: Any) -> Any:
            return factored.solve(vector)

    else:
        full_diagonal_cpu = resistance + 1j * omega * np.concatenate(
            [_inline_self_inductance(mesh, operator), _vertical_self_inductance(mesh, operator)]
        )
        diagonal = cp.asarray(full_diagonal_cpu[active_branches])
        schur = (reduced.T @ csp.diags(1.0 / diagonal) @ reduced).tocsc()
        factored = csl.splu(schur)

        factor_backend = "scipy_superlu_factor_cupy_triangular_solve"

        def precondition(vector: Any) -> Any:
            rhs_current = vector[:branches]
            rhs_node = vector[branches:]
            node = factored.solve(rhs_node + reduced.T @ (rhs_current / diagonal))
            current = (rhs_current + reduced @ node) / diagonal
            return cp.concatenate([current, node])

    # CuPy's ``M=`` path implements a right-preconditioned variable, unlike
    # SciPy's left-preconditioned GMRES used by the CPU reference.  Build
    # M^-1 A and M^-1 b explicitly so both paths solve the same Krylov system
    # and the reference-node closure does not accumulate the residual left
    # at every other node.
    def left_preconditioned_saddle(vector: Any) -> Any:
        return precondition(saddle(vector))

    linear_system = csl.LinearOperator((size, size), matvec=left_preconditioned_saddle, dtype=cp.complex128)
    cp.cuda.get_current_stream().synchronize()
    prepare_ms = shared_prepare_ms + (time.perf_counter() - group_started) * 1000.0
    effective_restart = min(restart, size)
    floor = 100.0 * np.finfo(np.float64).eps

    if lockstep > 1:
        try:
            _lockstep_gmres(
                cp, mesh, members, cases, given_list, saddle, saddle_many, factored, active_branches,
                active_branches_device, keep, branches, unknowns, size, tolerance, max_iterations, effective_restart,
                floor, finish, factor_backend, prepare_ms,
            )
        finally:
            factored.free()
        return
    try:
        for index in members:
            solve_started = time.perf_counter()
            injected_device = cp.asarray(cases[index][0])
            right_hand_side = cp.concatenate([cp.zeros(branches, dtype=cp.complex128), injected_device[keep]])
            left_right_hand_side = precondition(right_hand_side)
            counter = {"restarts": 0}

            def count(_value: Any) -> None:
                counter["restarts"] += 1

            rhs_norm = float(cp.linalg.norm(right_hand_side).get())
            left_norm = float(cp.linalg.norm(left_right_hand_side).get())
            given = given_list[index]
            if given is not None:
                result = _cuda_start(cp, saddle, given, mesh, active_branches, keep, branches, unknowns, right_hand_side)
                residual = float(cp.linalg.norm(saddle(result) - right_hand_side).get()) / max(rhs_norm, 1e-30)
            else:
                result = cp.zeros(size, dtype=cp.complex128)
                residual = 1.0 if rhs_norm > 0.0 else 0.0
            # GMRES is run one restart cycle at a time and the public saddle-point
            # residual is checked between cycles, so the solve stops when the
            # requested tolerance is met and ``max_iterations`` counts cycles as on
            # the CPU path.  Aiming CuPy's own (left-preconditioned) criterion
            # three decades below the tolerance ran every solve to its cap.
            cycles = 0
            while residual > tolerance and cycles < max_iterations:
                # Cycles grow from ten inner iterations towards ``restart``: a
                # well-preconditioned solve converges in a few tens of iterations
                # and must not pay for a full cycle to find out.
                cycle_length = min(effective_restart, 10 * 2**cycles)
                # CuPy judges the left-preconditioned residual; scale its target by
                # the current ratio of the two residuals so that meeting it brings
                # the saddle-point residual to the requested tolerance, and stop
                # when that target falls below what float64 can resolve.
                raw = saddle(result) - right_hand_side
                preconditioned = float(cp.linalg.norm(precondition(raw)).get()) / max(left_norm, 1e-30)
                ratio = preconditioned / max(residual, 1e-300) if residual > 0.0 else 1.0
                target = 0.5 * tolerance * ratio
                if target < floor:
                    break
                result, _info = csl.gmres(
                    linear_system, left_right_hand_side, x0=result, rtol=target, restart=cycle_length,
                    maxiter=cycle_length, callback=count, callback_type="pr_norm",
                )
                cycles += 1
                residual = float((cp.linalg.norm(saddle(result) - right_hand_side)).get()) / max(rhs_norm, 1e-30)
            current_device = cp.zeros(mesh.branch_count, dtype=cp.complex128)
            current_device[active_branches_device] = result[:branches]
            voltage_device = cp.zeros(mesh.node_count, dtype=cp.complex128)
            voltage_device[keep] = result[branches:]
            finish(index, voltage_device, current_device, counter["restarts"], residual, residual <= tolerance,
                   factor_backend, prepare_ms, solve_started)
    finally:
        release = getattr(factored, "free", None)
        if release is not None:
            release()


# Device memory the lockstep Krylov basis may take, as a share of the free memory.
LOCKSTEP_BASIS_SHARE = 0.5


def _lockstep_gmres(
    cp: Any, mesh: SheetMesh, members: list[int], cases: list, given_list: list, saddle: Any, saddle_many: Any,
    factored: Any, active_branches: np.ndarray, active_branches_device: Any, keep: np.ndarray, branches: int,
    unknowns: int, size: int, tolerance: float, max_iterations: int, restart: int, floor: float, finish: Any,
    factor_backend: str, prepare_ms: float,
) -> None:
    """Restarted, left-preconditioned GMRES for the excitations ``members``, all at once.

    Every excitation keeps its own Krylov basis and least-squares problem,
    but they advance together: each step applies the saddle-point operator
    to all the active ones in one batched product and the near factors to
    them in one multi-column cuDSS solve, so the stack and the factors --
    the bytes a step moves -- are read once for the lot.  Orthogonalization
    is classical Gram-Schmidt twice, batched; the Givens rotations and the
    residual estimates are on the host.  As on the single-excitation path,
    a cycle grows from ten steps towards ``restart``, stops when every
    active excitation's preconditioned residual meets a target scaled so
    that its saddle-point residual meets ``tolerance``, and an excitation
    leaves once its saddle-point residual is below it.  The basis is capped
    so that it fits in a share of the free device memory.
    """
    started = time.perf_counter()
    count = len(members)
    rhs = cp.empty((count, size), dtype=cp.complex128)
    rhs[:, :branches] = 0.0
    for row, index in enumerate(members):
        rhs[row, branches:] = cp.asarray(cases[index][0])[keep]
    x = cp.zeros((count, size), dtype=cp.complex128)
    for row, index in enumerate(members):
        given = given_list[index]
        if given is not None:
            x[row] = _cuda_start(cp, saddle, given, mesh, active_branches, keep, branches, unknowns, rhs[row])
    rhs_norm = cp.maximum(cp.linalg.norm(rhs, axis=1), 1e-30)
    residual = cp.asnumpy(cp.linalg.norm(saddle_many(x) - rhs, axis=1) / rhs_norm)
    residual[cp.asnumpy(cp.linalg.norm(rhs, axis=1)) == 0.0] = 0.0
    steps = np.zeros(count, dtype=np.int64)
    free = int(cp.cuda.runtime.memGetInfo()[0])
    cap = max(2, int(LOCKSTEP_BASIS_SHARE * free // (16 * size * count)) - 1)
    cycles = 0
    while (residual > tolerance).any() and cycles < max_iterations:
        active = np.flatnonzero(residual > tolerance)
        k = len(active)
        length = max(1, min(restart, 10 * 2**cycles, cap))
        current = x[active]
        start = factored.solve_many(rhs[active] - saddle_many(current))
        beta = cp.linalg.norm(start, axis=1)
        beta_host = cp.asnumpy(beta)
        # the preconditioned residual each excitation has to reach in this cycle
        target = 0.5 * tolerance * beta_host / np.maximum(residual[active], 1e-300)
        if (target < floor * np.maximum(beta_host, 1e-300)).all():
            break
        basis = cp.empty((k, length + 1, size), dtype=cp.complex128)
        basis[:, 0] = start / cp.maximum(beta, 1e-300)[:, None]
        hessenberg = np.zeros((k, length + 1, length), dtype=np.complex128)
        cosine = np.zeros((k, length), dtype=np.complex128)
        sine = np.zeros((k, length), dtype=np.complex128)
        g = np.zeros((k, length + 1), dtype=np.complex128)
        g[:, 0] = beta_host
        done = 0
        for j in range(length):
            w = factored.solve_many(saddle_many(basis[:, j]))
            column = cp.zeros((k, j + 1), dtype=cp.complex128)
            for _ in range(2):  # classical Gram-Schmidt, twice
                h = cp.matmul(basis[:, : j + 1].conj(), w[:, :, None])[:, :, 0]
                w = w - cp.matmul(h[:, None, :], basis[:, : j + 1])[:, 0]
                column += h
            norm = cp.linalg.norm(w, axis=1)
            basis[:, j + 1] = w / cp.maximum(norm, 1e-300)[:, None]
            hessenberg[:, : j + 1, j] = cp.asnumpy(column)
            hessenberg[:, j + 1, j] = cp.asnumpy(norm)
            # Givens rotations on the host: each excitation's residual estimate
            for i in range(k):
                for m in range(j):
                    a, b = hessenberg[i, m, j], hessenberg[i, m + 1, j]
                    hessenberg[i, m, j] = np.conj(cosine[i, m]) * a + np.conj(sine[i, m]) * b
                    hessenberg[i, m + 1, j] = -sine[i, m] * a + cosine[i, m] * b
                a, b = hessenberg[i, j, j], hessenberg[i, j + 1, j]
                radius = np.hypot(abs(a), abs(b))
                if radius == 0.0:
                    cosine[i, j], sine[i, j] = 1.0, 0.0
                else:
                    cosine[i, j], sine[i, j] = a / radius, b / radius
                hessenberg[i, j, j] = radius
                hessenberg[i, j + 1, j] = 0.0
                g[i, j + 1] = -sine[i, j] * g[i, j]
                g[i, j] = np.conj(cosine[i, j]) * g[i, j]
            done = j + 1
            steps[active] += 1
            if (np.abs(g[:, j + 1]) <= target).all():
                break
        # each excitation's least-squares update from its triangular system
        update = np.zeros((k, done), dtype=np.complex128)
        for i in range(k):
            r = hessenberg[i, :done, :done]
            update[i] = np.linalg.solve(np.triu(r), g[i, :done]) if done else update[i]
        current = current + cp.matmul(cp.asarray(update)[:, None, :], basis[:, :done])[:, 0]
        x[active] = current
        cycles += 1
        residual[active] = cp.asnumpy(cp.linalg.norm(saddle_many(current) - rhs[active], axis=1) / rhs_norm[active])
    for row, index in enumerate(members):
        current_device = cp.zeros(mesh.branch_count, dtype=cp.complex128)
        current_device[active_branches_device] = x[row, :branches]
        voltage_device = cp.zeros(mesh.node_count, dtype=cp.complex128)
        voltage_device[keep] = x[row, branches:]
        finish(index, voltage_device, current_device, int(steps[row]), float(residual[row]),
               bool(residual[row] <= tolerance), f"{factor_backend}_lockstep{count}", prepare_ms, started)


def _cuda_start(
    cp: Any, saddle: Any, given: SheetSolution, mesh: SheetMesh, active_branches: np.ndarray, keep: np.ndarray,
    branches: int, unknowns: int, right_hand_side: Any,
) -> Any:
    """The least-residual mix of a solution's real and imaginary currents and potentials (``initial``)."""
    current = cp.asarray(given.branch_current[active_branches])
    voltage = cp.asarray(given.node_voltage[keep])
    no_current = cp.zeros(branches, dtype=cp.complex128)
    no_voltage = cp.zeros(unknowns, dtype=cp.complex128)
    basis = cp.stack([
        cp.concatenate([current.real.astype(cp.complex128), no_voltage]),
        cp.concatenate([current.imag.astype(cp.complex128), no_voltage]),
        cp.concatenate([no_current, voltage.real.astype(cp.complex128)]),
        cp.concatenate([no_current, voltage.imag.astype(cp.complex128)]),
    ], axis=1)
    images = cp.stack([saddle(basis[:, k]) for k in range(4)], axis=1)
    coefficients = cp.linalg.lstsq(images, right_hand_side, rcond=None)[0]
    return basis @ coefficients
