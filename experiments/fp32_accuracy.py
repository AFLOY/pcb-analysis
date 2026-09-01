"""Measure what single precision costs the sheet-PEEC solver.

FP64 on a consumer Turing part runs at a thirty-second of the FP32 rate, so
asking whether this solver can work in FP32 is worth asking.  The question is
not one question.  Single precision enters at three separate places, and the
three fail for different reasons and at different magnitudes:

* **Kernel assembly.**  ``sheet_inductance`` evaluates Ruehli's partial mutual
  inductance as a signed sum of 64 corner terms.  The terms grow like the fifth
  power of the corner coordinates while their signed sum stays of order the
  answer, so the sum cancels its own precision away as the bars separate.
  ``docs/SHEET_PEEC.md`` already records the surviving fraction in FP64: 6.3e-2
  at one cell, 8.1e-8 at sixteen.  A surviving fraction ``r`` leaves a relative
  error of about ``eps/r``, and FP32's ``eps`` is 6e-8 rather than 1.1e-16, so
  this is where FP32 should break first and hardest.

* **The operator apply.**  The convolution is a transform, a spectral product,
  and an inverse transform.  Nothing cancels: parallel co-directed bars couple
  with one sign, so the accumulation is essentially positive.  This should cost
  about ``eps`` and no more.

* **The Krylov solve.**  The public contract asks for a relative residual of
  1e-10.  FP32 cannot represent that residual, let alone reach it, so the
  question here is what residual is reachable and what the reachable one leaves
  on the answer a gate reads.

Stage A runs on the host, where FP32 and FP64 are both exactly available and
the measurement is deterministic.  Stages B and C run on the device, because a
statement about CUDA execution that was not made on CUDA is not a statement.

Usage::

    python experiments/fp32_accuracy.py                # all stages, 48x48
    python experiments/fp32_accuracy.py --cells 128    # a size that matters
    python experiments/fp32_accuracy.py --stage a      # host stage only
    python experiments/fp32_accuracy.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from electrical.dice_peec.sheet_inductance import (
    _AXIS_SIGNS,
    _MU0_OVER_4PI,
    NEAR_RADIUS_CELLS,
    CellGeometry,
    closed_form_mutual_inductance,
    closed_form_precision,
    far_field_mutual_inductance,
)
from electrical.dice_peec.sheet_operator import SheetInductanceOperator, SheetLayer, SheetStackup
from electrical.dice_peec.sheet_peec import (
    SheetMesh,
    SheetSolution,
    Terminal,
    ViaBranch,
    _components,
    _components_with_terminals,
    _self_inductance,
    _source_vector,
    _vertical_self_inductance,
    solve_sheet_case,
    via_resistance,
)
from electrical.dice_peec.sheet_results import sheet_fields


# --------------------------------------------------------------------------
# Stage A: the closed-form kernel, in each precision
# --------------------------------------------------------------------------


def _primitive_at(x: Any, y: Any, z: Any, dtype: Any) -> Any:
    """Evaluate the Hoer and Love antiderivative wholly within ``dtype``.

    A copy of ``sheet_inductance._primitive`` with the hard-coded ``float64``
    lifted out.  The library's own version is deliberately fixed to double, so
    measuring the single-precision behaviour needs a version that is not; and
    the point of the measurement is the arithmetic, which has to happen in the
    dtype under test rather than being rounded to it at the end.
    """
    x = np.asarray(x, dtype=dtype)
    y = np.asarray(y, dtype=dtype)
    z = np.asarray(z, dtype=dtype)
    x2, y2, z2 = x * x, y * y, z * z
    rho = np.sqrt(x2 + y2 + z2)

    def safe_log(numerator: Any, denominator: Any) -> Any:
        numerator, denominator = np.broadcast_arrays(numerator, denominator)
        ratio = np.divide(
            numerator,
            denominator,
            out=np.ones(numerator.shape, dtype=dtype),
            where=(denominator > 0.0) & (numerator > 0.0),
        )
        return np.log(np.where(ratio > 0.0, ratio, dtype(1.0)))

    def safe_atan(numerator: Any, denominator: Any) -> Any:
        numerator, denominator = np.broadcast_arrays(numerator, denominator)
        return np.arctan(
            np.divide(
                numerator,
                denominator,
                out=np.zeros(numerator.shape, dtype=dtype),
                where=denominator != 0.0,
            )
        )

    quarter = dtype(4.0)
    twentyfourth = dtype(24.0)
    term = (
        (y2 * z2 / quarter - y2 * y2 / twentyfourth - z2 * z2 / twentyfourth)
        * x
        * safe_log(x + rho, np.sqrt(y2 + z2))
    )
    term += (
        (x2 * z2 / quarter - x2 * x2 / twentyfourth - z2 * z2 / twentyfourth)
        * y
        * safe_log(y + rho, np.sqrt(x2 + z2))
    )
    term += (
        (x2 * y2 / quarter - x2 * x2 / twentyfourth - y2 * y2 / twentyfourth)
        * z
        * safe_log(z + rho, np.sqrt(x2 + y2))
    )
    term += (
        (
            x2 * x2
            + y2 * y2
            + z2 * z2
            - dtype(3.0) * x2 * y2
            - dtype(3.0) * y2 * z2
            - dtype(3.0) * z2 * x2
        )
        * rho
        / dtype(60.0)
    )
    six = dtype(6.0)
    term -= (x * y * z * z2 / six) * safe_atan(x * y, z * rho)
    term -= (x * y * y2 * z / six) * safe_atan(x * z, y * rho)
    term -= (x2 * x * y * z / six) * safe_atan(y * z, x * rho)
    return term


def closed_form_in_dtype(
    cell: CellGeometry,
    other: CellGeometry,
    offset_m: tuple[float, float, float],
    dtype: Any,
) -> tuple[float, float]:
    """Give the closed-form mutual inductance and its surviving fraction.

    The second return value is the ratio of the signed sum to the largest term
    that went into it: the fraction of the magnitude that did not cancel.  It
    bounds the achievable relative error at about ``eps / retained``, in
    whichever precision the sum was taken.
    """
    extents = (
        (dtype(offset_m[0]), cell.length_m, other.length_m),
        (dtype(offset_m[1]), cell.width_m, other.width_m),
        (dtype(offset_m[2]), cell.thickness_m, other.thickness_m),
    )
    total = dtype(0.0)
    largest = dtype(0.0)
    half = dtype(0.5)
    for sx, tx in _AXIS_SIGNS:
        x = extents[0][0] + dtype(sx) * dtype(extents[0][1]) * half + dtype(
            tx
        ) * dtype(extents[0][2]) * half
        for sy, ty in _AXIS_SIGNS:
            y = extents[1][0] + dtype(sy) * dtype(extents[1][1]) * half + dtype(
                ty
            ) * dtype(extents[1][2]) * half
            for sz, tz in _AXIS_SIGNS:
                z = extents[2][0] + dtype(sz) * dtype(
                    extents[2][1]
                ) * half + dtype(tz) * dtype(extents[2][2]) * half
                value = dtype(_primitive_at(x, y, z, dtype))
                total = total + dtype(sx * tx * sy * ty * sz * tz) * value
                largest = max(largest, dtype(abs(value)))
    retained = float(abs(total) / largest) if largest > 0.0 else 1.0
    scale = dtype(_MU0_OVER_4PI) / (
        dtype(cell.cross_section_m2) * dtype(other.cross_section_m2)
    )
    return float(scale * total), retained


def stage_a(pitch_m: float, thickness_m: float) -> dict[str, Any]:
    """Compare the kernel in each precision against an exact reference.

    The reference is not FP64's own closed form -- that is the thing under
    test, and it cancels too.  Distant bars are checked against the
    centre-to-centre limit, which the library itself switches to past the near
    radius and which involves no cancellation at all.  Near bars, where the
    limit is not yet valid, are checked against the closed form evaluated in
    ``longdouble``: two more decimal digits than FP64, enough to resolve FP64's
    own error and far more than enough to resolve FP32's.
    """
    cell = CellGeometry(
        length_m=pitch_m, width_m=pitch_m, thickness_m=thickness_m
    )
    offsets = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    rows = []
    for cells in offsets:
        offset = (cells * pitch_m, 0.0, 0.0)
        exact, _ = closed_form_in_dtype(cell, cell, offset, np.longdouble)
        single, retained32 = closed_form_in_dtype(cell, cell, offset, np.float32)
        double, retained64 = closed_form_in_dtype(cell, cell, offset, np.float64)
        limit = (
            float(far_field_mutual_inductance(cell, cell, offset))
            if cells > 0
            else float("nan")
        )
        rows.append(
            {
                "offset_cells": cells,
                "exact_h": exact,
                "retained_fp64": retained64,
                "retained_fp32": retained32,
                "relative_error_fp64": abs(double - exact) / abs(exact),
                "relative_error_fp32": abs(single - exact) / abs(exact),
                "relative_error_far_field_limit": (
                    abs(limit - exact) / abs(exact) if cells > 0 else float("nan")
                ),
            }
        )

    # Where does FP32's closed form still carry a stated number of digits?  The
    # library's near radius is 24 cells; the useful comparison is how far FP32
    # could push a near radius before the closed form stops being an answer.
    usable = [
        row["offset_cells"]
        for row in rows
        if row["relative_error_fp32"] < 1e-3 and row["offset_cells"] > 0
    ]

    # A near radius is only safe where both regimes are good at once: inside it
    # the closed form must still be accurate, outside it the centre-to-centre
    # limit must already be.  The seam's quality is the worse of the two at the
    # crossover, minimised over where the crossover is put.  In FP64 that
    # minimum is 1e-4 and the window is wide.  Report it for each precision
    # rather than asserting the FP32 one is bad.
    seams = {}
    for precision in ("fp64", "fp32"):
        candidates = [
            (
                max(
                    row[f"relative_error_{precision}"],
                    row["relative_error_far_field_limit"],
                ),
                row["offset_cells"],
            )
            for row in rows
            if row["offset_cells"] > 0
        ]
        quality, where = min(candidates)
        seams[precision] = {
            "best_near_radius_cells": where,
            "achievable_seam_error": quality,
        }

    # The library refuses the closed form when the surviving fraction falls
    # under 1e-11.  That threshold was calibrated against FP64's epsilon.  In
    # FP32 the surviving fraction stops measuring the signal and starts
    # measuring the noise floor, so it reads high exactly where the answer is
    # worthless -- the guard would pass, not fire.
    guard_would_fire = [
        row["offset_cells"]
        for row in rows
        if row["retained_fp32"] < 1e-11
    ]
    misled = [
        row
        for row in rows
        if row["retained_fp32"] >= 1e-11 and row["relative_error_fp32"] > 1e-2
    ]
    return {
        "pitch_m": pitch_m,
        "thickness_m": thickness_m,
        "library_near_radius_cells": NEAR_RADIUS_CELLS,
        "fp32_last_offset_within_1e-3": max(usable) if usable else 0,
        "seams": seams,
        "precision_guard": {
            "threshold_retained": 1e-11,
            "fp32_offsets_where_guard_fires": guard_would_fire,
            "fp32_offsets_passing_guard_with_error_over_1e-2": [
                row["offset_cells"] for row in misled
            ],
            "worst_error_passing_guard": (
                max(row["relative_error_fp32"] for row in misled) if misled else 0.0
            ),
        },
        "rows": rows,
    }


# --------------------------------------------------------------------------
# The fixture stages B and C run on
# --------------------------------------------------------------------------


@dataclass
class Fixture:
    """A two-layer plane pair returning current through a via column."""

    mesh: SheetMesh
    operator: SheetInductanceOperator
    terminals: tuple[Terminal, ...]
    frequency_hz: float
    source_node: int
    sink_node: int
    current_a: float

    @property
    def description(self) -> dict[str, Any]:
        return {
            "shape": list(self.mesh.shape),
            "pitch_m": self.mesh.pitch_m,
            "layers": len(self.mesh.stackup),
            "nodes": self.mesh.node_count,
            "branches": self.mesh.branch_count,
            "vias": len(self.mesh.via_branches),
            "frequency_hz": self.frequency_hz,
            "operator_kernel_bytes": self.operator.kernel_bytes,
        }


def build_fixture(cells: int, frequency_hz: float) -> Fixture:
    """Build a power-plane loop: in on the front, across, down, back on the back.

    Both layers are solid copper, so every cell is a conductor and the mesh is
    the dense case rather than a favourable sparse one.  Current enters one edge
    of the front plane, spreads, returns through a column of vias at the far
    edge, and leaves the same edge of the back plane.  That is a loop with a
    real inductance, which is the quantity a PDN gate reads, so the accuracy
    that matters can be quoted on it rather than only on a residual.
    """
    pitch_m = 0.2e-3
    thickness_m = 35e-6
    board_m = 1.6e-3
    stackup = SheetStackup(
        layers=(
            SheetLayer(name="F.Cu", z_m=0.0, thickness_m=thickness_m),
            SheetLayer(name="B.Cu", z_m=-board_m, thickness_m=thickness_m),
        )
    )
    shape = (cells, cells)
    occupancy = np.ones((2, cells, cells), dtype=bool)

    barrel = via_resistance(
        barrel_length_m=board_m,
        drill_diameter_m=0.3e-3,
        plating_thickness_m=25e-6,
    )
    middle = cells // 2
    via_rows = range(max(0, middle - 2), min(cells, middle + 3))
    far_col = cells - 2
    vias = tuple(
        ViaBranch(
            row=row,
            col=far_col,
            lower_layer=1,
            upper_layer=0,
            resistance_ohm=barrel,
        )
        for row in via_rows
    )
    mesh = SheetMesh(
        shape=shape,
        pitch_m=pitch_m,
        stackup=stackup,
        occupancy=occupancy,
        vias=vias,
    )
    operator = SheetInductanceOperator(
        shape,
        pitch_m,
        stackup,
        vertical_levels=mesh.vertical_levels,
    )

    current_a = 1.0
    pad = tuple((row, 1) for row in via_rows)
    terminals = (
        Terminal(name="source", layer=0, cells=pad, current_a=current_a),
        Terminal(name="sink", layer=1, cells=pad, current_a=-current_a),
    )
    return Fixture(
        mesh=mesh,
        operator=operator,
        terminals=terminals,
        frequency_hz=frequency_hz,
        source_node=mesh.node_index[(0, middle, 1)],
        sink_node=mesh.node_index[(1, middle, 1)],
        current_a=current_a,
    )


def terminal_impedance(fixture: Fixture, voltage: np.ndarray) -> complex:
    """Read the power-conjugate impedance of the distributed excitation.

    A terminal current is spread evenly over every cell in its pad.  The
    corresponding port voltage is therefore the current-weighted mean of those
    node potentials, not the potential of one arbitrarily selected pad cell.
    """
    delivered_power = 0.0j
    for terminal in fixture.terminals:
        usable = [
            fixture.mesh.node_index[(terminal.layer, row, col)]
            for row, col in terminal.cells
            if (terminal.layer, row, col) in fixture.mesh.node_index
        ]
        if not usable:
            raise ValueError(f"terminal {terminal.name} has no cell on the conductor")
        share = terminal.current_a / len(usable)
        delivered_power += sum(
            voltage[index] * np.conjugate(share) for index in usable
        )
    scale = abs(fixture.current_a) ** 2
    if scale == 0.0:
        raise ValueError("fixture current must be non-zero")
    return complex(delivered_power / scale)


def loop_inductance_h(fixture: Fixture, voltage: np.ndarray) -> float:
    omega = 2.0 * math.pi * fixture.frequency_hz
    if omega == 0.0:
        return float("nan")
    return terminal_impedance(fixture, voltage).imag / omega


# --------------------------------------------------------------------------
# Stage B: the operator apply, on the device, in each precision
# --------------------------------------------------------------------------


def _device_spectra(operator: SheetInductanceOperator, cp: Any, complex_dtype: Any):
    return (
        {
            key: cp.asarray(value, dtype=complex_dtype)
            for key, value in operator._kernels.items()
        },
        {
            key: cp.asarray(value, dtype=complex_dtype)
            for key, value in operator._kernels_z.items()
        },
    )


def _apply_on_device(
    operator: SheetInductanceOperator,
    cp: Any,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray | None,
    real_dtype: Any,
    complex_dtype: Any,
    device_spectra: tuple[dict[Any, Any], dict[Any, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Run the convolution wholly in the given precision and bring it back.

    Mirrors ``CudaSheetInductanceOperator`` term for term.  The kernels are
    assembled in FP64 on the host and cast, so that this measures the
    transform and the spectral product alone -- stage A already measured what
    assembling them in FP32 would cost, and mixing the two would confound the
    two effects.
    """
    kernels, kernels_z = (
        _device_spectra(operator, cp, complex_dtype)
        if device_spectra is None
        else device_spectra
    )
    layers = len(operator.stackup)
    levels = len(operator.vertical_levels)
    rows, cols = operator.shape

    def kernel(axis: str, first: int, second: int):
        key = (axis, first, second) if first <= second else (axis, second, first)
        return kernels[key]

    def kernel_z(first: int, second: int):
        key = (first, second) if first <= second else (second, first)
        return kernels_z[key]

    def convolve(spectra, axis: str, count: int, table):
        out = cp.empty((count, rows, cols), dtype=real_dtype)
        for target in range(count):
            accumulated = cp.zeros_like(spectra[0])
            for source in range(count):
                accumulated += table(axis, target, source) * spectra[source]
            product = cp.fft.irfft2(accumulated, s=operator.padded)
            out[target] = product[:rows, :cols]
        return out

    device_x = cp.asarray(grid_x, dtype=real_dtype)
    device_y = cp.asarray(grid_y, dtype=real_dtype)
    spectra_x = cp.fft.rfft2(device_x, s=operator.padded, axes=(-2, -1))
    spectra_y = cp.fft.rfft2(device_y, s=operator.padded, axes=(-2, -1))
    flux_x = convolve(spectra_x, "x", layers, kernel)
    flux_y = convolve(spectra_y, "y", layers, kernel)
    flux_z = None
    if grid_z is not None and levels:
        device_z = cp.asarray(grid_z, dtype=real_dtype)
        spectra_z = cp.fft.rfft2(device_z, s=operator.padded, axes=(-2, -1))
        flux_z = convolve(
            spectra_z, "z", levels, lambda _axis, a, b: kernel_z(a, b)
        )
    cp.cuda.get_current_stream().synchronize()
    return (
        cp.asnumpy(flux_x).astype(np.float64),
        cp.asnumpy(flux_y).astype(np.float64),
        None if flux_z is None else cp.asnumpy(flux_z).astype(np.float64),
    )


def _gather_branch_flux(
    mesh: SheetMesh,
    flux_x: np.ndarray,
    flux_y: np.ndarray,
    flux_z: np.ndarray | None,
) -> np.ndarray:
    """Gather only values belonging to physical branches."""
    gathered = mesh.gather(flux_x, flux_y)
    if flux_z is not None:
        mesh.gather_vertical(flux_z, gathered)
    return gathered


def _host_branch_flux(fixture: Fixture, currents: np.ndarray) -> np.ndarray:
    """Apply the host operator to one real branch-current component."""
    mesh = fixture.mesh
    grid_x, grid_y = mesh.scatter(currents)
    grid_z = mesh.scatter_vertical(currents) if mesh.via_branches else None
    flux = fixture.operator.apply(
        grid_x, grid_y, grid_z if grid_z is not None else None
    )
    if grid_z is None:
        flux_x, flux_y = flux
        flux_z = None
    else:
        flux_x, flux_y, flux_z = flux
    return _gather_branch_flux(mesh, flux_x, flux_y, flux_z)


def stage_b(fixture: Fixture, currents: np.ndarray, cp: Any) -> dict[str, Any]:
    """Measure the complex branch flux of the solved current distribution.

    The currents are the ones the FP64 solve settled on, not a random vector: a
    random vector has no cancellation structure and would flatter or damn the
    transform for the wrong reason.  Only physical branches are compared; dummy
    grid positions used to carry the FFT embedding are excluded.
    """
    mesh = fixture.mesh
    reference = _host_branch_flux(fixture, currents.real).astype(np.complex128)
    reference += 1j * _host_branch_flux(fixture, currents.imag)
    results = {}
    for name, real_dtype, complex_dtype in (
        ("fp64", np.float64, np.complex128),
        ("fp32", np.float32, np.complex64),
    ):
        device_spectra = _device_spectra(fixture.operator, cp, complex_dtype)
        started = time.perf_counter()
        obtained = np.zeros(mesh.branch_count, dtype=np.complex128)
        for component, factor in ((currents.real, 1.0), (currents.imag, 1.0j)):
            grid_x, grid_y = mesh.scatter(component)
            grid_z = (
                mesh.scatter_vertical(component) if mesh.via_branches else None
            )
            got_x, got_y, got_z = _apply_on_device(
                fixture.operator,
                cp,
                grid_x,
                grid_y,
                grid_z,
                real_dtype,
                complex_dtype,
                device_spectra=device_spectra,
            )
            obtained += factor * _gather_branch_flux(
                mesh, got_x, got_y, got_z
            )
        elapsed = (time.perf_counter() - started) * 1000.0

        scale = np.linalg.norm(reference)
        difference = obtained - reference
        results[name] = {
            "relative_l2_error": float(np.linalg.norm(difference) / scale),
            "max_absolute_error_wb": float(np.max(np.abs(difference))),
            "max_relative_to_peak": float(
                np.max(np.abs(difference)) / np.max(np.abs(reference))
            ),
            "complex_apply_pair_ms": elapsed,
        }
    return {
        "reference": "host float64 operator, gathered physical branches",
        "branch_count": mesh.branch_count,
        "real_current_l2_norm_a": float(np.linalg.norm(currents.real)),
        "imag_current_l2_norm_a": float(np.linalg.norm(currents.imag)),
        **results,
    }


# --------------------------------------------------------------------------
# Stage C: the whole solve, on the device, in each precision
# --------------------------------------------------------------------------


@dataclass
class SolveOutcome:
    precision: str
    converged: bool
    working_precision_mixed_unit_residual: float
    requested_rtol: float
    iterations: int
    prepare_ms: float
    factor_ms: float
    solve_ms: float
    total_ms: float
    voltage: np.ndarray = field(repr=False)
    current: np.ndarray = field(repr=False)
    preconditioner_dtype: str = ""


def solve_on_device(
    fixture: Fixture,
    cp: Any,
    *,
    precision: str,
    rtol: float,
    max_iterations: int = 400,
    restart: int = 60,
) -> SolveOutcome:
    """Solve the saddle-point system on the device in one precision.

    Structurally this is ``sheet_cuda.solve_sheet_case_cuda`` with the dtypes
    lifted into parameters and the CPU-only preamble shared.  It is a copy
    rather than a call because the shipped solver fixes ``complex128``
    throughout; running both precisions through the same code here is what makes
    the difference attributable to the precision rather than to two code paths.
    """
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg as csl

    real_dtype, complex_dtype = (
        (np.float32, np.complex64)
        if precision == "fp32"
        else (np.float64, np.complex128)
    )
    mesh = fixture.mesh
    operator = fixture.operator
    omega = 2.0 * math.pi * fixture.frequency_hz

    incidence = mesh.incidence()
    resistance = mesh.resistances()
    injected = _source_vector(mesh, fixture.terminals)
    components, labels = _components(incidence, mesh.node_count)
    carried = _components_with_terminals(labels, components, injected)
    kept_nodes = np.isin(labels, list(carried))
    keep = kept_nodes.copy()
    for component in sorted(carried):
        keep[int(np.flatnonzero(labels == component)[0])] = False
    driven_endpoint_count = np.asarray(
        abs(incidence) @ kept_nodes.astype(np.int8)
    ).reshape(-1)
    active_branches = driven_endpoint_count == 2
    reduced_cpu = active_incidence = incidence[active_branches][:, keep]

    total_started = prepare_started = time.perf_counter()
    reduced = csp.csr_matrix(reduced_cpu.astype(complex_dtype))
    injected_device = cp.asarray(injected, dtype=complex_dtype)
    resistance_device = cp.asarray(resistance[active_branches], dtype=real_dtype)
    active_device = cp.asarray(active_branches)
    kernels, kernels_z = _device_spectra(operator, cp, complex_dtype)
    levels = mesh.vertical_levels
    has_vertical = bool(levels)

    rows, cols = mesh.shape
    layers = len(mesh.stackup)
    x_count = len(mesh.branch_x)
    y_count = len(mesh.branch_y)

    def axis_index(values):
        return tuple(
            cp.asarray([item[index] for item in values], dtype=cp.int32)
            for index in range(3)
        )

    x_layer, x_row, x_col = axis_index(mesh.branch_x)
    y_layer, y_row, y_col = axis_index(mesh.branch_y)
    level_of = {pair: index for index, pair in enumerate(levels)}
    z_level = cp.asarray(
        [level_of[(via.lower_layer, via.upper_layer)] for via in mesh.via_branches],
        dtype=cp.int32,
    )
    z_row = cp.asarray([via.row for via in mesh.via_branches], dtype=cp.int32)
    z_col = cp.asarray([via.col for via in mesh.via_branches], dtype=cp.int32)

    branches = int(active_branches.sum())
    unknowns = int(keep.sum())
    size = branches + unknowns

    def kernel(axis: str, first: int, second: int):
        key = (axis, first, second) if first <= second else (axis, second, first)
        return kernels[key]

    def kernel_z(first: int, second: int):
        key = (first, second) if first <= second else (second, first)
        return kernels_z[key]

    def convolve(spectra, count: int, table):
        out = cp.empty((count, rows, cols), dtype=real_dtype)
        for target in range(count):
            accumulated = cp.zeros_like(spectra[0])
            for source in range(count):
                accumulated += table(target, source) * spectra[source]
            product = cp.fft.irfft2(accumulated, s=operator.padded)
            out[target] = product[:rows, :cols]
        return out

    def flux(currents):
        full = cp.zeros(mesh.branch_count, dtype=real_dtype)
        full[active_device] = currents
        grid_x = cp.zeros((layers, rows, cols), dtype=real_dtype)
        grid_y = cp.zeros_like(grid_x)
        grid_x[x_layer, x_row, x_col] = full[:x_count]
        grid_y[y_layer, y_row, y_col] = full[x_count : x_count + y_count]
        spectra_x = cp.fft.rfft2(grid_x, s=operator.padded, axes=(-2, -1))
        spectra_y = cp.fft.rfft2(grid_y, s=operator.padded, axes=(-2, -1))
        flux_x = convolve(spectra_x, layers, lambda a, b: kernel("x", a, b))
        flux_y = convolve(spectra_y, layers, lambda a, b: kernel("y", a, b))
        out = cp.zeros(mesh.branch_count, dtype=real_dtype)
        out[:x_count] = flux_x[x_layer, x_row, x_col]
        out[x_count : x_count + y_count] = flux_y[y_layer, y_row, y_col]
        if has_vertical:
            grid_z = cp.zeros((len(levels), rows, cols), dtype=real_dtype)
            grid_z[z_level, z_row, z_col] = full[x_count + y_count :]
            spectra_z = cp.fft.rfft2(grid_z, s=operator.padded, axes=(-2, -1))
            flux_z = convolve(spectra_z, len(levels), kernel_z)
            out[x_count + y_count :] = flux_z[z_level, z_row, z_col]
        return out[active_device]

    def impedance(currents):
        linked = flux(currents.real).astype(complex_dtype)
        linked += 1j * flux(currents.imag)
        return resistance_device * currents + complex_dtype(1j) * omega * linked

    def saddle(vector):
        return cp.concatenate(
            [
                impedance(vector[:branches]) - reduced @ vector[branches:],
                reduced.T @ vector[:branches],
            ]
        )

    inline_count = x_count + y_count
    full_diagonal = resistance + 1j * omega * np.concatenate(
        [
            np.full(inline_count, _self_inductance(operator)),
            _vertical_self_inductance(mesh, operator),
        ]
    )
    diagonal = cp.asarray(full_diagonal[active_branches], dtype=complex_dtype)
    schur = (reduced.T @ csp.diags(1.0 / diagonal) @ reduced).tocsc()
    factor_started = time.perf_counter()
    try:
        factored = csl.splu(schur)
        preconditioner_dtype = str(schur.dtype)
    except (TypeError, ValueError, RuntimeError):
        # SuperLU may decline the narrower type.  Say so rather than silently
        # measuring a mixed-precision solve and reporting it as FP32.
        factored = csl.splu(schur.astype(np.complex128))
        preconditioner_dtype = "complex128 (fallback)"
    cp.cuda.get_current_stream().synchronize()
    factor_ms = (time.perf_counter() - factor_started) * 1000.0
    prepare_ms = (time.perf_counter() - prepare_started) * 1000.0

    def precondition(vector):
        rhs_current = vector[:branches]
        rhs_node = vector[branches:]
        node = factored.solve(
            (rhs_node + reduced.T @ (rhs_current / diagonal)).astype(
                factored.L.dtype if hasattr(factored, "L") else complex_dtype
            )
        ).astype(complex_dtype)
        current = (rhs_current + reduced @ node) / diagonal
        return cp.concatenate([current, node])

    right_hand_side = cp.concatenate(
        [cp.zeros(branches, dtype=complex_dtype), injected_device[keep]]
    )

    def left_preconditioned(vector):
        return precondition(saddle(vector))

    linear_system = csl.LinearOperator(
        (size, size), matvec=left_preconditioned, dtype=complex_dtype
    )
    counter = {"restarts": 0}

    solve_started = time.perf_counter()
    result, info = csl.gmres(
        linear_system,
        precondition(right_hand_side),
        rtol=rtol,
        restart=restart,
        maxiter=max_iterations * min(restart, size),
        callback=lambda _value: counter.__setitem__(
            "restarts", counter["restarts"] + 1
        ),
        callback_type="pr_norm",
    )
    working_precision_residual = float(
        (
            cp.linalg.norm(saddle(result) - right_hand_side)
            / cp.maximum(cp.linalg.norm(right_hand_side), cp.asarray(1e-30))
        ).get()
    )
    cp.cuda.get_current_stream().synchronize()
    solve_ms = (time.perf_counter() - solve_started) * 1000.0

    voltage = np.zeros(mesh.node_count, dtype=np.complex128)
    voltage[keep] = cp.asnumpy(result[branches:]).astype(np.complex128)
    current = np.zeros(mesh.branch_count, dtype=np.complex128)
    current[active_branches] = cp.asnumpy(result[:branches]).astype(np.complex128)
    total_ms = (time.perf_counter() - total_started) * 1000.0
    return SolveOutcome(
        precision=precision,
        converged=info == 0,
        working_precision_mixed_unit_residual=working_precision_residual,
        requested_rtol=rtol,
        iterations=counter["restarts"] * min(restart, size),
        prepare_ms=prepare_ms,
        factor_ms=factor_ms,
        solve_ms=solve_ms,
        total_ms=total_ms,
        voltage=voltage,
        current=current,
        preconditioner_dtype=preconditioner_dtype,
    )


def fp64_residuals(
    fixture: Fixture,
    voltage: np.ndarray,
    current: np.ndarray,
) -> dict[str, float]:
    """Re-evaluate KVL and KCL separately in the host FP64 system.

    The two blocks have different units, so they are normalized independently.
    Replaying them through the host operator also prevents an FP32 solve from
    grading its own residual with the arithmetic under test.
    """
    mesh = fixture.mesh
    incidence = mesh.incidence()
    injected = _source_vector(mesh, fixture.terminals)
    components, labels = _components(incidence, mesh.node_count)
    carried = _components_with_terminals(labels, components, injected)
    kept_nodes = np.isin(labels, list(carried))
    keep = kept_nodes.copy()
    for component in sorted(carried):
        keep[int(np.flatnonzero(labels == component)[0])] = False
    driven_endpoint_count = np.asarray(
        abs(incidence) @ kept_nodes.astype(np.int8)
    ).reshape(-1)
    active_branches = driven_endpoint_count == 2
    reduced = incidence[active_branches][:, keep]

    active_current = np.asarray(
        current[active_branches], dtype=np.complex128
    )
    linked = _host_branch_flux(fixture, current.real).astype(np.complex128)
    linked += 1j * _host_branch_flux(fixture, current.imag)
    impedance_drop = (
        mesh.resistances()[active_branches] * active_current
        + 1j
        * (2.0 * math.pi * fixture.frequency_hz)
        * linked[active_branches]
    )
    potential_drop = reduced @ np.asarray(voltage[keep], dtype=np.complex128)
    kvl_absolute = float(np.linalg.norm(impedance_drop - potential_drop))
    kvl_scale = max(
        float(np.linalg.norm(impedance_drop)),
        float(np.linalg.norm(potential_drop)),
        1e-30,
    )

    node_balance = reduced.T @ active_current
    source = injected[keep]
    kcl_absolute = float(np.linalg.norm(node_balance - source))
    kcl_scale = max(float(np.linalg.norm(source)), 1e-30)
    kvl_relative = kvl_absolute / kvl_scale
    kcl_relative = kcl_absolute / kcl_scale
    return {
        "fp64_kvl_absolute_residual_v": kvl_absolute,
        "fp64_kvl_relative_residual": kvl_relative,
        "fp64_kcl_absolute_residual_a": kcl_absolute,
        "fp64_kcl_relative_residual": kcl_relative,
        "fp64_max_block_relative_residual": max(kvl_relative, kcl_relative),
    }


_GATE_METRICS = (
    "bulk_p99_current_density_a_per_mm2",
    "max_current_density_a_per_mm2",
    "voltage_span_v",
    "max_vertical_connection_current_a",
    "i2r_loss_w",
)


def _relative_error(obtained: float, reference: float) -> float:
    return abs(float(obtained) - float(reference)) / max(abs(float(reference)), 1e-30)


def compare(
    fixture: Fixture,
    reference: SheetSolution,
    outcome: SolveOutcome,
) -> dict[str, Any]:
    reference_voltage = reference.node_voltage
    reference_current = reference.branch_current
    voltage_scale = float(np.max(np.abs(reference_voltage)))
    current_scale = float(np.max(np.abs(reference_current)))
    reference_z = terminal_impedance(fixture, reference_voltage)
    obtained_z = terminal_impedance(fixture, outcome.voltage)
    reference_l = loop_inductance_h(fixture, reference_voltage)
    obtained_l = loop_inductance_h(fixture, outcome.voltage)
    replay = fp64_residuals(fixture, outcome.voltage, outcome.current)
    solution = SheetSolution(
        node_voltage=outcome.voltage,
        branch_current=outcome.current,
        frequency_hz=fixture.frequency_hz,
        iterations=outcome.iterations,
        residual=replay["fp64_max_block_relative_residual"],
        converged=outcome.converged,
        grounded_node=reference.grounded_node,
        undriven_nodes=reference.undriven_nodes,
    )
    reference_metrics = sheet_fields(
        fixture.mesh, reference, fixture.terminals
    ).metrics
    obtained_metrics = sheet_fields(
        fixture.mesh, solution, fixture.terminals
    ).metrics
    gate_errors = {
        metric: _relative_error(obtained_metrics[metric], reference_metrics[metric])
        for metric in _GATE_METRICS
    }
    result = {
        "precision": outcome.precision,
        "requested_rtol": outcome.requested_rtol,
        "gmres_reported_convergence": outcome.converged,
        "working_precision_mixed_unit_residual": (
            outcome.working_precision_mixed_unit_residual
        ),
        "iterations": outcome.iterations,
        "prepare_ms": outcome.prepare_ms,
        "factor_ms": outcome.factor_ms,
        "solve_ms": outcome.solve_ms,
        "total_ms": outcome.total_ms,
        "preconditioner_dtype": outcome.preconditioner_dtype,
        "max_voltage_error_v": float(
            np.max(np.abs(outcome.voltage - reference_voltage))
        ),
        "relative_voltage_error": float(
            np.max(np.abs(outcome.voltage - reference_voltage)) / voltage_scale
        ),
        "relative_current_error": float(
            np.max(np.abs(outcome.current - reference_current)) / current_scale
        ),
        "resistance_ohm": obtained_z.real,
        "reference_resistance_ohm": reference_z.real,
        "relative_resistance_error": _relative_error(
            obtained_z.real, reference_z.real
        ),
        "loop_inductance_h": obtained_l,
        "reference_loop_inductance_h": reference_l,
        "relative_inductance_error": _relative_error(obtained_l, reference_l),
        "gate_metric_relative_errors": gate_errors,
        "max_gate_metric_relative_error": max(gate_errors.values(), default=0.0),
    }
    result.update(replay)
    result["fp64_residual_meets_requested_rtol"] = (
        replay["fp64_max_block_relative_residual"] <= outcome.requested_rtol
    )
    return result


def stage_c(
    fixture: Fixture,
    cp: Any,
    reference: Any,
    fp32_tolerances: tuple[float, ...],
    *,
    max_iterations: int,
) -> dict[str, Any]:
    """Solve in FP64 and at a range of FP32 targets, against the CPU answer.

    The CPU solve is the reference because it is the path the repository's
    existing acceptance already holds to independent references.  FP64 on the
    device is reported against it too, so the FP32 numbers can be read as a
    departure from a device path that is itself known to agree.
    """
    if not reference.converged:
        raise RuntimeError(
            f"CPU reference did not converge: residual={reference.residual:.3e}"
        )
    results = []
    double = solve_on_device(
        fixture,
        cp,
        precision="fp64",
        rtol=1e-13,
        max_iterations=max_iterations,
    )
    results.append(compare(fixture, reference, double))
    for rtol in fp32_tolerances:
        single = solve_on_device(
            fixture,
            cp,
            precision="fp32",
            rtol=rtol,
            max_iterations=max_iterations,
        )
        results.append(compare(fixture, reference, single))
        # Also state the FP32 departure from the device FP64 solve, which
        # isolates the precision from any difference the reference carries.
        results[-1]["relative_inductance_error_vs_device_fp64"] = abs(
            loop_inductance_h(fixture, single.voltage)
            - loop_inductance_h(fixture, double.voltage)
        ) / abs(loop_inductance_h(fixture, double.voltage))
    return {
        "reference": "CPU float64 solve_sheet_case",
        "reference_residual": reference.residual,
        "reference_converged": bool(reference.converged),
        "maximum_restart_cycles": max_iterations,
        "runs": results,
    }


# --------------------------------------------------------------------------


def _import_cupy() -> Any:
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise SystemExit("no CUDA device; stages B and C need one")
    return cp


def _device_identity(cp: Any) -> dict[str, Any]:
    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties["name"]
    return {
        "device": name.decode() if isinstance(name, bytes) else str(name),
        "compute_capability": f"{properties['major']}.{properties['minor']}",
        "cupy_version": cp.__version__,
        "driver_version": cp.cuda.runtime.driverGetVersion(),
        "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=48)
    parser.add_argument("--frequency", type=float, default=3e5)
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=20,
        help="maximum GMRES restart cycles per device solve (default: 20)",
    )
    parser.add_argument(
        "--stage", choices=("a", "b", "c", "all"), default="all"
    )
    parser.add_argument("--json", type=str, default="")
    arguments = parser.parse_args()
    if arguments.cells < 4:
        parser.error("--cells must be at least 4")
    if arguments.frequency <= 0.0 or not math.isfinite(arguments.frequency):
        parser.error("--frequency must be finite and positive")
    if arguments.max_restarts < 1:
        parser.error("--max-restarts must be positive")

    report: dict[str, Any] = {}
    if arguments.stage in ("a", "all"):
        report["stage_a_kernel"] = stage_a(0.2e-3, 35e-6)
        rows = report["stage_a_kernel"]["rows"]
        print("Stage A -- closed-form kernel, 0.2mm cell, along the current")
        print(
            f"{'offset':>7} {'retained fp64':>14} {'retained fp32':>14} "
            f"{'rel err fp64':>13} {'rel err fp32':>13} {'far-field':>11}"
        )
        for row in rows:
            print(
                f"{row['offset_cells']:>7} {row['retained_fp64']:>14.2e} "
                f"{row['retained_fp32']:>14.2e} "
                f"{row['relative_error_fp64']:>13.2e} "
                f"{row['relative_error_fp32']:>13.2e} "
                f"{row['relative_error_far_field_limit']:>11.2e}"
            )

    if arguments.stage in ("b", "c", "all"):
        cp = _import_cupy()
        report["device"] = _device_identity(cp)
        print(f"\nDevice: {report['device']['device']}")
        fixture = build_fixture(arguments.cells, arguments.frequency)
        report["fixture"] = fixture.description
        print(f"Fixture: {json.dumps(fixture.description)}")

        started = time.perf_counter()
        reference = solve_sheet_case(
            fixture.mesh,
            fixture.operator,
            fixture.terminals,
            frequency_hz=fixture.frequency_hz,
            tolerance=1e-11,
            max_iterations=400,
        )
        cpu_ms = (time.perf_counter() - started) * 1000.0
        report["cpu_reference_ms"] = cpu_ms
        print(
            f"CPU reference: residual {reference.residual:.2e}, "
            f"converged {reference.converged}, {cpu_ms:.0f} ms"
        )

        if arguments.stage in ("b", "all"):
            report["stage_b_operator"] = stage_b(
                fixture, reference.branch_current, cp
            )
            print("\nStage B -- operator apply on the device")
            for name in ("fp64", "fp32"):
                entry = report["stage_b_operator"][name]
                print(
                    f"  {name}: relative L2 {entry['relative_l2_error']:.2e}, "
                    f"peak-relative {entry['max_relative_to_peak']:.2e}, "
                    f"{entry['complex_apply_pair_ms']:.1f} ms"
                )

        if arguments.stage in ("c", "all"):
            report["stage_c_solve"] = stage_c(
                fixture,
                cp,
                reference,
                (1e-4, 1e-5, 1e-6),
                max_iterations=arguments.max_restarts,
            )
            print("\nStage C -- full solve on the device")
            print(
                f"{'precision':>9} {'rtol':>8} {'KVL res':>10} {'KCL res':>10} "
                f"{'iters':>6} {'gate err':>9} {'R err':>9} {'L err':>9} "
                f"{'total ms':>10}"
            )
            for run in report["stage_c_solve"]["runs"]:
                print(
                    f"{run['precision']:>9} {run['requested_rtol']:>8.0e} "
                    f"{run['fp64_kvl_relative_residual']:>10.2e} "
                    f"{run['fp64_kcl_relative_residual']:>10.2e} "
                    f"{run['iterations']:>6} "
                    f"{run['max_gate_metric_relative_error']:>9.2e} "
                    f"{run['relative_resistance_error']:>9.2e} "
                    f"{run['relative_inductance_error']:>9.2e} "
                    f"{run['total_ms']:>10.0f}"
                )

    if arguments.json:
        with open(arguments.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nWrote {arguments.json}")


if __name__ == "__main__":
    main()
