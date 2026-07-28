"""Predict what a PyPEEC voxel solve will ask the device for, before it asks.

The CUDA executor used to learn the size of a problem by running it: it set a
pool limit from the free VRAM and read the growth afterwards.  That is adequate
while every model is one copper layer thick.  It stops being adequate the
moment a model spans a board's height, because PyPEEC's coupling operators live
on a circulant embedding of the *bounding box*, and a two-sided board's box is
tens of times taller than its copper.  A model that cannot fit should say so
before it allocates, not after.

The numbers here come from PyPEEC 5.8.0's ``lib_matrix/multiply_fft.py``, which
embeds each axis at exactly twice its length, stores the prepared tensors as
``complex128``, and reports its own footprint as::

    nnz = (2 * nx) * (2 * ny) * (2 * nz) * nd_in
    footprint = NPCP.dtype(NPCP.complex128).itemsize * nnz

This module reproduces that expression and adds the terms PyPEEC does not log:
the per-product temporaries, the Krylov basis, and the index arrays.  It cannot
account for cuFFT's own plan workspaces, which are allocated outside CuPy's
pool; that share is carried as an explicit, named factor rather than folded in
silently.

Every quantity is a prediction.  It is meant to refuse an impossible solve, not
to report a measurement, and :attr:`MemoryEstimate.assumptions` says what it
took for granted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# PyPEEC 5.8.0 allocates its FFT tensors and solution vectors as complex128 and
# offers no way to ask for anything narrower.
COMPLEX_BYTES = 16

# Channels each prepared operator carries, as nd_in in PyPEEC's own footprint
# expression.  The potential operator is scalar; the inductance operator holds
# one tensor per voxel face direction; the electric-magnetic coupling operator
# holds three more and exists only when the problem defines a magnetic domain.
POTENTIAL_CHANNELS = 1
INDUCTANCE_CHANNELS = 3
COUPLING_CHANNELS = 3

# cuFFT plan workspaces and CuPy's plan cache sit outside the memory pool this
# executor caps, so they cannot be counted.  This is the fraction of the
# countable total held back for them.  It is a policy number, not a
# measurement: raise it if a device with a measured headroom still runs out.
DEFAULT_UNMEASURED_FRACTION = 0.35


@dataclass(frozen=True)
class MemoryEstimate:
    """What one PyPEEC solve is predicted to need on the device."""

    box: tuple[int, int, int]
    voxel_count: int
    embedded_count: int
    conductive_count: int
    unknown_count: int
    channels: dict[str, int]
    fft_tensor_bytes: int
    product_workspace_bytes: int
    krylov_bytes: int
    index_bytes: int
    unmeasured_fraction: float
    assumptions: tuple[str, ...] = ()

    @property
    def countable_bytes(self) -> int:
        """Sum the terms this module can derive from the problem statement."""
        return (
            self.fft_tensor_bytes
            + self.product_workspace_bytes
            + self.krylov_bytes
            + self.index_bytes
        )

    @property
    def required_bytes(self) -> int:
        """Give the countable total plus the share held back for cuFFT."""
        return int(math.ceil(self.countable_bytes * (1.0 + self.unmeasured_fraction)))

    def as_metrics(self) -> dict[str, Any]:
        return {
            "estimated_box": list(self.box),
            "estimated_voxel_count": self.voxel_count,
            "estimated_embedded_count": self.embedded_count,
            "estimated_conductive_count": self.conductive_count,
            "estimated_unknown_count": self.unknown_count,
            "estimated_fft_tensor_bytes": self.fft_tensor_bytes,
            "estimated_product_workspace_bytes": self.product_workspace_bytes,
            "estimated_krylov_bytes": self.krylov_bytes,
            "estimated_index_bytes": self.index_bytes,
            "estimated_countable_bytes": self.countable_bytes,
            "estimated_required_bytes": self.required_bytes,
            "estimated_unmeasured_fraction": self.unmeasured_fraction,
            "estimate_assumptions": list(self.assumptions),
        }


def _voxel_box(geometry: Mapping[str, Any]) -> tuple[int, int, int]:
    param = geometry["data_voxelize"]["param"]
    box = tuple(int(value) for value in param["n"])
    if len(box) != 3 or any(value < 1 for value in box):
        raise ValueError(f"voxel count must be three positive integers; got {box}")
    return box  # type: ignore[return-value]


def _conductive_count(geometry: Mapping[str, Any]) -> int:
    """Count the voxels the geometry actually assigns to a domain.

    A tall model is mostly laminate: the box says how large the operators are,
    and this says how large the solution vectors are.  The two diverge by more
    than an order of magnitude on a two-sided board, which is the whole reason
    an estimate is worth making rather than scaling off the box alone.
    """
    domains = geometry["data_voxelize"].get("domain_index") or {}
    seen: set[int] = set()
    for indices in domains.values():
        seen.update(int(value) for value in indices)
    return len(seen)


def _has_magnetic_domain(problem: Mapping[str, Any] | None) -> bool:
    for material in (problem or {}).get("material_def", {}).values():
        if str(material.get("material_type", "electric")) != "electric":
            return True
    return False


def _restart_length(tolerance: Mapping[str, Any] | None) -> int:
    """Read how many vectors the Krylov solver keeps before it restarts."""
    options = (tolerance or {}).get("solver_options", {})
    for key in ("direct_options", "segregated_options"):
        inner = options.get(key) or {}
        if "n_inner" in inner:
            return max(1, int(inner["n_inner"]))
        for nested in ("iter_electric_options", "iter_magnetic_options"):
            candidate = inner.get(nested) or {}
            if "n_inner" in candidate:
                return max(1, int(candidate["n_inner"]))
    return 30


def estimate_pypeec_memory(
    geometry: Mapping[str, Any],
    problem: Mapping[str, Any] | None = None,
    tolerance: Mapping[str, Any] | None = None,
    *,
    unmeasured_fraction: float = DEFAULT_UNMEASURED_FRACTION,
    unknowns_per_voxel: int = 4,
) -> MemoryEstimate:
    """Predict the device memory one PyPEEC solve of ``geometry`` will need.

    ``unknowns_per_voxel`` covers the three face currents and the one potential
    a conductive voxel contributes.  It is deliberately an upper bound: a voxel
    on the boundary of the conductor owns fewer faces than one inside it, and
    over-counting here costs a refused solve that would have fit, which is
    recoverable, rather than an accepted solve that does not, which is not.
    """
    if not 0.0 <= unmeasured_fraction < 4.0:
        raise ValueError("unmeasured_fraction must be in [0, 4)")
    if unknowns_per_voxel < 1:
        raise ValueError("unknowns_per_voxel must be positive")

    nx, ny, nz = _voxel_box(geometry)
    voxel_count = nx * ny * nz
    embedded_count = (2 * nx) * (2 * ny) * (2 * nz)
    conductive = _conductive_count(geometry)
    unknowns = conductive * unknowns_per_voxel

    channels = {
        "potential": POTENTIAL_CHANNELS,
        "inductance": INDUCTANCE_CHANNELS,
    }
    assumptions = [
        "PyPEEC 5.8.0 embeds each axis at exactly twice its length",
        "prepared operators and solution vectors are complex128",
        f"{unknowns_per_voxel} unknowns per conductive voxel",
    ]
    if _has_magnetic_domain(problem):
        channels["coupling"] = COUPLING_CHANNELS
        assumptions.append("the problem defines a magnetic domain, so the "
                           "electric-magnetic coupling operator is built")
    else:
        assumptions.append("the problem is electric only, so no coupling "
                           "operator is built")

    total_channels = sum(channels.values())
    fft_tensor_bytes = COMPLEX_BYTES * embedded_count * total_channels

    # One matrix-vector product materialises a tensor over the un-embedded box.
    # `split` trades that for a vector over the unknowns, which is what the
    # option is for; honour it, because on a tall box the difference is the
    # whole reason it exists.
    split = bool(((tolerance or {}).get("dense_options") or {}).get("split", True))
    if split:
        product_workspace_bytes = COMPLEX_BYTES * unknowns * 2
        assumptions.append("dense_options.split is on, so a product allocates "
                           "a vector over the unknowns rather than a tensor "
                           "over the box")
    else:
        product_workspace_bytes = (
            COMPLEX_BYTES * voxel_count * INDUCTANCE_CHANNELS * 2
        )
        assumptions.append("dense_options.split is off, so a product allocates "
                           "a tensor over the whole box")

    restart = _restart_length(tolerance)
    krylov_bytes = COMPLEX_BYTES * unknowns * (restart + 2)
    assumptions.append(f"the Krylov solver keeps {restart} vectors before it "
                       "restarts")

    # Index arrays are int64 over the conductive voxels, once per operator.
    index_bytes = 8 * conductive * max(1, len(channels))

    return MemoryEstimate(
        box=(nx, ny, nz),
        voxel_count=voxel_count,
        embedded_count=embedded_count,
        conductive_count=conductive,
        unknown_count=unknowns,
        channels=channels,
        fft_tensor_bytes=fft_tensor_bytes,
        product_workspace_bytes=product_workspace_bytes,
        krylov_bytes=krylov_bytes,
        index_bytes=index_bytes,
        unmeasured_fraction=float(unmeasured_fraction),
        assumptions=tuple(assumptions),
    )


def describe_estimate(estimate: MemoryEstimate) -> str:
    """Render an estimate for an error message, in megabytes."""

    def mib(value: int) -> str:
        return f"{value / (1024 ** 2):.1f} MiB"

    nx, ny, nz = estimate.box
    return (
        f"box {nx}x{ny}x{nz} ({estimate.voxel_count} voxels, "
        f"{estimate.conductive_count} conductive), embedded "
        f"{2 * nx}x{2 * ny}x{2 * nz}: operators {mib(estimate.fft_tensor_bytes)}, "
        f"products {mib(estimate.product_workspace_bytes)}, "
        f"Krylov {mib(estimate.krylov_bytes)}, "
        f"indices {mib(estimate.index_bytes)}, "
        f"plus {estimate.unmeasured_fraction:.0%} for cuFFT workspaces "
        f"= {mib(estimate.required_bytes)}"
    )
