"""Router-facing sparse operations for multilayer DICE-PEEC.

plane_opt (or any router) submits segment/via edits against a fixed stackup.
This module compiles them into occupancy deltas and via sets that the 2.5D
scorer understands.  Geometry DRC remains the router's responsibility (Gate 0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral
from typing import Iterable, Literal, Sequence

from .multilayer_peec import SparseDeltaML, ViaSet, ViaSpec
from .stackup import Stackup


Action = Literal["add", "remove"]


@dataclass(frozen=True)
class SegmentOp:
    """Add or remove copper occupancy along one layer."""

    action: Action
    layer: str | int
    cells: tuple[tuple[int, int], ...]  # (row, col)
    value: float = 1.0

    def __post_init__(self) -> None:
        if self.action not in {"add", "remove"}:
            raise ValueError("action must be add or remove")
        value = float(self.value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("segment value must be finite and non-negative")
        cells: list[tuple[int, int]] = []
        for cell in self.cells:
            if len(cell) != 2:
                raise ValueError("segment cells must be (row, col) pairs")
            row, col = cell
            if (
                isinstance(row, bool)
                or isinstance(col, bool)
                or not isinstance(row, Integral)
                or not isinstance(col, Integral)
            ):
                raise TypeError("segment row and col must be integers")
            if row < 0 or col < 0:
                raise ValueError("segment row and col must be non-negative")
            cells.append((int(row), int(col)))
        object.__setattr__(self, "cells", tuple(cells))
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class ViaOp:
    """Add or remove a via between two layers at one cell."""

    action: Action
    row: int
    col: int
    layer_from: str | int
    layer_to: str | int
    resistance_ohm: float = 0.0
    inductance_h: float = 1e-9
    important: bool = False
    mutual_weight: float = 1.0
    # When true, also place/remove unit occupancy on both end layers so the
    # via pads participate in the sheet interaction kernel.
    touch_pads: bool = True

    def __post_init__(self) -> None:
        if self.action not in {"add", "remove"}:
            raise ValueError("action must be add or remove")
        for name in ("row", "col"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"via {name} must be an integer")
            if value < 0:
                raise ValueError(f"via {name} must be non-negative")
            object.__setattr__(self, name, int(value))
        for name in ("resistance_ohm", "inductance_h", "mutual_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "important", bool(self.important))
        object.__setattr__(self, "touch_pads", bool(self.touch_pads))


@dataclass
class CandidateEdit:
    """One routing candidate as a bag of sparse operations."""

    segments: list[SegmentOp] = field(default_factory=list)
    vias: list[ViaOp] = field(default_factory=list)

    def extend(self, other: "CandidateEdit") -> "CandidateEdit":
        return CandidateEdit(
            segments=[*self.segments, *other.segments],
            vias=[*self.vias, *other.vias],
        )


@dataclass(frozen=True)
class CompiledCandidate:
    """Occupancy delta plus absolute candidate via set."""

    occupancy_delta: SparseDeltaML
    vias: ViaSet
    high_risk_topology: bool


def _sign(action: Action) -> float:
    return 1.0 if action == "add" else -1.0


def apply_via_ops(base_vias: ViaSet, ops: Sequence[ViaOp], stackup: Stackup) -> ViaSet:
    """Apply add/remove via ops onto a base via set (absolute result)."""
    by_key: dict[tuple[int, int, int, int], ViaSpec] = {
        via.normalized(): via for via in base_vias.vias
    }
    for op in ops:
        layer_from = stackup.index(op.layer_from)
        layer_to = stackup.index(op.layer_to)
        if layer_from == layer_to:
            raise ValueError("via op layers must differ")
        spec = ViaSpec(
            row=int(op.row),
            col=int(op.col),
            layer_from=layer_from,
            layer_to=layer_to,
            resistance_ohm=float(op.resistance_ohm),
            inductance_h=float(op.inductance_h),
            important=bool(op.important),
            mutual_weight=float(op.mutual_weight),
        )
        key = spec.normalized()
        if op.action == "add":
            previous = by_key.get(key)
            if previous is None:
                by_key[key] = spec
            else:
                # A physical via is a set element, not an additive impedance.
                # Identical adds are idempotent; conflicting properties are
                # rejected by ViaSet instead of being multiplied together.
                by_key[key] = ViaSet.from_iterable((previous, spec)).vias[0]
        else:
            by_key.pop(key, None)
    return ViaSet(vias=tuple(by_key.values()))


def compile_candidate(
    edit: CandidateEdit,
    stackup: Stackup,
    *,
    base_vias: ViaSet | None = None,
) -> CompiledCandidate:
    """Compile router ops into a scorer-ready occupancy delta and via set."""
    base_vias = base_vias or ViaSet()
    changes: dict[tuple[int, int, int], float] = {}
    high_risk = False

    def record_change(layer: int, row: int, col: int, value: float) -> None:
        if value == 0.0:
            return
        key = (layer, row, col)
        previous = changes.get(key)
        if previous is None:
            changes[key] = value
        elif previous != value:
            raise ValueError(
                f"conflicting occupancy edits at layer/row/col {key}: "
                f"{previous} and {value}"
            )
        # Repeated identical geometry (for example a segment ending on a via
        # pad) is a union and therefore contributes only once.

    for segment in edit.segments:
        layer = stackup.index(segment.layer)
        signed = _sign(segment.action) * float(segment.value)
        for row, col in segment.cells:
            record_change(layer, int(row), int(col), signed)

    for via in edit.vias:
        # Every layer transition edit changes topology, including removal of
        # an ordinary via.
        high_risk = True
        if via.touch_pads:
            layer_from = stackup.index(via.layer_from)
            layer_to = stackup.index(via.layer_to)
            signed = _sign(via.action)
            record_change(layer_from, int(via.row), int(via.col), signed)
            record_change(layer_to, int(via.row), int(via.col), signed)

    occupancy = SparseDeltaML.from_changes(
        (*key, value) for key, value in changes.items()
    )
    vias = apply_via_ops(base_vias, edit.vias, stackup)
    return CompiledCandidate(
        occupancy_delta=occupancy,
        vias=vias,
        high_risk_topology=high_risk,
    )


def compile_many(
    edits: Iterable[CandidateEdit],
    stackup: Stackup,
    *,
    base_vias: ViaSet | None = None,
) -> list[CompiledCandidate]:
    return [
        compile_candidate(edit, stackup, base_vias=base_vias) for edit in edits
    ]
