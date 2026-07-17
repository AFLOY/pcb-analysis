"""Router-facing sparse operations for multilayer DICE-PEEC.

plane_opt (or any router) submits segment/via edits against a fixed stackup.
This module compiles them into occupancy deltas and via sets that the 2.5D
scorer understands.  Geometry DRC remains the router's responsibility (Gate 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
        if not self.cells and self.value != 0.0:
            # empty is allowed; it simply no-ops
            pass


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
                by_key[key] = ViaSpec(
                    row=key[0],
                    col=key[1],
                    layer_from=key[2],
                    layer_to=key[3],
                    resistance_ohm=previous.resistance_ohm + spec.resistance_ohm,
                    inductance_h=previous.inductance_h + spec.inductance_h,
                    important=previous.important or spec.important,
                    mutual_weight=previous.mutual_weight + spec.mutual_weight,
                )
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
    changes: list[tuple[int, int, int, float]] = []
    high_risk = False

    for segment in edit.segments:
        layer = stackup.index(segment.layer)
        signed = _sign(segment.action) * float(segment.value)
        for row, col in segment.cells:
            changes.append((layer, int(row), int(col), signed))

    for via in edit.vias:
        if via.important or via.action == "add":
            # Layer transitions and new vias are treated as topology-risk.
            high_risk = True
        if via.touch_pads:
            layer_from = stackup.index(via.layer_from)
            layer_to = stackup.index(via.layer_to)
            signed = _sign(via.action)
            changes.append((layer_from, int(via.row), int(via.col), signed))
            changes.append((layer_to, int(via.row), int(via.col), signed))

    occupancy = SparseDeltaML.from_changes(changes)
    vias = apply_via_ops(base_vias, edit.vias, stackup)
    if any(via.important for via in vias.vias):
        high_risk = True
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
