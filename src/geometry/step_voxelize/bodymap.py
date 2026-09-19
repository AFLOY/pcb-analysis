"""The body map: what each STEP solid is, supplied by the caller.

Mechanical CAD does not say which solid is the board, which are copper, and
which is a heat sink; exporters disagree on names and colours.  The caller
therefore maps solid names (regular expressions, matched in full) to roles.
Every solid of the model must be claimed by exactly one entry or listed in
``ignore``; anything else is an error rather than a guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from thermal.matrix_free_mpir_fem import VoxelMaterial

from .reader import StepModel, StepSolid


@dataclass(frozen=True)
class LayerSpec:
    """One conductor layer of the board stackup."""

    name: str
    center_z_mm: float
    thickness_mm: float
    resistivity_ohm_m: float = 1.68e-8

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("layer name must not be empty")
        if not np.isfinite(self.center_z_mm):
            raise ValueError(f"{self.name}: center_z_mm must be finite")
        if self.thickness_mm <= 0.0 or not np.isfinite(self.thickness_mm):
            raise ValueError(f"{self.name}: thickness_mm must be positive")
        if self.resistivity_ohm_m <= 0.0 or not np.isfinite(self.resistivity_ohm_m):
            raise ValueError(f"{self.name}: resistivity_ohm_m must be positive")

    @property
    def bottom_z_mm(self) -> float:
        return self.center_z_mm - self.thickness_mm / 2.0

    @property
    def top_z_mm(self) -> float:
        return self.center_z_mm + self.thickness_mm / 2.0


@dataclass(frozen=True)
class BoardSpec:
    """The board solid and its conductor stackup (bottom layer first)."""

    solid: str
    layers: tuple[LayerSpec, ...]
    copper_conductivity_w_per_m_k: float = 385.0
    laminate_conductivity_w_per_m_k: float = 0.8
    laminate_through_conductivity_w_per_m_k: float = 0.3

    def __post_init__(self) -> None:
        layers = tuple(self.layers)
        if not layers:
            raise ValueError("a board needs at least one layer")
        z = [layer.center_z_mm for layer in layers]
        if z != sorted(z) or len(set(z)) != len(z):
            raise ValueError("layers must be ordered bottom to top with distinct z")
        names = [layer.name for layer in layers]
        if len(set(names)) != len(names):
            raise ValueError("layer names must be unique")
        for lower, upper in zip(layers, layers[1:]):
            if upper.bottom_z_mm < lower.top_z_mm - 1.0e-9:
                raise ValueError(f"layers {lower.name} and {upper.name} overlap")
        for value in (
            self.copper_conductivity_w_per_m_k,
            self.laminate_conductivity_w_per_m_k,
            self.laminate_through_conductivity_w_per_m_k,
        ):
            if value <= 0.0 or not np.isfinite(value):
                raise ValueError("board conductivities must be positive")
        object.__setattr__(self, "layers", layers)

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(layer.name for layer in self.layers)

    def layer(self, name: str) -> LayerSpec:
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise KeyError(f"unknown layer {name!r}")


@dataclass(frozen=True)
class CopperSpec:
    """Copper solids (pattern) that belong to one layer."""

    solids: str
    layer: str


@dataclass(frozen=True)
class ViaSpec:
    """Via barrels (pattern) joining two layers; the layers come from the z extent."""

    solids: str
    resistance_ohm: float = 0.0
    inductance_h: float = 1.0e-9


@dataclass(frozen=True)
class ContactSpec:
    """How a body touches the board: which side and the joint conductance."""

    board_side: str = "top"
    conductance_per_area_w_per_m2_k: float = 1.0e4

    def __post_init__(self) -> None:
        if self.board_side not in ("top", "bottom"):
            raise ValueError("board_side must be 'top' or 'bottom'")
        if self.conductance_per_area_w_per_m2_k <= 0.0:
            raise ValueError("conductance per area must be positive")

    @classmethod
    def from_interface_material(
        cls, *, conductivity_w_per_m_k: float, thickness_m: float, board_side: str = "top"
    ) -> "ContactSpec":
        return cls(board_side, conductivity_w_per_m_k / thickness_m)


@dataclass(frozen=True)
class BodySpec:
    """A 3D body: heat sink, enclosure, component package, standoff."""

    solids: str
    material: VoxelMaterial
    power_w: float = 0.0
    contact: ContactSpec | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.power_w) or self.power_w < 0.0:
            raise ValueError("power_w must be finite and non-negative")


@dataclass(frozen=True)
class BodyMap:
    board: BoardSpec
    copper: tuple[CopperSpec, ...] = ()
    vias: tuple[ViaSpec, ...] = ()
    bodies: tuple[BodySpec, ...] = ()
    ignore: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "copper", tuple(self.copper))
        object.__setattr__(self, "vias", tuple(self.vias))
        object.__setattr__(self, "bodies", tuple(self.bodies))
        object.__setattr__(self, "ignore", tuple(self.ignore))
        for spec in self.copper:
            self.board.layer(spec.layer)
        for pattern in (
            [self.board.solid]
            + [spec.solids for spec in self.copper]
            + [spec.solids for spec in self.vias]
            + [spec.solids for spec in self.bodies]
            + list(self.ignore)
        ):
            re.compile(pattern)


@dataclass(frozen=True)
class ResolvedBodies:
    """The model's solids sorted into the body map's roles."""

    board: StepSolid
    copper: tuple[tuple[CopperSpec, tuple[StepSolid, ...]], ...]
    vias: tuple[tuple[ViaSpec, tuple[StepSolid, ...]], ...]
    bodies: tuple[tuple[BodySpec, tuple[StepSolid, ...]], ...]
    ignored: tuple[StepSolid, ...] = field(default=())


def resolve_bodies(model: StepModel, body_map: BodyMap) -> ResolvedBodies:
    """Assign every solid to exactly one entry of the body map."""

    claimed: dict[str, str] = {}

    def claim(solids: Sequence[StepSolid], owner: str) -> None:
        for solid in solids:
            previous = claimed.get(solid.name)
            if previous is not None and previous != owner:
                raise ValueError(f"solid {solid.name!r} is claimed by both {previous} and {owner}")
            claimed[solid.name] = owner

    boards = model.matching(body_map.board.solid)
    if len(boards) != 1:
        raise ValueError(
            f"board pattern {body_map.board.solid!r} matches {len(boards)} solids, expected one"
        )
    claim(boards, "board")
    copper = []
    for index, spec in enumerate(body_map.copper):
        solids = model.matching(spec.solids)
        claim(solids, f"copper[{index}]:{spec.layer}")
        copper.append((spec, solids))
    vias = []
    for index, spec in enumerate(body_map.vias):
        solids = model.matching(spec.solids)
        claim(solids, f"vias[{index}]:{spec.solids}")
        vias.append((spec, solids))
    bodies = []
    for index, spec in enumerate(body_map.bodies):
        solids = model.matching(spec.solids)
        if not solids:
            raise ValueError(f"body pattern {spec.solids!r} matches no solid")
        claim(solids, f"bodies[{index}]:{spec.solids}")
        bodies.append((spec, solids))
    ignored = []
    for pattern in body_map.ignore:
        solids = model.matching(pattern)
        claim(solids, "ignore")
        ignored.extend(solids)
    unclaimed = [solid.name for solid in model.solids if solid.name not in claimed]
    if unclaimed:
        raise ValueError(f"solids not covered by the body map: {unclaimed}")
    return ResolvedBodies(
        board=boards[0],
        copper=tuple(copper),
        vias=tuple(vias),
        bodies=tuple(bodies),
        ignored=tuple(ignored),
    )


__all__ = [
    "BoardSpec",
    "BodyMap",
    "BodySpec",
    "ContactSpec",
    "CopperSpec",
    "LayerSpec",
    "ResolvedBodies",
    "ViaSpec",
    "resolve_bodies",
]
