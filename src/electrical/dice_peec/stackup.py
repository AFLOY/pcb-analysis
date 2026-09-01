"""PCB stackup description for the 2.5D DICE-PEEC operator.

Layer geometry is fixed for a local optimization epoch.  Horizontal FFTs use
the in-plane grid; interlayer coupling depends only on vertical separation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Stackup:
    """Conductive-sheet stackup with fixed layer order and z centers.

    Parameters
    ----------
    layer_names:
        Ordered layer identifiers (index 0 is the first sheet).
    z_mm:
        Layer center coordinates in millimetres, same order as ``layer_names``.
        Differences drive the interlayer kernel distance.
    """

    layer_names: tuple[str, ...]
    z_mm: tuple[float, ...]

    def __post_init__(self) -> None:
        names = tuple(self.layer_names)
        z_mm = tuple(float(value) for value in self.z_mm)
        if not names:
            raise ValueError("stackup requires at least one layer")
        if len(names) != len(z_mm):
            raise ValueError("layer_names and z_mm must have the same length")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("layer_names must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("layer_names must be unique")
        if not all(math.isfinite(value) for value in z_mm):
            raise ValueError("z_mm must contain only finite coordinates")
        object.__setattr__(self, "layer_names", names)
        object.__setattr__(self, "z_mm", z_mm)

    @classmethod
    def from_pairs(cls, pairs: Sequence[tuple[str, float]]) -> "Stackup":
        names = tuple(name for name, _ in pairs)
        z_mm = tuple(float(z) for _, z in pairs)
        return cls(layer_names=names, z_mm=z_mm)

    @classmethod
    def dual_sided(
        cls,
        *,
        board_thickness_mm: float = 1.6,
        front: str = "F.Cu",
        back: str = "B.Cu",
    ) -> "Stackup":
        """Common two-layer PCB with front at z=0 and back at -thickness."""
        thickness = float(board_thickness_mm)
        if not math.isfinite(thickness) or thickness <= 0.0:
            raise ValueError("board_thickness_mm must be finite and positive")
        return cls(layer_names=(front, back), z_mm=(0.0, -thickness))

    @property
    def n_layers(self) -> int:
        return len(self.layer_names)

    @property
    def z_m(self) -> np.ndarray:
        return np.asarray(self.z_mm, dtype=np.float64) * 1e-3

    def index(self, layer: str | int) -> int:
        if isinstance(layer, Integral) and not isinstance(layer, bool):
            index = int(layer)
            if index < 0 or index >= self.n_layers:
                raise IndexError(f"layer index out of range: {layer}")
            return index
        try:
            return self.layer_names.index(layer)
        except ValueError as exc:
            raise KeyError(f"unknown layer {layer!r}; known={self.layer_names}") from exc

    def resolve_many(self, layers: Iterable[str | int]) -> np.ndarray:
        return np.asarray([self.index(layer) for layer in layers], dtype=np.int32)

    def separation_m(self, layer_a: int, layer_b: int) -> float:
        index_a = self.index(layer_a)
        index_b = self.index(layer_b)
        return float(abs(self.z_m[index_a] - self.z_m[index_b]))

    def separation_cells(
        self, layer_a: int, layer_b: int, *, cell_size_m: float
    ) -> float:
        """Vertical separation expressed in horizontal cell units."""
        if not math.isfinite(cell_size_m) or cell_size_m <= 0.0:
            raise ValueError("cell_size_m must be finite and positive")
        return self.separation_m(layer_a, layer_b) / cell_size_m
