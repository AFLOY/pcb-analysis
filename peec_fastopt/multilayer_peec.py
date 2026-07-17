"""2.5D multilayer interaction operator and exact sparse-delta scoring.

Matches the DICE-PEEC design:

- conductive sheets on a fixed stackup;
- 2D in-plane FFTs with a layer-to-layer kernel matrix;
- candidates as sparse occupancy deltas on ``(layer, row, col)``;
- ordinary vias as frequency-dependent lumped elements;
- important vias retained for local 3D near-field treatment elsewhere.

This is still a scalar magnetoquasistatic interaction proxy, not a full
PEEC-MNA solve.  The identities and data layout are the production bridge
that plane_opt multilayer routing can feed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.fft import irfftn, next_fast_len, rfftn

from .stackup import Stackup


@dataclass(frozen=True)
class SparseDeltaML:
    """Sparse occupancy change on a multilayer grid."""

    layers: np.ndarray  # int32
    rows: np.ndarray  # int32
    cols: np.ndarray  # int32
    values: np.ndarray  # float64

    @classmethod
    def empty(cls) -> "SparseDeltaML":
        return cls(
            layers=np.empty(0, dtype=np.int32),
            rows=np.empty(0, dtype=np.int32),
            cols=np.empty(0, dtype=np.int32),
            values=np.empty(0, dtype=np.float64),
        )

    @classmethod
    def from_changes(
        cls,
        changes: Iterable[tuple[int, int, int, float] | tuple[int, int, float]],
        *,
        default_layer: int = 0,
    ) -> "SparseDeltaML":
        """Build from ``(layer, row, col, value)`` or 2D ``(row, col, value)``."""
        merged: dict[tuple[int, int, int], float] = {}
        for item in changes:
            if len(item) == 3:
                row, col, value = item  # type: ignore[misc]
                layer = default_layer
            elif len(item) == 4:
                layer, row, col, value = item  # type: ignore[misc]
            else:
                raise ValueError(
                    "each change must be (row, col, value) or "
                    "(layer, row, col, value)"
                )
            key = (int(layer), int(row), int(col))
            merged[key] = merged.get(key, 0.0) + float(value)
        merged = {key: value for key, value in merged.items() if value != 0.0}
        keys = list(merged)
        return cls(
            layers=np.asarray([key[0] for key in keys], dtype=np.int32),
            rows=np.asarray([key[1] for key in keys], dtype=np.int32),
            cols=np.asarray([key[2] for key in keys], dtype=np.int32),
            values=np.asarray([merged[key] for key in keys], dtype=np.float64),
        )

    @classmethod
    def concat(cls, parts: Sequence["SparseDeltaML"]) -> "SparseDeltaML":
        nonempty = [part for part in parts if part.size]
        if not nonempty:
            return cls.empty()
        return cls.from_changes(
            zip(
                np.concatenate([part.layers for part in nonempty]),
                np.concatenate([part.rows for part in nonempty]),
                np.concatenate([part.cols for part in nonempty]),
                np.concatenate([part.values for part in nonempty]),
            )
        )

    @property
    def size(self) -> int:
        return int(self.values.size)

    def dense(self, n_layers: int, shape: tuple[int, int]) -> np.ndarray:
        out = np.zeros((n_layers, shape[0], shape[1]), dtype=np.float64)
        if self.size == 0:
            return out
        if np.any(self.layers < 0) or np.any(self.layers >= n_layers):
            raise IndexError("delta layer index out of range")
        np.add.at(out, (self.layers, self.rows, self.cols), self.values)
        return out

    def points_xyz(
        self, stackup: Stackup, *, cell_size_m: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(N, 3)`` positions in cell units and values."""
        if self.size == 0:
            return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64)
        z_cells = stackup.z_m[self.layers] / cell_size_m
        points = np.column_stack(
            [
                self.rows.astype(np.float64),
                self.cols.astype(np.float64),
                z_cells.astype(np.float64),
            ]
        )
        return points, self.values.copy()


@dataclass(frozen=True)
class ViaSpec:
    """Vertical interconnect between two layers at one in-plane cell.

    Ordinary vias contribute a frequency-dependent lumped quadratic term.
    Important vias are also exposed for local 3D near-field audits.
    """

    row: int
    col: int
    layer_from: int
    layer_to: int
    resistance_ohm: float = 0.0
    inductance_h: float = 1e-9
    important: bool = False
    # Extra scalar mutual weight on the occupancy product at the via ends.
    mutual_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.layer_from == self.layer_to:
            raise ValueError("via layers must differ")
        if self.row < 0 or self.col < 0:
            raise ValueError("via row/col must be non-negative")

    def normalized(self) -> tuple[int, int, int, int]:
        lo, hi = sorted((self.layer_from, self.layer_to))
        return int(self.row), int(self.col), int(lo), int(hi)

    def lumped_weight(self, frequency_hz: float) -> float:
        """Scalar proxy weight at one frequency: R + jωL magnitude-like."""
        omega = 2.0 * np.pi * float(frequency_hz)
        return float(
            abs(self.resistance_ohm) + omega * abs(self.inductance_h)
        ) * float(self.mutual_weight)


@dataclass(frozen=True)
class ViaSet:
    """Deduplicated via collection for one layout or candidate delta."""

    vias: tuple[ViaSpec, ...] = ()

    @classmethod
    def from_iterable(cls, vias: Iterable[ViaSpec]) -> "ViaSet":
        merged: dict[tuple[int, int, int, int], ViaSpec] = {}
        for via in vias:
            key = via.normalized()
            previous = merged.get(key)
            if previous is None:
                merged[key] = via
            else:
                # Prefer the more important flag and accumulate weights.
                merged[key] = ViaSpec(
                    row=key[0],
                    col=key[1],
                    layer_from=key[2],
                    layer_to=key[3],
                    resistance_ohm=previous.resistance_ohm + via.resistance_ohm,
                    inductance_h=previous.inductance_h + via.inductance_h,
                    important=previous.important or via.important,
                    mutual_weight=previous.mutual_weight + via.mutual_weight,
                )
        return cls(vias=tuple(merged.values()))

    def __len__(self) -> int:
        return len(self.vias)

    def important(self) -> tuple[ViaSpec, ...]:
        return tuple(via for via in self.vias if via.important)

    def ordinary(self) -> tuple[ViaSpec, ...]:
        return tuple(via for via in self.vias if not via.important)

    def energy(
        self,
        occupancy: np.ndarray,
        *,
        frequency_hz: float,
    ) -> float:
        """Lumped via energy from end-cell occupancy products."""
        total = 0.0
        for via in self.vias:
            a = float(occupancy[via.layer_from, via.row, via.col])
            b = float(occupancy[via.layer_to, via.row, via.col])
            weight = via.lumped_weight(frequency_hz)
            # Frequency-dependent lumped proxy on the two pad occupancies:
            # self terms on each end plus a mutual term for the vertical path.
            # Equivalent quadratic form: w * [a, b] [[0.5, 0.5], [0.5, 0.5]] [a, b]^T
            # plus an extra series mismatch w/2 * (a-b)^2
            #   = w * (a^2 + b^2 + a*b).
            total += weight * (a * a + b * b + a * b)
        return float(total)

    def delta_energy(
        self,
        base: np.ndarray,
        delta: SparseDeltaML,
        *,
        frequency_hz: float,
    ) -> float:
        """Exact via energy change for a sparse occupancy update."""
        if not self.vias:
            return 0.0
        # Dense only the touched via cells would be ideal; via counts are small.
        occupied = base + delta.dense(base.shape[0], base.shape[1:])
        return self.energy(occupied, frequency_hz=frequency_hz) - self.energy(
            base, frequency_hz=frequency_hz
        )


class FFTInteraction25D:
    """Matrix-free 2.5D operator: planar FFT + interlayer kernel matrix.

    For occupancy ``x[l, y, x]`` the field is

    ``phi[l] = sum_{l'} G_{ll'} * x[l']``

    where ``*`` is a linear 2D convolution and

    ``G_{ll'}(dr, dc) = 1 / sqrt(dr^2 + dc^2 + dz_{ll'}^2 + softening^2)``.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        stackup: Stackup,
        *,
        cell_size_m: float = 0.2e-3,
        softening: float = 0.75,
    ) -> None:
        self.shape = (int(shape[0]), int(shape[1]))
        self.stackup = stackup
        self.cell_size_m = float(cell_size_m)
        if self.cell_size_m <= 0.0:
            raise ValueError("cell_size_m must be positive")
        self.softening = float(softening)
        self.n_layers = stackup.n_layers
        self.fft_shape = tuple(next_fast_len(3 * n - 2) for n in self.shape)
        self._kernel_fft = self._build_kernel_ffts()

    def _dz_cells(self, layer_a: int, layer_b: int) -> float:
        return self.stackup.separation_cells(
            layer_a, layer_b, cell_size_m=self.cell_size_m
        )

    def kernel_value(
        self,
        layer_a: np.ndarray | int,
        layer_b: np.ndarray | int,
        dr: np.ndarray,
        dc: np.ndarray,
    ) -> np.ndarray:
        la = np.asarray(layer_a, dtype=np.int32)
        lb = np.asarray(layer_b, dtype=np.int32)
        dz = (self.stackup.z_m[la] - self.stackup.z_m[lb]) / self.cell_size_m
        return 1.0 / np.sqrt(dr * dr + dc * dc + dz * dz + self.softening**2)

    def _embedded_kernel(self, layer_a: int, layer_b: int) -> np.ndarray:
        nr, nc = self.shape
        row_offsets = np.arange(-(nr - 1), nr, dtype=np.float64)
        col_offsets = np.arange(-(nc - 1), nc, dtype=np.float64)
        dr, dc = np.meshgrid(row_offsets, col_offsets, indexing="ij")
        dz = self._dz_cells(layer_a, layer_b)
        return 1.0 / np.sqrt(dr * dr + dc * dc + dz * dz + self.softening**2)

    def _build_kernel_ffts(self) -> np.ndarray:
        # Shape: (L, L, fft_y, fft_x_rfft)
        sample = rfftn(self._embedded_kernel(0, 0), self.fft_shape)
        kernels = np.empty(
            (self.n_layers, self.n_layers) + sample.shape, dtype=np.complex128
        )
        for layer_a in range(self.n_layers):
            for layer_b in range(self.n_layers):
                # Symmetry: G depends on |dz|.
                if layer_b < layer_a:
                    kernels[layer_a, layer_b] = kernels[layer_b, layer_a]
                else:
                    kernels[layer_a, layer_b] = rfftn(
                        self._embedded_kernel(layer_a, layer_b), self.fft_shape
                    )
        return kernels

    def apply(self, volume: np.ndarray) -> np.ndarray:
        volume = np.asarray(volume, dtype=np.float64)
        expected = (self.n_layers, self.shape[0], self.shape[1])
        if volume.shape != expected:
            raise ValueError(f"expected {expected}, got {volume.shape}")
        spectra = []
        for layer in range(self.n_layers):
            padded = np.zeros(self.fft_shape, dtype=np.float64)
            padded[: self.shape[0], : self.shape[1]] = volume[layer]
            spectra.append(rfftn(padded, self.fft_shape))
        starts = (self.shape[0] - 1, self.shape[1] - 1)
        fields = np.zeros_like(volume)
        for layer_a in range(self.n_layers):
            acc = np.zeros(spectra[0].shape, dtype=np.complex128)
            for layer_b in range(self.n_layers):
                acc += self._kernel_fft[layer_a, layer_b] * spectra[layer_b]
            full = irfftn(acc, self.fft_shape)
            fields[layer_a] = full[
                starts[0] : starts[0] + self.shape[0],
                starts[1] : starts[1] + self.shape[1],
            ]
        return fields

    def energy(self, volume: np.ndarray) -> float:
        field = self.apply(volume)
        return float(np.vdot(volume, field).real)


class MultilayerDeltaScorer:
    """Cache ``K x`` on the multilayer grid and score sparse candidates."""

    def __init__(
        self,
        operator: FFTInteraction25D,
        base: np.ndarray,
        *,
        vias: ViaSet | None = None,
        frequency_hz: float = 1e6,
    ) -> None:
        self.operator = operator
        self.base = np.asarray(base, dtype=np.float64)
        expected = (
            operator.n_layers,
            operator.shape[0],
            operator.shape[1],
        )
        if self.base.shape != expected:
            raise ValueError(f"base must have shape {expected}, got {self.base.shape}")
        self.base_field = operator.apply(self.base)
        self.base_interaction = float(np.vdot(self.base, self.base_field).real)
        self.vias = vias or ViaSet()
        self.frequency_hz = float(frequency_hz)
        self.base_via_energy = self.vias.energy(
            self.base, frequency_hz=self.frequency_hz
        )
        self.base_energy = self.base_interaction + self.base_via_energy

    def interaction_delta_energy(self, delta: SparseDeltaML) -> float:
        if delta.size == 0:
            return 0.0
        linear = 2.0 * float(
            np.dot(
                delta.values,
                self.base_field[delta.layers, delta.rows, delta.cols],
            )
        )
        dr = delta.rows[:, None] - delta.rows[None, :]
        dc = delta.cols[:, None] - delta.cols[None, :]
        la = delta.layers[:, None]
        lb = delta.layers[None, :]
        local_kernel = self.operator.kernel_value(la, lb, dr, dc)
        quadratic = float(delta.values @ local_kernel @ delta.values)
        return linear + quadratic

    def delta_energy(
        self,
        delta: SparseDeltaML,
        *,
        vias: ViaSet | None = None,
    ) -> float:
        """Return ``E(base+delta, vias) - E(base, base_vias)``.

        ``vias`` is the **full** candidate via set.  Omit it to keep the base
        vias and only change occupancy.
        """
        interaction = self.interaction_delta_energy(delta)
        vias_use = self.vias if vias is None else vias
        occupied = self.base + delta.dense(self.base.shape[0], self.base.shape[1:])
        via_term = (
            vias_use.energy(occupied, frequency_hz=self.frequency_hz)
            - self.base_via_energy
        )
        return interaction + via_term

    def energy(
        self,
        delta: SparseDeltaML,
        *,
        vias: ViaSet | None = None,
    ) -> float:
        return self.base_energy + self.delta_energy(delta, vias=vias)

    def full_energy(
        self,
        delta: SparseDeltaML,
        *,
        vias: ViaSet | None = None,
    ) -> float:
        volume = self.base + delta.dense(self.base.shape[0], self.base.shape[1:])
        interaction = self.operator.energy(volume)
        vias_use = self.vias if vias is None else vias
        return interaction + vias_use.energy(volume, frequency_hz=self.frequency_hz)
