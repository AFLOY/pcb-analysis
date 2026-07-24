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

import math
from dataclasses import dataclass
from numbers import Integral
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

    def __post_init__(self) -> None:
        coordinates: list[np.ndarray] = []
        int32 = np.iinfo(np.int32)
        for name in ("layers", "rows", "cols"):
            array = np.asarray(getattr(self, name))
            if array.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if array.size and not np.issubdtype(array.dtype, np.integer):
                raise TypeError(f"{name} must use an integer dtype")
            if array.size and (
                np.any(array < int32.min) or np.any(array > int32.max)
            ):
                raise OverflowError(f"{name} contains an index outside int32 range")
            coordinates.append(np.array(array, dtype=np.int32, copy=True))
        values = np.array(self.values, dtype=np.float64, copy=True)
        if values.ndim != 1:
            raise ValueError("values must be one-dimensional")
        lengths = {array.size for array in (*coordinates, values)}
        if len(lengths) != 1:
            raise ValueError("layers, rows, cols, and values must have equal lengths")
        if not np.all(np.isfinite(values)):
            raise ValueError("delta values must be finite")
        for name, array in zip(("layers", "rows", "cols"), coordinates):
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        values.flags.writeable = False
        object.__setattr__(self, "values", values)

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
            coordinates = (layer, row, col)
            if any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                for value in coordinates
            ):
                raise TypeError("delta layer, row, and col must be integers")
            key = tuple(int(value) for value in coordinates)
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
        n_layers = int(n_layers)
        if len(shape) != 2:
            raise ValueError("shape must contain exactly two dimensions")
        shape = (int(shape[0]), int(shape[1]))
        self.validate_bounds(n_layers, shape)
        out = np.zeros((n_layers, shape[0], shape[1]), dtype=np.float64)
        if self.size == 0:
            return out
        np.add.at(out, (self.layers, self.rows, self.cols), self.values)
        return out

    def validate_bounds(self, n_layers: int, shape: tuple[int, int]) -> None:
        if n_layers <= 0 or len(shape) != 2 or any(int(size) <= 0 for size in shape):
            raise ValueError("dense target dimensions must be positive")
        if (
            np.any(self.layers < 0)
            or np.any(self.layers >= n_layers)
            or np.any(self.rows < 0)
            or np.any(self.rows >= shape[0])
            or np.any(self.cols < 0)
            or np.any(self.cols >= shape[1])
        ):
            raise IndexError(
                f"delta coordinate is outside target shape {(n_layers, *shape)}"
            )

    def points_xyz(
        self, stackup: Stackup, *, cell_size_m: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(N, 3)`` positions in cell units and values."""
        cell_size_m = float(cell_size_m)
        if not math.isfinite(cell_size_m) or cell_size_m <= 0.0:
            raise ValueError("cell_size_m must be finite and positive")
        if self.size == 0:
            return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64)
        if (
            np.any(self.layers < 0)
            or np.any(self.layers >= stackup.n_layers)
            or np.any(self.rows < 0)
            or np.any(self.cols < 0)
        ):
            raise IndexError("delta contains an invalid point coordinate")
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
        for name in ("row", "col", "layer_from", "layer_to"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.layer_from == self.layer_to:
            raise ValueError("via layers must differ")
        if min(self.row, self.col, self.layer_from, self.layer_to) < 0:
            raise ValueError("via coordinates and layers must be non-negative")
        for name in ("resistance_ohm", "inductance_h", "mutual_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "important", bool(self.important))

    def normalized(self) -> tuple[int, int, int, int]:
        lo, hi = sorted((self.layer_from, self.layer_to))
        return int(self.row), int(self.col), int(lo), int(hi)

    def lumped_weight(self, frequency_hz: float) -> float:
        """Scalar proxy weight at one frequency: R + jωL magnitude-like."""
        frequency_hz = float(frequency_hz)
        if not math.isfinite(frequency_hz) or frequency_hz < 0.0:
            raise ValueError("frequency_hz must be finite and non-negative")
        omega = 2.0 * np.pi * frequency_hz
        return float(
            self.resistance_ohm + omega * self.inductance_h
        ) * float(self.mutual_weight)


@dataclass(frozen=True)
class ViaSet:
    """Deduplicated via collection for one layout or candidate delta."""

    vias: tuple[ViaSpec, ...] = ()

    def __post_init__(self) -> None:
        merged: dict[tuple[int, int, int, int], ViaSpec] = {}
        for via in self.vias:
            if not isinstance(via, ViaSpec):
                raise TypeError("vias must contain ViaSpec instances")
            key = via.normalized()
            canonical = ViaSpec(
                row=key[0],
                col=key[1],
                layer_from=key[2],
                layer_to=key[3],
                resistance_ohm=via.resistance_ohm,
                inductance_h=via.inductance_h,
                important=via.important,
                mutual_weight=via.mutual_weight,
            )
            previous = merged.get(key)
            if previous is None:
                merged[key] = canonical
                continue
            same_electrical = (
                previous.resistance_ohm == canonical.resistance_ohm
                and previous.inductance_h == canonical.inductance_h
                and previous.mutual_weight == canonical.mutual_weight
            )
            if not same_electrical:
                raise ValueError(f"conflicting duplicate via at {key}")
            if canonical.important and not previous.important:
                merged[key] = ViaSpec(
                    row=key[0],
                    col=key[1],
                    layer_from=key[2],
                    layer_to=key[3],
                    resistance_ohm=previous.resistance_ohm,
                    inductance_h=previous.inductance_h,
                    important=True,
                    mutual_weight=previous.mutual_weight,
                )
        object.__setattr__(self, "vias", tuple(merged.values()))

    @classmethod
    def from_iterable(cls, vias: Iterable[ViaSpec]) -> "ViaSet":
        """Deduplicate identical vias and reject conflicting specifications."""
        return cls(vias=tuple(vias))

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
        frequency_hz = float(frequency_hz)
        if not math.isfinite(frequency_hz) or frequency_hz < 0.0:
            raise ValueError("frequency_hz must be finite and non-negative")
        occupancy = np.asarray(occupancy, dtype=np.float64)
        if occupancy.ndim != 3 or any(size <= 0 for size in occupancy.shape):
            raise ValueError("occupancy must have shape (n_layers, ny, nx)")
        if not np.all(np.isfinite(occupancy)):
            raise ValueError("occupancy must contain only finite values")
        self._validate_bounds(occupancy.shape)
        total = 0.0
        for via in self.vias:
            a = float(occupancy[via.layer_from, via.row, via.col])
            b = float(occupancy[via.layer_to, via.row, via.col])
            weight = via.lumped_weight(frequency_hz)
            # Symmetric positive-definite scalar proxy with self terms at both
            # ends and a mutual term for the vertical path.
            total += weight * (a * a + b * b + a * b)
        return float(total)

    def _validate_bounds(self, shape: tuple[int, int, int]) -> None:
        for via in self.vias:
            if (
                via.layer_from >= shape[0]
                or via.layer_to >= shape[0]
                or via.row >= shape[1]
                or via.col >= shape[2]
            ):
                raise IndexError(
                    f"via {via.normalized()} is outside occupancy shape {shape}"
                )

    def energy_with_delta(
        self,
        base: np.ndarray,
        delta: SparseDeltaML,
        *,
        frequency_hz: float,
    ) -> float:
        """Evaluate via energy without materializing a full candidate volume."""
        frequency_hz = float(frequency_hz)
        if not math.isfinite(frequency_hz) or frequency_hz < 0.0:
            raise ValueError("frequency_hz must be finite and non-negative")
        base = np.asarray(base, dtype=np.float64)
        if base.ndim != 3 or any(size <= 0 for size in base.shape):
            raise ValueError("base must have shape (n_layers, ny, nx)")
        if not np.all(np.isfinite(base)):
            raise ValueError("base must contain only finite values")
        self._validate_bounds(base.shape)
        delta.validate_bounds(base.shape[0], base.shape[1:])
        changes: dict[tuple[int, int, int], float] = {}
        for layer, row, col, value in zip(
            delta.layers, delta.rows, delta.cols, delta.values
        ):
            key = (int(layer), int(row), int(col))
            changes[key] = changes.get(key, 0.0) + float(value)
        total = 0.0
        for via in self.vias:
            key_a = (via.layer_from, via.row, via.col)
            key_b = (via.layer_to, via.row, via.col)
            a = float(base[key_a]) + changes.get(key_a, 0.0)
            b = float(base[key_b]) + changes.get(key_b, 0.0)
            total += via.lumped_weight(frequency_hz) * (a * a + b * b + a * b)
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
        return self.energy_with_delta(
            base, delta, frequency_hz=frequency_hz
        ) - self.energy(base, frequency_hz=frequency_hz)


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
        if len(shape) != 2:
            raise ValueError("shape must contain exactly two dimensions")
        self.shape = (int(shape[0]), int(shape[1]))
        if any(size <= 0 for size in self.shape):
            raise ValueError("shape dimensions must be positive")
        if not isinstance(stackup, Stackup):
            raise TypeError("stackup must be a Stackup")
        self.stackup = stackup
        self.cell_size_m = float(cell_size_m)
        if not math.isfinite(self.cell_size_m) or self.cell_size_m <= 0.0:
            raise ValueError("cell_size_m must be finite and positive")
        self.softening = float(softening)
        if not math.isfinite(self.softening) or self.softening <= 0.0:
            raise ValueError("softening must be finite and positive")
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
        la_raw = np.asarray(layer_a)
        lb_raw = np.asarray(layer_b)
        if not np.issubdtype(la_raw.dtype, np.integer) or not np.issubdtype(
            lb_raw.dtype, np.integer
        ):
            raise TypeError("layer indices must use an integer dtype")
        la = la_raw.astype(np.int32, copy=False)
        lb = lb_raw.astype(np.int32, copy=False)
        if (
            np.any(la < 0)
            or np.any(la >= self.n_layers)
            or np.any(lb < 0)
            or np.any(lb >= self.n_layers)
        ):
            raise IndexError("kernel layer index out of range")
        dr = np.asarray(dr, dtype=np.float64)
        dc = np.asarray(dc, dtype=np.float64)
        if not np.all(np.isfinite(dr)) or not np.all(np.isfinite(dc)):
            raise ValueError("kernel offsets must be finite")
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
        sample = rfftn(
            self._embedded_kernel(0, 0), s=self.fft_shape, axes=(0, 1)
        )
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
                        self._embedded_kernel(layer_a, layer_b),
                        s=self.fft_shape,
                        axes=(0, 1),
                    )
        return kernels

    def apply(self, volume: np.ndarray) -> np.ndarray:
        volume = np.asarray(volume, dtype=np.float64)
        expected = (self.n_layers, self.shape[0], self.shape[1])
        if volume.shape != expected:
            raise ValueError(f"expected {expected}, got {volume.shape}")
        if not np.all(np.isfinite(volume)):
            raise ValueError("volume must contain only finite values")
        spectra = []
        for layer in range(self.n_layers):
            padded = np.zeros(self.fft_shape, dtype=np.float64)
            padded[: self.shape[0], : self.shape[1]] = volume[layer]
            spectra.append(rfftn(padded, s=self.fft_shape, axes=(0, 1)))
        starts = (self.shape[0] - 1, self.shape[1] - 1)
        fields = np.zeros_like(volume)
        for layer_a in range(self.n_layers):
            acc = np.zeros(spectra[0].shape, dtype=np.complex128)
            for layer_b in range(self.n_layers):
                acc += self._kernel_fft[layer_a, layer_b] * spectra[layer_b]
            full = irfftn(acc, s=self.fft_shape, axes=(0, 1))
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
        self.base = np.array(base, dtype=np.float64, copy=True)
        expected = (
            operator.n_layers,
            operator.shape[0],
            operator.shape[1],
        )
        if self.base.shape != expected:
            raise ValueError(f"base must have shape {expected}, got {self.base.shape}")
        if not np.all(np.isfinite(self.base)):
            raise ValueError("base must contain only finite values")
        self.base_field = operator.apply(self.base)
        self.base_interaction = float(np.vdot(self.base, self.base_field).real)
        self.vias = vias if vias is not None else ViaSet()
        if not isinstance(self.vias, ViaSet):
            raise TypeError("vias must be a ViaSet")
        self.frequency_hz = float(frequency_hz)
        if not math.isfinite(self.frequency_hz) or self.frequency_hz < 0.0:
            raise ValueError("frequency_hz must be finite and non-negative")
        self.base_via_energy = self.vias.energy(
            self.base, frequency_hz=self.frequency_hz
        )
        self.base_energy = self.base_interaction + self.base_via_energy
        self.base.flags.writeable = False
        self.base_field.flags.writeable = False

    def interaction_delta_energy(self, delta: SparseDeltaML) -> float:
        if delta.size == 0:
            return 0.0
        delta.validate_bounds(self.base.shape[0], self.base.shape[1:])
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
        if not isinstance(vias_use, ViaSet):
            raise TypeError("vias must be a ViaSet")
        via_term = vias_use.energy_with_delta(
            self.base, delta, frequency_hz=self.frequency_hz
        ) - self.base_via_energy
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
        if not isinstance(vias_use, ViaSet):
            raise TypeError("vias must be a ViaSet")
        return interaction + vias_use.energy(volume, frequency_hz=self.frequency_hz)
