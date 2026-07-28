"""The inductance operator of a layered sheet mesh, applied by 2D transforms.

The mesh is one grid of cells per copper layer.  A branch joins two adjacent
cells and carries current along x or along y; the bar it stands for is one
pitch long, one pitch wide, and as thick as the copper.

Three facts make the operator cheap on this mesh.

* Two branches carrying current along different axes have zero mutual partial
  inductance.  The defining integral carries ``dl_i . dl_j``, which vanishes
  for perpendicular bars.  So there is an x operator and a y operator and
  nothing between them, and a via, which carries current along z, couples to
  neither.  Vertical branches do couple to one another, being parallel; that
  operator is not built here.  See :class:`~plane_opt.sheet_peec.ViaBranch` for
  what its absence costs and where.
* Between two branches of the same axis, the coupling depends only on their
  offset and on the separation of their layers.  In plane that is a
  convolution, so one table per layer pair serves every branch of that pair.
* On a square mesh of square cells the y table is the x table transposed.  It
  is still built rather than inferred, because the spectrum of a real transform
  is not square and the identity holds only in the spatial domain.

The cost of a layer pair is therefore a pair of two-dimensional transforms over
the board, not one three-dimensional transform over the board's height.  For
the two-sided board this replaces a ``324x304x90`` embedding with eight
``348x348`` ones, which is the whole argument for meshing sheets rather than
voxels.  The saving shrinks as layers are added -- pairs grow as the square of
the layer count while the voxel height does not -- so this is a statement about
boards with few layers, which is what these are.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .sheet_inductance import (
    NEAR_RADIUS_CELLS,
    CellGeometry,
    build_kernel,
    build_vertical_kernel,
)

# Resistivity of the copper these boards are built from, in ohm-metres.
COPPER_RESISTIVITY_OHM_M = 1.724e-8


@dataclass(frozen=True)
class SheetLayer:
    """One copper layer: where it sits, how thick it is, what it is made of."""

    name: str
    z_m: float
    thickness_m: float
    resistivity_ohm_m: float = COPPER_RESISTIVITY_OHM_M

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("a layer needs a name")
        for field in ("thickness_m", "resistivity_ohm_m"):
            value = float(getattr(self, field))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")
            object.__setattr__(self, field, value)
        object.__setattr__(self, "z_m", float(self.z_m))

    @property
    def sheet_resistance_ohm(self) -> float:
        """Give the resistance of one square of this layer.

        A branch of a uniform grid is as long as it is wide, so its resistance
        is this regardless of the pitch.
        """
        return self.resistivity_ohm_m / self.thickness_m


@dataclass(frozen=True)
class SheetStackup:
    """The copper layers of a board, front to back."""

    layers: tuple[SheetLayer, ...]

    def __post_init__(self) -> None:
        layers = tuple(self.layers)
        if not layers:
            raise ValueError("a stackup needs at least one copper layer")
        names = [layer.name for layer in layers]
        if len(set(names)) != len(names):
            raise ValueError("layer names must be unique")
        object.__setattr__(self, "layers", layers)

    def __len__(self) -> int:
        return len(self.layers)

    def index(self, name: str) -> int:
        for position, layer in enumerate(self.layers):
            if layer.name == name:
                return position
        raise KeyError(f"no copper layer named {name!r}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(layer.name for layer in self.layers)

    def separation_m(self, first: int, second: int) -> float:
        return self.layers[second].z_m - self.layers[first].z_m


class SheetInductanceOperator:
    """Apply the partial inductance of a layered sheet mesh to branch currents.

    Currents are held as ``(layer, row, column)`` arrays, one for the branches
    along x and one for those along y.  Entry ``[l, r, c]`` of the x array is
    the branch of layer ``l`` running from cell ``(r, c)`` to ``(r, c + 1)``;
    the y array is the same with rows.  A cell outside the conductor simply
    carries zero, so the operator never needs to know the copper's shape and
    one built operator serves every candidate shape on that board.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        pitch_m: float,
        stackup: SheetStackup,
        *,
        vertical_levels: Sequence[tuple[int, int]] = (),
        near_radius_cells: int = NEAR_RADIUS_CELLS,
    ) -> None:
        rows, cols = (int(value) for value in shape)
        if rows < 1 or cols < 1:
            raise ValueError("the mesh shape must be positive")
        if float(pitch_m) <= 0.0:
            raise ValueError("pitch must be positive")
        self.shape = (rows, cols)
        self.pitch_m = float(pitch_m)
        self.stackup = stackup
        # A circular transform of a padded grid gives the linear convolution
        # the physics asks for; without the padding a branch at one edge of the
        # board would couple to one at the other.
        self.padded = (2 * rows, 2 * cols)

        # One prepared spectrum per layer pair per axis.  The two axes are
        # transposes of each other only when the mesh is square as well as the
        # cell, so both are built rather than one inferred from the other: the
        # spectrum of a real transform is not square and cannot be transposed
        # at all.
        self._kernels: dict[tuple[str, int, int], np.ndarray] = {}
        for first in range(len(stackup)):
            for second in range(first, len(stackup)):
                # The two layers need not be equally thick.  A conductor thick
                # against the skin depth is carried here as a stack of
                # filaments of graded thickness, so unequal pairs are the
                # normal case for an inlay rather than an exception.
                cell = CellGeometry(
                    length_m=self.pitch_m,
                    width_m=self.pitch_m,
                    thickness_m=stackup.layers[first].thickness_m,
                )
                other = CellGeometry(
                    length_m=self.pitch_m,
                    width_m=self.pitch_m,
                    thickness_m=stackup.layers[second].thickness_m,
                )
                for axis in ("x", "y"):
                    table = build_kernel(
                        self.padded,
                        cell,
                        stackup.separation_m(first, second),
                        other=other,
                        axis=axis,
                        near_radius_cells=near_radius_cells,
                    )
                    self._kernels[(axis, first, second)] = np.fft.rfft2(table)

        # Vertical branches carry current along z.  They couple to no in-plane
        # branch, being perpendicular to both, but they are parallel to one
        # another and so couple among themselves.  A level is one pair of layers
        # joined by vertical branches: every branch joining the same two layers
        # spans the same height, so the coupling of a level pair depends on the
        # in-plane offset alone and is a convolution like the others.
        #
        # The levels are stated rather than inferred.  Which layers a board
        # joins is a property of its vias and of how thick copper was cut into
        # filaments, neither of which the stackup alone says.  A mesh carrying a
        # level the operator was not told about is refused by the solver instead
        # of quietly losing that coupling.
        self.vertical_levels = tuple(
            (int(lower), int(upper)) for lower, upper in vertical_levels
        )
        for lower, upper in self.vertical_levels:
            if not (0 <= lower < len(stackup) and 0 <= upper < len(stackup)):
                raise ValueError(f"vertical level ({lower}, {upper}) names no layer")
            if lower == upper:
                raise ValueError("a vertical level joins two different layers")
        self._vertical_geometry = [
            (
                abs(stackup.separation_m(lower, upper)),
                (stackup.layers[lower].z_m + stackup.layers[upper].z_m) / 2.0,
            )
            for lower, upper in self.vertical_levels
        ]
        self._kernels_z: dict[tuple[int, int], np.ndarray] = {}
        for first in range(len(self.vertical_levels)):
            span_a, center_a = self._vertical_geometry[first]
            for second in range(first, len(self.vertical_levels)):
                span_b, center_b = self._vertical_geometry[second]
                table = build_vertical_kernel(
                    self.padded,
                    self.pitch_m,
                    span_a,
                    span_b,
                    center_b - center_a,
                    near_radius_cells=near_radius_cells,
                )
                self._kernels_z[(first, second)] = np.fft.rfft2(table)

    def _kernel(self, axis: str, first: int, second: int) -> np.ndarray:
        # The coupling of a pair does not depend on which of the two is asked
        # about first: reciprocity, and the separation enters the kernel only
        # through its magnitude for a symmetric cell.
        key = (axis, first, second) if first <= second else (axis, second, first)
        return self._kernels[key]

    def _kernel_z(self, first: int, second: int) -> np.ndarray:
        key = (first, second) if first <= second else (second, first)
        return self._kernels_z[key]

    @property
    def kernel_bytes(self) -> int:
        """Report what the prepared tables occupy."""
        return sum(table.nbytes for table in self._kernels.values()) + sum(
            table.nbytes for table in self._kernels_z.values()
        )

    def _convolve(self, spectra: list[np.ndarray], axis: str) -> np.ndarray:
        rows, cols = self.shape
        out = np.zeros((len(self.stackup), rows, cols), dtype=np.float64)
        for target in range(len(self.stackup)):
            accumulated = np.zeros_like(spectra[0])
            for source in range(len(self.stackup)):
                accumulated += self._kernel(axis, target, source) * spectra[source]
            product = np.fft.irfft2(accumulated, s=self.padded)
            out[target] = product[:rows, :cols]
        return out

    def _convolve_z(self, spectra: list[np.ndarray]) -> np.ndarray:
        rows, cols = self.shape
        out = np.zeros((len(self.vertical_levels), rows, cols), dtype=np.float64)
        for target in range(len(self.vertical_levels)):
            accumulated = np.zeros_like(spectra[0])
            for source in range(len(self.vertical_levels)):
                accumulated += self._kernel_z(target, source) * spectra[source]
            product = np.fft.irfft2(accumulated, s=self.padded)
            out[target] = product[:rows, :cols]
        return out

    def apply(
        self,
        currents_x: np.ndarray,
        currents_y: np.ndarray,
        currents_z: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Give the flux linkage of each branch, in webers, for these currents.

        Passing ``currents_z`` adds the vertical operator and returns three
        arrays instead of two.  Omitting it returns the two in-plane arrays, as
        it always has; that is correct only where no vertical current flows.
        """
        rows, cols = self.shape
        expected = (len(self.stackup), rows, cols)
        for name, array in (("currents_x", currents_x), ("currents_y", currents_y)):
            if array.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {array.shape}")

        padded_x = [
            np.fft.rfft2(currents_x[layer], s=self.padded)
            for layer in range(len(self.stackup))
        ]
        padded_y = [
            np.fft.rfft2(currents_y[layer], s=self.padded)
            for layer in range(len(self.stackup))
        ]
        flux_x = self._convolve(padded_x, "x")
        flux_y = self._convolve(padded_y, "y")
        if currents_z is None:
            return flux_x, flux_y

        vertical_expected = (len(self.vertical_levels), rows, cols)
        if currents_z.shape != vertical_expected:
            raise ValueError(
                f"currents_z must have shape {vertical_expected}, "
                f"got {currents_z.shape}"
            )
        if not self.vertical_levels:
            return flux_x, flux_y, np.zeros_like(currents_z)
        padded_z = [
            np.fft.rfft2(currents_z[level], s=self.padded)
            for level in range(len(self.vertical_levels))
        ]
        return flux_x, flux_y, self._convolve_z(padded_z)

    def dense_matrix(self, axis: str = "x") -> np.ndarray:
        """Assemble the same operator as a dense matrix, for checking.

        Only usable on a small mesh -- the matrix has one row per branch -- and
        present so that the transform path can be held against a direct sum
        rather than against itself.
        """
        if axis not in {"x", "y"}:
            raise ValueError("axis must be 'x' or 'y'")
        rows, cols = self.shape
        count = len(self.stackup) * rows * cols
        matrix = np.zeros((count, count), dtype=np.float64)
        basis_x = np.zeros((len(self.stackup), rows, cols))
        basis_y = np.zeros_like(basis_x)
        for column in range(count):
            flat = basis_x if axis == "x" else basis_y
            flat.flat[column] = 1.0
            out_x, out_y = self.apply(basis_x, basis_y)
            matrix[:, column] = (out_x if axis == "x" else out_y).reshape(-1)
            flat.flat[column] = 0.0
        return matrix
