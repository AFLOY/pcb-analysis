"""A board network's solution as radiating current elements, harmonic by harmonic.

``electrical.port_network.BoardNetwork`` solves the copper with every part
that joins its nets and gives, at each frequency, the current into each
copper port and through each part and source.  Two kinds of current element
follow from it:

- the copper's: the copper is linear, so its branch currents are the sum of
  the port solves kept with the port matrix
  (``SheetPortImpedance.superpose``) weighted by the port currents, read as
  dipoles like any sheet solve (:func:`dipoles_from_sheet_peec`);
- the parts' and sources': each carries its current from its pad ``a`` to
  its pad ``b`` along the straight line between the pads' centres, one
  element of moment ``I (r_b - r_a)`` at the midpoint.

Together they close every loop: at each pad the current the copper takes
equals what the parts deliver, so the moments of a solution sum to zero to
the solve's tolerance (the copper carries no charge in a sheet solve).  The
fields, the far-field pattern and the margin to a limit line are this
package's.

The copper's current is needed at each frequency, so a frequency whose
copper was interpolated (``NetworkSolution.copper_interpolated``) or solved
without its port solves kept is refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .far_field import SphereSampling, db_microvolt_per_m, far_field_pattern
from .fields import CurrentDipoles, evaluate_fields
from .limits import EmissionLimit, EmissionMargin, emission_margin
from .sources import dipoles_from_sheet_peec


def path_dipoles(
    currents: Mapping[str, complex],
    ends: Mapping[str, tuple[str, str]],
    pad_positions_m: Mapping[str, Sequence[float]],
) -> CurrentDipoles:
    """One element per named current: ``currents[name]`` from pad ``ends[name][0]`` to ``ends[name][1]``."""
    names = [name for name in currents if name in ends]
    missing = sorted({pad for name in names for pad in ends[name]} - set(pad_positions_m))
    if missing:
        raise ValueError(f"no position for pads {missing}")
    if not names:
        return CurrentDipoles(np.zeros((0, 3)), np.zeros((0, 3), dtype=complex))
    start = np.array([pad_positions_m[ends[name][0]] for name in names], dtype=np.float64)
    end = np.array([pad_positions_m[ends[name][1]] for name in names], dtype=np.float64)
    current = np.array([complex(currents[name]) for name in names], dtype=np.complex128)
    return CurrentDipoles(0.5 * (start + end), current[:, None] * (end - start))


def network_dipoles(
    network: Any,
    solution: Any,
    copper: Any,
    mesh: Any,
    pad_positions_m: Mapping[str, Sequence[float]],
) -> CurrentDipoles:
    """The copper's and the parts' current elements for one ``NetworkSolution``.

    ``network`` is the ``BoardNetwork`` that gave ``solution``; ``copper`` the
    ``SheetPortImpedance`` at the solution's frequency, solved with
    ``keep_solutions=True`` on ``mesh``; ``pad_positions_m`` every pad's
    centre (x, y, z) in the mesh's frame.
    """
    frequency = float(solution.frequency_hz)
    if getattr(solution, "copper_interpolated", False):
        raise ValueError(f"the copper at {frequency:g} Hz was interpolated: its current needs a solve there")
    if copper.solutions is None:
        raise ValueError("the copper's port solves were not kept; solve with keep_solutions=True")
    if float(copper.frequency_hz) != frequency:
        raise ValueError(f"the copper was solved at {float(copper.frequency_hz):g} Hz, the network at {frequency:g} Hz")
    if tuple(copper.ports) != tuple(network.ports):
        raise ValueError("the copper's ports are not the network's")
    copper_dipoles = dipoles_from_sheet_peec(mesh, copper.superpose(solution.copper_port_current_a))
    ends = {item.name: (item.a, item.b) for item in (*network.parts, *network.current_sources, *network.voltage_sources)}
    currents = {**solution.part_current_a, **solution.source_current_a}
    return copper_dipoles.concatenate(path_dipoles(currents, ends, pad_positions_m))


@dataclass(frozen=True)
class NetworkEmission:
    """One frequency of a network's emission."""

    frequency_hz: float
    far_field_v_per_m: float
    """The largest far-zone field over the sphere, at ``distance_m``."""
    distance_m: float
    radiated_power_w: float
    margin: EmissionMargin | None
    """Against the limit given, where it has a line at this frequency."""
    near_field_max_h_a_per_m: float | None
    """The largest |H| over the scan points given."""
    dipoles: CurrentDipoles = field(repr=False, compare=False)

    @property
    def far_field_dbuv_per_m(self) -> float:
        return float(db_microvolt_per_m(self.far_field_v_per_m))


def network_emission(
    network: Any,
    solutions: Sequence[Any],
    coppers: Mapping[float, Any],
    meshes: Any,
    pad_positions_m: Mapping[str, Sequence[float]],
    *,
    distance_m: float = 3.0,
    limit: EmissionLimit | None = None,
    detector: str = "quasi-peak",
    scan_points_m: np.ndarray | None = None,
    sampling: SphereSampling | None = None,
) -> list[NetworkEmission]:
    """The emission of each solution (a ``NetworkSpectrum``'s, say), frequency by frequency.

    ``coppers`` maps each frequency to its ``SheetPortImpedance`` (kept
    solves); ``meshes`` is one ``SheetMesh`` or a mapping by frequency (the
    port contract's ``PortImpedanceResult.meshes_by_frequency``; its
    ``meshes`` tuple holds no frequencies and is refused).  ``limit`` adds the
    margin where the line has a value; ``scan_points_m`` (n, 3) adds the
    largest near-field |H| over them.
    """
    if isinstance(meshes, Mapping):
        meshes = {float(f): mesh for f, mesh in meshes.items()}
    elif not hasattr(meshes, "branch_count"):
        raise TypeError(
            "meshes must be one SheetMesh or a mapping by frequency "
            "(PortImpedanceResult.meshes_by_frequency, not its meshes tuple)"
        )
    out = []
    for solution in solutions:
        frequency = float(solution.frequency_hz)
        if frequency not in coppers:
            raise ValueError(f"no copper solve at {frequency:g} Hz (have {sorted(coppers)})")
        if isinstance(meshes, Mapping):
            if frequency not in meshes:
                raise ValueError(f"no mesh at {frequency:g} Hz (have {sorted(meshes)})")
            mesh = meshes[frequency]
        else:
            mesh = meshes
        dipoles = network_dipoles(network, solution, coppers[frequency], mesh, pad_positions_m)
        pattern = far_field_pattern(dipoles, frequency, distance_m=distance_m, sampling=sampling)
        margin = None
        if limit is not None:
            try:
                margin = emission_margin(pattern.max_field_v_per_m, frequency, limit, distance_m=distance_m,
                                         detector=detector)
            except ValueError:  # the line has no value at this frequency
                margin = None
        near = None
        if scan_points_m is not None:
            near = float(evaluate_fields(dipoles, np.asarray(scan_points_m), frequency, electric=False)
                         .magnetic_magnitude_a_per_m.max())
        out.append(NetworkEmission(frequency, float(pattern.max_field_v_per_m), float(distance_m),
                                   float(pattern.radiated_power_w), margin, near, dipoles))
    return out


__all__ = ["NetworkEmission", "network_dipoles", "network_emission", "path_dipoles"]
