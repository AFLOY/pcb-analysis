"""The pad-port impedance of routed copper, stated and answered as serialisable mappings.

A caller that owns a board (a router, a placer) states its copper the way
``current-field-problem/v2`` does -- grid, layers, copper cells per layer,
cell regions that keep two nets apart, vertical connections -- and, instead
of one excitation, the pads of the parts, which of them are ports, each
port's reference pad on its own conductor, and the frequencies wanted.
pcb-analysis builds the sheet mesh, solves
:func:`~electrical.sheet_peec.sheet_ports.solve_port_impedance` at every
frequency and answers ``port-impedance-result/v1``: one impedance matrix
per frequency, which ``electrical.port_network.BoardNetwork`` takes as the
copper of a network.

``port-impedance-problem/v1``::

    {
      "schema": "port-impedance-problem/v1",
      "name": "...",
      "grid": {...}, "layers": [...], "copper_by_layer": {...},
      "cell_regions": {...}, "vertical_connections": [...],   # as current-field v2
      "frequencies_hz": [1e6, 1e7, ...],
      "pads": [{"name": "C3.1", "part": "C3", "net": "VIN",
                "cells": [{"layer": "F.Cu", "x": 10, "y": 4}, ...]}, ...],
      "ports": [{"pad": "C3.1", "reference": "J1.1"}, ...],
      "merge_pins": false,
      "source_board_sha256": null
    }

- A pad may cover cells on several layers.  Its current is spread evenly
  over all its cells and its voltage is their mean (``PortPad``), unlike a
  current-field terminal, which divides its current equally between layers.
- Every pad listed is a terminal of the solve, also when it is no port and
  no reference: a conductor that one of them touches is solved with no net
  current, so the eddy current the ports induce in it enters the matrix.
  Copper that no pad touches is left out.
- ``merge_pins``: the pads of one part on one net (both stated) become one
  pad named ``<part>:<net>`` (``merge_pads``); each part keeps its current,
  only the division between those pins is lost.  Off by default.

The thickness of each layer is cut into filaments per frequency
(``cells_per_skin_depth``, ``maximum_filaments`` in the settings); the
frequencies that cut alike share one mesh and one inductance operator.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

import numpy as np

from electrical.matrix_free_mpir_fem.grid import TensorGrid
from electrical.port_network.sweep import FrequencyCheck, PortImpedanceSweep, refine_frequencies
from electrical.port_network.frequency import match_frequency

from .current_field_contract import (
    CurrentFieldLayer,
    CurrentFieldVerticalSegment,
    NamedNode,
    _filament_cuts,
    _parse_geometry,
    _sheet_geometry,
)
from .sheet_peec import SheetMesh
from .sheet_ports import PortPad, SheetPortImpedance, merge_pads, solve_port_impedance

PORT_IMPEDANCE_PROBLEM_SCHEMA = "port-impedance-problem/v1"
PORT_IMPEDANCE_RESULT_SCHEMA = "port-impedance-result/v1"


@dataclass(frozen=True)
class ProblemPad:
    """A pad as the caller states it: cells by layer name, and optionally its part and net."""

    name: str
    cells: tuple[NamedNode, ...]
    part: str | None = None
    net: str | None = None


@dataclass(frozen=True)
class PortImpedanceProblem:
    name: str
    frequencies_hz: tuple[float, ...]
    rows: int
    columns: int
    pitch_mm: float | None
    layers: tuple[CurrentFieldLayer, ...]
    copper_by_layer: Mapping[str, frozenset[tuple[int, int]]]
    vertical_segments: tuple[tuple[str, tuple[int, int], CurrentFieldVerticalSegment], ...]
    pads: tuple[ProblemPad, ...]
    references: Mapping[str, str]
    """Each port pad's reference pad, in port order (pads as stated, before any merge)."""
    merge_pins: bool = False
    grid: TensorGrid | None = None
    cell_regions: Mapping[str, Mapping[tuple[int, int], int]] | None = None
    # Face coverage by layer, as the current-field contract parses it.
    copper_faces: Mapping[str, tuple[Mapping[tuple[int, int], tuple[float, float]], Mapping[tuple[int, int], tuple[float, float]]]] | None = None
    source_board_sha256: str | None = None

    @property
    def frequency_hz(self) -> float:
        """The first frequency; the conductor itself does not depend on it."""
        return self.frequencies_hz[0]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PortImpedanceProblem":
        schema = str(value.get("schema") or "")
        if schema != PORT_IMPEDANCE_PROBLEM_SCHEMA:
            raise ValueError(f"unsupported port-impedance problem schema: {schema}")
        geometry = _parse_geometry(value)
        copper = geometry["copper_by_layer"]

        frequencies = tuple(float(f) for f in value.get("frequencies_hz") or ())
        if not frequencies:
            raise ValueError("frequencies_hz must list at least one frequency")
        if any(not math.isfinite(f) or f <= 0.0 for f in frequencies):
            raise ValueError("every frequency must be finite and positive (DC is dc_network's)")
        if len(set(frequencies)) != len(frequencies):
            raise ValueError("frequencies_hz lists a frequency twice")

        pads: list[ProblemPad] = []
        for item in value.get("pads") or []:
            name = str(item.get("name") or "")
            if not name:
                raise ValueError("a pad needs a name")
            cells = tuple((str(cell["layer"]), int(cell["x"]), int(cell["y"])) for cell in item.get("cells") or [])
            if not cells:
                raise ValueError(f"pad {name} has no cells")
            for layer, x, y in cells:
                if layer not in copper:
                    raise ValueError(f"pad {name} names an unknown layer {layer!r}")
                if (x, y) not in copper[layer]:
                    raise ValueError(f"pad {name} cell {(x, y)} on {layer} is not conductor")
            part = item.get("part")
            net = item.get("net")
            pads.append(ProblemPad(name, cells, None if part in (None, "") else str(part),
                                   None if net in (None, "") else str(net)))
        names = [pad.name for pad in pads]
        if len(set(names)) != len(names):
            raise ValueError("pad names must be unique")

        references: dict[str, str] = {}
        for item in value.get("ports") or []:
            pad, reference = str(item.get("pad") or ""), str(item.get("reference") or "")
            if not pad or not reference:
                raise ValueError("a port needs a pad and a reference")
            if pad in references:
                raise ValueError(f"pad {pad} is a port twice")
            references[pad] = reference
        if not references:
            raise ValueError("at least one port is required")
        missing = sorted({*references, *references.values()} - set(names))
        if missing:
            raise ValueError(f"ports name pads that are not listed: {missing}")

        problem = cls(
            name=str(value.get("name") or ""),
            frequencies_hz=tuple(sorted(frequencies)),
            pads=tuple(pads),
            references=references,
            merge_pins=bool(value.get("merge_pins", False)),
            source_board_sha256=(
                None if value.get("source_board_sha256") is None else str(value.get("source_board_sha256"))
            ),
            **geometry,
        )
        if not problem.name:
            raise ValueError("problem name is required")
        problem.port_groups()  # refuse a merge that makes no ports
        return problem

    def pad_groups(self) -> dict[str, tuple[str, ...]]:
        """The pads of the solve, each the pads stated that it covers (one each without ``merge_pins``)."""
        groups: dict[str, list[str]] = {}
        for pad in self.pads:
            key = f"{pad.part}:{pad.net}" if self.merge_pins and pad.part and pad.net else pad.name
            groups.setdefault(key, []).append(pad.name)
        stated = {pad.name for pad in self.pads}
        clash = sorted(key for key, members in groups.items() if key in stated and members != [key])
        if clash:
            raise ValueError(f"merged pad names clash with pads stated: {clash}")
        return {key: tuple(members) for key, members in groups.items()}

    def pad_positions_m(self) -> dict[str, tuple[float, float, float]]:
        """Each pad of the solve (as :meth:`pad_groups` names it) at the centre of its cells, in metres.

        x and y are the mean of its cells' centres in the solve's grid frame,
        z the mean of their layers' centres: where a part's current enters
        and leaves the copper, for ``emc.tiled_dipole_superposition.network_dipoles``.
        """
        grid = self.grid if self.grid is not None else TensorGrid.uniform(self.pitch_mm * 1e-3, (self.rows, self.columns))
        xc = 0.5 * (np.asarray(grid.x_edges_m[:-1]) + np.asarray(grid.x_edges_m[1:]))
        yc = 0.5 * (np.asarray(grid.y_edges_m[:-1]) + np.asarray(grid.y_edges_m[1:]))
        z_of = {layer.name: layer.center_z_mm * 1e-3 for layer in self.layers}
        stated = {pad.name: pad for pad in self.pads}
        positions = {}
        for key, members in self.pad_groups().items():
            cells = [cell for name in members for cell in stated[name].cells]
            positions[key] = (
                float(np.mean([xc[x] for _, x, _ in cells])),
                float(np.mean([yc[y] for _, _, y in cells])),
                float(np.mean([z_of[layer] for layer, _, _ in cells])),
            )
        return positions

    def port_groups(self) -> dict[str, str]:
        """Each port of the solve (a pad group) and its reference group, in the order the ports were stated."""
        group_of = {name: key for key, members in self.pad_groups().items() for name in members}
        ports: dict[str, str] = {}
        for pad, reference in self.references.items():
            port, ref = group_of[pad], group_of[reference]
            if port == ref:
                raise ValueError(f"port {pad} and its reference {reference} merge into one pad {port}")
            if ports.setdefault(port, ref) != ref:
                raise ValueError(f"the pads merged into {port} name different references")
        if set(ports) & set(ports.values()):
            raise ValueError("a reference pad cannot also be a port")
        return ports


@dataclass(frozen=True)
class PortImpedanceResult:
    """The impedance matrix at every frequency of a problem, ascending."""

    name: str
    ports: tuple[str, ...]
    references: dict[str, str]
    pad_groups: dict[str, tuple[str, ...]]
    """Each pad of the solve and the pads stated that it covers."""
    impedances: tuple[SheetPortImpedance, ...]
    metrics: dict[str, Any] = field(default_factory=dict)
    meshes: tuple[SheetMesh, ...] | None = field(default=None, repr=False, compare=False)
    """The mesh each frequency was solved on, in the order of ``impedances``, with ``keep_solutions``
    (for superposed fields; :attr:`meshes_by_frequency` keys them by frequency)."""
    refinement: dict[str, Any] | None = None
    """With ``frequency_refinement``: the tolerance, whether it was met, the frequencies asked for, and every midpoint check."""

    def sweep(self) -> PortImpedanceSweep:
        """The matrices as an ``electrical.port_network.PortImpedanceSweep`` (with the checks, when refined)."""
        refinement = self.refinement or {}
        checks = [FrequencyCheck(**check) for check in refinement.get("checks", [])]
        return PortImpedanceSweep(self.impedances, checks, tolerance=refinement.get("relative_tolerance"),
                                  converged=refinement.get("converged"))

    @property
    def frequencies_hz(self) -> tuple[float, ...]:
        return tuple(z.frequency_hz for z in self.impedances)

    def at(self, frequency_hz: float) -> SheetPortImpedance:
        """The matrix at ``frequency_hz``, matched within ``electrical.port_network.FREQUENCY_RTOL``."""
        found = match_frequency(frequency_hz, self.frequencies_hz)
        for impedance in self.impedances:
            if found is not None and impedance.frequency_hz == found:
                return impedance
        raise ValueError(f"no matrix at {frequency_hz:g} Hz (have {list(self.frequencies_hz)})")

    @property
    def meshes_by_frequency(self) -> dict[float, SheetMesh]:
        """The mesh of each frequency, by frequency (what ``emc``'s ``network_emission`` takes as ``meshes``)."""
        if self.meshes is None:
            raise ValueError("the meshes were not kept; solve with keep_solutions=True")
        return {float(z.frequency_hz): mesh for z, mesh in zip(self.impedances, self.meshes)}

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": PORT_IMPEDANCE_RESULT_SCHEMA,
            "name": self.name,
            "ports": list(self.ports),
            "references": dict(self.references),
            "pad_groups": {key: list(members) for key, members in self.pad_groups.items()},
            "frequencies": [
                {
                    "frequency_hz": z.frequency_hz,
                    "z_ohm": {"real": np.real(z.z_ohm).tolist(), "imag": np.imag(z.z_ohm).tolist()},
                    "converged": list(z.converged),
                    "iterations": list(z.iterations),
                }
                for z in self.impedances
            ],
            "metrics": self.metrics,
            "refinement": self.refinement,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PortImpedanceResult":
        if str(value.get("schema") or "") != PORT_IMPEDANCE_RESULT_SCHEMA:
            raise ValueError(f"unsupported port-impedance result schema: {value.get('schema')}")
        ports = tuple(str(p) for p in value["ports"])
        references = {str(k): str(v) for k, v in value["references"].items()}
        impedances = []
        for item in value["frequencies"]:
            z = np.asarray(item["z_ohm"]["real"], dtype=float) + 1j * np.asarray(item["z_ohm"]["imag"], dtype=float)
            if z.shape != (len(ports), len(ports)):
                raise ValueError("an impedance matrix does not match the ports")
            impedances.append(SheetPortImpedance(
                ports=ports, references=dict(references), z_ohm=z, frequency_hz=float(item["frequency_hz"]),
                converged=tuple(bool(c) for c in item["converged"]),
                iterations=tuple(int(i) for i in item["iterations"]),
            ))
        return cls(
            name=str(value["name"]),
            ports=ports,
            references=references,
            pad_groups={str(k): tuple(str(m) for m in v) for k, v in value["pad_groups"].items()},
            impedances=tuple(impedances),
            metrics=dict(value.get("metrics") or {}),
            refinement=value.get("refinement"),
        )


def _cut_key(problem: PortImpedanceProblem, frequency_hz: float, settings: Mapping[str, Any]) -> tuple:
    cuts = _filament_cuts(problem.layers, frequency_hz, settings)
    # A barrel's resistance moves with frequency even where the cut does not;
    # a mesh is reused only while every barrel's Rac/Rdc agrees to 1e-9.
    layer_by_name = {layer.name: layer for layer in problem.layers}
    barrels = tuple(
        round(segment.resistance_at(frequency_hz, layer_by_name[segment.upper_layer].resistivity_ohm_m)
              / segment.resistance_ohm, 9)
        for _, _, segment in problem.vertical_segments
        if segment.skin_outer_diameter_mm is not None
    )
    # edge strands are cut above zero frequency only (their width is fixed per sweep)
    stranded = frequency_hz > 0.0 and settings.get("edge_strands", "auto") is not False
    return tuple((layer.name, tuple(cuts[layer.name].layers(layer.name, layer.resistivity_ohm_m)))
                 for layer in problem.layers) + (("barrel_skin", barrels), ("strands", stranded))


def _port_pads(problem: PortImpedanceProblem, filament_of: Mapping[str, tuple[int, ...]]) -> list[PortPad]:
    """The pads of the solve on the mesh: each on its layers' first filament, cells as (row, column)."""
    stated = {pad.name: pad for pad in problem.pads}
    order = {layer.name: layer.order for layer in problem.layers}
    pads = []
    for key, members in problem.pad_groups().items():
        parts = []
        for name in members:
            by_layer: dict[str, list[tuple[int, int]]] = {}
            for layer, x, y in stated[name].cells:
                by_layer.setdefault(layer, []).append((y, x))
            footprint = [(filament_of[layer][0], tuple(cells))
                         for layer, cells in sorted(by_layer.items(), key=lambda item: order[item[0]])]
            parts.append(PortPad(name, footprint[0][0], footprint[0][1], tuple(footprint[1:])))
        pads.append(parts[0] if members == (key,) else merge_pads(key, parts))
    return pads


def solve_port_impedance_problem(
    value: Mapping[str, Any] | PortImpedanceProblem,
    settings: Mapping[str, Any] | None = None,
    *,
    keep_solutions: bool = False,
) -> PortImpedanceResult:
    """Solve ``port-impedance-problem/v1`` at each of its frequencies.

    Settings, as the current-field contract reads them: ``relative_tolerance``
    (1e-9), ``maximum_iterations`` (200), ``restart`` (120),
    ``preconditioner`` (``auto``), ``near_tiles``, ``near_factor_threads``
    (1, or an integer or ``"budget"``), ``cells_per_skin_depth``,
    ``maximum_filaments``, ``operator`` and the ``pfft_*`` keys.
    ``execution_backend`` ``cuda`` (with ``device_id``) solves on the device,
    with no fallback to the CPU; the default is ``cpu``.
    ``keep_solutions`` keeps each port's solve and the mesh it is on, for
    :meth:`SheetPortImpedance.superpose`.

    ``warm_start`` (on by default): a frequency whose mesh an earlier one
    already solved starts each port's Krylov solve from that port's solve at
    the solved frequency nearest to it (in log f), as
    ``solve_port_impedance``'s ``initial`` does.  The matrix is the same to
    the tolerance, its bits depend on which frequencies were solved before
    it, and on a four-layer board of 28k cells and 30 ports it halved the
    solve (``docs/PORT_IMPEDANCE_SCALE_RESULTS.json``).  ``False`` solves
    every frequency from zero.

    ``frequency_refinement`` (``relative_tolerance`` 0.01, ``maximum_solves``
    24, ``minimum_ratio`` 1.05) solves the problem's frequencies and then the
    midpoints :func:`electrical.port_network.refine_frequencies` needs for R
    and L to interpolate within the tolerance between them; the result then
    holds every solved frequency and the checks (``result.refinement``,
    ``result.sweep()``).  Without it only the frequencies stated are solved.
    """
    settings = dict(settings or {})
    backend = str(settings.get("execution_backend", "cpu"))
    if backend not in {"cpu", "cuda"}:
        raise ValueError("execution_backend must be 'cpu' or 'cuda'")
    if settings.get("fallback_backend") is not None:
        raise ValueError("port-impedance solves do not allow implicit backend fallback")
    problem = value if isinstance(value, PortImpedanceProblem) else PortImpedanceProblem.from_mapping(value)
    ports = problem.port_groups()
    # Edge strands start half a skin depth wide; across a sweep they are cut
    # once, for the top frequency, so every frequency sees one mesh and the
    # impedance is a smooth function of frequency (cut per frequency, the
    # mesh changed under the refinement and it did not converge).
    if settings.get("edge_strands", "auto") is not False and settings.get("edge_strand_first_um") is None:
        top = max(problem.frequencies_hz)
        if top > 0.0:
            from electrical.sheet_peec.skin_filaments import skin_depth_m

            resistivity = min(layer.resistivity_ohm_m for layer in problem.layers)
            settings["edge_strand_first_um"] = 0.5e6 * skin_depth_m(top, resistivity)
    solve_arguments = {
        "tolerance": float(settings.get("relative_tolerance", 1e-9)),
        "max_iterations": int(settings.get("maximum_iterations", 200)),
        "restart": int(settings.get("restart", 120)),
        "preconditioner": str(settings.get("preconditioner", "auto")),
        "near_tiles": None if settings.get("near_tiles") is None else int(settings["near_tiles"]),
        "backend": backend,
        "device_id": int(settings.get("device_id", 0)),
        "near_factor_threads": settings.get("near_factor_threads"),
    }

    warm_start = bool(settings.get("warm_start", True))
    refinement = settings.get("frequency_refinement")
    built: dict[tuple, tuple[SheetMesh, Any, list[PortPad]]] = {}
    solved: dict[float, tuple[SheetPortImpedance, SheetMesh, dict[str, Any]]] = {}

    def solve_at(frequency: float) -> SheetPortImpedance:
        started = time.perf_counter()
        key = _cut_key(problem, frequency, settings)
        reused = key in built
        if not reused:
            mesh, operator, geometry = _sheet_geometry(problem, settings, frequency)
            built[key] = (mesh, operator, _port_pads(problem, geometry.filament_of))
        mesh, operator, pads = built[key]
        prepared = time.perf_counter()
        nearest = None
        if warm_start:
            same_mesh = [f for f, (_, other, _) in solved.items() if other is mesh]
            if same_mesh:
                nearest = min(same_mesh, key=lambda f: (abs(math.log(f / frequency)), f))
        impedance = solve_port_impedance(
            mesh, operator, pads, ports, frequency_hz=frequency, keep_solutions=keep_solutions or warm_start,
            initial=None if nearest is None else solved[nearest][0], **solve_arguments
        )
        solved[frequency] = (impedance, mesh, {
            "frequency_hz": frequency,
            "mesh_reused": reused,
            "sheet_layers": len(mesh.stackup),
            "nodes": int(mesh.node_count),
            "branches": int(mesh.branch_count),
            "build_seconds": prepared - started,
            "solve_seconds": time.perf_counter() - prepared,
            "converged": all(impedance.converged),
            "maximum_iterations": max(impedance.iterations),
            "started_from_hz": nearest,
        })
        return impedance

    refined: dict[str, Any] | None = None
    if refinement is None:
        for frequency in problem.frequencies_hz:
            solve_at(frequency)
    else:
        options = dict(refinement)
        unknown = sorted(set(options) - {"relative_tolerance", "maximum_solves", "minimum_ratio"})
        if unknown:
            raise ValueError(f"frequency_refinement takes relative_tolerance, maximum_solves, minimum_ratio; not {unknown}")
        sweep = refine_frequencies(
            solve_at, problem.frequencies_hz,
            tolerance=float(options.get("relative_tolerance", 0.01)),
            maximum_solves=int(options.get("maximum_solves", 24)),
            minimum_ratio=float(options.get("minimum_ratio", 1.05)),
        )
        refined = {
            "relative_tolerance": sweep.tolerance,
            "converged": sweep.converged,
            "requested_hz": list(problem.frequencies_hz),
            "checks": [asdict(check) for check in sweep.checks],
        }
    order = sorted(solved)
    impedances = [solved[f][0] if keep_solutions else replace(solved[f][0], solutions=None) for f in order]
    meshes = [solved[f][1] for f in order]
    records = [solved[f][2] for f in order]
    groups = problem.pad_groups()
    return PortImpedanceResult(
        name=problem.name,
        ports=tuple(ports),
        references=dict(ports),
        pad_groups=groups,
        impedances=tuple(impedances),
        metrics={
            "backend": "sheet_peec",
            "execution_backend": backend,
            "meshes_built": len(built),
            "relative_tolerance": solve_arguments["tolerance"],
            "frequencies": records,
        },
        meshes=tuple(meshes) if keep_solutions else None,
        refinement=refined,
    )


__all__ = [
    "PORT_IMPEDANCE_PROBLEM_SCHEMA",
    "PORT_IMPEDANCE_RESULT_SCHEMA",
    "PortImpedanceProblem",
    "PortImpedanceResult",
    "ProblemPad",
    "solve_port_impedance_problem",
]
