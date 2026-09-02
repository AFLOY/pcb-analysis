"""Consume plane-opt's backend-independent PEEC current-field schema.

This module intentionally imports no ``plane_opt`` code.  The optimizer owns
the board/config/scenario resolution; pcb-analysis receives the serialized
``plane-opt-current-field-problem/v1`` mapping and owns only the physical sheet
mesh and solve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .sheet_operator import SheetInductanceOperator, SheetStackup
from .sheet_peec import SheetMesh, Terminal, ViaBranch, solve_sheet_case
from .sheet_results import cell_current_density_phasor, sheet_fields
from .skin_filaments import filament_links, graded_filaments, skin_depth_m

PLANE_OPT_PROBLEM_SCHEMA = "plane-opt-current-field-problem/v1"
PLANE_OPT_RESULT_SCHEMA = "plane-opt-current-field-result/v1"

NamedNode = tuple[str, int, int]


def _complex_value(value: Any, *, field: str) -> complex:
    if isinstance(value, Mapping):
        number = complex(
            float(value.get("real", 0.0)),
            float(value.get("imag", 0.0)),
        )
    else:
        number = complex(value)
    if not math.isfinite(number.real) or not math.isfinite(number.imag):
        raise ValueError(f"{field} must be finite")
    return number


@dataclass(frozen=True)
class PlaneOptLayer:
    name: str
    order: int
    center_z_mm: float
    thickness_mm: float
    resistivity_ohm_m: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PlaneOptLayer":
        layer = cls(
            name=str(value.get("name") or ""),
            order=int(value["order"]),
            center_z_mm=float(value["center_z_mm"]),
            thickness_mm=float(value["thickness_mm"]),
            resistivity_ohm_m=float(value["resistivity_ohm_m"]),
        )
        if not layer.name:
            raise ValueError("layer name must not be empty")
        if layer.order < 0:
            raise ValueError(f"{layer.name}: layer order must be non-negative")
        if not math.isfinite(layer.center_z_mm):
            raise ValueError(f"{layer.name}: center_z_mm must be finite")
        if layer.thickness_mm <= 0.0 or not math.isfinite(layer.thickness_mm):
            raise ValueError(f"{layer.name}: thickness_mm must be positive")
        if (
            layer.resistivity_ohm_m <= 0.0
            or not math.isfinite(layer.resistivity_ohm_m)
        ):
            raise ValueError(
                f"{layer.name}: resistivity_ohm_m must be positive"
            )
        return layer


@dataclass(frozen=True)
class PlaneOptTerminal:
    name: str
    pad: str
    current_a: complex
    cells: tuple[NamedNode, ...]


@dataclass(frozen=True)
class PlaneOptVerticalSegment:
    upper_layer: str
    lower_layer: str
    resistance_ohm: float
    length_mm: float


@dataclass(frozen=True)
class PlaneOptProblem:
    name: str
    role: str
    frequency_hz: float
    rows: int
    columns: int
    pitch_mm: float
    layers: tuple[PlaneOptLayer, ...]
    copper_by_layer: Mapping[str, frozenset[tuple[int, int]]]
    vertical_segments: tuple[
        tuple[str, tuple[int, int], PlaneOptVerticalSegment], ...
    ]
    terminals: tuple[PlaneOptTerminal, ...]
    source_board_sha256: str | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PlaneOptProblem":
        schema = str(value.get("schema") or "")
        if schema != PLANE_OPT_PROBLEM_SCHEMA:
            raise ValueError(f"unsupported plane-opt problem schema: {schema}")
        grid = value.get("grid") or {}
        rows = int(grid["rows"])
        columns = int(grid["columns"])
        pitch_mm = float(grid["pitch_mm"])
        if (
            rows < 1
            or columns < 1
            or not math.isfinite(pitch_mm)
            or pitch_mm <= 0.0
        ):
            raise ValueError("grid shape and pitch must be positive")
        layers = tuple(
            sorted(
                (
                    PlaneOptLayer.from_mapping(item)
                    for item in (value.get("layers") or [])
                ),
                key=lambda item: item.order,
            )
        )
        if not layers:
            raise ValueError("at least one conductor layer is required")
        if tuple(layer.order for layer in layers) != tuple(range(len(layers))):
            raise ValueError("layers must use contiguous physical order")
        names = tuple(layer.name for layer in layers)
        names_set = set(names)
        if len(names) != len(names_set):
            raise ValueError("layer names must be unique")
        masks = value.get("copper_by_layer") or {}
        if set(masks) != names_set:
            raise ValueError("copper_by_layer must match the resolved layers")
        copper = {
            name: frozenset(
                (int(cell["x"]), int(cell["y"])) for cell in masks[name]
            )
            for name in names
        }
        for name, cells in copper.items():
            if not cells:
                raise ValueError(f"{name}: conductor mask must not be empty")
            if any(
                not (0 <= x < columns and 0 <= y < rows)
                for x, y in cells
            ):
                raise ValueError(f"{name}: conductor cell lies outside the grid")

        terminals: list[PlaneOptTerminal] = []
        for item in value.get("terminals") or []:
            cells = tuple(
                (
                    str(cell["layer"]),
                    int(cell["x"]),
                    int(cell["y"]),
                )
                for cell in item.get("cells") or []
            )
            if not cells:
                raise ValueError(f"terminal {item.get('name')} has no cells")
            if any(layer not in names_set for layer, _, _ in cells):
                raise ValueError(
                    f"terminal {item.get('name')} names an unknown layer"
                )
            if any(
                (x, y) not in copper[layer]
                for layer, x, y in cells
            ):
                raise ValueError(
                    f"terminal {item.get('name')} cell is not conductor"
                )
            name = str(item.get("name") or "")
            pad = str(item.get("pad") or "")
            if not name or not pad:
                raise ValueError("terminal name and pad are required")
            terminals.append(
                PlaneOptTerminal(
                    name=name,
                    pad=pad,
                    current_a=_complex_value(
                        item.get("current_a", 0.0),
                        field=f"terminal {item.get('name')} current",
                    ),
                    cells=cells,
                )
            )
        if len(terminals) < 2:
            raise ValueError("at least two terminals are required")
        if len({terminal.name for terminal in terminals}) != len(terminals):
            raise ValueError("terminal names must be unique")
        if len({terminal.pad for terminal in terminals}) != len(terminals):
            raise ValueError("terminal pads must be unique")
        current_sum = sum(
            (terminal.current_a for terminal in terminals), start=0.0j
        )
        tolerance = float(value.get("current_balance_tolerance_a", 1e-9))
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "current_balance_tolerance_a must be finite and non-negative"
            )
        if abs(current_sum) > tolerance:
            raise ValueError(f"terminal currents sum to {current_sum} A")
        if not any(abs(terminal.current_a) > tolerance for terminal in terminals):
            raise ValueError("terminal excitation must be non-zero")

        vertical: list[
            tuple[str, tuple[int, int], PlaneOptVerticalSegment]
        ] = []
        layer_order = {layer.name: layer.order for layer in layers}
        vertical_keys: set[tuple[int, int, str, str]] = set()
        for connection in value.get("vertical_connections") or []:
            cell_value = connection.get("cell") or {}
            cell = (int(cell_value["x"]), int(cell_value["y"]))
            for segment in connection.get("segments") or []:
                upper = str(segment["upper_layer"])
                lower = str(segment["lower_layer"])
                if upper not in names_set or lower not in names_set:
                    raise ValueError(
                        f"vertical connection {connection.get('name')} "
                        "names an unknown layer"
                    )
                if layer_order[upper] >= layer_order[lower]:
                    raise ValueError(
                        "vertical segment must be ordered front to back"
                    )
                if cell not in copper[upper] or cell not in copper[lower]:
                    raise ValueError(
                        f"vertical connection {connection.get('name')} "
                        "must land on conductor at both layers"
                    )
                resistance = float(segment["resistance_ohm"])
                length = float(segment["length_mm"])
                if (
                    not math.isfinite(resistance)
                    or not math.isfinite(length)
                    or resistance <= 0.0
                    or length <= 0.0
                ):
                    raise ValueError(
                        "vertical segment resistance and length must be positive"
                    )
                key = (cell[0], cell[1], upper, lower)
                if key in vertical_keys:
                    raise ValueError("duplicate vertical connection segment")
                vertical_keys.add(key)
                vertical.append(
                    (
                        str(connection.get("name") or ""),
                        cell,
                        PlaneOptVerticalSegment(
                            upper_layer=upper,
                            lower_layer=lower,
                            resistance_ohm=resistance,
                            length_mm=length,
                        ),
                    )
                )

        problem = cls(
            name=str(value.get("name") or ""),
            role=str(value.get("role") or ""),
            frequency_hz=float(value.get("frequency_hz", 0.0)),
            rows=rows,
            columns=columns,
            pitch_mm=pitch_mm,
            layers=layers,
            copper_by_layer=copper,
            vertical_segments=tuple(vertical),
            terminals=tuple(terminals),
            source_board_sha256=(
                None
                if value.get("source_board_sha256") is None
                else str(value.get("source_board_sha256"))
            ),
        )
        if not problem.name or not problem.role:
            raise ValueError("problem name and role are required")
        if problem.frequency_hz < 0.0 or not math.isfinite(problem.frequency_hz):
            raise ValueError("frequency_hz must be finite and non-negative")
        return problem


@dataclass
class PlaneOptSolveResult:
    """Sheet result mapped back to plane-opt's named nodes."""

    metrics: dict[str, Any]
    voltage: dict[NamedNode, float]
    current_density: dict[NamedNode, float]
    voltage_phasor: dict[NamedNode, complex]
    current_density_phasor: dict[NamedNode, tuple[complex, complex]]


@dataclass(frozen=True)
class _SheetContext:
    problem: PlaneOptProblem
    filament_of: Mapping[str, tuple[int, ...]]
    name_of_filament: Mapping[int, str]
    filament_counts: Mapping[str, int]
    skin_depth_by_layer_m: Mapping[str, float]


def _nearest_filament(
    indices: Sequence[int],
    target_z_m: float,
    stackup: SheetStackup,
) -> int:
    return min(
        indices,
        key=lambda index: abs(stackup.layers[index].z_m - target_z_m),
    )


def build_plane_opt_sheet_inputs(
    value: Mapping[str, Any] | PlaneOptProblem,
    settings: Mapping[str, Any] | None = None,
) -> tuple[SheetMesh, SheetInductanceOperator, list[Terminal], _SheetContext]:
    """Compile schema v1 into the sheet solver's native mesh."""
    problem = (
        value if isinstance(value, PlaneOptProblem) else PlaneOptProblem.from_mapping(value)
    )
    settings = dict(settings or {})
    pitch_m = problem.pitch_mm * 1e-3
    sheet_layers = []
    filament_of: dict[str, tuple[int, ...]] = {}
    cuts = {}
    for layer in problem.layers:
        cut = graded_filaments(
            layer.thickness_mm * 1e-3,
            layer.center_z_mm * 1e-3,
            problem.frequency_hz,
            resistivity_ohm_m=layer.resistivity_ohm_m,
            cells_per_skin_depth=float(
                settings.get("cells_per_skin_depth", 4.0)
            ),
            maximum_filaments=int(settings.get("maximum_filaments", 64)),
        )
        cuts[layer.name] = cut
        indices = tuple(
            range(len(sheet_layers), len(sheet_layers) + len(cut))
        )
        filament_of[layer.name] = indices
        sheet_layers.extend(cut.layers(layer.name, layer.resistivity_ohm_m))
    stackup = SheetStackup(tuple(sheet_layers))
    occupancy = np.zeros(
        (len(stackup), problem.rows, problem.columns), dtype=bool
    )
    for layer_name, cells in problem.copper_by_layer.items():
        for x, y in cells:
            if not (0 <= x < problem.columns and 0 <= y < problem.rows):
                raise ValueError(
                    f"{layer_name}: conductor cell {(x, y)} lies outside the grid"
                )
            for index in filament_of[layer_name]:
                occupancy[index, y, x] = True

    vias: list[ViaBranch] = []
    for layer in problem.layers:
        cut = cuts[layer.name]
        if len(cut) > 1:
            vias.extend(
                filament_links(
                    cut,
                    (problem.rows, problem.columns),
                    pitch_m,
                    first_layer=filament_of[layer.name][0],
                    resistivity_ohm_m=layer.resistivity_ohm_m,
                    occupancy=occupancy,
                )
            )
    layer_by_name = {layer.name: layer for layer in problem.layers}
    seen: set[tuple[int, int, int, int]] = set()
    for _, (x, y), segment in problem.vertical_segments:
        upper_indices = filament_of[segment.upper_layer]
        lower_indices = filament_of[segment.lower_layer]
        upper = _nearest_filament(
            upper_indices,
            layer_by_name[segment.lower_layer].center_z_mm * 1e-3,
            stackup,
        )
        lower = _nearest_filament(
            lower_indices,
            layer_by_name[segment.upper_layer].center_z_mm * 1e-3,
            stackup,
        )
        key = (y, x, lower, upper)
        if key in seen:
            continue
        seen.add(key)
        vias.append(
            ViaBranch(
                row=y,
                col=x,
                lower_layer=lower,
                upper_layer=upper,
                resistance_ohm=segment.resistance_ohm,
            )
        )
    mesh = SheetMesh(
        (problem.rows, problem.columns),
        pitch_m,
        stackup,
        occupancy,
        vias=tuple(vias),
    )
    operator = SheetInductanceOperator(
        mesh.shape,
        pitch_m,
        stackup,
        vertical_levels=mesh.vertical_levels,
    )

    terminals: list[Terminal] = []
    for terminal in problem.terminals:
        by_layer: dict[str, list[tuple[int, int]]] = {}
        for layer, x, y in terminal.cells:
            by_layer.setdefault(layer, []).append((y, x))
        share = terminal.current_a / len(by_layer)
        for layer_name, cells in sorted(
            by_layer.items(),
            key=lambda item: layer_by_name[item[0]].order,
        ):
            terminals.append(
                Terminal(
                    name=f"{terminal.pad}#{layer_name}",
                    layer=filament_of[layer_name][0],
                    cells=tuple(sorted(cells)),
                    current_a=share,
                )
            )
    name_of_filament = {
        index: name
        for name, indices in filament_of.items()
        for index in indices
    }
    context = _SheetContext(
        problem=problem,
        filament_of=filament_of,
        name_of_filament=name_of_filament,
        filament_counts={name: len(indices) for name, indices in filament_of.items()},
        skin_depth_by_layer_m={
            layer.name: skin_depth_m(
                problem.frequency_hz, layer.resistivity_ohm_m
            )
            for layer in problem.layers
        },
    )
    return mesh, operator, terminals, context


def solve_plane_opt_problem(
    value: Mapping[str, Any] | PlaneOptProblem,
    settings: Mapping[str, Any] | None = None,
) -> PlaneOptSolveResult:
    """Solve schema v1 and return fields keyed by original layer names."""
    settings = dict(settings or {})
    execution_backend = str(settings.get("execution_backend", "cpu"))
    if execution_backend not in {"cpu", "cuda"}:
        raise ValueError(
            "execution_backend must be 'cpu' or 'cuda'"
        )
    if settings.get("fallback_backend") is not None:
        raise ValueError(
            "plane-opt sheet solves do not allow implicit backend fallback"
        )
    mesh, operator, terminals, context = build_plane_opt_sheet_inputs(
        value, settings
    )
    solve_arguments = {
        "frequency_hz": context.problem.frequency_hz,
        "tolerance": float(settings.get("relative_tolerance", 1e-9)),
        "max_iterations": int(settings.get("maximum_iterations", 200)),
        "restart": int(settings.get("restart", 120)),
    }
    if execution_backend == "cuda":
        from .sheet_cuda import solve_sheet_case_cuda

        solution, telemetry = solve_sheet_case_cuda(
            mesh,
            operator,
            terminals,
            device_id=int(settings.get("device_id", 0)),
            **solve_arguments,
        )
        backend_metrics = telemetry.as_metrics()
    else:
        solution = solve_sheet_case(
            mesh,
            operator,
            terminals,
            **solve_arguments,
        )
        backend_metrics = {
            "backend": "sheet_peec",
            "resolved_backend": "numpy-scipy-sheet-peec",
            "fallback_used": False,
        }
    fields = sheet_fields(mesh, solution, terminals)
    phasor = cell_current_density_phasor(mesh, solution)

    voltage: dict[NamedNode, float] = {}
    voltage_phasor: dict[NamedNode, complex] = {}
    density: dict[NamedNode, float] = {}
    density_phasor: dict[NamedNode, tuple[complex, complex]] = {}
    for (index, row, col), value_at_cell in fields.current_density.items():
        name = context.name_of_filament[index]
        node = (name, col, row)
        # The optimizer has one cell through a physical copper layer's
        # thickness.  Preserve the peak volume density resolved by the
        # filaments rather than making the gate depend on filament count.
        if value_at_cell >= density.get(node, -1.0):
            density[node] = value_at_cell
            density_phasor[node] = phasor[(index, row, col)]
    for (index, row, col), mesh_index in mesh.node_index.items():
        name = context.name_of_filament[index]
        node = (name, col, row)
        if index == context.filament_of[name][0] or node not in voltage:
            voltage[node] = fields.voltage[(index, row, col)]
            voltage_phasor[node] = complex(solution.node_voltage[mesh_index])

    terminal_nodes = {
        cell
        for terminal in context.problem.terminals
        for cell in terminal.cells
    }
    density_values = np.fromiter(density.values(), dtype=float)
    bulk_density = np.fromiter(
        (
            value_at_cell
            for node, value_at_cell in density.items()
            if node not in terminal_nodes
        ),
        dtype=float,
    )
    metrics = dict(fields.metrics)
    metrics.update(backend_metrics)
    metrics.update(
        {
            "bulk_p99_current_density_a_per_mm2": (
                float(np.percentile(bulk_density, 99.0))
                if bulk_density.size
                else 0.0
            ),
            "max_current_density_a_per_mm2": (
                float(density_values.max()) if density_values.size else 0.0
            ),
            "problem_schema": PLANE_OPT_PROBLEM_SCHEMA,
            "result_schema": PLANE_OPT_RESULT_SCHEMA,
            "problem_name": context.problem.name,
            "role": context.problem.role,
            "source_board_sha256": context.problem.source_board_sha256,
            "resolved_layers": [
                {
                    "name": layer.name,
                    "order": layer.order,
                    "center_z_mm": layer.center_z_mm,
                    "thickness_mm": layer.thickness_mm,
                    "resistivity_ohm_m": layer.resistivity_ohm_m,
                }
                for layer in context.problem.layers
            ],
            "filament_counts": dict(context.filament_counts),
            "skin_depth_by_layer_m": dict(context.skin_depth_by_layer_m),
            "vertical_connection_segment_count": len(
                context.problem.vertical_segments
            ),
            "branch_count": mesh.branch_count,
            "vertical_level_count": len(mesh.vertical_levels),
            "operator_kernel_bytes": operator.kernel_bytes,
            "current_density_definition": (
                "maximum_over_thickness_filaments_of_"
                "two_axis_branch_mean_magnitude"
            ),
            "physical_conductor_cell_count": len(density),
        }
    )
    return PlaneOptSolveResult(
        metrics=metrics,
        voltage=voltage,
        current_density=density,
        voltage_phasor=voltage_phasor,
        current_density_phasor=density_phasor,
    )
