"""Turn rasters and voxel models into the solvers' own input types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from electrical.dice_peec import Stackup, ViaSet
from electrical.dice_peec import ViaSpec as PeecViaSpec
from electrical.sheet_peec.plane_opt_contract import PLANE_OPT_PROBLEM_SCHEMA, PLANE_OPT_PROBLEM_SCHEMA_V2
from thermal.matrix_free_mpir_fem import HeatSource, LayeredThermalMesh, VoxelSolidModel, VoxelThermalMesh

from .bodymap import BodySpec, ResolvedBodies
from .reader import MM
from .section import BoardRaster


def board_stackup(raster: BoardRaster) -> Stackup:
    return Stackup(
        layer_names=raster.spec.layer_names,
        z_mm=tuple(layer.center_z_mm for layer in raster.layers),
    )


def board_occupancy(raster: BoardRaster) -> np.ndarray:
    """The accepted-layout occupancy ``x0`` of shape ``(layers, rows, cols)``."""

    return raster.occupancy


def board_vias(resolved: ResolvedBodies, raster: BoardRaster) -> ViaSet:
    """Via barrels as ``ViaSpec`` entries: cell from the axis, layers from the z extent."""

    layers = raster.layers
    vias = []
    for spec, solids in resolved.vias:
        for solid in solids:
            lo, hi = solid.bounds_m
            cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
            row, col = raster.cell_of(cx, cy)
            spanned = [
                index
                for index, layer in enumerate(layers)
                if lo[2] - 1.0e-9 <= layer.center_z_mm * MM <= hi[2] + 1.0e-9
            ]
            if len(spanned) < 2:
                raise ValueError(f"via {solid.name!r} does not span two layers")
            resistance = spec.resistance_ohm
            if resistance <= 0.0:
                # DC resistance of the barrel from its geometry: length between
                # the two layer centres over the mean cross-section, with the
                # resistivity of the upper layer's copper.
                height = hi[2] - lo[2]
                cross_section = solid.volume_m3 / height
                length = (layers[spanned[-1]].center_z_mm - layers[spanned[0]].center_z_mm) * MM
                resistance = layers[spanned[-1]].resistivity_ohm_m * length / cross_section
            vias.append(
                PeecViaSpec(
                    row=row,
                    col=col,
                    layer_from=spanned[0],
                    layer_to=spanned[-1],
                    resistance_ohm=resistance,
                    inductance_h=spec.inductance_h,
                )
            )
    return ViaSet(tuple(vias))


@dataclass(frozen=True)
class BoardThermalModel:
    """The board's thermal mesh and where each conductor layer sits in it."""

    mesh: LayeredThermalMesh
    layer_slabs: tuple[int, ...]
    slab_names: tuple[str, ...]


def board_thermal_mesh(raster: BoardRaster, *, thickness_source: str = "stackup") -> BoardThermalModel:
    """Copper slabs blended by fill, laminate slabs between them, outline as mask.

    ``thickness_source="measured"`` sizes each copper slab from the copper
    solids' own thickness, keeping the layer centred where the stackup puts it.

    The board's own thickness beyond the outermost copper is not modelled:
    the mesh runs from the bottom of the lowest layer to the top of the
    highest, which is the extent the electrical layers see.
    """

    spec = raster.spec
    rows, cols = raster.shape
    thickness: list[float] = []
    in_plane: list[np.ndarray] = []
    through: list[np.ndarray] = []
    names: list[str] = []
    layer_slabs: list[int] = []
    laminate = np.full((rows, cols), spec.laminate_conductivity_w_per_m_k)
    laminate_z = np.full((rows, cols), spec.laminate_through_conductivity_w_per_m_k)
    def half(index: int) -> float:
        return raster.layer_thickness_mm(index, source=thickness_source) / 2.0

    for index, layer in enumerate(spec.layers):
        if index > 0:
            previous = spec.layers[index - 1]
            gap = (layer.center_z_mm - half(index)) - (previous.center_z_mm + half(index - 1))
            if gap < -1.0e-9:
                raise ValueError(f"layers {previous.name} and {layer.name} overlap with the measured thickness")
            if gap > 1.0e-9:
                thickness.append(gap * MM)
                in_plane.append(laminate)
                through.append(laminate_z)
                names.append(f"laminate:{previous.name}-{layer.name}")
        fill = raster.fill[index]
        copper = spec.copper_conductivity_w_per_m_k
        layer_slabs.append(len(thickness))
        thickness.append(2.0 * half(index) * MM)
        in_plane.append(fill * copper + (1.0 - fill) * laminate)
        # Through the thickness the copper and laminate act in parallel too.
        through.append(fill * copper + (1.0 - fill) * laminate_z)
        names.append(layer.name)
    active = np.broadcast_to(raster.outline[None], (len(thickness), rows, cols)).copy()
    mesh = LayeredThermalMesh(
        slab_thickness_m=tuple(thickness),
        pitch_x_m=raster.pitch_x_m,
        pitch_y_m=raster.pitch_y_m,
        conductivity_w_per_m_k=np.where(active, np.stack(in_plane), 1.0),
        through_plane_conductivity_w_per_m_k=np.where(active, np.stack(through), 1.0),
        active=active,
    )
    return BoardThermalModel(mesh, tuple(layer_slabs), tuple(names))


def body_thermal_mesh(model: VoxelSolidModel, **kwargs: Any) -> VoxelThermalMesh:
    return VoxelThermalMesh.from_solid_model(model, **kwargs)


def body_heat_sources(
    model: VoxelSolidModel, specs: Mapping[int, BodySpec], mesh: LayeredThermalMesh
) -> tuple[HeatSource, ...]:
    """Component power spread over the nodes of each powered body."""

    sources = []
    for material_id, spec in specs.items():
        if spec.power_w <= 0.0:
            continue
        elements = (model.material_id == material_id) & mesh.active  # type: ignore[operator]
        nodes = np.zeros(mesh.node_shape, dtype=bool)
        slabs, rows, cols = elements.shape
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    nodes[dz : dz + slabs, dy : dy + rows, dx : dx + cols] |= elements
        indices = tuple(tuple(int(v) for v in node) for node in np.argwhere(nodes))
        sources.append(HeatSource(indices, spec.power_w, name=spec.material.name))
    return tuple(sources)


def _plane_opt_grid(raster: BoardRaster) -> dict[str, Any]:
    """Schema v1 grid (one pitch) for a uniform raster, v2 grid lines in the array row order otherwise."""

    rows, cols = raster.shape
    if raster.is_uniform:
        return {"rows": rows, "columns": cols, "pitch_mm": raster.pitch_mm}
    grid = raster.grid
    assert grid is not None
    x_edges = grid.x_edges_m / MM
    y_edges = grid.y_edges_m / MM
    if raster.y_down:
        # Rows are stored top-down: the y lines run downwards from the top edge.
        y_edges = (grid.y_edges_m[-1] - grid.y_edges_m[::-1] + grid.y_edges_m[0]) / MM
    return {"rows": rows, "columns": cols, "x_edges_mm": x_edges.tolist(), "y_edges_mm": y_edges.tolist()}


def plane_opt_problem_mapping(
    raster: BoardRaster,
    *,
    terminals: Sequence[Mapping[str, Any]],
    frequency_hz: float = 0.0,
    name: str = "board",
    role: str = "step-geometry",
    vias: ViaSet | None = None,
    thickness_source: str = "stackup",
) -> dict[str, Any]:
    """The ``plane-opt-current-field-problem/v1`` mapping for this board.

    ``terminals`` are passed through as the schema expects them
    (``name``, ``pad``, ``current_a``, ``cells`` of ``layer``/``x``/``y``);
    the front end contributes grid, layers, conductor masks and vias.
    ``thickness_source="measured"`` writes the copper solids' own thickness
    per layer instead of the stackup value.  Layers too thick for a sheet at
    ``frequency_hz`` raise a warning (see :mod:`.skin`); the sheet PEEC's
    filaments handle everything below that limit.
    """

    from .skin import skin_report, warn_if_not_sheet

    rows, cols = raster.shape
    occupancy = raster.occupancy
    if frequency_hz > 0.0:
        warn_if_not_sheet(skin_report(raster, frequency_hz, thickness_source=thickness_source))
    layers = [
        {
            "name": layer.name,
            "order": index,
            "center_z_mm": layer.center_z_mm,
            "thickness_mm": raster.layer_thickness_mm(index, source=thickness_source),
            "resistivity_ohm_m": layer.resistivity_ohm_m,
        }
        for index, layer in enumerate(raster.layers)
    ]
    copper_by_layer = {
        layer.name: [
            {"x": int(col), "y": int(row)} for row, col in np.argwhere(occupancy[index] > 0.0)
        ]
        for index, layer in enumerate(raster.layers)
    }
    connections = []
    for via in (vias.vias if vias is not None else ()):
        segments = [
            {
                "upper_layer": raster.layers[upper].name,
                "lower_layer": raster.layers[upper - 1].name,
                "resistance_ohm": via.resistance_ohm,
                "length_mm": raster.layers[upper].center_z_mm - raster.layers[upper - 1].center_z_mm,
            }
            for upper in range(via.layer_from + 1, via.layer_to + 1)
        ]
        # plane-opt orders "upper" as the front (lower order index) layer.
        for segment in segments:
            segment["upper_layer"], segment["lower_layer"] = segment["lower_layer"], segment["upper_layer"]
        connections.append(
            {"name": f"via_{via.row}_{via.col}", "cell": {"x": via.col, "y": via.row}, "segments": segments}
        )
    return {
        "schema": PLANE_OPT_PROBLEM_SCHEMA if raster.is_uniform else PLANE_OPT_PROBLEM_SCHEMA_V2,
        "name": name,
        "role": role,
        "frequency_hz": float(frequency_hz),
        "grid": _plane_opt_grid(raster),
        "layers": layers,
        "copper_by_layer": copper_by_layer,
        "vertical_connections": connections,
        "terminals": list(terminals),
    }


__all__ = [
    "BoardThermalModel",
    "board_occupancy",
    "board_stackup",
    "board_thermal_mesh",
    "board_vias",
    "body_heat_sources",
    "body_thermal_mesh",
    "plane_opt_problem_mapping",
]
