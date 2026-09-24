# Geometry and CAD import (`geometry.cad_import`)

`geometry.cad_import` reads a STEP file (through OpenCASCADE) and turns it
into the arrays the solvers take: copper occupancy per layer on a grid, plated
barrels, a board thermal mesh and the electrical current-field problem. It owns no
solver. A KiCad export is the recommended input because the stackup and the
barrels come with it.

## Walkthrough: from a KiCad board to a current-field problem

### 1. Export the board from KiCad

```bash
kicad-cli pcb export step --include-tracks --include-pads --include-zones \
  --include-inner-copper --no-extra-pad-thickness --no-components --force \
  --output power_module.step power_module.kicad_pcb
```

- `--include-tracks --include-pads --include-zones` (and
  `--include-inner-copper` for inner layers) write the copper as solids.
- `--no-extra-pad-thickness` keeps pads in their layer's z window.
- Vias and through-holes come out as plated barrels: a thin tube standing in
  the drill. Keep that default representation; `board_barrels` reads it
  (step 5). Do not add `--fill-all-vias`.
- `--no-components` leaves out the 3D part models. If you keep them, add their
  solids to the body map's `ignore` (see `kicad_component_solids`).
- `export_kicad_step(board, "power_module.step", components=False)` runs the
  same export from Python.

### 2. Describe the stackup and the body map

A STEP file does not say which solid is the board and which is copper. The
body map does. For a KiCad export it is built from the `.kicad_pcb` stackup,
because KiCad tells the solids apart by z:

```python
from geometry.cad_import import kicad_step_body_map, layers_from_kicad_stackup, read_kicad_stackup

board = "power_module.kicad_pcb"
layers, board_top_mm = layers_from_kicad_stackup(read_kicad_stackup(board))   # LayerSpec per copper layer, bottom first
body_map = kicad_step_body_map(layers, board_top_z_mm=board_top_mm)
print(body_map.board.layer_names)      # ('B.Cu', 'F.Cu')
```

`body_map` is an ordinary `BodyMap` with three kinds of entry:

- a `BoardSpec`: the laminate solid and its layers;
- one `CopperSpec` per layer: the solids inside that layer's z window;
- a `ViaSpec`: the solids spanning from the lowest layer to the highest.

For a STEP file from another tool, write the same entries yourself. Each one
selects solids by name (a regular expression) and optionally by z window, and
every solid must be claimed by exactly one entry or listed in `ignore`.

### 3. Load the STEP file and sort its solids

```python
from geometry.cad_import import load_step, resolve_bodies

model = load_step("power_module.step")
resolved = resolve_bodies(model, body_map)
print(len(model.solids), "solids;", sum(len(s) for _, s in resolved.vias), "barrels")
```

### 4. Rasterise the board

`y_down=True` numbers rows from the top of the board, as KiCad does.

```python
from geometry.cad_import import rasterize_board

raster = rasterize_board(resolved, body_map.board, pitch_mm=0.25, y_down=True)
print(raster.shape, raster.occupancy.shape)             # (rows, cols), (layers, rows, cols)
print(f"F.Cu copper {raster.copper_area_m2(1) * 1e6:.1f} mm2")
```

`raster.occupancy` is the copper mask per layer, `raster.fill` the covered
fraction of each cell, and `raster.outline` the board area. A point in KiCad
coordinates maps to a cell with `raster.cell_of(x_mm * 1e-3, -y_mm * 1e-3)`,
because the export negates y.

### 5. Plated barrels

Each barrel is read from its solid rather than sampled. The grid cell under
it is copper on every layer it spans, even where the drill removed the
laminate.

```python
from geometry.cad_import import board_barrels, board_vias

barrels = board_barrels(resolved, raster)       # {(row, col): Barrel}
vias = board_vias(resolved, raster)             # the same holes as layer-to-layer connections
cell, barrel = next(iter(barrels.items()))
print(cell, barrel.as_dict())                   # drill and outer diameter, plating, wall area, z span
```

### 6. Input faces and the current-field problem

The electrical solvers take one current-field problem: grid, layers, copper
cells, vertical connections and terminals. A terminal is a set of copper cells
with a total current (positive into the board, summing to zero). The body map
has no pad or terminal entry: each pad's centre is read from the KiCad board
(the pad properties, in mm), and the copper cells under it become the
terminal's cells. This example takes one cell per pad.

```python
from geometry.cad_import import current_field_problem_mapping

def cells_at(layer, x_mm, y_mm):
    row, col = raster.cell_of(x_mm * 1e-3, -y_mm * 1e-3)
    return [{"layer": layer, "x": col, "y": row}]

terminals = [
    {"name": "VIN", "pad": "J1.3", "current_a": 1.0, "cells": cells_at("F.Cu", 129.0, 97.58)},
    {"name": "SRC", "pad": "Q1.3", "current_a": -1.0, "cells": cells_at("F.Cu", 149.26, 95.675)},
]
problem = current_field_problem_mapping(raster, terminals=terminals, vias=vias, barrels=barrels)
print(problem["schema"], len(problem["vertical_connections"]), "connections")
print(problem["vertical_connections"][0]["barrel"])
```

`problem` is what `electrical.sheet_peec.solve_current_field_problem` solves. The
[electrical README](../electrical/README.md) picks up from here with
terminals that cover whole pads.

### 7. The board as a thermal mesh

```python
from geometry.cad_import import board_thermal_mesh

thermal = board_thermal_mesh(raster)
print(thermal.slab_names, thermal.layer_slabs)   # copper and laminate slabs, and where each copper layer sits
```

`thermal.mesh` is a `LayeredThermalMesh`. The
[thermal README](../thermal/README.md) adds heat sources and boundaries to it.

## Further

Details are in [docs/GEOMETRY_CAD_IMPORT.md](../../docs/GEOMETRY_CAD_IMPORT.md):

- 3D part models (`kicad_component_solids`, `read_kicad_footprints`,
  `export_kicad_step(..., model_dir=)`) and component-driven graded grids
  (`component_boxes`, `board_refined_grid`, `rasterize_board(..., grid=)`);
- fused copper (`fuse_shapes=True`), point classification and the native kernels;
- heat sinks and enclosures as voxel bodies (`BodySpec`, `voxelize_bodies`)
  and the board-to-body contact map;
- thick conductors (`skin_report`, `conductor_problem_from_solids`) for the 3D
  voxel PEEC;
- synthetic solids for tests: `box_solid`, `cylinder_solid`, `drilled_solid`,
  `write_step`.

Other packages: [electrical](../electrical/README.md) ·
[thermal](../thermal/README.md) · [emc](../emc/README.md) ·
[multiphysics](../multiphysics/README.md) · [top-level README](../../README.md)
