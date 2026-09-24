# Geometry & CAD Import (`geometry.cad_import`)

Front end for importing mechanical CAD (STEP formats via OpenCASCADE / `cadquery-ocp`) and converting geometry into structured solver grids and masks.

Owns no field solver. Generates 2.5D rasterized planar occupancies for thin conductors, 3D voxel models for volumetric bodies (heat sinks, enclosures), contact maps between meshes, and component-driven graded tensor grids.

## Key Features

- **2.5D PCB rasterization**: Slices CAD solids at layer heights to produce 2D binary occupancies and stackups on uniform or graded tensor grids.
- **3D body voxelization**: Meshes non-board mechanical bodies into voxel solids, with accelerated point classification via OpenMP C++ kernels (`_voxelize_native`).
- **Contact interface mapping**: Automatically computes shared surface contact areas between the 2.5D board and 3D enclosures or heat sinks for thermal boundary coupling.
- **KiCad integration**: Automated STEP export using `kicad-cli`, stackup and footprint extraction, and component-driven local grid refinement.
- **Clean software boundary**: `reader.py` is the only file that imports OpenCASCADE (`OCP`); all returned data structures are NumPy arrays or standard dataclasses.

---

## Quick Start

### 1. Loading a STEP model and rasterizing a 2.5D board

```python
from geometry.cad_import import (
    BoardSpec,
    BodyMap,
    CopperSpec,
    LayerSpec,
    load_step,
    rasterize_board,
    resolve_bodies,
)

# Load STEP geometry
step_model = load_step("board_assembly.step")

# Define the layer stack (bottom first) and which CAD solids play which role.
# Solids are matched by name with regular expressions; every solid must be
# claimed by one entry or listed in `ignore`.
layers = (
    LayerSpec(name="B.Cu", center_z_mm=-0.0175, thickness_mm=0.035),
    LayerSpec(name="F.Cu", center_z_mm=1.6175, thickness_mm=0.035),
)
body_map = BodyMap(
    board=BoardSpec(solid="board", layers=layers),
    copper=(CopperSpec(solids="bottom_copper", layer="B.Cu"), CopperSpec(solids="top_copper", layer="F.Cu")),
)
resolved = resolve_bodies(step_model, body_map)

# Rasterize onto a 0.2 mm grid
raster = rasterize_board(resolved, body_map.board, pitch_mm=0.2)
print(f"Rasterized grid shape: {raster.occupancy.shape}")  # (n_layers, ny, nx)
```

### 2. Component-driven tensor grid refinement

```python
from geometry.cad_import import board_refined_grid, component_boxes, kicad_component_solids

# Refine under the component bodies of a KiCad STEP export (none in this model)
boxes = component_boxes(kicad_component_solids(step_model), min_size_m=1.0e-3)
grid = board_refined_grid(
    resolved.board,
    coarse_pitch_mm=0.5,
    fine_pitch_mm=0.1,
    boxes=boxes,
)
print(f"Tensor grid: {len(grid.x_edges_m) - 1} x {len(grid.y_edges_m) - 1} elements")
```

## Documentation

- [STEP Geometry Front End Contract & Methods](../../docs/GEOMETRY_CAD_IMPORT.md)
- [Matrix-free MPIR-FEM Contract](../../docs/MATRIX_FREE_MPIR_FEM.md)
- [Sheet PEEC Contract](../../docs/SHEET_PEEC.md)
- [Thermal MPIR-FEM Contract](../../docs/THERMAL_MPIR_FEM.md)
