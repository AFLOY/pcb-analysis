# Thermal analysis (`thermal.matrix_free_mpir_fem`)

`thermal.matrix_free_mpir_fem` solves steady and transient heat conduction
through the board's copper and laminate slabs. The result is a temperature
field with a closed heat budget. It uses the same matrix-free,
mixed-precision solver as the electrical FEM, on the CPU or with CUDA.

## Walkthrough: from a KiCad board to a temperature rise

### 1. Export the board from KiCad

```bash
kicad-cli pcb export step --include-tracks --include-pads --include-zones \
  --include-inner-copper --no-extra-pad-thickness --no-components --force \
  --output power_module.step power_module.kicad_pcb
```

### 2. Build the board's thermal mesh

Copper cells get the copper conductivity, blended by how much of each cell is
covered. The laminate slabs sit between them, and the board outline masks the
mesh.

```python
import numpy as np
from geometry.cad_import import (
    board_thermal_mesh, kicad_step_body_map, layers_from_kicad_stackup, load_step,
    rasterize_board, read_kicad_stackup, resolve_bodies,
)

board = "power_module.kicad_pcb"
layers, board_top_mm = layers_from_kicad_stackup(read_kicad_stackup(board))
body_map = kicad_step_body_map(layers, board_top_z_mm=board_top_mm)
resolved = resolve_bodies(load_step("power_module.step"), body_map)
raster = rasterize_board(resolved, body_map.board, pitch_mm=0.25, y_down=True)
thermal = board_thermal_mesh(raster)
mesh = thermal.mesh
print(thermal.slab_names, mesh.node_shape)     # nodes are (z, row, col); z = 0 is the bottom face
```

### 3. Define the input faces: heat sources

A fixed power goes on the top-face nodes under a part. The positions are the
centres KiCad shows, in mm: here the MOSFET tab (Q1 pad 2) and the diode (D1).

```python
from thermal.matrix_free_mpir_fem import HeatSource

def top_nodes(x_mm, y_mm, half_width_mm):
    row, col = raster.cell_of(x_mm * 1e-3, -y_mm * 1e-3)   # the STEP export negates KiCad's y
    r = int(np.ceil(half_width_mm / raster.pitch_mm))
    z = mesh.node_shape[0] - 1
    return tuple((z, y, x) for y in range(row - r, row + r + 2) for x in range(col - r, col + r + 2))

sources = (
    HeatSource(top_nodes(151.8, 107.105, 2.0), power_w=1.5, name="Q1"),
    HeatSource(top_nodes(139.9, 103.6, 1.5), power_w=0.5, name="D1"),
)
```

Heat can also come from an electrical solve. `element_joule_heat_w` and
`via_joule_heat_sources` place a matrix-free DC solution's copper loss on the
slabs. The [multiphysics README](../multiphysics/README.md) does this inside
the coupled loop.

### 4. Define the boundaries

Natural convection on both faces to a 25 °C ambient. `RadiationBoundary`
adds surface-to-ambient radiation, and `fixed_temperature_mask` holds nodes
at a temperature, such as a clamped edge.

```python
from thermal.matrix_free_mpir_fem import ConvectionBoundary, RadiationBoundary, ThermalConductionProblem

ambient_k = 298.15
problem = ThermalConductionProblem(
    mesh,
    convection=(ConvectionBoundary("top", 10.0, ambient_k), ConvectionBoundary("bottom", 10.0, ambient_k)),
    radiation=(RadiationBoundary("top", 0.9, ambient_k),),
    heat_sources=sources,
)
```

### 5. Run

```python
from thermal.matrix_free_mpir_fem import solve_thermal_conduction

solution = solve_thermal_conduction(problem)       # backend="cuda" runs the inner solve on the GPU
```

### 6. Read the result

```python
print(f"peak rise {solution.max_temperature_k - ambient_k:.1f} K, "
      f"heat in {solution.total_heat_input_w:.2f} W, balance error {solution.heat_balance_error_w:.1e} W")
top_face_k = solution.temperature_k[-1]            # (rows + 1, cols + 1) node temperatures of the top face
```

### 7. Transient heating (optional)

A transient solve needs a heat capacity per slab. It marches the problem by
backward Euler, one linear solve per step (and a Newton loop per step when
radiation is on), so the step count sets the cost. This example drops the
radiation and takes four steps over the first 15 s.

```python
import dataclasses
from thermal.matrix_free_mpir_fem import TimeSchedule, solve_thermal_transient

capacity = [3.45e6 if name in ("B.Cu", "F.Cu") else 2.0e6 for name in thermal.slab_names]   # J/(m3 K)
timed = dataclasses.replace(problem, radiation=(),
                            mesh=dataclasses.replace(mesh, volumetric_heat_capacity_j_per_m3_k=capacity))
transient = solve_thermal_transient(timed, TimeSchedule.geometric(1.0, 15.0, growth=2.0), initial_temperature_k=ambient_k)
print([f"{step.time_s:.1f} s: {step.max_temperature_k - ambient_k:.1f} K" for step in transient.history])
```

## Further

[docs/THERMAL_MPIR_FEM.md](../../docs/THERMAL_MPIR_FEM.md) covers:

- the discretisation, the heat budget, and why temperature rise is solved
  rather than absolute temperature;
- radiation (Newton linearisation), transient conduction and time schedules;
- the two-level preconditioner, CUDA and the fused C++ host path;
- graded grids, and the Joule-loss mapping from the electrical solves.

Heat sinks and enclosures meshed as voxel bodies:
[docs/GEOMETRY_CAD_IMPORT.md](../../docs/GEOMETRY_CAD_IMPORT.md) and
[docs/MULTIPHYSICS_SCENARIOS.md](../../docs/MULTIPHYSICS_SCENARIOS.md).

Other packages: [electrical](../electrical/README.md) ·
[emc](../emc/README.md) · [multiphysics](../multiphysics/README.md) ·
[geometry](../geometry/README.md) · [top-level README](../../README.md)
