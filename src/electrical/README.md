# Electrical analysis (`electrical`)

`electrical` solves how current spreads through a board's copper: the voltage
drop, current density and loss between the pads you drive, at DC or at a
frequency where skin effect and inductance matter. The main path is the 2.5D
sheet PEEC (`electrical.sheet_peec`). The matrix-free FEM
(`electrical.matrix_free_mpir_fem`) solves the same board at DC and feeds the
thermal and multiphysics packages.

## Walkthrough: from a KiCad board to a solved current

### 1. Export the board from KiCad

The flags write tracks, pads and zones as copper solids. Vias and
through-holes come out as the plated barrels KiCad models. See
[geometry](../geometry/README.md) for the details.

```bash
kicad-cli pcb export step --include-tracks --include-pads --include-zones \
  --include-inner-copper --no-extra-pad-thickness --no-components --force \
  --output power_module.step power_module.kicad_pcb
```

### 2. Put the copper on a grid

```python
import numpy as np
from geometry.cad_import import (
    board_barrels, board_vias, kicad_step_body_map, layers_from_kicad_stackup,
    load_step, current_field_problem_mapping, rasterize_board, read_kicad_stackup, resolve_bodies,
)

board = "power_module.kicad_pcb"
layers, board_top_mm = layers_from_kicad_stackup(read_kicad_stackup(board))
body_map = kicad_step_body_map(layers, board_top_z_mm=board_top_mm)
resolved = resolve_bodies(load_step("power_module.step"), body_map)
raster = rasterize_board(resolved, body_map.board, pitch_mm=0.25, y_down=True)
```

### 3. Define the input faces: terminals on pads

A terminal is a set of copper cells with a total current. Currents are
positive into the board and must sum to zero. The API has no pad object:
each pad's centre is read from the KiCad board (the pad properties, in mm),
and the copper cells under it become the terminal's cells.

```python
def pad_cells(layer, x_mm, y_mm, half_width_mm=0.5):
    """Copper cells of `layer` around a pad centre given in KiCad coordinates."""
    row, col = raster.cell_of(x_mm * 1e-3, -y_mm * 1e-3)   # the STEP export negates KiCad's y
    k = [spec.name for spec in raster.layers].index(layer)
    r = int(np.ceil(half_width_mm / raster.pitch_mm))
    rows, cols = raster.shape
    return [{"layer": layer, "x": c, "y": y}
            for y in range(max(row - r, 0), min(row + r + 1, rows))
            for c in range(max(col - r, 0), min(col + r + 1, cols)) if raster.occupancy[k, y, c] > 0]

terminals = [
    {"name": "VIN", "pad": "J1.3", "current_a": 1.0, "cells": pad_cells("F.Cu", 129.0, 97.58)},
    {"name": "SRC", "pad": "Q1.3", "current_a": -1.0, "cells": pad_cells("F.Cu", 149.26, 95.675)},
]
```

### 4. Define the current cases

One case is one current-field problem (`current_field_problem_mapping`): the
grid, copper, barrels, terminals and a frequency. The DC case gives the
resistive drop. At 100 kHz the copper's inductance starts to redistribute the
current. AC solves are iterative, so they cost more than the direct DC solve
and the cost grows with frequency; on this board both cases finish in a few
seconds on a desktop CPU.

```python
vias, barrels = board_vias(resolved, raster), board_barrels(resolved, raster)
cases = {
    name: current_field_problem_mapping(raster, terminals=terminals, frequency_hz=f, vias=vias, barrels=barrels)
    for name, f in (("dc", 0.0), ("100kHz", 1.0e5))
}
```

### 5. Run

```python
from electrical.sheet_peec import solve_current_field_problem

results = {name: solve_current_field_problem(problem) for name, problem in cases.items()}
```

Pass `{"execution_backend": "cuda"}` as the second argument to solve on the
GPU. The solve never falls back to the CPU silently.

### 6. Read the result

```python
for name, result in results.items():
    m = result.metrics
    print(f"{name}: span {m['voltage_span_v'] * 1e3:.3f} mV, peak J {m['max_current_density_a_per_mm2']:.2f} A/mm2, "
          f"loss {m['i2r_loss_w'] * 1e3:.3f} mW, converged {m['converged']}")
voltage = results["dc"].voltage          # {(layer, y, x): volts} for every copper cell
```

### 7. The same board with the matrix-free FEM

`solve_pcb_dc` solves DC conduction on the same cells. Its solution is what
`thermal` and `multiphysics` take as the Joule-heat source. Each copper cell
is an element, a terminal is the nodes at its cells' corners, and each barrel
becomes a via resistance ρ·h / A of its plating wall.

```python
from electrical.matrix_free_mpir_fem import (
    CurrentTerminal, LayeredPCBMesh, PCBConductionProblem, ViaConnection, solve_pcb_dc,
)

def cell_nodes(cells):
    k = {spec.name: i for i, spec in enumerate(raster.layers)}
    return tuple({(k[c["layer"]], c["y"] + dy, c["x"] + dx) for c in cells for dy in (0, 1) for dx in (0, 1)})

mesh = LayeredPCBMesh(
    element_active=raster.occupancy > 0,
    layer_thickness_m=[spec.thickness_mm * 1e-3 for spec in raster.layers],
    pitch_x_m=raster.pitch_x_m,
    pitch_y_m=raster.pitch_y_m,
)
fem_vias = tuple(
    ViaConnection((0, y, x), (1, y, x), 1.68e-8 * b.height_m / b.wall_area_m2) for (y, x), b in barrels.items()
)
fem_terminals = tuple(CurrentTerminal(cell_nodes(t["cells"]), t["current_a"], t["name"]) for t in terminals)
pcb_problem = PCBConductionProblem(mesh, fem_terminals, reference_node=fem_terminals[-1].nodes[0], vias=fem_vias)
dc = solve_pcb_dc(pcb_problem)
print(f"FEM: drop {np.nanmax(dc.potential_v) - np.nanmin(dc.potential_v):.3e} V, loss {dc.joule_loss_w:.3e} W")
```

## Further

- Sheet PEEC method, checks, vertical branches, thick copper and CUDA:
  [docs/SHEET_PEEC.md](../../docs/SHEET_PEEC.md)
- Graded grids and the problem schema `v2` (`rasterize_board(..., grid=)`):
  [docs/SHEET_PEEC.md](../../docs/SHEET_PEEC.md#graded-grids-precorrected-fft)
- Matrix-free FEM, 2D frequency-domain Maxwell, MPIR, preconditioners and CUDA:
  [docs/MATRIX_FREE_MPIR_FEM.md](../../docs/MATRIX_FREE_MPIR_FEM.md)
- 3D voxel PEEC (PyPEEC) for conductors too thick for a sheet:
  [docs/VOXEL_PEEC.md](../../docs/VOXEL_PEEC.md)
- DICE delta scoring for ranking candidate edits: [docs/DESIGN.md](../../docs/DESIGN.md)
- Requirements the solvers are held to: [docs/REQUIREMENTS.md](../../docs/REQUIREMENTS.md)

Other packages: [thermal](../thermal/README.md) ·
[emc](../emc/README.md) · [multiphysics](../multiphysics/README.md) ·
[geometry](../geometry/README.md) · [top-level README](../../README.md)
