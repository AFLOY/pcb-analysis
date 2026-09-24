# EMC analysis (`emc.tiled_dipole_superposition`)

`emc.tiled_dipole_superposition` turns a solved current distribution into
what an EMC engineer reads: the near field on a scan plane, the far-field
pattern and radiated power, and the margin to the CISPR 32 or FCC Part 15
limit lines. Every current element of the solve is a Hertzian dipole, and
their exact fields are summed on NumPy, CuPy or an OpenMP C++ kernel. There
is no separate field solve.

## Walkthrough: from a KiCad board to an emission margin

### 1. Export the board from KiCad

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

### 3. Define the input faces: the current at one frequency

The emission comes from the current the board carries at the frequency of
interest. Drive the pads with that harmonic's amplitude, here 0.1 A from
J1 pad 3 to Q1 pad 3. As in the [electrical README](../electrical/README.md),
each pad's centre is read from the KiCad board and the copper cells under it
become the terminal. Solve it with the sheet PEEC. A DC solve
(`solve_frequency_hz = 0.0`) gives the current's path in a second, and the
fields are then evaluated at 30 MHz, the lower edge of the CISPR 32 radiated
limits (a quasi-static estimate). Set `solve_frequency_hz = frequency_hz` to
include the copper's inductance; that solve is iterative, and at tens of MHz
it can take many minutes on a CPU.

```python
from electrical.sheet_peec import build_current_field_sheet_inputs, solve_sheet_case

def pad_cells(layer, x_mm, y_mm, half_width_mm=0.5):
    row, col = raster.cell_of(x_mm * 1e-3, -y_mm * 1e-3)   # the STEP export negates KiCad's y
    k = [spec.name for spec in raster.layers].index(layer)
    r = int(np.ceil(half_width_mm / raster.pitch_mm))
    return [{"layer": layer, "x": x, "y": y} for y in range(row - r, row + r + 1)
            for x in range(col - r, col + r + 1) if raster.occupancy[k, y, x] > 0]

frequency_hz, amplitude_a = 30e6, 0.1
solve_frequency_hz = 0.0
terminals = [
    {"name": "VIN", "pad": "J1.3", "current_a": amplitude_a, "cells": pad_cells("F.Cu", 129.0, 97.58)},
    {"name": "SRC", "pad": "Q1.3", "current_a": -amplitude_a, "cells": pad_cells("F.Cu", 149.26, 95.675)},
]
problem = current_field_problem_mapping(raster, terminals=terminals, frequency_hz=solve_frequency_hz,
                                    vias=board_vias(resolved, raster), barrels=board_barrels(resolved, raster))
sheet_mesh, operator, sheet_terminals, _ = build_current_field_sheet_inputs(problem)
current = solve_sheet_case(sheet_mesh, operator, sheet_terminals, frequency_hz=solve_frequency_hz)
```

### 4. Current to dipoles

```python
from emc.tiled_dipole_superposition import dipoles_from_sheet_peec

dipoles = dipoles_from_sheet_peec(sheet_mesh, current)
print(dipoles.position_m.shape[0], "current elements")
```

### 5. Evaluate: near-field scan and far field

```python
from emc.tiled_dipole_superposition import evaluate_fields, far_field_pattern, scan_plane

lo, hi = dipoles.position_m.min(axis=0), dipoles.position_m.max(axis=0)
probe = scan_plane(np.linspace(lo[0], hi[0], 30), np.linspace(lo[1], hi[1], 30), hi[2] + 5e-3)  # 5 mm above
near = evaluate_fields(dipoles, probe, frequency_hz, electric=False)   # H only; drop electric= for E too
pattern = far_field_pattern(dipoles, frequency_hz, distance_m=10.0)
```

`backend="cuda"` or `"auto"` runs both on the GPU.

### 6. Read the result against the limits

```python
from emc.tiled_dipole_superposition import CISPR32_CLASS_B, emission_margin

margin = emission_margin(pattern.max_polarised_field_v_per_m, frequency_hz, CISPR32_CLASS_B, distance_m=10.0)
print(f"peak H 5 mm above: {near.magnetic_magnitude_a_per_m.max():.3e} A/m; "
      f"radiated power {pattern.radiated_power_w:.3e} W")
print(f"{margin.predicted_dbuv_per_m:.1f} dBuV/m at 10 m, limit {margin.limit_dbuv_per_m:.1f}, "
      f"margin {margin.margin_db:+.1f} dB, compliant {margin.compliant}")
```

The other limit lines are `CISPR32_CLASS_A`, `FCC_PART15_CLASS_A` and
`FCC_PART15_CLASS_B`. `dipoles_from_pcb_dc` builds dipoles from a matrix-free
DC solution instead. The [multiphysics README](../multiphysics/README.md)
runs that for a heated board.

## Further

[docs/EMC_DIPOLE_SUPERPOSITION.md](../../docs/EMC_DIPOLE_SUPERPOSITION.md)
covers:

- the method and closing the current at the terminals;
- PEC ground-plane images (`ground_plane_z_m`) and dipole-moment proxies;
- accuracy limits (a current-only description, no dielectric or cable
  radiation);
- cost, tiling, CUDA and the native kernel.

Other packages: [electrical](../electrical/README.md) ·
[thermal](../thermal/README.md) · [multiphysics](../multiphysics/README.md) ·
[geometry](../geometry/README.md) · [top-level README](../../README.md)
