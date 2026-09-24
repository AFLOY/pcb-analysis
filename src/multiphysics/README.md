# Multiphysics coupling (`multiphysics.staggered_coupling`)

`multiphysics.staggered_coupling` chains the electrical, thermal and EMC
solvers into one analysis. Its central case is the electro-thermal fixed
point: copper conductivity falls as the board heats, which raises the loss,
which heats the board. It owns no solver; a scenario dataclass says what is
coupled, and `run_scenario` runs it.

## Walkthrough: from a KiCad board to a self-consistent hot board

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
    board_barrels, board_thermal_mesh, kicad_step_body_map, layers_from_kicad_stackup,
    load_step, rasterize_board, read_kicad_stackup, resolve_bodies,
)

board = "power_module.kicad_pcb"
layers, board_top_mm = layers_from_kicad_stackup(read_kicad_stackup(board))
body_map = kicad_step_body_map(layers, board_top_z_mm=board_top_mm)
resolved = resolve_bodies(load_step("power_module.step"), body_map)
raster = rasterize_board(resolved, body_map.board, pitch_mm=0.25, y_down=True)
```

### 3. The electrical side: terminals on pads, one DC current case

The coupled loop uses the matrix-free DC solver. Every copper cell is an
element, and a terminal is the nodes at the corners of a pad's cells. Here
5 A enter at J1 pad 3 and leave at Q1 pad 3. Each pad's centre is read
from the KiCad board (in mm), and the nodes of the copper cells under it
become the terminal.
The [electrical README](../electrical/README.md) explains each piece.

```python
from electrical.matrix_free_mpir_fem import CurrentTerminal, LayeredPCBMesh, PCBConductionProblem, ViaConnection

def pad_nodes(layer, x_mm, y_mm, half_width_mm=0.5):
    row, col = raster.cell_of(x_mm * 1e-3, -y_mm * 1e-3)   # the STEP export negates KiCad's y
    k = [spec.name for spec in raster.layers].index(layer)
    r = int(np.ceil(half_width_mm / raster.pitch_mm))
    return tuple({(k, y + dy, x + dx) for y in range(row - r, row + r + 1) for x in range(col - r, col + r + 1)
                  if raster.occupancy[k, y, x] > 0 for dy in (0, 1) for dx in (0, 1)})

mesh = LayeredPCBMesh(
    element_active=raster.occupancy > 0,
    layer_thickness_m=[spec.thickness_mm * 1e-3 for spec in raster.layers],
    pitch_x_m=raster.pitch_x_m,
    pitch_y_m=raster.pitch_y_m,
)
vias = tuple(ViaConnection((0, y, x), (1, y, x), 1.68e-8 * b.height_m / b.wall_area_m2)
             for (y, x), b in board_barrels(resolved, raster).items())
source, sink = pad_nodes("F.Cu", 129.0, 97.58), pad_nodes("F.Cu", 149.26, 95.675)
pcb_problem = PCBConductionProblem(
    mesh, (CurrentTerminal(source, 5.0, "VIN"), CurrentTerminal(sink, -5.0, "SRC")), reference_node=sink[0], vias=vias,
)
```

### 4. The thermal side: mesh and boundaries

`board_thermal_mesh` shares the electrical grid. `layer_slabs` says which
thermal slab holds each copper layer. The Joule loss becomes the heat source
automatically.

```python
from thermal.matrix_free_mpir_fem import ConvectionBoundary

thermal = board_thermal_mesh(raster)
ambient_k = 298.15
convection = (ConvectionBoundary("top", 10.0, ambient_k), ConvectionBoundary("bottom", 10.0, ambient_k))
```

### 5. The scenario that chains them, and run

```python
from multiphysics.staggered_coupling import ElectroThermalScenario, run_scenario

coupled_scenario = ElectroThermalScenario(pcb_problem, thermal.mesh, thermal.layer_slabs, convection=convection)
coupled = run_scenario(coupled_scenario)
```

### 6. Read the result

```python
print(f"converged {coupled.converged} in {coupled.iterations} iterations; "
      f"loss {coupled.cold_joule_loss_w:.3f} W cold -> {coupled.electrical.joule_loss_w:.3f} W hot "
      f"(x{coupled.loss_increase_ratio:.3f}); peak {coupled.thermal.max_temperature_k - ambient_k:.2f} K above ambient")
```

### 7. Add radiated emission of the hot board

`ElectroThermalEmissionScenario` evaluates emission from the converged
current and from the cold one. The current distribution is the DC one,
scaled to each frequency (a quasi-static estimate).

```python
from multiphysics.staggered_coupling import ElectroThermalEmissionScenario, EmissionScenario

heights_m = tuple(spec.center_z_mm * 1e-3 for spec in raster.layers)
chained = run_scenario(ElectroThermalEmissionScenario(coupled_scenario, heights_m, EmissionScenario((30e6, 100e6, 300e6))))
print(f"worst CISPR 32 class B margin {chained.emission.worst_margin_db:+.1f} dB, "
      f"heating shift {np.max(np.abs(chained.heating_shift_db)):.3f} dB")
```

## Scenarios

| Scenario | What it couples |
|---|---|
| `ElectricalScenario`, `ThermalScenario`, `ThermalTransientScenario` | one solver, same entry point |
| `ElectroThermalScenario` | DC conduction and heat conduction, with σ(T) copper and via R(T) |
| `ElectroThermalEnclosureScenario` | the same loop across the contact with separately meshed heat sinks or enclosures |
| `ElectroEmissionScenario` | emission of the cold DC current |
| `ElectroThermalEmissionScenario` | emission of the heated and the cold current |
| `SheetPeecEmissionScenario` | emission from one sheet-PEEC solve per frequency |

`run_scenarios` runs a list of them.

## Further

- Scenarios, the coupling iteration, Aitken relaxation and board/body
  interfaces: [docs/MULTIPHYSICS_SCENARIOS.md](../../docs/MULTIPHYSICS_SCENARIOS.md)
- Solver details: [docs/MATRIX_FREE_MPIR_FEM.md](../../docs/MATRIX_FREE_MPIR_FEM.md),
  [docs/THERMAL_MPIR_FEM.md](../../docs/THERMAL_MPIR_FEM.md),
  [docs/EMC_DIPOLE_SUPERPOSITION.md](../../docs/EMC_DIPOLE_SUPERPOSITION.md)

Other packages: [electrical](../electrical/README.md) ·
[thermal](../thermal/README.md) · [emc](../emc/README.md) ·
[geometry](../geometry/README.md) · [top-level README](../../README.md)
