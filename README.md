# Accelerated electrical, thermal, EMC, and coupled PCB analysis

[![CI](https://github.com/AFLOY/pcb-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/AFLOY/pcb-analysis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Accelerated PEEC and matrix-free FEM solvers for electrical and thermal PCB
analysis, radiated-emission evaluation for EMC, and PDN optimization.

The repository has three solver families: `electrical`, `thermal` and `emc`.

These solvers are designed to analyse PCBs faster, so that they can be embedded
in an automated, LLM-driven PCB design workflow.

## Features

- **electrical**: 2.5D sheet PEEC for current density, voltage drop and
  skin effect on multilayer copper; matrix-free FEM for DC conduction and 2D
  frequency-domain Maxwell; 3D voxel PEEC (PyPEEC) for thick conductors;
  DICE delta scoring to rank many candidate edits without a full solve.
- **thermal**: steady and transient heat conduction through the board stack,
  with convection, radiation and fixed-temperature boundaries.
- **emc**: near-field scans, far-field patterns and CISPR 32 / FCC Part 15
  margins computed from a solved current distribution.
- **multiphysics**: electro-thermal fixed point with temperature-dependent
  copper, and emission of the cold or heated current.
- **geometry**: reads a KiCad (or any mechanical) STEP export into the
  solvers' grids, keeping the plated via and through-hole barrels.
- NumPy on the CPU everywhere; CUDA (CuPy) and OpenMP C++ kernels where they
  pay off.

## Install

```console
pip install pcb-analysis            # CPU only
pip install 'pcb-analysis[cuda]'    # adds CuPy for the CUDA paths
```

Requirements:

- Python 3.11 or later. The base install includes OpenCASCADE (`cadquery-ocp`) and PyPEEC.
- Optional: an NVIDIA driver with CUDA 13.x for `[cuda]`. Run `nvidia-smi` to check it.
- `kicad-cli` (KiCad 8 or later) for the KiCad path below.

Each release is also attached to its GitHub release (wheel, sdist and
SHA-256 checksums). To install one without PyPI:

```console
pip install 'git+https://github.com/AFLOY/pcb-analysis.git@v0.8.2'
```

## How to use: start from KiCad

KiCad is the recommended environment. Design the board there, export it as
STEP with its copper, and let pcb-analysis read the copper back onto a grid.

**1. Export the board with its copper.**
`--include-tracks`, `--include-pads` and `--include-zones` write the copper
as solids. Vias and through-holes then come out as the plated barrels KiCad
models, and pcb-analysis reads those barrels as they are, so do not add
`--fill-all-vias`.

```bash
kicad-cli pcb export step --include-tracks --include-pads --include-zones \
  --include-inner-copper --no-extra-pad-thickness --no-components --force \
  --output power_module.step power_module.kicad_pcb
```

**2. Solve it.** The stackup comes from the `.kicad_pcb`. The body map sorts
the solids into the board, one copper sheet per layer and the barrels. Two
pads act as the input faces: 1 A enters at the input connector's VIN pin
(J1 pad 3) and leaves at the MOSFET's source pad (Q1 pad 3). The API has no
pad object: each pad's centre is read from the KiCad board (the pad
properties, in mm), and the copper cells under it become the terminal's cells.
`plane_opt_problem_mapping` collects the grid, copper, barrels and terminals
into one current-field problem.

```python
import numpy as np
from geometry.cad_import import (
    board_barrels, board_vias, kicad_step_body_map, layers_from_kicad_stackup,
    load_step, plane_opt_problem_mapping, rasterize_board, read_kicad_stackup, resolve_bodies,
)
from electrical.sheet_peec import solve_plane_opt_problem

board = "power_module.kicad_pcb"
layers, board_top_mm = layers_from_kicad_stackup(read_kicad_stackup(board))
body_map = kicad_step_body_map(layers, board_top_z_mm=board_top_mm)   # board, copper layers, barrels
resolved = resolve_bodies(load_step("power_module.step"), body_map)
raster = rasterize_board(resolved, body_map.board, pitch_mm=0.25, y_down=True)

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
problem = plane_opt_problem_mapping(raster, terminals=terminals, frequency_hz=0.0,
                                    vias=board_vias(resolved, raster), barrels=board_barrels(resolved, raster))
result = solve_plane_opt_problem(problem)             # DC sheet PEEC on the CPU
print(f"voltage span {result.metrics['voltage_span_v'] * 1e3:.3f} mV, "
      f"peak current density {result.metrics['max_current_density_a_per_mm2']:.2f} A/mm2")
```

On the example board (two layers, a 139 × 139 grid) this takes a few seconds
on a desktop CPU.

## Next steps

Each package README continues from the same export, one step at a time:

| Package | What you get |
|---|---|
| [electrical](https://github.com/AFLOY/pcb-analysis/blob/main/src/electrical/README.md) | terminals and current cases, DC and AC sheet PEEC, the matrix-free FEM |
| [thermal](https://github.com/AFLOY/pcb-analysis/blob/main/src/thermal/README.md) | board thermal mesh, heat sources, convection, temperature rise |
| [emc](https://github.com/AFLOY/pcb-analysis/blob/main/src/emc/README.md) | radiated field and limit margin from the solved current |
| [multiphysics](https://github.com/AFLOY/pcb-analysis/blob/main/src/multiphysics/README.md) | electro-thermal coupling and emission of the heated board |
| [geometry](https://github.com/AFLOY/pcb-analysis/blob/main/src/geometry/README.md) | body maps, rasterising, plated barrels, the current-field problem |

## Documentation

The methods, their limits and the measurements behind each design decision:

- [Architecture and design](https://github.com/AFLOY/pcb-analysis/blob/main/docs/DESIGN.md) and [requirements](https://github.com/AFLOY/pcb-analysis/blob/main/docs/REQUIREMENTS.md)
- [Sheet PEEC](https://github.com/AFLOY/pcb-analysis/blob/main/docs/SHEET_PEEC.md), [matrix-free MPIR-FEM](https://github.com/AFLOY/pcb-analysis/blob/main/docs/MATRIX_FREE_MPIR_FEM.md),
  [3D voxel PEEC](https://github.com/AFLOY/pcb-analysis/blob/main/docs/VOXEL_PEEC.md)
- [Thermal MPIR-FEM](https://github.com/AFLOY/pcb-analysis/blob/main/docs/THERMAL_MPIR_FEM.md)
- [EMC dipole superposition](https://github.com/AFLOY/pcb-analysis/blob/main/docs/EMC_DIPOLE_SUPERPOSITION.md)
- [Coupled scenarios](https://github.com/AFLOY/pcb-analysis/blob/main/docs/MULTIPHYSICS_SCENARIOS.md)
- [STEP and KiCad input](https://github.com/AFLOY/pcb-analysis/blob/main/docs/GEOMETRY_CAD_IMPORT.md)
- Reports: [sheet PEEC versus matrix-free FEM](https://github.com/AFLOY/pcb-analysis/blob/main/docs/PEEC_FEM_COMPARISON_REPORT.html),
  [Maxwell accuracy and CPU timing](https://github.com/AFLOY/pcb-analysis/blob/main/docs/MAXWELL_VALIDATION_REPORT.html)

Release and migration notes are in
[CHANGELOG.md](https://github.com/AFLOY/pcb-analysis/blob/main/CHANGELOG.md).

## Development

```console
pip install -e '.[test]'              # from a clone; CPU only, as CI runs it
python -m pytest tests/
pip install -e '.[native]'            # optional C++ kernels (pybind11, OpenMP)
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native
```

Tests that need the C++ kernels are skipped until they are built.
`CMakeLists.txt` lists the native targets; `-DPCB_NATIVE_OPENMP=OFF` drops
OpenMP and `-DPCB_NATIVE_MARCH=x86-64-v3` (for example) replaces
`-march=native`. The scripts in `experiments/` regenerate the numbers quoted
in `docs/`; run them from the repository root.

## License

[MIT](https://github.com/AFLOY/pcb-analysis/blob/main/LICENSE)
