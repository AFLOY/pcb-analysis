# EMC Analysis (`emc.tiled_dipole_superposition`)

Radiated-emission evaluation front end using exact Hertzian-dipole superposition of solved board currents.

Evaluates near-field probe scans, far-field radiation patterns, total radiated power, net electric/magnetic dipole moments, and compliance margins against regulatory emission limits (CISPR 32, FCC Part 15). Runs on NumPy, CuPy (CUDA), or OpenMP C++ kernels (`_native_dipole`).

## Key Features

- **No separate EM field solve needed**: Reads current distributions directly from DC conduction (`electrical.matrix_free_mpir_fem`) or full-wave sheet PEEC (`electrical.sheet_peec`).
- **Exact near-field scans**: Evaluates exact electric and magnetic fields (including near-zone $1/r^3$ and induction $1/r^2$ terms) across arbitrary scan grids and planes.
- **Far-field radiation patterns**: Computes radiation spherical patterns, maximum field strengths, and integrated total radiated power.
- **Regulatory limits**: Built-in limit lines for CISPR 32 Class A/B and FCC Part 15 Class A/B (at 3 m or 10 m test distances) with margin reporting.
- **Ground plane PEC images**: Analytical image theory support for conductors above an infinite ground plane.

## Quick Start

### Evaluating near-field scans and far-field emission margins

```python
import numpy as np
from emc.tiled_dipole_superposition import (
    CISPR32_CLASS_B,
    CurrentDipoles,
    emission_margin,
    evaluate_fields,
    far_field_pattern,
    scan_plane,
)

# Define a 20 mm current loop at 100 MHz carrying 10 mA (represented as 4 dipoles)
frequency_hz = 100e6
positions_m = np.array([
    [0.00, 0.00, 0.001],
    [0.02, 0.00, 0.001],
    [0.02, 0.02, 0.001],
    [0.00, 0.02, 0.001],
], dtype=np.float64)
# Current moments (I * dl) in Amperes * meters
moments_a_m = np.array([
    [0.02 * 0.01, 0.0, 0.0],
    [0.0, 0.02 * 0.01, 0.0],
    [-0.02 * 0.01, 0.0, 0.0],
    [0.0, -0.02 * 0.01, 0.0],
], dtype=np.complex128)

dipoles = CurrentDipoles(position_m=positions_m, moment_a_m=moments_a_m)

# 1. Near-field magnetic scan 5 mm above the board
probe = scan_plane(
    x_m=np.linspace(-0.01, 0.03, 40),
    y_m=np.linspace(-0.01, 0.03, 40),
    z_m=0.006,
)
near = evaluate_fields(dipoles, probe, frequency_hz, backend="auto")
print(f"Peak near-field H: {near.magnetic_magnitude_a_per_m.max():.3e} A/m")

# 2. Far-field pattern at a 10-meter measurement distance
pattern = far_field_pattern(dipoles, frequency_hz, distance_m=10.0)
print(f"Total radiated power: {pattern.radiated_power_w:.3e} W")
print(f"Peak E-field at 10 m: {pattern.max_polarised_field_v_per_m:.3e} V/m")

# 3. Regulatory margin check against CISPR 32 Class B
margin = emission_margin(
    field_v_per_m=pattern.max_polarised_field_v_per_m,
    frequency_hz=frequency_hz,
    limit=CISPR32_CLASS_B,
    distance_m=10.0,
)
print(f"Predicted: {margin.predicted_dbuv_per_m:.1f} dBµV/m")
print(f"CISPR 32 Class B margin: {margin.margin_db:+.1f} dB ({'PASS' if margin.compliant else 'FAIL'})")
```

### Loading solved currents from electrical solvers

```python
from emc.tiled_dipole_superposition import dipoles_from_pcb_dc, dipoles_from_sheet_peec

# From DC conduction: converts branch currents to dipoles with terminal closure
dipoles = dipoles_from_pcb_dc(
    problem=pcb_problem,
    solution=electrical_dc_solution,
    layer_height_m=(0.0, 1.6e-3),
    close_terminals=True,
)
```

## Documentation

- [EMC Dipole Superposition Contract & Methods](../../docs/EMC_DIPOLE_SUPERPOSITION.md)
- [Coupled Multiphysics Scenarios](../../docs/MULTIPHYSICS_SCENARIOS.md)
