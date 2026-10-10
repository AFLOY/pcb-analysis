# pcb-analysis design

## Source taxonomy

Source code is grouped first by analysis target and then by numerical method
plus acceleration strategy:

```text
src/electrical/{sheet_peec,voxel_peec}/
src/electrical/matrix_free_mpir_fem/
src/thermal/matrix_free_mpir_fem/
src/emc/tiled_dipole_superposition/
src/multiphysics/staggered_coupling/
```

The first directory holds the PEEC solvers: the 2.5D sheet PEEC
([SHEET_PEEC.md](SHEET_PEEC.md)) and the 3D voxel PEEC through PyPEEC
([VOXEL_PEEC.md](VOXEL_PEEC.md)). The
second contains matrix-free FEM accelerated by mixed-precision iterative
refinement (MPIR). The third is the thermal implementation: steady heat
conduction through the board stack on the same MPIR solver and low-precision
runtimes, documented in [THERMAL_MPIR_FEM.md](THERMAL_MPIR_FEM.md). The
solver and runtimes are imported from the electrical package rather than
copied; the discretisations and front ends stay apart. The fourth is the EMC
front end: radiated emissions by tiled superposition of the current elements
either electrical solve produces, documented in
[EMC_DIPOLE_SUPERPOSITION.md](EMC_DIPOLE_SUPERPOSITION.md). It owns no field
solve of its own. The fifth chains the others into coupled scenarios: a
staggered electro-thermal fixed point and the emission of its converged
current, documented in [MULTIPHYSICS_SCENARIOS.md](MULTIPHYSICS_SCENARIOS.md).

## C++ core

The numerical work runs in one C++ extension, `electrical._pcbcore`, built
from `cpp/` with CMake (the sources of the kernels that predate it stay beside
their packages and are compiled into the same module).  The Python packages
are facades: they validate inputs, convert dataclasses to arrays, call the
core with the GIL released and wrap the results; the NumPy (and CuPy)
implementations stay as references and as the path without a build.  The
only threading control is the process-wide budget of `electrical.threads`.

## Implementation status

| Component | Module | Status |
|---|---|---|
| Current-field schema mapping | `sheet_peec/current_field_contract.py` | implemented |
| Sheet PEEC CUDA execution | `sheet_peec/sheet_cuda.py` | implemented; no CPU fallback |
| Physical CUDA PyPEEC path | `voxel_peec/cuda_pypeec.py` | layer-agnostic executor; mapping-dependent |

## Current-field integration contract (used by plane_opt)

The full current-field solve consumes
`current-field-problem/v1` through
`current_field_contract.solve_current_field_problem()`. The mapping contains each
layer's own thickness and center Z, conductor cells, complex terminals,
scenario frequency, and explicit vertical segments; this package does not
import `plane_opt`.

## Matrix-free MPIR-FEM boundary

The electrical FEM path covers real, symmetric DC conduction on layered Q1 PCB
meshes and the complex 2D scalar `E_z` reduction of frequency-domain Maxwell.
The latter includes conductivity, displacement current, dielectric loss, eddy
currents, and skin effect. Outer reliable residuals and updates use host
float64/complex128; inner Jacobi-PCG or Jacobi-GMRES corrections and most
operator actions use float32/complex64. Element and via contributions are
applied directly, so there is no assembled global matrix.

It complements the sheet PEEC rather than replacing it: the full-wave front
end is scalar and z-invariant, not an arbitrary 3D vector PCB model. The algorithm, validation limits, CUDA runtime, and intended Tenstorrent
Blackhole port boundary are specified in
[MATRIX_FREE_MPIR_FEM.md](MATRIX_FREE_MPIR_FEM.md).
