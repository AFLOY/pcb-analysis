# 3D voxel PEEC for thick conductors

## Scope

`electrical.dice_peec.voxel_peec` solves conductors that are too thick for a
2.5D sheet (busbars, terminal blocks, heavy copper that `skin_report` marks
`3d`) with PyPEEC's voxel PEEC. The module owns the mapping from arrays to
PyPEEC's `geometry`, `problem` and `tolerance` and back; PyPEEC owns the
assembly and the solve, on the CPU or through `CudaPyPeecExecutor`. The
inputs are a `(nz, ny, nx)` conductor mask with pitch and origin, a
resistivity (one value or a material id per voxel), and lumped terminals as
voxel masks, one of them the reference held at 0 V and the others driven by a
current phasor. The output carries the current density, potential and loss
density per voxel, the terminal currents and voltages, the Joule loss, and
`element_heat_w()` for a `VoxelThermalMesh` on the same grid.

`geometry.step_voxelize.conductors` builds the problem from CAD solids:
`conductor_problem_from_solids` voxelises the solids (a voxel is conductor
when at least half of it is inside), marks the terminal voxels from solids
or boxes of the same model, and returns the matching `VoxelSolidModel`;
`conductor_heat_w` hands the loss to the thermal solve.

## Conventions

Voxel arrays are `(nz, ny, nx)` like the thermal meshes; PyPEEC's linear
index is `x + nx (y + ny z)`. PyPEEC's `P_c` field is a loss density in W/m³;
the Joule loss is its sum times the voxel volume, time-averaged for AC
(`|I|² R / 2` for a peak phasor). The frequency sweep is initialised from a
DC solve when the frequency is not zero.

## Acceptance (measured, adopted)

`experiments/voxel_peec_acceptance.py`, PyPEEC 5.8.0, NumPy
2.3.5; the adopted run is `VOXEL_PEEC_RESULTS.json`. A 20 × 4 × 2 mm
copper busbar with two-voxel terminal slabs at its ends, 10 A.

DC against `ρ L / A` between the terminal midplanes:

| Pitch (mm) | Voxels | R (Ω) | Analytic (Ω) | Error | Solve (s) |
|---|---|---|---|---|---|
| 0.50 | 1280 | 3.9113e-05 | 3.9900e-05 | 2.0 % | 0.2 |
| 0.25 | 10240 | 4.0556e-05 | 4.0950e-05 | 1.0 % | 0.4 |

AC at 0.5 mm pitch:

| f (Hz) | Skin depth (mm) | t / δ | R / R_dc | X (Ω) | Loss vs \|I\|²R/2 |
|---|---|---|---|---|---|
| 1000 | 2.063 | 1.0 | 1.03 | 5.67e-05 | 0.000 % |
| 10000 | 0.652 | 3.1 | 2.13 | 5.14e-04 | 0.001 % |
| 100000 | 0.206 | 9.7 | 4.13 | 4.70e-03 | 0.015 % |
| 1e+06 | 0.065 | 30.7 | 4.55 | 4.67e-02 | 0.023 % |

CUDA (NVIDIA GeForce GTX 1650, CuPy 14.1.1) at 100000 Hz: impedance within
2.1e-08 and loss within 2.6e-09 of the CPU
solve; 0.58 s against 0.17 s on the CPU for this small case, where the
device setup dominates.

The DC error falls from 2 % to 1 % when the pitch halves (voxel current
spreading at the terminal slabs). The AC resistance saturates at about 4.5
× R_dc from 100 kHz on because a 0.5 mm voxel cannot resolve a 65 µm skin
depth; `conductor_problem_from_solids` warns when the pitch exceeds the skin
depth. Resolving the skin effect of a thick conductor at MHz frequencies
needs voxels at or below the skin depth, or a boundary-layer refinement PyPEEC
does not offer; the coarse solve remains a lower bound on the loss.

## Software boundary

| Module | Responsibility |
|---|---|
| `electrical/dice_peec/voxel_peec.py` | `VoxelConductorProblem`, `VoxelTerminal`, PyPEEC mappings, `solve_voxel_peec`, `VoxelPeecSolution` |
| `electrical/dice_peec/cuda_pypeec.py` | CUDA executor the `cuda` backend uses (unchanged) |
| `geometry/step_voxelize/conductors.py` | solids to problem, terminal regions, loss to thermal load |

Import direction unchanged: `geometry` imports `electrical` and `thermal`;
`electrical` imports nothing outside itself.
