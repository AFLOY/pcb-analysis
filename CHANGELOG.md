# Changelog

Release notes for [pcb-analysis](https://pypi.org/project/pcb-analysis/).
Each version is published from the `vX.Y.Z` tag by the `Release` workflow.
Versions before 0.7.0 were not tagged; see the Git history.

## 0.8.2

Plated holes reach the electrical problem as the barrels the STEP export
carries. No import path moved.

- `geometry.cad_import.Barrel`, `barrel_of` and `barrels_of` recognise a plated
  barrel (via or through-hole pad) from its solid: axis, outer and drill
  radius, plating thickness, wall cross-section and z span. A solid that is
  not a recognisable barrel is left to the sampler.
- `rasterize_board` fills each barrel's footprint on every layer it spans, and
  `BoardRaster` widens the outline to every copper cell before masking with
  it, so a drill no longer deletes the plating and its annulus. A cell outside
  both the laminate and the copper stays masked.
- `board_barrels(resolved, raster)` keys the barrels by cell, and
  `plane_opt_problem_mapping(..., barrels=)` writes each one onto the matching
  `vertical_connections[*].barrel`.
- `drilled_solid` subtracts solids from a synthetic solid (drilled boards and
  tube-shaped barrels for tests).
- A bore below 1e-4 of the outer radius is treated as a solid pin rather than
  a plating a fraction of a nanometre thick.

## 0.8.1

Version 0.8.1 adds, without moving any import path: surface-to-ambient
radiation and backward-Euler transient conduction in `thermal`; the `σ(T)`
loop around the board/body interface (`ElectroThermalEnclosureScenario`);
graded tensor grids (`electrical.matrix_free_mpir_fem.grid`) in the thermal
and DC solvers, the section rasteriser and the sheet PEEC (precorrected FFT,
`electrical.sheet_peec.sheet_pfft`, with a C++ near-field kernel built like
the other native modules); component-driven grid refinement from KiCad STEP
exports; the current-field problem schema `v2` whose grid carries `x_edges_mm` /
`y_edges_mm` (v1 still accepted); a near-field preconditioner that is now the
default of the sheet solves (`preconditioner="diagonal"` restores the old
one); and a CUDA sheet solve that stops on the requested tolerance.
`LayeredThermalMesh.pitch_x_m` / `pitch_y_m` and `LayeredPCBMesh.pitch_x_m` /
`pitch_y_m` are now per-cell arrays (scalars are still accepted on input).

## 0.8.0

Version 0.8.0 makes `cadquery-ocp` and `pypeec` base dependencies and the
`cuda` extra carries CuPy only. `pip install 'pcb-analysis[cuda]'` still
installs everything it did; `[cad]` is an empty alias.

Version 0.8.0 also splits the electrical package by method: the sheet PEEC
(`sheet_peec`, `sheet_operator`, `sheet_inductance`, `sheet_results`,
`sheet_cuda`, `skin_filaments`, `skin_screen`, `plane_opt_contract`) moved to
`electrical.sheet_peec`; the PyPEEC wrapper (`cuda_pypeec`, `pypeec_memory`)
and the voxel contract (`voxel_peec` → `contract`) moved to
`electrical.voxel_peec`; `electrical.dice_peec` keeps the delta scorer,
`layout_ops`, `controller`, `backends`, `Stackup` and the CLI. The old paths
are not re-exported; update imports as
`from electrical.sheet_peec import solve_sheet_case` and
`from electrical.voxel_peec import CudaPyPeecExecutor`.

## 0.7.0

Version 0.7.0 adds the `geometry` top-level package (STEP input through
OpenCASCADE, optional `cad` extra) and the `native` extra with the CMake build
of the C++ kernels; nothing in `electrical`, `thermal`, `emc` or
`multiphysics` moved, so no import path changes for existing users.

## Before 0.7.0

The package is installed as `electrical`; the former top-level
`peec_fastopt` package now lives under `electrical` (split into `sheet_peec`,
`dice_peec` and `voxel_peec`), and the thermal and EMC front ends and the
coupled scenarios are the separate top-level packages `thermal`, `emc`, and
`multiphysics`. Re-run the editable install after pulling these changes so the
new packages are importable and the old path is not left on `sys.path`.
