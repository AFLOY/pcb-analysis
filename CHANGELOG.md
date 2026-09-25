# Changelog

Release notes for [pcb-analysis](https://pypi.org/project/pcb-analysis/).
Each version is published from the `vX.Y.Z` tag by the `Release` workflow.
Versions before 0.7.0 were not tagged; see the Git history.

## 0.9.1

Fused C++ host paths for the electro-thermal coupling, opt-in and result
preserving. No public name changes.

- `electrical.matrix_free_mpir_fem`: `MatrixFreePCBOperator` and
  `solve_pcb_dc` take `native=` and `native_threads=`; the layered DC
  conduction action (float32 and float64), the via links and the whole
  two-level inner PCG run in the new `_layered_dc_native` extension, built
  by `python -m electrical.matrix_free_mpir_fem.native.build` next to the
  Maxwell kernel. `PCB_NATIVE_Q1=1` selects it by default. Same iteration
  histories as the NumPy path; 1.1 to 2.8x end to end on one thread and
  10.7 to 31.6x on sixteen (Xeon Platinum 8581C, `DC_NATIVE_XEON_8581C_RESULTS.json`).
- `thermal.matrix_free_mpir_fem`: with `native=True` the FP64 action (outer
  residual and coarse assembly) also runs in C++
  (`high_operator_backend = "cpp-fused-node-gather-hex-q1-fp64"`);
  construction 1.3 to 2.5x faster, 202k-node solve 612 to 240 ms at sixteen
  threads (`THERMAL_NATIVE_FP64_XEON_8581C_*_RESULTS.json`).
- `multiphysics.staggered_coupling`: `run_electro_thermal(native=,
  native_threads=)` hands one selection to both solvers. A whole coupled
  solve is 1.6 to 5.9x faster on one thread and 2.9 to 14.7x on sixteen
  (`ELECTROTHERMAL_NATIVE_XEON_8581C_RESULTS.json`); the Kicad_PowerOpt
  adopted pipeline on `power_module` goes from 1,895 s to 1,551 s with the
  same candidate (`KICAD_POWEROPT_SYSTEM_BENCHMARK_XEON_8581C_RESULTS.json`).
- `emc.tiled_dipole_superposition`: `dipoles_from_sheet_peec` assembles
  its elements with NumPy indexing, 3.8 to 4.4x faster than the loop and
  bit-identical. A C++ kernel for it matched NumPy within 1.16x and was not
  adopted (`EMC_SOURCES_XEON_8581C_RESULTS.json`).
- New benchmarks `experiments/dc_native_benchmark.py` and
  `experiments/electrothermal_native_benchmark.py`; the thermal benchmark
  also reports construction and FP64 action timings.

## 0.9.0

**Breaking.** The current-field contract drops the `plane_opt` prefix: the
library is generic and the prefix named one of its consumers. Every name and
schema string below was renamed without an alias, so code and serialized
problems written for 0.8.x must be updated. No physics or numerics changed.

| 0.8.x | 0.9.0 |
|---|---|
| module `electrical.sheet_peec.plane_opt_contract` | `electrical.sheet_peec.current_field_contract` |
| `PLANE_OPT_PROBLEM_SCHEMA` | `CURRENT_FIELD_PROBLEM_SCHEMA` |
| `PLANE_OPT_PROBLEM_SCHEMA_V2` | `CURRENT_FIELD_PROBLEM_SCHEMA_V2` |
| `PLANE_OPT_RESULT_SCHEMA` | `CURRENT_FIELD_RESULT_SCHEMA` |
| `PlaneOptLayer` | `CurrentFieldLayer` |
| `PlaneOptProblem` | `CurrentFieldProblem` |
| `PlaneOptSolveResult` | `CurrentFieldSolveResult` |
| `PlaneOptTerminal` | `CurrentFieldTerminal` |
| `PlaneOptVerticalSegment` | `CurrentFieldVerticalSegment` |
| `build_plane_opt_sheet_inputs` | `build_current_field_sheet_inputs` |
| `solve_plane_opt_problem` | `solve_current_field_problem` |
| `geometry.cad_import.plane_opt_problem_mapping` | `geometry.cad_import.current_field_problem_mapping` |
| schema `plane-opt-current-field-problem/v1` | `current-field-problem/v1` |
| schema `plane-opt-current-field-problem/v2` | `current-field-problem/v2` |
| schema `plane-opt-current-field-result/v1` | `current-field-result/v1` |

Migration: replace the names as in the table. `CurrentFieldProblem.from_mapping`
accepts only the new schema strings and raises `ValueError` on the old ones, so
a stored problem JSON needs its `"schema"` field rewritten; the mapping's other
fields are unchanged. `current_field_problem_mapping` and the solve's
`problem_schema` / `result_schema` metrics emit the new strings, so a consumer
that compares them against the `plane-opt-` strings must switch too.

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
