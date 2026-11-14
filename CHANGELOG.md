# Changelog

Release notes for [pcb-analysis](https://pypi.org/project/pcb-analysis/).
Each version is published from the `vX.Y.Z` tag by the `Release` workflow.
Versions before 0.7.0 were not tagged; see the Git history.

## Unreleased

- `solve_current_field_problems` solves the conductors of one call side by
  side in worker processes (`spawn`), each on an even share of the thread
  budget and of the free memory, as many as fit in half of `MemAvailable` by
  `conductor_bytes` (4.6 + 3.4 n KB per saddle-point unknown with n filaments
  a layer, from near + MUMPS's measured peaks; 1.5 times that with SuperLU).
  Each result is the bits its conductor gets alone and carries
  `conductor_width`.  Below 100k unknowns in the largest conductor the call
  stays in the caller's process.
- The uniform grid's two edge corrections are on by default: `edge_strands`
  defaults to `"auto"` (cut on the CPU with the uniform operator and the near
  preconditioner, above zero frequency; `True` refuses a setup without them,
  `False` leaves them out), the `auto` preconditioner takes `near` for a
  stranded mesh, and `rasterize_board(faces="auto")` records the faces' copper
  coverage on a uniform grid when the geometry native extension is built.
  Results above zero frequency change: a 1.2 mm strip pair at 0.1 mm was 22 %
  (30 MHz) and 27 % (150 MHz) low without strands, a 45 degree strip 5 % high
  even at 1 MHz without the faces; with both, within about 2 % of a 2D FEM
  (45 degrees at 150 MHz: -4.8 %).
  The metrics record `edge_strands` and `face_coverage`; a port-impedance
  sweep cuts its strands once, for its top frequency; the far-field dipoles
  of a stranded mesh sum each branch's strands.  Vias on the uniform grid
  were checked against a 3D FEM: within -1.14..+0.34 % at 0.1 mm, DC-1 MHz,
  and a via array's current sharing within 3 % (docs/SHEET_PEEC.md).
- Sheet solves can correct the two edge errors of the uniform grid, both
  off by default (docs/SHEET_PEEC.md, "Edges: strands and face coverage"):
  `edge_strands` cuts the branches along an edge across their width into
  strands graded from half a skin depth, kept as repeating blocks in the C++
  core (R of a strip pair within -2.7..+0.4 % of a 2D FEM at 30-150 MHz,
  against 12-27 % low without); `rasterize_board(faces=True)` records each
  grid face's copper coverage, which the problem carries as
  `copper_faces_by_layer` and the mesh uses to narrow branches (a 45 degree
  strip's DC resistance +6.1 % -> -0.3 %).
- The AC `voltage_span_v` of a sheet solve is the largest phasor difference
  between two nodes, max |V_i - V_j| (the diameter of the phasors,
  `voltage_span_definition` `maximum_phasor_difference`), not the departure
  from the first node of the component in mesh order, which moved with the
  mesh: on a two-layer EVM's real fill a sliver cell moved the VIN and SW
  spans by +56 % and -35 % with the loss within 0.7 %; the diameter moved by
  under 0.4 %.  An AC span is now at least the old one, so gates reading it
  see larger values.  DC keeps max - min.
- Fixed: the FP32 inner PCG of the pure-Python matrix-free solves (no C++
  core) refused a well-posed system as not positive definite once the outer
  residual fell to ~1e-20, its dot products underflowing; a board held at
  its ambient by a stiff contact did that.  The correction's right-hand side
  is scaled by a power of two first, so other results keep their bits.
- Board-enclosure coupling can solve each body with a Robin condition to the
  board face's temperature (`InterfaceCouplingConfig.scheme="body_robin"`,
  `ContactMap.body_robin`, `ContactMap.board_nodal_heat_w`) instead of
  loading it with the interface heat: a part that loses little to its own
  film no longer swings by P / G_film on the first step, which made the
  exchange diverge on a two-layer buck EVM.  The default stays
  `"board_robin"`, so existing results do not move.

## 0.13.0

The sheet solve's defaults change: `preconditioner="auto"` is `near` at every
size where the core links MUMPS, and results are promised the same bits only
at the same thread counts.  Vias gain their skin effect and report the
current of each connection.

Migration: a caller that compared results to the bit across thread budgets
keeps the thread budget and `near_factor_threads` the same (they still agree
at the default of one MUMPS thread, but that is no longer promised).
`preconditioner="block"` gives the previous large-system solve.  A pure-Python
install (the PyPI wheel) factors `near` by SuperLU and keeps `block` from
400,000 unknowns, as before.

- A vertical connection's `barrel` or new `resistance_barrel` section scales
  its stated resistance by the plated tube's Rac/Rdc at the solve frequency
  (`skin_filaments.tube_internal_impedance`, `tube_resistance_ratio`): for a
  0.254 mm drill plated 25 um, 1.0016 at 1 MHz, 1.88 at 30 MHz, 4.27 at
  150 MHz.  `resistance_barrel` leaves the inductance to the cells' columns,
  for a barrel laid on the cells its wall runs through.  DC is unchanged.
- The current-field metrics gain `max_via_current_a` and
  `via_current_a_by_connection`: the branches of one named vertical
  connection are summed, so a barrel laid on several cells reports what the
  whole via carries.  A branch without a name is its own connection.
- The near factor of the sheet solve (`near_factorization`, default
  `"auto"`) is MUMPS where the core is built with `-DPCB_NATIVE_MUMPS=ON`
  (`cpp/third_party/build_mumps.sh`), SciPy's SuperLU otherwise, and with
  MUMPS `preconditioner="auto"` is `near` at every size: solves above
  400,000 saddle-point unknowns, which ran `block`, now run `near` and their
  bits move within the tolerance.  With SuperLU `auto` still takes `block`
  from 400,000.  On a filled two-layer board's ground conductor (1.05M branches,
  214 kHz) `near` with MUMPS took 84.5 s where `block` took 2,233.6 s and
  SuperLU ran out of 86 GB (docs/SHEET_PEEC.md).
- The MUMPS factor runs on one thread unless `near_factor_threads` (an
  integer, or `"budget"`; solve functions, current-field and port-impedance
  settings) asks for more.  Results are now promised the same bits only at
  the same thread counts (docs/DESIGN.md); with the default every thread
  budget still gives the same bits.
- The near preconditioner keeps the couplings between the skin filaments of
  one layer, which it had dropped as couplings between layers; the solution
  is unchanged and a cut mesh converges in far fewer iterations.
- The C++ sheet operator (`ConvolutionOperator`) applies the sets of a
  lockstep step together, keeps its spectra between applications instead of
  mapping fresh pages each time, transforms only the rows that carry data and
  spreads every plane over the thread budget.  On a hot-loop-shaped operator
  (8 layers, 7 levels, 225 x 320) ten sets on 8 threads take 19.94 ms a set
  where they took 58.29 (GCC 12.5.0); results are identical to the bit
  (`docs/SHEET_OPERATOR_BATCHING_RESULTS.json`).  A solve keeps up to 1 GiB
  of spectra in the operator while it lives (about 0.5 GB on that hot loop).
- `electrical.backend` sets `KMP_BLOCKTIME=0` before the core loads unless
  the environment sets it: a core built with Clang or AOCC runs on LLVM's
  libomp, whose threads otherwise spin for 200 ms after each parallel
  region and slow the solver's own thread pools.  GCC's libgomp ignores it.
- The core builds with AOCC 5.2.0 (Clang 17), which lacks
  `-fcx-limited-range`: the operator's complex products are spelled out.

## 0.12.1

Fixes from Kicad_PowerOpt's review of 0.12.0. Port-impedance solves on a
grid of oblong cells move, and `refine_frequencies` may take more solves
where the ports differ in size, its tolerance now being per entry.

- Equal oblong cells (`x_edges_mm` and `y_edges_mm` at different pitches,
  0.2 x 0.4 mm say) were solved as squares of their x pitch: the 5 x 40 cell
  strips' inductance was 4.98 nH, 3.99 nH as oblong cells (3.97 nH on the
  same copper in 0.2 mm squares). `TensorGrid.is_uniform` now asks for one
  square cell, so such grids take the pFFT operator and schema v2 as graded
  ones do, and `SheetMesh` refuses a `pitch_m` its grid's cells do not all
  have. The fault predates 0.12.0; every port-impedance solve on such a
  grid moves.
- `BoardNetwork` refuses a pad that is neither a copper port nor a reference
  and joins one element only: that element carried no current (1e-17 A)
  without a word, as when a part was stamped on `C2.1` where `merge_pins`
  names the copper's pad `C2:VIN`. The message names the copper's pads of
  the same part. A pad of the parts alone joined by two or more elements is
  a node as before.
- `dipoles_from_sheet_peec` places a uniform grid's elements from its first
  grid lines (`mesh.grid.origin_m`), as it did a graded grid's and as
  `PortImpedanceProblem.pad_positions_m()` places the pads. A board stated
  on grid lines off the origin had its parts' elements off the copper by
  that origin in `network_dipoles` and `network_emission`.
- `network_emission` takes the meshes as one `SheetMesh` or a mapping by
  frequency; `PortImpedanceResult.meshes_by_frequency` gives the mapping.
  Its docstring named `PortImpedanceResult.meshes`, a tuple without
  frequencies that then failed; that tuple is now refused with a pointer.
- Frequencies are matched within `FREQUENCY_RTOL` (1e-9 relative,
  `electrical.port_network.match_frequency`), not to the bit: by the
  sources' spectra (a missed key carried 0 A without a word), `BoardNetwork`'s
  copper and its cases, `PortImpedanceSweep.at`, `PortImpedanceResult.at`
  and `network_emission`. A solution reports the copper's own frequency.
- `NetworkSolution.copper_converged` (also on `NetworkTransfer` and
  `SweepPoint.converged`) says whether the port solves behind the copper's
  matrix converged; a matrix from solves that did not converge raises a
  `RuntimeWarning` once per frequency. `None` where the copper has no flags.
- `refine_frequencies` no longer reports a sweep unconverged because two
  frequencies asked for are closer than `minimum_ratio`; a split that gets
  that narrow still does.
- The refinement's tolerance is per entry against the entry's own ports,
  `|dX_ij| / sqrt(X_ii X_jj)` (`relative_errors`, public), not against the
  largest entry of the matrix, where a small port was checked against the
  largest one. `FrequencyCheck` errors are on this measure. Measured again
  (`docs/PORT_SWEEP_RESULTS.json`): the default forms stay; on the two
  coppers measured only the strips' `pchip-sqrt` rows moved (31 to 33 solves
  at 0.2 %, 0.58 % to 0.59 % in R at 1 %).

## 0.12.0

Copper's pad-port matrix over a band from a few solves
(`PortImpedanceSweep`), a solved board network as radiating current elements
(`network_emission`), and port solves at a placement candidate's size: each
frequency starts from the nearest one solved, `auto` keeps the near
preconditioner up to 400,000 unknowns, and the port solve runs on CUDA.
Solved matrices move within the solver tolerance (warm start, the `auto`
switch) and where the skin filaments' middle was thinner than their faces.

- The port-impedance contract starts each frequency from the port solves of
  the nearest frequency solved on its mesh (`settings["warm_start"]`, on by
  default; `started_from_hz` per frequency). On a four-layer 28k-cell board
  with 30 ports at 1/10/30/100 MHz this halved the solve (171 s against
  347 s at tolerance 1e-6, 341 s against 541 s at 1e-9,
  `docs/PORT_IMPEDANCE_SCALE_RESULTS.json`). The matrix agrees with a cold
  solve to the tolerance; its bits now depend on which frequencies were
  solved before it. `solve_sheet_cases(initial=...)` and
  `solve_port_impedance(initial=...)` expose the start.
- `preconditioner="auto"` keeps `near` up to 400,000 saddle-point unknowns
  (was 60,000). With the same-layer near field the near LU is faster than
  block at every size measured (0.30-0.53 of its time up to 599k unknowns)
  at 8-10 KB per unknown (`docs/SHEET_PRECONDITIONER_AUTO_RESULTS.json`); on
  four close layers block did not finish a 30-port solve in 15 minutes.
  Solves between the two sizes now use about twice block's memory and their
  bits move within the tolerance.
- Port impedance on CUDA: `solve_port_impedance(backend="cuda")` and the
  contract's `execution_backend="cuda"`. `solve_sheet_cases_cuda` solves
  several excitations of one mesh on the device, sharing the operator and
  the factored preconditioner, with `keep_terminal_components` and `initial`
  as on the CPU. It agrees with the CPU and is not faster: on a T4 the
  30-port fixture did not finish in 2400 s (CPU: 171 s on eight threads),
  the near LU's device triangular solves taking 90x the operator's time
  (`docs/SHEET_CUDA_RESULTS.md`).
- `electrical.port_network.PortImpedanceSweep` interpolates the copper's
  pad-port matrix over a band from a few solves (R: PCHIP in `sqrt f`, L:
  PCHIP in `log f`, chosen by measurement in
  `docs/PORT_SWEEP_RESULTS.json`), refusing outside it; `refine_frequencies`
  chooses the solves by checking every interval at its midpoint and keeps the
  checks. `BoardNetwork` takes a sweep as its copper and marks solutions on
  interpolated copper (`copper_interpolated`). The port-impedance contract
  refines with `settings["frequency_refinement"]` and returns
  `result.refinement` and `result.sweep()`. Without a sweep a frequency the
  copper was not solved at is refused, as before.
- `emc.tiled_dipole_superposition.network_dipoles` and `network_emission`:
  a `BoardNetwork` solution as current elements -- the copper's current
  superposed from the kept port solves, each part and source along the line
  between its pads -- and per frequency its far field, radiated power,
  margin to a limit line and near-field |H|. The elements close every loop
  (moments sum to 4e-16 of their magnitudes) and radiate as the same port
  currents solved directly (7e-13). `path_dipoles` is public;
  `PortImpedanceProblem.pad_positions_m()` gives the pads' centres.
- `dipoles_from_sheet_peec` takes graded grids (`SheetMesh.pitch_m is None`).
- `NetworkSolution.source_current_a` also holds each current source's own
  current (per case in `solve_cases`), not only the voltage sources'.
- `graded_filaments` no longer leaves a middle filament thinner than the face
  filaments: such a middle joins the two filaments beside it. A 35 um foil
  near 0.9 MHz had a 1 nm middle between two 17.5 um filaments, and every
  solve with `maximum_filaments` of 3 or more failed there ("partial
  inductance lost its precision"). The cut changes where the middle was
  thinner than the faces (35 um, default settings: 0.89-2.0, 8.0-10.9,
  44-50, 201-214 and 861-881 MHz); where the old cut solved, the two strips'
  impedance moved by 1e-4 of |Z| (R by 0.13 % at 1.67 MHz, 0.41 % at 50 MHz)
  and the skin slab of `tests/test_peec_fem_comparison.py` by 1.2e-4 in its
  AC/DC ratio. Above about 180 MHz on 0.2 mm cells the 1 um face filaments
  still exceed the closed form's precision, as before.

## 0.11.0

`electrical.dice_peec` is gone (breaking: see below). The C++ core joins the
Python solvers as a hybrid; copper becomes a network of pad ports closed by
the parts that join its nets (`electrical.port_network`,
`port-impedance-problem/v1`); and vias get their plated barrels and their
length.

- `electrical.port_network` is new: copper as a network of pad ports, closed
  by lumped parts (`docs/PORT_NETWORK.md`).
  - `realize_port_network` turns a pad-port impedance into resistors and
    coupled inductors, exact at the extraction frequency and resistive at DC.
  - `subckt_text` writes it as a SPICE `.subckt`.
  - `SeriesRLC` and `capacitor` model lumped parts. `nodal_voltages` solves a
    network of series R-L-C branches. `loop_impedance` and `mesh_matrices`
    close parallel meshes through shared copper.
  - `window_moments`, `close_kcl` and `port_block` reduce simulated port
    currents to the second moments a linear copper solve takes.
  - The code moved from Kicad_PowerOpt with the same arithmetic.
- `electrical.sheet_peec.solve_port_impedance` measures the pad-port
  impedance of copper in several conductors in one lockstep solve.
  `solve_sheet_case(s)(keep_terminal_components=True)` keeps the conductors
  a 0 A terminal touches, so the open-port voltages across conductors are
  read; the default is unchanged.
  - A `PortPad` may cover cells on several layers (`other_layers`), and
    `merge_pads` makes one port of several pads (the pins of one part on one
    net). A one-layer pad gives the same bits as before.
  - `keep_solutions=True` keeps each port's solve, and
    `SheetPortImpedance.superpose(port_currents)` returns the copper's
    `SheetSolution` for any port currents.
- `port-impedance-problem/v1` (`electrical.sheet_peec.port_impedance_contract`)
  states routed copper as a serialisable mapping: the conductor as in
  `current-field-problem/v2`, pads with their part and net, each port's
  reference pad, the frequencies, and `merge_pins`.
  `solve_port_impedance_problem` returns a `PortImpedanceResult` with one
  matrix per frequency for `BoardNetwork`, serialisable as
  `port-impedance-result/v1`. Frequencies that cut the copper alike share one
  mesh and operator. CPU only.
- `BoardNetwork` solves copper (`Z(f)` per frequency) with every part that
  joins its nets, by modified nodal analysis. Each part keeps its own current
  spectrum.
  - Parts: `Part` with `SeriesRLC`, `TabulatedImpedance` or `Tie`.
  - Sources: `CurrentSource` and `VoltageSource`.
  - `transfer(f)` factorises each frequency once for every source at unit
    strength (`NetworkTransfer`), and `solve_cases` solves many operating
    cases and their harmonics with it, each part's current spectrum per case.
- `realize_port_network` takes a reference pad per conductor. Breaking against
  the first version: `PortNetwork.reference_pad` is now a property of
  `references`. Resistors stay within a conductor, inductors couple across
  conductors, and the mutual resistance left out is recorded in
  `dropped_cross_resistance_ohm`.
- Another compiler builds the C++ core through CMake (`-DCMAKE_CXX_COMPILER=`,
  with its flags in `CMAKE_CXX_FLAGS`); every optional flag
  (`-fcx-limited-range`, `-march`) is checked before use and Clang builds the
  kernel library without FMA contraction. The `PCB_NATIVE_CXX` variable of
  the former per-package Python builders went with them. AMD AOCC 5.2
  (clang 17) built every kernel through those builders and passed the suite;
  measured on one host (Zen 5, 1 / 8 threads) against g++ 12: layered DC solve
  148 -> 44 ms / 20 -> 7.3 ms, thermal 2726 -> 1240 ms / 387 -> 188 ms,
  scalar Maxwell MPIR (256x256) 2.7 -> 4.4 s / 0.51 -> 0.77 s, dipole near
  field (8000 x 4096) 756 -> 791 ms / 98 -> 110 ms. AOCC through the CMake
  build has not been run. g++ stays the default.
- The sheet solve is a hybrid: the Python driver (lockstep excitations, the
  near block in strips, cell regions, `keep_terminal_components`) calls the
  C++ core for the mesh topology, the convolution and pFFT operators and the
  kernel tables. The core's own sheet solve driver (`core.sheet.solve_sheet`)
  is built but not called: it predates those features. A mesh with cell
  regions is built in Python. The suite takes 17 s with the core built
  against 80 s on the NumPy references (one host, not a benchmark).
- Vias couple to one another with their length: the vertical kernel's far
  form is two parallel filaments of the two spans
  (`parallel_filament_mutual_inductance`), not two points, and the pFFT
  couples via levels through line kernels instead of mid-plane points
  (`docs/SHEET_PEEC.md`, `docs/SHEET_VIA_RESULTS.json`).
- A via can be its plated barrel (`ViaBranch.barrel_radius_m`,
  `barrel_bore_radius_m`): its own inductance is then the barrel's, not the
  cell's column's. The current-field and port-impedance contracts read each
  vertical connection's `barrel` (as `geometry.cad_import` writes it from the
  STEP solid) and use it by default; the setting
  `via_self_inductance: "column"` keeps the previous model. **This moves the
  solved field of every mapping that carries barrels.** The CUDA sheet solve
  takes barrel vias as the CPU solve does.
- The CUDA sheet solve works with the C++ core built: the device
  convolution operator takes its spectra from the host operator's real-space
  tables when the core holds the spectra (it raised `KeyError` before).
- Excitations solved together keep the bits of their solve alone on OpenBLAS
  only: on Accelerate (macOS arm64) the C++ core's macOS CI job showed
  last-bit differences on the pFFT operator, iteration counts equal
  (`docs/SHEET_PEEC.md`). The guarantee is unchanged elsewhere.
- `CudaPeecConfig`'s precision error no longer names a consumer's
  configuration key.
- `electrical._pcbcore`: the C++ solver core (`cpp/`), one extension module
  built by the root `CMakeLists.txt`
  (`cmake -S . -B build/native && cmake --build build/native`). It bundles
  SuperLU 7.0.1 with its own CBLAS (fetched, SHA-256 pinned), so sparse
  factorisations never run on the system's BLAS, and fetches Eigen when none
  is installed. The kernels that were separate modules (`_layered_dc_native`,
  `_scalar_maxwell_native`, `_thermal_native`, `_dipole_native`,
  `_voxelize_native`, `_sheet_pfft_native`) are its submodules. It builds
  with GCC, Clang and MSVC on x86-64 and aarch64 Linux, macOS and Windows.
  `electrical.backend.native_available()` says whether it is built; without
  it the NumPy references answer. **The wheel is pure Python:** the core is
  built from a source checkout.
- The native kernels are the default CPU path whenever the core is built:
  layered DC and scalar Maxwell Q1, thermal hexahedral Q1 and the dipole
  direct summation; `native=False` keeps the NumPy path. The native MPIR runs
  the whole refinement in C++ (FP64 outer residual, FP32 inner PCG, two-level
  coarse space with Eigen); its answers agree with NumPy's to the solver
  tolerance, not bit for bit, and are bit-identical across thread budgets.
- `dc_network.solve_conductance_network` and `split_branch_sensitivity` run in
  the core when it is built (one SuperLU factorisation for the forward and
  every adjoint right-hand side); they agree with the NumPy references
  (`_dc_network_reference.py`) to roundoff. A singular network now gives
  non-finite potentials on both paths.
- `electrical._pcbcore_cuda` (built with `-DPCB_NATIVE_CUDA=ON`): the FP32
  inner PCG of the layered DC and thermal hex solves on a CUDA device, the
  FP64 refinement on the host. Opt-in through `operator.device_system()` and
  `solver.solve_mpir_device`; compiled in CI, not yet run on a GPU
  (`docs/CUDA_HANDOFF.md`).

### Breaking

- Removed `electrical.dice_peec` (the DICE delta scorer, its runtime
  controller and CUDA calibration) and the `pcb-analysis-calibrate` console
  script. Its scalar interaction kernel was a ranking proxy that no consumer
  used. Migration: `geometry.cad_import.board_vias` returns
  `geometry.cad_import.BoardVias` of `BoardVia` (same `row`, `col`,
  `layer_from`, `layer_to`, `resistance_ohm`, `inductance_h` fields as the
  old `ViaSpec`; duplicate vias still merge or are refused); `board_stackup`
  is gone (the raster's `layers` carry names and centre z); `voxel_peec`
  keeps `HardwareTelemetry` and `ExecutionReport` in
  `electrical.voxel_peec.telemetry`.
- The environment variables `PCB_NATIVE_Q1`, `PCB_NATIVE_THERMAL`,
  `PCB_NATIVE_EMC`, `PCB_NATIVE_ORTHO`, `PCB_NATIVE_DOT` and
  `PCB_GEOMETRY_CLASSIFY` are gone: the native kernels answer whenever the
  core is built, the Maxwell inner GMRES keeps modified Gram-Schmidt with
  float64 dots, and a point classification method is chosen per call with
  `method=`.
- The per-package `native/build.py` scripts (and their `PCB_NATIVE_CXX`) are
  gone; CMake is the one build.
- Barrel vias move solved fields: a current-field or port-impedance mapping
  whose vertical connections carry `barrel` (as `geometry.cad_import` writes
  them) is solved with the barrels' own inductance. Pass
  `via_self_inductance: "column"` to solve as 0.10.0 did (up to the far-field
  via coupling, which changed too).

## 0.10.0

One process-wide thread budget replaces every thread knob (breaking: see
below), excitations of one conductor are solved together, and the sheet-PEEC
near-field preconditioner keeps each branch's couplings on its own layer.
`electrical.sheet_peec.dc_network` is new.

- `solve_sheet_case(near_tiles=)`: the `block` preconditioner factors
  `R + j omega L_near` in that many strips, side by side on the budget.
  `DEFAULT_NEAR_TILES` is 4, so `block` (and `auto` above 60,000 unknowns)
  changes its iterations and its solution within the tolerance: within 4e-10
  of the whole block on the benchmark plane, 0.2 mm 39.6 s -> 15.8 s at 8
  threads (docs/SHEET_PEEC.md). `near_tiles=1` is the previous preconditioner.
- The sheet operator's flux apply runs every layer in one batched SciPy FFT
  on the budget.
- `solve_sheet_cases(mesh, operator, terminal_sets)` and
  `solve_current_field_problems(problems)`: excitations of one conductor
  share the system and the preconditioner, and their GMRES solves run in
  lockstep, each to the bits it gets alone (0.2 mm plane, 2 / 4
  excitations: 1.37x / 1.58x; docs/SHEET_PEEC.md).
  `SheetInductanceOperator.apply_many` applies the operator to several sets
  of currents in one batched FFT. Additive.
- `SheetInductanceOperator.near_inductance` assembles with array lookups
  instead of a per-branch walk, to the same bits (0.2 mm plane: 0.53 s ->
  0.06 s). The block triangular preconditioner forms were measured and not
  adopted (tag `work/sheet-block-triangular`).
- `solve_sheet_cases` cuts a group of excitations into lockstep columns
  that fit in half the memory available (`lockstep_column_bytes`, within
  10 % above the measured peak; docs/SHEET_PEEC.md). The bits do not
  change; `SheetSolution.lockstep_columns` and the contract's metrics say
  how many columns shared a solve.
- `electrical.sheet_peec.dc_network`: the zero-frequency sheet mesh as a bare
  conductance network. `ConductanceNetwork` (numbered nodes, branches in
  siemens), `solve_conductance_network(network, reference, injection,
  objective_weights=None)` (SuperLU factored once for the forward and every
  adjoint right-hand side when objectives are given, `spsolve` otherwise) and
  `split_branch_sensitivity` (per-node share of `G·ΔV·Δλ`, in-plane branches
  halved). The order of floating-point operations is fixed, so a network gives
  the same bits every time; Kicad_PowerOpt's search screen moved onto it with
  identical numbers. Additive.
- `electrical.threads`: one process-wide thread budget is the only threading
  control. `available_threads()` (the physical cores of the process's
  affinity mask; hyper-threads are not counted), `thread_budget()` (default
  `available_threads()`), `set_thread_budget(n | None)` (also limits the
  OpenMP pools of NumPy and SciPy to the budget through `threadpoolctl`, a
  new dependency; `None` sets the default budget) and
  `thread_budget_scope(n)`; all four are exported from `electrical`. BLAS
  runs on one thread, always: importing `electrical` sets
  `OPENBLAS_NUM_THREADS` and its MKL, BLIS and Accelerate counterparts to 1
  and limits the BLAS libraries already loaded. Every fused C++ kernel (layered DC and
  scalar Maxwell Q1, thermal hexahedral Q1, dipole fields, point-in-solid
  tests, sheet pFFT near-field products) runs an OpenMP team of the budget,
  passed explicitly, so `OMP_NUM_THREADS` no longer sets the pFFT kernel's
  team; PyPEEC's SciPy FFT runs on the budget. `dc_port_basis` splits the
  budget between its pool over the unit solves and each solve's OpenMP team
  internally (provisionally a serial pool and the whole budget in the team,
  to be replaced by a measured rule); `DCPortBasis.workers` still reports the
  pool width used. CUDA paths are unchanged.

- `SheetMesh(region=)` and the contract's `cell_regions`
  (`CurrentFieldProblem.cell_regions`, `{"cell_regions": {layer: [{x, y,
  region}]}}`): adjacent copper cells of two different non-zero regions are
  not joined, so two nets a clearance narrower than a cell separates do not
  short on the grid; region 0 joins any neighbour. Additive.
- The `near` and `block` preconditioners keep a branch's near-field
  couplings on its own layer only (and the vertical branches' among
  themselves), leaving the layers above and below to GMRES. Their
  iterations and their solutions change within the tolerance. On a
  four-layer 30 MHz hot loop (183k unknowns, seven excitations, tolerance
  1e-6) `near` went from 211 to 135 iterations and 264 s to 73 s, the loop
  inductance unchanged to 1e-6 of itself (docs/SHEET_PEEC.md, which also
  records the preconditioner changes measured and not adopted). `block`
  and the CUDA path were not measured again.
- The `near` preconditioner solves a lockstep group's columns in pairs on the
  thread budget (`NEAR_SOLVE_COLUMNS`), to the same bits: seven columns on
  that hot loop 0.91 s -> 0.37 s per application.

- Fixed: `solve_sheet_case_cuda` accepts `preconditioner="auto"` (as `near`;
  the CUDA path has no block form). Since 0.9.3 the current-field contract
  passed `auto` by default, so every CUDA current-field solve with default
  settings raised `ValueError`.

### Breaking

The default thread count of the native kernels changes from one to the
physical cores (hyper-threads excluded). Callers that set threads replace those settings with one
`electrical.set_thread_budget(n)` at start-up (or `thread_budget_scope(n)`
around a block). Removed:

- the `native_threads=` parameter of `solve_pcb_dc`, `MatrixFreePCBOperator`,
  `dc_port_basis`, `solve_thermal_conduction`, `MatrixFreeThermalOperator`,
  `solve_thermal_transient`, `run_electro_thermal`, `run_circuit_coupled`,
  `evaluate_fields` and `far_field_pattern`, and the `threads=` parameter of
  the internal native kernel wrappers;
- `dc_port_basis(workers=)`, `run_circuit_coupled(port_basis_workers=)` and
  the exported `electrical.matrix_free_mpir_fem.port_basis_workers()`;
- `geometry.cad_import.StepSolid.contains(threads=)` and
  `TriangleMesh.contains(threads=)` / `TriangleMesh.winding_numbers(threads=)`;
- the environment variables `PCB_NATIVE_THREADS` and `PCB_PORT_BASIS_WORKERS`
  and the `native_threads()` helpers of the native modules
  (`geometry.cad_import.mesh.native_threads` was in that module's `__all__`).
  `PCB_NATIVE_Q1`, `PCB_NATIVE_THERMAL` and `PCB_NATIVE_EMC`, which select a
  path rather than a thread count, stay;
- `electrical.voxel_peec.default_tolerance` no longer reads
  `settings["scipy_workers"]` and raises `ValueError` when it is given.

## 0.9.3

Memory of the sheet-PEEC solve and threads for the N-port basis. Additive;
the default preconditioner of `solve_sheet_case` changes from `near` to
`auto`, which is `near` below 60,000 saddle-point unknowns and `block` above
and gives the same solution to 1e-9.

- `electrical.matrix_free_mpir_fem.dc_port_basis(workers=)` runs the `n − 1`
  unit solves of an N-port basis on a thread pool (`PCB_PORT_BASIS_WORKERS`
  sets the default, which stays 1); `DCPortBasis.workers` reports the count
  used, `port_basis_workers()` is exported, and
  `multiphysics.staggered_coupling.run_circuit_coupled(port_basis_workers=)`
  passes it through. The result is the serial one bit for bit; four threads
  give 1.7 to 2.1× on an 8-core desktop, two pool threads over an OpenMP team
  of four up to 2.6× (`docs/PORT_BASIS_WORKERS_RESULTS.json`).
- `electrical.sheet_peec.solve_sheet_case(preconditioner="block")`: the near-field
  impedance applied by block elimination (`Z_near` LU and the diagonal variant's
  nodal admittance for the Schur complement) instead of one LU of the whole
  saddle-point matrix; about 1/4 of the memory from 64k branches on and faster
  there, the same solution to 1e-9. `"auto"` picks `near` below 60,000
  saddle-point unknowns and `block` above (`AUTO_BLOCK_FROM_UNKNOWNS`,
  `choose_preconditioner`) and is the new default of `solve_sheet_case` and of
  the current-field contract; pass `preconditioner="near"` for the previous
  behaviour. Measurements in `docs/SHEET_PRECONDITIONER_RESULTS.json`.

## 0.9.2

Voltage-driven DC conduction, the N-port reduction of a conductor, and the
electro-thermal loop driven by an external circuit. Additive; existing
current-driven calls are unchanged apart from a deprecation warning in the
coupled scenario.

- `electrical.matrix_free_mpir_fem`: new `VoltageTerminal(nodes, voltage_v,
  name)`. `PCBConductionProblem` takes `voltage_terminals=`, and
  `reference_node` now defaults to `None` (it is required only for a purely
  current-driven problem). `MatrixFreePCBOperator` takes `dirichlet_nodes=`,
  and `build_rhs` takes the voltage terminals and lifts their potentials.
  `PCBConductionSolution.voltage_terminal_current_a` reports the current each
  voltage terminal supplies. Verified on the NumPy and fused C++ paths; the
  CUDA runtime shares the same free-node mask but has not been run with it yet.
- `multiphysics.staggered_coupling`: `run_electro_thermal` accepts a
  voltage-driven problem as is. Heating lowers the current and the loss
  instead of raising them.
- `electrical.matrix_free_mpir_fem`: `PortSet`, `dc_port_basis` and
  `DCPortBasis` reduce one conductor to an N-port at its pads with `n − 1`
  unit voltage solves (conductance matrix, unit fields) and turn the second
  moments `⟨I Iᵀ⟩` of any port-current waveform into the time-averaged loss
  of every element and via and the RMS current density.
- `multiphysics.staggered_coupling`: `CircuitCoupledScenario` and
  `run_circuit_coupled` iterate the N-port of every conductor role, an
  external circuit (`PortCircuit` protocol; `LinearTheveninCircuit` built in)
  and the thermal solve to a self-consistent ρ(T) state. `via_temperature_k`
  takes a via sequence. `run_scenario` dispatches the new scenario.
- `thermal.matrix_free_mpir_fem`: `element_heat_from_losses` and
  `via_heat_sources` map loss arrays onto the thermal stack; the solution-
  based `element_joule_heat_w` and `via_joule_heat_sources` delegate to them.
- Deprecated: current-driven electro-thermal coupling. `ElectroThermalScenario`
  (and through it `ElectroThermalEnclosureScenario`) raises a
  `DeprecationWarning` for a problem with fewer than two voltage terminals,
  because its copper currents are imposed, the loss `I² R(T)` rises with
  temperature and the board can run away (one voltage terminal with current
  loads included). Behaviour
  is unchanged. Migration: replace the pair of `CurrentTerminal`s and the
  `reference_node` with `VoltageTerminal`s on the same pad nodes. To keep a
  known cold operating current, solve once at 1 V and scale the voltage. Read
  the supplied current from `electrical.voltage_terminal_current_a`. The
  coupled demo, the multiphysics README and the coupling tests now use
  voltage drive. The archived current-driven measurements are labelled as
  such.
- `emc.tiled_dipole_superposition`: `terminal_closure_dipoles` takes
  `voltage_terminal_current_a=`, and `dipoles_from_pcb_dc(close_terminals=True)`
  supplies it from the solution.

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
