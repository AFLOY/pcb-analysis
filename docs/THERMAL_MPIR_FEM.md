# Matrix-free MPIR-FEM for thermal PCB analysis

## Scope

`src/thermal/matrix_free_mpir_fem` solves steady heat conduction through a
layered PCB stack and reports the temperature field, the element heat flux, and
a closed heat budget. It shares the mixed-precision iterative-refinement solver
and the NumPy/CuPy runtimes of `electrical.matrix_free_mpir_fem`; only the
discretisation, the boundary model, and the preconditioner are thermal.

The front end is:

- a structured mesh of hexahedral trilinear (Q1) element slabs, one slab per
  copper or laminate layer, on the electrical solver's in-plane grid;
- element conductivity with separate in-plane and through-plane values, so
  copper, anisotropic laminate, and plated vias are all just element
  materials;
- heat input per element (Joule loss), per node array (heat handed over at
  a contact), or spread over named nodes (component power);
- Newton cooling on the top and bottom faces and fixed-temperature nodes;
- an optional active-element mask, so a heat sink or enclosure voxelised
  onto the same kind of grid (`VoxelThermalMesh`, from a `VoxelSolidModel`)
  is the same mesh with void elements carved out;
- Newton cooling on every exposed face of the active elements
  (`ExposedFaceConvection`, restrictable by direction), and a per-face
  ambient temperature on the top and bottom boundaries, which is how a
  separately meshed body presents its contact temperature to the board;
- grey surface-to-ambient radiation from the top or bottom face
  (`RadiationBoundary`) or from every exposed face (`ExposedFaceRadiation`),
  solved by Newton's method on the `T⁴` term inside
  `solve_thermal_conduction` (see "Radiation").

Transient conduction, temperature-dependent conductivity and view factors
between surfaces are not implemented. The nonlinear coupling back into the
electrical solve (copper resistivity rising with temperature) is left to
`multiphysics.staggered_coupling`.

## Discretisation

Each element is a box `hx × hy × hz` where `hz` is the slab thickness. The
element stiffness splits into an in-plane and a through-plane part so the two
conductivities enter separately:

```text
K_e = k_xy * [ (hy hz / hx) M⊗M⊗S + (hx hz / hy) M⊗S⊗M ]
    + k_z  *   (hx hy / hz) S⊗M⊗M
```

with the 1D stiffness `S = [[1, -1], [-1, 1]]` and mass `M = [[2, 1], [1, 2]] / 6`.
Local node ordering is `4 dz + 2 dy + dx`. Only the two `(slabs, 8, 8)` unit
tensors and the two `(slabs, rows, cols)` conductivity arrays are resident on
the low-precision path; no global matrix is assembled.

Void elements carry zero conductivity, so they drop out of the action without
any change to the kernels. A node touched by no active element has no
equation; it is treated as a fixed node with zero rise and reported as `nan`.
An exposed face is an active element face whose neighbour is void or lies
outside the grid; its film conductance `h A` is lumped equally onto its four
nodes, exactly as the top and bottom boundaries lump theirs. The array, CUDA
and C++ paths therefore run a masked mesh unchanged; `tests/test_thermal_voxel.py`
checks a block carved out of a void grid against the same block as a plain
layered mesh (agreement to `1e-7` relative), a fin against the 1D fin solution
(`2e-3` relative along the fin, base heat within 2 %), and the native and CUDA
paths against the array path on a masked mesh with an internal void.

`experiments/board_enclosure_acceptance.py` records the same checks with
timings; the adopted run is `BOARD_ENCLOSURE_ACCEPTANCE_RESULTS.json`
(Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz, NVIDIA GeForce GTX 1650, CuPy 14.1.1, NumPy 2.3.5).
A 13312-element block with a cavity, solved as a plain mesh
(15625 nodes, 536 ms) and inside a void grid (29791 nodes):

| Path | Solve (ms) | Inner iterations | Max difference to plain (K) |
|---|---|---|---|
| block in void, array | 1423 | 57 | 5.1e-13 |
| block in void, cpp-native-threads4 | 1256 | 56 | 1.2e-12 |
| block in void, cuda | 1224 | 56 | 8.0e-13 |

The masked grid carries 1.9× the nodes of the plain one for the same
body, which is the cost of a bounding-box grid; it buys the ability to
carve any body. The fin (1 mm thick, 20 mm long, `h = 10`, `k = 200`):

| Elements along the fin | Max relative temperature error | Base heat error |
|---|---|---|
| 20 | 1.6e-07 | 8.4e-06 |
| 40 | 4.1e-08 | 2.1e-06 |
| 80 | 1.0e-08 | 5.2e-07 |

Convection is a lumped Robin term. Every face element contributes
`h · hx · hy / 4` to each of its four corner nodes, so a zero-order face
integration adds a nodal conductance to the diagonal and `h A T_amb` to the
load. Element heat is lumped equally onto eight corners. Fixed-temperature rows
are replaced by the identity and their values move onto the free rows through
the unconstrained stiffness action, exactly as the frequency-domain Dirichlet
handling does.

The heat flux is `-k ∇T` at the element centre from the trilinear gradient.

## Heat budget

`ThermalConductionSolution` reports:

| Field | Meaning |
|---|---|
| `total_heat_input_w` | sum of element heat and nodal sources |
| `convective_heat_w` | heat removed by each convection boundary, in problem order |
| `radiative_heat_w` | heat removed by each radiation boundary, in problem order |
| `radiation_iterations`, `radiation_converged`, `radiation_change_k` | Newton steps on the radiation term, whether the last step moved the nodes less than the tolerance, and by how much |
| `fixed_temperature_heat_w` | heat absorbed by the fixed-temperature nodes |
| `heat_balance_error_w` | input minus removal; equals minus the solver residual summed over free nodes |

The fixed-node heat is `-(K T + R (T - T_amb) - q)` at those nodes, i.e. the
unconstrained row balance evaluated at the converged solution. On the tests
and the demo the balance error is at the 1e-11 to 1e-14 W level, which is
the FP64 residual, not a modelling approximation.

## Temperature rise, not absolute temperature

The solver works on `theta = T - T_ref`, with `T_ref` the ambient of the first
convective face or the mean fixed temperature. Solving for absolute kelvin
instead caps the attainable FP64 residual at roughly
`eps64 · ||A|| · 300 K / ||q||`, because the element action on a 300 K constant
is a cancellation. On a 4-slab, 13,005-node stack that floor was a relative
residual of `1.2e-10` with either an FP32 or an FP64 inner solve; solving for
the rise reaches the requested `1e-10` in seven outer steps.
`reference_temperature_k` overrides the default.

The same floor returns when the rise itself is large and nearly uniform: a
copper plate 120 K above ambient stalls near a relative residual of `3e-10`.
`solve_thermal_conduction` therefore defaults to 16 outer iterations and, if
the tolerance is still not met, re-references the unknown to the mean
free-node temperature and continues from the current iterate. The remaining
unknown is the small in-plane variation, whose residual is accurate, and the
solve finishes in a few more outer steps. The returned `MPIRResult` sums the
iteration counts of both stages and concatenates their histories.

## Radiation

A surface at `T` facing an environment at `T_amb` with emissivity `ε` loses
`q = ε σ (T⁴ − T_amb⁴)`. `RadiationBoundary(side, ε, T_amb)` puts it on the
top or bottom face with per-face arrays allowed, `ExposedFaceRadiation(ε,
T_amb, directions)` on every exposed face of the active elements with
per-element arrays allowed; both live in `problem.radiation`. The problem is
then nonlinear and `solve_thermal_conduction` iterates: each radiating face is
replaced by the Newton linearisation at the current iterate `T_k`,

```text
q ≈ h_k (T − T_eff,k),   h_k = 4 ε σ T_k³,   T_eff,k = T_k − (T_k⁴ − T_amb⁴) / (4 T_k³),
```

which is an ordinary `ConvectionBoundary` / `ExposedFaceConvection` with a
per-face coefficient and ambient (both accept arrays for this), the linear
problem is solved warm-started, and the two repeat until the nodes move less
than `radiation_tolerance_k` (default `1e-4 K`, at most
`radiation_max_iterations = 25`). Conduction is linear, so this is Newton's
method on the whole problem. The secant form `h = ε σ (T² + T_amb²)(T + T_amb)`
with the true ambient was not used: its fixed-point gain is about
`−3 (T − T_amb) / T`, so it oscillates once the rise exceeds a third of the
absolute temperature, while Newton converges monotonically from the ambient
start and needs one confirming step when warm-started from the answer.

The linearised coefficient is built each step from the mean corner temperature
of the face (top/bottom) or of the element (exposed faces); the operator and
its two-level coarse matrix are rebuilt per step. The model is grey, diffuse
and sees only its ambient: no view factors between surfaces, so a board inside
a case radiates to the case's *given* inner temperature, not to its computed
field.

Measured on a uniformly heated 8 × 10 × 1 mm plate radiating from its top
face, `ε = 0.9`, `T_amb = 298.15 K`, against the analytic surface temperature
`(T_amb⁴ + P / (ε σ A))^¼` (`ELECTROTHERMAL_ENCLOSURE_RESULTS.json`,
`experiments/electrothermal_enclosure_acceptance.py`):

| Boundary | P (W) | Surface rise (K) | Relative error | Newton steps | Heat balance (W) |
|---|---|---|---|---|---|
| RadiationBoundary top | 0.05 | 78.6 | -2.5e-14 | 5 | 1.1e-14 |
| ExposedFaceRadiation +z | 0.05 | 78.6 | 6.9e-12 | 5 | 1.4e-14 |
| RadiationBoundary top | 0.60 | 329.2 | -5.4e-14 | 9 | 1.4e-13 |
| ExposedFaceRadiation +z | 0.60 | 329.2 | 3.4e-10 | 9 | 1.4e-13 |
| RadiationBoundary top | 3.00 | 630.2 | 1.5e-11 | 13 | -1.8e-10 |
| ExposedFaceRadiation +z | 3.00 | 630.2 | 3.8e-09 | 13 | -2.8e-14 |

Decision: adopted. The bottom of the plate sits `q t / 2k` above the surface,
as it must for volumetric heating, and the radiated heat equals the input to
FP64 rounding.

## Two-level preconditioner

A cooled copper plate is thermally stiff in-plane and weakly coupled to the
air. The Jacobi-scaled operator of a 35 µm Cu / 1.5 mm FR-4 / 35 µm Cu stack on
a 12 × 12 grid has condition number `4.4e6`; a z-line block preconditioner only
brings that to `2.1e5`. The slow mode is the almost uniform temperature of each
plate, which no local smoother sees.

`AggregationCoarseCorrection` adds a coarse correction in the space of
functions constant on `b × b` node patches of each node layer:

```text
M⁻¹ r = D⁻¹ r + Z (Zᵀ A Z)⁻¹ Zᵀ r
```

- `Z` is never stored: `Zᵀ` is a pad-and-reshape sum, `Z` is a `repeat`.
- `Zᵀ A Z` is assembled exactly with 27 FP64 operator applications by
  colouring patches so that same-coloured patches never share an element.
- The coarse matrix is Cholesky-inverted once in FP64; the symmetrised
  inverse is applied on the low-precision runtime as one small matmul.
- Patches consisting only of fixed nodes get a unit diagonal so the coarse
  matrix stays SPD.
- `choose_block_size` picks the smallest `b ≥ 4` whose coarse space has at
  most 2,048 unknowns, so the dense inverse stays below 32 MiB.

Measured on the 3-slab copper stack, 40 × 40 elements, 6,724 nodes (CPU):

| Preconditioner | Inner PCG iterations per outer step | Total solve |
|---|---:|---:|
| Jacobi | 680–757 | 3.58 s |
| Two-level, `b = 2` | 24–28 | 0.33 s |
| Two-level, `b = 4` (default) | 43–53 | 0.29 s |
| Two-level, `b = 8` | 69–92 | 0.46 s |

With the default `MPIRConfig` (200 inner iterations per outer step) the Jacobi
variant does not converge on this stack; the two-level variant does. Pass
`preconditioner="jacobi"` to reproduce the comparison.

The construction depends only on the node grid, the free mask, and the FP64
action, so it also fits the electrical DC operator; it lives in the thermal
package because that is where it is needed today.

## CUDA execution

`backend="cuda"` runs the FP32 inner PCG on CuPy. The low action is one fused
`RawKernel` launch (`cuda-fused-node-gather-hex-q1`): a thread owns one output
node, visits its at most eight adjacent elements, applies the two 8 × 8 unit
tensors weighted by the element conductivities, adds the Robin diagonal, and
writes once. No atomics; repeated applications are bitwise identical. The
generic CuPy corner-product path remains as the reference for the kernel test.

The coarse correction runs on the device too: pad-reshape-sum, one
`(n_c × n_c)` float32 matmul, and `repeat`. The FP64 residual and the coarse
matrix assembly stay on the host.

Measured on an NVIDIA GeForce GTX 1650 (CuPy 14.1.1) against a 12-thread
Intel Core i7-8700 for a 4-slab stack (35 µm Cu / 0.7 mm FR-4 / 0.7 mm FR-4 /
35 µm Cu, 50 mm × 50 mm, convection on both faces, a heated trace on top),
default `MPIRConfig`, default two-level preconditioner, requested relative
residual `1e-10`:

| Elements | Nodes | Coarse size | CPU action | CUDA action | CPU solve | CUDA solve | Outer / inner |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 × 50 | 13,005 | 845 | 1.92 ms | 0.028 ms | 1.36 s | 0.56 s | 7 / 586 |
| 100 × 100 | 51,005 | 1,445 | 4.09 ms | 0.042 ms | 4.09 s | 0.72 s | 7 / 806 |
| 200 × 200 | 202,005 | 1,805 | 11.0 ms | 0.125 ms | 15.7 s | 1.12 s | 7 / 1,025 |

Action timings are the mean of 20 synchronized float32 applications. Solve
timings exclude operator construction, which includes the 27 FP64 applications
and the dense coarse inverse (0.2–1.4 s across these sizes on either backend,
because it runs on the host). The CUDA and CPU peak temperatures agree to about `1e-9` K.
The remaining CUDA solve time is dominated by the host-side FP64 residual
(seven applications of the NumPy corner-product action) and per-iteration
reductions, so the CUDA advantage grows with the mesh.

### Measured alternatives, not adopted

`experiments/thermal_sparse_fp32_benchmark.py` (branch
`exp/thermal-sparse-fp32-coarse`, results in the Git-ignored
`benchmark-results/thermal-sparse-fp32.json`) compared three bandwidth
reductions of the CUDA FP32 inner solve on the 4-slab fixture above, GTX 1650,
CuPy 14.1.1, 2026-09-02:

| Nodes | CSR fine action / matrix-free | Sparse-LU coarse / dense inverse | Tapered sparse coarse solve / dense |
|---:|---:|---:|---:|
| 13,005 | 0.46× (2.4 MB vs 0.15 MB) | 5.22 ms vs 0.030 ms | 0.72× (817 vs 577 inner) |
| 51,005 | 0.41× (9.6 MB vs 0.58 MB) | 11.1 ms vs 0.079 ms | 0.67× (1,233 vs 795 inner) |
| 202,005 | 0.39× (38 MB vs 2.3 MB) | 14.8 ms vs 0.124 ms | 0.58× (2,019 vs 1,047 inner) |

An assembled FP32 CSR operator is 2.2–2.5× slower than the fused kernel and
stores 16× more; the exact sparse-LU coarse solve is two orders slower than the
dense matmul; the Bartlett-tapered sparse coarse inverse (radius 4, SPD) keeps
the solution to `1e-12` but needs 1.4–1.9× the inner iterations and never wins.
The matrix-free action and the dense coarse inverse stay. The coarse-space cap
remains the open item for boards beyond a few hundred thousand nodes.

## Fused C++ host path (measured on `exp/cpp-thermal-emc`)

The array-corner-product action issues 64 whole-grid products per
application and the two-level PCG spends the rest of an inner iteration in
NumPy vector calls, the patch restriction and a `(n_c × n_c)` matmul.  The
opt-in native path (`MatrixFreeThermalOperator(native=True)` or
`solve_thermal_conduction(native=True, native_threads=...)`,
`PCB_NATIVE_THERMAL=1` for the process, after
`python -m thermal.matrix_free_mpir_fem.native.build`) moves the float32 inner
work into one pybind11 extension:

- the node-owned gather of the hexahedral Q1 action in the CUDA kernel's
  ordering, written per node line with the two x-neighbour elements and the
  eight local columns unrolled so the x-loop vectorises;
- the whole inner PCG with the same two-level preconditioner (Jacobi
  scaling, patch restriction, the dense float32 coarse inverse applied as a
  row-partitioned matvec, prolongation) in one SPMD OpenMP region; each
  thread owns a static range of node lines, reductions are per-thread
  partials summed in thread order, FTZ/DAZ is set per thread.

The preconditioner construction (27 FP64 applications and the dense coarse
inverse) and the FP64 outer residual stay in NumPy.  The solver dispatches
through the optional `native_inner_pcg` hook; the CUDA runtime keeps its path.

Measured on an Intel Xeon Platinum 8581C (16 cores / 32 threads, AVX-512),
GCC 14.2.1, NumPy 2.3.5, `OPENBLAS_NUM_THREADS=1`, the 4-slab fixture of the
CUDA table above, default `MPIRConfig(max_outer_iterations=16)`, default
two-level preconditioner; solve medians of three runs, operator construction
excluded (`experiments/thermal_native_benchmark.py`,
`THERMAL_NATIVE_XEON_8581C_RESULTS.json`):

| Nodes | Coarse size | NumPy solve | Construction | C++ 1 thread | 2 | 4 | 8 | 16 | Inner NumPy / C++ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13,005 | 845 | 812 ms | 187 ms | 222 ms (3.66×) | 124 ms | 81 ms | 54 ms | 45 ms (17.9×) | 584 / 585 |
| 51,005 | 1,445 | 2,807 ms | 585 ms | 933 ms (3.01×) | 505 ms | 304 ms | 191 ms | 142 ms (19.8×) | 807 / 794 |
| 202,005 | 1,805 | 14,849 ms | 1,606 ms | 3,158 ms (4.70×) | 1,736 ms | 1,095 ms | 717 ms | 524 ms (28.3×) | 1,025 / 1,020 |

The operator alone is 9.8× to 15.3× faster than the corner products on one
thread and 54× to 85× on sixteen.  Converged solutions differ from the NumPy
path by at most 1.6e-12 (relative), and the heat balance closes to the same
FP64 level.  The inner iteration counts differ by a few because the float32
rounding of the coarse matvec differs.  On sixteen threads the largest case
runs 6.0× faster than on one; the 13,005-node case flattens at eight threads
because its per-iteration barriers (six per PCG step) cost as much as the
vector work.  The preconditioner construction, still NumPy, is now the larger
part of a cold solve at every size and is the next candidate for the native
path.

Decision recorded in the JSON: the benchmark criteria (every case at least 2×
on one thread, identical convergence outcome, converged solutions within 1e-6)
are met.  The default stays the array path and one thread.

## Electrothermal coupling

`solve_pcb_dc` now reports `element_joule_loss_w` (the exact element
quadratic form `σ t vᵀ K_e v`) and `via_joule_loss_w`. The coupling module maps
them onto the thermal mesh without interpolation, because both solvers share
the in-plane grid:

- `element_joule_heat_w(solution, thermal_mesh, layer_slabs)` places layer
  `L`'s element loss into slab `layer_slabs[L]`;
- `via_joule_heat_sources(problem, solution, thermal_mesh, layer_slabs)`
  gives each via endpoint half the via loss, spread over the two node faces
  bounding its copper slab.

The sum of the thermal heat input equals the electrical Joule loss to FP64
rounding; the coupling test checks that the convective removal matches it.
`examples/electrothermal_demo.py` runs the complete chain on a two-layer
board with a via field.

## Software boundary

| Module | Responsibility |
|---|---|
| `mesh.py` | `LayeredThermalMesh`, active mask, exposed faces, corner views, unit element matrices |
| `boundaries.py` | `ConvectionBoundary`, `ExposedFaceConvection`, `HeatSource`, each lumping its own nodal conductance and load |
| `radiation.py` | `RadiationBoundary`, `ExposedFaceRadiation`, their Newton linearisation into the convection types |
| `problem.py` | `ThermalConductionProblem` validation |
| `operator.py` | matrix-free hex Q1 operator on NumPy, CuPy or C++, RHS and heat-budget post-processing |
| `solve.py` | `solve_thermal_conduction` (linear solve, Newton loop over the radiation boundaries), `ThermalConductionSolution` |
| `two_level.py` | Jacobi + aggregation coarse correction on a layered node grid |
| `cuda.py` | fused node-owned gather kernel for the float32 action |
| `native_hex.py`, `native/` | opt-in C++ action and two-level inner PCG (built in place) |
| `coupling.py` | Joule loss of a `PCBConductionSolution` as thermal load |
| `voxel.py` | `VoxelSolidModel` (material ids, fill, pitch, origin) and `VoxelThermalMesh` |
| `contact.py` | `ContactMap` between a board face and a body, `planar_contact_map` from the two grids |

The MPIR solver gained optional hooks: a system may define
`precondition_low(vector)`, which replaces Jacobi scaling in the inner PCG and
GMRES, and `native_inner_pcg(rhs, config)`, which runs the whole inner PCG
natively and returns `None` to decline. Systems without them behave exactly as
before.

## Limitations and next increments

- Structured rectangular mesh; pads and barrels are element columns, curved
  bodies are voxel staircases whose partially filled voxels carry a
  fill-scaled conductivity.
- Convection is a film coefficient per face; no buoyancy correlation, so a
  natural-convection `h(ΔT, orientation)` has to be iterated by the caller.
- Radiation is surface-to-ambient only (no view factors, no enclosure
  radiosity); a board inside a case radiates to a given case temperature.
- Steady state only, with `k` independent of temperature; `k(T)` would join
  the radiation Newton loop. The thermal-electrical feedback through `ρ(T)`
  lives in `multiphysics.staggered_coupling`.
- The coarse space is capped at 2,048 unknowns by a dense inverse. Boards
  beyond a few hundred thousand nodes will want a sparse coarse solve or a
  third level.
- A heat sink or enclosure is solved as its own `VoxelThermalMesh`; its
  interface coupling to the board is specified in
  `GEOMETRY_CAD_IMPORT.md`.
