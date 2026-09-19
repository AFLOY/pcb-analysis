# Sheet PEEC: solving a board as sheets rather than voxels

## Why

PyPEEC solves an arbitrary three-dimensional voxel model, so a multilayer board
is not beyond it. What it costs is set by the mesher taking one cell size for
the whole model: the copper and the laminate share a `z` step. A step fine
enough to hold 35um of copper makes a 1.6mm board 45 steps tall, and the
coupling operators live on a circulant embedding of that box.

Measured on `power_module` (34.7mm square, 0.2mm grid, F.Cu + B.Cu):

| | voxel box | prepared operators |
|---|---|---|
| one copper layer | 98 x 94 x 1 | 4.5 MiB |
| two copper layers | 162 x 152 x 45 | 541.1 MiB |

The conductor roughly doubled. The operators grew by a factor of 120, and none
of that growth is copper: 1,108,080 voxels of box hold 29,006 of conductor.

A PCB is not an arbitrary volume. It is a few thin sheets at heights the
stackup states. Meshing it that way pays for the copper alone.

## The mesh

One grid of cells per copper layer. A branch joins two adjacent occupied cells
and carries current along x or along y; the bar it stands for is one pitch
long, one pitch wide, and as thick as the copper. A via adds a branch between
the two layers it joins.

Three properties of this mesh make the operator cheap.

- Two branches carrying current along different axes have zero mutual partial
  inductance: the defining integral carries `dl_i . dl_j`. So there is an x
  operator and a y operator and nothing between them, and a via, carrying
  current along z, couples to neither. Vertical branches do couple to one
  another, being parallel, and that coupling is a second convolution: a level is
  one pair of layers joined by vertical branches, every branch of a level spans
  the same height, so the coupling of a level pair depends on the in-plane
  offset alone. See "Vertical branches" below.
- Between two branches of the same axis the coupling depends only on their
  in-plane offset and their layers' separation. In plane that is a convolution:
  one table per layer pair per axis serves every branch of that pair.
- On a square mesh of square cells the y table is the x table transposed.

The result on the same board:

| | prepared operators |
|---|---|
| 3D voxel, PyPEEC | 541.1 MiB |
| 2.5D sheet | 5.6 MiB |

The saving shrinks as layers are added -- pairs grow as the square of the layer
count while the voxel height does not -- so this is a statement about boards of
few layers, which is what these are.

## The kernel

`sheet_inductance` evaluates Ruehli's partial mutual inductance between two
parallel rectangular bars,

    L_ij = (mu0 / 4 pi a_i a_j) * int_Vi int_Vj 1/|r - r'| dV' dV

in closed form (Hoer and Love), as a signed sum of a sixfold antiderivative
over the corner differences.

Quadrature was tried and abandoned. It fails exactly where this mesh needs the
answer: the self term places two bars on top of each other, where a product
rule puts nodes at zero separation and the result diverges; and the nearest
neighbours of a 0.2mm cell are close enough that no affordable order resolves
the peak of `1/r` -- a six-point rule was 150% wrong on a near term.

The closed form is exact in exact arithmetic. In floating point its 64 terms
individually grow like the fifth power of the corner coordinates while their
signed sum stays of order the answer, so distant bars cancel their own
precision away. Measured for a 0.2mm cell, along the current:

| offset, cells | surviving fraction | centre-to-centre limit, relative error |
|---|---|---|
| 1 | 6.3e-2 | 1.0e-1 |
| 4 | 1.5e-4 | 5.1e-3 |
| 16 | 8.1e-8 | 3.2e-4 |
| 24 | 7.8e-9 | 1.4e-4 |
| 64 | 2.5e-11 | -- |
| 128 | 4.0e-13 | -- |

The two regimes overlap widely, so `NEAR_RADIUS_CELLS = 24` sits far from
either one's difficulty: the closed form still carries seven digits there and
the centre-to-centre limit is already within 1.4e-4. Outside the near radius
the limit is used. Asked for the closed form where it cannot deliver, the
module raises rather than returning the noise.

## Checks

The kernel is held against formulas derived independently of it:

| check | agreement |
|---|---|
| self term vs Grover's rectangular-bar formula, aspect ratios 10 to 1000 | 0.007% to 0.17% |
| mutual term vs the exact parallel-filament formula | 0.0002% to 0.006% |
| reciprocity | 1e-11 |

The operator is held against a direct sum over the same closed form, with no
transform anywhere in the reference:

| check | result |
|---|---|
| transform path vs direct assembly, both axes | 3.2e-11 |
| x current producing y flux | exactly zero |
| positive definite | yes |
| padding suppresses wrap-around between opposite edges | yes |

The solver is held against two references:

| check | result |
|---|---|
| zero frequency vs an independent dense resistor network | 1.6e-19 on a scale of 7.2e-4 |
| the only path between two layers carries the whole current | 1.000000 A |
| transform path vs a dense direct solve, 0 / 3e5 / 1e8 Hz | 1.2e-14 / 4.2e-11 / 3.7e-11 |
| departure from the resistive answer against frequency | linear, to 3 decimal places |

The zero-frequency check is the load-bearing one. There the inductance drops
out and what remains is exactly a resistor network, so any error in the
incidence, the resistances, the source handling or the grounding shows up
separated from any question about the inductance.

## Vertical branches

A branch running through the board couples to no in-plane branch and to every
other vertical branch. The second half of that was missing: only a scalar self
term existed, defaulting to zero.

It could not be fixed by supplying the self term alone. A neighbouring column
carries 46% to 69% of the branch's own coupling, measured across the filament
interfaces of a 3.5mm inlay, and redistribution currents in neighbouring columns
often oppose one another -- so the mutual terms cancel much of the loop
inductance the self terms would claim. Self without mutual overstates it.

The fix is a third convolution, built the same way as the two in-plane ones. A
*level* is one pair of layers joined by vertical branches. Every branch of a
level spans the same height, so the coupling of a level pair depends on the
in-plane offset alone. For a 3.5mm inlay cut into eleven filaments there are ten
levels and 55 tables, 1.1 MiB, 2.1s to build.

The levels are stated, not inferred: which layers a board joins is a property of
its vias and of how thick copper was cut, neither of which the stackup says. A
mesh carrying a level the operator was not built for is refused rather than
solved with that coupling silently dropped.

### What it changed

Measured on a 1.6 x 3.5mm inlay bar at 300kHz, the resistance ratio over a fixed
1.6mm window at the centre, against the bar's length:

| length | L/t | Rac/Rdc |
|---|---|---|
| 2.4mm | 0.69 | 2.774 |
| 4.8mm | 1.37 | 4.155 |
| 9.6mm | 2.74 | 4.998 |
| 19.2mm | 5.49 | 5.313 |

against the field solve's 5.92 for the same bar. It rises with length and is
still rising at 19.2mm, which is what an entry length of roughly the largest
transverse dimension over pi -- about 1.1mm here -- predicts: the excitation
fixes the end-face current density to the DC profile and the bar has to relax
out of it.

A note on what this cannot be compared against. An earlier version of this
document said a value above the projection ceiling was one "a converged solve
cannot produce". That was wrong. The ceiling is the loss the mesh holds when
given the *true* current distribution; a solve is not required to reproduce that
distribution, and a coarse in-plane grid that cannot resolve the side-wall skin
layer will concentrate current differently from the way physics does. The
measurement above exceeds the ceiling of 4.93 at 9.6mm and beyond, which
demonstrates the point: the ceiling bounds what the mesh can represent, not what
the solve returns.

So the vertical operator's justification rests on the physics rather than on
that comparison. Vertical branches are parallel and therefore couple; the
coupling to a neighbouring column is 46% to 69% of a branch's own; and
redistribution currents in neighbouring columns oppose one another, so the
mutual terms cancel much of the loop inductance the self terms would claim.
Supplying the self term alone would have been worse than supplying neither.

## Thick copper: the inlay

The sheet form assumes the current is uniform through a layer's thickness. For
foil that holds: 35um against a 121um skin depth at 300kHz is uniform to a few
per cent. A copper inlay is a different conductor.

| frequency | skin depth | 35um foil, t/d | 3.5mm inlay, t/d | inlay Rac/Rdc |
|---|---|---|---|---|
| 1 kHz | 2.090 mm | 0.017 | 1.67 | 1.04 |
| 10 kHz | 0.661 mm | 0.053 | 5.30 | 2.64 |
| 100 kHz | 0.209 mm | 0.167 | 16.75 | 8.37 |
| 300 kHz | 0.121 mm | 0.290 | 29.01 | 14.50 |
| 1 MHz | 0.066 mm | 0.530 | 52.96 | 26.48 |

A 3.5mm inlay is uniform through its thickness only below about 89 Hz. At the
switching frequency it is 29 skin depths thick and its resistance is 14.5 times
its direct-current value. One filament would miss that by the same factor and
would put the current where it is not.

`skin_filaments` cuts a conductor into filaments thinner than the skin depth,
graded so they are fine at the faces where the current is and coarse in the
middle where it is not. Each filament is a layer of the existing stackup, so
nothing else in the formulation changes, and the proximity effect between
filaments falls out of the mutual inductance already there rather than being
modelled separately. The 3.5mm inlay at 300kHz takes nine filaments:

    60um  121um  241um  483um  1690um  483um  241um  121um  60um

The foil is returned as one filament unchanged, so the same call applies to
every layer of a board.

This is where placing conductors at stated heights pays hardest against a
uniform voxel grid. For the same board at 300kHz:

| | prepared operators |
|---|---|
| 3D voxel, dz = 60um -- cannot represent the 35um foil at all | 0.95 GiB |
| 3D voxel, dz = 35um, nz = 114 | 1.65 GiB |
| 2.5D sheet, 11 filaments | 122.7 MiB |

The voxel grid has one `dz` for the whole board, so resolving a 60um filament
in the inlay forces every millimetre of the board's height to be cut at 60um --
and since that is coarser than the foil, it has to go finer still. Filaments
are placed only in the copper.

The sheet advantage narrows from 97x to 13x, because eleven filaments make 66
layer pairs. It does not disappear, and grading the cut is available only on
this side.

### What the cut is worth, against a field solve

An independent finite-difference solve of the eddy-current problem on the bar's
cross-section is the reference: infinitely long, so no end effects, and sharing
no code with this formulation. For a 1.6 x 3.5 mm copper bar at 300 kHz it gives
`Rac/Rdc = 5.10 / 5.70 / 5.87` at `h = 100 / 50 / 25 um`, extrapolating to
**5.92 +/- 0.03**.

Two things were then measured. First, whether the operator represents a
developed state at all: the field solve's own converged complex current density
was integrated onto the sheet mesh's branches, repeated along every column, and
`u = R I + j w L I` applied directly -- no solve, no terminals, and no vertical
current, so the missing vertical-branch mutual operator cannot enter. For a
developed mode `u_b / pitch` has to be one common axial field for every branch
of the cross section, and it is: the dominant imaginary part spans +/-3.7%, the
worst departure over all branches 6.1%.

Second, the loss that mesh can hold given that exact profile -- its ceiling,
independent of how well any solve converges:

| in-plane pitch | filaments per skin depth | filaments | ceiling Rac/Rdc | of the field solve |
|---|---|---|---|---|
| 200 um | 2 | 9 | 4.53 | 77% |
| 100 um | 2 | 9 | 5.16 | 88% |
| 50 um | 2 | 9 | 5.37 | 92% |
| 200 um | 4 | 11 | 4.93 | 84% |
| 200 um | 8 | 13 | 5.28 | 90% |
| 100 um | 4 | 11 | 5.56 | 95% |

So the two directions cost about the same and neither alone is enough. At the
0.2 mm grid the optimizer currently uses, with the default two filaments per
skin depth, the sheet mesh **understates this bar's AC resistance by 23%** and no
amount of solver convergence will recover it. Reaching 95% takes a 100 um
in-plane grid together with four filaments per skin depth.

That is a bounded, measured error rather than an unknown one, and it is a
statement about resolution, not about the formulation. Whether 23% is
acceptable is a question for whoever reads the gate; it is not a question this
module can settle.

## Graded grids: precorrected FFT

The convolution operator above needs one pitch: the coupling of two branches
then depends on their offset alone.  A graded tensor grid
(`electrical.matrix_free_mpir_fem.grid`, fine under the parts and along
narrow traces, coarse elsewhere) has branches of different lengths and widths
at irregular positions, and no convolution exists.  `sheet_pfft` keeps the
transform with the precorrected FFT:

1. each branch's current-length product `I_b l_b` is projected onto a
   uniform grid of the coarse pitch through a tensor-product Lagrange
   stencil of `order + 1` points per axis (moments matched to that order);
2. the grid sources are convolved by FFT with `mu0 / (4 pi r)` between grid
   points, one table per layer separation;
3. the vector potential is interpolated back to the branch centres with the
   same stencil and scaled by `l_b`;
4. for pairs within `near_radius_cells` grid cells the grid contribution is
   subtracted and the exact Hoer–Love partial inductance of the two bars
   (per-pair dimensions, `closed_form_mutual_inductance_arrays`) is added, as
   one sparse precorrection built once.  The self terms are therefore exact
   and the preconditioner reads them per branch.  Bars are sampled at Gauss
   points over their extent (three along, two across) so the far field of a
   bar longer than a grid cell stays right, and the near radius is measured
   from the bars' edges.

Vertical branches (barrels and filament links) get the same treatment on
their level pairs.  `SheetMesh` takes the grid (`grid=`, `pitch_m=None`),
gives every branch its bar (`branch_geometry`), scales resistances by
`length / width` and current densities by the cell's transverse extent; the
plane-opt contract gained schema `v2` whose `grid` carries `x_edges_mm` and
`y_edges_mm` instead of `pitch_mm` (`build_plane_opt_sheet_inputs` picks the
pFFT operator for a graded grid, `operator="pfft"` forces it), and the CUDA
solve has the same operator on the device (`CudaPfftSheetInductanceOperator`).

Measured (`SHEET_PFFT_RESULTS.json`, `experiments/sheet_pfft_acceptance.py`,
Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz). Operator accuracy as the relative Frobenius error of the dense
in-plane operator (x / y) against the convolution operator on a uniform
8 × 10 mesh at 0.2 mm and against the closed form for every pair on a graded
0.1–0.5 mm mesh; the diagonals agree to `1e-12` in every case:

| Order | Near radius (cells) | Uniform x / y | Graded x / y | Build, graded (ms) |
|---|---|---|---|---|
| 1 | 2 | 2.5e-03 / 2.4e-03 | 3.0e-03 / 3.8e-03 | 222 |
| 2 | 3 | 1.4e-04 / 2.0e-04 | 3.7e-04 / 5.2e-04 | 385 |
| 3 | 3 | 1.2e-04 / 1.4e-04 | 3.3e-04 / 4.5e-04 | 452 |
| 3 | 4 | 3.0e-10 / 2.8e-10 | 6.7e-05 / 1.3e-04 | 623 |
| 3 | 5 | 3.0e-10 / 2.8e-10 | 2.7e-05 / 5.0e-05 | 806 |
| 4 | 5 | 3.0e-10 / 2.8e-10 | 5.7e-06 / 1.2e-05 | 1010 |

The default is order 3, radius 3, on a projection grid of twice the finest
cell. The near terms (one closed form per distinct bar pair and offset, one
grid stencil product per distinct configuration) are the build's cost; the
grid products run in C++ with OpenMP when `electrical.sheet_peec.native` is
built (`native_available()`), else in NumPy. A 12 × 1 mm strip line at 1e+06 Hz
(F.Cu go, B.Cu return, 35 µm copper cut into filaments), loss relative to the
uniform 0.1 mm convolution solve:

| Grid | Operator | Branches | Kernel (MB) | Build (ms) | Apply (ms) | Diagonal: solve (ms) / GMRES | Near: solve (ms) / GMRES | Loss rel. diff |
|---|---|---|---|---|---|---|---|---|
| uniform 0.1 mm | SheetInductanceOperator | 4550 | 0.27 | 84 | 0.5 | 1247 / 221 | 198 / 14 | +0.00e+00 |
| uniform 0.5 mm | SheetInductanceOperator | 142 | 0.01 | 53 | 0.2 | 65 / 69 | 6 / 2 | -1.15e-01 |
| uniform 0.1 mm, pFFT | PfftSheetInductanceOperator | 4550 | 15.00 | 203 | 2.6 | 2227 / 227 | 375 / 13 | -2.10e-04 |
| graded 0.1 mm at the ends, 0.5 mm between | PfftSheetInductanceOperator | 1928 | 4.17 | 270 | 1.3 | 1044 / 216 | 136 / 15 | -2.54e-04 |

### Near-field preconditioner

The saddle-point GMRES was preconditioned with `Z` replaced by its diagonal
(self terms only), which left every mutual term to the Krylov iterations:
about 220 on the strip line, 1,150 to 1,800 on `power_module`. Both operators
now report the exact partial inductance between each branch and its
neighbours within one projection cell (pFFT: kept from the precorrection's
near pairs; convolution: read off the kernel tables), and the default
preconditioner (`preconditioner="near"`) factors the saddle-point matrix with
`R + j omega L_near` in place of `Z` by sparse LU (`"diagonal"` keeps the old
Schur-complement form). The strip-line rows below carry both; the solution
is the same to `1e-10`. On CUDA the factors are SuperLU's applied by device
triangular solves and GMRES runs in cycles growing from ten iterations, which
on `power_module` (uniform, 1 MHz) gives 8 s against 3 s on the CPU and 48 s
with the diagonal form; the near preconditioner makes the CPU path the faster
one for these sizes.

`power_module` from KiCad (fused copper, y-down raster, terminals at the two
ends of the largest B.Cu copper piece, barrels off the copper dropped),
through `solve_plane_opt_problem`, one filament per layer:

| Grid | f (Hz) | Schema | Operator | Cells | Branches | Kernel (MB) | Wall, diagonal (s) | Wall, near (s) | J max (A/mm²) | J p99 (A/mm²) |
|---|---|---|---|---|---|---|---|---|---|---|
| uniform 0.25 mm (v1) | 0e+00 | v1 | SheetInductanceOperator | 19321 | 23230 | 4.4 | — | 0.3 | 77.270 | 12.482 |
| uniform 0.25 mm (v1) | 1e+06 | v1 | SheetInductanceOperator | 19321 | 23230 | 4.4 | 82.9 | 3.5 | 75.704 | 15.175 |
| graded 0.1 mm under components (v2) | 0e+00 | v2 | PfftSheetInductanceOperator | 32200 | 51545 | 659.3 | — | 29.4 | 45.047 | 12.992 |
| graded 0.1 mm under components (v2) | 1e+06 | v2 | PfftSheetInductanceOperator | 32200 | 51545 | 659.3 | 460.2 | 45.7 | 44.247 | 19.577 |

Decision: adopted. With the near-field preconditioner the 1 MHz solve of
`power_module` takes 3.5 s on the uniform 0.25 mm grid (83 s with the
diagonal one) and 46 s on the component-driven graded grid (460 s), of which
29 s is the pFFT operator build; the graded grid has 2.2× the branches of the
0.25 mm grid, so the fine pitch, not the operator, sets what is left. The
pFFT pays off where the fine cells are a small part of the board (the strip
line: 42 % of the branches, the same loss). The uniform convolution operator
stays the choice for uniform grids: it is exact to its 24-cell seam and
cheaper to build.

## What is not done

- The barrel's partial self inductance is taken as a given scalar. Nothing
  computes it from the hole.
- A barrel's own partial inductance is still whatever the caller states as a
  scalar; nothing derives it from the hole. It does now take part in the
  vertical operator, but as a column of the in-plane cell's cross-section rather
  than as a plated annulus, so it is understated in the same way its resistance
  once was.
- Skin effect within a barrel is not modelled. A 0.3mm hole plated 25um thick
  is thin against the skin depth at these frequencies, so the wall is uniform;
  a thicker plating or a filled via would need the same treatment as the inlay.
- In-plane current crowding at a conductor's edge is not resolved at the
  optimizer's 0.2mm grid: the skin depth at 300kHz is 0.121mm, so current
  hugging an inlay's vertical side wall falls inside one cell. Measured cost:
  see the ceiling table above. This matters most for a bulk current density
  percentile, which is exactly what the gate reads.
- The end-to-end solve has not been shown to reach its own ceiling. The
  excitation used so far fixes the end-face current density to the DC profile,
  which is a Neumann condition the bar then has to relax out of over an entry
  length of roughly the largest transverse dimension over pi -- about 1.1mm
  here, against measurement windows only 0.6 to 2.4mm from the ends. An
  equipotential end face, or the field solve's own profile injected with its
  phase, would settle it. `Terminal.current_a` now accepts complex current, but
  the equipotential terminal model is still absent.
- Resistance was read from the mean node potential over a column, which has no
  unique meaning where the cross section is not an equipotential. The
  observable to use is the Joule loss, `sum R_b |I_b|^2 / |I|^2`, which is what
  the field solve computes. Not `Re(I^H Z I)` over a sub-region: cutting a dense
  mutual inductance in half lets reactive power exchange across the cut appear
  in the real part.

## CUDA execution

`sheet_cuda.py` uses the same mesh and closed-form kernels as the CPU solver.
Prepared spectra, batched 2D transforms, sparse incidence products, Krylov
vectors, and preconditioner triangular solves reside in CuPy. There is no
implicit CPU fallback. CuPy 14 builds the sparse LU factors with SciPy
SuperLU, then uploads the factors and performs each triangular solve on CUDA;
the result records this as
`scipy_superlu_factor_cupy_triangular_solve`.
`max_iterations` means restart cycles on both paths. The CUDA adapter runs
CuPy's GMRES one restart cycle at a time and checks the public saddle-point
residual between cycles: CuPy judges the left-preconditioned residual, whose
norm differs from the public one by a problem-dependent factor, so each
cycle's internal target is the tolerance scaled by the current ratio of the
two residuals (halved), and the loop stops when the public residual meets
the tolerance or when that target falls below `100 eps`. Before this the
adapter aimed three decades below the tolerance, ran every solve to its
iteration cap and reported `converged=False` with residuals well inside the
tolerance; at `1e-9` it now converges in three cycles on the test problems,
matching the CPU currents to `1e-10`.

An undriven copper island is omitted as both nodes and branches. Omitting only
its KCL columns while retaining its KVL rows allows the dense inductance
operator to induce current in a component with no closure equation. The
returned current of every omitted branch is zero and `undriven_node_count`
records its size.

Measured device acceptance and CPU comparisons are in
`docs/SHEET_CUDA_RESULTS.md`.
