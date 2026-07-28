# DICE-PEEC: optimization-oriented incremental PEEC

## Objective

Minimize total PCB routing optimization time, rather than the latency of one
high-accuracy PEEC solve. Routing optimizers normally generate many local
variants around the same accepted layout. Re-solving all variants to the same
tolerance discards this structure.

DICE-PEEC (Delta-Incremental Cascade Evaluation for PEEC) combines four ideas:

1. a fixed-grid matrix-free FFT-PEEC operator;
2. exact sparse-delta evaluation for quadratic interaction metrics;
3. adjoint screening plus a residual-based error indicator;
4. selective warm-started correction solves only for promising or uncertain
   candidates.

## Core state

For the currently accepted layout `g0`, cache:

- the converged PEEC solution `x0`;
- the matrix-free operator state and FFT Green tensors;
- the fields `Lp * i0` and `P * q0`;
- one adjoint vector per optimization objective;
- the sparse local preconditioner;
- the previous Krylov recycle space.

The grid, board outline, stackup, and FFT kernels remain fixed during a local
optimization epoch. A candidate is represented by a sparse change list instead
of a complete board copy.

## Candidate cascade

### Gate 0: geometry

Reject DRC violations and dominated routes. Calculate length, bend, via, and
return-path discontinuity features in a batch.

### Gate 1: exact incremental quadratic physics

For a cached interaction field `phi = K x0` and local change `d`:

```text
E(x0 + d) - E(x0) = 2 d^T phi + d^T K d
```

The first term is `O(s)` and the local self-interaction is `O(s^2)`, where `s`
is the number of changed cells. This identity can score inductive-energy,
charge-interaction, and fixed-current crosstalk proxies without a new FFT.

### Gate 2: adjoint estimate

For `A(g) x = b(g)` and objective `J`, solve the adjoint once at the accepted
layout. Each candidate then uses only contractions with its local matrix change:

```text
Delta J ~= J_g Delta g - lambda^H (Delta A x0 - Delta b)
```

This turns one solve per candidate into one solve per accepted-layout epoch.

### Gate 3: residual trust test

Evaluate the old solution under the candidate operator:

```text
r_c = b_c - A_c x0
eta_c = ||r_c|| / ||b_c||
```

If `eta_c` is small and the candidate is not near an objective constraint, the
incremental score is accepted. Otherwise the candidate enters the correction
queue. The threshold is calibrated against exact solves, not chosen only from
the linear-system tolerance.

### Gate 4: selective correction

Correct only the best and most uncertain candidates with warm-started FGMRES.
Reuse the base preconditioner and recycle space. Update only local Schwarz or
ILU blocks touched by the candidate. Evaluate candidates in a GPU batch grouped
by changed-cell count and frequency set.

### Commit

After accepting a candidate, refresh the full solution, adjoints, fields, and
preconditioner. Proposals rejected during that epoch never trigger a full
refresh.

## CUDA mapping

- cuFFT: batched tensor convolutions for `Lp` and `P`.
- cuSPARSE: incidence matrices and local preconditioner.
- cuBLAS: Krylov orthogonalization and block vector operations.
- custom kernels: sparse candidate gather, delta contractions, residual norms,
  and batched objective reduction.
- CUDA Graphs: one captured correction-iteration graph per batch shape.

Candidate data should use structure-of-arrays storage. Keep FFT kernels,
accepted-layout vectors, and adjoints resident on the GPU. Transfer only compact
candidate deltas and final scalar scores.

## Important limitation

The exact delta identity applies directly to quadratic metrics with a fixed or
explicitly updated current/charge vector. It is not an exact replacement for
solving the candidate PEEC circuit because currents and charges also change.
The adjoint estimate, residual gate, and selective correction solve are what
control that approximation.

## Practical default policy

- Score every candidate with Gates 0-2.
- Correct the best 2% by predicted score.
- Also correct candidates whose residual indicator is in the worst 5%, whose
  predicted constraint margin is small, or whose topology changes a port or
  return path.
- Randomly audit 1% of rejected candidates to detect ranking drift.
- Tighten or relax the gates online to keep top-candidate recall above 99%.

This policy makes total time approximately

```text
T = T_refresh + C * T_delta + K * T_correction
```

instead of `C * T_full`, with `K` much smaller than candidate count `C`.

## Low-memory 4/8 GB operator

A PCB must not be split into electromagnetically independent tiles. Distant
conductors remain coupled. Instead split the interaction operator:

```text
K = K_near + K_far
```

`K_near` is evaluated exactly in streamed overlapping tiles. `K_far` is
represented on a global coarse 2.5D layer grid. This gives every tile a global
coarse correction and avoids artificial electromagnetic boundaries.

The PCB stackup is represented as conductive sheets. FFTs are two-dimensional
in the board plane, while layer-to-layer coupling is a small matrix for each
spatial frequency. Kernel channels and layer pairs are streamed through a
bounded GPU cache rather than stored simultaneously. Important vias remain
local 3D objects; ordinary vias are frequency-dependent lumped elements.

For a 4 GB device, cap application allocation at about 3.0 GB. Keep only one
near-field tile, one or two kernel spectra, 8-12 short-recurrence solver
vectors, and the accepted-layout cache resident. For an 8 GB device, increase
the kernel cache and process two tiles concurrently. Candidate states remain
sparse deltas in either case.

### Error-driven fidelity cascade

The scalar calibration experiment supports this default:

1. all candidates: block width 2, exact near radius 8, dipole far field,
   float32 field arithmetic with float64 reductions;
2. best 20% and high-uncertainty candidates: increase exact near radius to 16;
3. finalists: streamed full 2.5D PEEC correction solve;
4. accepted candidate: tight residual and grid-refinement audit.

Do not accept a ranking solely because two scalar scores differ. Candidate `a`
is confidently better than `b` only when

```text
J_a + eta_a < J_b - eta_b
```

where `eta` combines split, iterative, precision, and discretization error
estimates. Overlapping intervals trigger the next fidelity level.

Use the following error indicators:

- split error: difference between near radii 8 and 16;
- iterative error: adjoint-weighted residual `abs(lambda^H r)`;
- precision error: periodic float64 replay of the objective and residual;
- mesh error: selected candidates repeated on a refined grid;
- ranking drift: random exact audits of rejected candidates.

This policy deliberately spends more time only near decision boundaries.

## Integration boundary

The router should submit a base-layout identifier and sparse operations such as
`add_segment`, `remove_segment`, `add_via`, and `remove_via`. DICE-PEEC returns
the predicted objective vector, uncertainty/residual indicator, fidelity level,
and whether a correction solve was executed. The LLM receives only these
structured diagnostics and does not select numerical solver tolerances.

## Dynamic controller implemented in this prototype

`controller.py` turns the policy above into an executable state machine. It
does not hard-code a 4 GB or 8 GB configuration. On every stage it reads total
and free device memory, reserves platform-dependent headroom, searches tile,
candidate-batch, and kernel-batch combinations, and returns the largest safe
throughput plan.

The controller consumes execution evidence and changes subsequent plans:

- OOM: discard the failing tile size and retry with a smaller tile;
- Krylov stagnation or non-convergence: replace BiCGSTAB(2) with restarted
  GMRES;
- float32/float64 replay gap above `1e-5`: use complex128 field storage while
  retaining float64 reductions;
- missed promising candidate in an audit: enlarge the shortlist and double the
  near-field radii within bounded limits;
- 100 stable audits: cautiously relax toward the calibrated defaults.

Candidate promotion is interval-based. Split, iterative, precision, and mesh
errors are added into a conservative score interval. The normal shortlist,
overlapping intervals, excessive relative uncertainty, and topology-risk flags
all trigger promotion. This keeps memory adaptation and accuracy adaptation in
the same feedback loop without allowing available VRAM to determine accuracy.

The analytic memory estimator is deliberately conservative until complete
operator telemetry is available. Partial cuFFT calibration reports are marked
`memory_complete=False`, so they update timing but cannot incorrectly scale the
whole PEEC memory model.

## Multilayer implementation status (peec-cuda)

plane_opt will own multilayer pathfinding and via insertion.  peec-cuda owns
the fixed-grid 2.5D interaction model and sparse candidate scoring that those
edits feed.

| Component | Module | Status |
|---|---|---|
| Stackup / layer z | `stackup.py` | implemented |
| Sparse multilayer delta | `multilayer_peec.SparseDeltaML` | implemented |
| 2.5D FFT + interlayer kernel matrix | `multilayer_peec.FFTInteraction25D` | implemented |
| Exact delta quadratic identity | `multilayer_peec.MultilayerDeltaScorer` | implemented |
| Lumped via energy (R + jωL proxy) | `multilayer_peec.ViaSpec` / `ViaSet` | implemented |
| Router ops → delta | `layout_ops` (`add`/`remove` segment & via) | implemented |
| Near/far 2.5D cascade | `lowmem_25d.py` | implemented |
| Dynamic memory controller `layers` | `controller.py` | already parameterized |
| Full PEEC-MNA / adjoint / residual gates | — | not yet |
| plane_opt current-field schema mapping | `plane_opt_contract.py` | implemented |
| Sheet PEEC CUDA execution | `sheet_cuda.py` | implemented; no CPU fallback |
| Physical CUDA PyPEEC path | `cuda_pypeec.py` | layer-agnostic executor; mapping-dependent |

### Integration contract for plane_opt

The full current-field solve consumes
`plane-opt-current-field-problem/v1` through
`plane_opt_contract.solve_plane_opt_problem()`. The mapping contains each
layer's own thickness and center Z, conductor cells, complex terminals,
scenario frequency, and explicit vertical segments; this package does not
import `plane_opt`.

The sparse candidate-scoring path remains:

1. Build a `Stackup` (layer names + z centers in mm) for the board.
2. Represent the accepted layout as occupancy `x0` with shape
   `(n_layers, ny, nx)` and a base `ViaSet`.
3. For each candidate, emit `CandidateEdit` with `SegmentOp` / `ViaOp`, or
   construct `SparseDeltaML` + `ViaSet` directly.
4. Score with `MultilayerDeltaScorer` (Gate 1).  Use `high_risk_topology` from
   `compile_candidate` when vias or layer transitions are present (feeds Gate 3
   promotion policy in `controller.select_for_promotion`).
5. Keep full PyPEEC/CUDA correction solves for promoted candidates only.

Cell coordinates are in the fixed routing grid.  Vertical distances use
`stackup.z_mm` converted by `cell_size_m` so interlayer kernels stay consistent
with the planar FFT grid.
