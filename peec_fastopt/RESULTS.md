# DICE-PEEC design and prototype results

## Conclusion

The shortest optimization time is obtained by avoiding a converged PEEC solve
for most routing proposals. The proposed DICE-PEEC pipeline scores local
candidates from cached fields and adjoints, estimates whether the cached
solution is still trustworthy, and solves only promising or uncertain
candidates.

This differs from merely moving a dense PEEC solver to CUDA. CUDA accelerates
the remaining FFT and Krylov work, while sparse-delta evaluation removes most
of that work from the candidate loop.

## Measured prototype result

Environment: CPU-only NumPy/SciPy runtime. The benchmark uses a scalar
translation-invariant magnetoquasistatic interaction proxy and local rectangular
reroutes. Full scoring performs one FFT convolution per candidate. Delta
scoring caches the base field and evaluates the exact quadratic identity.

| Grid | Candidates | Mean changed cells | Full FFT estimated | Delta score | Speedup | Maximum relative error | Top 10% overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 x 128 | 500 | 13.28 | 1.127 s | 0.00594 s | 190x | 2.51e-16 | 100% |
| 256 x 256 | 1000 | 21.51 | 10.519 s | 0.01312 s | 802x | 4.43e-16 | 100% |
| 512 x 512 | 1000 | 38.12 | 57.788 s | 0.01679 s | 3441x | 3.99e-16 | 100% |

The full-FFT time is estimated from 100 validation candidates for the first two
rows and 50 for the final row. Delta time covers every candidate. Timings are a
single run and should not be treated as a hardware-independent performance
claim.

## Proposed full optimization loop

1. Keep the accepted-layout PEEC solution, FFT fields, adjoints, local
   preconditioner, and Krylov recycle vectors resident on the GPU.
2. Represent all proposals as sparse changes from the accepted layout.
3. Reject DRC failures and dominated geometric candidates.
4. Apply exact delta interaction scores and adjoint objective estimates to all
   remaining candidates in one GPU batch.
5. Compute a residual-based trust indicator. Route topology, port, layer-change,
   and return-path edits are marked high risk even if the scalar residual is
   small.
6. Run warm-started FGMRES only for the best predicted candidates and the
   high-uncertainty candidates.
7. Fully refresh cached state only after a candidate is accepted.
8. Randomly audit rejected candidates and adapt thresholds so that recall of
   genuinely good candidates remains above the required level.

## Expected total-loop effect

If `C` candidates would otherwise require a full solve and DICE-PEEC corrects
only a fraction `p`, the dominant solve count falls from `C` to approximately
`p*C + 1` per accepted-layout epoch. Before considering warm starts or CUDA,
correcting 2%, 5%, or 10% of candidates therefore gives upper-bound solve-count
reductions of roughly 50x, 20x, or 10x. Actual speedup is lower because delta
evaluation, residual checks, audits, and state refreshes are not free.

The prototype shows that the delta interaction portion is cheap enough not to
dominate this budget. The real determinant will be how small the correction
fraction can be while preserving the best-candidate ranking.

## Implementation sequence

### Phase 1: useful before a full CUDA PEEC solver

Connect the router to sparse candidate deltas and use current-path templates to
calculate incremental inductive energy, mutual coupling, return-path distance,
and loop-area scores. Validate the top candidates with the existing solver.

### Phase 2: matrix-free CUDA solver

Implement the accepted-layout solve with fixed-grid aoFFT-PEEC using cuFFT,
cuSPARSE, and FGMRES. Cache Green tensors and FFT plans across candidates and
frequencies.

### Phase 3: adjoint and residual gates

Add adjoints for the actual optimization objectives, candidate residual norms,
warm starts, and local preconditioner updates. Calibrate correction thresholds
from exact-solve audits.

### Phase 4: adaptive fidelity

Use coarse magnetoquasistatic evaluation for most proposals, capacitance-aware
PEEC for finalists, and full-wave/sign-off analysis only at milestones.

## Evidence base

- FFT-PEEC reduces dense interaction products to FFT operations without forming
  the dense matrices: https://doi.org/10.1109/TPEL.2021.3092431
- The 2026 PCB-focused solver combines anisotropic FFT, thin-conductor
  treatment, and via replacement for million-unknown PCB models:
  https://doi.org/10.1109/TEMC.2025.3626010
- PyPEEC provides an open FFT-accelerated implementation and optional
  CuPy/CUDA backend: https://doi.org/10.21105/joss.06644

## Scope warning

The prototype does not yet solve the complete PEEC MNA system and does not
establish the correction fraction obtainable on a real KiCad board. Its result
establishes only that local quadratic interaction terms can be updated exactly
and extremely cheaply. Real-board validation must measure top-candidate recall,
not just scalar score error.

## Low-memory near/far calibration

A second experiment used exact pairwise interaction energy as the reference and
approximated distant sources with spatial blocks. Near sources were restored
exactly. Both monopole and dipole far-field expansions were tested.

For a 256 x 256 routing grid, 100 local reroutes, block width 2, dipole far
field, and mixed float32 field/float64 reduction:

| Exact near radius | Maximum relative error | Top 10% overlap | True top 10% recall in 20% shortlist |
|---:|---:|---:|---:|
| 8 cells | about 1.54e-4 | 90% | 100% |
| 16 cells | about 3.78e-5 | 100% | 100% |

The radius-8 experiment was repeated with seeds 3, 7, 11, and 17. Every run
retained all truly top-10% candidates inside the approximate top-20% shortlist.
The radius-16 pass recovered 100% top-10% overlap in all four runs.

Using block width 4 reduced the far-field representation size, but it was not
safe for final ranking. At radius 16 its maximum scalar error was only about
6.63e-5, yet top-10% overlap was 60% in the examined seed. This demonstrates
that absolute field error alone is not a sufficient acceptance criterion when
candidate scores are close.

Float32 field arithmetic with float64 global reductions differed from the
float64 result by at most about 6.98e-9 in the block-2/radius-16 experiment.
The dominant error was spatial coarse-graining, not floating-point storage.

These results motivate a two-pass low-memory cascade: radius 8 for all
candidates, radius 16 for the best 20%, and a full correction solve only for
overlapping error intervals. The result is specific to the scalar validation
model; a complete PEEC implementation must recalibrate the thresholds for each
objective and frequency range.

## Controller verification

The backend-neutral controller is covered by 11 CPU-side unit tests. The test
suite verifies exact delta scoring, 4/8 GB safe-budget compliance, smaller-tile
OOM recovery, BiCGSTAB-to-GMRES fallback, complex128 precision promotion,
interval/topology candidate promotion, audit-driven radius and shortlist
tightening, and protection against applying partial cuFFT allocation data to
the complete operator model.

For the default synthetic 1024 x 1024, four-layer, two-million-unknown profile,
all fidelity stages fit the emulated Windows 4 GB safe allocation budget. This
is a controller/model check, not proof that a complete physical PEEC operator
has that footprint. The next measurement must replace estimates with actual
peak VRAM from the integrated CUDA executor.
