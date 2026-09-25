# Matrix-free MPIR-FEM for electrical PCB analysis

## Scope

The implementation in `src/electrical/matrix_free_mpir_fem` has two physical
front ends:

- layered-PCB DC conduction with structured bilinear Q1 elements and resistive
  vertical connections;
- 2D scalar-polarised (`E_z`) frequency-domain Maxwell with element-wise
  permittivity, permeability, conductivity, and dielectric loss tangent.

The frequency-domain equation retains displacement current and wave
propagation. Conductivity produces eddy currents and skin effect; complex
permittivity produces dielectric loss. It is an exact Maxwell reduction for
geometry and material fields invariant in `z`, not an arbitrary 3D vector
Maxwell solver. DICE-PEEC remains the applicable path for the existing
magneto-quasistatic PCB conductor model.

## Why MPIR

MPIR means **mixed-precision iterative refinement**. The algorithm separates
accuracy from the precision used for most work:

```text
x64 = initial guess
repeat:
    r64 = b64 - A64(x64)                 # reliable FP64 residual
    stop only from ||r64|| / ||b64||
    approximately solve A32 d32 = r32    # FP32 PCG / complex64 GMRES
    x64 = x64 + cast_fp64(d32)           # FP64 update
```

The inner relative tolerance is intentionally loose. One FP32 correction does
not have to reach the requested final accuracy; each FP64 reliable update
removes the error that the preceding low-precision solve left behind. The
solver records high- and low-precision operator-application counts separately,
so a run cannot merely claim mixed precision without showing where its work
occurred.

The DC operator is real SPD and uses FP32 PCG. The frequency-domain operator is
complex symmetric and uses restarted complex64 right-Jacobi GMRES. Both share
the same complex128/FP64 reliable-update loop. A future 3D curl-curl operator
should implement this split-precision contract with Nédélec edge degrees of
freedom rather than weakening the vector conformity requirement.

## Matrix-free Q1 action

For each active element `e`, the solver evaluates

```text
y_e += sigma_e * thickness_e * K_Q1 * x_e
```

and accumulates its four nodal contributions. `K_Q1 = (h_y / h_x) U_x + (h_x /
h_y) U_y` is built from two constant 4-by-4 unit tensors (`U_x = M ⊗ S`, `U_y
= S ⊗ M` with the 1D stiffness `S` and mass `M`), so the element dimensions
enter only through two per-element coefficients `σ t h_y / h_x` and `σ t h_x /
h_y`. The grid is therefore a tensor product of arbitrary column widths and
row heights (`LayeredPCBMesh.pitch_x_m` / `pitch_y_m` scalar or per cell;
`grid.py` builds graded grids, see "Graded tensor grids"). Vias add pairwise
conductance actions `g * (v_a - v_b)`. The global stiffness matrix is never
assembled or stored. Only these items are resident for the low-precision path:

- the two FP32 coefficient arrays and the two 4-by-4 unit tensors;
- the active/free-node mask;
- compact via endpoint and conductance arrays;
- the diagonal used by the Jacobi preconditioner and the small dense inverse
  of the two-level coarse space (`two_level.py`, default on);
- a small number of FP32 PCG vectors.

The FP64 host path holds the physical coefficients and solution vector and is
called once per outer refinement step. This deliberately avoids spending the
majority of execution on CUDA FP64 arithmetic while still making final
convergence a high-precision statement.

## Software boundary

The package is divided into three parts:

| Module | Responsibility |
|---|---|
| `solver.py` | Backend-independent outer MPIR plus inner PCG/GMRES control flow |
| `runtime.py` | float32/complex64 vector primitives for NumPy or CuPy |
| `pcb.py` | Q1 PCB mesh, matrix-free element/via action, physical outputs |
| `grid.py` | `TensorGrid`, graded grid-line generation (`graded_edges`, `refined_grid`), cell overlap fractions |
| `two_level.py` | Jacobi plus patch-constant aggregation coarse correction on a layered node grid (used by the DC and the thermal operators) |
| `frequency_domain.py` | Scalar full-wave Q1 operator, fields, currents, and losses |

`MatrixFreeMPIRSystem` is the solver-facing contract. A new physical operator
provides `apply_high`, `apply_low`, and `diagonal_low`. It may also provide
`precondition_low`, an SPD approximate inverse on the low-precision runtime
that replaces the default Jacobi scaling in every inner solver; the DC and
the thermal operators use this hook for the two-level preconditioner. A new accelerator
provides the low-precision vector runtime. CUDA is optional and imported only
when a CuPy runtime is constructed.

## CUDA execution

Pass `backend="cuda"` to `solve_scalar_maxwell` or `solve_pcb_dc`, or use
`backend="auto"` for CUDA-with-CPU-fallback selection. `device_id` selects a
visible CUDA device. The scalar-Maxwell low operator uses one fused CUDA launch
per application:

1. one thread owns one output node;
2. it gathers the free-node values from at most four adjacent Q1 elements;
3. it applies the complex64 stiffness/reaction tensor locally;
4. it writes one output without atomics.

This node-owned ordering is deterministic and never materialises the global
matrix. CUDA GMRES stores its Arnoldi and preconditioned bases on the device.
Two-pass classical Gram-Schmidt batches each projection into GEMV operations;
only a small Hessenberg column and scalar norms return to the host. The
Hessenberg least-squares solve remains complex128 on the CPU.

Measured on an NVIDIA GeForce GTX 1650 (compute capability 7.5, CuPy 14.1.1):

| Workload | Unknowns | CPU median | CUDA median | CPU/CUDA |
|---|---:|---:|---:|---:|
| Q1 operator | 4,225 | 0.249 ms | 0.0284 ms | 8.78× |
| Q1 operator | 16,641 | 0.792 ms | 0.0354 ms | 22.36× |
| Q1 operator | 66,049 | 3.053 ms | 0.0757 ms | 40.36× |
| Complete MPIR solve | 4,369 | 1,752 ms | 2,067 ms | 0.85× |
| Complete MPIR solve | 16,705 | 4,699 ms | 2,365 ms | 1.99× |

CUDA timings synchronize the stream and exclude operator construction. The
small solve is a negative result: launch/reduction synchronization outweighs
the fast operator until the vectors are large enough. CUDA/CPU action error is
about `8.5e-8`; the 16,705-unknown reliable solutions differ by `4.4e-9` and
both meet the requested complex128 residual tolerance. Raw measurements are in
`MAXWELL_CUDA_RESULTS.json` and `MAXWELL_CUDA_SCALE_RESULTS.json`.

### Pipelined CUDA inner GMRES (measured on `exp/cpp-inner-krylov`)

Profiling one Arnoldi column of the CuPy implementation at 16,705 unknowns
gave about 530 µs, of which the operator was 20 µs. The rest was host-side:
`basis.conj()` copied the whole active basis twice per column (43 µs each),
`norm` and `to_host` each synchronised, the normalising `axpy` was three
kernels, the per-column `lstsq` took 57 µs, and every step waited for the
previous one because the host needed the column before launching the next.
The rewritten `_inner_gmres_cuda` keeps CGS2 and the device-resident bases
and changes the schedule:

- the four projections per column are raw cuBLAS `cgemv` calls on the stored
  basis (`CUBLAS_OP_C` for the conjugate dot products, `CUBLAS_OP_N` with
  `beta = 1` for the in-place updates), so no basis copy or temporary exists;
- the candidate norm is written to device memory by `nrm2`, and an
  elementwise kernel normalises the next basis vector from that device value,
  so column `j+1` is enqueued before the host has seen column `j`;
- each column has its own scratch row, pinned host copy and event; the host
  waits on the event of column `j`, reduces the column with complex128 Givens
  rotations in plain Python complex arithmetic, and decides convergence while
  the device runs column `j+1`. A column enqueued after convergence is
  discarded; its operator application is still counted;
- the correction update `Z y` is one GEMV per cycle.

NVIDIA GeForce GTX 1650 (compute capability 7.5, CuPy 14.1.1, CUDA runtime
13.2), same fixture and `MPIRConfig` as above, five repeats, baseline measured
on the same day with the previous implementation
(`MAXWELL_CUDA_GTX1650_BASELINE_{16x256,64x256,256x256}_RESULTS.json`,
`MAXWELL_CUDA_GTX1650_PIPELINED_{16x256,64x256,256x256}_RESULTS.json`):

| Unknowns | CPU NumPy | CUDA before | CUDA after | Before / after | Per inner iteration after | Inner iterations before / after | Solution difference vs CPU after |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,369 | 1,719 ms | 1,988 ms | 637 ms | 3.12× | 0.193 ms | 3,299 / 3,299 | 3.96e-8 |
| 16,705 | 4,547 ms | 2,166 ms | 697 ms | 3.11× | 0.211 ms | 3,300 / 3,300 | 1.62e-8 |
| 66,049 | 22,067 ms | 5,703 ms | 2,471 ms | 2.31× | 0.516 ms | 4,790 / 4,791 | not converged (2.2e-6 reached) |

Both CUDA variants converge alike; the converged solutions differ from the
CPU path by at most 4e-8 (the Givens reduction replaces `lstsq`, so the
rounding is not identical to before). The remaining column time is split
between the four GEMV passes over the basis, which stream about 2.3 MB per
call at 16,705 unknowns and are bandwidth bound on this device (each about
25 µs), and roughly 150 µs of Python launch overhead that now overlaps with
them. At 66,049 unknowns the GEMV passes dominate (36 MB per column) and the
device is the limit. Against the threaded C++ host path on the same machine
(122 / 275 / 2,410 ms on six cores) the GTX 1650 is slower at the two smaller
sizes and level at the largest; a fused three-pass CGS2 kernel that reads the
basis three times instead of four, and fusing the Jacobi division into the
operator launch, are the remaining device-side items.

## Fused C++ host path (measured on `exp/cpp-inner-krylov`)

The portable NumPy low path spends about half of each inner iteration in the
Q1 operator, which expands the 4-by-4 tensor into roughly eighty whole-array
passes, and the rest in Python-level Gram-Schmidt bookkeeping, a per-iteration
`lstsq`, and temporary allocation. The experiment keeps the Python front end,
the `MatrixFreeMPIRSystem` contract, and the FP64 outer loop, and moves only
the complex64 inner work into one pybind11 extension:

- a node-owned gather Q1 operator with the same ordering as the CUDA kernel,
  written as a branch-free 16-term interior stencil so it vectorises;
- the whole restarted right-Jacobi GMRES cycle (modified Gram-Schmidt, Givens
  rotations on a complex128 Hessenberg, back substitution) per outer step;
- flush-to-zero for subnormal complex64 values during the native call only.
  Subnormal corrections made FP32 SIMD arithmetic several times slower on the
  66,049-unknown case; the FP64 residual remains the acceptance criterion.

The path is opt-in: `MatrixFreeScalarMaxwellOperator(problem, native=True)`
after `python -m electrical.matrix_free_mpir_fem.native.build`. Without the
build, `native=True` raises and the default behaviour is unchanged. The solver
dispatches through the optional `native_inner_gmres` hook; `precondition_low`
systems and the CUDA runtime keep their existing paths.

Measured on an Intel Xeon Platinum 8581C, GCC 14.2.1, NumPy 2.3.5, one
operator thread, `OPENBLAS_NUM_THREADS=1`, same fixture and `MPIRConfig` as
the CUDA benchmark (`experiments/maxwell_native_benchmark.py`,
`MAXWELL_NATIVE_RESULTS.json`):

| Unknowns | Operator NumPy | Operator C++ | Operator ratio | Solve NumPy | Solve C++ | Solve ratio | Inner iterations | Solution difference |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,369 | 0.253 ms | 0.0417 ms | 6.07× | 1,830 ms | 316 ms | 5.79× | 3,299 / 3,299 | 1.15e-8 |
| 16,705 | 0.692 ms | 0.114 ms | 6.06× | 3,990 ms | 976 ms | 4.09× | 2,900 / 2,900 | 9.70e-10 |
| 66,049 | 3.31 ms | 0.430 ms | 7.68× | 27,929 ms | 6,397 ms | 4.37× | 4,790 / 4,792 | not converged |

The 66,049-unknown case stalls on both paths at the 12-outer/400-inner limit
(reached relative residuals 4.9e-7 and 1.6e-6), so its solution difference is
not a correctness statement; the Jacobi-preconditioned inner solve, not the
implementation language, limits that size. With the default OpenBLAS thread
pool (`MAXWELL_NATIVE_DEFAULT_ENV_RESULTS.json`) the spinning BLAS threads
contend with the single native thread and the 16,705-unknown ratio drops to
2.94×; the other two cases are within noise of the table above. Operator
threads (`PCB_NATIVE_THREADS`) gain another 1.2× at 16,705 unknowns with eight
threads because Gram-Schmidt stays single-threaded and L3-bound.

Second environment, Intel Core i7-8700 (6 cores, AVX2, no AVX-512), GCC
14.3.1, NumPy 2.3.5, Python 3.12.14, one operator thread,
`OPENBLAS_NUM_THREADS=1` (`MAXWELL_NATIVE_I7_8700_RESULTS.json`):

| Unknowns | Operator NumPy | Operator C++ | Operator ratio | Solve NumPy | Solve C++ | Solve ratio | Inner iterations | Solution difference |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,369 | 0.249 ms | 0.0402 ms | 6.19× | 1,687 ms | 334 ms | 5.06× | 3,299 / 3,299 | 8.10e-9 |
| 16,705 | 0.734 ms | 0.117 ms | 6.30× | 4,411 ms | 1,071 ms | 4.12× | 3,300 / 2,900 | 5.60e-9 |
| 66,049 | 2.31 ms | 0.435 ms | 5.31× | 21,736 ms | 8,484 ms | 2.56× | 4,790 / 4,792 | not converged |

The 66,049-unknown case again stalls on both paths (reached residuals 1.1e-6
and 1.6e-6). The portable 16,705-unknown solve took one more outer step on this
AVX2 host than on the Xeon (3,300 versus 2,900 inner iterations); the native
path took 2,900 on both, and both reached the requested residual, so the
outer-loop count depends on the host's complex64 rounding rather than on the
implementation. The relative action difference between the two operators is
8.4e-8 on every case. With the default OpenBLAS thread pool
(`MAXWELL_NATIVE_I7_8700_DEFAULT_ENV_RESULTS.json`) the 16,705-unknown ratio
drops from 4.12× to 3.65× and the largest case from 2.56× to 2.48×. With six
operator threads (`PCB_NATIVE_THREADS=6`,
`MAXWELL_NATIVE_I7_8700_THREADS6_RESULTS.json`) the operator alone is 25×
faster than NumPy and the end-to-end solve gains a further 1.21× at 16,705
unknowns (886 ms) and 1.20× at 66,049 unknowns (7,059 ms); the remaining time
is the single-threaded Gram-Schmidt cycle. The adoption criteria hold in this
environment too, with a smaller margin on the largest case (minimum 2.56×
against the 4.09× minimum on the Xeon).

### Threaded inner GMRES (SPMD region)

With only the operator threaded, the modified Gram-Schmidt cycle, norms,
Jacobi division and correction update were about ninety percent of an inner
iteration on six threads: at 16,705 unknowns the iteration took 0.305 ms of
which the operator was 0.029 ms. The restart-32 cycle streams about 65 vector
lengths per iteration against two or three for the operator. `gmres_q1` now
runs the whole solve inside one OpenMP parallel region. Each thread owns a
static range of node rows and applies every vector operation to that range;
the stencil reads neighbouring rows after a barrier; reductions are written
per thread and summed in thread order, so a fixed thread count gives bitwise
reproducible results (checked with two 6-thread solves at 16,705 unknowns);
the 32-by-32 Hessenberg bookkeeping is repeated on thread-private copies.
Flush-to-zero is set per thread, which the earlier version did only on the
calling thread. Convergence is unchanged: inner iterations 3,299 / 2,900 /
4,792 and the same reached residuals at every thread count.

Core i7-8700, `OPENBLAS_NUM_THREADS=1`, `PCB_NATIVE_THREADS` swept
(`MAXWELL_NATIVE_I7_8700_SPMD_T{1,2,3,4,6,12}_RESULTS.json`; T1 and T6 use
five repeats, the others three):

| Threads | Solve C++ 16,705 | Ratio vs NumPy | Solve C++ 66,049 | Ratio vs NumPy | Operator C++ 16,705 | Operator C++ 66,049 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,098 ms | 4.14× | 8,282 ms | 2.64× | 0.111 ms | 0.401 ms |
| 2 | 599 ms | 7.27× | 4,713 ms | 4.59× | 0.0610 ms | 0.212 ms |
| 3 | 423 ms | 10.42× | 3,596 ms | 6.04× | 0.0432 ms | 0.147 ms |
| 4 | 359 ms | 12.31× | 2,991 ms | 7.16× | 0.0360 ms | 0.116 ms |
| 6 | 275 ms | 16.29× | 2,410 ms | 8.95× | 0.0275 ms | 0.0833 ms |
| 12 | 285 ms | 15.59× | 2,549 ms | 8.73× | 0.0313 ms | 0.0831 ms |

Against the operator-only threading above (886 ms and 7,059 ms on six
threads) the SPMD cycle is 3.2× and 2.9× faster; the single-thread times are
within noise of the earlier build (1,071 ms and 8,484 ms). Scaling is close to
linear up to three threads and flattens at the six physical cores; the twelve
hyper-threads gain nothing because the cycle is bound by L3 and DRAM bandwidth,
not by issue rate. The 4,369-unknown case reaches 122 ms on six threads
(14.2×). The default remains one thread; a caller who wants the parallel
cycle sets `PCB_NATIVE_THREADS` to the physical core count.

### Threaded inner GMRES on the Xeon Platinum 8581C

The SPMD cycle was re-measured on the first environment, Intel Xeon Platinum
8581C (16 physical cores, 32 hyper-threads, AVX-512, one socket), GCC 14.2.1,
NumPy 2.3.5, Python 3.12.9, `OPENBLAS_NUM_THREADS=1`, runs sequential,
`PCB_NATIVE_THREADS` swept
(`MAXWELL_NATIVE_XEON_8581C_SPMD_T{1,2,4,8,16,32}_RESULTS.json`; T1 and T16
use five repeats, the others three). Solve medians, NumPy ratio in brackets:

| Threads | 4,369 | 16,705 | 66,049 | Operator C++ 16,705 | Operator C++ 66,049 |
|---:|---:|---:|---:|---:|---:|
| 1 | 234 ms (5.46×) | 719 ms (4.24×) | 4,926 ms (4.33×) | 0.0686 ms | 0.258 ms |
| 2 | 147 ms (8.86×) | 409 ms (7.43×) | 2,520 ms (8.92×) | 0.0374 ms | 0.125 ms |
| 4 | 114 ms (11.59×) | 242 ms (12.55×) | 1,450 ms (15.21×) | 0.0224 ms | 0.0685 ms |
| 8 | 104 ms (12.68×) | 176 ms (17.48×) | 852 ms (25.74×) | 0.0160 ms | 0.0388 ms |
| 16 | 132 ms (9.91×) | 180 ms (16.99×) | 659 ms (33.31×) | 0.0160 ms | 0.0258 ms |
| 32 | 130 ms (10.02×) | 220 ms (13.88×) | 745 ms (28.74×) | 0.0172 ms | 0.0310 ms |

Inner iteration counts are 3,299 / 2,900 / 4,792 at every thread count, as on
the Core i7-8700. The one-thread times match the earlier Xeon build (316 /
976 / 6,397 ms were measured on a busier machine; the portable path also ran
faster in this session, so the ratios are the comparable figures). Scaling
continues to eight threads on all three sizes and to sixteen only on the
largest case, where the 66,049-unknown cycle reaches 7.5× over one thread;
the 4,369-unknown case is best at eight threads and loses ground beyond
because its per-column barriers cost as much as the vector work. The
hyper-threads (32) are slower than sixteen cores everywhere. Reached
residuals differ between thread counts at the double-rounding level (the
4,369-unknown case reaches 9.1e-12 on sixteen threads against 4.2e-12 on the
others) while the converged solutions agree with the portable path within
1.6e-8.

Float-accumulated dot products on the same host
(`MAXWELL_NATIVE_XEON_8581C_FLOAT_DOTS_T{1,16}_RESULTS.json`):

| Unknowns | float64 1 thread | float32 1 thread | float64 16 threads | float32 16 threads | Inner iterations float64 / float32 (1 thread) |
|---:|---:|---:|---:|---:|---:|
| 4,369 | 234 ms | 185 ms | 132 ms | 124 ms | 3,299 / 3,299 |
| 16,705 | 719 ms | 725 ms | 180 ms | 161 ms | 2,900 / 3,300 |
| 66,049 | 4,926 ms | 4,372 ms | 659 ms | 589 ms | 4,792 / 4,792 |

On this AVX-512 host the float32 accumulation gains 11 to 21 percent on one
thread where the iteration count is unchanged, and 6 to 11 percent on sixteen
threads; the 16,705-unknown one-thread case took one more outer step and
ended level. The criteria hold for both settings (minimum ratio 4.23× on one
thread, 10.51× on sixteen; converged solutions within 1.7e-8). The default
stays `float64` and one thread, as recorded above.

### CGS2 orthogonalisation (measured, not adopted)

The restart-32 MGS cycle streams each basis vector twice per column (dot
product, then update). Classical Gram-Schmidt with one reorthogonalisation
reads the new vector once per cache block and each basis vector once per pass,
and needs two barriers per column instead of one per basis vector. It is
available as `PCB_NATIVE_ORTHO=cgs2` (`native_orthogonalization="cgs2"`) and
was measured with `--orthogonalization cgs2` on the same fixture
(`MAXWELL_NATIVE_I7_8700_CGS2_T1_RESULTS.json`,
`MAXWELL_NATIVE_I7_8700_CGS2_T6_RESULTS.json`):

| Unknowns | MGS 1 thread | CGS2 1 thread | MGS 6 threads | CGS2 6 threads | Inner iterations MGS / CGS2 | CGS2 solution difference |
|---:|---:|---:|---:|---:|---:|---:|
| 4,369 | 355 ms | 595 ms | 122 ms | 152 ms | 3,299 / 2,899 | 4.31e-8 |
| 16,705 | 1,098 ms | 2,201 ms | 275 ms | 457 ms | 2,900 / 2,900 | 1.99e-9 |
| 66,049 | 8,282 ms | 16,692 ms | 2,410 ms | 5,411 ms | 4,792 / 4,792 | not converged |

CGS2 is 2.0× slower on one thread and 1.7× slower on six. The traffic model
behind the idea was wrong for this size range: a single-core micro-benchmark
of the complex64 dot product with double accumulation runs at about 24 GB/s at
all three sizes, well below L2 and L3 bandwidth, so the MGS cycle is bound by
the float-to-double conversions and double FMAs, not by memory, and the second
CGS2 pass doubles exactly that work. The same micro-benchmark gives 43 to
53 GB/s when the dot product accumulates in float per 1,024-element block and
sums the blocks in double, which is the next candidate. CGS2 changes the
complex64 rounding, so the 4,369-unknown case takes one outer step fewer and
the converged solutions differ from the portable path by up to 4.3e-8; both
paths still reach the requested FP64 residual.

### Fused vector passes (measured, not adopted)

Three rounding-neutral fusions were tried on the MGS cycle: writing
`Z(col+1) = V(col+1) / diag` in the same pass that scales `V(col+1)`, folding
the Arnoldi vector norm into the last Gram-Schmidt update, and folding the
residual norm into the residual update. Together they remove three or four of
the roughly 65 vector passes per iteration
(`MAXWELL_NATIVE_I7_8700_FUSED_PASSES_T1_RESULTS.json`,
`MAXWELL_NATIVE_I7_8700_FUSED_PASSES_T6_RESULTS.json`):

| Unknowns | MGS 1 thread | Fused 1 thread | MGS 6 threads | Fused 6 threads | Inner iterations MGS / fused |
|---:|---:|---:|---:|---:|---:|
| 4,369 | 355 ms (0.107 ms/it) | 339 ms (0.117 ms/it) | 122 ms | 113 ms | 3,299 / 2,899 |
| 16,705 | 1,098 ms (0.379 ms/it) | 1,347 ms (0.408 ms/it) | 275 ms | 334 ms | 2,900 / 3,300 |
| 66,049 | 8,282 ms (1.73 ms/it) | 8,901 ms (1.86 ms/it) | 2,410 ms | 2,546 ms | 4,792 / 4,792 |

The time per inner iteration rose by 7 to 10 percent on one thread and 5 to
6 percent on six: the loops that mix a complex64 store with a double
reduction vectorise worse than the separate passes, and the traffic saved does
not matter for a compute-bound cycle. The different double summation order in
the fused norms also changes the complex64 rounding, so two of the three cases
took one outer step more or fewer; the FP64 residual was still reached in
both converged cases. The change was reverted; the wall-clock gain at 4,369
unknowns is the shorter iteration count, not the fusion.

### Float-accumulated dot products (measured, criteria met, opt-in)

Following the CGS2 finding, the Gram-Schmidt dot products can accumulate
1,024-element blocks in float and sum the blocks in double
(`PCB_NATIVE_DOT=float32`, `native_dot_accumulation="float32"`,
`--dot-accumulation float32`). Norms stay in double. Same fixture and
`MPIRConfig` (`MAXWELL_NATIVE_I7_8700_FLOAT_DOTS_T1_RESULTS.json`,
`MAXWELL_NATIVE_I7_8700_FLOAT_DOTS_T6_RESULTS.json`):

| Unknowns | MGS float64 1 thread | float32 1 thread | MGS float64 6 threads | float32 6 threads | Inner iterations float64 / float32 (1 thread) | float32 solution difference |
|---:|---:|---:|---:|---:|---:|---:|
| 4,369 | 355 ms (0.107 ms/it) | 289 ms (0.087 ms/it) | 122 ms (0.037 ms/it) | 107 ms (0.032 ms/it) | 3,299 / 3,299 | 1.28e-8 |
| 16,705 | 1,098 ms (0.379 ms/it) | 977 ms (0.296 ms/it) | 275 ms (0.095 ms/it) | 227 ms (0.078 ms/it) | 2,900 / 3,300 | 4.02e-9 (1 thread), 2.42e-8 (6 threads) |
| 66,049 | 8,282 ms (1.73 ms/it) | 7,153 ms (1.49 ms/it) | 2,410 ms (0.503 ms/it) | 2,310 ms (0.482 ms/it) | 4,792 / 4,792 | not converged |

The inner iteration is 14 to 22 percent shorter on one thread and 4 to
18 percent on six; the gain shrinks where the cycle becomes bandwidth bound.
The benchmark's criteria against the portable path hold (minimum end-to-end
ratio 3.04× on one thread, converged solutions within 2.4e-8). Because the
complex64 rounding of the coefficients changes, the outer iteration count can
differ by one from the float64 path and the result depends on the thread
count's block boundaries; the default stays `float64`, which reproduces the
portable rounding, and switching the default is a separate decision.

Decision recorded in the JSON: the benchmark criteria (every case at least 2×,
identical convergence outcome, converged solutions within 1e-6) are met. The
extension is not packaged in the wheel and CUDA execution was not measured in
this environment, so integration into `feature/` requires the packaging and
CI work described in `AGENTS.md` before the default path changes.

## Roofline of the matrix-free action (measured on `exp/roofline-matrix-free`)

`experiments/roofline_matrix_free.py` (with `roofline_matrix_free.cpp`) measures
how the shipped C++ Q1 action uses the host's bandwidth and FMA throughput. It
compiles the production kernels in from their sources and compares them with the
same operator assembled by probing `apply_low` with 9 coloured vectors. The
assembled operator is stored two ways: as a structured stencil with one
complex64 coefficient array per neighbour offset and no indices, and as CSR
with int32 indices. All three use the same static row partition and a
preallocated output. The assembled actions agree with the matrix-free action to
8.4e-8. This KVM guest has no hardware counters, so flop and byte counts are
analytic: the flops the loop executes per interior node, and the compulsory
traffic of one pass with no write-allocate. Host: Xeon Platinum 8581C, GCC
14.2.1, NumPy 2.3.5, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`
(`ROOFLINE_MATRIX_FREE_XEON_8581C_RESULTS.json`).

Machine probes (AVX-512 FMA with 16 independent chains; float32 read sums with
eight accumulators, repeated inside one parallel region):

| Threads | FP32 FMA peak | FP64 FMA peak | L2 read | L3 read | DRAM read | DRAM triad |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 239 GFLOP/s | 120 GFLOP/s | 181 GB/s | 34 GB/s | 18 GB/s | 13 GB/s |
| 4 | 957 GFLOP/s | 472 GFLOP/s | 627 GB/s | 128 GB/s | 72 GB/s | 53 GB/s |
| 8 | 1,899 GFLOP/s | 916 GFLOP/s | 1,266 GB/s | 231 GB/s | 139 GB/s | 105 GB/s |
| 16 | 3,338 GFLOP/s | 1,698 GFLOP/s | 2,457 GB/s | 340 GB/s | 198 GB/s | 171 GB/s |
| 32 | 3,298 GFLOP/s | 1,661 GFLOP/s | 1,093 GB/s | 335 GB/s | 237 GB/s | 181 GB/s |

On sixteen cores the FP32 machine balance is 17 flop/byte against DRAM, 10
against L3, and 1.4 against L2.

Per interior node, the matrix-free action executes 391 flops on 36 compulsory
bytes (10.9 flop/byte):

- 16 (element, local column) terms. Each term forms its coefficient
  `imu*K + rea*M` (14 flops), applies the Dirichlet mask (2), and does the
  complex multiply-add (8).
- The Dirichlet blend adds 7 flops.

The assembled stencil needs 72 flops on 88 bytes (0.82 flop/byte), and CSR
needs 72 flops on 128 bytes. The matrix-free form therefore executes 5.4 times
the useful arithmetic. Of the 391 flops:

- 57 % forms coefficients;
- 8 % applies the mask;
- 14 % is the duplicate products of the 16 element terms that the stencil
  merges into 9.

Action medians:

| Nodes | Storage MF / stencil / CSR | Threads | Matrix-free | Stencil | CSR | Stencil / MF | MF GFLOP/s (of FP32 peak) | Stencil GB/s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 66,049 | 1.4 / 4.8 / 7.4 MB | 1 | 0.245 ms | 0.168 ms | 0.551 ms | 0.69 | 105 (44%) | 35 |
| 66,049 | 1.4 / 4.8 / 7.4 MB | 16 | 0.026 ms | 0.012 ms | 0.043 ms | 0.45 | 1,007 (30%) | 500 |
| 1,050,625 | 22.0 / 75.6 / 117.5 MB | 1 | 3.045 ms | 2.666 ms | 8.616 ms | 0.88 | 135 (56%) | 35 |
| 1,050,625 | 22.0 / 75.6 / 117.5 MB | 16 | 0.242 ms | 0.336 ms | 0.648 ms | 1.39 | 1,697 (51%) | 275 |
| 16,785,409 | 352.4 / 1,208.5 / 1,879.4 MB | 1 | 50.666 ms | 112.814 ms | 249.210 ms | 2.23 | 130 (54%) | 13 |
| 16,785,409 | 352.4 / 1,208.5 / 1,879.4 MB | 16 | 3.760 ms | 8.114 ms | 17.205 ms | 2.16 | 1,746 (52%) | 182 |

The matrix-free action is compute-bound at 30 to 56 % of the FP32 FMA peak at
every size. The assembled stencil is bandwidth-bound: it reaches the L3 read
rate on one thread and 92 % of the DRAM read rate on sixteen. The result
depends on where the working set lives:

- **Cache-resident** (the 66,049-node documented fixture): the stencil is 1.4
  to 2.2 times faster, because only the flops remain.
- **DRAM** (16.8 million nodes): the matrix-free action is 2.2 times faster
  and needs 3.4 times less storage than the stencil and 5.3 times less than
  CSR. It then runs at 81 % of the DRAM read rate and 52 % of the FMA peak
  together, which is close to balanced on this host.

CSR is slower than the matrix-free action everywhere.

The action itself is a small part of an MPIR solve. The next table uses the
66,049-node fixture with the `maxwell_native_benchmark.py` configuration. The
solve did not converge in 12 outer steps, as recorded above. Operator shares
are applications times the isolated action median; "other" is the inner
Gram-Schmidt and vector updates plus the outer bookkeeping. The stencil column
is an estimate, not a measured solve.

| Threads | Solve | Low operator (4,942 applications) | FP64 high operator | Other | Estimated solve with stencil |
|---:|---:|---:|---:|---:|---:|
| 1 | 4,680 ms | 26% | 2% (13 x 6.8 ms) | 72% | 4,300 ms |
| 4 | 1,330 ms | 25% | 6% (13 x 6.2 ms) | 69% | 1,131 ms |
| 8 | 793 ms | 23% | 10% (13 x 6.2 ms) | 67% | 700 ms |
| 16 | 606 ms | 21% | 14% (13 x 6.4 ms) | 65% | 537 ms |

Replacing the action with the assembled stencil would save an estimated 8 to
15 % at this size and lose 2.2 times at DRAM size, while needing 3.4 times the
storage. **Decision:** the assembled stencil and CSR were measured, not
adopted. The waste worth attacking is:

- the Gram-Schmidt and vector passes (two thirds of the solve);
- the arithmetic inside the action. Coefficient formation is 57 % of it and
  could be cut by applying the element matrices `K` and `M` to the gathered
  values before weighting them by `imu` and `rea` (not measured).
- the single-threaded NumPy FP64 outer operator, whose share grows with the
  thread count.

## Tenstorrent Blackhole migration

The intended first Blackhole port keeps FP64 outer refinement on the host and
moves the repeated FP32 correction solve to the accelerator. This requires a
TT runtime plus a TT implementation of the low operator; it does not require a
rewrite of the convergence policy or PCB input/result contracts.

Recommended dataflow decomposition:

1. Keep coefficients, masks, via arrays, diagonal, and Krylov vectors resident on
   the device for all inner iterations.
2. Stream structured element tiles through local memory and apply the 4-by-4
   tensor without materialising element matrices.
3. Accumulate nodal contributions with node-owned tiles or deterministic
   element colouring; do not make unordered floating-point atomics part of the
   numerical contract.
4. Fuse Jacobi scaling and vector updates where profiling shows that memory
   traffic dominates.
5. Perform hierarchical FP32/complex64 dot and norm reductions on device. Return scalar
   convergence data, not whole vectors, during an inner solve.
6. Transfer one FP32 residual to the device and one correction back per outer
   step. The FP64 residual remains the sole acceptance criterion.

Before a TT backend is accepted, compare it with the NumPy runtime on the same
operator and require:

- the requested FP64 outer residual is reached;
- voltage, current density, via current, and Joule loss meet explicit error
  bounds against a dense small-system reference;
- low operator applications materially outnumber high applications;
- no assembled global matrix appears in device memory telemetry;
- repeated runs are deterministic within the documented reduction tolerance.

## Graded tensor grids

`grid.graded_edges(start, end, coarse_pitch_m=, fine_pitch_m=, refine_m=[(a,
b), ...], margin_m=, growth=)` places grid lines so that cells are
`fine_pitch_m` wide over every refined interval widened by the margin,
`coarse_pitch_m` far from them, and grow geometrically by about `growth`
between: the target size `s(x) = min(H, h + (r − 1) d)` (distance `d` to the
nearest interval) is integrated as `∫ dx / s`, and lines sit where the
integral crosses whole numbers, so the last cell ends exactly at `end` and
the count is the rounded integral. `refined_grid` does both axes from
refinement boxes; `TensorGrid.cell_overlap_fraction` spreads a footprint
power over exactly its area on any grid. Refining a box refines the whole
strip of the board in each axis, the usual tensor-grid cost of keeping every
kernel structured. The thermal and DC meshes take the resulting per-cell
pitches; the electro-thermal scenarios require both to share the grid lines.

The DC operator now defaults to the two-level preconditioner: with Jacobi
alone a wide plane or a graded grid with elongated cells costs several
hundred inner PCG iterations per outer step and hit the 200-iteration cap.
Measured (`TENSOR_GRID_RESULTS.json`, `experiments/tensor_grid_acceptance.py`,
Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz), `MPIRConfig(max_inner_iterations=4000)` so that Jacobi converges too:

| Case | Elements | Two-level inner iterations | Jacobi inner iterations | Resistance error |
|---|---|---|---|---|
| 0.2 mm trace, graded length (one row) | 59 | 96 (15 ms) | 419 (35 ms) | 3.9e-13 |
| 0.2 mm trace, graded length and rows 0.05/0.1/0.05 | 177 | 255 (34 ms) | 1208 (112 ms) | 2.1e-04 |
| 40 x 60 mm plane, uniform 0.5 mm | 9600 | 54 (84 ms) | 821 (267 ms) | 3.0e-04 |

The one-row trace is exact on the graded grid, as Q1 must be for a uniform
current on any rectangle grid. The two multi-row cases carry a `2e-4` to
`3e-4` error that is the terminals' equal nodal current shares (a uniform
injection would weight the nodes by their row heights), not the
discretisation. Decision: adopted. The thermal side of the same measurement
is in `THERMAL_MPIR_FEM.md`.

## Current limitations and next physics increments

- The meshes are structured tensor grids (graded, but rectangular); curved
  pads and barrels are not geometrically resolved, and a refinement box
  refines its whole row and column strips.
- Conductivity is scalar and isotropic within an element.
- Field and current density are reported at element centres.
- One reference node supplies the voltage gauge. Disconnected conductive
  components should be solved separately or explicitly connected.
- The full-wave front end is the 2D scalar `E_z` reduction. Arbitrary 3D vector
  fields require curl-conforming Nédélec edge elements.
- Ports, PML/open radiation boundaries, S-parameters, dispersive material
  models, and nonlinear magnetics are not implemented.

The accuracy and CPU timing audit is in
[MAXWELL_VALIDATION_REPORT.html](MAXWELL_VALIDATION_REPORT.html), backed by
`MAXWELL_SMALL_RESULTS.json`. A matched DC strip, CUDA timing, operator-memory,
skin-effect, and capability comparison with Sheet PEEC is in
[PEEC_FEM_COMPARISON_REPORT.html](PEEC_FEM_COMPARISON_REPORT.html), backed by
`PEEC_FEM_COMPARISON_RESULTS.json`. The next 3D increment needs edge elements,
ports, and an absorbing boundary. The electrothermal extension lives under
`src/thermal/matrix_free_mpir_fem` and receives the per-element and per-via
Joule loss that `solve_pcb_dc` now reports; see
[THERMAL_MPIR_FEM.md](THERMAL_MPIR_FEM.md).
