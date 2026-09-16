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

and accumulates its four nodal contributions. `K_Q1` is a constant 4-by-4
rectangular-element tensor. Vias add pairwise conductance actions
`g * (v_a - v_b)`. The global stiffness matrix is never assembled or stored.
Only these items are resident for the low-precision path:

- FP32 sheet-conductance coefficients and the 4-by-4 element tensor;
- the active/free-node mask;
- compact via endpoint and conductance arrays;
- the diagonal used by the Jacobi preconditioner;
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
| `frequency_domain.py` | Scalar full-wave Q1 operator, fields, currents, and losses |

`MatrixFreeMPIRSystem` is the solver-facing contract. A new physical operator
provides `apply_high`, `apply_low`, and `diagonal_low`. It may also provide
`precondition_low`, an SPD approximate inverse on the low-precision runtime
that replaces the default Jacobi scaling in every inner solver; the thermal
package uses this hook for its two-level preconditioner. A new accelerator
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

Decision recorded in the JSON: the benchmark criteria (every case at least 2×,
identical convergence outcome, converged solutions within 1e-6) are met. The
extension is not packaged in the wheel and CUDA execution was not measured in
this environment, so integration into `feature/` requires the packaging and
CI work described in `AGENTS.md` before the default path changes.

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

## Current limitations and next physics increments

- The meshes are structured and rectangular; curved pads and barrels are not
  geometrically resolved.
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
