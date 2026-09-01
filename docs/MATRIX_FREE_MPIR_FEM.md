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
provides `apply_high`, `apply_low`, and `diagonal_low`. A new accelerator
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
ports, and an absorbing boundary. An electrothermal extension belongs under
`src/thermal/<method+acceleration>` and can exchange Joule-loss fields through
a physics-neutral coupling layer.
