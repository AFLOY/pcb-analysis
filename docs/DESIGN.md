# pcb-analysis design

## Source taxonomy

Source code is grouped first by analysis target and then by numerical method
plus acceleration strategy:

```text
src/electrical/{sheet_peec,voxel_peec}/
src/electrical/matrix_free_mpir_fem/
src/thermal/matrix_free_mpir_fem/
src/emc/tiled_dipole_superposition/
src/multiphysics/staggered_coupling/
```

The first directory holds the PEEC solvers: the 2.5D sheet PEEC
([SHEET_PEEC.md](SHEET_PEEC.md)) and the 3D voxel PEEC through PyPEEC
([VOXEL_PEEC.md](VOXEL_PEEC.md)). The
second contains matrix-free FEM accelerated by mixed-precision iterative
refinement (MPIR). The third is the thermal implementation: steady heat
conduction through the board stack on the same MPIR solver and low-precision
runtimes, documented in [THERMAL_MPIR_FEM.md](THERMAL_MPIR_FEM.md). The
solver and runtimes are imported from the electrical package rather than
copied; the discretisations and front ends stay apart. The fourth is the EMC
front end: radiated emissions by tiled superposition of the current elements
either electrical solve produces, documented in
[EMC_DIPOLE_SUPERPOSITION.md](EMC_DIPOLE_SUPERPOSITION.md). It owns no field
solve of its own. The fifth chains the others into coupled scenarios: a
staggered electro-thermal fixed point and the emission of its converged
current, documented in [MULTIPHYSICS_SCENARIOS.md](MULTIPHYSICS_SCENARIOS.md).

## C++ core

The numerical work runs in one C++ extension, `electrical._pcbcore`, built
from `cpp/` with CMake (the sources of the kernels that predate it stay beside
their packages and are compiled into the same module).  The Python packages
are facades: they validate inputs, convert dataclasses to arrays, call the
core with the GIL released and wrap the results; the NumPy (and CuPy)
implementations stay as references and as the path without a build.  The
only threading control is the process-wide budget of `electrical.threads`.

## Reproducibility

**Same bits, at the same thread counts.** On one machine and one build, the
same input solved with the same thread budget and the same
`near_factor_threads` gives the same bits in every process, whatever
`PYTHONHASHSEED`, the memory free (which sets how a lockstep group is cut) or
how many sets the operator applies at once.  This is the promise; other
thread counts agree within the solver's tolerance.

The MUMPS near factor (the CPU sheet solve's default where the core links
MUMPS) is what ties the bits to a thread count.  Its OpenMP factorization
sums in an order set by its team: with its L0 tree parallelism
(`ICNTL(48)`) the order changed from run to run even at one count, and with
it off (as set) the bits are fixed per count but differ between counts (on a
small mesh 1 and 2, 3, and 4 and more threads gave three digests; on
Flow2's EPYC 9965, 8 and 48 threads agreed and 1 did not).  So the factor
runs on one thread unless `near_factor_threads` asks for more, and with the
default every part of the solve still gives the same bits at any thread
budget (the tests below check it).  That is a property of the present code,
not a promise: a caller that compares bits keeps the thread counts the same.

- Every sum runs in a fixed order.  A native kernel gives each output to
  one task, which sums in layer (or entry) order; a lockstep group solves each
  column as it would be solved alone; BLAS is held to one thread; MUMPS runs
  without its L0 tree parallelism on `near_factor_threads` threads.

- No set or `frozenset` is iterated where its order could reach a number.
  Sets of names, or of cells keyed by a layer's name, iterate in another
  order in every process, because Python randomises string hashes.  The
  only sets iterated hold integers, and they are sorted or used where order
  does not count.
- Evidence:
  - `tests/test_reproducibility.py` solves both contracts in three
    processes, each with its own hash seed and thread budget, and compares
    every float they return, in the order returned.
  - `tests/test_native_sheet_operator.py` and `tests/test_sheet_peec.py`
    check thread budgets, operator batches and lockstep cuts.
  - `tests/test_sheet_peec_mumps_near.py` repeats a solve with the MUMPS
    factor on three threads and compares the bits.
  - Kicad_PowerOpt's LM5156 hot loop (eight excitations, `near`) gave the
    same results with lockstep widths 1, 2 and 8, at thread budgets 1 and 8,
    and in separate processes.

**Not the same bits, but the same answer within the solver's tolerance:**

- Another CPU, another BLAS (OpenBLAS picks kernels by instruction set;
  Apple's Accelerate solves several right-hand sides differently, see
  [SHEET_PEEC.md](SHEET_PEEC.md)), or other NumPy, SciPy or pocketfft
  versions.  The core's operator gave the same bits built with GCC 12.5.0 and
  with AOCC 5.2.0 on one machine (`docs/SHEET_OPERATOR_BATCHING_RESULTS.json`),
  but no compiler is promised.
- Another thread budget or another `near_factor_threads`.
- MUMPS against SuperLU as the near factor (`near_factorization`): the same
  preconditioner factored two ways, so the GMRES iterates differ.
- The CUDA path against the CPU.
- A port-impedance frequency solved with a warm start (the default) depends
  on which frequencies were solved before it on the same mesh.  The
  frequencies of one problem are solved in ascending order, so one problem
  always gives the same bits.  The same frequency asked for in another set
  of frequencies agrees within the tolerance, not to the bit (measured
  against cold starts in [SHEET_PEEC.md](SHEET_PEEC.md)).

**For callers.** A sum taken over a set of named nodes moves in its last bits
from one process to the next.  An example is a pad's mean voltage over the
cells of its set.  Sorting the nodes, or the values, before summing makes the
sum the same in every process.

## Implementation status

| Component | Module | Status |
|---|---|---|
| Current-field schema mapping | `sheet_peec/current_field_contract.py` | implemented |
| Sheet PEEC CUDA execution | `sheet_peec/sheet_cuda.py` | implemented; no CPU fallback |
| Physical CUDA PyPEEC path | `voxel_peec/cuda_pypeec.py` | layer-agnostic executor; mapping-dependent |
| Port-impedance schema mapping | `sheet_peec/port_impedance_contract.py` | implemented; CPU only |

## Current-field integration contract (used by plane_opt)

The full current-field solve consumes
`current-field-problem/v1` through
`current_field_contract.solve_current_field_problem()`. The mapping contains each
layer's own thickness and center Z, conductor cells, complex terminals,
scenario frequency, and explicit vertical segments; this package does not
import `plane_opt`.

Routed copper whose pads are ports goes through `port-impedance-problem/v1`
and `port_impedance_contract.solve_port_impedance_problem()`: the same
conductor statement, pads with their part and net, each port's reference pad,
and the frequencies.  It returns one pad-port impedance matrix per frequency
for `electrical.port_network.BoardNetwork` (docs/SHEET_PEEC.md).

## Matrix-free MPIR-FEM boundary

The electrical FEM path covers real, symmetric DC conduction on layered Q1 PCB
meshes and the complex 2D scalar `E_z` reduction of frequency-domain Maxwell.
The latter includes conductivity, displacement current, dielectric loss, eddy
currents, and skin effect. Outer reliable residuals and updates use host
float64/complex128; inner Jacobi-PCG or Jacobi-GMRES corrections and most
operator actions use float32/complex64. Element and via contributions are
applied directly, so there is no assembled global matrix.

It complements the sheet PEEC rather than replacing it: the full-wave front
end is scalar and z-invariant, not an arbitrary 3D vector PCB model. The algorithm, validation limits, CUDA runtime, and intended Tenstorrent
Blackhole port boundary are specified in
[MATRIX_FREE_MPIR_FEM.md](MATRIX_FREE_MPIR_FEM.md).
