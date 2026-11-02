# Sheet-PEEC CUDA acceptance

## Scope

This records execution of the multilayer, per-layer-thickness sheet solver. It
does not replace KiCad refill, DRC, unconnected, real-fill physics, or the full
adopted-pipeline gates.

CUDA execution was performed outside the sandbox on 2026-07-28. A CUDA skip or
CPU fallback is not counted as acceptance.

## Runtime

| Item | Value |
|---|---|
| Device | NVIDIA GeForce GTX 1650, compute capability 7.5 |
| Device memory | 4096 MiB |
| Driver | 610.43.02 (`driverGetVersion=13030`) |
| CUDA runtime | 13.2 (`runtimeGetVersion=13020`) |
| CuPy | 14.1.1 |
| Resolved backend | `cupy-sheet-peec:0` |
| Fallback | false |

## Nonuniform three-layer contract fixture

Input: 35 / 70 / 105 um copper, 123456 Hz, two explicit three-layer vertical
connections. Filament counts were 1 / 1 / 3.

| Check | Result |
|---|---|
| CPU converged | true |
| CUDA converged | true |
| CUDA original-system residual | 2.91e-14 |
| CUDA current closure | 1.25e-14 A |
| Maximum voltage phasor difference | 1.48e-14 V |
| Maximum density phasor difference | 2.60e-10 A/mm2 |
| Maximum relative metric difference | 1.50e-11 |
| CUDA warm total | 792.8 ms |
| CPU total | 287.2 ms |
| CUDA device-pool peak | 111616 bytes |

The fixture is too small to benefit from CUDA; it is a correctness check.
`tests/test_sheet_cuda_device.py` is the repeatable real-device acceptance
test.

## Power-module full-domain smoke

Input: `power_module`, `pgnd_off_return_2a`, 300 kHz, 0.2 mm grid, F.Cu/B.Cu
35 um, 69040 branches, 35268 physical conductor cells. Fifteen undriven nodes
were excluded together with their branches.

| Check | CPU | CUDA |
|---|---:|---:|
| Converged | true | true |
| Original-system relative residual | 6.86e-10 | 3.38e-12 |
| Current closure | 2.00e-11 A | 8.72e-14 A |
| Voltage span | 0.0034247345467 V | 0.0034247345403 V |
| Iterations | 764 | 960 |
| Solve time | 109.18 s | 107.92 s |

CUDA used a 327183872-byte device memory-pool peak; prepared operator kernels
were 5651968 bytes. The voltage-span relative difference was 1.85e-9.

This run proves real-device execution and numerical agreement, not a useful
speedup on this GPU. Operator and sparse-factor reuse across cases, CUDA Graph
capture, or a custom Krylov implementation are still needed before claiming a
performance advantage.

## Port impedance at a placement candidate's size (`SHEET_CUDA_PORT_T4_RESULTS.json`)

The 30-port four-layer fixture of `SHEET_PEEC.md` (28,195 cells, 81,152
unknowns in the near LU, 1/10/30/100 MHz, tolerance 1e-6, warm starts) on a
Colab Tesla T4 (CuPy 14.0.1, two host CPUs, `_pcbcore_cuda` built for sm_75).
The suite passed there (562 passed; the CUDA tests all ran), and the CUDA
port matrix agrees with the CPU's to 1e-7 of its largest entry
(`tests/test_sheet_cuda_device.py`).  The time does not:

| Run | Result |
|---|---|
| CUDA, four frequencies | stopped after 2400 s, unfinished |
| CPU on the same host, thread budget 2 | 1508 s (125 / 238 / 364 / 781 s; 37 / 67 / 101 / 271 iterations) |
| CUDA, 3 ports at 1 MHz | 5.0 s preparation, then 40 s per port |
| one near LU application on the device / one operator application | 486 ms / 5.4 ms |
| near LU applied on the host instead (probe), 3 ports | 8 s per port at 1 MHz, 29 s at 30 MHz |

The device triangular solves of the near LU take nearly all of it: one
application costs 90 times the operator's.  Applying the same SuperLU
factors on the host and copying the vectors was five times faster per port,
and still slower than the CPU's lockstep solve of all 30 ports (the CPU path
took 171 s on eight threads, `PORT_IMPEDANCE_SCALE_RESULTS.json`).
**Decision:** the CUDA port path stays as a working path, not a faster one;
the host-applied LU is measured, not adopted (one fixture, one device; the
probe is not kept).  Solving the ports in lockstep on the device, or a
preconditioner without triangular solves, would be the next candidates.

## Solver boundary

The CUDA path keeps kernel spectra, batched FFTs, sparse incidence products,
Krylov vectors, and preconditioner triangular solves on the device. CuPy 14's
`splu()` creates the sparse LU factors with SciPy SuperLU on the CPU and then
uploads them; metrics record
`scipy_superlu_factor_cupy_triangular_solve`. No implicit fallback is
implemented.
