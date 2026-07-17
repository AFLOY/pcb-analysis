# CUDA integration handoff

## Current boundary

The optimization control plane is ready for device calibration. It already
owns fidelity selection, memory budgeting, retry policy, solver fallback,
precision escalation, uncertainty-based promotion, and accuracy-audit
feedback. The remaining device-specific work is to connect the real
matrix-free PEEC operator and report measurements through the existing backend
contract.

`CuPyCalibrationBackend` is a hardware probe and representative 2D FFT
microbenchmark. It is not the physical PEEC-MNA solver.

## Backend contract

A CUDA executor implements two methods from `CalibrationBackend`:

```python
def probe() -> HardwareTelemetry: ...
def execute(plan: ExecutionPlan, problem: ProblemProfile) -> ExecutionReport: ...
```

The executor must obey `ExecutionPlan` tile size, candidate/kernel batch,
near-field radius, storage/reduction precision, solver, restart, and tolerance.
It must report actual peak device bytes, elapsed time, iterations, relative
residual, convergence, OOM, stagnation, and float32/float64 replay gap. Set
`memory_complete=True` only when peak memory includes FFT workspaces, kernels,
geometry, preconditioner, Krylov vectors, candidate data, and allocator pools.

## First run on the CUDA machine

1. Install the CuPy wheel matching the installed CUDA major version.
2. Run the CPU/controller tests:

   ```sh
   python3 -m unittest -v \
     peec_fastopt/test_delta_peec.py peec_fastopt/test_controller.py
   ```

3. Collect allocation and FFT calibration:

   ```sh
   python3 -m peec_fastopt.cuda_calibrate \
     --grid 1024 --layers 4 --unknowns 2000000 --repeats 20
   ```

4. Connect the real operator behind `CalibrationBackend`, initially with one
   frequency and one accepted layout. Feed every report to
   `DynamicController.observe` and call `make_plan` again after OOM,
   stagnation, or a precision gap.

## Required measurement matrix

Run each case on both 4 GB and 8 GB targets, and on WDDM if Windows support is
required:

| Axis | Values |
|---|---|
| Grid | 512, 1024, production ROI |
| Layers | 2, 4, maximum supported |
| Stage | near-coarse, near-fine, correction, refined |
| Precision | complex64/float64 reduction; complex128 replay |
| Topology | segment edit, layer change, via/return-path edit |
| Frequency | low, middle, highest optimization frequency |

## Acceptance gates

- No unrecovered OOM: one automatic smaller-tile retry must produce a plan
  below the live safe budget; otherwise fail clearly and reduce the ROI.
- Complete measured peak VRAM remains below the controller budget with at least
  10% of total VRAM or the platform reserve still free, whichever is larger.
- Accepted candidates meet the requested linear residual; stagnation causes a
  GMRES retry rather than silent acceptance.
- The complex64 objective/residual replay gap remains below `1e-5`; otherwise
  the next plan uses complex128 storage.
- Radius-8 screening retains at least 99% of the true promising candidates in
  the promoted set. Any miss immediately tightens the dynamic policy.
- Final ranking and objectives are compared against the existing trusted PEEC
  solver on real boards. Scalar-proxy agreement alone is insufficient.

## CUDA tuning order

After correctness gates pass, tune in this order: cuFFT plan reuse and
workspace policy, kernel-batch size, tile overlap/stream concurrency,
candidate-batch size, CUDA Graph capture, then Krylov kernel fusion. Re-run the
accuracy audits after any change that alters precision, summation order, near
radius, or far-field representation.
