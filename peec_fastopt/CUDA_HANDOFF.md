# CUDA integration status and handoff

## Implemented boundary

The production MVP is connected end to end.  `CudaPyPeecExecutor` runs the
physical PyPEEC 5.8 solve with CuPy/cuFFT, reserves configurable device memory,
caches geometry-keyed voxelizations, records timing and memory telemetry, and
fails explicitly on CUDA initialization or allocation errors.  It resets
PyPEEC's process-global FFT selector before each solve so a preceding CPU solve
cannot silently force the CUDA request back to SciPy.

`plane_opt` exposes this executor as the `cuda_peec` current-field backend and
maps CPU and CUDA solutions through one result-conversion function.  CPU
fallback is off by default; setting `fallback_backend` to `pypeec` enables it
and records the requested backend and reason in the result metadata.

The package also contains a CuPy/cuFFT plus RawKernel implementation of exact
batched sparse-delta quadratic scoring.  The adaptive controller and scalar
delta experiments remain available for later candidate-screening work.

## Environment and first run

Create an isolated environment and install both local projects:

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[cuda]' -e ../plane_opt
nvidia-smi
.venv/bin/python -c \
  'import cupy as cp; print(cp.cuda.runtime.getDeviceCount(), cp.show_config())'
```

On Linux, if the driver modules are loaded but `/dev/nvidia*` is missing, create
the device nodes and retry the probe:

```sh
sudo nvidia-modprobe -u -c=0
```

Run the allocation/FFT smoke calibration with:

```sh
.venv/bin/peec-cuda-calibrate \
  --grid 1024 --layers 4 --unknowns 2000000 --repeats 20
```

## Real-board acceptance benchmark

```sh
.venv/bin/peec-cuda-benchmark \
  --extract ../plane_opt/topology_variants/pgnd_board_left_expanded_0p4/analysis/extract.json \
  --candidate ../plane_opt/topology_variants/pgnd_board_left_expanded_0p4/analysis/search.json \
  --config ../plane_opt/topology_refinement_config.json \
  --warmups 1 --repeats 3 \
  --output benchmark-results/cuda-vs-cpu.json
```

The command exits nonzero unless the global median speedup is at least 2x, no
individual case regresses by more than 10%, physical metrics remain within 1%,
current closure stays within 1e-6 A, and meaningful metric rankings agree.

The validated GTX 1650 run completed all nine scenarios at 624.697 ms CPU
median versus 309.441 ms CUDA median, or 2.019x.  Its maximum physical-metric
difference was 1.60e-11 and every gate passed.

## Known limits and next optimization targets

- PyPEEC 5.8 controls the physical solver dtype, so `complex64` is accepted as
  requested policy metadata but the effective solve remains complex128.
- The current `plane_opt` physical mapping is single-layer; multilayer/via
  coupling requires extending that mapping and its validation fixtures.
- Memory telemetry includes the CuPy pool and observed free-memory delta, but
  not a guaranteed complete cuFFT workspace peak.  It therefore reports
  `memory_measurement_complete=false`.
- The voxel cache is process-local and geometry-keyed.  Changing geometry
  correctly causes a remesh; restarting the process starts with a cold cache.
- Small VIN/VOUT cases show less benefit than larger SW/PGND cases.  The next
  tuning work should focus on cuFFT plan reuse, lower launch overhead, and
  measured mixed precision before changing numerical acceptance thresholds.
