# Delta-Cascade PEEC prototype

This prototype tests one component of an optimization-oriented PEEC workflow:
exact incremental evaluation of a quadratic electromagnetic interaction metric.

For a fixed grid and interaction operator `K`, a candidate changes the current
occupancy vector from `x` to `x + d`. Instead of recomputing

```text
E(x + d) = (x + d)^T K (x + d)
```

with a new FFT for every candidate, the cached field `phi = K x` is used:

```text
Delta E = 2 d^T phi + d^T K d
```

When a reroute changes only `s` cells, the first term costs `O(s)` and the
second `O(s^2)`. This is exact for the quadratic metric and avoids an
`O(N log N)` FFT for every candidate.

The code is deliberately backend-neutral. `numpy.fft` can be replaced by
CuPy/cuFFT without changing the delta-scoring interface.

## Run

```sh
python3 peec_fastopt/benchmark.py --grid 256 --candidates 1000
```

Low-memory near/far parameters can be calibrated against exact interaction
energies with:

```sh
python3 -m peec_fastopt.autotune_lowmem --grid 256 --candidates 100 --summary
```

Inspect the dynamic plans for a memory-constrained device without CUDA:

```sh
python3 -m peec_fastopt.controller_demo --vram-gb 4 --platform windows
python3 -m peec_fastopt.controller_demo --vram-gb 8 --platform linux
```

On a CUDA machine with the matching CuPy package installed, collect real VRAM
and cuFFT calibration data:

```sh
python3 -m peec_fastopt.cuda_calibrate \
  --grid 1024 --layers 4 --unknowns 2000000
```

The controller automatically chooses tile and batch sizes from live free VRAM,
replans with a smaller tile after OOM, changes BiCGSTAB(2) to restarted GMRES
after stagnation, promotes storage to complex128 when a precision replay shows
a gap, and tightens the near radius/shortlist when an accuracy audit finds a
miss. See `CUDA_HANDOFF.md` for the backend contract and acceptance gates.

The model is a scalar magnetoquasistatic interaction proxy, not a complete
PEEC field solver. It validates the incremental-scoring identity and measures
the optimization-loop speedup that the full implementation can exploit.

`lowmem_peec.py` is a second scalar validation model. It tests exact near-field
plus block-multipole far-field splitting and is used to choose a low-memory
fidelity cascade from measured ranking error.

`controller.py` is the backend-neutral control plane. `cupy_calibrator.py`
measures real CUDA allocation and FFT behavior, but is intentionally not a
complete PEEC-MNA executor. The physical CUDA operator must report the same
`ExecutionReport` fields when it is connected.
