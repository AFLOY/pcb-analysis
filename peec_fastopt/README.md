# PEEC-CUDA and Delta-Cascade PEEC

The package now contains a production integration path for the single-layer,
multi-terminal PyPEEC problems used by `plane_opt`.  The `cuda_peec` backend
keeps PyPEEC's physical formulation and iterative solver but selects its CuPy
/ cuFFT operator, applies a device-memory reserve, and returns measured runtime
metadata through the existing `SolveResult` contract.  Sparse quadratic delta
scores also have a batched CuPy/RawKernel implementation.

Install the CPU development package with:

```sh
python3 -m venv .venv
.venv/bin/pip install -e .
```

After the NVIDIA driver is healthy, install the CUDA extras and both sibling
projects.  The CUDA 13 wheel uses NumPy 2 in the CUDA environment, while the
base package remains compatible with NumPy 1.26 for CPU-only workflows:

```sh
.venv/bin/pip install -e '.[cuda]' -e ../plane_opt
```

Verify the device with `nvidia-smi`.  If Linux has loaded the NVIDIA modules
but the `/dev/nvidia*` nodes are absent, run `sudo nvidia-modprobe -u -c=0` and
probe again.

Select `"backend": "cuda_peec"` in `current_field_solver`; CPU fallback is
disabled by default and is enabled only with `"fallback_backend": "pypeec"`.

Run the real-board acceptance benchmark with:

```sh
.venv/bin/peec-cuda-benchmark \
  --extract ../plane_opt/topology_variants/pgnd_board_left_expanded_0p4/analysis/extract.json \
  --candidate ../plane_opt/topology_variants/pgnd_board_left_expanded_0p4/analysis/search.json \
  --config ../plane_opt/topology_refinement_config.json \
  --output benchmark-results/cuda-vs-cpu.json
```

The command requires at least 2x median end-to-end speedup, metrics within 1%,
current closure within 1e-6 A, stable rankings, and no case slower by more than
10%.  It exits nonzero when any gate fails.

## Multilayer 2.5D scoring (DICE scaffolding)

While `plane_opt` implements multilayer routing and via connectivity, this
package provides the document-aligned 2.5D interaction model:

- `Stackup` — layer names and z centers
- `FFTInteraction25D` — planar FFT with interlayer kernel matrix
- `SparseDeltaML` / `MultilayerDeltaScorer` — exact sparse delta energy
- `ViaSpec` / `ViaSet` — frequency-dependent lumped vias
- `layout_ops` — `add_segment` / `remove_segment` / `add_via` / `remove_via`
- `lowmem_25d` — near/far cascade on multilayer occupancy

```python
from peec_fastopt import (
    Stackup,
    FFTInteraction25D,
    MultilayerDeltaScorer,
    CandidateEdit,
    SegmentOp,
    ViaOp,
    compile_candidate,
)
import numpy as np

stack = Stackup.dual_sided(board_thickness_mm=1.6)
shape = (64, 64)
base = np.zeros((stack.n_layers, *shape))
base[0, 10:20, 10:30] = 1.0
op = FFTInteraction25D(shape, stack, cell_size_m=0.2e-3)
scorer = MultilayerDeltaScorer(op, base, frequency_hz=3e5)
edit = CandidateEdit(
    segments=[SegmentOp("add", "B.Cu", ((12, 12), (12, 13), (12, 14)))],
    vias=[ViaOp("add", 12, 14, "F.Cu", "B.Cu", important=True)],
)
compiled = compile_candidate(edit, stack)
score = scorer.energy(compiled.occupancy_delta, vias=compiled.vias)
```

Single-layer physical CUDA solves remain the production path through
`cuda_peec` until plane_opt supplies multilayer PyPEEC mappings.

## Research prototype

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
