# peec-fastopt

[![CI](https://github.com/AFLOY/peec-cuda/actions/workflows/ci.yml/badge.svg)](https://github.com/AFLOY/peec-cuda/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

CUDA-accelerated PEEC (Partial Element Equivalent Circuit) solver for PCB PDN optimization.

This library provides exact incremental delta-scoring for local PCB reroutes, a 2.5D multilayer interaction operator, and an adaptive runtime controller — all designed for tight integration into topology-optimization loops.

## Features

- **Exact sparse delta scoring** — `O(s)` + `O(s²)` incremental evaluation instead of `O(N log N)` full FFT per candidate
- **2.5D multilayer operator** — planar FFT with interlayer kernel matrix and frequency-dependent via model
- **CUDA acceleration** — CuPy/cuFFT backend with RawKernel batched scoring
- **Adaptive controller** — memory-bounded, error-driven fidelity cascade with automatic OOM recovery
- **Low-memory mode** — near/far field splitting for 4–8 GB VRAM devices

## Installation

### CPU only

```bash
pip install -e .
```

### With CUDA support

```bash
pip install -e '.[cuda]'
```

> **Note**: Requires NVIDIA driver and CUDA 13.x. Verify with `nvidia-smi`.

## Quick start

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

## Project structure

```
peec_fastopt/     Core library (solvers, operators, controller)
tests/            Test suite (pytest)
docs/             Design documents and benchmark results
examples/         Demo scripts and benchmarks
```

## The two solvers

`cuda_pypeec` runs PyPEEC's own three-dimensional voxel solve on CUDA. PyPEEC
owns the physics; this owns the device policy. `pypeec_memory` predicts what a
model costs before it is attempted, which matters once a model spans a board's
height instead of one copper layer: on a two-sided 34.7mm board the prepared
operators go from 4.5 MiB to 541 MiB while the conductor merely doubles.

`sheet_peec` solves the same physics on a mesh built for what a PCB is -- a few
thin sheets at heights the stackup states. Its inductance operator is a
two-dimensional transform per layer pair rather than a three-dimensional one
over the board's height, which brings the same board's operators to 5.6 MiB.
At zero frequency it reproduces an independent resistor network to machine
precision; above it, the transform path matches a dense assembly of the same
operator. See [docs/SHEET_PEEC.md](docs/SHEET_PEEC.md).

`multilayer_peec` is neither. It is a scalar interaction proxy for ranking many
candidate shapes cheaply, and it solves for no current or potential.

## Development

```bash
# Install with test dependencies
pip install -e '.[test]'

# Run tests
pytest tests/ -v
```

## Documentation

- [Architecture & Design](docs/DESIGN.md)
- [CUDA Backend Handoff](docs/CUDA_HANDOFF.md)
- [Benchmark Results](docs/RESULTS.md)

## License

[MIT](LICENSE)
