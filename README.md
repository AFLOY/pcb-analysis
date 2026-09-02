# Accelerated electrical, thermal, EMC, and coupled PCB analysis

[![CI](https://github.com/AFLOY/pcb-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/AFLOY/pcb-analysis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Accelerated PEEC and matrix-free FEM solvers for electrical and thermal PCB
analysis, radiated-emission evaluation for EMC, and PDN optimization.

The repository holds two solver families under the `electrical` package, one
under the `thermal` package, one evaluation front end under `emc`, and the
coupled scenarios that chain them under `multiphysics`. `electrical.dice_peec` provides exact incremental
delta-scoring for local PCB reroutes, a 2.5D multilayer interaction operator, a
thin-sheet PEEC field solver, and an adaptive runtime controller for
topology-optimization loops. `electrical.matrix_free_mpir_fem` provides a
matrix-free Q1 finite-element solver for DC conduction and 2D frequency-domain
Maxwell fields, driven by mixed-precision iterative refinement (MPIR) on NumPy
or CuPy. `thermal.matrix_free_mpir_fem` solves steady heat conduction through
the whole board stack with the same MPIR solver and runtimes, and takes the
Joule loss of an electrical solve as its heat load.
`emc.tiled_dipole_superposition` reads the current distribution of either
electrical solve as Hertzian dipoles and evaluates near-field scans, far-field
patterns, radiated power, and margins to CISPR 32 and FCC Part 15 limits.
`multiphysics.staggered_coupling` runs the scenarios that couple them: the
electro-thermal fixed point with temperature-dependent copper, and the emission
of a cold or thermally converged current distribution.

## Features

- **Exact sparse delta scoring** — `O(s)` + `O(s²)` incremental evaluation instead of `O(N log N)` full FFT per candidate
- **2.5D multilayer operator** — planar FFT with interlayer kernel matrix and frequency-dependent via model
- **CUDA acceleration** — CuPy/cuFFT backend with RawKernel batched scoring
- **Adaptive controller** — memory-bounded, error-driven fidelity cascade with automatic OOM recovery
- **Low-memory mode** — near/far field splitting for 4–8 GB VRAM devices
- **Matrix-free MPIR-FEM** — Q1 PCB conduction FEM with an FP32 inner PCG and
  FP64 outer reliable updates
- **Frequency-domain Maxwell** — 2D scalar-polarisation wave, dielectric-loss,
  eddy-current, and skin-effect solve with complex64 inner GMRES and complex128
  reliable updates
- **CUDA matrix-free FEM** — fused, node-owned Q1 gather kernel plus batched
  complex64 GMRES; no assembled matrix and no global atomics
- **Accelerator boundary** — NumPy/CuPy runtimes with a narrow low-precision
  interface that remains suitable for the later Tenstorrent port
- **Thermal MPIR-FEM** — matrix-free trilinear hexahedral heat conduction
  through the copper/laminate stack with anisotropic laminate conductivity,
  convective faces, fixed-temperature nodes, and a closed heat budget
- **Electrothermal coupling** — per-element and per-via Joule loss from the DC
  solve mapped onto the thermal stack without interpolation
- **Radiated emissions** — exact Hertzian-dipole superposition of the solved
  currents for near-field scans and far-field patterns, radiated power,
  dipole-moment proxies, PEC ground-plane images, and margins to CISPR 32 /
  FCC Part 15 limit lines, tiled on NumPy or CuPy
- **Coupled scenarios** — staggered electro-thermal iteration with `σ(T)` and
  via `R(T)`, warm starts and Aitken relaxation; electro-emission and
  electro-thermal-emission chains; one `run_scenario` entry point

## Installation

### From PyPI

```bash
pip install pcb-analysis
# with CuPy and PyPEEC
pip install 'pcb-analysis[cuda]'
```

Each release is uploaded to [PyPI](https://pypi.org/project/pcb-analysis/)
by the `Release` workflow with Trusted Publishing, and the same wheel, source
distribution, and SHA-256 checksums are attached to the GitHub release. To
install a specific release without PyPI, point pip at the release asset or
at the tag:

```bash
pip install https://github.com/AFLOY/pcb-analysis/releases/download/v0.6.0/pcb_analysis-0.6.0-py3-none-any.whl
pip install 'git+https://github.com/AFLOY/pcb-analysis.git@v0.6.0'
```

The release tag has to match the version in `pyproject.toml`.

### From a checkout, CPU only

```bash
pip install -e .
```

### With CUDA support

```bash
pip install -e '.[cuda]'
```

> **Note**: Requires an NVIDIA driver and CUDA 13.x. Verify with `nvidia-smi`.
> The package is installed as `electrical`; the former top-level
> `peec_fastopt` package now lives at `electrical.dice_peec`, and the thermal
> and EMC front ends and the coupled scenarios are the separate top-level
> packages `thermal`, `emc`, and `multiphysics`. Re-run the editable install
> after pulling these changes so the new packages are importable and the old
> path is not left on `sys.path`.

## Quick start

```python
from electrical.dice_peec import (
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

### Matrix-free PCB conduction

```python
import numpy as np
from electrical.matrix_free_mpir_fem import (
    CurrentTerminal,
    LayeredPCBMesh,
    PCBConductionProblem,
    solve_pcb_dc,
)

mesh = LayeredPCBMesh(
    element_active=np.ones((1, 1, 32), dtype=bool),
    layer_thickness_m=(35e-6,),
    pitch_x_m=0.2e-3,
    pitch_y_m=0.2e-3,
)
problem = PCBConductionProblem(
    mesh=mesh,
    terminals=(
        CurrentTerminal(((0, 0, 0), (0, 1, 0)), 1.0, "source"),
        CurrentTerminal(((0, 0, 32), (0, 1, 32)), -1.0, "sink"),
    ),
    reference_node=(0, 0, 32),
)
solution = solve_pcb_dc(problem)
assert solution.solve.converged
```

### Frequency-domain Maxwell

The frequency-domain front end solves the `E_z` Maxwell reduction for
z-invariant geometries and reports electric field, magnetic field,
eddy-current density, conductor loss, and dielectric loss. Dirichlet nodes
are given by a boolean mask over the node grid plus the field value held at
each masked node.

```python
import numpy as np
from electrical.matrix_free_mpir_fem import (
    ScalarMaxwellMesh2D,
    ScalarMaxwellProblem,
    solve_scalar_maxwell,
)

mesh = ScalarMaxwellMesh2D(
    (8, 64),                # elements: rows x columns
    pitch_x_m=0.1e-3,
    pitch_y_m=0.1e-3,
    relative_permittivity=4.0,
    dielectric_loss_tangent=0.01,
)
mask = np.zeros(mesh.node_shape, dtype=bool)
mask[:, 0] = mask[:, -1] = True          # drive the left edge, ground the right
value = np.zeros(mesh.node_shape, dtype=np.complex128)
value[:, 0] = 1.0
problem = ScalarMaxwellProblem(mesh, 1e9, mask, value)

solution = solve_scalar_maxwell(problem)          # NumPy complex64 inner solve
assert solution.solve.converged
loss_w_per_m = solution.dielectric_loss_w_per_m
```

### CUDA full-wave solve

```python
solution = solve_scalar_maxwell(
    problem,
    backend="cuda",  # or "auto" to fall back to NumPy when no GPU is visible
    device_id=0,
)
assert solution.solve.low_runtime == "cupy-complex64"
```

The complex128 reliable residual stays on the CPU. The correction RHS crosses
to CUDA once per outer iteration, while the complex64 Krylov basis,
preconditioner, material coefficients, and repeated Q1 actions remain on the
GPU. On the repository's GTX 1650 audit, the fused operator is 8.8–40.4×
faster than the NumPy complex64 action for 4,225–66,049 unknowns. The complete
solve is slower at 4,369 unknowns (0.85×) but 1.99× faster at 16,705 unknowns,
so CUDA should not be selected solely for tiny meshes.

### Steady thermal conduction

The thermal front end meshes the whole stack as slabs of hexahedral Q1
elements. Copper, laminate, and vias are element conductivities; laminates may
have a different through-plane value. Heat enters per element or per node and
leaves through convective faces and fixed-temperature nodes. The solution
reports nodal temperature, element heat flux, and a heat budget whose error is
the solver residual.

```python
import numpy as np
from thermal.matrix_free_mpir_fem import (
    ConvectionBoundary,
    HeatSource,
    LayeredThermalMesh,
    ThermalConductionProblem,
    solve_thermal_conduction,
)

# 35 µm copper / 1.5 mm FR-4 / 35 µm copper, 50 mm x 50 mm on a 0.5 mm grid.
mesh = LayeredThermalMesh(
    slab_thickness_m=(35e-6, 1.5e-3, 35e-6),
    pitch_x_m=0.5e-3,
    pitch_y_m=0.5e-3,
    conductivity_w_per_m_k=(385.0, 0.8, 385.0),
    through_plane_conductivity_w_per_m_k=(385.0, 0.3, 385.0),
    element_shape=(100, 100),
)
ambient = 298.15
problem = ThermalConductionProblem(
    mesh,
    convection=(
        ConvectionBoundary("top", 10.0, ambient),
        ConvectionBoundary("bottom", 10.0, ambient),
    ),
    heat_sources=(HeatSource(((3, 50, 50), (3, 50, 51)), 0.5, "regulator"),),
)
solution = solve_thermal_conduction(problem, initial_temperature_k=ambient)
assert solution.solve.converged
hottest_k = solution.max_temperature_k
assert abs(solution.heat_balance_error_w) < 1e-9
```

The default preconditioner adds a patch-constant coarse correction to Jacobi
scaling. A cooled copper plate is stiff in-plane and weakly coupled to the air,
so plain Jacobi PCG needs hundreds of inner iterations per decade on such a
stack; the two-level variant needs tens. Pass `preconditioner="jacobi"` to
compare, and `coarse_block_nodes` to set the patch width.

### Electrothermal coupling

`element_joule_heat_w` places the per-element copper loss of a
`PCBConductionSolution` onto the thermal slabs that hold each electrical layer,
and `via_joule_heat_sources` turns each via's loss into nodal heat at its
endpoints. Both meshes must share the in-plane element grid.

```python
from thermal.matrix_free_mpir_fem import element_joule_heat_w, via_joule_heat_sources

electrical = solve_pcb_dc(pcb_problem)          # copper on layers 0 and 1
layer_slabs = (0, 2)                            # bottom and top copper slabs
thermal_problem = ThermalConductionProblem(
    mesh,
    convection=(ConvectionBoundary("top", 10.0, ambient),),
    element_heat_w=element_joule_heat_w(electrical, mesh, layer_slabs),
    heat_sources=via_joule_heat_sources(pcb_problem, electrical, mesh, layer_slabs),
)
solution = solve_thermal_conduction(thermal_problem, backend="auto")
```

The thermal operator runs its FP32 inner PCG on CuPy with `backend="cuda"`; the
FP64 reliable update stays on the host exactly as for the electrical solves.

### Radiated emissions

Every current element of a solved board is a Hertzian dipole. The EMC front end
sums their exact fields for a near-field scan and their far-zone terms for the
pattern at a test-site distance, then compares the maximum with a limit line.

```python
import numpy as np
from emc.tiled_dipole_superposition import (
    CISPR32_CLASS_B,
    dipole_moments,
    dipoles_from_pcb_dc,
    emission_margin,
    evaluate_fields,
    far_field_pattern,
    scan_plane,
)

electrical = solve_pcb_dc(pcb_problem)            # two copper layers
dipoles = dipoles_from_pcb_dc(
    pcb_problem, electrical, layer_height_m=(0.0, 1.6e-3), close_terminals=True
)
frequency_hz = 100e6                              # DC pattern used as a phasor

probe = scan_plane(np.linspace(0, 0.05, 50), np.linspace(0, 0.02, 20), z_m=6.6e-3)
near = evaluate_fields(dipoles, probe, frequency_hz, backend="auto")
h_max_a_per_m = near.magnetic_magnitude_a_per_m.max()

pattern = far_field_pattern(dipoles, frequency_hz, distance_m=10.0)
margin = emission_margin(
    pattern.max_polarised_field_v_per_m, frequency_hz, CISPR32_CLASS_B, distance_m=10.0
)
print(pattern.radiated_power_w, margin.predicted_dbuv_per_m, margin.margin_db)

moments = dipole_moments(dipoles, frequency_hz)   # |P| ≈ 0 once the loop closes
```

`dipole_moments` reports the net electric moment: a non-zero value for a board
whose terminal currents balance means the return path through the component is
not in the model, and `close_terminals=True` adds it as a straight element.
The magnetic near field and the far field are the robust outputs; the electric
near field of a current-only description is sensitive to how continuously the
element chain closes. `dipoles_from_sheet_peec` does the same for a
frequency-resolved sheet-PEEC solve, without the quasi-static assumption.

### Coupled scenarios

Scenario dataclasses name what is coupled to what; `run_scenario` dispatches on
the type. The electro-thermal chain iterates the two conduction solves until
the copper conductivity `σ(T) = σ_ref / (1 + α (T − T_ref))` is self-consistent
with the temperature it produces, warm-starting both solves and relaxing the
temperature update with Aitken's Δ² estimate.

```python
from multiphysics.staggered_coupling import (
    ElectroThermalEmissionScenario,
    ElectroThermalScenario,
    EmissionScenario,
    run_scenario,
)

coupled_scenario = ElectroThermalScenario(
    pcb_problem,                    # PCBConductionProblem, conductivity at 293.15 K
    thermal_mesh,                   # LayeredThermalMesh on the same (rows, cols)
    layer_slabs=(0, 2),             # thermal slab holding each copper layer
    convection=(ConvectionBoundary("top", 10.0, 298.15), ConvectionBoundary("bottom", 10.0, 298.15)),
)
coupled = run_scenario(coupled_scenario)
print(coupled.converged, coupled.iterations, coupled.loss_increase_ratio)

chained = run_scenario(
    ElectroThermalEmissionScenario(
        coupled_scenario,
        layer_height_m=(0.0, 1.6e-3),
        emission=EmissionScenario((30e6, 100e6, 300e6)),   # CISPR 32 Class B at 10 m
    )
)
print(chained.emission.margin_db, chained.heating_shift_db)
```

`ElectricalScenario`, `ThermalScenario`, `ElectroEmissionScenario`, and
`SheetPeecEmissionScenario` (one sheet-PEEC solve per frequency) complete the
set; `run_scenarios` runs a list. See
[MULTIPHYSICS_SCENARIOS.md](docs/MULTIPHYSICS_SCENARIOS.md).

## Project structure

```text
src/
  electrical/                         Analysis target
    dice_peec/                         PEEC + DICE/2.5D FFT acceleration
    matrix_free_mpir_fem/              FEM + matrix-free/MPIR acceleration
  thermal/                            Analysis target
    matrix_free_mpir_fem/              Heat-conduction FEM + matrix-free/MPIR acceleration
  emc/                                Analysis target
    tiled_dipole_superposition/        Radiated emission by dipole superposition, tiled NumPy/CuPy
  multiphysics/                       Coupled analysis
    staggered_coupling/                Partitioned fixed-point coupling with warm starts and Aitken relaxation
tests/                                 Test suite (pytest)
docs/                                  Design documents, validation reports, raw results
examples/                              Demo scripts and benchmarks
experiments/                           Repeatable accuracy and timing audits
requirements.txt                       Editable install with CUDA/test/build tooling
```

Future physics should follow the same rule: the first level names the analysis
target and the second names the method plus its acceleration strategy.

## Electrical solvers

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

`skin_filaments` extends `sheet_peec` to conductors thick against the skin
depth by cutting them into graded filaments joined through the thickness, so
the solve itself decides how the current divides between the faces and the
interior.

`matrix_free_mpir_fem` solves real SPD DC conduction on layered Q1 PCB meshes
and complex 2D scalar-polarised frequency-domain Maxwell fields. Its global
matrix is never assembled. Low-precision inner corrections contain most
operator applications, while high-precision host residuals determine final
convergence. Scope and the Blackhole port boundary are documented in
[docs/MATRIX_FREE_MPIR_FEM.md](docs/MATRIX_FREE_MPIR_FEM.md).

## Experiments and audits

The scripts in `experiments/` regenerate the numbers quoted in `docs/`. They
run from the repository root because the FP32 audit is also imported by the
test suite. Results default to the ignored `benchmark-results/` directory;
pass `--output` (or `--json` for the FP32 audit) to write elsewhere.

```bash
# What single precision costs the sheet-PEEC solver (host stage, then CUDA)
python experiments/fp32_accuracy.py --cells 128 --json benchmark-results/fp32.json

# Maxwell accuracy against closed-form 1D solutions, plus CPU timing
python experiments/maxwell_small_benchmark.py

# Fused CUDA operator throughput and complete-solve timing (needs a GPU)
python experiments/maxwell_cuda_benchmark.py --operator-sides 64 128 256

# Sheet PEEC versus matrix-free FEM on matched DC strips and a skin-effect slab.
# Both methods solve both scenarios; the PEEC skin bar takes ~90 s at 64 cells.
python experiments/peec_fem_comparison.py --no-cuda --peec-skin-lengths 16 32
```

The skin-effect scenario is the same 0.5 mm copper slab at 1 MHz for both
methods. FEM solves the field through the slab's thickness. Sheet PEEC has no
infinite slab, so it solves a bar of graded filaments and reads the AC/DC
resistance ratio from the filament current division at the middle of the bar,
where it is slab-like. The closed-form slab impedance is the shared reference.

## Development

```bash
# CPU-only, as CI runs it
pip install -e '.[test]'

# Or: the ignored local environment with CUDA, test, and build tooling
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Run tests from the repository root
python -m pytest tests/ -v
```

CI runs the suite on Python 3.11–3.13, then builds a wheel and source
distribution with the oldest supported setuptools and imports both
subpackages from a clean environment, so a module missing from the
distribution fails the build rather than a user's install.

## Documentation

- [Architecture & Design](docs/DESIGN.md)
- [CUDA Backend Handoff](docs/CUDA_HANDOFF.md)
- [Matrix-free MPIR-FEM](docs/MATRIX_FREE_MPIR_FEM.md)
- [Thermal MPIR-FEM](docs/THERMAL_MPIR_FEM.md)
- [EMC radiated emissions](docs/EMC_DIPOLE_SUPERPOSITION.md)
- [Coupled analysis scenarios](docs/MULTIPHYSICS_SCENARIOS.md)
- [Sheet PEEC](docs/SHEET_PEEC.md) and [Sheet PEEC CUDA results](docs/SHEET_CUDA_RESULTS.md)
- [Requirements](docs/REQUIREMENTS.md)
- [Sheet PEEC / matrix-free FEM comparison](docs/PEEC_FEM_COMPARISON_REPORT.html)
  ([raw data](docs/PEEC_FEM_COMPARISON_RESULTS.json))
- [Maxwell accuracy and CPU timing report](docs/MAXWELL_VALIDATION_REPORT.html)
  ([raw data](docs/MAXWELL_SMALL_RESULTS.json))
- [Maxwell CUDA benchmark data](docs/MAXWELL_CUDA_RESULTS.json) and
  [CUDA scaling data](docs/MAXWELL_CUDA_SCALE_RESULTS.json)
- [Benchmark Results](docs/RESULTS.md)

## License

[MIT](LICENSE)
