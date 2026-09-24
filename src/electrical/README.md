# Electrical Analysis (`electrical`)

The `electrical` package contains four complementary electromagnetic and circuit solvers/evaluators tailored for accelerated PCB analysis and topology optimization.

## Subpackage Taxonomy

| Subpackage | Method | Primary Use Case | Hardware Backend |
|---|---|---|---|
| [`electrical.sheet_peec`](sheet_peec/) | 2.5D Sheet PEEC (pFFT & 2D convolution) | Frequency-dependent PCB current distributions, skin effect, partial inductance | CPU (NumPy, OpenMP C++) & CUDA (CuPy) |
| [`electrical.matrix_free_mpir_fem`](matrix_free_mpir_fem/) | Matrix-free Q1 FEM (MPIR) | Multi-layer DC conduction & 2D scalar Maxwell wave/loss fields | CPU (NumPy, C++) & CUDA (CuPy FP32 gather kernel) |
| [`electrical.dice_peec`](dice_peec/) | Delta-incremental cascade scoring | Exact $O(s) + O(s^2)$ candidate re-scoring for routing optimization loops | CPU (NumPy) & CUDA (CuPy / RawKernel) |
| [`electrical.voxel_peec`](voxel_peec/) | 3D Voxel PEEC (PyPEEC) | Thick conductors, volumetric vias, validation reference | CPU & CUDA (CuPy executor) |

---

## Quick Start

### 1. Matrix-free DC Conduction (`electrical.matrix_free_mpir_fem`)

Solves multi-layer PCB DC conduction without assembling a global matrix.

```python
import numpy as np
from electrical.matrix_free_mpir_fem import (
    CurrentTerminal,
    LayeredPCBMesh,
    PCBConductionProblem,
    solve_pcb_dc,
)

# 1-layer active trace on a 0.2 mm grid (32 elements long)
mesh = LayeredPCBMesh(
    element_active=np.ones((1, 1, 32), dtype=bool),
    layer_thickness_m=(35e-6,),
    pitch_x_m=0.2e-3,
    pitch_y_m=0.2e-3,
)
problem = PCBConductionProblem(
    mesh=mesh,
    terminals=(
        CurrentTerminal(nodes=((0, 0, 0), (0, 1, 0)), current_a=1.0, name="source"),
        CurrentTerminal(nodes=((0, 0, 32), (0, 1, 32)), current_a=-1.0, name="sink"),
    ),
    reference_node=(0, 0, 32),
)

solution = solve_pcb_dc(problem, backend="auto")
assert solution.solve.converged
print(f"DC voltage drop: {np.nanmax(solution.potential_v) - np.nanmin(solution.potential_v):.4e} V")
```

### 2. 2D Frequency-Domain Maxwell (`electrical.matrix_free_mpir_fem`)

Solves the scalar-polarised $E_z$ Maxwell reduction for dielectric loss, eddy currents, and skin effect with complex64 inner GMRES and complex128 outer reliable updates.

```python
import numpy as np
from electrical.matrix_free_mpir_fem import (
    ScalarMaxwellMesh2D,
    ScalarMaxwellProblem,
    solve_scalar_maxwell,
)

mesh = ScalarMaxwellMesh2D(
    element_shape=(8, 64),
    pitch_x_m=0.1e-3,
    pitch_y_m=0.1e-3,
    relative_permittivity=4.0,
    dielectric_loss_tangent=0.01,
)
mask = np.zeros(mesh.node_shape, dtype=bool)
mask[:, 0] = mask[:, -1] = True  # Drive left edge, ground right edge
value = np.zeros(mesh.node_shape, dtype=np.complex128)
value[:, 0] = 1.0

problem = ScalarMaxwellProblem(mesh, frequency_hz=1e9, dirichlet_mask=mask, dirichlet_electric_field_v_per_m=value)
solution = solve_scalar_maxwell(problem, backend="auto")
assert solution.solve.converged
print(f"Dielectric loss: {solution.dielectric_loss_w_per_m:.4e} W/m")
```

### 3. Exact Incremental Delta Scoring (`electrical.dice_peec`)

Scores candidate routing edits in $O(s) + O(s^2)$ operations (where $s$ is changed cells) using the exact quadratic identity rather than running a full FFT.

```python
import numpy as np
from electrical.dice_peec import (
    CandidateEdit,
    FFTInteraction25D,
    MultilayerDeltaScorer,
    SegmentOp,
    Stackup,
    compile_candidate,
)

stack = Stackup.dual_sided(board_thickness_mm=1.6)
base_occupancy = np.zeros((stack.n_layers, 64, 64))
base_occupancy[0, 10:20, 10:30] = 1.0

op = FFTInteraction25D(shape=(64, 64), stackup=stack, cell_size_m=0.2e-3)
scorer = MultilayerDeltaScorer(op, base_occupancy, frequency_hz=3e5)

# Candidate edit: add a segment on bottom copper
edit = CandidateEdit(segments=[SegmentOp("add", "B.Cu", ((12, 12), (12, 13), (12, 14)))])
compiled = compile_candidate(edit, stack)
energy_score = scorer.energy(compiled.occupancy_delta, vias=compiled.vias)
print(f"Incremental interaction score (scalar proxy, no unit): {energy_score:.4e}")
```

---

## Documentation

- [Sheet PEEC Contract & Methods](../../docs/SHEET_PEEC.md)
- [Matrix-free MPIR-FEM Contract & Architecture](../../docs/MATRIX_FREE_MPIR_FEM.md)
- [DICE-PEEC Design & Adaptive Controller](../../docs/DESIGN.md)
- [3D Voxel PEEC (PyPEEC) Contract](../../docs/VOXEL_PEEC.md)
