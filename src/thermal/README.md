# Thermal Analysis (`thermal.matrix_free_mpir_fem`)

Accelerated matrix-free Q1 finite-element solver for steady and transient heat conduction in PCB layer stacks and 3D voxel bodies (enclosures, heat sinks).

Driven by Mixed-Precision Iterative Refinement (MPIR) with an FP32 inner PCG and FP64 outer reliable updates. The MPIR solver runtime is imported directly from `electrical.matrix_free_mpir_fem` without duplication.

## Key Features

- **Matrix-free Q1 hexahedral FEM**: No assembled global stiffness matrix, enabling large multi-layer meshes with minimal memory overhead.
- **Two-level preconditioner**: Patch-constant coarse correction combined with Jacobi scaling, resolving ill-conditioned thin copper/FR-4 anisotropic stacks in tens of iterations rather than hundreds.
- **Boundary conditions**: Convective heat transfer, Stefan-Boltzmann surface-to-ambient radiation (Newton linearisation), fixed-temperature Dirichlet nodes, and volumetric/nodal heat sources.
- **Transient conduction**: Backward-Euler time stepping on uniform or geometric time schedules, optionally until steady state.
- **Hardware acceleration**: CPU (NumPy / OpenMP C++ kernel) and CUDA (CuPy FP32 gather kernel) backends.
- **Electrothermal integration**: Maps per-element and per-via Joule heat from `electrical` solves directly onto the thermal stack.

## Quick Start

### 1. Steady-state thermal conduction

```python
import numpy as np
from thermal.matrix_free_mpir_fem import (
    ConvectionBoundary,
    HeatSource,
    LayeredThermalMesh,
    ThermalConductionProblem,
    solve_thermal_conduction,
)

# 35 µm copper / 1.5 mm FR-4 / 35 µm copper stack, 50 mm x 50 mm on a 0.5 mm grid
mesh = LayeredThermalMesh(
    slab_thickness_m=(35e-6, 1.5e-3, 35e-6),
    pitch_x_m=0.5e-3,
    pitch_y_m=0.5e-3,
    conductivity_w_per_m_k=(385.0, 0.8, 385.0),
    through_plane_conductivity_w_per_m_k=(385.0, 0.3, 385.0),
    volumetric_heat_capacity_j_per_m3_k=(3.45e6, 2.0e6, 3.45e6),  # needed by the transient solve
    element_shape=(100, 100),
)
ambient_k = 298.15  # 25 °C

# Top and bottom natural convection (h = 10 W/m²·K) plus a 0.5 W point regulator heat source
problem = ThermalConductionProblem(
    mesh=mesh,
    convection=(
        ConvectionBoundary("top", coefficient_w_per_m2_k=10.0, ambient_temperature_k=ambient_k),
        ConvectionBoundary("bottom", coefficient_w_per_m2_k=10.0, ambient_temperature_k=ambient_k),
    ),
    heat_sources=(
        HeatSource(nodes=((2, 50, 50), (2, 50, 51)), power_w=0.5, name="regulator"),
    ),
)

# Solve using the two-level preconditioner (backend="cuda" for GPU)
solution = solve_thermal_conduction(problem, initial_temperature_k=ambient_k, backend="auto")

assert solution.solve.converged
print(f"Max temperature: {solution.max_temperature_k:.2f} K ({solution.max_temperature_k - 273.15:.2f} °C)")
print(f"Residual heat balance error: {solution.heat_balance_error_w:.2e} W")
```

### 2. Transient heat conduction (backward Euler)

```python
from thermal.matrix_free_mpir_fem import (
    TimeSchedule,
    solve_thermal_transient,
)

# Run a 10-second heating transient with 0.1-second time steps
schedule = TimeSchedule.uniform(step_s=0.1, end_s=10.0)
transient_solution = solve_thermal_transient(
    problem=problem,
    schedule=schedule,
    initial_temperature_k=ambient_k,
    backend="auto",
)

print(f"Final peak temperature: {transient_solution.history[-1].max_temperature_k:.2f} K")
```

## Documentation

For full details on the mathematical formulation, heat budgets, two-level preconditioner, CUDA/native kernels, and electrothermal mappings:
- [Thermal MPIR-FEM Contract & Architecture](../../docs/THERMAL_MPIR_FEM.md)
- [Coupled Multiphysics Scenarios](../../docs/MULTIPHYSICS_SCENARIOS.md)
