# Multiphysics Coupling (`multiphysics.staggered_coupling`)

Partitioned (staggered) multiphysics workflow chaining the `electrical`, `thermal`, and `emc` solvers.

Iterates coupled physics to self-consistent fixed points with warm starts, Aitken $\Delta^2$ relaxation, and direct Joule-loss mapping without spatial interpolation. Dispatched through a unified `run_scenario` entry point.

## Supported Coupled Scenarios

| Scenario Class | Coupled Physics | Formulation / Solver Chain |
|---|---|---|
| `ElectroThermalScenario` | DC Conduction + Thermal | Fixed-point iteration with temperature-dependent copper $\sigma(T) = \sigma_0 / (1 + \alpha (T - T_0))$ and via $R(T)$, relaxed with Aitken acceleration |
| `ElectroThermalEnclosureScenario` | Board + 3D Bodies + $\sigma(T)$ | Electro-thermal loop coupled across contact interfaces to separately meshed heat sinks/enclosures |
| `ElectroThermalEmissionScenario` | DC + Thermal + EMC | Evaluates radiated emissions (CISPR 32 / FCC Part 15) using the thermally converged current distribution |
| `ElectroEmissionScenario` | DC Conduction + EMC | Evaluates near/far-field emissions directly from cold DC conduction currents |
| `SheetPeecEmissionScenario` | Sheet PEEC + EMC | Evaluates radiated emissions frequency-by-frequency from full-wave sheet PEEC currents |

---

## Quick Start

### Electro-Thermal Fixed-Point Coupling

```python
from multiphysics.staggered_coupling import (
    ElectroThermalEmissionScenario,
    ElectroThermalScenario,
    EmissionScenario,
    run_scenario,
)
from thermal.matrix_free_mpir_fem import ConvectionBoundary

# 1. Define the coupled electro-thermal scenario
# (Assumes pcb_problem: PCBConductionProblem, thermal_mesh: LayeredThermalMesh)
coupled_scenario = ElectroThermalScenario(
    electrical=pcb_problem,
    thermal_mesh=thermal_mesh,
    layer_slabs=(0, 2),  # Thermal slab index for each electrical layer
    convection=(
        ConvectionBoundary("top", coefficient_w_per_m2_k=10.0, ambient_temperature_k=298.15),
        ConvectionBoundary("bottom", coefficient_w_per_m2_k=10.0, ambient_temperature_k=298.15),
    ),
)

# 2. Run the fixed-point iteration
coupled_result = run_scenario(coupled_scenario)
print(f"Converged: {coupled_result.converged} in {coupled_result.iterations} iterations")
print(f"Resistance/loss increase due to Joule heating: {coupled_result.loss_increase_ratio:.3f}x")

# 3. Chain with radiated-emission evaluation on the hot board
chained_scenario = ElectroThermalEmissionScenario(
    electro_thermal=coupled_scenario,
    layer_height_m=(0.0, 1.6e-3),
    emission=EmissionScenario(frequencies_hz=(30e6, 100e6, 300e6)),  # Test at CISPR 32 bands
)
chained_result = run_scenario(chained_scenario)
print(f"Worst margin to limit line: {chained_result.emission.worst_margin_db:+.1f} dB")
```

## Documentation

- [Coupled Multiphysics Scenarios Contract & Architecture](../../docs/MULTIPHYSICS_SCENARIOS.md)
- [Matrix-free MPIR-FEM Contract](../../docs/MATRIX_FREE_MPIR_FEM.md)
- [Thermal MPIR-FEM Contract](../../docs/THERMAL_MPIR_FEM.md)
- [EMC Dipole Superposition Contract](../../docs/EMC_DIPOLE_SUPERPOSITION.md)
