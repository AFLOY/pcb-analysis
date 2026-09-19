"""Staggered (partitioned) coupling of the electrical, thermal, and EMC solvers.

The electro-thermal chain iterates the two conduction solves to a
self-consistent copper temperature with warm starts and Aitken relaxation.
The emission chains hand a converged current distribution to the dipole
superposition.  ``run_scenario`` dispatches on the scenario dataclass.
"""

from .board_enclosure import (
    BoardEnclosureThermalResult,
    BoardEnclosureThermalScenario,
    BodyContact,
    InterfaceCouplingConfig,
    InterfaceStep,
    run_board_enclosure_thermal,
)
from .electro_thermal import (
    COPPER_TEMPERATURE_COEFFICIENT_PER_K,
    CouplingConfig,
    CouplingStep,
    ElectroThermalResult,
    ElectroThermalScenario,
    TemperatureFixedPoint,
    conductivity_at_temperature,
    electrical_layer_temperature_k,
    heated_electrical_problem,
    run_electro_thermal,
    slab_element_temperature_k,
    thermal_problem_with_joule_heat,
    via_node_temperature_k,
)
from .electro_thermal_enclosure import (
    ElectroThermalEnclosureResult,
    ElectroThermalEnclosureScenario,
    ElectroThermalEnclosureStep,
    run_electro_thermal_enclosure,
)
from .emission import (
    EmissionPoint,
    EmissionResult,
    EmissionScenario,
    ScanPlane,
    evaluate_emission,
    run_pcb_dc_emission,
    run_sheet_peec_emission,
)
from .scenarios import (
    ElectricalScenario,
    ElectroEmissionResult,
    ElectroEmissionScenario,
    ElectroThermalEmissionResult,
    ElectroThermalEmissionScenario,
    SheetPeecEmissionResult,
    SheetPeecEmissionScenario,
    ThermalScenario,
    ThermalTransientScenario,
    run_scenario,
    run_scenarios,
)

__all__ = [
    "BoardEnclosureThermalResult",
    "BoardEnclosureThermalScenario",
    "BodyContact",
    "COPPER_TEMPERATURE_COEFFICIENT_PER_K",
    "InterfaceCouplingConfig",
    "InterfaceStep",
    "CouplingConfig",
    "CouplingStep",
    "ElectricalScenario",
    "ElectroEmissionResult",
    "ElectroEmissionScenario",
    "ElectroThermalEmissionResult",
    "ElectroThermalEmissionScenario",
    "ElectroThermalEnclosureResult",
    "ElectroThermalEnclosureScenario",
    "ElectroThermalEnclosureStep",
    "ElectroThermalResult",
    "ElectroThermalScenario",
    "EmissionPoint",
    "EmissionResult",
    "EmissionScenario",
    "ScanPlane",
    "SheetPeecEmissionResult",
    "SheetPeecEmissionScenario",
    "TemperatureFixedPoint",
    "ThermalScenario",
    "ThermalTransientScenario",
    "conductivity_at_temperature",
    "electrical_layer_temperature_k",
    "evaluate_emission",
    "heated_electrical_problem",
    "run_board_enclosure_thermal",
    "run_electro_thermal",
    "run_electro_thermal_enclosure",
    "run_pcb_dc_emission",
    "run_scenario",
    "run_scenarios",
    "run_sheet_peec_emission",
    "slab_element_temperature_k",
    "thermal_problem_with_joule_heat",
    "via_node_temperature_k",
]
