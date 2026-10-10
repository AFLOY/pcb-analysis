// The staggered electro-thermal coupling loops (multiphysics/staggered_coupling):
// a layered DC conduction problem solved at a temperature, the copper and
// via resistivity law, the Joule heat it hands to the thermal mesh, the
// relaxed fixed point on the temperature field, and the loops around them.
#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "pcbcore/fem/dc_ports.hpp"
#include "pcbcore/fem/layered_dc_system.hpp"
#include "pcbcore/fem/mpir.hpp"
#include "pcbcore/thermal/thermal_system.hpp"

namespace pcbcore::coupling {

using Index = std::int64_t;

// One conductor role: its mesh at the reference temperature, vias, gauge and drive.
struct DCProblem {
    int layers{0};
    int rows{0};
    int cols{0};
    std::vector<std::uint8_t> active;     // (layers, rows, cols)
    std::vector<double> thickness;        // (layers,)
    std::vector<double> pitch_x;          // (cols,)
    std::vector<double> pitch_y;          // (rows,)
    std::vector<double> conductivity;     // (layers, rows, cols), at the reference temperature
    std::vector<Index> via_lower;         // flat nodes
    std::vector<Index> via_upper;
    std::vector<double> via_resistance;   // at the reference temperature
    Index reference_node{-1};
    std::vector<Index> current_offsets{0};
    std::vector<Index> current_nodes;
    std::vector<double> current;
    std::vector<Index> voltage_offsets{0};
    std::vector<Index> voltage_nodes;
    std::vector<double> voltage;
    bool two_level{true};
    int block{0};
    fem::MpirConfig config;

    [[nodiscard]] fem::NodeGroupsView current_groups() const noexcept;
    [[nodiscard]] fem::NodeGroupsView voltage_groups() const noexcept;
    [[nodiscard]] Index nodes() const noexcept { return static_cast<Index>(layers) * (rows + 1) * (cols + 1); }
};

struct DCSolution {
    std::vector<double> potential;  // nodes, NaN off the copper
    std::vector<double> solution;   // the MPIR solution
    fem::LayeredDCPost post;
    std::vector<double> terminal_current;  // per voltage terminal
    fem::MpirResult solve;
};

// solve_pcb_dc at the given conductivity and via resistances; ``initial``
// (a reported potential, NaN read as zero) warm-starts the free nodes.
[[nodiscard]] DCSolution solve_dc(const DCProblem& problem, const std::vector<double>& conductivity,
                                  const std::vector<double>& via_resistance, const double* initial, int threads);

// How the copper of one conductor sits in the thermal mesh, and rho(T).
struct Placement {
    std::vector<int> layer_slabs;  // one slab per electrical layer
    double reference_temperature{293.15};
    double coefficient{3.93e-3};
};

// Element temperature of each electrical layer, (layers, rows, cols).
[[nodiscard]] std::vector<double> layer_temperature(const thermal::ThermalMesh& mesh, const Placement& placement,
                                                    const double* temperature);
// sigma / (1 + alpha (T - T_ref)) per element; throws when 1 + alpha (T - T_ref) <= 0.
[[nodiscard]] std::vector<double> conductivity_at(const DCProblem& problem, const thermal::ThermalMesh& mesh,
                                                  const Placement& placement, const double* temperature);
// R (1 + alpha (T_via - T_ref)), T_via the mean of the slab faces at both ends.
[[nodiscard]] std::vector<double> via_resistance_at(const DCProblem& problem, const thermal::ThermalMesh& mesh,
                                                    const Placement& placement, const double* temperature);

// Heat of one conductor's losses: element losses into their slabs, half of
// each via's loss shared by the two slab faces at each end.
struct HeatShare {
    const DCProblem* problem{nullptr};
    const Placement* placement{nullptr};
    std::vector<double> element_loss;  // (layers, rows, cols)
    std::vector<double> via_loss;      // (vias,)
};

// The nodal load of the thermal problem: ``extra_element_heat`` (or null)
// added to the conductors' element heat, ``nodal_heat``, then the via heat
// and ``extra sources`` (CSR over thermal nodes).
struct ExtraHeat {
    std::vector<double> element_heat;  // empty: none
    std::vector<double> nodal_heat;    // nodes
    std::vector<Index> source_offsets{0};
    std::vector<Index> source_nodes;
    std::vector<double> source_power;
};
[[nodiscard]] std::vector<double> coupled_load(const thermal::ThermalMesh& mesh, const std::vector<HeatShare>& shares,
                                               const ExtraHeat& extra);

struct FixedPointConfig {
    int max_iterations{20};
    double temperature_tolerance{1.0e-3};
    double relative_loss_tolerance{1.0e-6};
    double relaxation{1.0};
    bool aitken{true};
    double max_relaxation{4.0};
};

// Relaxed fixed point with Aitken's delta-squared rescaling; below ``floor``
// an over-relaxed step falls back to the plain one.
class TemperatureFixedPoint {
public:
    TemperatureFixedPoint(const FixedPointConfig& config, double floor);
    // Accept a proposed field; the largest change of the relaxed iterate.
    double update(const std::vector<double>& proposed);
    [[nodiscard]] bool started() const noexcept { return started_; }
    [[nodiscard]] const std::vector<double>& temperature() const noexcept { return temperature_; }
    [[nodiscard]] double relaxation() const noexcept { return relaxation_; }

private:
    FixedPointConfig config_;
    double floor_;
    double relaxation_;
    bool started_{false};
    std::vector<double> temperature_;
    std::vector<double> previous_increment_;
};

// The floor that keeps 1 + alpha (T - T_ref) >= 1e-3.
[[nodiscard]] double resistivity_floor(const Placement& placement) noexcept;

struct CouplingStep {
    int iteration{0};
    double joule_loss{0.0};
    double max_temperature{0.0};
    double temperature_change{0.0};
    double relative_loss_change{0.0};
    double relaxation{1.0};
    int electrical_inner_iterations{0};
    int thermal_inner_iterations{0};
};

struct ElectroThermalResult {
    DCSolution electrical;
    thermal::ThermalSolution thermal;
    std::vector<double> conductivity;
    std::vector<double> via_resistance;
    std::vector<double> element_temperature;
    bool converged{false};
    std::vector<CouplingStep> history;
    double cold_loss{std::numeric_limits<double>::quiet_NaN()};
};

// run_electro_thermal: ``board`` is the thermal problem without the
// conductor's heat (its load is replaced every iteration).
[[nodiscard]] ElectroThermalResult run_electro_thermal(const DCProblem& problem, const Placement& placement,
                                                       const thermal::ThermalProblem& board, const ExtraHeat& extra,
                                                       const FixedPointConfig& config,
                                                       const thermal::SolveOptions& thermal_options, int threads);

// One conductor role of a circuit-coupled board: its copper (``problem``;
// the voltage groups are its port pads, all held as Dirichlet nodes), the
// reference port, where it sits, and how its unit solves share the budget.
struct PortConductor {
    const DCProblem* problem{nullptr};
    Index reference_port{0};
    Placement placement;
    int width{1};
    int team{1};
};

// A port-current correlation restricted to the driven ports, and its scale.
struct ReducedCorrelation {
    std::vector<double> reduced;  // (ports - 1, ports - 1)
    double scale{1.0};
};

// The circuit: the conductance matrix of every conductor in, one reduced
// correlation per conductor out.  Called once per iteration, outside every
// parallel region.
using PortCircuit = std::function<std::vector<ReducedCorrelation>(const std::vector<std::vector<double>>&)>;

struct ConductorOutcome {
    std::shared_ptr<fem::LayeredDCSystem> system;
    fem::DCPortBasisResult basis;
    std::vector<double> conductivity;
    std::vector<double> via_resistance;
    std::vector<double> element_loss;
    std::vector<double> via_loss;
    std::vector<double> rms_current_density;
    std::vector<double> element_temperature;
};

struct CircuitCoupledResult {
    std::vector<ConductorOutcome> conductors;
    thermal::ThermalSolution thermal;
    bool converged{false};
    std::vector<CouplingStep> history;
    double cold_loss{std::numeric_limits<double>::quiet_NaN()};
};

// run_circuit_coupled: N-port bases, the circuit, the thermal solve and the
// fixed point, iterated in the core with one circuit call per iteration.
// ``placement`` of every conductor shares T_ref and alpha (the scenario's).
[[nodiscard]] CircuitCoupledResult run_circuit_coupled(const std::vector<PortConductor>& conductors,
                                                       const thermal::ThermalProblem& board, const ExtraHeat& extra,
                                                       const FixedPointConfig& config,
                                                       const thermal::SolveOptions& thermal_options,
                                                       const PortCircuit& circuit, int threads);

}  // namespace pcbcore::coupling
