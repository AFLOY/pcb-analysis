#include "pcbcore/coupling/coupling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "pcbcore/errors.hpp"
#include "pcbcore/lane_sum.hpp"

namespace pcbcore::coupling {

namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

[[nodiscard]] Index thermal_node(const thermal::ThermalMesh& mesh, const Index slab, const Index row, const Index col) {
    return (slab * (mesh.rows + 1) + row) * (mesh.cols + 1) + col;
}

struct NodeIndex {
    Index layer;
    Index row;
    Index col;
};

[[nodiscard]] NodeIndex split(const DCProblem& problem, const Index node) {
    const Index plane = static_cast<Index>(problem.rows + 1) * (problem.cols + 1);
    const Index in_plane = node % plane;
    return {node / plane, in_plane / (problem.cols + 1), in_plane % (problem.cols + 1)};
}

void check_placement(const DCProblem& problem, const thermal::ThermalMesh& mesh, const Placement& placement) {
    if (static_cast<int>(placement.layer_slabs.size()) != problem.layers) {
        throw InvalidInput("layer_slabs must name one thermal slab per electrical layer");
    }
    for (const int slab : placement.layer_slabs) {
        if (slab < 0 || slab >= mesh.slabs) {
            throw InvalidInput("layer_slabs must name one thermal slab per electrical layer");
        }
    }
    if (problem.rows != mesh.rows || problem.cols != mesh.cols) {
        throw InvalidInput("electrical and thermal meshes must share (rows, cols)");
    }
}

[[nodiscard]] double relative_change(const double loss, const double previous) {
    return (std::isfinite(previous) && loss != 0.0) ? std::abs(loss - previous) / std::abs(loss) : kInfinity;
}

}  // namespace

fem::NodeGroupsView DCProblem::current_groups() const noexcept {
    return {static_cast<Index>(current.size()), current_offsets.data(), current_nodes.data()};
}

fem::NodeGroupsView DCProblem::voltage_groups() const noexcept {
    return {static_cast<Index>(voltage.size()), voltage_offsets.data(), voltage_nodes.data()};
}

DCSolution solve_dc(const DCProblem& problem, const std::vector<double>& conductivity,
                    const std::vector<double>& via_resistance, const double* const initial, const int threads) {
    fem::LayeredDCMeshView mesh;
    mesh.layers = problem.layers;
    mesh.rows = problem.rows;
    mesh.cols = problem.cols;
    mesh.element_active = problem.active.data();
    mesh.layer_thickness_m = problem.thickness.data();
    mesh.pitch_x_m = problem.pitch_x.data();
    mesh.pitch_y_m = problem.pitch_y.data();
    mesh.conductivity_s_per_m = conductivity.data();
    std::vector<double> conductance(via_resistance.size(), 0.0);
    for (std::size_t k = 0; k < via_resistance.size(); ++k) {
        if (!std::isfinite(via_resistance[k]) || !(via_resistance[k] > 0.0)) {
            throw InvalidInput("via resistance must be finite and positive");
        }
        conductance[k] = 1.0 / via_resistance[k];
    }
    const fem::ViaLinksView vias{static_cast<Index>(conductance.size()), problem.via_lower.data(),
                                 problem.via_upper.data(), conductance.data()};
    const fem::LayeredDCSystem system(mesh, vias, problem.reference_node, problem.voltage_nodes.data(),
                                      static_cast<Index>(problem.voltage_nodes.size()), problem.two_level,
                                      problem.block, threads);
    const fem::NodeGroupsView current = problem.current_groups();
    const fem::NodeGroupsView voltage = problem.voltage_groups();
    const std::vector<double> rhs = system.build_rhs(current, problem.current.data(), voltage,
                                                     problem.voltage.data(), threads);
    for (const double value : rhs) {
        if (!std::isfinite(value)) {
            throw InvalidInput("rhs must contain only finite values");
        }
    }
    const std::vector<double> fixed = system.dirichlet_potential(voltage, problem.voltage.data());
    const Index n = system.size();
    DCSolution out;
    out.solution.assign(static_cast<std::size_t>(n), 0.0);
    const auto& free = system.free_nodes();
    if (initial != nullptr) {
        for (Index i = 0; i < n; ++i) {
            const double guess = std::isnan(initial[i]) ? 0.0 : initial[i];
            out.solution[i] = free[i] != 0U ? guess : fixed[i];
        }
    } else if (voltage.count > 0) {
        out.solution = fixed;
    }
    out.solve = system.solve(rhs.data(), out.solution.data(), problem.config, threads);
    const auto& active = system.active_nodes();
    out.potential.assign(static_cast<std::size_t>(n), std::numeric_limits<double>::quiet_NaN());
    for (Index i = 0; i < n; ++i) {
        if (active[i] != 0U) {
            out.potential[i] = out.solution[i];
        }
    }
    out.post = system.post_process(out.solution.data(), threads);
    out.terminal_current = system.group_currents(out.solution.data(), voltage, threads);
    return out;
}

std::vector<double> layer_temperature(const thermal::ThermalMesh& mesh, const Placement& placement,
                                      const double* const t) {
    const int layers = static_cast<int>(placement.layer_slabs.size());
    std::vector<double> out(static_cast<std::size_t>(layers) * mesh.rows * mesh.cols, 0.0);
    for (int l = 0; l < layers; ++l) {
        const Index slab = placement.layer_slabs[l];
        for (int r = 0; r < mesh.rows; ++r) {
            for (int c = 0; c < mesh.cols; ++c) {
                const Index lo = thermal_node(mesh, slab, r, c);
                const Index hi = thermal_node(mesh, slab + 1, r, c);
                const Index nc = mesh.cols + 1;
                const double sum = t[lo] + t[lo + 1] + t[lo + nc] + t[lo + nc + 1] + t[hi] + t[hi + 1] + t[hi + nc] +
                                   t[hi + nc + 1];
                out[(static_cast<Index>(l) * mesh.rows + r) * mesh.cols + c] = 0.125 * sum;
            }
        }
    }
    return out;
}

std::vector<double> conductivity_at(const DCProblem& problem, const thermal::ThermalMesh& mesh,
                                    const Placement& placement, const double* const temperature) {
    const std::vector<double> layer = layer_temperature(mesh, placement, temperature);
    std::vector<double> out(layer.size(), 0.0);
    for (std::size_t e = 0; e < layer.size(); ++e) {
        const double factor = 1.0 + placement.coefficient * (layer[e] - placement.reference_temperature);
        if (!(factor > 0.0)) {
            throw InvalidInput("the linear resistivity model left the valid range (σ ≤ 0)");
        }
        out[e] = problem.conductivity[e] / factor;
    }
    return out;
}

std::vector<double> via_resistance_at(const DCProblem& problem, const thermal::ThermalMesh& mesh,
                                      const Placement& placement, const double* const t) {
    std::vector<double> out(problem.via_resistance.size(), 0.0);
    for (std::size_t k = 0; k < out.size(); ++k) {
        double ends[2] = {0.0, 0.0};
        const Index nodes[2] = {problem.via_lower[k], problem.via_upper[k]};
        for (int end = 0; end < 2; ++end) {
            const NodeIndex node = split(problem, nodes[end]);
            const Index slab = placement.layer_slabs[node.layer];
            ends[end] = 0.5 * (t[thermal_node(mesh, slab, node.row, node.col)] +
                               t[thermal_node(mesh, slab + 1, node.row, node.col)]);
        }
        const double factor = 1.0 + placement.coefficient * (0.5 * (ends[0] + ends[1]) - placement.reference_temperature);
        out[k] = problem.via_resistance[k] * factor;
    }
    return out;
}

std::vector<double> coupled_load(const thermal::ThermalMesh& mesh, const std::vector<HeatShare>& shares,
                                 const ExtraHeat& extra) {
    const Index elements = mesh.elements();
    const Index plane = static_cast<Index>(mesh.rows) * mesh.cols;
    std::vector<double> heat(static_cast<std::size_t>(elements), 0.0);
    std::vector<Index> offsets{0};
    std::vector<Index> nodes;
    std::vector<double> power;
    for (const HeatShare& share : shares) {
        const DCProblem& problem = *share.problem;
        const Placement& placement = *share.placement;
        check_placement(problem, mesh, placement);
        // element_heat_from_losses: one conductor's layers into their slabs, then added.
        std::vector<double> conductor(static_cast<std::size_t>(elements), 0.0);
        for (int l = 0; l < problem.layers; ++l) {
            const Index slab = placement.layer_slabs[l];
            for (Index i = 0; i < plane; ++i) {
                conductor[slab * plane + i] += share.element_loss[l * plane + i];
            }
        }
        for (Index e = 0; e < elements; ++e) {
            heat[e] += conductor[e];
        }
        // via_heat_sources: half of each via's loss at each end, on the two
        // node faces bounding that end's slab.
        for (std::size_t k = 0; k < share.via_loss.size(); ++k) {
            for (const Index end : {problem.via_lower[k], problem.via_upper[k]}) {
                const NodeIndex node = split(problem, end);
                if (node.row > mesh.rows || node.col > mesh.cols) {
                    throw InvalidInput("a via node lies outside the thermal footprint");
                }
                const Index slab = placement.layer_slabs[node.layer];
                nodes.push_back(thermal_node(mesh, slab, node.row, node.col));
                nodes.push_back(thermal_node(mesh, slab + 1, node.row, node.col));
                offsets.push_back(static_cast<Index>(nodes.size()));
                power.push_back(0.5 * share.via_loss[k]);
            }
        }
    }
    if (!extra.element_heat.empty()) {
        for (Index e = 0; e < elements; ++e) {
            heat[e] = heat[e] + extra.element_heat[e];
        }
    }
    for (std::size_t k = 0; k < extra.source_power.size(); ++k) {
        for (Index i = extra.source_offsets[k]; i < extra.source_offsets[k + 1]; ++i) {
            nodes.push_back(extra.source_nodes[i]);
        }
        offsets.push_back(static_cast<Index>(nodes.size()));
        power.push_back(extra.source_power[k]);
    }
    return thermal::nodal_load(mesh, heat.data(), extra.nodal_heat.data(), static_cast<Index>(power.size()),
                               offsets.data(), nodes.data(), power.data());
}

TemperatureFixedPoint::TemperatureFixedPoint(const FixedPointConfig& config, const double floor)
    : config_(config), floor_(floor), relaxation_(config.relaxation) {}

double TemperatureFixedPoint::update(const std::vector<double>& proposed) {
    if (!started_) {
        temperature_ = proposed;
        previous_increment_.clear();
        started_ = true;
        return kInfinity;
    }
    const Index n = static_cast<Index>(proposed.size());
    std::vector<double> increment(proposed.size(), 0.0);
    for (Index i = 0; i < n; ++i) {
        increment[i] = proposed[i] - temperature_[i];
    }
    if (config_.aitken && !previous_increment_.empty()) {
        std::vector<double> difference(proposed.size(), 0.0);
        for (Index i = 0; i < n; ++i) {
            difference[i] = increment[i] - previous_increment_[i];
        }
        const double denominator = lane_sum(n, [&](const std::ptrdiff_t i) { return difference[i] * difference[i]; });
        if (denominator > 0.0) {
            const double numerator =
                lane_sum(n, [&](const std::ptrdiff_t i) { return previous_increment_[i] * difference[i]; });
            relaxation_ = -relaxation_ * numerator / denominator;
            relaxation_ = std::min(std::max(relaxation_, 0.05), config_.max_relaxation);
        }
    }
    std::vector<double> candidate(proposed.size(), 0.0);
    double candidate_min = kInfinity;
    double proposed_min = kInfinity;
    for (Index i = 0; i < n; ++i) {
        candidate[i] = temperature_[i] + relaxation_ * increment[i];
        candidate_min = std::min(candidate_min, candidate[i]);
        proposed_min = std::min(proposed_min, proposed[i]);
    }
    if (candidate_min < floor_ && floor_ <= proposed_min) {
        relaxation_ = 1.0;
        candidate = proposed;
    }
    double change = 0.0;
    for (Index i = 0; i < n; ++i) {
        change = std::max(change, std::abs(candidate[i] - temperature_[i]));
    }
    temperature_ = std::move(candidate);
    previous_increment_ = std::move(increment);
    return change;
}

double resistivity_floor(const Placement& placement) noexcept {
    constexpr double kMinimumFactor = 1.0e-3;
    return placement.coefficient > 0.0
               ? placement.reference_temperature - (1.0 - kMinimumFactor) / placement.coefficient
               : -kInfinity;
}

ElectroThermalResult run_electro_thermal(const DCProblem& problem, const Placement& placement,
                                         const thermal::ThermalProblem& board, const ExtraHeat& extra,
                                         const FixedPointConfig& config, const thermal::SolveOptions& thermal_options,
                                         const int threads) {
    check_placement(problem, board.mesh, placement);
    TemperatureFixedPoint fixed_point(config, resistivity_floor(placement));
    ElectroThermalResult out;
    std::vector<double> conductivity = problem.conductivity;
    std::vector<double> via_resistance = problem.via_resistance;
    std::vector<double> potential;
    double previous_loss = std::numeric_limits<double>::quiet_NaN();
    thermal::ThermalProblem loaded = board;
    thermal::SystemCache cache;
    for (int iteration = 1; iteration <= config.max_iterations; ++iteration) {
        out.electrical = solve_dc(problem, conductivity, via_resistance, potential.empty() ? nullptr : potential.data(),
                                  threads);
        potential = out.electrical.potential;
        const double loss = out.electrical.post.joule_loss;
        if (iteration == 1) {
            out.cold_loss = loss;
        }
        HeatShare share{&problem, &placement, out.electrical.post.element_joule_loss, out.electrical.post.via_joule_loss};
        loaded.load = coupled_load(board.mesh, {share}, extra);
        out.thermal = thermal::solve_steady(loaded, thermal_options,
                                            fixed_point.started() ? fixed_point.temperature().data() : nullptr,
                                            nullptr, nullptr, &cache);
        std::vector<double> proposed(out.thermal.temperature);
        for (double& value : proposed) {
            if (!std::isfinite(value)) {
                value = out.thermal.min_temperature;
            }
        }
        const double change = fixed_point.update(proposed);
        const auto& temperature = fixed_point.temperature();
        const double relative = relative_change(loss, previous_loss);
        previous_loss = loss;
        CouplingStep step;
        step.iteration = iteration;
        step.joule_loss = loss;
        step.max_temperature = *std::max_element(temperature.begin(), temperature.end());
        step.temperature_change = change;
        step.relative_loss_change = relative;
        step.relaxation = fixed_point.relaxation();
        step.electrical_inner_iterations = out.electrical.solve.inner_iterations;
        step.thermal_inner_iterations = out.thermal.solve.inner_iterations;
        out.history.push_back(step);
        const bool solved = out.electrical.solve.converged && out.thermal.solve.converged;
        if (placement.coefficient == 0.0) {
            out.converged = solved;
            break;
        }
        if (change <= config.temperature_tolerance && relative <= config.relative_loss_tolerance && solved) {
            out.converged = true;
            break;
        }
        conductivity = conductivity_at(problem, board.mesh, placement, temperature.data());
        via_resistance = via_resistance_at(problem, board.mesh, placement, temperature.data());
    }
    out.conductivity = conductivity;
    out.via_resistance = via_resistance;
    out.element_temperature = layer_temperature(board.mesh, placement, fixed_point.temperature().data());
    return out;
}

namespace {

[[nodiscard]] std::shared_ptr<fem::LayeredDCSystem> port_system(const DCProblem& problem,
                                                                const std::vector<double>& conductivity,
                                                                const std::vector<double>& via_resistance,
                                                                const int threads) {
    fem::LayeredDCMeshView mesh;
    mesh.layers = problem.layers;
    mesh.rows = problem.rows;
    mesh.cols = problem.cols;
    mesh.element_active = problem.active.data();
    mesh.layer_thickness_m = problem.thickness.data();
    mesh.pitch_x_m = problem.pitch_x.data();
    mesh.pitch_y_m = problem.pitch_y.data();
    mesh.conductivity_s_per_m = conductivity.data();
    std::vector<double> conductance(via_resistance.size(), 0.0);
    for (std::size_t k = 0; k < via_resistance.size(); ++k) {
        if (!std::isfinite(via_resistance[k]) || !(via_resistance[k] > 0.0)) {
            throw InvalidInput("via resistance must be finite and positive");
        }
        conductance[k] = 1.0 / via_resistance[k];
    }
    const fem::ViaLinksView vias{static_cast<Index>(conductance.size()), problem.via_lower.data(),
                                 problem.via_upper.data(), conductance.data()};
    return std::make_shared<fem::LayeredDCSystem>(mesh, vias, -1, problem.voltage_nodes.data(),
                                                  static_cast<Index>(problem.voltage_nodes.size()), problem.two_level,
                                                  problem.block, threads);
}

}  // namespace

CircuitCoupledResult run_circuit_coupled(const std::vector<PortConductor>& conductors,
                                         const thermal::ThermalProblem& board, const ExtraHeat& extra,
                                         const FixedPointConfig& config, const thermal::SolveOptions& thermal_options,
                                         const PortCircuit& circuit, const int threads) {
    if (conductors.empty()) {
        throw InvalidInput("at least one conductor is required");
    }
    for (const PortConductor& conductor : conductors) {
        check_placement(*conductor.problem, board.mesh, conductor.placement);
    }
    const Placement& model = conductors.front().placement;
    TemperatureFixedPoint fixed_point(config, resistivity_floor(model));
    CircuitCoupledResult out;
    out.conductors.resize(conductors.size());
    std::vector<fem::CorrelationModes> modes(conductors.size());
    double previous_loss = std::numeric_limits<double>::quiet_NaN();
    thermal::ThermalProblem loaded = board;
    thermal::SystemCache cache;
    for (int iteration = 1; iteration <= config.max_iterations; ++iteration) {
        std::vector<std::vector<double>> conductances;
        int electrical_inner = 0;
        bool electrical_converged = true;
        for (std::size_t c = 0; c < conductors.size(); ++c) {
            const PortConductor& conductor = conductors[c];
            const DCProblem& problem = *conductor.problem;
            ConductorOutcome& state = out.conductors[c];
            if (fixed_point.started()) {
                state.conductivity = conductivity_at(problem, board.mesh, conductor.placement,
                                                     fixed_point.temperature().data());
                state.via_resistance = via_resistance_at(problem, board.mesh, conductor.placement,
                                                         fixed_point.temperature().data());
            } else {
                state.conductivity = problem.conductivity;
                state.via_resistance = problem.via_resistance;
            }
            state.system = port_system(problem, state.conductivity, state.via_resistance, threads);
            const std::vector<double> previous = std::move(state.basis.unit_voltage);
            state.basis = fem::dc_port_basis(*state.system, problem.voltage_groups(), conductor.reference_port,
                                             previous.empty() ? nullptr : previous.data(), problem.config,
                                             conductor.width, conductor.team);
            for (const auto& solve : state.basis.solves) {
                electrical_inner += solve.inner_iterations;
                electrical_converged = electrical_converged && solve.converged;
            }
            conductances.push_back(state.basis.conductance);
        }
        const std::vector<ReducedCorrelation> correlations = circuit(conductances);
        if (correlations.size() != conductors.size()) {
            throw InvalidInput("the circuit must return one correlation per conductor");
        }
        double loss = 0.0;
        std::vector<HeatShare> shares;
        for (std::size_t c = 0; c < conductors.size(); ++c) {
            ConductorOutcome& state = out.conductors[c];
            const Index driven = state.basis.ports - 1;
            if (static_cast<Index>(correlations[c].reduced.size()) != driven * driven) {
                throw InvalidInput("the reduced correlation must be (ports - 1) x (ports - 1)");
            }
            modes[c] = fem::correlation_modes(correlations[c].reduced.data(), driven, correlations[c].scale,
                                              state.basis.unit_current.data(), state.system->size());
            fem::ModalLoss modal = fem::modal_loss(*state.system, modes[c], threads);
            state.element_loss = std::move(modal.element);
            state.via_loss = std::move(modal.via);
            loss += lane_sum(static_cast<std::ptrdiff_t>(state.element_loss.size()),
                             [&](const std::ptrdiff_t i) { return state.element_loss[i]; }) +
                    lane_sum(static_cast<std::ptrdiff_t>(state.via_loss.size()),
                             [&](const std::ptrdiff_t i) { return state.via_loss[i]; });
            shares.push_back({conductors[c].problem, &conductors[c].placement, state.element_loss, state.via_loss});
        }
        if (iteration == 1) {
            out.cold_loss = loss;
        }
        loaded.load = coupled_load(board.mesh, shares, extra);
        out.thermal = thermal::solve_steady(loaded, thermal_options,
                                            fixed_point.started() ? fixed_point.temperature().data() : nullptr,
                                            nullptr, nullptr, &cache);
        std::vector<double> proposed(out.thermal.temperature);
        for (double& value : proposed) {
            if (!std::isfinite(value)) {
                value = out.thermal.min_temperature;
            }
        }
        const double change = fixed_point.update(proposed);
        const auto& temperature = fixed_point.temperature();
        const double relative = relative_change(loss, previous_loss);
        previous_loss = loss;
        CouplingStep step;
        step.iteration = iteration;
        step.joule_loss = loss;
        step.max_temperature = *std::max_element(temperature.begin(), temperature.end());
        step.temperature_change = change;
        step.relative_loss_change = relative;
        step.relaxation = fixed_point.relaxation();
        step.electrical_inner_iterations = electrical_inner;
        step.thermal_inner_iterations = out.thermal.solve.inner_iterations;
        out.history.push_back(step);
        const bool solved = electrical_converged && out.thermal.solve.converged;
        if (model.coefficient == 0.0) {
            out.converged = solved;
            break;
        }
        if (change <= config.temperature_tolerance && relative <= config.relative_loss_tolerance && solved) {
            out.converged = true;
            break;
        }
    }
    for (std::size_t c = 0; c < conductors.size(); ++c) {
        ConductorOutcome& state = out.conductors[c];
        state.rms_current_density = fem::modal_rms_current_density(*state.system, modes[c], threads);
        state.element_temperature = layer_temperature(board.mesh, conductors[c].placement,
                                                      fixed_point.temperature().data());
    }
    return out;
}

}  // namespace pcbcore::coupling
