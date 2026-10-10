#include "pcbcore/coupling/enclosure.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "pcbcore/errors.hpp"
#include "pcbcore/lane_sum.hpp"

namespace pcbcore::coupling {

namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

// Local corners (4 dz + 2 dy + dx) of each face direction, FACE_DIRECTIONS order.
constexpr int kFaceCorners[6][4] = {
    {0, 2, 4, 6}, {1, 3, 5, 7}, {0, 1, 4, 5}, {2, 3, 6, 7}, {0, 1, 2, 3}, {4, 5, 6, 7},
};

[[nodiscard]] std::vector<double> face_mean(const thermal::ThermalMesh& mesh, const double* const t,
                                            const std::vector<Index>& cells, const int stride, const int fixed_slab,
                                            const int face) {
    const Index pairs = static_cast<Index>(cells.size()) / stride;
    std::vector<double> out(static_cast<std::size_t>(pairs), 0.0);
    for (Index p = 0; p < pairs; ++p) {
        const int slab = stride == 2 ? fixed_slab : static_cast<int>(cells[p * stride]);
        const int row = static_cast<int>(cells[p * stride + stride - 2]);
        const int col = static_cast<int>(cells[p * stride + stride - 1]);
        double sum = 0.0;
        for (const int corner : kFaceCorners[face]) {
            sum += t[mesh.corner_node(slab, row, col, corner)];
        }
        out[p] = sum / 4.0;
    }
    return out;
}

[[nodiscard]] double relative_change(const double value, const double previous) {
    return (std::isfinite(previous) && value != 0.0) ? std::abs(value - previous) / std::abs(value) : kInfinity;
}

[[nodiscard]] double sum_of(const std::vector<double>& values) {
    return lane_sum(static_cast<std::ptrdiff_t>(values.size()), [&](const std::ptrdiff_t i) { return values[i]; });
}

[[nodiscard]] std::vector<double> finite_or_min(const thermal::ThermalSolution& solution) {
    std::vector<double> out(solution.temperature);
    for (double& value : out) {
        if (std::isnan(value)) {
            value = solution.min_temperature;
        }
    }
    return out;
}

}  // namespace

std::vector<double> ContactMap::board_face_temperature(const thermal::ThermalMesh& board, const double* const t) const {
    return face_mean(board, t, board_cells, 2, board_top ? board.slabs - 1 : 0, board_top ? 5 : 4);
}

std::vector<double> ContactMap::body_face_temperature(const thermal::ThermalMesh& body, const double* const t) const {
    return face_mean(body, t, body_cells, 3, 0, body_face);
}

thermal::FaceBoundary ContactMap::board_robin(const thermal::ThermalMesh& board,
                                              const std::vector<double>& body_temperature) const {
    if (static_cast<Index>(body_temperature.size()) != size()) {
        throw InvalidInput("one body temperature per contact pair is required");
    }
    const Index plane = static_cast<Index>(board.rows) * board.cols;
    std::vector<double> conductance_sum(static_cast<std::size_t>(plane), 0.0);
    std::vector<double> weighted(static_cast<std::size_t>(plane), 0.0);
    for (Index p = 0; p < size(); ++p) {
        const Index at = board_cells[2 * p] * board.cols + board_cells[2 * p + 1];
        conductance_sum[at] += conductance[p];
        weighted[at] += conductance[p] * body_temperature[p];
    }
    thermal::FaceBoundary out;
    out.kind = board_top ? thermal::FaceKind::Top : thermal::FaceKind::Bottom;
    out.directions = {board_top ? 5 : 4};
    out.coefficient.assign(static_cast<std::size_t>(board.elements()), 0.0);
    out.ambient.assign(static_cast<std::size_t>(board.elements()), 0.0);
    const Index base = static_cast<Index>(board_top ? board.slabs - 1 : 0) * plane;
    for (int r = 0; r < board.rows; ++r) {
        for (int c = 0; c < board.cols; ++c) {
            const Index at = static_cast<Index>(r) * board.cols + c;
            const bool touched = conductance_sum[at] > 0.0;
            out.ambient[base + at] = touched ? weighted[at] / conductance_sum[at] : 0.0;
            out.coefficient[base + at] = conductance_sum[at] / (board.pitch_y[r] * board.pitch_x[c]);
        }
    }
    out.mean_ambient = lane_sum(plane, [&](const std::ptrdiff_t i) { return out.ambient[base + i]; }) /
                       static_cast<double>(plane);
    return out;
}

std::vector<double> ContactMap::body_nodal_heat(const thermal::ThermalMesh& body,
                                                const std::vector<double>& pair_heat) const {
    std::vector<double> load(static_cast<std::size_t>(body.nodes()), 0.0);
    for (Index p = 0; p < size(); ++p) {
        const double share = pair_heat[p] / 4.0;
        for (const int corner : kFaceCorners[body_face]) {
            load[body.corner_node(static_cast<int>(body_cells[3 * p]), static_cast<int>(body_cells[3 * p + 1]),
                                  static_cast<int>(body_cells[3 * p + 2]), corner)] += share;
        }
    }
    return load;
}

double EnclosureResult::interface_heat() const {
    double total = 0.0;
    for (const auto& heat : contact_heat) {
        total += sum_of(heat);
    }
    return total;
}

EnclosureResult run_board_enclosure(const thermal::ThermalProblem& board, const std::vector<Body>& bodies,
                                    const InterfaceConfig& config, const EnclosureResult* const initial) {
    if (bodies.empty()) {
        throw InvalidInput("at least one body is required");
    }
    std::vector<std::vector<double>> body_temperature;
    for (const Body& body : bodies) {
        body_temperature.emplace_back(static_cast<std::size_t>(body.contact.size()), body.start);
    }
    std::vector<double> board_guess;
    std::vector<std::vector<double>> body_guess(bodies.size());
    if (initial != nullptr) {
        if (initial->bodies.size() != bodies.size()) {
            throw InvalidInput("initial must come from a scenario with the same bodies and contacts");
        }
        for (std::size_t b = 0; b < bodies.size(); ++b) {
            if (initial->contact_temperature[b].size() != body_temperature[b].size()) {
                throw InvalidInput("initial must come from a scenario with the same bodies and contacts");
            }
        }
        body_temperature = initial->contact_temperature;
        board_guess = finite_or_min(initial->board);
        for (std::size_t b = 0; b < bodies.size(); ++b) {
            body_guess[b] = finite_or_min(initial->bodies[b]);
        }
    }
    // Every body keeps one problem whose load is summed again with the
    // contact heat each exchange, so its prepared system is reused.
    std::vector<thermal::ThermalProblem> loaded;
    loaded.reserve(bodies.size());
    for (const Body& body : bodies) {
        loaded.push_back(*body.problem);
    }
    std::vector<thermal::SystemCache> caches(bodies.size());
    thermal::ThermalProblem board_problem = board;

    EnclosureResult out;
    std::vector<double> previous_increment;
    double relaxation = config.relaxation;
    double previous_heat = std::numeric_limits<double>::quiet_NaN();
    for (int iteration = 1; iteration <= config.max_iterations; ++iteration) {
        board_problem.convection = board.convection;
        for (std::size_t b = 0; b < bodies.size(); ++b) {
            board_problem.convection.push_back(bodies[b].contact.board_robin(board.mesh, body_temperature[b]));
        }
        out.board = thermal::solve_steady(board_problem, config.board, board_guess.empty() ? nullptr : board_guess.data(),
                                          nullptr, nullptr);
        board_guess = finite_or_min(out.board);

        out.bodies.clear();
        out.contact_heat.clear();
        std::vector<double> current_all;
        std::vector<double> proposed_all;
        int body_inner = 0;
        for (std::size_t b = 0; b < bodies.size(); ++b) {
            const Body& body = bodies[b];
            const std::vector<double> board_face =
                body.contact.board_face_temperature(board.mesh, out.board.temperature.data());
            std::vector<double> heat(board_face.size(), 0.0);
            for (std::size_t p = 0; p < heat.size(); ++p) {
                heat[p] = body.contact.conductance[p] * (board_face[p] - body_temperature[b][p]);
            }
            const std::vector<double> contact = body.contact.body_nodal_heat(body.problem->mesh, heat);
            thermal::ThermalProblem& problem = loaded[b];
            for (std::size_t n = 0; n < contact.size(); ++n) {
                problem.nodal_heat[n] = body.problem->nodal_heat[n] + contact[n];
            }
            problem.sum_load();
            thermal::ThermalSolution solution = thermal::solve_steady(
                problem, config.body, body_guess[b].empty() ? nullptr : body_guess[b].data(), nullptr, nullptr,
                &caches[b]);
            body_guess[b] = finite_or_min(solution);
            body_inner += solution.solve.inner_iterations;
            const std::vector<double> proposed =
                body.contact.body_face_temperature(body.problem->mesh, solution.temperature.data());
            current_all.insert(current_all.end(), body_temperature[b].begin(), body_temperature[b].end());
            proposed_all.insert(proposed_all.end(), proposed.begin(), proposed.end());
            out.bodies.push_back(std::move(solution));
            out.contact_heat.push_back(std::move(heat));
        }
        // Relaxed fixed point on the stacked contact temperatures, with
        // Aitken's delta-squared rescaling from two successive increments.
        const Index n = static_cast<Index>(current_all.size());
        std::vector<double> increment(current_all.size(), 0.0);
        for (Index i = 0; i < n; ++i) {
            increment[i] = proposed_all[i] - current_all[i];
        }
        if (config.aitken && !previous_increment.empty()) {
            std::vector<double> difference(increment.size(), 0.0);
            for (Index i = 0; i < n; ++i) {
                difference[i] = increment[i] - previous_increment[i];
            }
            const double denominator = lane_sum(n, [&](const std::ptrdiff_t i) { return difference[i] * difference[i]; });
            if (denominator > 0.0) {
                relaxation = -relaxation *
                             lane_sum(n, [&](const std::ptrdiff_t i) { return previous_increment[i] * difference[i]; }) /
                             denominator;
                relaxation = std::min(std::max(relaxation, 0.05), config.max_relaxation);
            }
        }
        // np.max: NaN wins.
        double change = 0.0;
        bool not_a_number = false;
        Index at = 0;
        for (std::size_t b = 0; b < bodies.size(); ++b) {
            for (double& value : body_temperature[b]) {
                const double updated = current_all[at] + relaxation * increment[at];
                const double step = std::abs(updated - current_all[at]);
                not_a_number = not_a_number || std::isnan(step);
                change = std::max(change, step);
                value = updated;
                ++at;
            }
        }
        if (not_a_number) {
            change = std::numeric_limits<double>::quiet_NaN();
        }
        previous_increment = std::move(increment);

        const double total_heat = out.interface_heat();
        const double relative = relative_change(total_heat, previous_heat);
        previous_heat = total_heat;
        out.history.push_back({iteration, total_heat, change, relative, relaxation, out.board.solve.inner_iterations,
                               body_inner});
        if (change <= config.temperature_tolerance && relative <= config.relative_heat_tolerance) {
            out.converged = out.board.solve.converged;
            for (const auto& solution : out.bodies) {
                out.converged = out.converged && solution.solve.converged;
            }
            break;
        }
        if (!std::isfinite(change) || change > config.divergence_temperature) {
            // A hard contact on a stiff body has a fixed-point gain near one;
            // without relaxation the exchange oscillates with growing
            // amplitude.  Stop before the solves see infinities.
            break;
        }
    }
    out.contact_temperature = body_temperature;
    return out;
}

ElectroThermalEnclosureResult run_electro_thermal_enclosure(const DCProblem& problem, const Placement& placement,
                                                            const thermal::ThermalProblem& board,
                                                            const ExtraHeat& extra, const std::vector<Body>& bodies,
                                                            const FixedPointConfig& config,
                                                            const InterfaceConfig& interface, const int threads) {
    TemperatureFixedPoint fixed_point(config, resistivity_floor(placement));
    ElectroThermalEnclosureResult out;
    std::vector<double> conductivity = problem.conductivity;
    std::vector<double> via_resistance = problem.via_resistance;
    std::vector<double> potential;
    double previous_loss = std::numeric_limits<double>::quiet_NaN();
    thermal::ThermalProblem loaded = board;
    bool have_thermal = false;
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
        out.thermal = run_board_enclosure(loaded, bodies, interface, have_thermal ? &out.thermal : nullptr);
        have_thermal = true;
        std::vector<double> proposed = finite_or_min(out.thermal.board);
        for (double& value : proposed) {
            if (!std::isfinite(value)) {
                value = out.thermal.board.min_temperature;
            }
        }
        const double change = fixed_point.update(proposed);
        const auto& temperature = fixed_point.temperature();
        const double relative = relative_change(loss, previous_loss);
        previous_loss = loss;
        int thermal_inner = out.thermal.board.solve.inner_iterations;
        for (const auto& body : out.thermal.bodies) {
            thermal_inner += body.solve.inner_iterations;
        }
        EnclosureStep step;
        step.step = {iteration,
                     loss,
                     *std::max_element(temperature.begin(), temperature.end()),
                     change,
                     relative,
                     fixed_point.relaxation(),
                     out.electrical.solve.inner_iterations,
                     thermal_inner};
        step.interface_iterations = static_cast<int>(out.thermal.history.size());
        step.interface_heat = out.thermal.interface_heat();
        out.history.push_back(step);
        if (placement.coefficient == 0.0) {
            out.converged = out.electrical.solve.converged && out.thermal.converged;
            break;
        }
        if (change <= config.temperature_tolerance && relative <= config.relative_loss_tolerance &&
            out.electrical.solve.converged && out.thermal.converged) {
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

}  // namespace pcbcore::coupling
