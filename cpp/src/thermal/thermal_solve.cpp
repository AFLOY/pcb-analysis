// The linear solve with its heat budget, the radiation Newton loop and the
// backward-Euler march (thermal/matrix_free_mpir_fem/{solve,transient}.py).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>

#include "pcbcore/errors.hpp"
#include "pcbcore/lane_sum.hpp"
#include "pcbcore/thermal/thermal_system.hpp"

namespace pcbcore::thermal {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

[[nodiscard]] double sequential_sum(const std::vector<double>& values) noexcept {
    double total = 0.0;
    for (const double value : values) {
        total += value;
    }
    return total;
}

[[nodiscard]] fem::MpirResult merged(const fem::MpirResult& first, const fem::MpirResult& resumed) {
    fem::MpirResult out = resumed;
    out.outer_iterations = first.outer_iterations + resumed.outer_iterations;
    out.inner_iterations = first.inner_iterations + resumed.inner_iterations;
    out.high_operator_applications = first.high_operator_applications + resumed.high_operator_applications;
    out.low_operator_applications = first.low_operator_applications + resumed.low_operator_applications;
    out.history = first.history;
    out.history.insert(out.history.end(), resumed.history.begin(), resumed.history.end());
    return out;
}

[[nodiscard]] std::vector<LumpedRobin> lump_all(const ThermalMesh& mesh, const std::vector<FaceBoundary>& boundaries) {
    std::vector<LumpedRobin> out;
    out.reserve(boundaries.size());
    for (const FaceBoundary& boundary : boundaries) {
        out.push_back(lump(mesh, boundary));
    }
    return out;
}

}  // namespace

ThermalSolution solve_linear(const ThermalSystem& system, const SolveOptions& options, const double* const initial,
                             const double* const previous) {
    const Index n = system.size();
    const ThermalProblem& problem = system.problem();
    const int threads = std::max(1, options.threads);
    std::vector<double> previous_clean;
    const double* prev = nullptr;
    if (previous != nullptr) {
        previous_clean.assign(previous, previous + n);
        for (double& value : previous_clean) {
            if (!std::isfinite(value)) {
                value = 0.0;
            }
        }
        prev = previous_clean.data();
    }
    double reference = options.reference.has_value() ? *options.reference : system.default_reference();
    if (!std::isfinite(reference)) {
        throw InvalidInput("reference_temperature_k must be finite");
    }
    std::vector<double> rhs = system.build_rhs(reference, prev, threads);
    for (const double value : rhs) {
        if (!std::isfinite(value)) {
            throw InvalidInput("rhs must contain only finite values");
        }
    }
    const auto& free = system.free_nodes();
    const auto& active = system.active_nodes();
    const auto& fixed_temperature = system.fixed_temperature();
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    if (initial != nullptr) {
        for (Index i = 0; i < n; ++i) {
            const double start =
                free[i] != 0U ? initial[i] : fixed_temperature[i] + reference * (active[i] != 0U ? 0.0 : 1.0);
            x[i] = start - reference;
        }
    }
    fem::MpirResult result = system.solve(rhs.data(), x.data(), options.config, threads);
    Index free_count = 0;
    for (Index i = 0; i < n; ++i) {
        free_count += free[i] != 0U ? 1 : 0;
    }
    if (!result.converged && free_count > 0) {
        // The FP64 residual of K theta floors at eps ||K|| ||theta||; a plate
        // far above the reference, almost uniformly, hits that floor first.
        // Re-reference to the mean rise and continue from the iterate.
        std::vector<double> free_values;
        free_values.reserve(static_cast<std::size_t>(free_count));
        for (Index i = 0; i < n; ++i) {
            if (free[i] != 0U) {
                free_values.push_back(x[i]);
            }
        }
        const double shift = lane_sum(free_count, [&](const std::ptrdiff_t i) { return free_values[i]; }) /
                             static_cast<double>(free_count);
        if (std::abs(shift) > 0.0) {
            reference += shift;
            rhs = system.build_rhs(reference, prev, threads);
            for (double& value : x) {
                value -= shift;
            }
            const fem::MpirResult resumed = system.solve(rhs.data(), x.data(), options.config, threads);
            result = merged(result, resumed);
        }
    }

    ThermalSolution out;
    out.rise = x;
    out.temperature.assign(static_cast<std::size_t>(n), 0.0);
    for (Index i = 0; i < n; ++i) {
        out.temperature[i] = x[i] + reference;
    }
    const double* const t = out.temperature.data();
    const std::vector<double> residual = system.unconstrained_residual(t, prev, threads);
    std::vector<double> fixed_residual;
    for (Index i = 0; i < n; ++i) {
        if (free[i] == 0U) {
            fixed_residual.push_back(residual[i]);
        }
    }
    out.convective_heat = system.convective_heat(t);
    out.fixed_temperature_heat =
        -lane_sum(static_cast<std::ptrdiff_t>(fixed_residual.size()), [&](const std::ptrdiff_t i) { return fixed_residual[i]; });
    const std::vector<double> stored = system.stored_heat(t, prev);
    out.stored_heat = lane_sum(n, [&](const std::ptrdiff_t i) { return stored[i]; });
    out.total_heat_input = lane_sum(n, [&](const std::ptrdiff_t i) { return problem.load[i]; });
    out.heat_flux = system.element_heat_flux(t, threads);
    const ThermalMesh& mesh = problem.mesh;
    if (!mesh.full()) {
        for (Index e = 0; e < mesh.elements(); ++e) {
            if (mesh.active[e] == 0U) {
                out.heat_flux[3 * e] = out.heat_flux[3 * e + 1] = out.heat_flux[3 * e + 2] = 0.0;
            }
        }
        for (Index i = 0; i < n; ++i) {
            if (active[i] == 0U) {
                out.temperature[i] = kNaN;
            }
        }
    }
    double maximum = -std::numeric_limits<double>::infinity();
    double minimum = std::numeric_limits<double>::infinity();
    for (const double value : out.temperature) {
        if (!std::isnan(value)) {
            maximum = std::max(maximum, value);
            minimum = std::min(minimum, value);
        }
    }
    out.max_temperature = maximum;
    out.min_temperature = minimum;
    out.heat_balance_error =
        ((out.total_heat_input - sequential_sum(out.convective_heat)) - out.fixed_temperature_heat) - out.stored_heat;
    out.solve = std::move(result);
    return out;
}

ThermalSolution solve_steady(const ThermalProblem& problem, const SolveOptions& options, const double* const initial,
                             const double* const capacity_per_s, const double* const previous, SystemCache* const cache) {
    const ThermalMesh& mesh = problem.mesh;
    const Index n = mesh.nodes();
    const int threads = std::max(1, options.threads);
    if (problem.radiation.empty()) {
        std::shared_ptr<ThermalSystem> system;
        const bool transient = capacity_per_s != nullptr;
        if (cache != nullptr && cache->system != nullptr && cache->problem == &problem &&
            cache->transient == transient &&
            (!transient || (static_cast<Index>(cache->capacity.size()) == n &&
                            std::equal(cache->capacity.begin(), cache->capacity.end(), capacity_per_s)))) {
            system = cache->system;
        } else {
            system = std::make_shared<ThermalSystem>(problem, lump_all(mesh, problem.convection), capacity_per_s,
                                                     options.two_level, options.block, threads);
            if (cache != nullptr) {
                cache->problem = &problem;
                cache->transient = transient;
                cache->capacity.assign(capacity_per_s, capacity_per_s + (transient ? n : 0));
                cache->system = system;
            }
        }
        return solve_linear(*system, options, initial, previous);
    }
    if (options.radiation_max_iterations < 1) {
        throw InvalidInput("radiation_max_iterations must be positive");
    }
    if (!(options.radiation_tolerance > 0.0)) {
        throw InvalidInput("radiation_tolerance_k must be positive");
    }

    std::vector<double> current(static_cast<std::size_t>(n), 0.0);
    const double radiative_ambient = problem.radiation.front().mean_ambient;
    if (initial == nullptr) {
        const double start = problem.convection.empty() ? radiative_ambient : problem.convection.front().mean_ambient;
        std::fill(current.begin(), current.end(), start);
    } else {
        for (Index i = 0; i < n; ++i) {
            current[i] = std::isfinite(initial[i]) ? initial[i] : radiative_ambient;
        }
    }
    const std::vector<LumpedRobin> convection = lump_all(mesh, problem.convection);
    ThermalSolution solution;
    double change = std::numeric_limits<double>::infinity();
    bool converged = false;
    int iteration = 0;
    for (iteration = 1; iteration <= options.radiation_max_iterations; ++iteration) {
        std::vector<LumpedRobin> robin = convection;
        for (const FaceBoundary& boundary : problem.radiation) {
            robin.push_back(lump(mesh, linearize(mesh, boundary, current.data())));
        }
        const ThermalSystem system(problem, std::move(robin), capacity_per_s, options.two_level, options.block,
                                   threads);
        solution = solve_linear(system, options, current.data(), previous);
        change = 0.0;
        for (Index i = 0; i < n; ++i) {
            const double proposed = std::isfinite(solution.temperature[i]) ? solution.temperature[i] : current[i];
            change = std::max(change, std::abs(proposed - current[i]));
            current[i] = proposed;
        }
        if (change <= options.radiation_tolerance) {
            converged = true;
            break;
        }
    }
    iteration = std::min(iteration, options.radiation_max_iterations);
    const std::size_t count = problem.convection.size();
    solution.radiative_heat.assign(solution.convective_heat.begin() + static_cast<std::ptrdiff_t>(count),
                                   solution.convective_heat.end());
    solution.convective_heat.resize(count);
    solution.radiation_iterations = iteration;
    solution.radiation_converged = converged && solution.solve.converged;
    solution.radiation_change = change;
    return solution;
}

TransientSolution solve_transient(const ThermalProblem& problem, const SolveOptions& options,
                                  const std::vector<double>& times, const double* const initial,
                                  const double* const capacity, const bool store_all, const bool until_steady,
                                  const double steady_tolerance) {
    const ThermalMesh& mesh = problem.mesh;
    const Index n = mesh.nodes();
    if (times.size() < 2 || times.front() != 0.0) {
        throw InvalidInput("times_s must start at 0 and hold at least one step");
    }
    const std::vector<std::uint8_t> active = mesh.active_nodes();
    bool any_active = false;
    for (const std::uint8_t a : active) {
        any_active = any_active || a != 0U;
    }
    std::vector<double> current(initial, initial + n);
    const auto masked = [&](const std::vector<double>& field) {
        std::vector<double> out(field);
        for (Index i = 0; i < n; ++i) {
            if (active[i] == 0U) {
                out[i] = kNaN;
            }
        }
        return out;
    };
    TransientSolution out;
    std::vector<double> last = masked(current);
    out.temperature = last;
    out.times.push_back(0.0);
    std::vector<double> scaled(static_cast<std::size_t>(n), 0.0);
    SystemCache cache;
    for (std::size_t index = 1; index < times.size(); ++index) {
        const double step = times[index] - times[index - 1];
        for (Index i = 0; i < n; ++i) {
            scaled[i] = capacity[i] / step;
        }
        const std::vector<double> previous(current);
        out.final = solve_steady(problem, options, previous.data(), scaled.data(), previous.data(), &cache);
        double change = 0.0;
        for (Index i = 0; i < n; ++i) {
            const double proposed = std::isfinite(out.final.temperature[i]) ? out.final.temperature[i] : current[i];
            if (active[i] != 0U) {
                change = std::max(change, std::abs(proposed - current[i]));
            }
            current[i] = proposed;
        }
        if (!any_active) {
            change = 0.0;
        }
        last = masked(current);
        if (store_all) {
            out.times.push_back(times[index]);
            out.temperature.insert(out.temperature.end(), last.begin(), last.end());
        } else {
            out.times.assign(1, times[index]);
            out.temperature = last;
        }
        TransientStep record;
        record.index = static_cast<int>(index);
        record.time = times[index];
        record.step = step;
        double maximum = -std::numeric_limits<double>::infinity();
        for (const double value : last) {
            if (!std::isnan(value)) {
                maximum = std::max(maximum, value);
            }
        }
        record.max_temperature = maximum;
        record.max_change = change;
        record.stored_heat = out.final.stored_heat;
        record.convective_heat = sequential_sum(out.final.convective_heat);
        record.radiative_heat = sequential_sum(out.final.radiative_heat);
        record.heat_balance_error = out.final.heat_balance_error;
        record.inner_iterations = out.final.solve.inner_iterations;
        record.radiation_iterations = out.final.radiation_iterations;
        record.converged = out.final.solve.converged && out.final.radiation_converged;
        out.history.push_back(record);
        if (until_steady && change / step <= steady_tolerance) {
            out.reached_steady = true;
            break;
        }
    }
    return out;
}

}  // namespace pcbcore::thermal
