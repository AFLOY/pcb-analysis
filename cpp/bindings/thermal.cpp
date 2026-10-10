// Bindings of the thermal conduction core (``electrical._pcbcore.thermal``).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include "arrays.hpp"
#include "mpir_convert.hpp"
#include "thermal_convert.hpp"
#include "pcbcore/errors.hpp"
#include "pcbcore/thermal/thermal_system.hpp"

namespace pcbcore_bindings {

namespace {

using pcbcore::thermal::FaceBoundary;
using pcbcore::thermal::ThermalProblem;
using pcbcore::thermal::ThermalSolution;
using pcbcore::thermal::ThermalSystem;
using Problem = std::shared_ptr<ThermalProblem>;

// (kind, directions, coefficient (slabs, rows, cols), ambient (slabs, rows, cols), mean ambient)
using BoundaryTuple = std::tuple<int, std::vector<int>, Input<double>, Input<double>, double>;

std::vector<FaceBoundary> boundaries(const std::vector<BoundaryTuple>& tuples, const std::int64_t elements) {
    std::vector<FaceBoundary> out;
    for (const auto& [kind, directions, coefficient, ambient, mean] : tuples) {
        if (kind < 0 || kind > 2) {
            throw pcbcore::InvalidInput("boundary kind is 0 (bottom), 1 (top) or 2 (exposed faces)");
        }
        FaceBoundary boundary;
        boundary.kind = static_cast<pcbcore::thermal::FaceKind>(kind);
        boundary.directions = directions;
        const double* c = sized(coefficient, elements, "boundary coefficient");
        const double* a = sized(ambient, elements, "boundary ambient");
        boundary.coefficient.assign(c, c + elements);
        boundary.ambient.assign(a, a + elements);
        boundary.mean_ambient = mean;
        out.push_back(std::move(boundary));
    }
    return out;
}

Problem make_problem(const Input<std::uint8_t>& active, const Input<double>& thickness, const Input<double>& pitch_x,
                     const Input<double>& pitch_y, const Input<double>& k_in, const Input<double>& k_z,
                     const Input<std::uint8_t>& fixed_mask, const Input<double>& fixed_values,
                     const std::vector<BoundaryTuple>& convection, const std::vector<BoundaryTuple>& radiation,
                     const Input<double>& element_heat, const Input<double>& nodal_heat,
                     const Input<std::int64_t>& source_offsets, const Input<std::int64_t>& source_nodes,
                     const Input<double>& source_power) {
    if (active.ndim() != 3) {
        throw pcbcore::InvalidInput("active must be (slabs, rows, cols)");
    }
    auto problem = std::make_shared<ThermalProblem>();
    auto& mesh = problem->mesh;
    mesh.slabs = static_cast<int>(active.shape(0));
    mesh.rows = static_cast<int>(active.shape(1));
    mesh.cols = static_cast<int>(active.shape(2));
    const std::int64_t elements = mesh.elements();
    const std::int64_t nodes = mesh.nodes();
    const auto copy = [](const auto* data, const std::int64_t count) {
        return std::vector<std::remove_cv_t<std::remove_pointer_t<decltype(data)>>>(data, data + count);
    };
    mesh.active = copy(active.data(), elements);
    mesh.thickness = copy(sized(thickness, mesh.slabs, "thickness"), mesh.slabs);
    mesh.pitch_x = copy(sized(pitch_x, mesh.cols, "pitch_x"), mesh.cols);
    mesh.pitch_y = copy(sized(pitch_y, mesh.rows, "pitch_y"), mesh.rows);
    mesh.k_in = copy(sized(k_in, elements, "k_in"), elements);
    mesh.k_z = copy(sized(k_z, elements, "k_z"), elements);
    problem->fixed_mask = copy(sized(fixed_mask, nodes, "fixed_mask"), nodes);
    problem->fixed_values = copy(sized(fixed_values, nodes, "fixed_values"), nodes);
    problem->convection = boundaries(convection, elements);
    problem->radiation = boundaries(radiation, elements);
    const std::int64_t sources = source_power.size();
    if (source_offsets.size() != sources + 1 || source_offsets.data()[sources] != source_nodes.size()) {
        throw pcbcore::InvalidInput("heat sources need offsets (sources + 1) ending at the node count");
    }
    problem->load = pcbcore::thermal::nodal_load(mesh, sized(element_heat, elements, "element_heat"),
                                                 sized(nodal_heat, nodes, "nodal_heat"), sources,
                                                 source_offsets.data(), source_nodes.data(), source_power.data());
    return problem;
}

const double* optional_nodes(const py::object& value, Input<double>& holder, const std::int64_t nodes,
                             const char* name) {
    if (value.is_none()) {
        return nullptr;
    }
    holder = value.cast<Input<double>>();
    return sized(holder, nodes, name);
}

pcbcore::thermal::SolveOptions options(const double relative_tolerance, const double absolute_tolerance,
                                       const double inner_relative_tolerance, const int max_outer_iterations,
                                       const int max_inner_iterations, const bool two_level, const int block,
                                       const std::optional<double> reference, const int radiation_max_iterations,
                                       const double radiation_tolerance, const int threads) {
    pcbcore::thermal::SolveOptions out;
    out.config = mpir_config(relative_tolerance, absolute_tolerance, inner_relative_tolerance, max_outer_iterations,
                             max_inner_iterations);
    out.two_level = two_level;
    out.block = block;
    out.reference = reference;
    out.radiation_max_iterations = radiation_max_iterations;
    out.radiation_tolerance = radiation_tolerance;
    out.threads = threads;
    return out;
}

#define PCB_OPTION_ARGS                                                                                   \
    py::arg("relative_tolerance"), py::arg("absolute_tolerance"), py::arg("inner_relative_tolerance"),     \
        py::arg("max_outer_iterations"), py::arg("max_inner_iterations"), py::arg("two_level"),            \
        py::arg("block"), py::arg("reference"), py::arg("radiation_max_iterations"),                       \
        py::arg("radiation_tolerance"), py::arg("threads")

}  // namespace

void register_thermal(py::module_& m) {
    py::class_<ThermalProblem, Problem>(m, "ThermalProblem")
        .def(py::init(&make_problem), py::arg("active"), py::arg("thickness"), py::arg("pitch_x"),
             py::arg("pitch_y"), py::arg("k_in"), py::arg("k_z"), py::arg("fixed_mask"), py::arg("fixed_values"),
             py::arg("convection"), py::arg("radiation"), py::arg("element_heat"), py::arg("nodal_heat"),
             py::arg("source_offsets"), py::arg("source_nodes"), py::arg("source_power"))
        .def_property_readonly("load",
                               [](const ThermalProblem& p) { return copy_array(p.load, node_shape(p.mesh)); });

    py::class_<ThermalSystem, std::shared_ptr<ThermalSystem>>(m, "ThermalSystem")
        .def(py::init([](const Problem& problem, const py::object& capacity, const bool two_level, const int block,
                         const int threads) {
                 Input<double> holder;
                 const double* c = optional_nodes(capacity, holder, problem->mesh.nodes(), "capacity_per_s");
                 py::gil_scoped_release release;
                 std::vector<pcbcore::thermal::LumpedRobin> robin;
                 for (const auto& boundary : problem->convection) {
                     robin.push_back(pcbcore::thermal::lump(problem->mesh, boundary));
                 }
                 return std::make_shared<ThermalSystem>(*problem, std::move(robin), c, two_level, block, threads);
             }),
             py::arg("problem"), py::arg("capacity_per_s"), py::arg("two_level"), py::arg("block"),
             py::arg("threads"), py::keep_alive<1, 2>())
        .def_property_readonly("size", &ThermalSystem::size)
        .def_property_readonly("block", &ThermalSystem::block)
        .def_property_readonly("two_level", &ThermalSystem::two_level)
        .def_property_readonly("active_nodes",
                               [](const ThermalSystem& s) {
                                   return mask_array(s.active_nodes(), node_shape(s.problem().mesh));
                               })
        .def_property_readonly("free_nodes",
                               [](const ThermalSystem& s) {
                                   return mask_array(s.free_nodes(), node_shape(s.problem().mesh));
                               })
        .def_property_readonly("fixed_temperature",
                               [](const ThermalSystem& s) {
                                   return copy_array(s.fixed_temperature(), node_shape(s.problem().mesh));
                               })
        .def_property_readonly("coefficients",
                               [](const ThermalSystem& s) {
                                   const auto& m = s.problem().mesh;
                                   return copy_array(s.coefficients(), {3, m.slabs, m.rows, m.cols});
                               })
        .def_property_readonly("unit", [](const ThermalSystem& s) { return copy_array(s.unit(), {3, 8, 8}); })
        .def_property_readonly("robin_total",
                               [](const ThermalSystem& s) { return copy_array(s.robin_total(), {s.size()}); })
        .def_property_readonly("diagonal", [](const ThermalSystem& s) { return copy_array(s.diagonal(), {s.size()}); })
        .def_property_readonly("coarse_matrix",
                               [](const ThermalSystem& s) {
                                   const auto n = static_cast<py::ssize_t>(s.coarse().size);
                                   return copy_array(s.coarse().matrix, {n, n});
                               })
        .def_property_readonly("coarse_inverse",
                               [](const ThermalSystem& s) {
                                   const auto n = static_cast<py::ssize_t>(s.coarse().size);
                                   return copy_array(s.coarse().inverse, {n, n});
                               })
        .def(
            "apply_high",
            [](const ThermalSystem& s, const Input<double>& x, const int threads) {
                std::vector<double> y(static_cast<std::size_t>(s.size()));
                s.apply_high(sized(x, s.size(), "vector"), y.data(), threads);
                return to_array(std::move(y));
            },
            py::arg("vector"), py::arg("threads"))
        .def(
            "apply_low",
            [](const ThermalSystem& s, const Input<float>& x, const int threads) {
                std::vector<float> y(static_cast<std::size_t>(s.size()));
                s.apply_low(sized(x, s.size(), "vector"), y.data(), threads);
                return to_array(std::move(y));
            },
            py::arg("vector"), py::arg("threads"))
        .def("default_reference", &ThermalSystem::default_reference)
        .def(
            "build_rhs",
            [](const ThermalSystem& s, const double reference, const py::object& previous, const int threads) {
                Input<double> holder;
                const double* p = optional_nodes(previous, holder, s.size(), "previous_temperature_k");
                return to_array(s.build_rhs(reference, p, threads));
            },
            py::arg("reference"), py::arg("previous"), py::arg("threads"))
        .def(
            "solve",
            [](const ThermalSystem& s, const Input<double>& rhs, const py::object& initial,
               const double relative_tolerance, const double absolute_tolerance,
               const double inner_relative_tolerance, const int max_outer_iterations,
               const int max_inner_iterations, const int threads) {
                const double* b = sized(rhs, s.size(), "rhs");
                std::vector<double> x(static_cast<std::size_t>(s.size()), 0.0);
                Input<double> holder;
                if (const double* g = optional_nodes(initial, holder, s.size(), "initial_guess")) {
                    std::copy(g, g + s.size(), x.begin());
                }
                const auto config = mpir_config(relative_tolerance, absolute_tolerance, inner_relative_tolerance,
                                                max_outer_iterations, max_inner_iterations);
                pcbcore::fem::MpirResult result;
                {
                    py::gil_scoped_release release;
                    result = s.solve(b, x.data(), config, threads);
                }
                py::dict out = result_dict(result);
                out["solution"] = to_array(std::move(x));
                return out;
            },
            py::arg("rhs"), py::arg("initial_guess"), py::arg("relative_tolerance"), py::arg("absolute_tolerance"),
            py::arg("inner_relative_tolerance"), py::arg("max_outer_iterations"), py::arg("max_inner_iterations"),
            py::arg("threads"))
        .def(
            "stored_heat",
            [](const ThermalSystem& s, const Input<double>& temperature, const py::object& previous) {
                Input<double> holder;
                const double* p = optional_nodes(previous, holder, s.size(), "previous_temperature_k");
                return to_array(s.stored_heat(sized(temperature, s.size(), "temperature"), p));
            },
            py::arg("temperature"), py::arg("previous"))
        .def(
            "unconstrained_residual",
            [](const ThermalSystem& s, const Input<double>& temperature, const py::object& previous,
               const int threads) {
                Input<double> holder;
                const double* p = optional_nodes(previous, holder, s.size(), "previous_temperature_k");
                return to_array(s.unconstrained_residual(sized(temperature, s.size(), "temperature"), p, threads));
            },
            py::arg("temperature"), py::arg("previous"), py::arg("threads"))
        .def(
            "convective_heat",
            [](const ThermalSystem& s, const Input<double>& temperature) {
                return to_array(s.convective_heat(sized(temperature, s.size(), "temperature")));
            },
            py::arg("temperature"))
        .def(
            "element_heat_flux",
            [](const ThermalSystem& s, const Input<double>& temperature, const int threads) {
                const auto& m = s.problem().mesh;
                return to_array(s.element_heat_flux(sized(temperature, s.size(), "temperature"), threads),
                                {m.slabs, m.rows, m.cols, 3});
            },
            py::arg("temperature"), py::arg("threads"));

    m.def(
        "solve_steady",
        [](const Problem& problem, const py::object& initial, const py::object& capacity, const py::object& previous,
           const double relative_tolerance, const double absolute_tolerance, const double inner_relative_tolerance,
           const int max_outer_iterations, const int max_inner_iterations, const bool two_level, const int block,
           const std::optional<double> reference, const int radiation_max_iterations,
           const double radiation_tolerance, const int threads) {
            const std::int64_t nodes = problem->mesh.nodes();
            Input<double> initial_holder;
            Input<double> capacity_holder;
            Input<double> previous_holder;
            const double* i = optional_nodes(initial, initial_holder, nodes, "initial_temperature_k");
            const double* c = optional_nodes(capacity, capacity_holder, nodes, "capacity_per_s");
            const double* p = optional_nodes(previous, previous_holder, nodes, "previous_temperature_k");
            const auto o = options(relative_tolerance, absolute_tolerance, inner_relative_tolerance,
                                   max_outer_iterations, max_inner_iterations, two_level, block, reference,
                                   radiation_max_iterations, radiation_tolerance, threads);
            ThermalSolution solution;
            {
                py::gil_scoped_release release;
                solution = pcbcore::thermal::solve_steady(*problem, o, i, c, p);
            }
            return solution_dict(std::move(solution), problem->mesh);
        },
        py::arg("problem"), py::arg("initial"), py::arg("capacity_per_s"), py::arg("previous"), PCB_OPTION_ARGS);

    m.def(
        "solve_transient",
        [](const Problem& problem, const std::vector<double>& times, const Input<double>& initial,
           const Input<double>& capacity, const bool store_all, const bool until_steady,
           const double steady_tolerance, const double relative_tolerance, const double absolute_tolerance,
           const double inner_relative_tolerance, const int max_outer_iterations, const int max_inner_iterations,
           const bool two_level, const int block, const std::optional<double> reference,
           const int radiation_max_iterations, const double radiation_tolerance, const int threads) {
            const std::int64_t nodes = problem->mesh.nodes();
            const double* i = sized(initial, nodes, "initial_temperature_k");
            const double* c = sized(capacity, nodes, "capacity");
            const auto o = options(relative_tolerance, absolute_tolerance, inner_relative_tolerance,
                                   max_outer_iterations, max_inner_iterations, two_level, block, reference,
                                   radiation_max_iterations, radiation_tolerance, threads);
            pcbcore::thermal::TransientSolution run;
            {
                py::gil_scoped_release release;
                run = pcbcore::thermal::solve_transient(*problem, o, times, i, c, store_all, until_steady,
                                                        steady_tolerance);
            }
            auto shape = node_shape(problem->mesh);
            shape.insert(shape.begin(), static_cast<py::ssize_t>(run.times.size()));
            py::list history;
            for (const auto& step : run.history) {
                py::dict record;
                record["index"] = step.index;
                record["time_s"] = step.time;
                record["step_s"] = step.step;
                record["max_temperature_k"] = step.max_temperature;
                record["max_change_k"] = step.max_change;
                record["stored_heat_w"] = step.stored_heat;
                record["convective_heat_w"] = step.convective_heat;
                record["radiative_heat_w"] = step.radiative_heat;
                record["heat_balance_error_w"] = step.heat_balance_error;
                record["inner_iterations"] = step.inner_iterations;
                record["radiation_iterations"] = step.radiation_iterations;
                record["converged"] = step.converged;
                history.append(record);
            }
            py::dict out;
            out["times"] = to_array(std::move(run.times));
            out["temperature"] = to_array(std::move(run.temperature), shape);
            out["history"] = history;
            out["final"] = solution_dict(std::move(run.final), problem->mesh);
            out["reached_steady"] = run.reached_steady;
            return out;
        },
        py::arg("problem"), py::arg("times"), py::arg("initial"), py::arg("capacity"), py::arg("store_all"),
        py::arg("until_steady"), py::arg("steady_tolerance"), PCB_OPTION_ARGS);
}

}  // namespace pcbcore_bindings
