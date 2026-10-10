// Bindings of the coupling loops (``electrical._pcbcore.coupling``).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "arrays.hpp"
#include "mpir_convert.hpp"
#include "pcbcore/coupling/coupling.hpp"
#include "pcbcore/errors.hpp"
#include "thermal_convert.hpp"

namespace pcbcore_bindings {

namespace {

using pcbcore::coupling::DCProblem;
using pcbcore::coupling::Index;

template <typename T>
std::vector<T> vector_of(const Input<T>& array) {
    return std::vector<T>(array.data(), array.data() + array.size());
}

std::shared_ptr<DCProblem> make_dc_problem(
    const Input<std::uint8_t>& active, const Input<double>& thickness, const Input<double>& pitch_x,
    const Input<double>& pitch_y, const Input<double>& conductivity, const Input<std::int64_t>& via_lower,
    const Input<std::int64_t>& via_upper, const Input<double>& via_resistance, const std::int64_t reference_node,
    const Input<std::int64_t>& current_offsets, const Input<std::int64_t>& current_nodes, const Input<double>& current,
    const Input<std::int64_t>& voltage_offsets, const Input<std::int64_t>& voltage_nodes, const Input<double>& voltage,
    const bool two_level, const int block, const double relative_tolerance, const double absolute_tolerance,
    const double inner_relative_tolerance, const int max_outer_iterations, const int max_inner_iterations) {
    if (active.ndim() != 3) {
        throw pcbcore::InvalidInput("element_active must be (layers, rows, cols)");
    }
    auto p = std::make_shared<DCProblem>();
    p->layers = static_cast<int>(active.shape(0));
    p->rows = static_cast<int>(active.shape(1));
    p->cols = static_cast<int>(active.shape(2));
    const py::ssize_t elements = active.size();
    p->active = vector_of(active);
    p->thickness = std::vector<double>(sized(thickness, p->layers, "thickness"), thickness.data() + p->layers);
    p->pitch_x = std::vector<double>(sized(pitch_x, p->cols, "pitch_x"), pitch_x.data() + p->cols);
    p->pitch_y = std::vector<double>(sized(pitch_y, p->rows, "pitch_y"), pitch_y.data() + p->rows);
    p->conductivity = std::vector<double>(sized(conductivity, elements, "conductivity"), conductivity.data() + elements);
    const py::ssize_t vias = via_lower.size();
    p->via_lower = vector_of(via_lower);
    p->via_upper = std::vector<Index>(sized(via_upper, vias, "via_upper"), via_upper.data() + vias);
    p->via_resistance = std::vector<double>(sized(via_resistance, vias, "via_resistance"), via_resistance.data() + vias);
    p->reference_node = reference_node;
    p->current_offsets = vector_of(current_offsets);
    p->current_nodes = vector_of(current_nodes);
    p->current = vector_of(current);
    p->voltage_offsets = vector_of(voltage_offsets);
    p->voltage_nodes = vector_of(voltage_nodes);
    p->voltage = vector_of(voltage);
    if (p->current_offsets.size() != p->current.size() + 1 || p->voltage_offsets.size() != p->voltage.size() + 1) {
        throw pcbcore::InvalidInput("terminal offsets must hold one entry more than the terminals");
    }
    p->two_level = two_level;
    p->block = block;
    p->config = mpir_config(relative_tolerance, absolute_tolerance, inner_relative_tolerance, max_outer_iterations,
                            max_inner_iterations);
    return p;
}

py::dict dc_solution_dict(pcbcore::coupling::DCSolution&& s, const DCProblem& p) {
    const std::vector<py::ssize_t> elements{p.layers, p.rows, p.cols};
    py::dict out;
    out["potential"] = to_array(std::move(s.potential), {p.layers, p.rows + 1, p.cols + 1});
    out["current_density"] = to_array(std::move(s.post.current_density), {p.layers, p.rows, p.cols, 2});
    out["via_current"] = to_array(std::move(s.post.via_current));
    out["joule_loss"] = s.post.joule_loss;
    out["element_joule_loss"] = to_array(std::move(s.post.element_joule_loss), elements);
    out["via_joule_loss"] = to_array(std::move(s.post.via_joule_loss));
    out["max_current_density"] = s.post.max_current_density;
    out["terminal_current"] = to_array(std::move(s.terminal_current));
    py::dict solve = result_dict(s.solve);
    solve["solution"] = to_array(std::move(s.solution));
    out["solve"] = solve;
    return out;
}

pcbcore::coupling::ExtraHeat extra_heat(const pcbcore::thermal::ThermalMesh& mesh, const py::object& element_heat,
                                        const Input<double>& nodal_heat, const Input<std::int64_t>& offsets,
                                        const Input<std::int64_t>& nodes, const Input<double>& power) {
    pcbcore::coupling::ExtraHeat extra;
    if (!element_heat.is_none()) {
        const auto heat = element_heat.cast<Input<double>>();
        extra.element_heat = std::vector<double>(sized(heat, mesh.elements(), "extra_element_heat_w"),
                                                 heat.data() + mesh.elements());
    }
    extra.nodal_heat = std::vector<double>(sized(nodal_heat, mesh.nodes(), "nodal_heat"), nodal_heat.data() + mesh.nodes());
    extra.source_offsets = vector_of(offsets);
    extra.source_nodes = vector_of(nodes);
    extra.source_power = vector_of(power);
    if (extra.source_offsets.size() != extra.source_power.size() + 1) {
        throw pcbcore::InvalidInput("heat source offsets must hold one entry more than the sources");
    }
    return extra;
}

py::list history_list(const std::vector<pcbcore::coupling::CouplingStep>& steps) {
    py::list history;
    for (const auto& step : steps) {
        py::dict record;
        record["iteration"] = step.iteration;
        record["joule_loss_w"] = step.joule_loss;
        record["max_temperature_k"] = step.max_temperature;
        record["temperature_change_k"] = step.temperature_change;
        record["relative_loss_change"] = step.relative_loss_change;
        record["relaxation"] = step.relaxation;
        record["electrical_inner_iterations"] = step.electrical_inner_iterations;
        record["thermal_inner_iterations"] = step.thermal_inner_iterations;
        history.append(record);
    }
    return history;
}

pcbcore::thermal::SolveOptions thermal_options(const double relative_tolerance, const double absolute_tolerance,
                                               const double inner_relative_tolerance, const int max_outer_iterations,
                                               const int max_inner_iterations, const bool two_level, const int block,
                                               const int radiation_max_iterations, const double radiation_tolerance,
                                               const int threads) {
    pcbcore::thermal::SolveOptions options;
    options.config = mpir_config(relative_tolerance, absolute_tolerance, inner_relative_tolerance,
                                 max_outer_iterations, max_inner_iterations);
    options.two_level = two_level;
    options.block = block;
    options.radiation_max_iterations = radiation_max_iterations;
    options.radiation_tolerance = radiation_tolerance;
    options.threads = threads;
    return options;
}

// (problem, reference port, layer slabs, pool width, team)
using ConductorTuple = std::tuple<std::shared_ptr<DCProblem>, std::int64_t, std::vector<int>, int, int>;

}  // namespace

void register_coupling(py::module_& m) {
    py::class_<DCProblem, std::shared_ptr<DCProblem>>(m, "DCProblem")
        .def(py::init(&make_dc_problem), py::arg("element_active"), py::arg("thickness"), py::arg("pitch_x"),
             py::arg("pitch_y"), py::arg("conductivity"), py::arg("via_lower"), py::arg("via_upper"),
             py::arg("via_resistance"), py::arg("reference_node"), py::arg("current_offsets"),
             py::arg("current_nodes"), py::arg("current"), py::arg("voltage_offsets"), py::arg("voltage_nodes"),
             py::arg("voltage"), py::arg("two_level"), py::arg("block"), py::arg("relative_tolerance"),
             py::arg("absolute_tolerance"), py::arg("inner_relative_tolerance"), py::arg("max_outer_iterations"),
             py::arg("max_inner_iterations"));

    m.def(
        "solve_dc",
        [](const DCProblem& problem, const py::object& initial, const int threads) {
            Input<double> holder;
            const double* guess = nullptr;
            if (!initial.is_none()) {
                holder = initial.cast<Input<double>>();
                guess = sized(holder, problem.nodes(), "initial_potential_v");
            }
            pcbcore::coupling::DCSolution solution;
            {
                py::gil_scoped_release release;
                solution = pcbcore::coupling::solve_dc(problem, problem.conductivity, problem.via_resistance, guess,
                                                       threads);
            }
            return dc_solution_dict(std::move(solution), problem);
        },
        py::arg("problem"), py::arg("initial"), py::arg("threads"));

    m.def(
        "run_electro_thermal",
        [](const DCProblem& problem, const std::vector<int>& layer_slabs, const double reference_temperature,
           const double coefficient, const std::shared_ptr<pcbcore::thermal::ThermalProblem>& board,
           const py::object& extra_element_heat, const Input<double>& nodal_heat,
           const Input<std::int64_t>& source_offsets, const Input<std::int64_t>& source_nodes,
           const Input<double>& source_power, const int max_iterations, const double temperature_tolerance,
           const double relative_loss_tolerance, const double relaxation, const bool aitken,
           const double max_relaxation, const double relative_tolerance, const double absolute_tolerance,
           const double inner_relative_tolerance, const int max_outer_iterations, const int max_inner_iterations,
           const bool two_level, const int block, const int radiation_max_iterations,
           const double radiation_tolerance, const int threads) {
            pcbcore::coupling::Placement placement{layer_slabs, reference_temperature, coefficient};
            const auto extra = extra_heat(board->mesh, extra_element_heat, nodal_heat, source_offsets, source_nodes,
                                          source_power);
            pcbcore::coupling::FixedPointConfig config{max_iterations,  temperature_tolerance, relative_loss_tolerance,
                                                       relaxation,      aitken,                max_relaxation};
            pcbcore::thermal::SolveOptions options;
            options.config = mpir_config(relative_tolerance, absolute_tolerance, inner_relative_tolerance,
                                         max_outer_iterations, max_inner_iterations);
            options.two_level = two_level;
            options.block = block;
            options.radiation_max_iterations = radiation_max_iterations;
            options.radiation_tolerance = radiation_tolerance;
            options.threads = threads;
            pcbcore::coupling::ElectroThermalResult result;
            {
                py::gil_scoped_release release;
                result = pcbcore::coupling::run_electro_thermal(problem, placement, *board, extra, config, options,
                                                                threads);
            }
            py::list history = history_list(result.history);
            const std::vector<py::ssize_t> elements{problem.layers, problem.rows, problem.cols};
            py::dict out;
            out["electrical"] = dc_solution_dict(std::move(result.electrical), problem);
            out["thermal"] = solution_dict(std::move(result.thermal), board->mesh);
            out["conductivity"] = to_array(std::move(result.conductivity), elements);
            out["via_resistance"] = to_array(std::move(result.via_resistance));
            out["element_temperature"] = to_array(std::move(result.element_temperature), elements);
            out["converged"] = result.converged;
            out["history"] = history;
            out["cold_loss"] = result.cold_loss;
            return out;
        },
        py::arg("problem"), py::arg("layer_slabs"), py::arg("reference_temperature"), py::arg("coefficient"),
        py::arg("board"), py::arg("extra_element_heat"), py::arg("nodal_heat"), py::arg("source_offsets"),
        py::arg("source_nodes"), py::arg("source_power"), py::arg("max_iterations"),
        py::arg("temperature_tolerance"), py::arg("relative_loss_tolerance"), py::arg("relaxation"),
        py::arg("aitken"), py::arg("max_relaxation"), py::arg("relative_tolerance"), py::arg("absolute_tolerance"),
        py::arg("inner_relative_tolerance"), py::arg("max_outer_iterations"), py::arg("max_inner_iterations"),
        py::arg("two_level"), py::arg("block"), py::arg("radiation_max_iterations"), py::arg("radiation_tolerance"),
        py::arg("threads"));

    m.def(
        "run_circuit_coupled",
        [](const std::vector<ConductorTuple>& conductor_tuples, const double reference_temperature,
           const double coefficient, const std::shared_ptr<pcbcore::thermal::ThermalProblem>& board,
           const py::object& extra_element_heat, const Input<double>& nodal_heat,
           const Input<std::int64_t>& source_offsets, const Input<std::int64_t>& source_nodes,
           const Input<double>& source_power, const py::function& circuit, const int max_iterations,
           const double temperature_tolerance, const double relative_loss_tolerance, const double relaxation,
           const bool aitken, const double max_relaxation, const double relative_tolerance,
           const double absolute_tolerance, const double inner_relative_tolerance, const int max_outer_iterations,
           const int max_inner_iterations, const bool two_level, const int block, const int radiation_max_iterations,
           const double radiation_tolerance, const int threads) {
            std::vector<pcbcore::coupling::PortConductor> conductors;
            for (const auto& [problem, reference, slabs, width, team] : conductor_tuples) {
                conductors.push_back({problem.get(), reference, {slabs, reference_temperature, coefficient}, width,
                                      team});
            }
            const auto extra = extra_heat(board->mesh, extra_element_heat, nodal_heat, source_offsets, source_nodes,
                                          source_power);
            pcbcore::coupling::FixedPointConfig config{max_iterations,  temperature_tolerance, relative_loss_tolerance,
                                                       relaxation,      aitken,                max_relaxation};
            const auto options = thermal_options(relative_tolerance, absolute_tolerance, inner_relative_tolerance,
                                                 max_outer_iterations, max_inner_iterations, two_level, block,
                                                 radiation_max_iterations, radiation_tolerance, threads);
            // The circuit runs in Python, with the GIL, once per iteration.
            const pcbcore::coupling::PortCircuit port_circuit =
                [&circuit](const std::vector<std::vector<double>>& conductances) {
                    py::gil_scoped_acquire acquire;
                    py::list matrices;
                    for (const auto& conductance : conductances) {
                        const auto n = static_cast<py::ssize_t>(std::llround(std::sqrt(conductance.size())));
                        matrices.append(copy_array(conductance, {n, n}));
                    }
                    const py::object answer = circuit(matrices);
                    std::vector<pcbcore::coupling::ReducedCorrelation> out;
                    for (const py::handle item : answer) {
                        const auto pair = item.cast<py::tuple>();
                        const auto reduced = pair[0].cast<Input<double>>();
                        out.push_back({vector_of(reduced), pair[1].cast<double>()});
                    }
                    return out;
                };
            pcbcore::coupling::CircuitCoupledResult result;
            {
                py::gil_scoped_release release;
                result = pcbcore::coupling::run_circuit_coupled(conductors, *board, extra, config, options,
                                                                port_circuit, threads);
            }
            py::list states;
            for (std::size_t c = 0; c < result.conductors.size(); ++c) {
                auto& state = result.conductors[c];
                const DCProblem& p = *std::get<0>(conductor_tuples[c]);
                const std::vector<py::ssize_t> elements{p.layers, p.rows, p.cols};
                const auto n = static_cast<py::ssize_t>(state.basis.ports);
                const std::vector<py::ssize_t> fields{n - 1, p.layers, p.rows + 1, p.cols + 1};
                py::list solves;
                for (const auto& solve : state.basis.solves) {
                    solves.append(result_dict(solve));
                }
                py::dict record;
                record["system"] = py::cast(state.system);
                record["conductance"] = to_array(std::move(state.basis.conductance), {n, n});
                record["unit_voltage"] = to_array(std::move(state.basis.unit_voltage), fields);
                record["unit_current"] = to_array(std::move(state.basis.unit_current), fields);
                record["solves"] = solves;
                record["conductivity"] = to_array(std::move(state.conductivity), elements);
                record["via_resistance"] = to_array(std::move(state.via_resistance));
                record["element_loss"] = to_array(std::move(state.element_loss), elements);
                record["via_loss"] = to_array(std::move(state.via_loss));
                record["rms_current_density"] = to_array(std::move(state.rms_current_density), elements);
                record["element_temperature"] = to_array(std::move(state.element_temperature), elements);
                states.append(record);
            }
            py::dict out;
            out["conductors"] = states;
            out["thermal"] = solution_dict(std::move(result.thermal), board->mesh);
            out["converged"] = result.converged;
            out["history"] = history_list(result.history);
            out["cold_loss"] = result.cold_loss;
            return out;
        },
        py::arg("conductors"), py::arg("reference_temperature"), py::arg("coefficient"), py::arg("board"),
        py::arg("extra_element_heat"), py::arg("nodal_heat"), py::arg("source_offsets"), py::arg("source_nodes"),
        py::arg("source_power"), py::arg("circuit"), py::arg("max_iterations"), py::arg("temperature_tolerance"),
        py::arg("relative_loss_tolerance"), py::arg("relaxation"), py::arg("aitken"), py::arg("max_relaxation"),
        py::arg("relative_tolerance"), py::arg("absolute_tolerance"), py::arg("inner_relative_tolerance"),
        py::arg("max_outer_iterations"), py::arg("max_inner_iterations"), py::arg("two_level"), py::arg("block"),
        py::arg("radiation_max_iterations"), py::arg("radiation_tolerance"), py::arg("threads"));
}

}  // namespace pcbcore_bindings
