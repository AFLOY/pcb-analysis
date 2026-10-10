// Bindings of the prepared FEM systems (``electrical._pcbcore.fem``).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <tuple>
#include <vector>

#include "arrays.hpp"
#include "pcbcore/errors.hpp"
#include "pcbcore/fem/dc_ports.hpp"
#include "pcbcore/fem/layered_dc_system.hpp"

namespace pcbcore_bindings {

namespace {

using pcbcore::fem::LayeredDCSystem;
using System = std::shared_ptr<LayeredDCSystem>;

template <typename T>
const T* sized(const Input<T>& array, const py::ssize_t expected, const char* const name) {
    if (array.size() != expected) {
        throw pcbcore::InvalidInput(std::string(name) + " has the wrong size");
    }
    return array.data();
}

py::array_t<bool> mask_array(const std::vector<std::uint8_t>& values, std::vector<py::ssize_t> shape) {
    py::array_t<bool> out(std::move(shape));
    std::memcpy(out.mutable_data(), values.data(), values.size());
    return out;
}

pcbcore::fem::NodeGroupsView groups_view(const Input<std::int64_t>& offsets, const Input<std::int64_t>& nodes) {
    if (offsets.ndim() != 1 || nodes.ndim() != 1 || offsets.size() < 1) {
        throw pcbcore::InvalidInput("node groups need 1-D offsets (groups + 1) and nodes");
    }
    const std::int64_t count = offsets.size() - 1;
    if (offsets.data()[count] != nodes.size()) {
        throw pcbcore::InvalidInput("the last group offset must equal the node count");
    }
    return {count, offsets.data(), nodes.data()};
}

System make_system(const Input<std::uint8_t>& element_active, const Input<double>& layer_thickness,
                   const Input<double>& pitch_x, const Input<double>& pitch_y, const Input<double>& conductivity,
                   const Input<std::int64_t>& via_lower, const Input<std::int64_t>& via_upper,
                   const Input<double>& via_conductance, const std::int64_t reference_node,
                   const Input<std::int64_t>& dirichlet_nodes, const bool two_level, const int block,
                   const int threads) {
    if (element_active.ndim() != 3) {
        throw pcbcore::InvalidInput("element_active must be (layers, rows, cols)");
    }
    pcbcore::fem::LayeredDCMeshView mesh;
    mesh.layers = static_cast<int>(element_active.shape(0));
    mesh.rows = static_cast<int>(element_active.shape(1));
    mesh.cols = static_cast<int>(element_active.shape(2));
    mesh.element_active = element_active.data();
    mesh.layer_thickness_m = sized(layer_thickness, mesh.layers, "layer_thickness");
    mesh.pitch_x_m = sized(pitch_x, mesh.cols, "pitch_x");
    mesh.pitch_y_m = sized(pitch_y, mesh.rows, "pitch_y");
    mesh.conductivity_s_per_m = sized(conductivity, element_active.size(), "conductivity");
    pcbcore::fem::ViaLinksView vias;
    vias.count = via_lower.size();
    vias.lower = via_lower.data();
    vias.upper = sized(via_upper, vias.count, "via_upper");
    vias.conductance_s = sized(via_conductance, vias.count, "via_conductance");
    py::gil_scoped_release release;
    return std::make_shared<LayeredDCSystem>(mesh, vias, reference_node, dirichlet_nodes.data(),
                                             dirichlet_nodes.size(), two_level, block, threads);
}

template <typename T, typename Action>
py::array_t<T> node_action(const LayeredDCSystem& system, const Input<T>& vector, Action action) {
    const T* x = sized(vector, system.size(), "vector");
    std::vector<T> y(static_cast<std::size_t>(system.size()));
    {
        py::gil_scoped_release release;
        action(x, y.data());
    }
    return to_array(std::move(y));
}

pcbcore::fem::MpirConfig mpir_config(const double relative_tolerance, const double absolute_tolerance,
                                     const double inner_relative_tolerance, const int max_outer_iterations,
                                     const int max_inner_iterations) {
    pcbcore::fem::MpirConfig config;
    config.relative_tolerance = relative_tolerance;
    config.absolute_tolerance = absolute_tolerance;
    config.inner_relative_tolerance = inner_relative_tolerance;
    config.max_outer_iterations = max_outer_iterations;
    config.max_inner_iterations = max_inner_iterations;
    return config;
}

py::dict result_dict(const pcbcore::fem::MpirResult& result) {
    std::vector<std::tuple<int, double, int, double>> history;
    for (const auto& step : result.history) {
        history.emplace_back(step.outer_iteration, step.high_relative_residual, step.inner_iterations,
                             step.inner_relative_residual);
    }
    py::dict out;
    out["converged"] = result.converged;
    out["outer_iterations"] = result.outer_iterations;
    out["inner_iterations"] = result.inner_iterations;
    out["relative_residual"] = result.relative_residual;
    out["high_operator_applications"] = result.high_operator_applications;
    out["low_operator_applications"] = result.low_operator_applications;
    out["history"] = history;
    return out;
}

py::dict solve(const LayeredDCSystem& system, const Input<double>& rhs, const py::object& initial_guess,
               const double relative_tolerance, const double absolute_tolerance,
               const double inner_relative_tolerance, const int max_outer_iterations,
               const int max_inner_iterations, const int threads) {
    const double* b = sized(rhs, system.size(), "rhs");
    std::vector<double> x(static_cast<std::size_t>(system.size()), 0.0);
    if (!initial_guess.is_none()) {
        const auto guess = initial_guess.cast<Input<double>>();
        const double* g = sized(guess, system.size(), "initial_guess");
        std::copy(g, g + system.size(), x.begin());
    }
    const auto config = mpir_config(relative_tolerance, absolute_tolerance, inner_relative_tolerance,
                                    max_outer_iterations, max_inner_iterations);
    pcbcore::fem::MpirResult result;
    {
        py::gil_scoped_release release;
        result = system.solve(b, x.data(), config, threads);
    }
    py::dict out = result_dict(result);
    out["solution"] = to_array(std::move(x));
    return out;
}

std::vector<py::ssize_t> element_shape(const LayeredDCSystem& s) { return {s.layers(), s.rows(), s.cols()}; }
std::vector<py::ssize_t> node_shape(const LayeredDCSystem& s) { return {s.layers(), s.rows() + 1, s.cols() + 1}; }

py::dict port_basis(const LayeredDCSystem& system, const Input<std::int64_t>& offsets,
                    const Input<std::int64_t>& nodes, const std::int64_t reference, const py::object& initial,
                    const double relative_tolerance, const double absolute_tolerance,
                    const double inner_relative_tolerance, const int max_outer_iterations,
                    const int max_inner_iterations, const int width, const int team) {
    const auto ports = groups_view(offsets, nodes);
    const auto config = mpir_config(relative_tolerance, absolute_tolerance, inner_relative_tolerance,
                                    max_outer_iterations, max_inner_iterations);
    Input<double> previous;
    const double* initial_data = nullptr;
    if (!initial.is_none()) {
        previous = initial.cast<Input<double>>();
        initial_data = sized(previous, (ports.count - 1) * system.size(), "initial");
    }
    pcbcore::fem::DCPortBasisResult result;
    {
        py::gil_scoped_release release;
        result = pcbcore::fem::dc_port_basis(system, ports, reference, initial_data, config, width, team);
    }
    const auto n = static_cast<py::ssize_t>(result.ports);
    auto fields = node_shape(system);
    fields.insert(fields.begin(), n - 1);
    py::list solves;
    for (const auto& solve : result.solves) {
        solves.append(result_dict(solve));
    }
    py::dict out;
    out["conductance"] = to_array(std::move(result.conductance), {n, n});
    out["unit_voltage"] = to_array(std::move(result.unit_voltage), fields);
    out["unit_current"] = to_array(std::move(result.unit_current), fields);
    out["solves"] = solves;
    return out;
}

pcbcore::fem::CorrelationModes modes_of(const LayeredDCSystem& system, const Input<double>& reduced,
                                        const double scale, const Input<double>& unit_current) {
    const auto driven = static_cast<std::int64_t>(reduced.ndim() == 2 ? reduced.shape(0) : -1);
    if (driven < 0 || reduced.shape(1) != driven) {
        throw pcbcore::InvalidInput("the reduced correlation must be square");
    }
    const double* fields = sized(unit_current, driven * system.size(), "unit_current");
    return pcbcore::fem::correlation_modes(reduced.data(), driven, scale, fields, system.size());
}

}  // namespace

void register_fem(py::module_& m) {
    py::class_<LayeredDCSystem, System>(m, "LayeredDCSystem")
        .def(py::init(&make_system), py::arg("element_active"), py::arg("layer_thickness"), py::arg("pitch_x"),
             py::arg("pitch_y"), py::arg("conductivity"), py::arg("via_lower"), py::arg("via_upper"),
             py::arg("via_conductance"), py::arg("reference_node"), py::arg("dirichlet_nodes"),
             py::arg("two_level"), py::arg("block"), py::arg("threads"))
        .def_property_readonly("size", &LayeredDCSystem::size)
        .def_property_readonly("block", &LayeredDCSystem::block)
        .def_property_readonly("two_level", &LayeredDCSystem::two_level)
        .def_property_readonly("low_bytes", &LayeredDCSystem::low_bytes)
        .def_property_readonly("active_nodes",
                               [](const LayeredDCSystem& s) { return mask_array(s.active_nodes(), node_shape(s)); })
        .def_property_readonly("free_nodes",
                               [](const LayeredDCSystem& s) { return mask_array(s.free_nodes(), node_shape(s)); })
        .def_property_readonly("coefficients",
                               [](const LayeredDCSystem& s) {
                                   auto shape = element_shape(s);
                                   shape.insert(shape.begin(), 2);
                                   return copy_array(s.coefficients(), shape);
                               })
        .def_property_readonly("unit", [](const LayeredDCSystem& s) { return copy_array(s.unit(), {2, 4, 4}); })
        .def_property_readonly("diagonal", [](const LayeredDCSystem& s) { return copy_array(s.diagonal(), {s.size()}); })
        .def_property_readonly("coarse_matrix",
                               [](const LayeredDCSystem& s) {
                                   const auto n = static_cast<py::ssize_t>(s.coarse().size);
                                   return copy_array(s.coarse().matrix, {n, n});
                               })
        .def_property_readonly("coarse_inverse",
                               [](const LayeredDCSystem& s) {
                                   const auto n = static_cast<py::ssize_t>(s.coarse().size);
                                   return copy_array(s.coarse().inverse, {n, n});
                               })
        .def(
            "apply_high",
            [](const LayeredDCSystem& s, const Input<double>& x, const int threads) {
                return node_action(s, x, [&](const double* in, double* out) { s.apply_high(in, out, threads); });
            },
            py::arg("vector"), py::arg("threads"))
        .def(
            "apply_low",
            [](const LayeredDCSystem& s, const Input<float>& x, const int threads) {
                return node_action(s, x, [&](const float* in, float* out) { s.apply_low(in, out, threads); });
            },
            py::arg("vector"), py::arg("threads"))
        .def(
            "apply_full",
            [](const LayeredDCSystem& s, const Input<double>& x, const int threads) {
                return node_action(s, x, [&](const double* in, double* out) { s.apply_full(in, out, threads); });
            },
            py::arg("vector"), py::arg("threads"))
        .def(
            "dirichlet_potential",
            [](const LayeredDCSystem& s, const Input<std::int64_t>& offsets, const Input<std::int64_t>& nodes,
               const Input<double>& voltage) {
                const auto groups = groups_view(offsets, nodes);
                const double* v = sized(voltage, groups.count, "voltage");
                return to_array(s.dirichlet_potential(groups, v));
            },
            py::arg("offsets"), py::arg("nodes"), py::arg("voltage"))
        .def(
            "build_rhs",
            [](const LayeredDCSystem& s, const Input<std::int64_t>& current_offsets,
               const Input<std::int64_t>& current_nodes, const Input<double>& current,
               const Input<std::int64_t>& voltage_offsets, const Input<std::int64_t>& voltage_nodes,
               const Input<double>& voltage, const int threads) {
                const auto currents = groups_view(current_offsets, current_nodes);
                const auto voltages = groups_view(voltage_offsets, voltage_nodes);
                const double* i = sized(current, currents.count, "current");
                const double* v = sized(voltage, voltages.count, "voltage");
                std::vector<double> rhs;
                {
                    py::gil_scoped_release release;
                    rhs = s.build_rhs(currents, i, voltages, v, threads);
                }
                return to_array(std::move(rhs));
            },
            py::arg("current_offsets"), py::arg("current_nodes"), py::arg("current"), py::arg("voltage_offsets"),
            py::arg("voltage_nodes"), py::arg("voltage"), py::arg("threads"))
        .def(
            "group_currents",
            [](const LayeredDCSystem& s, const Input<double>& potential, const Input<std::int64_t>& offsets,
               const Input<std::int64_t>& nodes, const int threads) {
                const auto groups = groups_view(offsets, nodes);
                const double* p = sized(potential, s.size(), "potential");
                std::vector<double> out;
                {
                    py::gil_scoped_release release;
                    out = s.group_currents(p, groups, threads);
                }
                return to_array(std::move(out));
            },
            py::arg("potential"), py::arg("offsets"), py::arg("nodes"), py::arg("threads"))
        .def("solve", &solve, py::arg("rhs"), py::arg("initial_guess"), py::arg("relative_tolerance"),
             py::arg("absolute_tolerance"), py::arg("inner_relative_tolerance"), py::arg("max_outer_iterations"),
             py::arg("max_inner_iterations"), py::arg("threads"))
        .def(
            "element_electric_field",
            [](const LayeredDCSystem& s, const Input<double>& potential, const int threads) {
                const double* p = sized(potential, s.size(), "potential");
                auto shape = element_shape(s);
                shape.push_back(2);
                return to_array(s.element_electric_field(p, threads), shape);
            },
            py::arg("potential"), py::arg("threads"))
        .def(
            "element_joule_loss",
            [](const LayeredDCSystem& s, const Input<double>& potential, const int threads) {
                const double* p = sized(potential, s.size(), "potential");
                return to_array(s.element_joule_loss(p, threads), element_shape(s));
            },
            py::arg("potential"), py::arg("threads"))
        .def(
            "via_current",
            [](const LayeredDCSystem& s, const Input<double>& potential) {
                return to_array(s.via_current(sized(potential, s.size(), "potential")));
            },
            py::arg("potential"))
        .def(
            "via_joule_loss",
            [](const LayeredDCSystem& s, const Input<double>& potential) {
                return to_array(s.via_joule_loss(sized(potential, s.size(), "potential")));
            },
            py::arg("potential"))
        .def(
            "post_process",
            [](const LayeredDCSystem& s, const Input<double>& potential, const int threads) {
                const double* p = sized(potential, s.size(), "potential");
                pcbcore::fem::LayeredDCPost post;
                {
                    py::gil_scoped_release release;
                    post = s.post_process(p, threads);
                }
                auto vector_shape = element_shape(s);
                vector_shape.push_back(2);
                py::dict out;
                out["electric_field"] = to_array(std::move(post.electric_field), vector_shape);
                out["current_density"] = to_array(std::move(post.current_density), vector_shape);
                out["max_current_density"] = post.max_current_density;
                out["element_joule_loss"] = to_array(std::move(post.element_joule_loss), element_shape(s));
                out["via_current"] = to_array(std::move(post.via_current));
                out["via_joule_loss"] = to_array(std::move(post.via_joule_loss));
                out["joule_loss"] = post.joule_loss;
                return out;
            },
            py::arg("potential"), py::arg("threads"));
    m.def("dc_port_basis", &port_basis, py::arg("system"), py::arg("offsets"), py::arg("nodes"),
          py::arg("reference"), py::arg("initial"), py::arg("relative_tolerance"), py::arg("absolute_tolerance"),
          py::arg("inner_relative_tolerance"), py::arg("max_outer_iterations"), py::arg("max_inner_iterations"),
          py::arg("width"), py::arg("team"));
    m.def(
        "port_modal_loss",
        [](const LayeredDCSystem& system, const Input<double>& reduced, const double scale,
           const Input<double>& unit_current, const int threads) {
            pcbcore::fem::ModalLoss loss;
            {
                py::gil_scoped_release release;
                loss = pcbcore::fem::modal_loss(system, modes_of(system, reduced, scale, unit_current), threads);
            }
            return py::make_tuple(to_array(std::move(loss.element), element_shape(system)),
                                  to_array(std::move(loss.via)));
        },
        py::arg("system"), py::arg("reduced"), py::arg("scale"), py::arg("unit_current"), py::arg("threads"));
    m.def(
        "port_rms_current_density",
        [](const LayeredDCSystem& system, const Input<double>& reduced, const double scale,
           const Input<double>& unit_current, const int threads) {
            std::vector<double> rms;
            {
                py::gil_scoped_release release;
                rms = pcbcore::fem::modal_rms_current_density(system, modes_of(system, reduced, scale, unit_current),
                                                              threads);
            }
            return to_array(std::move(rms), element_shape(system));
        },
        py::arg("system"), py::arg("reduced"), py::arg("scale"), py::arg("unit_current"), py::arg("threads"));
    m.def("choose_block_size", &pcbcore::fem::choose_block_size, py::arg("layers"), py::arg("node_rows"),
          py::arg("node_cols"), py::arg("max_coarse_size") = pcbcore::fem::kDefaultMaxCoarseSize,
          py::arg("minimum") = 4);
}

}  // namespace pcbcore_bindings
