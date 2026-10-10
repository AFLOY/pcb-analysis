// The pcbcore extension module, ``electrical._pcbcore``.
//
// Bindings only convert: NumPy arrays in (viewed without a copy when their
// dtype and layout already match), the C++ call with the GIL released, NumPy
// arrays out (moved, not copied).  The Python facades wrap the results in the
// package's dataclasses.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "pcbcore/errors.hpp"
#include "pcbcore/network/dc_network.hpp"

namespace py = pybind11;

namespace {

template <typename T>
using Input = py::array_t<T, py::array::c_style | py::array::forcecast>;

// A NumPy array that owns ``values``: the vector moves into a capsule.
template <typename T>
py::array_t<T> to_array(std::vector<T>&& values, std::vector<py::ssize_t> shape) {
    auto* owner = new std::vector<T>(std::move(values));
    py::capsule release(owner, [](void* pointer) { delete static_cast<std::vector<T>*>(pointer); });
    return py::array_t<T>(std::move(shape), owner->data(), release);
}

template <typename T>
py::array_t<T> to_array(std::vector<T>&& values) {
    const auto size = static_cast<py::ssize_t>(values.size());
    return to_array(std::move(values), {size});
}

pcbcore::network::ConductanceNetworkView network_view(const std::int64_t node_count,
                                                      const Input<std::int64_t>& left,
                                                      const Input<std::int64_t>& right,
                                                      const Input<double>& conductance) {
    if (left.ndim() != 1 || right.ndim() != 1 || conductance.ndim() != 1 || left.size() != right.size() ||
        left.size() != conductance.size()) {
        throw pcbcore::InvalidInput("left, right and conductance must be 1-D and equally long");
    }
    return {node_count, static_cast<std::int64_t>(left.size()), left.data(), right.data(), conductance.data()};
}

py::dict solve_conductance_network(const std::int64_t node_count, const Input<std::int64_t>& left,
                                   const Input<std::int64_t>& right, const Input<double>& conductance,
                                   const std::int64_t reference, const Input<double>& injection,
                                   const Input<double>& weights) {
    const auto network = network_view(node_count, left, right, conductance);
    if (injection.ndim() != 1 || injection.size() != node_count) {
        throw pcbcore::InvalidInput("injection must have one entry per node");
    }
    if (weights.ndim() != 2 || weights.shape(1) != node_count) {
        throw pcbcore::InvalidInput("objective_weights must be objectives x node_count");
    }
    const std::int64_t objectives = weights.shape(0);
    pcbcore::network::DCNetworkResult result;
    {
        py::gil_scoped_release release;
        result = pcbcore::network::solve_conductance_network(network, reference, injection.data(),
                                                             objectives ? weights.data() : nullptr, objectives);
    }
    py::dict out;
    out["voltage_unknowns"] = to_array(std::move(result.voltage_unknowns));
    out["node_voltage"] = to_array(std::move(result.node_voltage));
    out["edge_current"] = to_array(std::move(result.edge_current));
    out["node_current"] = to_array(std::move(result.node_current));
    out["adjoint_voltage"] =
        to_array(std::move(result.adjoint_voltage), {static_cast<py::ssize_t>(objectives), static_cast<py::ssize_t>(node_count)});
    out["loss_w"] = result.loss_w;
    out["relative_residual"] = result.relative_residual;
    out["singular"] = result.singular;
    return out;
}

py::tuple split_branch_sensitivity(const std::int64_t node_count, const Input<std::int64_t>& left,
                                   const Input<std::int64_t>& right, const Input<double>& conductance,
                                   const Input<double>& branch_product, const Input<bool>& in_plane) {
    const auto network = network_view(node_count, left, right, conductance);
    if (branch_product.ndim() != 1 || in_plane.ndim() != 1 || branch_product.size() != left.size() ||
        in_plane.size() != left.size()) {
        throw pcbcore::InvalidInput("branch_product and in_plane must have one entry per branch");
    }
    pcbcore::network::BranchSensitivity result;
    {
        py::gil_scoped_release release;
        result = pcbcore::network::split_branch_sensitivity(network, branch_product.data(), in_plane.data());
    }
    return py::make_tuple(to_array(std::move(result.node_sensitivity)),
                          to_array(std::move(result.branch_sensitivity)), result.vertical_total);
}

}  // namespace

PYBIND11_MODULE(_pcbcore, m) {
    m.doc() = "pcb-analysis C++ core: solvers and their kernels";

    // pybind11 already maps std::invalid_argument (InvalidInput) to ValueError,
    // std::runtime_error (Singular) to RuntimeError and std::bad_alloc to
    // MemoryError; NonFinite needs its own class.
    py::register_exception_translator([](std::exception_ptr pointer) {
        try {
            if (pointer) {
                std::rethrow_exception(pointer);
            }
        } catch (const pcbcore::NonFinite& error) {
            PyErr_SetString(PyExc_FloatingPointError, error.what());
        }
    });

    py::module_ network = m.def_submodule("network", "Conductance networks: the zero-frequency sheet mesh");
    network.def("solve_conductance_network", &solve_conductance_network, py::arg("node_count"), py::arg("left"),
                py::arg("right"), py::arg("conductance"), py::arg("reference"), py::arg("injection"),
                py::arg("weights"));
    network.def("split_branch_sensitivity", &split_branch_sensitivity, py::arg("node_count"), py::arg("left"),
                py::arg("right"), py::arg("conductance"), py::arg("branch_product"), py::arg("in_plane"));
}
