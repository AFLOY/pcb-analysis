// ``electrical._pcbcore_cuda``: the FP32 inner solves of prepared systems on
// a CUDA device, the FP64 outer refinement on the host (pcbcore/cuda).
//
// The systems are the ones ``electrical._pcbcore`` builds; this module
// imports it so their Python types are known, and keeps each Python object
// alive for as long as its device copy lives.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "arrays.hpp"
#include "mpir_convert.hpp"
#include "pcbcore/cuda/inner_pcg.hpp"
#include "pcbcore/errors.hpp"
#include "pcbcore/fem/layered_dc_system.hpp"
#include "pcbcore/thermal/thermal_system.hpp"

namespace py = pybind11;
using pcbcore_bindings::Input;
using pcbcore_bindings::mpir_config;
using pcbcore_bindings::result_dict;
using pcbcore_bindings::sized;
using pcbcore_bindings::to_array;

namespace {

// One prepared system with its inner solve resident on a device.
struct DeviceSystem {
    py::object owner;  // the Python system: keeps it and what it keeps alive
    std::function<void(const double*, double*, int)> apply_high;
    std::unique_ptr<pcbcore::cuda::InnerSolver> inner;
    long long size{0};
};

DeviceSystem layered_dc(const py::object& owner, const int device) {
    const auto system = owner.cast<std::shared_ptr<pcbcore::fem::LayeredDCSystem>>();
    DeviceSystem out;
    out.owner = owner;
    out.size = system->size();
    const auto& cinv = system->coarse_inverse_low();
    out.inner = pcbcore::cuda::make_layered_dc(system->low_view(), system->diagonal_low().data(), system->block(),
                                               cinv.empty() ? nullptr : cinv.data(), device);
    out.apply_high = [system](const double* x, double* y, const int threads) { system->apply_high(x, y, threads); };
    return out;
}

DeviceSystem thermal(const py::object& owner, const int device) {
    const auto system = owner.cast<std::shared_ptr<pcbcore::thermal::ThermalSystem>>();
    DeviceSystem out;
    out.owner = owner;
    out.size = system->size();
    const auto& cinv = system->coarse_inverse_low();
    out.inner = pcbcore::cuda::make_thermal_hex(system->low_view(), system->diagonal_low().data(), system->block(),
                                                cinv.empty() ? nullptr : cinv.data(), device);
    out.apply_high = [system](const double* x, double* y, const int threads) { system->apply_high(x, y, threads); };
    return out;
}

py::dict solve(DeviceSystem& d, const Input<double>& rhs, const py::object& initial,
               const double relative_tolerance, const double absolute_tolerance,
               const double inner_relative_tolerance, const int max_outer_iterations,
               const int max_inner_iterations, const int threads) {
    const double* b = sized(rhs, d.size, "rhs");
    std::vector<double> x(static_cast<std::size_t>(d.size), 0.0);
    if (!initial.is_none()) {
        const auto guess = initial.cast<Input<double>>();
        const double* g = sized(guess, d.size, "initial_guess");
        std::copy(g, g + d.size, x.begin());
    }
    const auto config = mpir_config(relative_tolerance, absolute_tolerance, inner_relative_tolerance,
                                    max_outer_iterations, max_inner_iterations);
    pcbcore::fem::MpirResult result;
    {
        py::gil_scoped_release release;
        const auto apply = [&d, threads](const double* in, double* out) { d.apply_high(in, out, threads); };
        result = pcbcore::cuda::solve_mpir(apply, *d.inner, b, x.data(), config);
    }
    py::dict out = result_dict(result);
    out["solution"] = to_array(std::move(x));
    return out;
}

}  // namespace

PYBIND11_MODULE(_pcbcore_cuda, m) {
    m.doc() = "pcb-analysis C++ core: inner solves on a CUDA device";
    // The system types live in the CPU module; registering them first lets
    // this module accept its objects.
    py::module_::import("electrical._pcbcore");

    m.def("device_count", &pcbcore::cuda::device_count);
    py::class_<DeviceSystem>(m, "DeviceSystem")
        .def_static("layered_dc", &layered_dc, py::arg("system"), py::arg("device") = 0)
        .def_static("thermal", &thermal, py::arg("system"), py::arg("device") = 0)
        .def_property_readonly("size", [](const DeviceSystem& d) { return d.size; })
        .def("solve", &solve, py::arg("rhs"), py::arg("initial_guess"), py::arg("relative_tolerance"),
             py::arg("absolute_tolerance"), py::arg("inner_relative_tolerance"), py::arg("max_outer_iterations"),
             py::arg("max_inner_iterations"), py::arg("threads"));
}
