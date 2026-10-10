// Bindings of the EMC post-processing (``electrical._pcbcore.emc``).
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <complex>
#include <cstdint>
#include <vector>

#include "arrays.hpp"
#include "pcbcore/emc/post.hpp"
#include "pcbcore/errors.hpp"

namespace pcbcore_bindings {

namespace {

using Complex = std::complex<double>;

py::tuple dipole_arrays(pcbcore::emc::Dipoles&& dipoles) {
    const auto count = static_cast<py::ssize_t>(dipoles.position.size() / 3);
    return py::make_tuple(to_array(std::move(dipoles.position), {count, 3}),
                          to_array(std::move(dipoles.moment), {count, 3}));
}

}  // namespace

void register_emc(py::module_& m) {
    m.def(
        "far_field",
        [](const Input<Complex>& pattern, const Input<double>& theta, const Input<double>& phi,
           const Input<double>& weight, const double wavenumber, const double distance, const double impedance) {
            const py::ssize_t n = theta.size();
            pcbcore::emc::FarField out = pcbcore::emc::far_field(
                sized(pattern, 3 * n, "pattern"), theta.data(), sized(phi, n, "phi"), sized(weight, n, "weight"), n,
                wavenumber, distance, impedance);
            return py::make_tuple(to_array(std::move(out.field), {n, 3}), to_array(std::move(out.e_theta)),
                                  to_array(std::move(out.e_phi)), out.radiated_power);
        },
        py::arg("pattern"), py::arg("theta"), py::arg("phi"), py::arg("weight"), py::arg("wavenumber"),
        py::arg("distance"), py::arg("impedance"));
    m.def(
        "dipole_moments",
        [](const Input<double>& position, const Input<Complex>& moment, const py::object& origin) {
            const py::ssize_t n = position.size() / 3;
            Input<double> holder;
            const double* o = nullptr;
            if (!origin.is_none()) {
                holder = origin.cast<Input<double>>();
                o = sized(holder, 3, "origin_m");
            }
            const auto out = pcbcore::emc::dipole_moments(sized(position, 3 * n, "position"),
                                                          sized(moment, 3 * n, "moment"), n, o);
            return py::make_tuple(to_array(std::vector<Complex>(out.begin(), out.begin() + 3)),
                                  to_array(std::vector<Complex>(out.begin() + 3, out.end())));
        },
        py::arg("position"), py::arg("moment"), py::arg("origin"));
    m.def(
        "dc_dipoles",
        [](const Input<std::uint8_t>& active, const Input<double>& thickness, const Input<double>& pitch_x,
           const Input<double>& pitch_y, const Input<double>& heights, const Input<double>& density,
           const Input<std::int64_t>& via_lower, const Input<std::int64_t>& via_upper,
           const Input<double>& via_current) {
            if (active.ndim() != 3) {
                throw pcbcore::InvalidInput("element_active must be (layers, rows, cols)");
            }
            const int layers = static_cast<int>(active.shape(0));
            const int rows = static_cast<int>(active.shape(1));
            const int cols = static_cast<int>(active.shape(2));
            const py::ssize_t vias = via_lower.size();
            return dipole_arrays(pcbcore::emc::dc_dipoles(
                layers, rows, cols, active.data(), sized(thickness, layers, "thickness"), sized(pitch_x, cols, "pitch_x"),
                sized(pitch_y, rows, "pitch_y"), sized(heights, layers, "heights"),
                sized(density, 2 * active.size(), "current density"), vias, via_lower.data(),
                sized(via_upper, vias, "via_upper"), sized(via_current, vias, "via_current")));
        },
        py::arg("element_active"), py::arg("thickness"), py::arg("pitch_x"), py::arg("pitch_y"), py::arg("heights"),
        py::arg("density"), py::arg("via_lower"), py::arg("via_upper"), py::arg("via_current"));
    m.def(
        "terminal_closure",
        [](const int rows, const int cols, const Input<double>& pitch_x, const Input<double>& pitch_y,
           const Input<double>& heights, const Input<std::int64_t>& offsets, const Input<std::int64_t>& nodes,
           const Input<double>& current) {
            const py::ssize_t terminals = current.size();
            if (offsets.size() != terminals + 1 || offsets.data()[terminals] * 3 != nodes.size()) {
                throw pcbcore::InvalidInput("terminals need offsets (terminals + 1) over (layer, row, col) triples");
            }
            return dipole_arrays(pcbcore::emc::terminal_closure(rows, cols, sized(pitch_x, cols, "pitch_x"),
                                                                sized(pitch_y, rows, "pitch_y"), heights.data(),
                                                                terminals, offsets.data(), nodes.data(),
                                                                current.data()));
        },
        py::arg("rows"), py::arg("cols"), py::arg("pitch_x"), py::arg("pitch_y"), py::arg("heights"),
        py::arg("offsets"), py::arg("nodes"), py::arg("current"));
}

}  // namespace pcbcore_bindings
