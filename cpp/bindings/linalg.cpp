// ``electrical._pcbcore.linalg``: sparse factorisations the Python solvers use
// directly.  ``SymmetricLdlt`` is present only in a build linked against MUMPS
// (``-DPCB_NATIVE_MUMPS=ON``); ``linalg.mumps_version`` is ``None`` otherwise.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrays.hpp"
#include "pcbcore/errors.hpp"

#ifdef PCB_HAVE_MUMPS
#include "pcbcore/linalg/symmetric_ldlt.hpp"
#endif

namespace pcbcore_bindings {

namespace py = pybind11;

#ifdef PCB_HAVE_MUMPS
namespace {

using pcbcore::linalg::SymmetricLdlt;
using pcbcore::linalg::SymmetricOrdering;
using Complex = std::complex<double>;

SymmetricOrdering parse_ordering(const std::string& name) {
    if (name == "auto") {
        return SymmetricOrdering::automatic;
    }
    if (name == "amd") {
        return SymmetricOrdering::amd;
    }
    if (name == "metis") {
        return SymmetricOrdering::metis;
    }
    if (name == "scotch") {
        return SymmetricOrdering::scotch;
    }
    throw pcbcore::InvalidInput("ordering must be 'auto', 'amd', 'metis' or 'scotch'");
}

std::unique_ptr<SymmetricLdlt> make_ldlt(const std::int64_t size, const Input<std::int64_t>& row_start,
                                         const Input<std::int64_t>& column, const std::string& ordering,
                                         const int threads, const py::dict& controls) {
    if (row_start.ndim() != 1 || row_start.size() != size + 1 || column.ndim() != 1) {
        throw pcbcore::InvalidInput("row_start must have size + 1 entries and column must be 1-D");
    }
    if (column.size() != row_start.data()[size]) {
        throw pcbcore::InvalidInput("column must have row_start[-1] entries");
    }
    const SymmetricOrdering parsed = parse_ordering(ordering);
    std::vector<std::pair<int, int>> settings;
    for (const auto& item : controls) {
        settings.emplace_back(item.first.cast<int>(), item.second.cast<int>());
    }
    py::gil_scoped_release release;
    return std::make_unique<SymmetricLdlt>(size, row_start.data(), column.data(), parsed, threads, settings);
}

void factor_ldlt(SymmetricLdlt& ldlt, const Input<Complex>& values, const double pivot_threshold,
                 const int threads) {
    if (values.ndim() != 1 || values.size() != ldlt.nonzeros()) {
        throw pcbcore::InvalidInput("values must be 1-D with one entry per pattern entry");
    }
    py::gil_scoped_release release;
    ldlt.factor(values.data(), pivot_threshold, threads);
}

// Solves for a vector (n,) or a block (n, k); returns new arrays of the same shape.
py::array_t<Complex> solve_ldlt(SymmetricLdlt& ldlt, const py::array_t<Complex>& right_hand_sides,
                                const int threads) {
    const auto n = static_cast<py::ssize_t>(ldlt.size());
    if (right_hand_sides.ndim() == 1) {
        if (right_hand_sides.shape(0) != n) {
            throw pcbcore::InvalidInput("the right-hand side must have size() entries");
        }
        py::array_t<Complex> out(n);
        auto source = right_hand_sides.unchecked<1>();
        Complex* const target = out.mutable_data();
        for (py::ssize_t i = 0; i < n; ++i) {
            target[i] = source(i);
        }
        {
            py::gil_scoped_release release;
            ldlt.solve(target, 1, threads);
        }
        return out;
    }
    if (right_hand_sides.ndim() != 2 || right_hand_sides.shape(0) != n) {
        throw pcbcore::InvalidInput("right-hand sides must be (size(),) or (size(), k)");
    }
    const py::ssize_t columns = right_hand_sides.shape(1);
    // Column-major for MUMPS: an (n, k) array in Fortran order.
    py::array_t<Complex, py::array::f_style> out({n, columns});
    auto source = right_hand_sides.unchecked<2>();
    Complex* const target = out.mutable_data();
    for (py::ssize_t c = 0; c < columns; ++c) {
        for (py::ssize_t i = 0; i < n; ++i) {
            target[c * n + i] = source(i, c);
        }
    }
    {
        py::gil_scoped_release release;
        ldlt.solve(target, columns, threads);
    }
    return out;
}

}  // namespace
#endif

void register_linalg(py::module_& m) {
#ifdef PCB_HAVE_MUMPS
    m.attr("mumps_version") = pcbcore::linalg::mumps_version();
    py::class_<SymmetricLdlt>(m, "SymmetricLdlt",
                              "LDL^T of a complex symmetric sparse matrix by MUMPS: the upper triangle's "
                              "pattern is analysed at construction, factor() takes its values")
        .def(py::init(&make_ldlt), py::arg("size"), py::arg("row_start"), py::arg("column"),
             py::arg("ordering") = "metis", py::arg("threads") = 1, py::arg("controls") = py::dict())
        .def("factor", &factor_ldlt, py::arg("values"), py::arg("pivot_threshold") = 0.01, py::arg("threads") = 1)
        .def("solve", &solve_ldlt, py::arg("right_hand_sides"), py::arg("threads") = 1)
        .def_property_readonly("size", &SymmetricLdlt::size)
        .def_property_readonly("nonzeros", &SymmetricLdlt::nonzeros)
        .def_property_readonly("factored", &SymmetricLdlt::factored)
        .def_property_readonly("factor_entries", &SymmetricLdlt::factor_entries)
        .def_property_readonly("analysis_seconds", &SymmetricLdlt::analysis_seconds)
        .def_property_readonly("factor_seconds", &SymmetricLdlt::factor_seconds)
        .def("info", &SymmetricLdlt::info, py::arg("k"), "MUMPS INFOG(k), one-based");
#else
    m.attr("mumps_version") = py::none();
#endif
}

}  // namespace pcbcore_bindings
