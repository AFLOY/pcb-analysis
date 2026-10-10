// Near-field terms of the precorrected-FFT sheet inductance operator.
//
// For every listed pair of branches (i, j) with sparse projection stencils
// (node index, weight) the grid path credits the pair with
//     sum_a sum_b w_i[a] w_j[b] K[(y_b - y_a) mod rows, (x_b - x_a) mod cols],
// K being the wrapped kernel table of the projection grid.  The Python path
// gathers a (pairs, s, s) table slice per chunk; this loop does the same sum
// pair by pair with OpenMP over pairs and no temporaries.
//
// Compliant with MISRA-C++ principles: explicit types, no C-style casts,
// strict const correctness, RAII, noexcept specifications, and internal
// implementation details hidden within an anonymous namespace.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// Registered as electrical._pcbcore.sheet_pfft_near.
namespace pcb_sheet_pfft_near {

namespace {

using ArrF64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ArrI64 = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using ArrI32 = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;

[[nodiscard]] py::array_t<double> grid_pair_coupling(
    const ArrI32& indptr, const ArrI32& indices, const ArrF64& data,
    const int nodes_x, const ArrF64& table,
    const ArrI64& pair_i, const ArrI64& pair_j, const int threads) {
    if (table.ndim() != 2) {
        throw std::invalid_argument("table must be (rows, cols)");
    }
    const py::ssize_t rows = table.shape(0);
    const py::ssize_t cols = table.shape(1);
    if (pair_i.size() != pair_j.size()) {
        throw std::invalid_argument("pair_i and pair_j must have the same length");
    }
    const py::ssize_t count = pair_i.size();
    const py::ssize_t branch_count = indptr.size() - 1;
    py::array_t<double> result(count);
    const std::int32_t* const ptr = indptr.data();
    const std::int32_t* const idx = indices.data();
    const double* const wgt = data.data();
    const double* const tab = table.data();
    const std::int64_t* const pi = pair_i.data();
    const std::int64_t* const pj = pair_j.data();
    double* const out = result.mutable_data();

    for (py::ssize_t p = 0; p < count; ++p) {
        if (pi[p] < 0 || pi[p] >= branch_count || pj[p] < 0 || pj[p] >= branch_count) {
            throw std::invalid_argument("pair index outside the projection rows");
        }
    }

    {
        py::gil_scoped_release release;
        // The team is named on the region: omp_set_num_threads would change the
        // calling thread's default for every later region, ours or not.
        const int team = (threads > 0) ? threads : 1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(team) if (team > 1)
#endif
        for (py::ssize_t p = 0; p < count; ++p) {
            const std::int64_t i = pi[p];
            const std::int64_t j = pj[p];
            double acc{0.0};
            const std::int32_t a_start = ptr[i];
            const std::int32_t a_end = ptr[i + 1];
            const std::int32_t b_start = ptr[j];
            const std::int32_t b_end = ptr[j + 1];

            for (std::int32_t a = a_start; a < a_end; ++a) {
                const std::int64_t node_a = idx[a];
                const std::int64_t ya = node_a / static_cast<std::int64_t>(nodes_x);
                const std::int64_t xa = node_a - (ya * static_cast<std::int64_t>(nodes_x));
                const double wa = wgt[a];
                double inner{0.0};

                for (std::int32_t b = b_start; b < b_end; ++b) {
                    const std::int64_t node_b = idx[b];
                    const std::int64_t yb = node_b / static_cast<std::int64_t>(nodes_x);
                    const std::int64_t xb = node_b - (yb * static_cast<std::int64_t>(nodes_x));
                    std::int64_t dy = (yb - ya) % rows;
                    if (dy < 0) {
                        dy += rows;
                    }
                    std::int64_t dx = (xb - xa) % cols;
                    if (dx < 0) {
                        dx += cols;
                    }
                    inner += wgt[b] * tab[(dy * cols) + dx];
                }
                acc += wa * inner;
            }
            out[p] = acc;
        }
    }
    return result;
}

}  // namespace

void register_module(py::module_& m) {
    m.doc() = "Near-field grid coupling of the precorrected-FFT sheet inductance operator";
    m.def("grid_pair_coupling", &grid_pair_coupling, py::arg("indptr"), py::arg("indices"), py::arg("data"),
          py::arg("nodes_x"), py::arg("table"), py::arg("pair_i"), py::arg("pair_j"), py::arg("threads") = 0);
}

}  // namespace pcb_sheet_pfft_near

