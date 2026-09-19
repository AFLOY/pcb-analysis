// Near-field terms of the precorrected-FFT sheet inductance operator.
//
// For every listed pair of branches (i, j) with sparse projection stencils
// (node index, weight) the grid path credits the pair with
//     sum_a sum_b w_i[a] w_j[b] K[(y_b - y_a) mod rows, (x_b - x_a) mod cols],
// K being the wrapped kernel table of the projection grid.  The Python path
// gathers a (pairs, s, s) table slice per chunk; this loop does the same sum
// pair by pair with OpenMP over pairs and no temporaries.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

using ArrF64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ArrI64 = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using ArrI32 = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;

py::array_t<double> grid_pair_coupling(ArrI32 indptr, ArrI32 indices, ArrF64 data, int nodes_x, ArrF64 table,
                                       ArrI64 pair_i, ArrI64 pair_j, int threads) {
    if (table.ndim() != 2) throw std::invalid_argument("table must be (rows, cols)");
    const py::ssize_t rows = table.shape(0), cols = table.shape(1);
    if (pair_i.size() != pair_j.size()) throw std::invalid_argument("pair_i and pair_j must have the same length");
    const py::ssize_t count = pair_i.size();
    const py::ssize_t branch_count = indptr.size() - 1;
    py::array_t<double> result(count);
    const std::int32_t* ptr = indptr.data();
    const std::int32_t* idx = indices.data();
    const double* wgt = data.data();
    const double* tab = table.data();
    const std::int64_t* pi = pair_i.data();
    const std::int64_t* pj = pair_j.data();
    double* out = result.mutable_data();
    for (py::ssize_t p = 0; p < count; ++p) {
        if (pi[p] < 0 || pi[p] >= branch_count || pj[p] < 0 || pj[p] >= branch_count)
            throw std::invalid_argument("pair index outside the projection rows");
    }
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
        if (threads > 0) omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#endif
        for (py::ssize_t p = 0; p < count; ++p) {
            const std::int64_t i = pi[p], j = pj[p];
            double acc = 0.0;
            for (std::int32_t a = ptr[i]; a < ptr[i + 1]; ++a) {
                const std::int64_t node_a = idx[a];
                const std::int64_t ya = node_a / nodes_x, xa = node_a - ya * nodes_x;
                const double wa = wgt[a];
                double inner = 0.0;
                for (std::int32_t b = ptr[j]; b < ptr[j + 1]; ++b) {
                    const std::int64_t node_b = idx[b];
                    const std::int64_t yb = node_b / nodes_x, xb = node_b - yb * nodes_x;
                    std::int64_t dy = (yb - ya) % rows;
                    if (dy < 0) dy += rows;
                    std::int64_t dx = (xb - xa) % cols;
                    if (dx < 0) dx += cols;
                    inner += wgt[b] * tab[dy * cols + dx];
                }
                acc += wa * inner;
            }
            out[p] = acc;
        }
    }
    return result;
}

}  // namespace

PYBIND11_MODULE(_sheet_pfft_native, m) {
    m.doc() = "Near-field grid coupling of the precorrected-FFT sheet inductance operator";
    m.def("grid_pair_coupling", &grid_pair_coupling, py::arg("indptr"), py::arg("indices"), py::arg("data"),
          py::arg("nodes_x"), py::arg("table"), py::arg("pair_i"), py::arg("pair_j"), py::arg("threads") = 0);
}
