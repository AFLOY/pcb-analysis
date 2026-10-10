// Fused CPU kernels for the layered-PCB DC conduction matrix-free MPIR path.
//
// The operator is the node-owned gather of the bilinear (sheet Q1) conduction
// action of ``pcb.py``: one output node visits its at most four adjacent
// elements on its own copper layer, applies the two per-element 4x4 unit
// tensors weighted by the element sheet conductances, adds the resistive via
// links that end at the node, and is written once.  The same gather runs in
// float32 for the low path and in float64 for the outer MPIR residual (and
// the coarse-space assembly of the two-level preconditioner).  The inner PCG
// with the two-level preconditioner (Jacobi plus patch-constant coarse
// correction through a dense float32 inverse) runs as one SPMD OpenMP region
// per outer MPIR step, with the control flow of ``solver._inner_pcg``.
//
// Vias arrive as a node-owned adjacency in CSR form (``via_ptr``,
// ``via_nbr``, ``via_g``): every link a-b is stored once under a and once
// under b, so no thread writes another thread's node and no atomics are
// needed.
//
// Compliant with MISRA-C++ principles: explicit types, no C-style casts,
// strict const correctness, RAII, noexcept specifications, and internal
// implementation details hidden within an anonymous namespace.

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#include "pcbcore/flush_subnormals.hpp"
#include "pcbcore/pragmas.hpp"

#include "pcbcore/fem/layered_dc.hpp"
#include "pcbcore/lane_sum.hpp"

namespace py = pybind11;

// Registered as electrical._pcbcore.layered_dc.
namespace pcb_layered_dc {

namespace {

using ArrF32 = py::array_t<float, py::array::c_style | py::array::forcecast>;
using ArrF64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ArrU8 = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;
using ArrI64 = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
template <typename T>
using Arr = py::array_t<T, py::array::c_style | py::array::forcecast>;

using pcbcore::FlushSubnormals;

template <typename T, int F>
[[nodiscard]] const T* data_of(const py::array_t<T, F>& a, py::ssize_t expected, const char* const name) {
    if (a.size() != expected) {
        throw std::invalid_argument(std::string(name) + " has the wrong size");
    }
    return a.data();
}

template <typename T>
struct LayeredOperatorT final {
    const T* coef{nullptr};             // (2, layers, rows, cols): c_x, c_y per element
    const T* unit{nullptr};             // (2, 4, 4): U_x, U_y; local node index 2 dy + dx
    const std::uint8_t* free_nodes{nullptr};
    const T* free_mask{nullptr};        // 1 free, 0 fixed
    const std::int64_t* via_ptr{nullptr};   // (nodes + 1,)
    const std::int64_t* via_nbr{nullptr};   // (links,)
    const T* via_g{nullptr};                // (links,)
    int layers{0};
    int rows{0};
    int cols{0};
    int threads{1};

    [[nodiscard]] int node_rows() const noexcept { return rows + 1; }
    [[nodiscard]] int node_cols() const noexcept { return cols + 1; }
    [[nodiscard]] int lines() const noexcept { return layers * node_rows(); }
    [[nodiscard]] py::ssize_t element_count() const noexcept {
        return static_cast<py::ssize_t>(layers) * static_cast<py::ssize_t>(rows) * static_cast<py::ssize_t>(cols);
    }
    [[nodiscard]] py::ssize_t node_count() const noexcept {
        return static_cast<py::ssize_t>(lines()) * static_cast<py::ssize_t>(node_cols());
    }

    [[nodiscard]] T via_terms(const T* const x, const py::ssize_t node) const noexcept {
        T acc{static_cast<T>(0)};
        const T xn = x[node] * free_mask[node];
        const std::int64_t start_idx = via_ptr[node];
        const std::int64_t end_idx = via_ptr[node + 1];
        for (std::int64_t k = start_idx; k < end_idx; ++k) {
            const std::int64_t nbr = via_nbr[k];
            acc += via_g[k] * (xn - (x[nbr] * free_mask[nbr]));
        }
        return acc;
    }

    PCBCORE_NO_FP_CONTRACT
    [[nodiscard]] T gather(const T* const x, const int l, const int y, const int xi) const noexcept {
        const int nr = node_rows();
        const int nc = node_cols();
        const py::ssize_t node = (static_cast<py::ssize_t>(l) * nr + static_cast<py::ssize_t>(y)) * nc + static_cast<py::ssize_t>(xi);
        if (free_nodes[node] == 0U) {
            return x[node];
        }
        const int y0 = (y > 0) ? (y - 1) : 0;
        const int y1 = (y < rows) ? y : (rows - 1);
        const int x0 = (xi > 0) ? (xi - 1) : 0;
        const int x1 = (xi < cols) ? xi : (cols - 1);
        const py::ssize_t ne = element_count();
        const T* const ux = unit;
        const T* const uy = unit + 16;
        T acc{static_cast<T>(0)};
        for (int ey = y0; ey <= y1; ++ey) {
            for (int ex = x0; ex <= x1; ++ex) {
                const int lr = 2 * (y - ey) + (xi - ex);
                const py::ssize_t e = (static_cast<py::ssize_t>(l) * static_cast<py::ssize_t>(rows) + static_cast<py::ssize_t>(ey)) * static_cast<py::ssize_t>(cols) + static_cast<py::ssize_t>(ex);
                const T a = coef[e];
                const T b = coef[ne + e];
                const py::ssize_t corner = (static_cast<py::ssize_t>(l) * static_cast<py::ssize_t>(nr) + static_cast<py::ssize_t>(ey)) * static_cast<py::ssize_t>(nc) + static_cast<py::ssize_t>(ex);
                for (int c = 0; c < 4; ++c) {
                    const py::ssize_t cn = corner + static_cast<py::ssize_t>(c >> 1) * nc + static_cast<py::ssize_t>(c & 1);
                    acc += ((a * ux[4 * lr + c]) + (b * uy[4 * lr + c])) * (x[cn] * free_mask[cn]);
                }
            }
        }
        return acc + via_terms(x, node);
    }

    PCBCORE_NO_FP_CONTRACT
    void apply_lines(const T* const x, T* const out, const int line_begin, const int line_end) const noexcept {
        const int nr = node_rows();
        const int nc = node_cols();
        for (int line = line_begin; line < line_end; ++line) {
            const int l = line / nr;
            const int y = line - (l * nr);
            const py::ssize_t base = static_cast<py::ssize_t>(line) * static_cast<py::ssize_t>(nc);
            out[base] = gather(x, l, y, 0);
            out[base + cols] = gather(x, l, y, cols);
            if (cols < 2) {
                continue;
            }
            const int y0 = (y > 0) ? (y - 1) : 0;
            const int y1 = (y < rows) ? y : (rows - 1);
            T* const o = out + base;
            for (int xi = 1; xi < cols; ++xi) {
                o[xi] = static_cast<T>(0);
            }
            const py::ssize_t ne = element_count();
            const T* const ux = unit;
            const T* const uy = unit + 16;
            for (int ey = y0; ey <= y1; ++ey) {
                const int lr_base = 2 * (y - ey);
                const py::ssize_t e_row = (static_cast<py::ssize_t>(l) * static_cast<py::ssize_t>(rows) + static_cast<py::ssize_t>(ey)) * static_cast<py::ssize_t>(cols);
                const T* const ax = coef + e_row;
                const T* const ay = coef + ne + e_row;
                const py::ssize_t corner_row = (static_cast<py::ssize_t>(l) * static_cast<py::ssize_t>(nr) + static_cast<py::ssize_t>(ey)) * static_cast<py::ssize_t>(nc);
PCBCORE_OMP_SIMD
                for (int xi = 1; xi < cols; ++xi) {
                    T acc{static_cast<T>(0)};
#pragma GCC unroll 2
                    for (int dx = 0; dx < 2; ++dx) {
                        const int ex = xi - 1 + dx;
                        const int lr = lr_base + (1 - dx);
                        const T a = ax[ex];
                        const T b = ay[ex];
                        const py::ssize_t corner = corner_row + static_cast<py::ssize_t>(ex);
#pragma GCC unroll 4
                        for (int c = 0; c < 4; ++c) {
                            const py::ssize_t cn = corner + static_cast<py::ssize_t>(c >> 1) * nc + static_cast<py::ssize_t>(c & 1);
                            const T w = (a * ux[4 * lr + c]) + (b * uy[4 * lr + c]);
                            acc += w * (x[cn] * free_mask[cn]);
                        }
                    }
                    o[xi] += acc;
                }
            }
            for (int xi = 1; xi < cols; ++xi) {
                const py::ssize_t node = base + static_cast<py::ssize_t>(xi);
                if (via_ptr[node] != via_ptr[node + 1]) {
                    o[xi] += via_terms(x, node);
                }
            }
PCBCORE_OMP_SIMD
            for (int xi = 1; xi < cols; ++xi) {
                const py::ssize_t node = base + static_cast<py::ssize_t>(xi);
                const T f = free_mask[node];
                o[xi] = (f * o[xi]) + ((static_cast<T>(1) - f) * x[node]);
            }
        }
    }

    static void thread_lines(const int lines_cnt, int& begin, int& end) noexcept {
#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int t = omp_get_thread_num();
        begin = static_cast<int>((static_cast<long long>(lines_cnt) * t) / nt);
        end = static_cast<int>((static_cast<long long>(lines_cnt) * (t + 1)) / nt);
#else
        begin = 0;
        end = lines_cnt;
#endif
    }
};

template <typename T>
[[nodiscard]] LayeredOperatorT<T> make_operator(const Arr<T>& coef, const Arr<T>& unit, const ArrU8& free_nodes,
                                                const Arr<T>& free_mask, const ArrI64& via_ptr, const ArrI64& via_nbr,
                                                const Arr<T>& via_g, const int layers, const int rows, const int cols,
                                                const int threads) {
    if (layers < 1 || rows < 1 || cols < 1) {
        throw std::invalid_argument("element grid must be positive");
    }
    LayeredOperatorT<T> op;
    op.layers = layers;
    op.rows = rows;
    op.cols = cols;
    op.threads = (threads < 1) ? 1 : threads;
    const py::ssize_t elements = op.element_count();
    const py::ssize_t nodes = op.node_count();
    op.coef = data_of(coef, 2 * elements, "coefficients");
    op.unit = data_of(unit, static_cast<py::ssize_t>(32), "unit");
    op.free_nodes = data_of(free_nodes, nodes, "free_nodes");
    op.free_mask = data_of(free_mask, nodes, "free_mask");
    op.via_ptr = data_of(via_ptr, nodes + 1, "via_ptr");
    const py::ssize_t links = static_cast<py::ssize_t>(op.via_ptr[nodes]);
    if (op.via_ptr[0] != 0 || links < 0) {
        throw std::invalid_argument("via_ptr must start at zero");
    }
    op.via_nbr = data_of(via_nbr, links, "via_nbr");
    op.via_g = data_of(via_g, links, "via_g");
    for (py::ssize_t k = 0; k < links; ++k) {
        if (op.via_nbr[k] < 0 || op.via_nbr[k] >= nodes) {
            throw std::invalid_argument("via_nbr names a node outside the mesh");
        }
    }
    return op;
}

// The same operator over the non-owning arrays of the C++ interface.
template <typename T>
[[nodiscard]] LayeredOperatorT<T> operator_from(const pcbcore::fem::layered_dc::OperatorView<T>& v, const int threads) {
    if (v.layers < 1 || v.rows < 1 || v.cols < 1) {
        throw std::invalid_argument("element grid must be positive");
    }
    if (v.coef == nullptr || v.unit == nullptr || v.free_nodes == nullptr || v.free_mask == nullptr ||
        v.via_ptr == nullptr) {
        throw std::invalid_argument("operator arrays are missing");
    }
    LayeredOperatorT<T> op;
    op.coef = v.coef;
    op.unit = v.unit;
    op.free_nodes = v.free_nodes;
    op.free_mask = v.free_mask;
    op.via_ptr = v.via_ptr;
    op.via_nbr = v.via_nbr;
    op.via_g = v.via_g;
    op.layers = v.layers;
    op.rows = v.rows;
    op.cols = v.cols;
    op.threads = (threads < 1) ? 1 : threads;
    const py::ssize_t nodes = op.node_count();
    const py::ssize_t links = static_cast<py::ssize_t>(op.via_ptr[nodes]);
    if (op.via_ptr[0] != 0 || links < 0) {
        throw std::invalid_argument("via_ptr must start at zero");
    }
    for (py::ssize_t k = 0; k < links; ++k) {
        if (op.via_nbr[k] < 0 || op.via_nbr[k] >= nodes) {
            throw std::invalid_argument("via_nbr names a node outside the mesh");
        }
    }
    return op;
}

template <typename T>
[[nodiscard]] pcbcore::fem::layered_dc::OperatorView<T> view_of(const LayeredOperatorT<T>& op) {
    pcbcore::fem::layered_dc::OperatorView<T> v;
    v.coef = op.coef;
    v.unit = op.unit;
    v.free_nodes = op.free_nodes;
    v.free_mask = op.free_mask;
    v.via_ptr = op.via_ptr;
    v.via_nbr = op.via_nbr;
    v.via_g = op.via_g;
    v.layers = op.layers;
    v.rows = op.rows;
    v.cols = op.cols;
    return v;
}

struct alignas(64) Partial final {
    double a{0.0};
    double pad[7]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

// The whole action on the operator's team, for callers already outside the
// GIL (the coarse assembly and the outer MPIR residual).  Lines are split as
// in apply_impl, so the result does not depend on the team size.
template <typename T>
void apply_parallel(const LayeredOperatorT<T>& op, const T* const x, T* const y) noexcept {
    const int lines = op.lines();
    const int team = std::max(1, std::min(op.threads, lines));
#pragma omp parallel num_threads(team) if (team > 1)
    {
        int b{0};
        int e{lines};
        LayeredOperatorT<T>::thread_lines(lines, b, e);
        op.apply_lines(x, y, b, e);
    }
}

template <typename T>
[[nodiscard]] Arr<T> apply_impl(const LayeredOperatorT<T>& op, const Arr<T>& vector, const bool flush_subnormals) {
    const py::ssize_t n = op.node_count();
    const T* const x = data_of(vector, n, "vector");
    Arr<T> out(n);
    T* const y = out.mutable_data();
    {
        py::gil_scoped_release release;
        const int lines = op.lines();
        const int team = std::max(1, std::min(op.threads, lines));
#pragma omp parallel num_threads(team) if (team > 1)
        {
            if (flush_subnormals) {
                const FlushSubnormals flush{};
                int b{0};
                int e{lines};
                LayeredOperatorT<T>::thread_lines(lines, b, e);
                op.apply_lines(x, y, b, e);
            } else {
                int b{0};
                int e{lines};
                LayeredOperatorT<T>::thread_lines(lines, b, e);
                op.apply_lines(x, y, b, e);
            }
        }
    }
    return out;
}

// Internal PCG core routine shared between Python entry and full MPIR solver.
void pcg_layered_dc_q1_core(
    const LayeredOperatorT<float>& op,
    const double* const rhs_in,
    const float* const diag,
    const int block,
    const float* const cinv,
    const double inner_relative_tolerance,
    const int max_inner_iterations,
    float* const xsol,
    int& total_iterations,
    int& applications,
    double& relative_residual,
    bool& not_spd) {

    const py::ssize_t n = op.node_count();
    const int nr = op.node_rows();
    const int nc = op.node_cols();
    const int coarse_rows = (nr + block - 1) / block;
    const int coarse_cols = (nc + block - 1) / block;
    const py::ssize_t ncoarse = static_cast<py::ssize_t>(op.layers) * static_cast<py::ssize_t>(coarse_rows) * static_cast<py::ssize_t>(coarse_cols);
    const bool two_level = (cinv != nullptr);

    const int lines = op.lines();
    const int team = std::max(1, std::min(op.threads, lines));
    std::vector<float> rhs(static_cast<size_t>(n), 0.0f);
    std::vector<float> r(static_cast<size_t>(n), 0.0f);
    std::vector<float> z(static_cast<size_t>(n), 0.0f);
    std::vector<float> p(static_cast<size_t>(n), 0.0f);
    std::vector<float> q(static_cast<size_t>(n), 0.0f);
    std::vector<float> coarse_r(static_cast<size_t>(two_level ? ncoarse : 0), 0.0f);
    std::vector<float> coarse_z(static_cast<size_t>(two_level ? ncoarse : 0), 0.0f);
    // One partial per slot and node line, summed in line order: the inner
    // products do not depend on the team size or on where the vectors sit.
    std::vector<double> line_partials(3U * static_cast<size_t>(lines), 0.0);

#pragma omp parallel num_threads(team) if (team > 1)
    {
        const FlushSubnormals flush{};
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
#else
        const int tid = 0;
        const int nt = 1;
#endif
        int lb{0};
        int le{lines};
        LayeredOperatorT<float>::thread_lines(lines, lb, le);
        const py::ssize_t lo = static_cast<py::ssize_t>(lb) * static_cast<py::ssize_t>(nc);
        const py::ssize_t len = static_cast<py::ssize_t>(le - lb) * static_cast<py::ssize_t>(nc);
        auto store = [&](const int s, const float* const a, const float* const b) noexcept {
            for (int line = lb; line < le; ++line) {
                const float* const al = a + static_cast<py::ssize_t>(line) * static_cast<py::ssize_t>(nc);
                const float* const bl = b + static_cast<py::ssize_t>(line) * static_cast<py::ssize_t>(nc);
                line_partials[static_cast<size_t>(s) * static_cast<size_t>(lines) + static_cast<size_t>(line)] =
                    pcbcore::lane_sum(nc, [al, bl](const std::ptrdiff_t i) {
                        return static_cast<double>(al[i]) * static_cast<double>(bl[i]);
                    });
            }
        };
        auto reduce = [&](const int s) noexcept -> double {
            const double* const part = line_partials.data() + static_cast<size_t>(s) * static_cast<size_t>(lines);
            return pcbcore::ordered_sum(lines, [part](const std::ptrdiff_t line) { return part[line]; });
        };
        auto precondition = [&]() noexcept {
            Eigen::Map<Eigen::VectorXf>(z.data() + lo, len) =
                Eigen::Map<const Eigen::VectorXf>(r.data() + lo, len)
                    .cwiseQuotient(Eigen::Map<const Eigen::VectorXf>(diag + lo, len));

            if (!two_level) {
                return;
            }
            // Restriction patch by patch: each coarse value is one thread's sum
            // over its patch in node order, whichever thread owns it.
            const py::ssize_t cb = (ncoarse * static_cast<py::ssize_t>(tid)) / static_cast<py::ssize_t>(nt);
            const py::ssize_t ce = (ncoarse * static_cast<py::ssize_t>(tid + 1)) / static_cast<py::ssize_t>(nt);
            const py::ssize_t per_layer = static_cast<py::ssize_t>(coarse_rows) * static_cast<py::ssize_t>(coarse_cols);
            for (py::ssize_t i = cb; i < ce; ++i) {
                const int l = static_cast<int>(i / per_layer);
                const int yc = static_cast<int>((i - static_cast<py::ssize_t>(l) * per_layer) / coarse_cols);
                const int xc = static_cast<int>(i - static_cast<py::ssize_t>(l) * per_layer - static_cast<py::ssize_t>(yc) * coarse_cols);
                double acc{0.0};
                for (int yy = yc * block; yy < std::min(nr, (yc + 1) * block); ++yy) {
                    const py::ssize_t row = (static_cast<py::ssize_t>(l) * nr + yy) * static_cast<py::ssize_t>(nc);
                    for (int xi = xc * block; xi < std::min(nc, (xc + 1) * block); ++xi) {
                        acc += static_cast<double>(r[static_cast<size_t>(row + xi)] * op.free_mask[row + xi]);
                    }
                }
                coarse_r[static_cast<size_t>(i)] = static_cast<float>(acc);
            }
#pragma omp barrier
            for (py::ssize_t i = cb; i < ce; ++i) {
                const float* const ci = cinv + i * ncoarse;
                coarse_z[static_cast<size_t>(i)] = static_cast<float>(pcbcore::lane_sum(ncoarse, [ci, &coarse_r](const std::ptrdiff_t j) {
                    return static_cast<double>(ci[j]) * static_cast<double>(coarse_r[static_cast<size_t>(j)]);
                }));
            }
#pragma omp barrier
            for (int line = lb; line < le; ++line) {
                const int l = line / nr;
                const int yy = line - (l * nr);
                const py::ssize_t cbase = (static_cast<py::ssize_t>(l) * static_cast<py::ssize_t>(coarse_rows) + static_cast<py::ssize_t>(yy / block)) * static_cast<py::ssize_t>(coarse_cols);
                float* const zl = z.data() + static_cast<py::ssize_t>(line) * static_cast<py::ssize_t>(nc);
                const float* const ml = op.free_mask + static_cast<py::ssize_t>(line) * static_cast<py::ssize_t>(nc);
                for (int xi = 0; xi < nc; ++xi) {
                    zl[xi] += ml[xi] * coarse_z[cbase + static_cast<py::ssize_t>(xi / block)];
                }
            }
        };

        int iterations{0};
        int applied{0};
        double rel{1.0};
        bool bad{false};
        for (py::ssize_t i = lo; i < lo + len; ++i) {
            rhs[i] = static_cast<float>(rhs_in[i]);
            xsol[i] = 0.0f;
            r[i] = rhs[i];
        }
        store(0, rhs.data(), rhs.data());
#pragma omp barrier
        const double rhs_norm = std::sqrt(reduce(0));
        if (rhs_norm == 0.0) {
            rel = 0.0;
        } else {
            precondition();
            Eigen::Map<Eigen::VectorXf>(p.data() + lo, len) = Eigen::Map<const Eigen::VectorXf>(z.data() + lo, len);
            store(2, r.data(), z.data());
#pragma omp barrier
            double rz = reduce(2);
            while (iterations < max_inner_iterations) {
                op.apply_lines(p.data(), q.data(), lb, le);
                ++applied;
                store(1, p.data(), q.data());
#pragma omp barrier
                const double curvature = reduce(1);
                if (!std::isfinite(curvature) || curvature <= 0.0) {
                    bad = true;
                    break;
                }
                const float alpha = static_cast<float>(rz / curvature);
                Eigen::Map<Eigen::VectorXf>(xsol + lo, len) += alpha * Eigen::Map<const Eigen::VectorXf>(p.data() + lo, len);
                Eigen::Map<Eigen::VectorXf>(r.data() + lo, len) -= alpha * Eigen::Map<const Eigen::VectorXf>(q.data() + lo, len);

                store(0, r.data(), r.data());
#pragma omp barrier
                rel = std::sqrt(reduce(0)) / rhs_norm;
                ++iterations;
                if (rel <= inner_relative_tolerance) {
                    break;
                }
                precondition();
                store(2, r.data(), z.data());
#pragma omp barrier
                const double next_rz = reduce(2);
                if (!std::isfinite(next_rz) || rz == 0.0) {
                    break;
                }
                const float beta = static_cast<float>(next_rz / rz);
                Eigen::Map<Eigen::VectorXf>(p.data() + lo, len) =
                    Eigen::Map<const Eigen::VectorXf>(z.data() + lo, len) +
                    beta * Eigen::Map<const Eigen::VectorXf>(p.data() + lo, len);
                rz = next_rz;
#pragma omp barrier
            }
        }
        if (tid == 0) {
            total_iterations = iterations;
            applications = applied;
            relative_residual = rel;
            not_spd = bad;
        }
    }
}

}  // namespace

ArrF32 apply_layered_dc_q1(ArrF32 vector, ArrF32 coef, ArrF32 unit, ArrU8 free_nodes, ArrF32 free_mask,
                           ArrI64 via_ptr, ArrI64 via_nbr, ArrF32 via_g, int layers, int rows, int cols,
                           int threads) {
    const LayeredOperatorT<float> op = make_operator<float>(coef, unit, free_nodes, free_mask, via_ptr, via_nbr,
                                                             via_g, layers, rows, cols, threads);
    return apply_impl<float>(op, vector, true);
}

ArrF64 apply_layered_dc_q1_f64(ArrF64 vector, ArrF64 coef, ArrF64 unit, ArrU8 free_nodes, ArrF64 free_mask,
                               ArrI64 via_ptr, ArrI64 via_nbr, ArrF64 via_g, int layers, int rows, int cols,
                               int threads) {
    const LayeredOperatorT<double> op = make_operator<double>(coef, unit, free_nodes, free_mask, via_ptr,
                                                               via_nbr, via_g, layers, rows, cols, threads);
    return apply_impl<double>(op, vector, false);
}

py::tuple pcg_layered_dc_q1(ArrF64 rhs_high, ArrF32 diagonal, ArrF32 coef, ArrF32 unit, ArrU8 free_nodes,
                            ArrF32 free_mask, ArrI64 via_ptr, ArrI64 via_nbr, ArrF32 via_g, int layers,
                            int rows, int cols, int block, ArrF32 coarse_inverse,
                            double inner_relative_tolerance, int max_inner_iterations, int threads) {
    const LayeredOperatorT<float> op = make_operator<float>(coef, unit, free_nodes, free_mask, via_ptr, via_nbr,
                                                             via_g, layers, rows, cols, threads);
    const py::ssize_t n = op.node_count();
    const double* const rhs_in = data_of(rhs_high, n, "rhs");
    const float* const diag = data_of(diagonal, n, "diagonal");
    if (max_inner_iterations < 1) {
        throw std::invalid_argument("max_inner_iterations must be positive");
    }
    if (block < 1) {
        throw std::invalid_argument("block must be positive");
    }
    const int nr = op.node_rows();
    const int nc = op.node_cols();
    const int coarse_rows = (nr + block - 1) / block;
    const int coarse_cols = (nc + block - 1) / block;
    const py::ssize_t ncoarse = static_cast<py::ssize_t>(layers) * static_cast<py::ssize_t>(coarse_rows) * static_cast<py::ssize_t>(coarse_cols);
    const bool two_level = (coarse_inverse.size() > 0);
    const float* cinv{nullptr};
    if (two_level) {
        cinv = data_of(coarse_inverse, ncoarse * ncoarse, "coarse_inverse");
    }

    ArrF32 correction_out(n);
    float* const xsol = correction_out.mutable_data();
    int total_iterations{0};
    int applications{0};
    double relative_residual{1.0};
    bool not_spd{false};

    {
        py::gil_scoped_release release;
        pcg_layered_dc_q1_core(op, rhs_in, diag, block, cinv, inner_relative_tolerance,
                               max_inner_iterations, xsol, total_iterations, applications,
                               relative_residual, not_spd);
    }
    if (not_spd) {
        throw std::runtime_error("inner PCG requires a finite symmetric positive-definite operator");
    }
    return py::make_tuple(correction_out, total_iterations, relative_residual, applications);
}

// The two-level coarse matrix Z^T A Z (27 coloured FP64 applications) and its
// inverse (Eigen LLT); the work is pcbcore::fem::layered_dc::assemble_coarse.
py::tuple assemble_coarse_dc(
    ArrF64 coef, ArrF64 unit, ArrU8 free_nodes, ArrF64 free_mask,
    ArrI64 via_ptr, ArrI64 via_nbr, ArrF64 via_g,
    int layers, int rows, int cols, int block, int threads) {

    const LayeredOperatorT<double> op = make_operator<double>(
        coef, unit, free_nodes, free_mask, via_ptr, via_nbr, via_g, layers, rows, cols, threads);
    if (block < 1) {
        throw std::invalid_argument("block must be positive");
    }
    pcbcore::fem::CoarseSpace coarse;
    {
        py::gil_scoped_release release;
        coarse = pcbcore::fem::layered_dc::assemble_coarse(view_of(op), block, op.threads);
    }
    ArrF64 matrix_out(static_cast<py::ssize_t>(coarse.matrix.size()));
    std::copy(coarse.matrix.begin(), coarse.matrix.end(), matrix_out.mutable_data());
    ArrF64 out(static_cast<py::ssize_t>(coarse.inverse.size()));
    std::copy(coarse.inverse.begin(), coarse.inverse.end(), out.mutable_data());
    return py::make_tuple(matrix_out, out);
}

// The end-to-end MPIR solve (FP64 outer residual, FP32 inner PCG) of
// pcbcore::fem::layered_dc::solve_mpir, behind array checks.
py::tuple solve_mpir_layered_dc_q1(
    ArrF64 rhs_high, ArrF64 initial_guess, ArrF32 diagonal,
    ArrF32 coef_f32, ArrF32 unit_f32, ArrU8 free_nodes, ArrF32 free_mask_f32,
    ArrI64 via_ptr, ArrI64 via_nbr, ArrF32 via_g_f32,
    ArrF64 coef_f64, ArrF64 unit_f64, ArrF64 free_mask_f64, ArrF64 via_g_f64,
    int layers, int rows, int cols, int block, ArrF32 coarse_inverse,
    double relative_tolerance, double absolute_tolerance,
    double inner_relative_tolerance, int max_outer_iterations,
    int max_inner_iterations, int threads) {

    const LayeredOperatorT<float> op_low = make_operator<float>(
        coef_f32, unit_f32, free_nodes, free_mask_f32, via_ptr, via_nbr, via_g_f32,
        layers, rows, cols, threads);
    const LayeredOperatorT<double> op_high = make_operator<double>(
        coef_f64, unit_f64, free_nodes, free_mask_f64, via_ptr, via_nbr, via_g_f64,
        layers, rows, cols, threads);

    const py::ssize_t n = op_low.node_count();
    const double* const rhs_in = data_of(rhs_high, n, "rhs_high");
    const float* const diag = data_of(diagonal, n, "diagonal");
    if (block < 1) {
        throw std::invalid_argument("block must be positive");
    }
    const int coarse_rows = (op_low.node_rows() + block - 1) / block;
    const int coarse_cols = (op_low.node_cols() + block - 1) / block;
    const py::ssize_t ncoarse = static_cast<py::ssize_t>(layers) * static_cast<py::ssize_t>(coarse_rows) * static_cast<py::ssize_t>(coarse_cols);
    const float* cinv{nullptr};
    if (coarse_inverse.size() > 0) {
        cinv = data_of(coarse_inverse, ncoarse * ncoarse, "coarse_inverse");
    }

    ArrF64 solution_out(n);
    double* const sol = solution_out.mutable_data();
    if (initial_guess.size() == n) {
        const double* const init_ptr = initial_guess.data();
        std::copy(init_ptr, init_ptr + n, sol);
    } else {
        std::fill(sol, sol + n, 0.0);
    }
    pcbcore::fem::MpirConfig config;
    config.relative_tolerance = relative_tolerance;
    config.absolute_tolerance = absolute_tolerance;
    config.inner_relative_tolerance = inner_relative_tolerance;
    config.max_outer_iterations = max_outer_iterations;
    config.max_inner_iterations = max_inner_iterations;
    pcbcore::fem::MpirResult result;
    {
        py::gil_scoped_release release;
        result = pcbcore::fem::layered_dc::solve_mpir(view_of(op_low), view_of(op_high), rhs_in, sol, diag, block,
                                                      cinv, config, op_low.threads);
    }
    std::vector<std::tuple<int, double, int, double>> history;
    for (const auto& step : result.history) {
        history.emplace_back(step.outer_iteration, step.high_relative_residual, step.inner_iterations,
                             step.inner_relative_residual);
    }
    return py::make_tuple(
        solution_out, result.converged, result.outer_iterations, result.inner_iterations,
        result.relative_residual, result.high_operator_applications, result.low_operator_applications,
        py::cast(history));
}

void register_module(py::module_& m) {
    m.doc() = "Fused C++ layered-PCB DC conduction operator (float32 and float64), coarse assembly and full MPIR solver";
    m.def("apply_layered_dc_q1", &apply_layered_dc_q1, py::arg("vector"), py::arg("coefficients"), py::arg("unit"),
          py::arg("free_nodes"), py::arg("free_mask"), py::arg("via_ptr"), py::arg("via_nbr"), py::arg("via_g"),
          py::arg("layers"), py::arg("rows"), py::arg("cols"), py::arg("threads") = 1);
    m.def("apply_layered_dc_q1_f64", &apply_layered_dc_q1_f64, py::arg("vector"), py::arg("coefficients"),
          py::arg("unit"), py::arg("free_nodes"), py::arg("free_mask"), py::arg("via_ptr"), py::arg("via_nbr"),
          py::arg("via_g"), py::arg("layers"), py::arg("rows"), py::arg("cols"), py::arg("threads") = 1);
    m.def("pcg_layered_dc_q1", &pcg_layered_dc_q1, py::arg("rhs_high"), py::arg("diagonal"), py::arg("coefficients"),
          py::arg("unit"), py::arg("free_nodes"), py::arg("free_mask"), py::arg("via_ptr"), py::arg("via_nbr"),
          py::arg("via_g"), py::arg("layers"), py::arg("rows"), py::arg("cols"), py::arg("block"),
          py::arg("coarse_inverse"), py::arg("inner_relative_tolerance"), py::arg("max_inner_iterations"),
          py::arg("threads") = 1);
    m.def("assemble_coarse_dc", &assemble_coarse_dc,
          py::arg("coefficients"), py::arg("unit"), py::arg("free_nodes"), py::arg("free_mask"),
          py::arg("via_ptr"), py::arg("via_nbr"), py::arg("via_g"),
          py::arg("layers"), py::arg("rows"), py::arg("cols"), py::arg("block"), py::arg("threads") = 1);
    m.def("solve_mpir_layered_dc_q1", &solve_mpir_layered_dc_q1,
          py::arg("rhs_high"), py::arg("initial_guess"), py::arg("diagonal"),
          py::arg("coefficients_f32"), py::arg("unit_f32"), py::arg("free_nodes"), py::arg("free_mask_f32"),
          py::arg("via_ptr"), py::arg("via_nbr"), py::arg("via_g_f32"),
          py::arg("coefficients_f64"), py::arg("unit_f64"), py::arg("free_mask_f64"), py::arg("via_g_f64"),
          py::arg("layers"), py::arg("rows"), py::arg("cols"), py::arg("block"), py::arg("coarse_inverse"),
          py::arg("relative_tolerance"), py::arg("absolute_tolerance"),
          py::arg("inner_relative_tolerance"), py::arg("max_outer_iterations"),
          py::arg("max_inner_iterations"), py::arg("threads") = 1);
    m.attr("openmp") =
#ifdef _OPENMP
        true;
#else
        false;
#endif
}

}  // namespace pcb_layered_dc

namespace pcbcore::fem::layered_dc {

using pcb_layered_dc::LayeredOperatorT;

void apply_high(const OperatorView<double>& view, const double* const x, double* const y, const int threads) {
    const LayeredOperatorT<double> op = pcb_layered_dc::operator_from(view, threads);
    pcb_layered_dc::apply_parallel(op, x, y);
}

void apply_low(const OperatorView<float>& view, const float* const x, float* const y, const int threads) {
    const LayeredOperatorT<float> op = pcb_layered_dc::operator_from(view, threads);
    pcb_layered_dc::apply_parallel(op, x, y);
}

CoarseSpace assemble_coarse(const OperatorView<double>& view, const int block, const int threads) {
    if (block < 1) {
        throw std::invalid_argument("block must be positive");
    }
    const LayeredOperatorT<double> op = pcb_layered_dc::operator_from(view, threads);
    const int layers = op.layers;
    const int nr = op.node_rows();
    const int nc = op.node_cols();
    const int coarse_rows = (nr + block - 1) / block;
    const int coarse_cols = (nc + block - 1) / block;
    const py::ssize_t ncoarse = static_cast<py::ssize_t>(layers) * static_cast<py::ssize_t>(coarse_rows) * static_cast<py::ssize_t>(coarse_cols);
    const py::ssize_t n_nodes = op.node_count();

    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(ncoarse, ncoarse);
    Eigen::VectorXd counts = Eigen::VectorXd::Zero(ncoarse);

    std::vector<double> fine(static_cast<size_t>(n_nodes), 0.0);
    std::vector<double> action(static_cast<size_t>(n_nodes), 0.0);
    std::vector<double> restricted(static_cast<size_t>(ncoarse), 0.0);

    for (int l = 0; l < layers; ++l) {
        for (int y = 0; y < nr; ++y) {
            const int yc = y / block;
            for (int x = 0; x < nc; ++x) {
                const int xc = x / block;
                const py::ssize_t node = (static_cast<py::ssize_t>(l) * nr + static_cast<py::ssize_t>(y)) * nc + static_cast<py::ssize_t>(x);
                if (op.free_nodes[node] != 0U) {
                    const py::ssize_t patch = (static_cast<py::ssize_t>(l) * coarse_rows + yc) * coarse_cols + xc;
                    counts[patch] += 1.0;
                }
            }
        }
    }

    {
        for (int colour = 0; colour < 27; ++colour) {
            bool has_colour{false};
            for (py::ssize_t i = 0; i < n_nodes; ++i) {
                fine[static_cast<size_t>(i)] = 0.0;
            }
            for (int l = 0; l < layers; ++l) {
                const int cl = l % 3;
                for (int y = 0; y < nr; ++y) {
                    const int yc = y / block;
                    const int cy = yc % 3;
                    for (int x = 0; x < nc; ++x) {
                        const int xc = x / block;
                        const int cx = xc % 3;
                        const int col_idx = (cl * 9) + (cy * 3) + cx;
                        if (col_idx == colour) {
                            const py::ssize_t node = (static_cast<py::ssize_t>(l) * nr + static_cast<py::ssize_t>(y)) * nc + static_cast<py::ssize_t>(x);
                            if (op.free_nodes[node] != 0U) {
                                fine[static_cast<size_t>(node)] = 1.0;
                                has_colour = true;
                            }
                        }
                    }
                }
            }
            if (!has_colour) {
                continue;
            }

            apply_parallel(op, fine.data(), action.data());

            std::fill(restricted.begin(), restricted.end(), 0.0);
            for (int l = 0; l < layers; ++l) {
                for (int y = 0; y < nr; ++y) {
                    const int yc = y / block;
                    for (int x = 0; x < nc; ++x) {
                        const int xc = x / block;
                        const py::ssize_t node = (static_cast<py::ssize_t>(l) * nr + static_cast<py::ssize_t>(y)) * nc + static_cast<py::ssize_t>(x);
                        if (op.free_nodes[node] != 0U) {
                            const py::ssize_t patch = (static_cast<py::ssize_t>(l) * coarse_rows + yc) * coarse_cols + xc;
                            restricted[static_cast<size_t>(patch)] += action[static_cast<size_t>(node)];
                        }
                    }
                }
            }

            for (int l = 0; l < layers; ++l) {
                for (int yc = 0; yc < coarse_rows; ++yc) {
                    for (int xc = 0; xc < coarse_cols; ++xc) {
                        const py::ssize_t I = (static_cast<py::ssize_t>(l) * coarse_rows + yc) * coarse_cols + xc;
                        for (int dl = -1; dl <= 1; ++dl) {
                            const int sl = l + dl;
                            if (sl < 0 || sl >= layers) continue;
                            const int ncl = sl % 3;
                            for (int dy = -1; dy <= 1; ++dy) {
                                const int syc = yc + dy;
                                if (syc < 0 || syc >= coarse_rows) continue;
                                const int ncy = syc % 3;
                                for (int dx = -1; dx <= 1; ++dx) {
                                    const int sxc = xc + dx;
                                    if (sxc < 0 || sxc >= coarse_cols) continue;
                                    const int ncx = sxc % 3;
                                    const int ncolour = (ncl * 9) + (ncy * 3) + ncx;
                                    if (ncolour == colour) {
                                        const py::ssize_t J = (static_cast<py::ssize_t>(sl) * coarse_rows + syc) * coarse_cols + sxc;
                                        matrix(I, J) = restricted[static_cast<size_t>(I)];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for (py::ssize_t i = 0; i < ncoarse; ++i) {
            if (counts[i] <= 0.0) {
                matrix.row(i).setZero();
                matrix.col(i).setZero();
                matrix(i, i) = 1.0;
            }
        }
    }

    Eigen::LLT<Eigen::MatrixXd> llt(matrix);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("coarse matrix is not positive definite");
    }
    Eigen::MatrixXd inv = llt.solve(Eigen::MatrixXd::Identity(ncoarse, ncoarse));
    inv = 0.5 * (inv + inv.transpose());

    CoarseSpace out;
    out.size = static_cast<std::int64_t>(ncoarse);
    out.matrix.resize(static_cast<std::size_t>(ncoarse * ncoarse));
    out.inverse.resize(static_cast<std::size_t>(ncoarse * ncoarse));
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        out.matrix.data(), ncoarse, ncoarse) = matrix;
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        out.inverse.data(), ncoarse, ncoarse) = inv;
    return out;
}

MpirResult solve_mpir(const OperatorView<float>& low_view, const OperatorView<double>& high_view,
                      const double* const rhs_in, double* const sol, const float* const diag, const int block,
                      const float* const cinv, const MpirConfig& config, const int threads) {
    const LayeredOperatorT<float> op_low = pcb_layered_dc::operator_from(low_view, threads);
    const LayeredOperatorT<double> op_high = pcb_layered_dc::operator_from(high_view, threads);
    if (block < 1) {
        throw std::invalid_argument("block must be positive");
    }
    const py::ssize_t n = op_low.node_count();
    const double relative_tolerance = config.relative_tolerance;
    const double absolute_tolerance = config.absolute_tolerance;
    const double inner_relative_tolerance = config.inner_relative_tolerance;
    const int max_outer_iterations = config.max_outer_iterations;
    const int max_inner_iterations = config.max_inner_iterations;
    int outer_iterations{0};
    int total_inner_iterations{0};
    // One (outer step, FP64 relative residual before it, inner iterations,
    // inner relative residual) per correction, as the Python loop records.
    std::vector<std::tuple<int, double, int, double>> history;
    int total_high_apps{0};
    int total_low_apps{0};
    double relative_residual{1.0};
    bool converged{false};

    {
        const double rhs_norm = std::sqrt(pcbcore::lane_sum(n, [rhs_in](const std::ptrdiff_t i) { return rhs_in[i] * rhs_in[i]; }));
        const double scale = (rhs_norm > 0.0) ? rhs_norm : 1.0;
        const double target = absolute_tolerance + (relative_tolerance * scale);

        std::vector<double> residual(static_cast<size_t>(n), 0.0);
        std::vector<double> Ax(static_cast<size_t>(n), 0.0);
        std::vector<float> correction(static_cast<size_t>(n), 0.0f);

        for (int outer = 0; outer <= max_outer_iterations; ++outer) {
            outer_iterations = outer;
            apply_parallel(op_high, sol, Ax.data());
            ++total_high_apps;

            for (py::ssize_t i = 0; i < n; ++i) {
                residual[static_cast<size_t>(i)] = rhs_in[i] - Ax[static_cast<size_t>(i)];
            }
            const double res_norm = std::sqrt(pcbcore::lane_sum(n, [&residual](const std::ptrdiff_t i) {
                return residual[static_cast<size_t>(i)] * residual[static_cast<size_t>(i)];
            }));
            relative_residual = res_norm / scale;

            if (res_norm <= target) {
                converged = true;
                break;
            }
            if (outer == max_outer_iterations) {
                break;
            }

            int inner_iters{0};
            int inner_apps{0};
            double inner_rel{1.0};
            bool not_spd{false};

            pcg_layered_dc_q1_core(
                op_low, residual.data(), diag, block, cinv,
                inner_relative_tolerance, max_inner_iterations,
                correction.data(), inner_iters, inner_apps, inner_rel, not_spd);

            if (not_spd) {
                throw std::runtime_error("inner PCG requires a finite SPD operator");
            }

            total_inner_iterations += inner_iters;
            total_low_apps += inner_apps;
            history.emplace_back(outer + 1, relative_residual, inner_iters, inner_rel);

            Eigen::Map<Eigen::VectorXd>(sol, n) +=
                Eigen::Map<const Eigen::VectorXf>(correction.data(), n).cast<double>();
        }
    }

    MpirResult result;
    result.converged = converged;
    result.outer_iterations = outer_iterations;
    result.inner_iterations = total_inner_iterations;
    result.relative_residual = relative_residual;
    result.high_operator_applications = total_high_apps;
    result.low_operator_applications = total_low_apps;
    for (const auto& [step, high, inner, inner_rel] : history) {
        result.history.push_back(MpirStep{step, high, inner, inner_rel});
    }
    return result;
}

}  // namespace pcbcore::fem::layered_dc

