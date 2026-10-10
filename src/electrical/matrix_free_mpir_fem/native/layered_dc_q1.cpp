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
#include <pybind11/pybind11.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <xmmintrin.h>

namespace py = pybind11;

namespace {

using ArrF32 = py::array_t<float, py::array::c_style | py::array::forcecast>;
using ArrF64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ArrU8 = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;
using ArrI64 = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
template <typename T>
using Arr = py::array_t<T, py::array::c_style | py::array::forcecast>;

class FlushSubnormals final {
   public:
    explicit FlushSubnormals() noexcept : saved_(_mm_getcsr()) {
        _mm_setcsr(saved_ | 0x8040u);
    }
    ~FlushSubnormals() noexcept {
        _mm_setcsr(saved_);
    }
    FlushSubnormals(const FlushSubnormals&) = delete;
    FlushSubnormals& operator=(const FlushSubnormals&) = delete;
    FlushSubnormals(FlushSubnormals&&) = delete;
    FlushSubnormals& operator=(FlushSubnormals&&) = delete;

   private:
    unsigned int saved_{0U};
};

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

    // Resistive links of one node: sum g (x_n - x_nbr) over the vias that
    // end there.  Fixed neighbours contribute zero, like the masked gather.
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

    // The corner weights are formed as fl(fl(a U_x) + fl(b U_y)) with two
    // separate roundings, like the NumPy path.  With one rounding (an FMA)
    // the four weights a row of a uniform element contributes to a constant
    // vector no longer cancel exactly, and every element then leaves a bias
    // of order eps32 * a in the constant mode -- the one mode a DC problem
    // pins only through its reference node, so the inner PCG would spend
    // its iterations on rounding noise.  Contraction is therefore off in the
    // two gather functions.
    __attribute__((optimize("-ffp-contract=off")))
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

    // y = A x on node lines [line_begin, line_end).  A line is one (layer, y)
    // row of node_cols nodes.  Interior x use a branch-free form over the two
    // adjacent element rows, unrolled over the two x-neighbour elements and
    // the four local columns, so the x-loop vectorises; the ends of the line
    // use the generic gather.
    __attribute__((optimize("-ffp-contract=off")))
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
                const py::ssize_t e_row = (static_cast<py::ssize_t>(l) * static_cast<py::ssize_t>(rows) + static_cast<py::ssize_t>(ey)) * static_cast<py::ssize_t>(cols);  // element ex = 0
                const T* const ax = coef + e_row;
                const T* const ay = coef + ne + e_row;
                const py::ssize_t corner_row = (static_cast<py::ssize_t>(l) * static_cast<py::ssize_t>(nr) + static_cast<py::ssize_t>(ey)) * static_cast<py::ssize_t>(nc);  // corner ex = 0
#pragma omp simd
                for (int xi = 1; xi < cols; ++xi) {
                    T acc{static_cast<T>(0)};
#pragma GCC unroll 2
                    for (int dx = 0; dx < 2; ++dx) {
                        const int ex = xi - 1 + dx;        // element left (dx=0) or right (dx=1)
                        const int lr = lr_base + (1 - dx);  // local x index of the node in it
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
            // Via links end at few nodes; a sparse pass over the line.
            for (int xi = 1; xi < cols; ++xi) {
                const py::ssize_t node = base + static_cast<py::ssize_t>(xi);
                if (via_ptr[node] != via_ptr[node + 1]) {
                    o[xi] += via_terms(x, node);
                }
            }
#pragma omp simd
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

struct alignas(64) Partial final {
    double a{0.0};
    double pad[7]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

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

}  // namespace

ArrF32 apply_layered_dc_q1(ArrF32 vector, ArrF32 coef, ArrF32 unit, ArrU8 free_nodes, ArrF32 free_mask,
                           ArrI64 via_ptr, ArrI64 via_nbr, ArrF32 via_g, int layers, int rows, int cols,
                           int threads) {
    const LayeredOperatorT<float> op = make_operator<float>(coef, unit, free_nodes, free_mask, via_ptr, via_nbr,
                                                             via_g, layers, rows, cols, threads);
    return apply_impl<float>(op, vector, true);
}

// y = A x in float64 for the outer MPIR residual and the coarse-space
// assembly.  The MXCSR is left untouched so FP64 subnormals keep IEEE semantics.
ArrF64 apply_layered_dc_q1_f64(ArrF64 vector, ArrF64 coef, ArrF64 unit, ArrU8 free_nodes, ArrF64 free_mask,
                               ArrI64 via_ptr, ArrI64 via_nbr, ArrF64 via_g, int layers, int rows, int cols,
                               int threads) {
    const LayeredOperatorT<double> op = make_operator<double>(coef, unit, free_nodes, free_mask, via_ptr,
                                                               via_nbr, via_g, layers, rows, cols, threads);
    return apply_impl<double>(op, vector, false);
}

// Inner PCG in float32 with the two-level preconditioner
//   M^-1 r = D^-1 r + Z (Z^T A Z)^-1 Z^T r
// (or Jacobi only when ``coarse_inverse`` is empty).  ``block`` is the
// in-plane patch width; the coarse index of node (l, y, x) is
// (l * coarse_rows + y / block) * coarse_cols + x / block.  Returns
// (correction float32, iterations, relative_residual, applications) with the
// control flow of solver._inner_pcg.  Threads run SPMD over a static partition
// of node lines; reductions are per-thread partials summed in thread order.
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
        const int lines = op.lines();
        const int team = std::max(1, std::min(op.threads, lines));
        std::vector<float> rhs(static_cast<size_t>(n), 0.0f);
        std::vector<float> r(static_cast<size_t>(n), 0.0f);
        std::vector<float> z(static_cast<size_t>(n), 0.0f);
        std::vector<float> p(static_cast<size_t>(n), 0.0f);
        std::vector<float> q(static_cast<size_t>(n), 0.0f);
        std::vector<double> coarse_part(static_cast<size_t>(team) * static_cast<size_t>(two_level ? ncoarse : 0), 0.0);
        std::vector<float> coarse_r(static_cast<size_t>(two_level ? ncoarse : 0), 0.0f);
        std::vector<float> coarse_z(static_cast<size_t>(two_level ? ncoarse : 0), 0.0f);
        // Slots: 0 norms, 1 curvature, 2 rz.
        std::vector<Partial> partials(3U * static_cast<size_t>(team));

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
            auto slot = [&](const int s) noexcept -> double& {
                return partials[static_cast<size_t>(s) * static_cast<size_t>(nt) + static_cast<size_t>(tid)].a;
            };
            auto reduce = [&](const int s) noexcept -> double {
                double acc{0.0};
                for (int t = 0; t < nt; ++t) {
                    acc += partials[static_cast<size_t>(s) * static_cast<size_t>(nt) + static_cast<size_t>(t)].a;
                }
                return acc;
            };
            auto local_dot = [&](const float* const a, const float* const b) noexcept -> double {
                double acc{0.0};
#pragma omp simd reduction(+ : acc)
                for (py::ssize_t i = lo; i < lo + len; ++i) {
                    acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
                }
                return acc;
            };
            // z = M^-1 r on the owned lines.  Three barriers when two-level.
            auto precondition = [&]() noexcept {
                // Vectorized diagonal Jacobi preconditioner using Eigen Map
                Eigen::Map<Eigen::VectorXf>(z.data() + lo, len) =
                    Eigen::Map<const Eigen::VectorXf>(r.data() + lo, len)
                        .cwiseQuotient(Eigen::Map<const Eigen::VectorXf>(diag + lo, len));

                if (!two_level) {
                    return;
                }
                double* const part = coarse_part.data() + static_cast<size_t>(tid) * static_cast<size_t>(ncoarse);
                std::fill(part, part + ncoarse, 0.0);
                for (int line = lb; line < le; ++line) {
                    const int l = line / nr;
                    const int yy = line - (l * nr);
                    const py::ssize_t cbase = (static_cast<py::ssize_t>(l) * static_cast<py::ssize_t>(coarse_rows) + static_cast<py::ssize_t>(yy / block)) * static_cast<py::ssize_t>(coarse_cols);
                    const float* const rl = r.data() + static_cast<py::ssize_t>(line) * static_cast<py::ssize_t>(nc);
                    const float* const ml = op.free_mask + static_cast<py::ssize_t>(line) * static_cast<py::ssize_t>(nc);
                    for (int xi = 0; xi < nc; ++xi) {
                        part[cbase + static_cast<py::ssize_t>(xi / block)] += static_cast<double>(rl[xi] * ml[xi]);
                    }
                }
#pragma omp barrier
                const py::ssize_t cb = (ncoarse * static_cast<py::ssize_t>(tid)) / static_cast<py::ssize_t>(nt);
                const py::ssize_t ce = (ncoarse * static_cast<py::ssize_t>(tid + 1)) / static_cast<py::ssize_t>(nt);
                for (py::ssize_t i = cb; i < ce; ++i) {
                    double acc{0.0};
                    for (int t = 0; t < nt; ++t) {
                        acc += coarse_part[static_cast<size_t>(t) * static_cast<size_t>(ncoarse) + static_cast<size_t>(i)];
                    }
                    coarse_r[i] = static_cast<float>(acc);
                }
#pragma omp barrier
                // Eigen GEMV acceleration for dense coarse-space solve:
                // coarse_z[cb..ce) = cinv[cb..ce, :] * coarse_r[:]
                if (ce > cb) {
                    const auto cinv_block = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                        cinv + cb * ncoarse, ce - cb, ncoarse);
                    const auto cr_vec = Eigen::Map<const Eigen::VectorXf>(coarse_r.data(), ncoarse);
                    Eigen::Map<Eigen::VectorXf>(coarse_z.data() + cb, ce - cb) = cinv_block * cr_vec;
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
            slot(0) = local_dot(rhs.data(), rhs.data());
#pragma omp barrier
            const double rhs_norm = std::sqrt(reduce(0));
            if (rhs_norm == 0.0) {
                rel = 0.0;
            } else {
                precondition();
                // Vector copy p = z via Eigen Map
                Eigen::Map<Eigen::VectorXf>(p.data() + lo, len) = Eigen::Map<const Eigen::VectorXf>(z.data() + lo, len);
                slot(2) = local_dot(r.data(), z.data());
#pragma omp barrier
                double rz = reduce(2);
                while (iterations < max_inner_iterations) {
                    op.apply_lines(p.data(), q.data(), lb, le);
                    ++applied;
                    slot(1) = local_dot(p.data(), q.data());
#pragma omp barrier
                    const double curvature = reduce(1);
                    if (!std::isfinite(curvature) || curvature <= 0.0) {
                        bad = true;
                        break;
                    }
                    const float alpha = static_cast<float>(rz / curvature);
                    // Eigen vectorized AXPY for state updates:
                    // xsol += alpha * p
                    // r    -= alpha * q
                    Eigen::Map<Eigen::VectorXf>(xsol + lo, len) += alpha * Eigen::Map<const Eigen::VectorXf>(p.data() + lo, len);
                    Eigen::Map<Eigen::VectorXf>(r.data() + lo, len) -= alpha * Eigen::Map<const Eigen::VectorXf>(q.data() + lo, len);

                    slot(0) = local_dot(r.data(), r.data());
#pragma omp barrier
                    rel = std::sqrt(reduce(0)) / rhs_norm;
                    ++iterations;
                    if (rel <= inner_relative_tolerance) {
                        break;
                    }
                    precondition();
                    slot(2) = local_dot(r.data(), z.data());
#pragma omp barrier
                    const double next_rz = reduce(2);
                    if (!std::isfinite(next_rz) || rz == 0.0) {
                        break;
                    }
                    const float beta = static_cast<float>(next_rz / rz);
                    // Eigen vectorized update: p = z + beta * p
                    Eigen::Map<Eigen::VectorXf>(p.data() + lo, len) =
                        Eigen::Map<const Eigen::VectorXf>(z.data() + lo, len) +
                        beta * Eigen::Map<const Eigen::VectorXf>(p.data() + lo, len);
                    rz = next_rz;
#pragma omp barrier  // p complete before the next stencil
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
    if (not_spd) {
        throw std::runtime_error("inner PCG requires a finite symmetric positive-definite operator");
    }
    return py::make_tuple(correction_out, total_iterations, relative_residual, applications);
}

PYBIND11_MODULE(_layered_dc_native, m) {
    m.doc() = "Fused C++ layered-PCB DC conduction operator (float32 and float64) and two-level inner PCG";
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
    m.attr("openmp") =
#ifdef _OPENMP
        true;
#else
        false;
#endif
}
