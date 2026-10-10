// Fused CPU kernels for the thermal matrix-free MPIR low path.
//
// The operator is the node-owned gather of the trilinear (hexahedral Q1)
// conduction action used by the CUDA kernel in ``cuda.py``: one output node
// visits its at most eight adjacent element slabs, applies the two per-slab
// 8x8 unit tensors weighted by the element conductivities, adds the lumped
// Robin conductance, and is written once.  The same gather runs in float32
// for the low path and in float64 for the outer MPIR residual.  The inner PCG with the two-level
// preconditioner (Jacobi plus patch-constant coarse correction through a
// dense float32 inverse) runs as one SPMD OpenMP region per outer MPIR step.
//
// Compliant with MISRA-C++ principles: explicit types, no C-style casts,
// strict const correctness, RAII, noexcept specifications, and internal
// implementation details hidden within an anonymous namespace.

#include <pybind11/numpy.h>
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
struct HexOperatorT final {
    const T* coef{nullptr};       // (3, slabs, rows, cols): a_x, a_y, a_z per element
    const T* unit{nullptr};       // (3, 8, 8): U_x, U_y, U_z
    const T* robin{nullptr};      // (nodes,)
    const std::uint8_t* free_nodes{nullptr};
    const T* free_mask{nullptr};  // 1 free, 0 fixed
    int slabs{0};
    int rows{0};
    int cols{0};   // element grid
    int threads{1};

    [[nodiscard]] int node_layers() const noexcept { return slabs + 1; }
    [[nodiscard]] int node_rows() const noexcept { return rows + 1; }
    [[nodiscard]] int node_cols() const noexcept { return cols + 1; }
    [[nodiscard]] int lines() const noexcept { return node_layers() * node_rows(); }
    [[nodiscard]] py::ssize_t element_count() const noexcept {
        return static_cast<py::ssize_t>(slabs) * static_cast<py::ssize_t>(rows) * static_cast<py::ssize_t>(cols);
    }
    [[nodiscard]] py::ssize_t node_count() const noexcept {
        return static_cast<py::ssize_t>(lines()) * static_cast<py::ssize_t>(node_cols());
    }

    // Generic gather for one node, same ordering as the CUDA kernel.
    [[nodiscard]] T gather(const T* const x, const int z, const int y, const int xi) const noexcept {
        const int nr = node_rows();
        const int nc = node_cols();
        const int plane = nr * nc;
        const py::ssize_t node = (static_cast<py::ssize_t>(z) * nr + static_cast<py::ssize_t>(y)) * nc + static_cast<py::ssize_t>(xi);
        if (free_nodes[node] == 0U) {
            return x[node];
        }
        const int z0 = (z > 0) ? (z - 1) : 0;
        const int z1 = (z < slabs) ? z : (slabs - 1);
        const int y0 = (y > 0) ? (y - 1) : 0;
        const int y1 = (y < rows) ? y : (rows - 1);
        const int x0 = (xi > 0) ? (xi - 1) : 0;
        const int x1 = (xi < cols) ? xi : (cols - 1);
        const py::ssize_t ne = element_count();
        const T* const ux = unit;
        const T* const uy = unit + 64;
        const T* const uz = unit + 128;
        T acc{static_cast<T>(0)};
        for (int ez = z0; ez <= z1; ++ez) {
            for (int ey = y0; ey <= y1; ++ey) {
                for (int ex = x0; ex <= x1; ++ex) {
                    const int lr = (4 * (z - ez)) + (2 * (y - ey)) + (xi - ex);
                    const py::ssize_t e = (static_cast<py::ssize_t>(ez * rows + ey)) * cols + static_cast<py::ssize_t>(ex);
                    const T a = coef[e];
                    const T b = coef[ne + e];
                    const T d = coef[(2 * ne) + e];
                    const py::ssize_t corner = (static_cast<py::ssize_t>(ez) * nr + static_cast<py::ssize_t>(ey)) * nc + static_cast<py::ssize_t>(ex);
                    for (int c = 0; c < 8; ++c) {
                        const py::ssize_t cn = corner + (static_cast<py::ssize_t>(c >> 2) * plane) + (static_cast<py::ssize_t>((c >> 1) & 1) * nc) + static_cast<py::ssize_t>(c & 1);
                        if (free_nodes[cn] == 0U) {
                            continue;
                        }
                        acc += ((a * ux[(8 * lr) + c]) + (b * uy[(8 * lr) + c]) + (d * uz[(8 * lr) + c])) * x[cn];
                    }
                }
            }
        }
        return acc + (robin[node] * x[node]);
    }

    void apply_lines(const T* const x, T* const out, const int line_begin, const int line_end) const noexcept {
        const int nr = node_rows();
        const int nc = node_cols();
        const int plane = nr * nc;
        for (int line = line_begin; line < line_end; ++line) {
            const int z = line / nr;
            const int y = line - (z * nr);
            const py::ssize_t base = static_cast<py::ssize_t>(line) * static_cast<py::ssize_t>(nc);
            out[base] = gather(x, z, y, 0);
            out[base + cols] = gather(x, z, y, cols);
            if (cols < 2) {
                continue;
            }
            const int z0 = (z > 0) ? (z - 1) : 0;
            const int z1 = (z < slabs) ? z : (slabs - 1);
            const int y0 = (y > 0) ? (y - 1) : 0;
            const int y1 = (y < rows) ? y : (rows - 1);
            T* const o = out + base;
            for (int xi = 1; xi < cols; ++xi) {
                o[xi] = static_cast<T>(0);
            }
            const py::ssize_t ne = element_count();
            const T* const ux = unit;
            const T* const uy = unit + 64;
            const T* const uz = unit + 128;
            for (int ez = z0; ez <= z1; ++ez) {
                for (int ey = y0; ey <= y1; ++ey) {
                    const int lr_base = (4 * (z - ez)) + (2 * (y - ey));
                    const py::ssize_t e_row = (static_cast<py::ssize_t>(ez * rows + ey)) * cols;
                    const T* const ax = coef + e_row;
                    const T* const ay = coef + ne + e_row;
                    const T* const az = coef + (2 * ne) + e_row;
                    const py::ssize_t corner_row = (static_cast<py::ssize_t>(ez * nr + ey)) * nc;
#pragma omp simd
                    for (int xi = 1; xi < cols; ++xi) {
                        T acc{static_cast<T>(0)};
#pragma GCC unroll 2
                        for (int dx = 0; dx < 2; ++dx) {
                            const int ex = xi - 1 + dx;
                            const int lr = lr_base + (1 - dx);
                            const T a = ax[ex];
                            const T b = ay[ex];
                            const T d = az[ex];
                            const py::ssize_t corner = corner_row + static_cast<py::ssize_t>(ex);
#pragma GCC unroll 8
                            for (int c = 0; c < 8; ++c) {
                                const py::ssize_t cn = corner + (static_cast<py::ssize_t>(c >> 2) * plane) + (static_cast<py::ssize_t>((c >> 1) & 1) * nc) + static_cast<py::ssize_t>(c & 1);
                                const T w = (a * ux[(8 * lr) + c]) + (b * uy[(8 * lr) + c]) + (d * uz[(8 * lr) + c]);
                                acc += w * (x[cn] * free_mask[cn]);
                            }
                        }
                        o[xi] += acc;
                    }
                }
            }
#pragma omp simd
            for (int xi = 1; xi < cols; ++xi) {
                const py::ssize_t node = base + static_cast<py::ssize_t>(xi);
                const T f = free_mask[node];
                o[xi] = (f * (o[xi] + (robin[node] * x[node]))) + ((static_cast<T>(1) - f) * x[node]);
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

using HexOperator = HexOperatorT<float>;
using HexOperatorHigh = HexOperatorT<double>;

template <typename T>
[[nodiscard]] HexOperatorT<T> make_operator_t(
    const Arr<T>& coef, const Arr<T>& unit, const Arr<T>& robin,
    const ArrU8& free_nodes, const Arr<T>& free_mask,
    const int slabs, const int rows, const int cols, const int threads) {

    if (slabs < 1 || rows < 1 || cols < 1) {
        throw std::invalid_argument("element grid must be positive");
    }
    HexOperatorT<T> op;
    op.slabs = slabs;
    op.rows = rows;
    op.cols = cols;
    op.threads = (threads < 1) ? 1 : threads;
    const py::ssize_t elements = op.element_count();
    const py::ssize_t nodes = op.node_count();
    op.coef = data_of(coef, 3 * elements, "coefficients");
    op.unit = data_of(unit, static_cast<py::ssize_t>(192), "unit");
    op.robin = data_of(robin, nodes, "robin");
    op.free_nodes = data_of(free_nodes, nodes, "free_nodes");
    op.free_mask = data_of(free_mask, nodes, "free_mask");
    return op;
}

struct alignas(64) Partial final {
    double a{0.0};
    double pad[7]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

template <typename T>
[[nodiscard]] Arr<T> apply_impl(const HexOperatorT<T>& op, const Arr<T>& vector, const bool flush_subnormals) {
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
                HexOperatorT<T>::thread_lines(lines, b, e);
                op.apply_lines(x, y, b, e);
            } else {
                int b{0};
                int e{lines};
                HexOperatorT<T>::thread_lines(lines, b, e);
                op.apply_lines(x, y, b, e);
            }
        }
    }
    return out;
}

// Internal PCG core routine for Hex Q1 thermal operator.
void pcg_hex_q1_core(
    const HexOperator& op,
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
    const int nl = op.node_layers();
    const int coarse_rows = (nr + block - 1) / block;
    const int coarse_cols = (nc + block - 1) / block;
    const py::ssize_t ncoarse = static_cast<py::ssize_t>(nl) * static_cast<py::ssize_t>(coarse_rows) * static_cast<py::ssize_t>(coarse_cols);
    const bool two_level = (cinv != nullptr);

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
        HexOperator::thread_lines(lines, lb, le);
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
        auto precondition = [&]() noexcept {
            Eigen::Map<Eigen::VectorXf>(z.data() + lo, len) =
                Eigen::Map<const Eigen::VectorXf>(r.data() + lo, len)
                    .cwiseQuotient(Eigen::Map<const Eigen::VectorXf>(diag + lo, len));

            if (!two_level) {
                return;
            }
            double* const part = coarse_part.data() + static_cast<size_t>(tid) * static_cast<size_t>(ncoarse);
            std::fill(part, part + ncoarse, 0.0);
            for (int line = lb; line < le; ++line) {
                const int zz = line / nr;
                const int yy = line - (zz * nr);
                const py::ssize_t cbase = (static_cast<py::ssize_t>(zz) * static_cast<py::ssize_t>(coarse_rows) + static_cast<py::ssize_t>(yy / block)) * static_cast<py::ssize_t>(coarse_cols);
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
                const int zz = line / nr;
                const int yy = line - (zz * nr);
                const py::ssize_t cbase = (static_cast<py::ssize_t>(zz) * static_cast<py::ssize_t>(coarse_rows) + static_cast<py::ssize_t>(yy / block)) * static_cast<py::ssize_t>(coarse_cols);
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

ArrF32 apply_hex_q1(ArrF32 vector, ArrF32 coefficients, ArrF32 unit, ArrF32 robin,
                    ArrU8 free_nodes, ArrF32 free_mask, int slabs, int rows, int cols,
                    int threads) {
    const HexOperator op = make_operator_t<float>(
        coefficients, unit, robin, free_nodes, free_mask, slabs, rows, cols, threads);
    return apply_impl<float>(op, vector, true);
}

ArrF64 apply_hex_q1_f64(ArrF64 vector, ArrF64 coefficients, ArrF64 unit, ArrF64 robin,
                        ArrU8 free_nodes, ArrF64 free_mask, int slabs, int rows, int cols,
                        int threads) {
    const HexOperatorHigh op = make_operator_t<double>(
        coefficients, unit, robin, free_nodes, free_mask, slabs, rows, cols, threads);
    return apply_impl<double>(op, vector, false);
}

py::tuple pcg_hex_q1(ArrF64 rhs_high, ArrF32 diagonal, ArrF32 coef, ArrF32 unit,
                     ArrF32 robin, ArrU8 free_nodes, ArrF32 free_mask,
                     int slabs, int rows, int cols, int block,
                     ArrF32 coarse_inverse, double inner_relative_tolerance,
                     int max_inner_iterations, int threads) {
    const HexOperator op = make_operator_t<float>(
        coef, unit, robin, free_nodes, free_mask, slabs, rows, cols, threads);
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
    const int nl = op.node_layers();
    const int coarse_rows = (nr + block - 1) / block;
    const int coarse_cols = (nc + block - 1) / block;
    const py::ssize_t ncoarse = static_cast<py::ssize_t>(nl) * static_cast<py::ssize_t>(coarse_rows) * static_cast<py::ssize_t>(coarse_cols);
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
        pcg_hex_q1_core(op, rhs_in, diag, block, cinv, inner_relative_tolerance,
                        max_inner_iterations, xsol, total_iterations, applications,
                        relative_residual, not_spd);
    }
    if (not_spd) {
        throw std::runtime_error("inner PCG requires a finite symmetric positive-definite operator");
    }
    return py::make_tuple(correction_out, total_iterations, relative_residual, applications);
}

// Full C++ Two-Level Coarse Matrix Assembly & Inversion for Thermal Hexahedral Operator.
ArrF64 assemble_coarse_inverse_hex(
    ArrF64 coef, ArrF64 unit, ArrF64 robin, ArrU8 free_nodes, ArrF64 free_mask,
    int slabs, int rows, int cols, int block, int threads) {

    const HexOperatorHigh op = make_operator_t<double>(
        coef, unit, robin, free_nodes, free_mask, slabs, rows, cols, threads);

    const int nl = op.node_layers();
    const int nr = op.node_rows();
    const int nc = op.node_cols();
    const int coarse_rows = (nr + block - 1) / block;
    const int coarse_cols = (nc + block - 1) / block;
    const py::ssize_t ncoarse = static_cast<py::ssize_t>(nl) * static_cast<py::ssize_t>(coarse_rows) * static_cast<py::ssize_t>(coarse_cols);
    const py::ssize_t n_nodes = op.node_count();

    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(ncoarse, ncoarse);
    Eigen::VectorXd counts = Eigen::VectorXd::Zero(ncoarse);

    std::vector<double> fine(static_cast<size_t>(n_nodes), 0.0);
    std::vector<double> action(static_cast<size_t>(n_nodes), 0.0);
    std::vector<double> restricted(static_cast<size_t>(ncoarse), 0.0);

    for (int l = 0; l < nl; ++l) {
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
        py::gil_scoped_release release;
        for (int colour = 0; colour < 27; ++colour) {
            bool has_colour{false};
            std::fill(fine.begin(), fine.end(), 0.0);
            for (int l = 0; l < nl; ++l) {
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

            op.apply_lines(fine.data(), action.data(), 0, op.lines());

            std::fill(restricted.begin(), restricted.end(), 0.0);
            for (int l = 0; l < nl; ++l) {
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

            for (int l = 0; l < nl; ++l) {
                for (int yc = 0; yc < coarse_rows; ++yc) {
                    for (int xc = 0; xc < coarse_cols; ++xc) {
                        const py::ssize_t I = (static_cast<py::ssize_t>(l) * coarse_rows + yc) * coarse_cols + xc;
                        for (int dl = -1; dl <= 1; ++dl) {
                            const int sl = l + dl;
                            if (sl < 0 || sl >= nl) continue;
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

    ArrF64 out(ncoarse * ncoarse);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        out.mutable_data(), ncoarse, ncoarse) = inv;
    return out;
}

// Complete C++ End-to-End MPIR Solver for Thermal Hexahedral Q1 Conduction.
py::tuple solve_mpir_thermal_hex(
    ArrF64 rhs_high, ArrF64 initial_guess, ArrF32 diagonal,
    ArrF32 coef_f32, ArrF32 unit_f32, ArrF32 robin_f32, ArrU8 free_nodes, ArrF32 free_mask_f32,
    ArrF64 coef_f64, ArrF64 unit_f64, ArrF64 robin_f64, ArrF64 free_mask_f64,
    int slabs, int rows, int cols, int block, ArrF32 coarse_inverse,
    double relative_tolerance, double absolute_tolerance,
    double inner_relative_tolerance, int max_outer_iterations,
    int max_inner_iterations, int threads) {

    const HexOperator op_low = make_operator_t<float>(
        coef_f32, unit_f32, robin_f32, free_nodes, free_mask_f32, slabs, rows, cols, threads);
    const HexOperatorHigh op_high = make_operator_t<double>(
        coef_f64, unit_f64, robin_f64, free_nodes, free_mask_f64, slabs, rows, cols, threads);

    const py::ssize_t n = op_low.node_count();
    const double* const rhs_in = data_of(rhs_high, n, "rhs_high");
    const float* const diag = data_of(diagonal, n, "diagonal");

    const int nl = op_low.node_layers();
    const int nr = op_low.node_rows();
    const int nc = op_low.node_cols();
    const int coarse_rows = (nr + block - 1) / block;
    const int coarse_cols = (nc + block - 1) / block;
    const py::ssize_t ncoarse = static_cast<py::ssize_t>(nl) * static_cast<py::ssize_t>(coarse_rows) * static_cast<py::ssize_t>(coarse_cols);
    const bool two_level = (coarse_inverse.size() > 0);
    const float* cinv{nullptr};
    if (two_level) {
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

    int outer_iterations{0};
    int total_inner_iterations{0};
    int total_high_apps{0};
    int total_low_apps{0};
    double relative_residual{1.0};
    bool converged{false};

    {
        py::gil_scoped_release release;
        const double rhs_norm = Eigen::Map<const Eigen::VectorXd>(rhs_in, n).norm();
        const double scale = (rhs_norm > 0.0) ? rhs_norm : 1.0;
        const double target = absolute_tolerance + (relative_tolerance * scale);

        std::vector<double> residual(static_cast<size_t>(n), 0.0);
        std::vector<double> Ax(static_cast<size_t>(n), 0.0);
        std::vector<float> correction(static_cast<size_t>(n), 0.0f);

        for (int outer = 0; outer <= max_outer_iterations; ++outer) {
            outer_iterations = outer;
            op_high.apply_lines(sol, Ax.data(), 0, op_high.lines());
            ++total_high_apps;

            for (py::ssize_t i = 0; i < n; ++i) {
                residual[static_cast<size_t>(i)] = rhs_in[i] - Ax[static_cast<size_t>(i)];
            }
            const double res_norm = Eigen::Map<const Eigen::VectorXd>(residual.data(), n).norm();
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

            pcg_hex_q1_core(
                op_low, residual.data(), diag, block, cinv,
                inner_relative_tolerance, max_inner_iterations,
                correction.data(), inner_iters, inner_apps, inner_rel, not_spd);

            if (not_spd) {
                throw std::runtime_error("inner PCG requires a finite SPD operator");
            }

            total_inner_iterations += inner_iters;
            total_low_apps += inner_apps;

            Eigen::Map<Eigen::VectorXd>(sol, n) +=
                Eigen::Map<const Eigen::VectorXf>(correction.data(), n).cast<double>();
        }
    }

    return py::make_tuple(
        solution_out, converged, outer_iterations, total_inner_iterations,
        relative_residual, total_high_apps, total_low_apps);
}

PYBIND11_MODULE(_thermal_native, m) {
    m.doc() = "Fused C++ hexahedral Q1 conduction operator (float32 and float64), coarse assembly and full MPIR solver";
    m.def("apply_hex_q1", &apply_hex_q1, py::arg("vector"), py::arg("coefficients"), py::arg("unit"),
          py::arg("robin"), py::arg("free_nodes"),
          py::arg("free_mask"), py::arg("slabs"), py::arg("rows"), py::arg("cols"),
          py::arg("threads") = 1);
    m.def("apply_hex_q1_f64", &apply_hex_q1_f64, py::arg("vector"), py::arg("coefficients"),
          py::arg("unit"), py::arg("robin"), py::arg("free_nodes"), py::arg("free_mask"),
          py::arg("slabs"), py::arg("rows"), py::arg("cols"), py::arg("threads") = 1);
    m.def("pcg_hex_q1", &pcg_hex_q1, py::arg("rhs_high"), py::arg("diagonal"), py::arg("coefficients"),
          py::arg("unit"), py::arg("robin"),
          py::arg("free_nodes"), py::arg("free_mask"), py::arg("slabs"), py::arg("rows"),
          py::arg("cols"), py::arg("block"), py::arg("coarse_inverse"),
          py::arg("inner_relative_tolerance"), py::arg("max_inner_iterations"),
          py::arg("threads") = 1);
    m.def("assemble_coarse_inverse_hex", &assemble_coarse_inverse_hex,
          py::arg("coefficients"), py::arg("unit"), py::arg("robin"),
          py::arg("free_nodes"), py::arg("free_mask"),
          py::arg("slabs"), py::arg("rows"), py::arg("cols"), py::arg("block"), py::arg("threads") = 1);
    m.def("solve_mpir_thermal_hex", &solve_mpir_thermal_hex,
          py::arg("rhs_high"), py::arg("initial_guess"), py::arg("diagonal"),
          py::arg("coefficients_f32"), py::arg("unit_f32"), py::arg("robin_f32"),
          py::arg("free_nodes"), py::arg("free_mask_f32"),
          py::arg("coefficients_f64"), py::arg("unit_f64"), py::arg("robin_f64"),
          py::arg("free_mask_f64"),
          py::arg("slabs"), py::arg("rows"), py::arg("cols"), py::arg("block"), py::arg("coarse_inverse"),
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
