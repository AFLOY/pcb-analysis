// Fused CPU kernels for the scalar Maxwell matrix-free MPIR low path.
//
// The operator uses the same node-owned gather ordering as the CUDA kernel in
// ``cuda.py``: one output node visits at most four adjacent Q1 elements and is
// written once, so no assembled matrix and no atomics exist.  The inner GMRES
// keeps the whole restarted Arnoldi cycle in C++ and returns one complex64
// correction per outer MPIR step, exactly like the NumPy implementation.
//
// Compliant with MISRA-C++ principles: explicit types, no C-style casts,
// strict const correctness, RAII, noexcept specifications, and internal
// implementation details hidden within an anonymous namespace.

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
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

#include "pcbcore/fem/scalar_maxwell.hpp"
#include "pcbcore/lane_sum.hpp"

namespace py = pybind11;

// Registered as electrical._pcbcore.scalar_maxwell.
namespace pcb_scalar_maxwell {

using c64 = std::complex<float>;
using c128 = std::complex<double>;

namespace {

template <typename T>
struct Q1OperatorT final {
    using Complex = std::complex<T>;
    const Complex* inverse_mu{nullptr};
    const Complex* reaction{nullptr};
    const Complex* stiffness{nullptr};  // 4x4 row-major, local index 2*ly+lx
    const Complex* mass{nullptr};
    const std::uint8_t* free_nodes{nullptr};
    const T* free_mask{nullptr};  // 1.0 for free nodes, 0.0 for Dirichlet nodes
    int element_rows{0};
    int element_columns{0};
    int threads{1};

    [[nodiscard]] int node_columns() const noexcept { return element_columns + 1; }
    [[nodiscard]] int node_count() const noexcept {
        return (element_rows + 1) * (element_columns + 1);
    }

    [[nodiscard]] Complex gather(const Complex* const x, const int node_y, const int node_x) const noexcept {
        const int ncols = node_columns();
        const int node = (node_y * ncols) + node_x;
        if (free_nodes[node] == 0U) {
            return x[node];
        }
        const int ey0 = (node_y > 0) ? (node_y - 1) : 0;
        const int ey1 = (node_y < element_rows) ? node_y : (element_rows - 1);
        const int ex0 = (node_x > 0) ? (node_x - 1) : 0;
        const int ex1 = (node_x < element_columns) ? node_x : (element_columns - 1);
        Complex acc(static_cast<T>(0), static_cast<T>(0));
        for (int ey = ey0; ey <= ey1; ++ey) {
            for (int ex = ex0; ex <= ex1; ++ex) {
                const int element = (ey * element_columns) + ex;
                const int local_row = (2 * (node_y - ey)) + (node_x - ex);
                const int top_left = (ey * ncols) + ex;
                const int nodes[4] = {top_left, top_left + 1, top_left + ncols, top_left + ncols + 1};
                const Complex imu = inverse_mu[element];
                const Complex rea = reaction[element];
                for (int lc = 0; lc < 4; ++lc) {
                    const int cn = nodes[lc];
                    if (free_nodes[cn] == 0U) {
                        continue;
                    }
                    const int li = (4 * local_row) + lc;
                    acc += ((imu * stiffness[li]) + (rea * mass[li])) * x[cn];
                }
            }
        }
        return acc;
    }

    void apply(const Complex* const x, Complex* const y) const noexcept {
#pragma omp parallel num_threads(threads) if (threads > 1)
        {
            int row_begin{0};
            int row_end{element_rows + 1};
            thread_rows(element_rows + 1, row_begin, row_end);
            apply_rows(x, y, row_begin, row_end);
        }
    }

    static void thread_rows(const int rows, int& row_begin, int& row_end) noexcept {
#ifdef _OPENMP
        const int t = omp_get_num_threads();
        const int id = omp_get_thread_num();
#else
        const int t = 1;
        const int id = 0;
#endif
        row_begin = static_cast<int>((static_cast<long long>(rows) * id) / t);
        row_end = static_cast<int>((static_cast<long long>(rows) * (id + 1)) / t);
    }

    void apply_rows(const Complex* const x, Complex* const y, const int row_begin, const int row_end) const noexcept {
        const int N = node_columns();
        const int C = element_columns;
        const T* const xr = reinterpret_cast<const T*>(x);
        T* const yr = reinterpret_cast<T*>(y);
        const T* const imu = reinterpret_cast<const T*>(inverse_mu);
        const T* const rea = reinterpret_cast<const T*>(reaction);
        const int offsets[4][4] = {
            {-N - 1, -N, -1, 0}, {-N, -N + 1, 0, 1}, {-1, 0, N - 1, N}, {0, 1, N, N + 1}};
        const int local_rows[4] = {3, 2, 1, 0};
        const int element_offsets[4] = {-C - 1, -C, -1, 0};
        T Kr[16]{};
        T Ki[16]{};
        T Mr[16]{};
        T Mi[16]{};
        for (int i = 0; i < 16; ++i) {
            Kr[i] = stiffness[i].real();
            Ki[i] = stiffness[i].imag();
            Mr[i] = mass[i].real();
            Mi[i] = mass[i].imag();
        }

        for (int node_y = row_begin; node_y < row_end; ++node_y) {
            if (node_y == 0 || node_y == element_rows) {
                for (int node_x = 0; node_x < N; ++node_x) {
                    y[(node_y * N) + node_x] = gather(x, node_y, node_x);
                }
                continue;
            }
            y[node_y * N] = gather(x, node_y, 0);
            y[(node_y * N) + C] = gather(x, node_y, C);
            const int row0 = node_y * N;
            const int e_here = node_y * C;
#pragma omp simd
            for (int node_x = 1; node_x < C; ++node_x) {
                const int node = row0 + node_x;
                const int e = e_here + node_x;
                T acc_r{static_cast<T>(0)};
                T acc_i{static_cast<T>(0)};
#pragma GCC unroll 4
                for (int p = 0; p < 4; ++p) {
                    const int el = e + element_offsets[p];
                    const T ar = imu[2 * el];
                    const T ai = imu[(2 * el) + 1];
                    const T br = rea[2 * el];
                    const T bi = rea[(2 * el) + 1];
                    const int lr = local_rows[p];
#pragma GCC unroll 4
                    for (int lc = 0; lc < 4; ++lc) {
                        const int li = (4 * lr) + lc;
                        const T cr = (ar * Kr[li]) - (ai * Ki[li]) + (br * Mr[li]) - (bi * Mi[li]);
                        const T ci = (ar * Ki[li]) + (ai * Kr[li]) + (br * Mi[li]) + (bi * Mr[li]);
                        const int cn = node + offsets[p][lc];
                        const T m = free_mask[cn];
                        const T vr = xr[2 * cn] * m;
                        const T vi = xr[(2 * cn) + 1] * m;
                        acc_r += (cr * vr) - (ci * vi);
                        acc_i += (cr * vi) + (ci * vr);
                    }
                }
                const T f = free_mask[node];
                yr[2 * node] = (f * acc_r) + ((static_cast<T>(1) - f) * xr[2 * node]);
                yr[(2 * node) + 1] = (f * acc_i) + ((static_cast<T>(1) - f) * xr[(2 * node) + 1]);
            }
        }
    }
};

using Q1Operator = Q1OperatorT<float>;
using Q1OperatorHigh = Q1OperatorT<double>;

using pcbcore::FlushSubnormals;

template <typename T>
[[nodiscard]] const T* data_of(const py::array_t<T, py::array::c_style | py::array::forcecast>& a,
                               const py::ssize_t expected, const char* const name) {
    if (a.size() != expected) {
        throw std::invalid_argument(std::string(name) + " has the wrong size");
    }
    return a.data();
}

using ArrC64 = py::array_t<c64, py::array::c_style | py::array::forcecast>;
using ArrC128 = py::array_t<c128, py::array::c_style | py::array::forcecast>;
using ArrU8 = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;
using ArrF32 = py::array_t<float, py::array::c_style | py::array::forcecast>;
using ArrF64 = py::array_t<double, py::array::c_style | py::array::forcecast>;

template <typename T>
[[nodiscard]] Q1OperatorT<T> make_operator_t(
    const py::array_t<std::complex<T>, py::array::c_style | py::array::forcecast>& inverse_mu,
    const py::array_t<std::complex<T>, py::array::c_style | py::array::forcecast>& reaction,
    const py::array_t<std::complex<T>, py::array::c_style | py::array::forcecast>& stiffness,
    const py::array_t<std::complex<T>, py::array::c_style | py::array::forcecast>& mass,
    const ArrU8& free_nodes,
    const py::array_t<T, py::array::c_style | py::array::forcecast>& free_mask,
    const int element_rows, const int element_columns, const int threads) {

    if (element_rows < 1 || element_columns < 1) {
        throw std::invalid_argument("element shape must be positive");
    }
    const py::ssize_t elements = static_cast<py::ssize_t>(element_rows) * static_cast<py::ssize_t>(element_columns);
    const py::ssize_t nodes =
        static_cast<py::ssize_t>(element_rows + 1) * static_cast<py::ssize_t>(element_columns + 1);
    Q1OperatorT<T> op;
    op.inverse_mu = data_of(inverse_mu, elements, "inverse_mu");
    op.reaction = data_of(reaction, elements, "reaction");
    op.stiffness = data_of(stiffness, 16, "stiffness");
    op.mass = data_of(mass, 16, "mass");
    op.free_nodes = data_of(free_nodes, nodes, "free_nodes");
    op.free_mask = data_of(free_mask, nodes, "free_mask");
    op.element_rows = element_rows;
    op.element_columns = element_columns;
    op.threads = (threads < 1) ? 1 : threads;
    return op;
}

// The same operator over the non-owning arrays of the C++ interface.
template <typename T>
[[nodiscard]] Q1OperatorT<T> operator_from(const pcbcore::fem::scalar_maxwell::OperatorView<T>& v, const int threads) {
    if (v.element_rows < 1 || v.element_columns < 1) {
        throw std::invalid_argument("element shape must be positive");
    }
    if (v.inverse_mu == nullptr || v.reaction == nullptr || v.stiffness == nullptr || v.mass == nullptr ||
        v.free_nodes == nullptr || v.free_mask == nullptr) {
        throw std::invalid_argument("operator arrays are missing");
    }
    Q1OperatorT<T> op;
    op.inverse_mu = v.inverse_mu;
    op.reaction = v.reaction;
    op.stiffness = v.stiffness;
    op.mass = v.mass;
    op.free_nodes = v.free_nodes;
    op.free_mask = v.free_mask;
    op.element_rows = v.element_rows;
    op.element_columns = v.element_columns;
    op.threads = (threads < 1) ? 1 : threads;
    return op;
}

template <typename T>
[[nodiscard]] pcbcore::fem::scalar_maxwell::OperatorView<T> view_of(const Q1OperatorT<T>& op) {
    pcbcore::fem::scalar_maxwell::OperatorView<T> v;
    v.inverse_mu = op.inverse_mu;
    v.reaction = op.reaction;
    v.stiffness = op.stiffness;
    v.mass = op.mass;
    v.free_nodes = op.free_nodes;
    v.free_mask = op.free_mask;
    v.element_rows = op.element_rows;
    v.element_columns = op.element_columns;
    return v;
}

// Element-wise vector updates written out: Eigen's complex expressions take
// the leading elements up to an aligned address with a scalar formula and the
// rest with a SIMD one, which round complex products differently, so where the
// vector happened to be allocated changed the bits.  One formula per element
// here, vectorised or not.
inline void axpy_neg(const c64 h, const c64* const v, c64* const w, const py::ssize_t n) noexcept {
    const float hr = h.real();
    const float hi = h.imag();
    const float* const fv = reinterpret_cast<const float*>(v);
    float* const fw = reinterpret_cast<float*>(w);
    for (py::ssize_t i = 0; i < n; ++i) {
        const float vr = fv[2 * i];
        const float vi = fv[(2 * i) + 1];
        fw[2 * i] -= (hr * vr) - (hi * vi);
        fw[(2 * i) + 1] -= (hr * vi) + (hi * vr);
    }
}

inline void axpy_add(const c64 h, const c64* const v, c64* const w, const py::ssize_t n) noexcept {
    const float hr = h.real();
    const float hi = h.imag();
    const float* const fv = reinterpret_cast<const float*>(v);
    float* const fw = reinterpret_cast<float*>(w);
    for (py::ssize_t i = 0; i < n; ++i) {
        const float vr = fv[2 * i];
        const float vi = fv[(2 * i) + 1];
        fw[2 * i] += (hr * vr) - (hi * vi);
        fw[(2 * i) + 1] += (hr * vi) + (hi * vr);
    }
}

inline void scale_into(const float s, const c64* const v, c64* const out, const py::ssize_t n) noexcept {
    const float* const fv = reinterpret_cast<const float*>(v);
    float* const fo = reinterpret_cast<float*>(out);
    for (py::ssize_t i = 0; i < 2 * n; ++i) {
        fo[i] = s * fv[i];
    }
}

inline void divide_into(const c64* const v, const c64* const d, c64* const z, const py::ssize_t n) noexcept {
    const float* const fv = reinterpret_cast<const float*>(v);
    const float* const fd = reinterpret_cast<const float*>(d);
    float* const fz = reinterpret_cast<float*>(z);
#pragma omp simd
    for (py::ssize_t i = 0; i < n; ++i) {
        const float vr = fv[2 * i];
        const float vi = fv[(2 * i) + 1];
        const float dr = fd[2 * i];
        const float di = fd[(2 * i) + 1];
        const float inv = 1.0f / ((dr * dr) + (di * di));
        fz[2 * i] = ((vr * dr) + (vi * di)) * inv;
        fz[(2 * i) + 1] = ((vi * dr) - (vr * di)) * inv;
    }
}

constexpr py::ssize_t kBlock = 1024;

void block_axpy_neg(const c64* const basis, const py::ssize_t n, const c64* const h, const int rows,
                    c64* const w, const py::ssize_t lo, const py::ssize_t len) noexcept {
    for (py::ssize_t i0 = lo; i0 < lo + len; i0 += kBlock) {
        const py::ssize_t m = std::min(kBlock, lo + len - i0);
        for (int k = 0; k < rows; ++k) {
            axpy_neg(h[k], basis + (static_cast<size_t>(k) * static_cast<size_t>(n)) + static_cast<size_t>(i0), w + i0, m);
        }
    }
}

struct alignas(64) Partial final {
    double a{0.0};
    double b{0.0};
    double pad[6]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

void gmres_q1_core(
    const Q1Operator& op,
    const c128* const rhs_in,
    const c64* const diag,
    const double inner_relative_tolerance,
    const int max_inner_iterations,
    const int restart,
    const bool cgs2,
    const bool float_dots,
    c64* const correction,
    int& total_iterations,
    int& applications,
    double& relative_residual) {

    const py::ssize_t n = op.node_count();
    const int max_cycle = restart;
    const int node_rows = op.element_rows + 1;
    const int ncols = op.node_columns();
    const int team = std::max(1, std::min(op.threads, node_rows));
    std::vector<c64> rhs(static_cast<size_t>(n), c64(0.0f, 0.0f));
    std::vector<c64> residual(static_cast<size_t>(n), c64(0.0f, 0.0f));
    std::vector<c64> w(static_cast<size_t>(n), c64(0.0f, 0.0f));
    std::vector<c64> basis(static_cast<size_t>(max_cycle + 1) * static_cast<size_t>(n), c64(0.0f, 0.0f));
    std::vector<c64> zbasis(static_cast<size_t>(max_cycle) * static_cast<size_t>(n), c64(0.0f, 0.0f));
    const int second_pass_slot = max_cycle + 2;
    const int norm_slot = (2 * max_cycle) + 3;
    // One (re, im) partial per slot and node row, summed in row order: the
    // inner products do not depend on the team size or on where the vectors sit.
    std::vector<double> row_partials(static_cast<size_t>(norm_slot + 1) * static_cast<size_t>(node_rows) * 2U, 0.0);
    auto V = [&](const int k) noexcept -> c64* { return basis.data() + (static_cast<size_t>(k) * static_cast<size_t>(n)); };
    auto Z = [&](const int k) noexcept -> c64* { return zbasis.data() + (static_cast<size_t>(k) * static_cast<size_t>(n)); };

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
        int row_begin{0};
        int row_end{node_rows};
        Q1Operator::thread_rows(node_rows, row_begin, row_end);
        const py::ssize_t lo = static_cast<py::ssize_t>(row_begin) * static_cast<py::ssize_t>(ncols);
        const py::ssize_t len = static_cast<py::ssize_t>(row_end - row_begin) * static_cast<py::ssize_t>(ncols);
        auto partial = [&](const int s, const int row) noexcept -> double* {
            return row_partials.data() + (static_cast<size_t>(s) * static_cast<size_t>(node_rows) + static_cast<size_t>(row)) * 2U;
        };
        auto reduce = [&](const int s, double& a, double& b) noexcept {
            a = pcbcore::ordered_sum(node_rows, [&](const std::ptrdiff_t row) { return partial(s, static_cast<int>(row))[0]; });
            b = pcbcore::ordered_sum(node_rows, [&](const std::ptrdiff_t row) { return partial(s, static_cast<int>(row))[1]; });
        };
        auto store_norm2 = [&](const int s, const c64* const v) noexcept {
            for (int row = row_begin; row < row_end; ++row) {
                const float* const f = reinterpret_cast<const float*>(v + static_cast<py::ssize_t>(row) * ncols);
                double* const out = partial(s, row);
                out[0] = pcbcore::lane_sum(2 * static_cast<std::ptrdiff_t>(ncols), [f](const std::ptrdiff_t i) {
                    return static_cast<double>(f[i]) * static_cast<double>(f[i]);
                });
                out[1] = 0.0;
            }
        };
        // conj(a) . b over one node row.
        auto row_vdot = [&](const c64* const a, const c64* const b, const int row, double& re, double& im) noexcept {
            const float* const fa = reinterpret_cast<const float*>(a + static_cast<py::ssize_t>(row) * ncols);
            const float* const fb = reinterpret_cast<const float*>(b + static_cast<py::ssize_t>(row) * ncols);
            if (!float_dots) {
                re = pcbcore::lane_sum(ncols, [fa, fb](const std::ptrdiff_t i) {
                    return (static_cast<double>(fa[2 * i]) * static_cast<double>(fb[2 * i])) +
                           (static_cast<double>(fa[2 * i + 1]) * static_cast<double>(fb[2 * i + 1]));
                });
                im = pcbcore::lane_sum(ncols, [fa, fb](const std::ptrdiff_t i) {
                    return (static_cast<double>(fa[2 * i]) * static_cast<double>(fb[2 * i + 1])) -
                           (static_cast<double>(fa[2 * i + 1]) * static_cast<double>(fb[2 * i]));
                });
                return;
            }
            // Float lanes inside blocks of kBlock nodes, the blocks summed in double.
            double sr{0.0};
            double si{0.0};
            for (py::ssize_t i0 = 0; i0 < ncols; i0 += kBlock) {
                const py::ssize_t m = std::min(kBlock, static_cast<py::ssize_t>(ncols) - i0);
                float lr[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                float li[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                for (py::ssize_t i = 0; i < m; ++i) {
                    const float ar = fa[2 * (i0 + i)];
                    const float ai = fa[(2 * (i0 + i)) + 1];
                    const float br = fb[2 * (i0 + i)];
                    const float bi = fb[(2 * (i0 + i)) + 1];
                    lr[i % 8] += (ar * br) + (ai * bi);
                    li[i % 8] += (ar * bi) - (ai * br);
                }
                sr += ((static_cast<double>(lr[0]) + lr[1]) + (static_cast<double>(lr[2]) + lr[3])) +
                      ((static_cast<double>(lr[4]) + lr[5]) + (static_cast<double>(lr[6]) + lr[7]));
                si += ((static_cast<double>(li[0]) + li[1]) + (static_cast<double>(li[2]) + li[3])) +
                      ((static_cast<double>(li[4]) + li[5]) + (static_cast<double>(li[6]) + li[7]));
            }
            re = sr;
            im = si;
        };
        auto store_vdot = [&](const int s, const c64* const a, const c64* const b) noexcept {
            for (int row = row_begin; row < row_end; ++row) {
                double* const out = partial(s, row);
                row_vdot(a, b, row, out[0], out[1]);
            }
        };

        std::vector<c128> hess(static_cast<size_t>(max_cycle + 1) * static_cast<size_t>(max_cycle), c128(0.0, 0.0));
        std::vector<c128> g(static_cast<size_t>(max_cycle + 1), c128(0.0, 0.0));
        std::vector<c128> cs(static_cast<size_t>(max_cycle), c128(0.0, 0.0));
        std::vector<c128> y(static_cast<size_t>(max_cycle), c128(0.0, 0.0));
        std::vector<double> sn(static_cast<size_t>(max_cycle), 0.0);
        std::vector<double> hre(static_cast<size_t>(max_cycle), 0.0);
        std::vector<double> him(static_cast<size_t>(max_cycle), 0.0);
        std::vector<c64> hcoef(static_cast<size_t>(max_cycle), c64(0.0f, 0.0f));
        auto H = [&](const int r, const int c) noexcept -> c128& {
            return hess[static_cast<size_t>(r) * static_cast<size_t>(max_cycle) + static_cast<size_t>(c)];
        };
        int iterations{0};
        int applied{0};
        double rel{1.0};

        for (py::ssize_t i = lo; i < lo + len; ++i) {
            rhs[static_cast<size_t>(i)] = c64(static_cast<float>(rhs_in[i].real()),
                                              static_cast<float>(rhs_in[i].imag()));
            correction[i] = c64(0.0f, 0.0f);
            residual[static_cast<size_t>(i)] = rhs[static_cast<size_t>(i)];
        }
        store_norm2(0, rhs.data());
#pragma omp barrier
        double rhs_sq{0.0};
        double unused{0.0};
        reduce(0, rhs_sq, unused);
        // Every thread has read slot 0 before any writes the residual norm
        // into it below; without this barrier a fast thread could overwrite
        // it while a slow one still read the right-hand side's norm.
#pragma omp barrier
        const double rhs_norm = std::sqrt(rhs_sq);
        if (rhs_norm == 0.0) {
            rel = 0.0;
        } else {
            const float eps32 = std::numeric_limits<float>::epsilon();
            while (iterations < max_inner_iterations) {
                if (iterations != 0) {
                    op.apply_rows(correction, w.data(), row_begin, row_end);
                    ++applied;
                    for (py::ssize_t i = lo; i < lo + len; ++i) {
                        residual[static_cast<size_t>(i)] = rhs[static_cast<size_t>(i)] - w[static_cast<size_t>(i)];
                    }
                }
                store_norm2(0, residual.data());
#pragma omp barrier
                double beta_sq{0.0};
                reduce(0, beta_sq, unused);
                const double beta = std::sqrt(beta_sq);
                rel = beta / rhs_norm;
                if (rel <= inner_relative_tolerance) {
                    break;
                }

                const int cycle = std::min(restart, max_inner_iterations - iterations);
                const float inv_beta = static_cast<float>(1.0 / beta);
                scale_into(inv_beta, residual.data() + lo, V(0) + lo, len);
                std::fill(hess.begin(), hess.end(), c128(0.0, 0.0));
                std::fill(g.begin(), g.end(), c128(0.0, 0.0));
                g[0] = c128(static_cast<float>(beta), 0.0);
                int accepted{0};
                bool breakdown{false};

                for (int col = 0; col < cycle; ++col) {
                    c64* const z = Z(col);
                    const c64* const v = V(col);
                    divide_into(v + lo, diag + lo, z + lo, len);
#pragma omp barrier
                    op.apply_rows(z, w.data(), row_begin, row_end);
                    ++applied;

                    if (!cgs2) {
                        for (int row = 0; row <= col; ++row) {
                            store_vdot(1 + row, V(row), w.data());
#pragma omp barrier
                            double re{0.0};
                            double im{0.0};
                            reduce(1 + row, re, im);
                            const c64 h(static_cast<float>(re), static_cast<float>(im));
                            H(row, col) = c128(h.real(), h.imag());
                            axpy_neg(h, V(row) + lo, w.data() + lo, len);
                        }
                    } else {
                        for (int pass = 0; pass < 2; ++pass) {
                            const int base = (pass == 0) ? 1 : second_pass_slot;
                            for (int row = 0; row <= col; ++row) {
                                store_vdot(base + row, V(row), w.data());
                            }
#pragma omp barrier
                            for (int row = 0; row <= col; ++row) {
                                double re{0.0};
                                double im{0.0};
                                reduce(base + row, re, im);
                                hcoef[static_cast<size_t>(row)] = c64(static_cast<float>(re), static_cast<float>(im));
                                H(row, col) += c128(hcoef[static_cast<size_t>(row)].real(), hcoef[static_cast<size_t>(row)].imag());
                            }
                            block_axpy_neg(basis.data(), n, hcoef.data(), col + 1, w.data(), lo, len);
                        }
                    }
                    store_norm2(norm_slot, w.data());
#pragma omp barrier
                    double next_sq{0.0};
                    reduce(norm_slot, next_sq, unused);
                    const double next_norm = std::sqrt(next_sq);
                    const float next_norm32 = static_cast<float>(next_norm);
                    H(col + 1, col) = c128(next_norm32, 0.0);
                    breakdown = !(next_norm32 > (eps32 * static_cast<float>(beta)));
                    if (!breakdown) {
                        scale_into(1.0f / next_norm32, w.data() + lo, V(col + 1) + lo, len);
                    }

                    for (int row = 0; row < col; ++row) {
                        const c128 a = H(row, col);
                        const c128 b = H(row + 1, col);
                        H(row, col) = (std::conj(cs[static_cast<size_t>(row)]) * a) + (sn[static_cast<size_t>(row)] * b);
                        H(row + 1, col) = (-sn[static_cast<size_t>(row)] * a) + (cs[static_cast<size_t>(row)] * b);
                    }
                    {
                        const c128 a = H(col, col);
                        const c128 b = H(col + 1, col);
                        const double na = std::abs(a);
                        const double nb = std::abs(b);
                        const double r = std::hypot(na, nb);
                        if (r == 0.0) {
                            cs[static_cast<size_t>(col)] = c128(1.0, 0.0);
                            sn[static_cast<size_t>(col)] = 0.0;
                        } else if (na == 0.0) {
                            cs[static_cast<size_t>(col)] = c128(0.0, 0.0);
                            sn[static_cast<size_t>(col)] = 1.0;
                        } else {
                            cs[static_cast<size_t>(col)] = (a / na) * (na / r);
                            sn[static_cast<size_t>(col)] = nb / r;
                        }
                        H(col, col) = (std::conj(cs[static_cast<size_t>(col)]) * a) + (sn[static_cast<size_t>(col)] * b);
                        H(col + 1, col) = c128(0.0, 0.0);
                        const c128 g0 = g[static_cast<size_t>(col)];
                        g[static_cast<size_t>(col)] = std::conj(cs[static_cast<size_t>(col)]) * g0;
                        g[static_cast<size_t>(col + 1)] = -sn[static_cast<size_t>(col)] * g0;
                    }
                    accepted = col + 1;
                    rel = std::abs(g[static_cast<size_t>(accepted)]) / rhs_norm;
                    ++iterations;
                    if (rel <= inner_relative_tolerance || breakdown) {
                        break;
                    }
                }

                for (int row = accepted - 1; row >= 0; --row) {
                    c128 acc = g[static_cast<size_t>(row)];
                    for (int c = row + 1; c < accepted; ++c) {
                        acc -= H(row, c) * y[static_cast<size_t>(c)];
                    }
                    y[static_cast<size_t>(row)] = acc / H(row, row);
                }
                for (int k = 0; k < accepted; ++k) {
                    const c64 yk(static_cast<float>(y[static_cast<size_t>(k)].real()),
                                 static_cast<float>(y[static_cast<size_t>(k)].imag()));
                    axpy_add(yk, Z(k) + lo, correction + lo, len);
                }
#pragma omp barrier
                if (rel <= inner_relative_tolerance) {
                    break;
                }
            }
        }
        if (tid == 0) {
            total_iterations = iterations;
            applications = applied;
            relative_residual = rel;
        }
    }
}

}  // namespace

ArrC64 apply_q1(ArrC64 vector, ArrC64 inverse_mu, ArrC64 reaction, ArrC64 stiffness,
                ArrC64 mass, ArrU8 free_nodes, ArrF32 free_mask, int element_rows,
                int element_columns, int threads) {
    const Q1Operator op = make_operator_t<float>(inverse_mu, reaction, stiffness, mass, free_nodes,
                                                 free_mask, element_rows, element_columns, threads);
    const py::ssize_t n = op.node_count();
    const c64* const x = data_of(vector, n, "vector");
    ArrC64 out(n);
    {
        py::gil_scoped_release release;
        const FlushSubnormals flush{};
        op.apply(x, out.mutable_data());
    }
    return out;
}

ArrC128 apply_q1_f64(ArrC128 vector, ArrC128 inverse_mu, ArrC128 reaction, ArrC128 stiffness,
                     ArrC128 mass, ArrU8 free_nodes, ArrF64 free_mask, int element_rows,
                     int element_columns, int threads) {
    const Q1OperatorHigh op = make_operator_t<double>(inverse_mu, reaction, stiffness, mass, free_nodes,
                                                      free_mask, element_rows, element_columns, threads);
    const py::ssize_t n = op.node_count();
    const c128* const x = data_of(vector, n, "vector");
    ArrC128 out(n);
    {
        py::gil_scoped_release release;
        op.apply(x, out.mutable_data());
    }
    return out;
}

py::tuple gmres_q1(ArrC128 rhs_high, ArrC64 diagonal, ArrC64 inverse_mu, ArrC64 reaction,
                   ArrC64 stiffness, ArrC64 mass, ArrU8 free_nodes, ArrF32 free_mask,
                   int element_rows, int element_columns, double inner_relative_tolerance,
                   int max_inner_iterations, int restart, int threads, bool cgs2,
                   bool float_dots) {
    const Q1Operator op = make_operator_t<float>(inverse_mu, reaction, stiffness, mass, free_nodes,
                                                 free_mask, element_rows, element_columns, threads);
    const py::ssize_t n = op.node_count();
    const c128* const rhs_in = data_of(rhs_high, n, "rhs");
    const c64* const diag = data_of(diagonal, n, "diagonal");
    if (max_inner_iterations < 1 || restart < 2) {
        throw std::invalid_argument("iteration limits are invalid");
    }

    ArrC64 correction_out(n);
    c64* const correction = correction_out.mutable_data();
    int total_iterations{0};
    int applications{0};
    double relative_residual{1.0};

    {
        py::gil_scoped_release release;
        gmres_q1_core(op, rhs_in, diag, inner_relative_tolerance, max_inner_iterations,
                      restart, cgs2, float_dots, correction, total_iterations,
                      applications, relative_residual);
    }
    return py::make_tuple(correction_out, total_iterations, relative_residual, applications);
}

// The end-to-end MPIR solve of pcbcore::fem::scalar_maxwell::solve_mpir,
// behind array checks.
py::tuple solve_mpir_scalar_maxwell_q1(
    ArrC128 rhs_high, ArrC128 initial_guess, ArrC64 diagonal,
    ArrC64 inverse_mu_f32, ArrC64 reaction_f32, ArrC64 stiffness_f32, ArrC64 mass_f32,
    ArrU8 free_nodes, ArrF32 free_mask_f32,
    ArrC128 inverse_mu_f64, ArrC128 reaction_f64, ArrC128 stiffness_f64, ArrC128 mass_f64,
    ArrF64 free_mask_f64,
    int element_rows, int element_columns,
    double relative_tolerance, double absolute_tolerance, double inner_relative_tolerance,
    int max_outer_iterations, int max_inner_iterations, int restart,
    int threads, bool cgs2, bool float_dots) {

    const Q1Operator op_low = make_operator_t<float>(
        inverse_mu_f32, reaction_f32, stiffness_f32, mass_f32, free_nodes, free_mask_f32,
        element_rows, element_columns, threads);
    const Q1OperatorHigh op_high = make_operator_t<double>(
        inverse_mu_f64, reaction_f64, stiffness_f64, mass_f64, free_nodes, free_mask_f64,
        element_rows, element_columns, threads);

    const py::ssize_t n = op_low.node_count();
    const c128* const rhs_in = data_of(rhs_high, n, "rhs_high");
    const c64* const diag = data_of(diagonal, n, "diagonal");

    ArrC128 solution_out(n);
    c128* const sol = solution_out.mutable_data();
    if (initial_guess.size() == n) {
        const c128* const init_ptr = initial_guess.data();
        std::copy(init_ptr, init_ptr + n, sol);
    } else {
        std::fill(sol, sol + n, c128(0.0, 0.0));
    }
    pcbcore::fem::MpirConfig config;
    config.relative_tolerance = relative_tolerance;
    config.absolute_tolerance = absolute_tolerance;
    config.inner_relative_tolerance = inner_relative_tolerance;
    config.max_outer_iterations = max_outer_iterations;
    config.max_inner_iterations = max_inner_iterations;
    pcbcore::fem::scalar_maxwell::GmresOptions gmres;
    gmres.restart = restart;
    gmres.cgs2 = cgs2;
    gmres.float_dots = float_dots;
    pcbcore::fem::MpirResult result;
    {
        py::gil_scoped_release release;
        result = pcbcore::fem::scalar_maxwell::solve_mpir(view_of(op_low), view_of(op_high), rhs_in, sol, diag,
                                                          config, gmres, op_low.threads);
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

int default_threads() noexcept {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void register_module(py::module_& m) {
    m.doc() = "Fused C++ Q1 scalar Maxwell operator, complex64 inner GMRES and end-to-end MPIR solver";
    m.def("apply_q1", &apply_q1, py::arg("vector"), py::arg("inverse_mu"), py::arg("reaction"),
          py::arg("stiffness"), py::arg("mass"), py::arg("free_nodes"), py::arg("free_mask"),
          py::arg("element_rows"), py::arg("element_columns"), py::arg("threads") = 1);
    m.def("apply_q1_f64", &apply_q1_f64, py::arg("vector"), py::arg("inverse_mu"), py::arg("reaction"),
          py::arg("stiffness"), py::arg("mass"), py::arg("free_nodes"), py::arg("free_mask"),
          py::arg("element_rows"), py::arg("element_columns"), py::arg("threads") = 1);
    m.def("gmres_q1", &gmres_q1, py::arg("rhs_high"), py::arg("diagonal"), py::arg("inverse_mu"),
          py::arg("reaction"), py::arg("stiffness"), py::arg("mass"), py::arg("free_nodes"),
          py::arg("free_mask"), py::arg("element_rows"), py::arg("element_columns"),
          py::arg("inner_relative_tolerance"), py::arg("max_inner_iterations"),
          py::arg("restart"), py::arg("threads") = 1, py::arg("cgs2") = false,
          py::arg("float_dots") = false);
    m.def("solve_mpir_scalar_maxwell_q1", &solve_mpir_scalar_maxwell_q1,
          py::arg("rhs_high"), py::arg("initial_guess"), py::arg("diagonal"),
          py::arg("inverse_mu_f32"), py::arg("reaction_f32"), py::arg("stiffness_f32"), py::arg("mass_f32"),
          py::arg("free_nodes"), py::arg("free_mask_f32"),
          py::arg("inverse_mu_f64"), py::arg("reaction_f64"), py::arg("stiffness_f64"), py::arg("mass_f64"),
          py::arg("free_mask_f64"),
          py::arg("element_rows"), py::arg("element_columns"),
          py::arg("relative_tolerance"), py::arg("absolute_tolerance"), py::arg("inner_relative_tolerance"),
          py::arg("max_outer_iterations"), py::arg("max_inner_iterations"), py::arg("restart"),
          py::arg("threads") = 1, py::arg("cgs2") = false, py::arg("float_dots") = false);
    m.def("default_threads", &default_threads);
    m.attr("openmp") =
#ifdef _OPENMP
        true;
#else
        false;
#endif
}

}  // namespace pcb_scalar_maxwell

namespace pcbcore::fem::scalar_maxwell {

using pcb_scalar_maxwell::Q1OperatorT;

void apply_high(const OperatorView<double>& view, const std::complex<double>* const x, std::complex<double>* const y,
                const int threads) {
    const Q1OperatorT<double> op = pcb_scalar_maxwell::operator_from(view, threads);
    op.apply(x, y);
}

void apply_low(const OperatorView<float>& view, const std::complex<float>* const x, std::complex<float>* const y,
               const int threads) {
    const Q1OperatorT<float> op = pcb_scalar_maxwell::operator_from(view, threads);
    op.apply(x, y);
}

MpirResult solve_mpir(const OperatorView<float>& low_view, const OperatorView<double>& high_view,
                      const std::complex<double>* const rhs_in, std::complex<double>* const sol,
                      const std::complex<float>* const diag, const MpirConfig& config, const GmresOptions& options,
                      const int threads) {
    using pcb_scalar_maxwell::c128;
    using pcb_scalar_maxwell::c64;
    const Q1OperatorT<float> op_low = pcb_scalar_maxwell::operator_from(low_view, threads);
    const Q1OperatorT<double> op_high = pcb_scalar_maxwell::operator_from(high_view, threads);
    const py::ssize_t n = op_low.node_count();
    const double relative_tolerance = config.relative_tolerance;
    const double absolute_tolerance = config.absolute_tolerance;
    const double inner_relative_tolerance = config.inner_relative_tolerance;
    const int max_outer_iterations = config.max_outer_iterations;
    const int max_inner_iterations = config.max_inner_iterations;
    const int restart = options.restart;
    const bool cgs2 = options.cgs2;
    const bool float_dots = options.float_dots;
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
        double rhs_sq{0.0};
        for (py::ssize_t i = 0; i < n; ++i) {
            const double r = rhs_in[i].real();
            const double im = rhs_in[i].imag();
            rhs_sq += (r * r) + (im * im);
        }
        const double rhs_norm = std::sqrt(rhs_sq);
        const double scale = (rhs_norm > 0.0) ? rhs_norm : 1.0;
        const double target = absolute_tolerance + (relative_tolerance * scale);

        std::vector<c128> residual(static_cast<size_t>(n), c128(0.0, 0.0));
        std::vector<c128> Ax(static_cast<size_t>(n), c128(0.0, 0.0));
        std::vector<c64> correction(static_cast<size_t>(n), c64(0.0f, 0.0f));

        for (int outer = 0; outer <= max_outer_iterations; ++outer) {
            outer_iterations = outer;
            op_high.apply(sol, Ax.data());
            ++total_high_apps;

            double res_sq{0.0};
            for (py::ssize_t i = 0; i < n; ++i) {
                residual[static_cast<size_t>(i)] = rhs_in[i] - Ax[static_cast<size_t>(i)];
                const double rr = residual[static_cast<size_t>(i)].real();
                const double ri = residual[static_cast<size_t>(i)].imag();
                res_sq += (rr * rr) + (ri * ri);
            }
            const double res_norm = std::sqrt(res_sq);
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

            gmres_q1_core(
                op_low, residual.data(), diag, inner_relative_tolerance, max_inner_iterations,
                restart, cgs2, float_dots, correction.data(), inner_iters, inner_apps, inner_rel);

            total_inner_iterations += inner_iters;
            total_low_apps += inner_apps;
            history.emplace_back(outer + 1, relative_residual, inner_iters, inner_rel);

            for (py::ssize_t i = 0; i < n; ++i) {
                sol[i] += c128(static_cast<double>(correction[static_cast<size_t>(i)].real()),
                               static_cast<double>(correction[static_cast<size_t>(i)].imag()));
            }
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

}  // namespace pcbcore::fem::scalar_maxwell

