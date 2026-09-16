// Fused CPU kernels for the scalar Maxwell matrix-free MPIR low path.
//
// The operator uses the same node-owned gather ordering as the CUDA kernel in
// ``cuda.py``: one output node visits at most four adjacent Q1 elements and is
// written once, so no assembled matrix and no atomics exist.  The inner GMRES
// keeps the whole restarted Arnoldi cycle in C++ and returns one complex64
// correction per outer MPIR step, exactly like the NumPy implementation.

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <xmmintrin.h>

namespace py = pybind11;

using c64 = std::complex<float>;
using c128 = std::complex<double>;

namespace {

struct Q1Operator {
    const c64* inverse_mu;
    const c64* reaction;
    const c64* stiffness;  // 4x4 row-major, local index 2*ly+lx
    const c64* mass;
    const std::uint8_t* free_nodes;
    const float* free_mask;  // 1.0f for free nodes, 0.0f for Dirichlet nodes
    int element_rows;
    int element_columns;
    int threads;

    int node_columns() const { return element_columns + 1; }
    int node_count() const { return (element_rows + 1) * (element_columns + 1); }

    // Generic node-owned gather for one node: identical ordering to the CUDA
    // kernel.  Used for the boundary rows/columns.
    c64 gather(const c64* x, int node_y, int node_x) const {
        const int ncols = node_columns();
        const int node = node_y * ncols + node_x;
        if (!free_nodes[node]) return x[node];
        const int ey0 = node_y > 0 ? node_y - 1 : 0;
        const int ey1 = node_y < element_rows ? node_y : element_rows - 1;
        const int ex0 = node_x > 0 ? node_x - 1 : 0;
        const int ex1 = node_x < element_columns ? node_x : element_columns - 1;
        c64 acc(0.0f, 0.0f);
        for (int ey = ey0; ey <= ey1; ++ey) {
            for (int ex = ex0; ex <= ex1; ++ex) {
                const int element = ey * element_columns + ex;
                const int local_row = 2 * (node_y - ey) + (node_x - ex);
                const int top_left = ey * ncols + ex;
                const int nodes[4] = {top_left, top_left + 1, top_left + ncols,
                                      top_left + ncols + 1};
                const c64 imu = inverse_mu[element];
                const c64 rea = reaction[element];
                for (int lc = 0; lc < 4; ++lc) {
                    const int cn = nodes[lc];
                    if (!free_nodes[cn]) continue;
                    const int li = 4 * local_row + lc;
                    acc += (imu * stiffness[li] + rea * mass[li]) * x[cn];
                }
            }
        }
        return acc;
    }

    // y = A x  (complex64).  Interior nodes use a branch-free 16-term form
    // over the four adjacent elements so the x-loop vectorises; the Dirichlet
    // mask enters as a 0/1 multiplier, which is the same arithmetic as
    // skipping the column.  Dirichlet rows copy x.
    void apply(const c64* x, c64* y) const {
#pragma omp parallel num_threads(threads) if (threads > 1)
        {
            int row_begin = 0, row_end = element_rows + 1;
            thread_rows(element_rows + 1, row_begin, row_end);
            apply_rows(x, y, row_begin, row_end);
        }
    }

    // Static row partition of ``rows`` node rows for the calling OpenMP thread.
    static void thread_rows(int rows, int& row_begin, int& row_end) {
#ifdef _OPENMP
        const int t = omp_get_num_threads();
        const int id = omp_get_thread_num();
#else
        const int t = 1, id = 0;
#endif
        row_begin = static_cast<int>(static_cast<long long>(rows) * id / t);
        row_end = static_cast<int>(static_cast<long long>(rows) * (id + 1) / t);
    }

    // y[rows row_begin..row_end) = (A x)[same rows].  Reads x on neighbouring
    // rows, so callers must finish writing x before entering.
    void apply_rows(const c64* x, c64* y, int row_begin, int row_end) const {
        const int N = node_columns();
        const int C = element_columns;
        const int nrows = element_rows + 1;
        const float* xr = reinterpret_cast<const float*>(x);
        float* yr = reinterpret_cast<float*>(y);
        const float* imu = reinterpret_cast<const float*>(inverse_mu);
        const float* rea = reinterpret_cast<const float*>(reaction);
        // Per element position (0: e(y-1,x-1) local row 3, 1: e(y-1,x) row 2,
        // 2: e(y,x-1) row 1, 3: e(y,x) row 0) and local column lc, the node
        // offset relative to the owned node.
        const int offsets[4][4] = {
            {-N - 1, -N, -1, 0}, {-N, -N + 1, 0, 1}, {-1, 0, N - 1, N}, {0, 1, N, N + 1}};
        const int local_rows[4] = {3, 2, 1, 0};
        const int element_offsets[4] = {-C - 1, -C, -1, 0};
        float Kr[16], Ki[16], Mr[16], Mi[16];
        for (int i = 0; i < 16; ++i) {
            Kr[i] = stiffness[i].real();
            Ki[i] = stiffness[i].imag();
            Mr[i] = mass[i].real();
            Mi[i] = mass[i].imag();
        }

        (void)nrows;
        for (int node_y = row_begin; node_y < row_end; ++node_y) {
            if (node_y == 0 || node_y == element_rows) {
                for (int node_x = 0; node_x < N; ++node_x) {
                    y[node_y * N + node_x] = gather(x, node_y, node_x);
                }
                continue;
            }
            y[node_y * N] = gather(x, node_y, 0);
            y[node_y * N + C] = gather(x, node_y, C);
            const int row0 = node_y * N;
            const int e_here = node_y * C;
#pragma omp simd
            for (int node_x = 1; node_x < C; ++node_x) {
                const int node = row0 + node_x;
                const int e = e_here + node_x;
                float acc_r = 0.0f, acc_i = 0.0f;
#pragma GCC unroll 4
                for (int p = 0; p < 4; ++p) {
                    const int el = e + element_offsets[p];
                    const float ar = imu[2 * el], ai = imu[2 * el + 1];
                    const float br = rea[2 * el], bi = rea[2 * el + 1];
                    const int lr = local_rows[p];
#pragma GCC unroll 4
                    for (int lc = 0; lc < 4; ++lc) {
                        const int li = 4 * lr + lc;
                        // coefficient = imu*K + rea*M
                        const float cr = ar * Kr[li] - ai * Ki[li] + br * Mr[li] - bi * Mi[li];
                        const float ci = ar * Ki[li] + ai * Kr[li] + br * Mi[li] + bi * Mr[li];
                        const int cn = node + offsets[p][lc];
                        const float m = free_mask[cn];
                        const float vr = xr[2 * cn] * m, vi = xr[2 * cn + 1] * m;
                        acc_r += cr * vr - ci * vi;
                        acc_i += cr * vi + ci * vr;
                    }
                }
                const float f = free_mask[node];
                yr[2 * node] = f * acc_r + (1.0f - f) * xr[2 * node];
                yr[2 * node + 1] = f * acc_i + (1.0f - f) * xr[2 * node + 1];
            }
        }
    }
};

// Flush subnormal complex64 values to zero for the duration of a native call.
// The low-precision correction is only an approximation refined by the FP64
// outer residual, and subnormal operands make FP32 SIMD arithmetic tens of
// times slower.  The previous MXCSR state is restored on exit.
class FlushSubnormals {
   public:
    FlushSubnormals() : saved_(_mm_getcsr()) { _mm_setcsr(saved_ | 0x8040u); }
    ~FlushSubnormals() { _mm_setcsr(saved_); }
    FlushSubnormals(const FlushSubnormals&) = delete;
    FlushSubnormals& operator=(const FlushSubnormals&) = delete;

   private:
    unsigned int saved_;
};

template <typename T>
const T* data_of(const py::array_t<T, py::array::c_style | py::array::forcecast>& a,
                 py::ssize_t expected, const char* name) {
    if (a.size() != expected) {
        throw std::invalid_argument(std::string(name) + " has the wrong size");
    }
    return a.data();
}

using ArrC64 = py::array_t<c64, py::array::c_style | py::array::forcecast>;
using ArrC128 = py::array_t<c128, py::array::c_style | py::array::forcecast>;
using ArrU8 = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;

using ArrF32 = py::array_t<float, py::array::c_style | py::array::forcecast>;

Q1Operator make_operator(const ArrC64& inverse_mu, const ArrC64& reaction,
                         const ArrC64& stiffness, const ArrC64& mass,
                         const ArrU8& free_nodes, const ArrF32& free_mask,
                         int element_rows, int element_columns, int threads) {
    if (element_rows < 1 || element_columns < 1) {
        throw std::invalid_argument("element shape must be positive");
    }
    const py::ssize_t elements = static_cast<py::ssize_t>(element_rows) * element_columns;
    const py::ssize_t nodes =
        static_cast<py::ssize_t>(element_rows + 1) * (element_columns + 1);
    Q1Operator op;
    op.inverse_mu = data_of(inverse_mu, elements, "inverse_mu");
    op.reaction = data_of(reaction, elements, "reaction");
    op.stiffness = data_of(stiffness, 16, "stiffness");
    op.mass = data_of(mass, 16, "mass");
    op.free_nodes = data_of(free_nodes, nodes, "free_nodes");
    op.free_mask = data_of(free_mask, nodes, "free_mask");
    op.element_rows = element_rows;
    op.element_columns = element_columns;
    op.threads = threads < 1 ? 1 : threads;
    return op;
}

double norm2(const c64* v, py::ssize_t n) {
    const float* f = reinterpret_cast<const float*>(v);
    double s = 0.0;
#pragma omp simd reduction(+ : s)
    for (py::ssize_t i = 0; i < 2 * n; ++i) {
        s += static_cast<double>(f[i]) * static_cast<double>(f[i]);
    }
    return std::sqrt(s);
}

// conj(a) . b accumulated in double, returned rounded to complex64 like the
// NumPy runtime's complex64 vdot.
c64 vdot(const c64* a, const c64* b, py::ssize_t n) {
    const float* fa = reinterpret_cast<const float*>(a);
    const float* fb = reinterpret_cast<const float*>(b);
    double re = 0.0, im = 0.0;
#pragma omp simd reduction(+ : re, im)
    for (py::ssize_t i = 0; i < n; ++i) {
        const double ar = fa[2 * i], ai = fa[2 * i + 1];
        const double br = fb[2 * i], bi = fb[2 * i + 1];
        re += ar * br + ai * bi;
        im += ar * bi - ai * br;
    }
    return c64(static_cast<float>(re), static_cast<float>(im));
}

// w -= h * v  (complex64), written on interleaved floats so it vectorises.
inline void axpy_neg(c64 h, const c64* v, c64* w, py::ssize_t n) {
    const float hr = h.real(), hi = h.imag();
    const float* fv = reinterpret_cast<const float*>(v);
    float* fw = reinterpret_cast<float*>(w);
#pragma omp simd
    for (py::ssize_t i = 0; i < n; ++i) {
        const float vr = fv[2 * i], vi = fv[2 * i + 1];
        fw[2 * i] -= hr * vr - hi * vi;
        fw[2 * i + 1] -= hr * vi + hi * vr;
    }
}

inline void axpy_add(c64 h, const c64* v, c64* w, py::ssize_t n) {
    axpy_neg(c64(-h.real(), -h.imag()), v, w, n);
}

inline void scale_into(float s, const c64* v, c64* out, py::ssize_t n) {
    const float* fv = reinterpret_cast<const float*>(v);
    float* fo = reinterpret_cast<float*>(out);
#pragma omp simd
    for (py::ssize_t i = 0; i < 2 * n; ++i) fo[i] = fv[i] * s;
}

// z = v / d  (complex64 elementwise), limited-range complex division.
inline void divide_into(const c64* v, const c64* d, c64* z, py::ssize_t n) {
    const float* fv = reinterpret_cast<const float*>(v);
    const float* fd = reinterpret_cast<const float*>(d);
    float* fz = reinterpret_cast<float*>(z);
#pragma omp simd
    for (py::ssize_t i = 0; i < n; ++i) {
        const float vr = fv[2 * i], vi = fv[2 * i + 1];
        const float dr = fd[2 * i], di = fd[2 * i + 1];
        const float inv = 1.0f / (dr * dr + di * di);
        fz[2 * i] = (vr * dr + vi * di) * inv;
        fz[2 * i + 1] = (vi * dr - vr * di) * inv;
    }
}

// Fused dot products conj(V_k) . w for k < rows on [lo, lo+len), accumulated in
// double.  w is visited one cache block at a time so it stays in L1 while the
// basis vectors stream past once each.
constexpr py::ssize_t kBlock = 1024;  // complex64 elements, 8 KiB

void block_dots(const c64* basis, py::ssize_t n, const c64* w, int rows, py::ssize_t lo,
                py::ssize_t len, std::vector<double>& re, std::vector<double>& im) {
    for (int k = 0; k < rows; ++k) {
        re[k] = 0.0;
        im[k] = 0.0;
    }
    for (py::ssize_t i0 = lo; i0 < lo + len; i0 += kBlock) {
        const py::ssize_t m = std::min(kBlock, lo + len - i0);
        const float* fb = reinterpret_cast<const float*>(w + i0);
        for (int k = 0; k < rows; ++k) {
            const float* fa = reinterpret_cast<const float*>(basis + static_cast<size_t>(k) * n + i0);
            double sr = 0.0, si = 0.0;
#pragma omp simd reduction(+ : sr, si)
            for (py::ssize_t i = 0; i < m; ++i) {
                const double ar = fa[2 * i], ai = fa[2 * i + 1];
                const double br = fb[2 * i], bi = fb[2 * i + 1];
                sr += ar * br + ai * bi;
                si += ar * bi - ai * br;
            }
            re[k] += sr;
            im[k] += si;
        }
    }
}

// w -= sum_k h_k V_k on [lo, lo+len), blocked the same way.
void block_axpy_neg(const c64* basis, py::ssize_t n, const c64* h, int rows, c64* w,
                    py::ssize_t lo, py::ssize_t len) {
    for (py::ssize_t i0 = lo; i0 < lo + len; i0 += kBlock) {
        const py::ssize_t m = std::min(kBlock, lo + len - i0);
        for (int k = 0; k < rows; ++k) {
            axpy_neg(h[k], basis + static_cast<size_t>(k) * n + i0, w + i0, m);
        }
    }
}

}  // namespace

ArrC64 apply_q1(ArrC64 vector, ArrC64 inverse_mu, ArrC64 reaction, ArrC64 stiffness,
                ArrC64 mass, ArrU8 free_nodes, ArrF32 free_mask, int element_rows,
                int element_columns, int threads) {
    const Q1Operator op = make_operator(inverse_mu, reaction, stiffness, mass, free_nodes,
                                        free_mask, element_rows, element_columns, threads);
    const py::ssize_t n = op.node_count();
    const c64* x = data_of(vector, n, "vector");
    ArrC64 out(n);
    {
        py::gil_scoped_release release;
        FlushSubnormals flush;
        op.apply(x, out.mutable_data());
    }
    return out;
}

// Restarted right-Jacobi GMRES in complex64 with a complex128 Hessenberg.
// Returns (correction complex64, total_iterations, relative_residual,
// operator_applications) with the same control flow as solver._inner_gmres.
// One cache line per thread for deterministic two-stage reductions.
struct alignas(64) Partial {
    double a, b;
    double pad[6];
};

// Restarted right-Jacobi GMRES in complex64 with a complex128 Hessenberg.
// Returns (correction complex64, total_iterations, relative_residual,
// operator_applications) with the same control flow as solver._inner_gmres.
//
// Threads run SPMD over one static node-row partition for the whole solve:
// every vector operation touches only the owned rows, the operator reads the
// neighbouring rows after a barrier, and each reduction is written per thread
// and summed by every thread in thread order, so the result does not depend on
// scheduling for a fixed thread count.  The 32x32 Hessenberg bookkeeping is
// repeated redundantly on private copies, which keeps all threads on the same
// control path without broadcasts.
py::tuple gmres_q1(ArrC128 rhs_high, ArrC64 diagonal, ArrC64 inverse_mu, ArrC64 reaction,
                   ArrC64 stiffness, ArrC64 mass, ArrU8 free_nodes, ArrF32 free_mask,
                   int element_rows, int element_columns, double inner_relative_tolerance,
                   int max_inner_iterations, int restart, int threads, bool cgs2,
                   bool float_dots) {
    const Q1Operator op = make_operator(inverse_mu, reaction, stiffness, mass, free_nodes,
                                        free_mask, element_rows, element_columns, threads);
    const py::ssize_t n = op.node_count();
    const c128* rhs_in = data_of(rhs_high, n, "rhs");
    const c64* diag = data_of(diagonal, n, "diagonal");
    if (max_inner_iterations < 1 || restart < 2) {
        throw std::invalid_argument("iteration limits are invalid");
    }

    ArrC64 correction_out(n);
    c64* correction = correction_out.mutable_data();
    int total_iterations = 0;
    int applications = 0;
    double relative_residual = 1.0;

    {
        py::gil_scoped_release release;

        const int max_cycle = restart;
        const int node_rows = element_rows + 1;
        const int ncols = op.node_columns();
        const int team = std::max(1, std::min(op.threads, node_rows));
        std::vector<c64> rhs(n), residual(n), w(n);
        std::vector<c64> basis(static_cast<size_t>(max_cycle + 1) * n);
        std::vector<c64> zbasis(static_cast<size_t>(max_cycle) * n);
        // Reduction slots.  A slot may be rewritten only after every thread has
        // passed at least one barrier since it last read the slot, so each
        // reduction inside a column has its own slot: 0 is the rhs and cycle
        // residual norm, 1+row the Gram-Schmidt coefficient of row (first
        // CGS2 pass or MGS), max_cycle+2+row the second CGS2 pass, and
        // 2*max_cycle+3 the Arnoldi vector norm.
        const int second_pass_slot = max_cycle + 2;
        const int norm_slot = 2 * max_cycle + 3;
        std::vector<Partial> partials(static_cast<size_t>(norm_slot + 1) * team);
        auto V = [&](int k) { return basis.data() + static_cast<size_t>(k) * n; };
        auto Z = [&](int k) { return zbasis.data() + static_cast<size_t>(k) * n; };

#pragma omp parallel num_threads(team) if (team > 1)
        {
            FlushSubnormals flush;  // MXCSR is per thread
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
            const int nt = omp_get_num_threads();
#else
            const int tid = 0, nt = 1;
#endif
            int row_begin = 0, row_end = node_rows;
            Q1Operator::thread_rows(node_rows, row_begin, row_end);
            const py::ssize_t lo = static_cast<py::ssize_t>(row_begin) * ncols;
            const py::ssize_t len = static_cast<py::ssize_t>(row_end - row_begin) * ncols;
            auto slot = [&](int s) -> Partial& { return partials[static_cast<size_t>(s) * nt + tid]; };
            auto reduce = [&](int s, double& a, double& b) {
                a = 0.0;
                b = 0.0;
                for (int t = 0; t < nt; ++t) {
                    a += partials[static_cast<size_t>(s) * nt + t].a;
                    b += partials[static_cast<size_t>(s) * nt + t].b;
                }
            };
            // Local reductions on the owned range (double accumulation).
            auto local_norm2 = [&](const c64* v) {
                const float* f = reinterpret_cast<const float*>(v + lo);
                double acc = 0.0;
#pragma omp simd reduction(+ : acc)
                for (py::ssize_t i = 0; i < 2 * len; ++i) {
                    acc += static_cast<double>(f[i]) * static_cast<double>(f[i]);
                }
                return acc;
            };
            auto local_vdot = [&](const c64* a, const c64* b, double& re, double& im) {
                const float* fa = reinterpret_cast<const float*>(a + lo);
                const float* fb = reinterpret_cast<const float*>(b + lo);
                if (!float_dots) {
                    double sr = 0.0, si = 0.0;
#pragma omp simd reduction(+ : sr, si)
                    for (py::ssize_t i = 0; i < len; ++i) {
                        const double ar = fa[2 * i], ai = fa[2 * i + 1];
                        const double br = fb[2 * i], bi = fb[2 * i + 1];
                        sr += ar * br + ai * bi;
                        si += ar * bi - ai * br;
                    }
                    re = sr;
                    im = si;
                    return;
                }
                // Float accumulation per cache block, blocks summed in double:
                // twice the SIMD width and no float-to-double conversion in
                // the inner loop; the block sum error is about sqrt(kBlock)
                // ulp of the partial, far below the complex64 rounding of h.
                double sr = 0.0, si = 0.0;
                for (py::ssize_t i0 = 0; i0 < len; i0 += kBlock) {
                    const py::ssize_t m = std::min(kBlock, len - i0);
                    float br_acc = 0.0f, bi_acc = 0.0f;
#pragma omp simd reduction(+ : br_acc, bi_acc)
                    for (py::ssize_t i = 0; i < m; ++i) {
                        const float ar = fa[2 * (i0 + i)], ai = fa[2 * (i0 + i) + 1];
                        const float br = fb[2 * (i0 + i)], bi = fb[2 * (i0 + i) + 1];
                        br_acc += ar * br + ai * bi;
                        bi_acc += ar * bi - ai * br;
                    }
                    sr += br_acc;
                    si += bi_acc;
                }
                re = sr;
                im = si;
            };

            // Thread-private Hessenberg bookkeeping, identical on every thread.
            std::vector<c128> hess(static_cast<size_t>(max_cycle + 1) * max_cycle);
            std::vector<c128> g(max_cycle + 1), cs(max_cycle), y(max_cycle);
            std::vector<double> sn(max_cycle);
            std::vector<double> hre(max_cycle), him(max_cycle);
            std::vector<c64> hcoef(max_cycle);
            auto H = [&](int r, int c) -> c128& { return hess[static_cast<size_t>(r) * max_cycle + c]; };
            int iterations = 0;
            int applied = 0;
            double rel = 1.0;

            for (py::ssize_t i = lo; i < lo + len; ++i) {
                rhs[i] = c64(static_cast<float>(rhs_in[i].real()),
                             static_cast<float>(rhs_in[i].imag()));
                correction[i] = c64(0.0f, 0.0f);
                residual[i] = rhs[i];
            }
            slot(0).a = local_norm2(rhs.data());
            slot(0).b = 0.0;
#pragma omp barrier
            double rhs_sq, unused;
            reduce(0, rhs_sq, unused);
            const double rhs_norm = std::sqrt(rhs_sq);
            if (rhs_norm == 0.0) {
                rel = 0.0;
            } else {
                const float eps32 = std::numeric_limits<float>::epsilon();
                while (iterations < max_inner_iterations) {
                    if (iterations) {
                        // All threads finished correction += Z y (barrier at
                        // the end of the previous cycle) before this read.
                        op.apply_rows(correction, w.data(), row_begin, row_end);
                        ++applied;
                        const float* fr = reinterpret_cast<const float*>(rhs.data() + lo);
                        const float* fw = reinterpret_cast<const float*>(w.data() + lo);
                        float* fo = reinterpret_cast<float*>(residual.data() + lo);
#pragma omp simd
                        for (py::ssize_t i = 0; i < 2 * len; ++i) fo[i] = fr[i] - fw[i];
                    }
                    slot(0).a = local_norm2(residual.data());
#pragma omp barrier
                    double beta_sq;
                    reduce(0, beta_sq, unused);
                    const double beta = std::sqrt(beta_sq);
                    rel = beta / rhs_norm;
                    if (rel <= inner_relative_tolerance) break;

                    const int cycle = std::min(restart, max_inner_iterations - iterations);
                    const float inv_beta = static_cast<float>(1.0 / beta);
                    scale_into(inv_beta, residual.data() + lo, V(0) + lo, len);
                    std::fill(hess.begin(), hess.end(), c128(0.0, 0.0));
                    std::fill(g.begin(), g.end(), c128(0.0, 0.0));
                    g[0] = c128(static_cast<float>(beta), 0.0);
                    int accepted = 0;
                    bool breakdown = false;

                    for (int col = 0; col < cycle; ++col) {
                        c64* z = Z(col);
                        const c64* v = V(col);
                        divide_into(v + lo, diag + lo, z + lo, len);
#pragma omp barrier  // z complete on every row before the stencil reads it
                        op.apply_rows(z, w.data(), row_begin, row_end);
                        ++applied;

                        if (!cgs2) {
                            // Modified Gram-Schmidt in complex64 with complex64-rounded
                            // coefficients, matching the portable runtime.
                            for (int row = 0; row <= col; ++row) {
                                local_vdot(V(row), w.data(), slot(1 + row).a, slot(1 + row).b);
#pragma omp barrier
                                double re, im;
                                reduce(1 + row, re, im);
                                const c64 h(static_cast<float>(re), static_cast<float>(im));
                                H(row, col) = c128(h.real(), h.imag());
                                axpy_neg(h, V(row) + lo, w.data() + lo, len);
                            }
                        } else {
                            // Classical Gram-Schmidt with one reorthogonalisation
                            // (CGS2).  Each pass reads w once per cache block and
                            // every basis vector once, and needs one barrier
                            // instead of col+1.  Coefficients are rounded to
                            // complex64 per pass like the MGS path; H keeps the
                            // complex128 sum of both passes.
                            for (int pass = 0; pass < 2; ++pass) {
                                const int base = pass == 0 ? 1 : second_pass_slot;
                                block_dots(basis.data(), n, w.data(), col + 1, lo, len, hre, him);
                                for (int row = 0; row <= col; ++row) {
                                    slot(base + row).a = hre[row];
                                    slot(base + row).b = him[row];
                                }
#pragma omp barrier
                                for (int row = 0; row <= col; ++row) {
                                    double re, im;
                                    reduce(base + row, re, im);
                                    hcoef[row] = c64(static_cast<float>(re), static_cast<float>(im));
                                    H(row, col) += c128(hcoef[row].real(), hcoef[row].imag());
                                }
                                block_axpy_neg(basis.data(), n, hcoef.data(), col + 1, w.data(), lo, len);
                            }
                        }
                        slot(norm_slot).a = local_norm2(w.data());
#pragma omp barrier
                        double next_sq;
                        reduce(norm_slot, next_sq, unused);
                        const double next_norm = std::sqrt(next_sq);
                        const float next_norm32 = static_cast<float>(next_norm);
                        H(col + 1, col) = c128(next_norm32, 0.0);
                        breakdown = !(next_norm32 > eps32 * static_cast<float>(beta));
                        if (!breakdown) {
                            scale_into(1.0f / next_norm32, w.data() + lo, V(col + 1) + lo, len);
                        }

                        // Givens rotations: apply the earlier ones, then zero the
                        // subdiagonal of this column.  Equivalent to the NumPy
                        // least-squares solve on the same Hessenberg.
                        for (int row = 0; row < col; ++row) {
                            const c128 a = H(row, col), b = H(row + 1, col);
                            H(row, col) = std::conj(cs[row]) * a + sn[row] * b;
                            H(row + 1, col) = -sn[row] * a + cs[row] * b;
                        }
                        {
                            const c128 a = H(col, col), b = H(col + 1, col);
                            const double na = std::abs(a), nb = std::abs(b);
                            const double r = std::hypot(na, nb);
                            if (r == 0.0) {
                                cs[col] = c128(1.0, 0.0);
                                sn[col] = 0.0;
                            } else if (na == 0.0) {
                                cs[col] = c128(0.0, 0.0);
                                sn[col] = 1.0;
                            } else {
                                cs[col] = (a / na) * (na / r);
                                sn[col] = nb / r;
                            }
                            H(col, col) = std::conj(cs[col]) * a + sn[col] * b;
                            H(col + 1, col) = c128(0.0, 0.0);
                            const c128 g0 = g[col];
                            g[col] = std::conj(cs[col]) * g0;
                            g[col + 1] = -sn[col] * g0;
                        }
                        accepted = col + 1;
                        rel = std::abs(g[accepted]) / rhs_norm;
                        ++iterations;
                        if (rel <= inner_relative_tolerance || breakdown) break;
                    }

                    // Back substitution R y = g, then correction += Z y.
                    for (int row = accepted - 1; row >= 0; --row) {
                        c128 acc = g[row];
                        for (int c = row + 1; c < accepted; ++c) acc -= H(row, c) * y[c];
                        y[row] = acc / H(row, row);
                    }
                    for (int k = 0; k < accepted; ++k) {
                        const c64 yk(static_cast<float>(y[k].real()), static_cast<float>(y[k].imag()));
                        axpy_add(yk, Z(k) + lo, correction + lo, len);
                    }
#pragma omp barrier  // correction complete before the next residual stencil
                    if (rel <= inner_relative_tolerance) break;
                }
            }
            if (tid == 0) {
                total_iterations = iterations;
                applications = applied;
                relative_residual = rel;
            }
        }
    }
    return py::make_tuple(correction_out, total_iterations, relative_residual, applications);
}


int default_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

PYBIND11_MODULE(_scalar_maxwell_native, m) {
    m.doc() = "Fused C++ Q1 scalar Maxwell operator and complex64 inner GMRES";
    m.def("apply_q1", &apply_q1, py::arg("vector"), py::arg("inverse_mu"), py::arg("reaction"),
          py::arg("stiffness"), py::arg("mass"), py::arg("free_nodes"), py::arg("free_mask"),
          py::arg("element_rows"), py::arg("element_columns"), py::arg("threads") = 1);
    m.def("gmres_q1", &gmres_q1, py::arg("rhs_high"), py::arg("diagonal"), py::arg("inverse_mu"),
          py::arg("reaction"), py::arg("stiffness"), py::arg("mass"), py::arg("free_nodes"),
          py::arg("free_mask"), py::arg("element_rows"), py::arg("element_columns"),
          py::arg("inner_relative_tolerance"), py::arg("max_inner_iterations"),
          py::arg("restart"), py::arg("threads") = 1, py::arg("cgs2") = false,
          py::arg("float_dots") = false);
    m.def("default_threads", &default_threads);
    m.attr("openmp") =
#ifdef _OPENMP
        true;
#else
        false;
#endif
}
