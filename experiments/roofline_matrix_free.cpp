// Roofline probes for the matrix-free Q1 kernels on the host CPU.
//
// The production operators are compiled in from their sources (not copied),
// each inside its own namespace, so the timed matrix-free code is exactly the
// shipped kernel.  Next to them are the machine probes (STREAM-like bandwidth
// per working-set size, AVX-512 FMA peak) and the assembled alternatives the
// matrix-free form is compared with: a structured stencil with one coefficient
// array per neighbour offset ("DIA", no indices) and a CSR SpMV.  Every kernel
// uses the same static node-row partition and a preallocated output, and each
// call is timed inside C++.

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <immintrin.h>
#include <omp.h>
#include <xmmintrin.h>

#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable) [[maybe_unused]] static void name##_unused(pybind11::module_& variable)
namespace mx {
#include "../src/electrical/matrix_free_mpir_fem/native/scalar_maxwell_q1.cpp"
}
namespace th {
#include "../src/thermal/matrix_free_mpir_fem/native/hex_q1.cpp"
}
#undef PYBIND11_MODULE
#pragma pop_macro("PYBIND11_MODULE")

namespace py = pybind11;
using c64 = std::complex<float>;
using ArrF32 = py::array_t<float, py::array::c_style | py::array::forcecast>;
using ArrC64 = py::array_t<c64, py::array::c_style | py::array::forcecast>;
using ArrI32 = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;
using ArrU8 = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;

namespace {

using Clock = std::chrono::steady_clock;

// Per-call seconds of ``fn`` after ``warmups`` untimed calls.
template <typename F>
std::vector<double> timed(F&& fn, int repeats, int warmups) {
    for (int i = 0; i < warmups; ++i) fn();
    std::vector<double> samples;
    samples.reserve(repeats);
    for (int i = 0; i < repeats; ++i) {
        const auto t0 = Clock::now();
        fn();
        samples.push_back(std::chrono::duration<double>(Clock::now() - t0).count());
    }
    return samples;
}

void partition(py::ssize_t n, py::ssize_t& b, py::ssize_t& e) {
    const int nt = omp_get_num_threads(), t = omp_get_thread_num();
    b = n * t / nt;
    e = n * (t + 1) / nt;
}

void partition_rows(int rows, int& b, int& e) {
    const int nt = omp_get_num_threads(), t = omp_get_thread_num();
    b = static_cast<int>(static_cast<long long>(rows) * t / nt);
    e = static_cast<int>(static_cast<long long>(rows) * (t + 1) / nt);
}

}  // namespace

// ---------------------------------------------------------------- machine ---

// STREAM-like kernels on float32 arrays of ``n`` elements each, first touched
// by the owning thread.  kind: 0 read (sum), 1 copy, 2 triad a = b + s c.
// One call runs ``inner`` sweeps of each thread's own range inside one
// parallel region, so small working sets are not dominated by the fork; the read
// sum keeps eight 512-bit accumulators so it is not add-latency bound.
// Returns per-call seconds; the caller divides by ``inner`` and counts bytes.
std::vector<double> stream(py::ssize_t n, int kind, int threads, int repeats, int inner) {
    float* pa = static_cast<float*>(std::aligned_alloc(64, ((n * 4 + 63) / 64) * 64));
    float* pb = static_cast<float*>(std::aligned_alloc(64, ((n * 4 + 63) / 64) * 64));
    float* pc = static_cast<float*>(std::aligned_alloc(64, ((n * 4 + 63) / 64) * 64));
    std::vector<double> sink(threads * 8, 0.0);
#pragma omp parallel num_threads(threads)
    {
        py::ssize_t lo, hi;
        partition(n, lo, hi);
        for (py::ssize_t i = lo; i < hi; ++i) {
            pa[i] = 1.0f;
            pb[i] = 2.0f;
            pc[i] = 0.5f;
        }
    }
    const float s = 1.0001f;
    auto run = [&]() {
#pragma omp parallel num_threads(threads)
        {
            py::ssize_t lo, hi;
            partition(n, lo, hi);
            for (int sweep = 0; sweep < inner; ++sweep) {
                if (kind == 0) {
                    __m512 acc[8];
                    for (int k = 0; k < 8; ++k) acc[k] = _mm512_setzero_ps();
                    py::ssize_t i = lo;
                    for (; i + 128 <= hi; i += 128) {
#pragma GCC unroll 8
                        for (int k = 0; k < 8; ++k) acc[k] = _mm512_add_ps(acc[k], _mm512_loadu_ps(pb + i + 16 * k));
                    }
                    float tail = 0.0f;
                    for (; i < hi; ++i) tail += pb[i];
                    for (int k = 1; k < 8; ++k) acc[0] = _mm512_add_ps(acc[0], acc[k]);
                    sink[omp_get_thread_num() * 8] += _mm512_reduce_add_ps(acc[0]) + tail;
                } else if (kind == 1) {
#pragma omp simd
                    for (py::ssize_t i = lo; i < hi; ++i) pa[i] = pb[i];
                } else {
#pragma omp simd
                    for (py::ssize_t i = lo; i < hi; ++i) pa[i] = pb[i] + s * pc[i];
                }
            }
        }
    };
    auto samples = timed(run, repeats, 3);
    std::free(pa);
    std::free(pb);
    std::free(pc);
    return samples;
}

// Peak FP32 (or FP64) FMA throughput: 16 independent 512-bit accumulator
// chains per thread, all in registers.  Returns seconds for ``iterations``
// rounds; flops = threads * iterations * 16 chains * lanes * 2.
double fma_peak(long iterations, int threads, bool fp64) {
    std::vector<double> sink(threads * 8, 0.0);
    const auto t0 = Clock::now();
#pragma omp parallel num_threads(threads)
    {
        if (!fp64) {
            __m512 acc[16];
            for (int k = 0; k < 16; ++k) acc[k] = _mm512_set1_ps(1.0f + 1e-3f * k);
            const __m512 m = _mm512_set1_ps(0.999999f), a = _mm512_set1_ps(1e-7f);
            for (long it = 0; it < iterations; ++it) {
#pragma GCC unroll 16
                for (int k = 0; k < 16; ++k) acc[k] = _mm512_fmadd_ps(acc[k], m, a);
            }
            __m512 s = acc[0];
            for (int k = 1; k < 16; ++k) s = _mm512_add_ps(s, acc[k]);
            sink[omp_get_thread_num() * 8] = _mm512_reduce_add_ps(s);
        } else {
            __m512d acc[16];
            for (int k = 0; k < 16; ++k) acc[k] = _mm512_set1_pd(1.0 + 1e-3 * k);
            const __m512d m = _mm512_set1_pd(0.999999), a = _mm512_set1_pd(1e-7);
            for (long it = 0; it < iterations; ++it) {
#pragma GCC unroll 16
                for (int k = 0; k < 16; ++k) acc[k] = _mm512_fmadd_pd(acc[k], m, a);
            }
            __m512d s = acc[0];
            for (int k = 1; k < 16; ++k) s = _mm512_add_pd(s, acc[k]);
            sink[omp_get_thread_num() * 8] = _mm512_reduce_add_pd(s);
        }
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    volatile double keep = sink[0];
    (void)keep;
    return seconds;
}

// ---------------------------------------------------- 2D Q1 scalar Maxwell ---

// Production matrix-free action (mx::Q1Operator::apply), preallocated output.
py::tuple maxwell_matrix_free(ArrC64 x, ArrC64 inverse_mu, ArrC64 reaction, ArrC64 stiffness,
                              ArrC64 mass, ArrU8 free_nodes, int element_rows,
                              int element_columns, int threads, int repeats) {
    const ArrF32 mask = ArrF32(py::array(free_nodes).attr("astype")("float32"));
    const mx::Q1Operator op = mx::make_operator(inverse_mu, reaction, stiffness, mass, free_nodes,
                                                mask, element_rows, element_columns, threads);
    ArrC64 y(op.node_count());
    std::vector<double> samples;
    {
        py::gil_scoped_release release;
        mx::FlushSubnormals flush;
        samples = timed([&]() { op.apply(x.data(), y.mutable_data()); }, repeats, 3);
    }
    return py::make_tuple(y, samples);
}

// Assembled 9-point stencil in complex64: coef is (9, nodes), offset k =
// 3*(dy+1)+(dx+1); entries whose neighbour lies outside the grid are zero.
py::tuple maxwell_stencil(ArrC64 x, ArrC64 coef, int node_rows, int node_columns, int threads,
                          int repeats) {
    const py::ssize_t n = static_cast<py::ssize_t>(node_rows) * node_columns;
    if (x.size() != n || coef.size() != 9 * n) throw std::invalid_argument("stencil shape");
    ArrC64 y(n);
    const float* xr = reinterpret_cast<const float*>(x.data());
    const float* cr = reinterpret_cast<const float*>(coef.data());
    float* yr = reinterpret_cast<float*>(y.mutable_data());
    const int N = node_columns;
    auto edge = [&](int ny, int nx) {
        const py::ssize_t node = static_cast<py::ssize_t>(ny) * N + nx;
        float ar = 0.0f, ai = 0.0f;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int my = ny + dy, mxx = nx + dx;
                if (my < 0 || my >= node_rows || mxx < 0 || mxx >= N) continue;
                const py::ssize_t m = static_cast<py::ssize_t>(my) * N + mxx;
                const float* c = cr + 2 * ((3 * (dy + 1) + dx + 1) * n + node);
                ar += c[0] * xr[2 * m] - c[1] * xr[2 * m + 1];
                ai += c[0] * xr[2 * m + 1] + c[1] * xr[2 * m];
            }
        }
        yr[2 * node] = ar;
        yr[2 * node + 1] = ai;
    };
    auto run = [&]() {
#pragma omp parallel num_threads(threads)
        {
            int rb, re;
            partition_rows(node_rows, rb, re);
            for (int ny = rb; ny < re; ++ny) {
                if (ny == 0 || ny == node_rows - 1) {
                    for (int nx = 0; nx < N; ++nx) edge(ny, nx);
                    continue;
                }
                edge(ny, 0);
                edge(ny, N - 1);
                const py::ssize_t row0 = static_cast<py::ssize_t>(ny) * N;
#pragma omp simd
                for (int nx = 1; nx < N - 1; ++nx) {
                    const py::ssize_t node = row0 + nx;
                    float ar = 0.0f, ai = 0.0f;
#pragma GCC unroll 9
                    for (int k = 0; k < 9; ++k) {
                        const py::ssize_t m = node + (k / 3 - 1) * N + (k % 3 - 1);
                        const float c0 = cr[2 * (k * n + node)], c1 = cr[2 * (k * n + node) + 1];
                        const float v0 = xr[2 * m], v1 = xr[2 * m + 1];
                        ar += c0 * v0 - c1 * v1;
                        ai += c0 * v1 + c1 * v0;
                    }
                    yr[2 * node] = ar;
                    yr[2 * node + 1] = ai;
                }
            }
        }
    };
    std::vector<double> samples;
    {
        py::gil_scoped_release release;
        mx::FlushSubnormals flush;
        samples = timed(run, repeats, 3);
    }
    return py::make_tuple(y, samples);
}

// CSR SpMV, int32 indices, rows split by the same static node-row partition.
template <typename T>
py::tuple csr(py::array_t<T, py::array::c_style | py::array::forcecast> x, ArrI32 indptr,
              ArrI32 indices, py::array_t<T, py::array::c_style | py::array::forcecast> data,
              int threads, int repeats) {
    const py::ssize_t n = indptr.size() - 1;
    py::array_t<T> y(n);
    const T* px = x.data();
    const std::int32_t* pp = indptr.data();
    const std::int32_t* pi = indices.data();
    const T* pd = data.data();
    T* py_ = y.mutable_data();
    auto run = [&]() {
#pragma omp parallel num_threads(threads)
        {
            py::ssize_t lo, hi;
            partition(n, lo, hi);
            for (py::ssize_t r = lo; r < hi; ++r) {
                T acc{};
                for (std::int32_t k = pp[r]; k < pp[r + 1]; ++k) acc += pd[k] * px[pi[k]];
                py_[r] = acc;
            }
        }
    };
    std::vector<double> samples;
    {
        py::gil_scoped_release release;
        mx::FlushSubnormals flush;
        samples = timed(run, repeats, 3);
    }
    return py::make_tuple(y, samples);
}

// ------------------------------------------------------ 3D hex Q1 thermal ---

py::tuple thermal_matrix_free(ArrF32 x, ArrF32 coef, ArrF32 unit, ArrF32 robin, ArrU8 free_nodes,
                              int slabs, int rows, int cols, int threads, int repeats) {
    const ArrF32 mask = ArrF32(py::array(free_nodes).attr("astype")("float32"));
    const th::HexOperator op =
        th::make_operator(coef, unit, robin, free_nodes, mask, slabs, rows, cols, threads);
    ArrF32 y(op.node_count());
    const int lines = op.lines();
    const int team = std::max(1, std::min(op.threads, lines));
    std::vector<double> samples;
    {
        py::gil_scoped_release release;
        auto run = [&]() {
#pragma omp parallel num_threads(team)
            {
                th::FlushSubnormals flush;
                int b = 0, e = lines;
                th::HexOperator::thread_lines(lines, b, e);
                op.apply_lines(x.data(), y.mutable_data(), b, e);
            }
        };
        samples = timed(run, repeats, 3);
    }
    return py::make_tuple(y, samples);
}

// Assembled 27-point stencil in float32: coef is (27, nodes), offset k =
// 9*(dz+1)+3*(dy+1)+(dx+1).  Node lines (z, y) are split like the production
// kernel; interior x vectorises, the two line ends and grid faces bound-check.
py::tuple thermal_stencil(ArrF32 x, ArrF32 coef, int layers, int node_rows, int node_cols,
                          int threads, int repeats) {
    const py::ssize_t n = static_cast<py::ssize_t>(layers) * node_rows * node_cols;
    if (x.size() != n || coef.size() != 27 * n) throw std::invalid_argument("stencil shape");
    ArrF32 y(n);
    const float* px = x.data();
    const float* pc = coef.data();
    float* py_ = y.mutable_data();
    const int nc = node_cols;
    const py::ssize_t plane = static_cast<py::ssize_t>(node_rows) * nc;
    auto edge = [&](int z, int yy, int xi) {
        const py::ssize_t node = (static_cast<py::ssize_t>(z) * node_rows + yy) * nc + xi;
        float acc = 0.0f;
        for (int k = 0; k < 27; ++k) {
            const int dz = k / 9 - 1, dy = (k / 3) % 3 - 1, dx = k % 3 - 1;
            const int mz = z + dz, my = yy + dy, mxx = xi + dx;
            if (mz < 0 || mz >= layers || my < 0 || my >= node_rows || mxx < 0 || mxx >= nc) continue;
            acc += pc[k * n + node] * px[node + dz * plane + dy * nc + dx];
        }
        py_[node] = acc;
    };
    const int lines = layers * node_rows;
    auto run = [&]() {
#pragma omp parallel num_threads(threads)
        {
            int lb, le;
            partition_rows(lines, lb, le);
            for (int line = lb; line < le; ++line) {
                const int z = line / node_rows, yy = line - z * node_rows;
                edge(z, yy, 0);
                edge(z, yy, nc - 1);
                const py::ssize_t base = static_cast<py::ssize_t>(line) * nc;
                if (z == 0 || z == layers - 1 || yy == 0 || yy == node_rows - 1) {
                    // Face line: accumulate over the neighbour lines that
                    // exist, three x offsets each, still vectorised in x.
                    float* o = py_ + base;
                    for (int xi = 1; xi < nc - 1; ++xi) o[xi] = 0.0f;
                    for (int dz = -1; dz <= 1; ++dz) {
                        if (z + dz < 0 || z + dz >= layers) continue;
                        for (int dy = -1; dy <= 1; ++dy) {
                            if (yy + dy < 0 || yy + dy >= node_rows) continue;
                            const int k0 = 9 * (dz + 1) + 3 * (dy + 1);
                            const float* c0 = pc + k0 * n + base;
                            const float* c1 = c0 + n;
                            const float* c2 = c1 + n;
                            const float* xl = px + base + dz * plane + dy * nc;
#pragma omp simd
                            for (int xi = 1; xi < nc - 1; ++xi) {
                                o[xi] += c0[xi] * xl[xi - 1] + c1[xi] * xl[xi] + c2[xi] * xl[xi + 1];
                            }
                        }
                    }
                    continue;
                }
#pragma omp simd
                for (int xi = 1; xi < nc - 1; ++xi) {
                    const py::ssize_t node = base + xi;
                    float acc = 0.0f;
#pragma GCC unroll 27
                    for (int k = 0; k < 27; ++k) {
                        const py::ssize_t m = node + (k / 9 - 1) * plane + ((k / 3) % 3 - 1) * nc + (k % 3 - 1);
                        acc += pc[k * n + node] * px[m];
                    }
                    py_[node] = acc;
                }
            }
        }
    };
    std::vector<double> samples;
    {
        py::gil_scoped_release release;
        mx::FlushSubnormals flush;
        samples = timed(run, repeats, 3);
    }
    return py::make_tuple(y, samples);
}

PYBIND11_MODULE(_roofline_matrix_free, m) {
    m.doc() = "Roofline probes: bandwidth, FMA peak, matrix-free vs assembled Q1 actions";
    m.def("stream", &stream, py::arg("n"), py::arg("kind"), py::arg("threads"), py::arg("repeats"), py::arg("inner"));
    m.def("fma_peak", &fma_peak, py::arg("iterations"), py::arg("threads"), py::arg("fp64") = false);
    m.def("maxwell_matrix_free", &maxwell_matrix_free);
    m.def("maxwell_stencil", &maxwell_stencil);
    m.def("csr_c64", &csr<c64>);
    m.def("csr_f32", &csr<float>);
    m.def("thermal_matrix_free", &thermal_matrix_free);
    m.def("thermal_stencil", &thermal_stencil);
}
