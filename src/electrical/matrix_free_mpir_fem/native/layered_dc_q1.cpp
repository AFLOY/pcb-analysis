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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
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

class FlushSubnormals {
   public:
    FlushSubnormals() : saved_(_mm_getcsr()) { _mm_setcsr(saved_ | 0x8040u); }
    ~FlushSubnormals() { _mm_setcsr(saved_); }
    FlushSubnormals(const FlushSubnormals&) = delete;
    FlushSubnormals& operator=(const FlushSubnormals&) = delete;

   private:
    unsigned int saved_;
};

template <typename T, int F>
const T* data_of(const py::array_t<T, F>& a, py::ssize_t expected, const char* name) {
    if (a.size() != expected) {
        throw std::invalid_argument(std::string(name) + " has the wrong size");
    }
    return a.data();
}

template <typename T>
struct LayeredOperatorT {
    const T* coef;             // (2, layers, rows, cols): c_x, c_y per element
    const T* unit;             // (2, 4, 4): U_x, U_y; local node index 2 dy + dx
    const std::uint8_t* free_nodes;
    const T* free_mask;        // 1 free, 0 fixed
    const std::int64_t* via_ptr;   // (nodes + 1,)
    const std::int64_t* via_nbr;   // (links,)
    const T* via_g;                // (links,)
    int layers, rows, cols;    // element grid per layer
    int threads;

    int node_rows() const { return rows + 1; }
    int node_cols() const { return cols + 1; }
    int lines() const { return layers * node_rows(); }
    py::ssize_t element_count() const { return static_cast<py::ssize_t>(layers) * rows * cols; }
    py::ssize_t node_count() const { return static_cast<py::ssize_t>(lines()) * node_cols(); }

    // Resistive links of one node: sum g (x_n - x_nbr) over the vias that
    // end there.  Fixed neighbours contribute zero, like the masked gather.
    T via_terms(const T* x, py::ssize_t node) const {
        T acc = T(0);
        const T xn = x[node] * free_mask[node];
        for (std::int64_t k = via_ptr[node]; k < via_ptr[node + 1]; ++k) {
            const std::int64_t nbr = via_nbr[k];
            acc += via_g[k] * (xn - x[nbr] * free_mask[nbr]);
        }
        return acc;
    }

    // Generic gather for one node.
    T gather(const T* x, int l, int y, int xi) const {
        const int nr = node_rows(), nc = node_cols();
        const py::ssize_t node = (static_cast<py::ssize_t>(l) * nr + y) * nc + xi;
        if (!free_nodes[node]) return x[node];
        const int y0 = y > 0 ? y - 1 : 0, y1 = y < rows ? y : rows - 1;
        const int x0 = xi > 0 ? xi - 1 : 0, x1 = xi < cols ? xi : cols - 1;
        const py::ssize_t ne = element_count();
        const T* ux = unit;
        const T* uy = unit + 16;
        T acc = T(0);
        for (int ey = y0; ey <= y1; ++ey) {
            for (int ex = x0; ex <= x1; ++ex) {
                const int lr = 2 * (y - ey) + (xi - ex);
                const py::ssize_t e = (static_cast<py::ssize_t>(l) * rows + ey) * cols + ex;
                const T a = coef[e], b = coef[ne + e];
                const py::ssize_t corner = (static_cast<py::ssize_t>(l) * nr + ey) * nc + ex;
                for (int c = 0; c < 4; ++c) {
                    const py::ssize_t cn = corner + (c >> 1) * nc + (c & 1);
                    acc += (a * ux[4 * lr + c] + b * uy[4 * lr + c]) * (x[cn] * free_mask[cn]);
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
    void apply_lines(const T* x, T* out, int line_begin, int line_end) const {
        const int nr = node_rows(), nc = node_cols();
        for (int line = line_begin; line < line_end; ++line) {
            const int l = line / nr, y = line - l * nr;
            const py::ssize_t base = static_cast<py::ssize_t>(line) * nc;
            out[base] = gather(x, l, y, 0);
            out[base + cols] = gather(x, l, y, cols);
            if (cols < 2) continue;
            const int y0 = y > 0 ? y - 1 : 0, y1 = y < rows ? y : rows - 1;
            T* o = out + base;
            for (int xi = 1; xi < cols; ++xi) o[xi] = T(0);
            const py::ssize_t ne = element_count();
            const T* ux = unit;
            const T* uy = unit + 16;
            for (int ey = y0; ey <= y1; ++ey) {
                const int lr_base = 2 * (y - ey);
                const py::ssize_t e_row = (static_cast<py::ssize_t>(l) * rows + ey) * cols;  // element ex = 0
                const T* ax = coef + e_row;
                const T* ay = coef + ne + e_row;
                const py::ssize_t corner_row = (static_cast<py::ssize_t>(l) * nr + ey) * nc;  // corner ex = 0
#pragma omp simd
                for (int xi = 1; xi < cols; ++xi) {
                    T acc = T(0);
#pragma GCC unroll 2
                    for (int dx = 0; dx < 2; ++dx) {
                        const int ex = xi - 1 + dx;        // element left (dx=0) or right (dx=1)
                        const int lr = lr_base + (1 - dx);  // local x index of the node in it
                        const T a = ax[ex], b = ay[ex];
                        const py::ssize_t corner = corner_row + ex;
#pragma GCC unroll 4
                        for (int c = 0; c < 4; ++c) {
                            const py::ssize_t cn = corner + (c >> 1) * nc + (c & 1);
                            const T w = a * ux[4 * lr + c] + b * uy[4 * lr + c];
                            acc += w * (x[cn] * free_mask[cn]);
                        }
                    }
                    o[xi] += acc;
                }
            }
            // Via links end at few nodes; a sparse pass over the line.
            for (int xi = 1; xi < cols; ++xi) {
                const py::ssize_t node = base + xi;
                if (via_ptr[node] != via_ptr[node + 1]) o[xi] += via_terms(x, node);
            }
#pragma omp simd
            for (int xi = 1; xi < cols; ++xi) {
                const py::ssize_t node = base + xi;
                const T f = free_mask[node];
                o[xi] = f * o[xi] + (T(1) - f) * x[node];
            }
        }
    }

    static void thread_lines(int lines, int& begin, int& end) {
#ifdef _OPENMP
        const int nt = omp_get_num_threads(), t = omp_get_thread_num();
        begin = static_cast<int>(static_cast<long long>(lines) * t / nt);
        end = static_cast<int>(static_cast<long long>(lines) * (t + 1) / nt);
#else
        begin = 0;
        end = lines;
#endif
    }
};

template <typename T>
LayeredOperatorT<T> make_operator(const Arr<T>& coef, const Arr<T>& unit, const ArrU8& free_nodes,
                                  const Arr<T>& free_mask, const ArrI64& via_ptr, const ArrI64& via_nbr,
                                  const Arr<T>& via_g, int layers, int rows, int cols, int threads) {
    if (layers < 1 || rows < 1 || cols < 1) throw std::invalid_argument("element grid must be positive");
    LayeredOperatorT<T> op;
    op.layers = layers;
    op.rows = rows;
    op.cols = cols;
    op.threads = threads < 1 ? 1 : threads;
    const py::ssize_t elements = op.element_count();
    const py::ssize_t nodes = op.node_count();
    op.coef = data_of(coef, 2 * elements, "coefficients");
    op.unit = data_of(unit, static_cast<py::ssize_t>(32), "unit");
    op.free_nodes = data_of(free_nodes, nodes, "free_nodes");
    op.free_mask = data_of(free_mask, nodes, "free_mask");
    op.via_ptr = data_of(via_ptr, nodes + 1, "via_ptr");
    const py::ssize_t links = static_cast<py::ssize_t>(op.via_ptr[nodes]);
    if (op.via_ptr[0] != 0 || links < 0) throw std::invalid_argument("via_ptr must start at zero");
    op.via_nbr = data_of(via_nbr, links, "via_nbr");
    op.via_g = data_of(via_g, links, "via_g");
    for (py::ssize_t k = 0; k < links; ++k) {
        if (op.via_nbr[k] < 0 || op.via_nbr[k] >= nodes) throw std::invalid_argument("via_nbr names a node outside the mesh");
    }
    return op;
}

struct alignas(64) Partial {
    double a;
    double pad[7];
};

template <typename T>
Arr<T> apply_impl(const LayeredOperatorT<T>& op, const Arr<T>& vector, bool flush_subnormals) {
    const py::ssize_t n = op.node_count();
    const T* x = data_of(vector, n, "vector");
    Arr<T> out(n);
    T* y = out.mutable_data();
    {
        py::gil_scoped_release release;
        const int lines = op.lines();
        const int team = std::max(1, std::min(op.threads, lines));
#pragma omp parallel num_threads(team) if (team > 1)
        {
            if (flush_subnormals) {
                FlushSubnormals flush;
                int b = 0, e = lines;
                LayeredOperatorT<T>::thread_lines(lines, b, e);
                op.apply_lines(x, y, b, e);
            } else {
                int b = 0, e = lines;
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
    const double* rhs_in = data_of(rhs_high, n, "rhs");
    const float* diag = data_of(diagonal, n, "diagonal");
    if (max_inner_iterations < 1) throw std::invalid_argument("max_inner_iterations must be positive");
    if (block < 1) throw std::invalid_argument("block must be positive");
    const int nr = op.node_rows(), nc = op.node_cols();
    const int coarse_rows = (nr + block - 1) / block, coarse_cols = (nc + block - 1) / block;
    const py::ssize_t ncoarse = static_cast<py::ssize_t>(layers) * coarse_rows * coarse_cols;
    const bool two_level = coarse_inverse.size() > 0;
    const float* cinv = nullptr;
    if (two_level) cinv = data_of(coarse_inverse, ncoarse * ncoarse, "coarse_inverse");

    ArrF32 correction_out(n);
    float* xsol = correction_out.mutable_data();
    int total_iterations = 0, applications = 0;
    double relative_residual = 1.0;
    bool not_spd = false;

    {
        py::gil_scoped_release release;
        const int lines = op.lines();
        const int team = std::max(1, std::min(op.threads, lines));
        std::vector<float> rhs(n), r(n), z(n), p(n), q(n);
        std::vector<double> coarse_part(static_cast<size_t>(team) * (two_level ? ncoarse : 0));
        std::vector<float> coarse_r(two_level ? ncoarse : 0), coarse_z(two_level ? ncoarse : 0);
        // Slots: 0 norms, 1 curvature, 2 rz.
        std::vector<Partial> partials(3 * static_cast<size_t>(team));

#pragma omp parallel num_threads(team) if (team > 1)
        {
            FlushSubnormals flush;
#ifdef _OPENMP
            const int tid = omp_get_thread_num(), nt = omp_get_num_threads();
#else
            const int tid = 0, nt = 1;
#endif
            int lb = 0, le = lines;
            LayeredOperatorT<float>::thread_lines(lines, lb, le);
            const py::ssize_t lo = static_cast<py::ssize_t>(lb) * nc;
            const py::ssize_t len = static_cast<py::ssize_t>(le - lb) * nc;
            auto slot = [&](int s) -> double& { return partials[static_cast<size_t>(s) * nt + tid].a; };
            auto reduce = [&](int s) {
                double acc = 0.0;
                for (int t = 0; t < nt; ++t) acc += partials[static_cast<size_t>(s) * nt + t].a;
                return acc;
            };
            auto local_dot = [&](const float* a, const float* b) {
                double acc = 0.0;
#pragma omp simd reduction(+ : acc)
                for (py::ssize_t i = lo; i < lo + len; ++i) acc += static_cast<double>(a[i]) * b[i];
                return acc;
            };
            // z = M^-1 r on the owned lines.  Three barriers when two-level.
            auto precondition = [&]() {
#pragma omp simd
                for (py::ssize_t i = lo; i < lo + len; ++i) z[i] = r[i] / diag[i];
                if (!two_level) return;
                double* part = coarse_part.data() + static_cast<size_t>(tid) * ncoarse;
                std::fill(part, part + ncoarse, 0.0);
                for (int line = lb; line < le; ++line) {
                    const int l = line / nr, yy = line - l * nr;
                    const py::ssize_t cbase = (static_cast<py::ssize_t>(l) * coarse_rows + yy / block) * coarse_cols;
                    const float* rl = r.data() + static_cast<py::ssize_t>(line) * nc;
                    const float* ml = op.free_mask + static_cast<py::ssize_t>(line) * nc;
                    for (int xi = 0; xi < nc; ++xi) part[cbase + xi / block] += static_cast<double>(rl[xi] * ml[xi]);
                }
#pragma omp barrier
                const py::ssize_t cb = ncoarse * tid / nt, ce = ncoarse * (tid + 1) / nt;
                for (py::ssize_t i = cb; i < ce; ++i) {
                    double acc = 0.0;
                    for (int t = 0; t < nt; ++t) acc += coarse_part[static_cast<size_t>(t) * ncoarse + i];
                    coarse_r[i] = static_cast<float>(acc);
                }
#pragma omp barrier
                for (py::ssize_t i = cb; i < ce; ++i) {
                    const float* row = cinv + i * ncoarse;
                    double acc = 0.0;
#pragma omp simd reduction(+ : acc)
                    for (py::ssize_t j = 0; j < ncoarse; ++j) acc += static_cast<double>(row[j]) * coarse_r[j];
                    coarse_z[i] = static_cast<float>(acc);
                }
#pragma omp barrier
                for (int line = lb; line < le; ++line) {
                    const int l = line / nr, yy = line - l * nr;
                    const py::ssize_t cbase = (static_cast<py::ssize_t>(l) * coarse_rows + yy / block) * coarse_cols;
                    float* zl = z.data() + static_cast<py::ssize_t>(line) * nc;
                    const float* ml = op.free_mask + static_cast<py::ssize_t>(line) * nc;
                    for (int xi = 0; xi < nc; ++xi) zl[xi] += ml[xi] * coarse_z[cbase + xi / block];
                }
            };

            int iterations = 0, applied = 0;
            double rel = 1.0;
            bool bad = false;
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
                for (py::ssize_t i = lo; i < lo + len; ++i) p[i] = z[i];
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
#pragma omp simd
                    for (py::ssize_t i = lo; i < lo + len; ++i) {
                        xsol[i] += alpha * p[i];
                        r[i] -= alpha * q[i];
                    }
                    slot(0) = local_dot(r.data(), r.data());
#pragma omp barrier
                    rel = std::sqrt(reduce(0)) / rhs_norm;
                    ++iterations;
                    if (rel <= inner_relative_tolerance) break;
                    precondition();
                    slot(2) = local_dot(r.data(), z.data());
#pragma omp barrier
                    const double next_rz = reduce(2);
                    if (!std::isfinite(next_rz) || rz == 0.0) break;
                    const float beta = static_cast<float>(next_rz / rz);
#pragma omp simd
                    for (py::ssize_t i = lo; i < lo + len; ++i) p[i] = z[i] + beta * p[i];
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
