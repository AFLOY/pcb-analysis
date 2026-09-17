// Fused CPU kernels for the thermal matrix-free MPIR low path.
//
// The operator is the node-owned gather of the trilinear (hexahedral Q1)
// conduction action used by the CUDA kernel in ``cuda.py``: one output node
// visits its at most eight adjacent element slabs, applies the two per-slab
// 8x8 unit tensors weighted by the element conductivities, adds the lumped
// Robin conductance, and is written once.  The inner PCG with the two-level
// preconditioner (Jacobi plus patch-constant coarse correction through a
// dense float32 inverse) runs as one SPMD OpenMP region per outer MPIR step.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
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

struct HexOperator {
    const float* in_plane;   // (slabs, rows, cols)
    const float* through;    // (slabs, rows, cols)
    const float* local_in;   // (slabs, 8, 8)
    const float* local_thr;  // (slabs, 8, 8)
    const float* robin;      // (nodes,)
    const std::uint8_t* free_nodes;
    const float* free_mask;  // 1.0f free, 0.0f fixed
    int slabs, rows, cols;   // element grid
    int threads;

    int node_layers() const { return slabs + 1; }
    int node_rows() const { return rows + 1; }
    int node_cols() const { return cols + 1; }
    int lines() const { return node_layers() * node_rows(); }
    py::ssize_t node_count() const { return static_cast<py::ssize_t>(lines()) * node_cols(); }

    // Generic gather for one node, same ordering as the CUDA kernel.
    float gather(const float* x, int z, int y, int xi) const {
        const int nr = node_rows(), nc = node_cols(), plane = nr * nc;
        const int node = (z * nr + y) * nc + xi;
        if (!free_nodes[node]) return x[node];
        const int z0 = z > 0 ? z - 1 : 0, z1 = z < slabs ? z : slabs - 1;
        const int y0 = y > 0 ? y - 1 : 0, y1 = y < rows ? y : rows - 1;
        const int x0 = xi > 0 ? xi - 1 : 0, x1 = xi < cols ? xi : cols - 1;
        float acc = 0.0f;
        for (int ez = z0; ez <= z1; ++ez) {
            const float* kin = local_in + ez * 64;
            const float* kz = local_thr + ez * 64;
            for (int ey = y0; ey <= y1; ++ey) {
                for (int ex = x0; ex <= x1; ++ex) {
                    const int lr = 4 * (z - ez) + 2 * (y - ey) + (xi - ex);
                    const int e = (ez * rows + ey) * cols + ex;
                    const float a = in_plane[e], b = through[e];
                    const int corner = (ez * nr + ey) * nc + ex;
                    for (int c = 0; c < 8; ++c) {
                        const int cn = corner + (c >> 2) * plane + ((c >> 1) & 1) * nc + (c & 1);
                        if (!free_nodes[cn]) continue;
                        acc += (a * kin[8 * lr + c] + b * kz[8 * lr + c]) * x[cn];
                    }
                }
            }
        }
        return acc + robin[node] * x[node];
    }

    // y = A x on node lines [line_begin, line_end).  A line is one (z, y) row
    // of node_cols nodes.  Interior x use a branch-free form over the adjacent
    // element slabs and rows, unrolled over the two x-neighbour elements and
    // the eight local columns, so the x-loop vectorises; the ends of the line
    // use the generic gather.
    void apply_lines(const float* x, float* out, int line_begin, int line_end) const {
        const int nr = node_rows(), nc = node_cols(), plane = nr * nc;
        for (int line = line_begin; line < line_end; ++line) {
            const int z = line / nr, y = line - z * nr;
            const int base = line * nc;
            out[base] = gather(x, z, y, 0);
            out[base + cols] = gather(x, z, y, cols);
            if (cols < 2) continue;
            const int z0 = z > 0 ? z - 1 : 0, z1 = z < slabs ? z : slabs - 1;
            const int y0 = y > 0 ? y - 1 : 0, y1 = y < rows ? y : rows - 1;
            // acc over interior x; robin and the identity blend at the end.
            float* o = out + base;
            for (int xi = 1; xi < cols; ++xi) o[xi] = 0.0f;
            for (int ez = z0; ez <= z1; ++ez) {
                const float* kin = local_in + ez * 64;
                const float* kz = local_thr + ez * 64;
                for (int ey = y0; ey <= y1; ++ey) {
                    const int lr_base = 4 * (z - ez) + 2 * (y - ey);
                    const int e_row = (ez * rows + ey) * cols;      // element index of ex = 0
                    const int corner_row = (ez * nr + ey) * nc;    // node index of corner ex = 0
#pragma omp simd
                    for (int xi = 1; xi < cols; ++xi) {
                        float acc = 0.0f;
#pragma GCC unroll 2
                        for (int dx = 0; dx < 2; ++dx) {
                            const int ex = xi - 1 + dx;           // element left (dx=0) or right (dx=1)
                            const int lr = lr_base + (1 - dx);     // local x index of the node in it
                            const float a = in_plane[e_row + ex], b = through[e_row + ex];
                            const int corner = corner_row + ex;
#pragma GCC unroll 8
                            for (int c = 0; c < 8; ++c) {
                                const int cn = corner + (c >> 2) * plane + ((c >> 1) & 1) * nc + (c & 1);
                                const float w = a * kin[8 * lr + c] + b * kz[8 * lr + c];
                                acc += w * (x[cn] * free_mask[cn]);
                            }
                        }
                        o[xi] += acc;
                    }
                }
            }
#pragma omp simd
            for (int xi = 1; xi < cols; ++xi) {
                const int node = base + xi;
                const float f = free_mask[node];
                o[xi] = f * (o[xi] + robin[node] * x[node]) + (1.0f - f) * x[node];
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

HexOperator make_operator(const ArrF32& in_plane, const ArrF32& through, const ArrF32& local_in,
                          const ArrF32& local_thr, const ArrF32& robin, const ArrU8& free_nodes,
                          const ArrF32& free_mask, int slabs, int rows, int cols, int threads) {
    if (slabs < 1 || rows < 1 || cols < 1) throw std::invalid_argument("element grid must be positive");
    HexOperator op;
    op.slabs = slabs;
    op.rows = rows;
    op.cols = cols;
    op.threads = threads < 1 ? 1 : threads;
    const py::ssize_t elements = static_cast<py::ssize_t>(slabs) * rows * cols;
    const py::ssize_t nodes = op.node_count();
    op.in_plane = data_of(in_plane, elements, "in_plane");
    op.through = data_of(through, elements, "through");
    op.local_in = data_of(local_in, static_cast<py::ssize_t>(slabs) * 64, "local_in_plane");
    op.local_thr = data_of(local_thr, static_cast<py::ssize_t>(slabs) * 64, "local_through");
    op.robin = data_of(robin, nodes, "robin");
    op.free_nodes = data_of(free_nodes, nodes, "free_nodes");
    op.free_mask = data_of(free_mask, nodes, "free_mask");
    return op;
}

struct alignas(64) Partial {
    double a;
    double pad[7];
};

}  // namespace

ArrF32 apply_hex_q1(ArrF32 vector, ArrF32 in_plane, ArrF32 through, ArrF32 local_in,
                    ArrF32 local_thr, ArrF32 robin, ArrU8 free_nodes, ArrF32 free_mask, int slabs,
                    int rows, int cols, int threads) {
    const HexOperator op = make_operator(in_plane, through, local_in, local_thr, robin, free_nodes,
                                         free_mask, slabs, rows, cols, threads);
    const py::ssize_t n = op.node_count();
    const float* x = data_of(vector, n, "vector");
    ArrF32 out(n);
    float* y = out.mutable_data();
    {
        py::gil_scoped_release release;
        const int lines = op.lines();
        const int team = std::max(1, std::min(op.threads, lines));
#pragma omp parallel num_threads(team) if (team > 1)
        {
            FlushSubnormals flush;
            int b = 0, e = lines;
            HexOperator::thread_lines(lines, b, e);
            op.apply_lines(x, y, b, e);
        }
    }
    return out;
}

// Inner PCG in float32 with the two-level preconditioner
//   M^-1 r = D^-1 r + Z (Z^T A Z)^-1 Z^T r
// (or Jacobi only when ``coarse_inverse`` is empty).  ``block`` is the
// in-plane patch width; the coarse index of node (z, y, x) is
// (z * coarse_rows + y / block) * coarse_cols + x / block.  Returns
// (correction float32, iterations, relative_residual, applications) with the
// control flow of solver._inner_pcg.  Threads run SPMD over a static partition
// of node lines; reductions are per-thread partials summed in thread order.
py::tuple pcg_hex_q1(ArrF64 rhs_high, ArrF32 diagonal, ArrF32 in_plane, ArrF32 through,
                     ArrF32 local_in, ArrF32 local_thr, ArrF32 robin, ArrU8 free_nodes,
                     ArrF32 free_mask, int slabs, int rows, int cols, int block,
                     ArrF32 coarse_inverse, double inner_relative_tolerance,
                     int max_inner_iterations, int threads) {
    const HexOperator op = make_operator(in_plane, through, local_in, local_thr, robin, free_nodes,
                                         free_mask, slabs, rows, cols, threads);
    const py::ssize_t n = op.node_count();
    const double* rhs_in = data_of(rhs_high, n, "rhs");
    const float* diag = data_of(diagonal, n, "diagonal");
    if (max_inner_iterations < 1) throw std::invalid_argument("max_inner_iterations must be positive");
    if (block < 1) throw std::invalid_argument("block must be positive");
    const int nr = op.node_rows(), nc = op.node_cols(), nl = op.node_layers();
    const int coarse_rows = (nr + block - 1) / block, coarse_cols = (nc + block - 1) / block;
    const py::ssize_t ncoarse = static_cast<py::ssize_t>(nl) * coarse_rows * coarse_cols;
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
            HexOperator::thread_lines(lines, lb, le);
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
            // z = M^-1 r on the owned lines.  Two barriers when two-level.
            auto precondition = [&]() {
#pragma omp simd
                for (py::ssize_t i = lo; i < lo + len; ++i) z[i] = r[i] / diag[i];
                if (!two_level) return;
                double* part = coarse_part.data() + static_cast<size_t>(tid) * ncoarse;
                std::fill(part, part + ncoarse, 0.0);
                for (int line = lb; line < le; ++line) {
                    const int zz = line / nr, yy = line - zz * nr;
                    const py::ssize_t cbase = (static_cast<py::ssize_t>(zz) * coarse_rows + yy / block) * coarse_cols;
                    const float* rl = r.data() + static_cast<py::ssize_t>(line) * nc;
                    const float* ml = op.free_mask + static_cast<py::ssize_t>(line) * nc;
                    for (int xi = 0; xi < nc; ++xi) part[cbase + xi / block] += static_cast<double>(rl[xi] * ml[xi]);
                }
#pragma omp barrier
                // Sum the partial restrictions in thread order for the coarse
                // rows this thread owns, then apply the dense inverse rows.
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
                    const int zz = line / nr, yy = line - zz * nr;
                    const py::ssize_t cbase = (static_cast<py::ssize_t>(zz) * coarse_rows + yy / block) * coarse_cols;
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
                    // p complete on every line (barrier above or at loop end).
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

PYBIND11_MODULE(_thermal_native, m) {
    m.doc() = "Fused C++ hexahedral Q1 conduction operator and two-level inner PCG";
    m.def("apply_hex_q1", &apply_hex_q1, py::arg("vector"), py::arg("in_plane"), py::arg("through"),
          py::arg("local_in_plane"), py::arg("local_through"), py::arg("robin"), py::arg("free_nodes"),
          py::arg("free_mask"), py::arg("slabs"), py::arg("rows"), py::arg("cols"),
          py::arg("threads") = 1);
    m.def("pcg_hex_q1", &pcg_hex_q1, py::arg("rhs_high"), py::arg("diagonal"), py::arg("in_plane"),
          py::arg("through"), py::arg("local_in_plane"), py::arg("local_through"), py::arg("robin"),
          py::arg("free_nodes"), py::arg("free_mask"), py::arg("slabs"), py::arg("rows"),
          py::arg("cols"), py::arg("block"), py::arg("coarse_inverse"),
          py::arg("inner_relative_tolerance"), py::arg("max_inner_iterations"),
          py::arg("threads") = 1);
    m.attr("openmp") =
#ifdef _OPENMP
        true;
#else
        false;
#endif
}
