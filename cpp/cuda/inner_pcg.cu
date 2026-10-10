// CUDA inner PCG of the layered DC and thermal hex MPIR solves; see
// pcbcore/cuda/inner_pcg.hpp.  Built with --fmad=false: no product is fused
// into a sum, as in the CPU gathers.
#include "pcbcore/cuda/inner_pcg.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "pcbcore/lane_sum.hpp"

namespace pcbcore::cuda {

namespace {

using Index = long long;

void check(const cudaError_t status, const char* const what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(status));
    }
}

// A device array that frees itself.
template <typename T>
class DeviceArray {
public:
    DeviceArray() = default;
    explicit DeviceArray(const Index count) { allocate(count); }
    DeviceArray(const T* const host, const Index count) {
        allocate(count);
        upload(host, count);
    }
    ~DeviceArray() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    void allocate(const Index count) {
        count_ = count;
        if (count > 0) {
            check(cudaMalloc(reinterpret_cast<void**>(&data_), static_cast<std::size_t>(count) * sizeof(T)),
                  "allocation");
        }
    }
    void upload(const T* const host, const Index count) {
        if (count > 0) {
            check(cudaMemcpy(data_, host, static_cast<std::size_t>(count) * sizeof(T), cudaMemcpyHostToDevice),
                  "upload");
        }
    }
    void download(T* const host, const Index count) const {
        if (count > 0) {
            check(cudaMemcpy(host, data_, static_cast<std::size_t>(count) * sizeof(T), cudaMemcpyDeviceToHost),
                  "download");
        }
    }
    [[nodiscard]] T* get() const noexcept { return data_; }

private:
    T* data_{nullptr};
    Index count_{0};
};

constexpr int kThreads = 256;

[[nodiscard]] int blocks_for(const Index count) {
    return static_cast<int>((count + kThreads - 1) / kThreads);
}

// ----------------------------------------------------------------- operators

// The layered DC sheet operator: nodes (layers, rows + 1, cols + 1), 4-node
// elements, vias as a node-owned CSR (layered_dc_q1.cpp).
struct DCOperator {
    const float* coef;
    const float* unit;
    const std::uint8_t* free_nodes;
    const float* free_mask;
    const long long* via_ptr;
    const long long* via_nbr;
    const float* via_g;
    int layers;
    int rows;
    int cols;

    __host__ __device__ int node_rows() const { return rows + 1; }
    __host__ __device__ int node_cols() const { return cols + 1; }

    __device__ float via_terms(const float* x, const Index node) const {
        float acc = 0.0f;
        const float xn = x[node] * free_mask[node];
        for (long long k = via_ptr[node]; k < via_ptr[node + 1]; ++k) {
            const long long nbr = via_nbr[k];
            acc += via_g[k] * (xn - (x[nbr] * free_mask[nbr]));
        }
        return acc;
    }

    // LayeredOperatorT::gather (the end columns of every line).
    __device__ float gather(const float* x, const int l, const int y, const int xi) const {
        const int nr = node_rows();
        const int nc = node_cols();
        const Index node = (static_cast<Index>(l) * nr + y) * nc + xi;
        if (free_nodes[node] == 0U) {
            return x[node];
        }
        const int y0 = y > 0 ? y - 1 : 0;
        const int y1 = y < rows ? y : rows - 1;
        const int x0 = xi > 0 ? xi - 1 : 0;
        const int x1 = xi < cols ? xi : cols - 1;
        const Index ne = static_cast<Index>(layers) * rows * cols;
        float acc = 0.0f;
        for (int ey = y0; ey <= y1; ++ey) {
            for (int ex = x0; ex <= x1; ++ex) {
                const int lr = 2 * (y - ey) + (xi - ex);
                const Index e = (static_cast<Index>(l) * rows + ey) * cols + ex;
                const float a = coef[e];
                const float b = coef[ne + e];
                const Index corner = (static_cast<Index>(l) * nr + ey) * nc + ex;
                for (int c = 0; c < 4; ++c) {
                    const Index cn = corner + static_cast<Index>(c >> 1) * nc + (c & 1);
                    acc += ((a * unit[4 * lr + c]) + (b * unit[16 + 4 * lr + c])) * (x[cn] * free_mask[cn]);
                }
            }
        }
        return acc + via_terms(x, node);
    }

    // LayeredOperatorT::apply_lines for one node.
    __device__ float apply(const float* x, const Index node) const {
        const int nr = node_rows();
        const int nc = node_cols();
        const int line = static_cast<int>(node / nc);
        const int xi = static_cast<int>(node - static_cast<Index>(line) * nc);
        const int l = line / nr;
        const int y = line - l * nr;
        if (xi == 0 || xi == cols || cols < 2) {
            return gather(x, l, y, xi);
        }
        const int y0 = y > 0 ? y - 1 : 0;
        const int y1 = y < rows ? y : rows - 1;
        const Index ne = static_cast<Index>(layers) * rows * cols;
        float out = 0.0f;
        for (int ey = y0; ey <= y1; ++ey) {
            const int lr_base = 2 * (y - ey);
            const Index e_row = (static_cast<Index>(l) * rows + ey) * cols;
            const Index corner_row = (static_cast<Index>(l) * nr + ey) * nc;
            float acc = 0.0f;
            for (int dx = 0; dx < 2; ++dx) {
                const int ex = xi - 1 + dx;
                const int lr = lr_base + (1 - dx);
                const float a = coef[e_row + ex];
                const float b = coef[ne + e_row + ex];
                const Index corner = corner_row + ex;
                for (int c = 0; c < 4; ++c) {
                    const Index cn = corner + static_cast<Index>(c >> 1) * nc + (c & 1);
                    const float w = (a * unit[4 * lr + c]) + (b * unit[16 + 4 * lr + c]);
                    acc += w * (x[cn] * free_mask[cn]);
                }
            }
            out += acc;
        }
        if (via_ptr[node] != via_ptr[node + 1]) {
            out += via_terms(x, node);
        }
        const float f = free_mask[node];
        return (f * out) + ((1.0f - f) * x[node]);
    }
};

// The thermal hex operator: nodes (slabs + 1, rows + 1, cols + 1), 8-node
// elements, a lumped Robin conductance per node (hex_q1.cpp).
struct HexOperator {
    const float* coef;
    const float* unit;
    const float* robin;
    const std::uint8_t* free_nodes;
    const float* free_mask;
    int slabs;
    int rows;
    int cols;

    __host__ __device__ int node_rows() const { return rows + 1; }
    __host__ __device__ int node_cols() const { return cols + 1; }

    // HexOperatorT::gather.
    __device__ float gather(const float* x, const int z, const int y, const int xi) const {
        const int nr = node_rows();
        const int nc = node_cols();
        const int plane = nr * nc;
        const Index node = (static_cast<Index>(z) * nr + y) * nc + xi;
        if (free_nodes[node] == 0U) {
            return x[node];
        }
        const int z0 = z > 0 ? z - 1 : 0;
        const int z1 = z < slabs ? z : slabs - 1;
        const int y0 = y > 0 ? y - 1 : 0;
        const int y1 = y < rows ? y : rows - 1;
        const int x0 = xi > 0 ? xi - 1 : 0;
        const int x1 = xi < cols ? xi : cols - 1;
        const Index ne = static_cast<Index>(slabs) * rows * cols;
        float acc = 0.0f;
        for (int ez = z0; ez <= z1; ++ez) {
            for (int ey = y0; ey <= y1; ++ey) {
                for (int ex = x0; ex <= x1; ++ex) {
                    const int lr = 4 * (z - ez) + 2 * (y - ey) + (xi - ex);
                    const Index e = (static_cast<Index>(ez) * rows + ey) * cols + ex;
                    const float a = coef[e];
                    const float b = coef[ne + e];
                    const float d = coef[2 * ne + e];
                    const Index corner = (static_cast<Index>(ez) * nr + ey) * nc + ex;
                    for (int c = 0; c < 8; ++c) {
                        const Index cn =
                            corner + static_cast<Index>(c >> 2) * plane + static_cast<Index>((c >> 1) & 1) * nc + (c & 1);
                        if (free_nodes[cn] == 0U) {
                            continue;
                        }
                        acc += ((a * unit[8 * lr + c]) + (b * unit[64 + 8 * lr + c]) + (d * unit[128 + 8 * lr + c])) *
                               x[cn];
                    }
                }
            }
        }
        return acc + (robin[node] * x[node]);
    }

    // HexOperatorT::apply_lines for one node.
    __device__ float apply(const float* x, const Index node) const {
        const int nr = node_rows();
        const int nc = node_cols();
        const int plane = nr * nc;
        const int line = static_cast<int>(node / nc);
        const int xi = static_cast<int>(node - static_cast<Index>(line) * nc);
        const int z = line / nr;
        const int y = line - z * nr;
        if (xi == 0 || xi == cols || cols < 2) {
            return gather(x, z, y, xi);
        }
        const int z0 = z > 0 ? z - 1 : 0;
        const int z1 = z < slabs ? z : slabs - 1;
        const int y0 = y > 0 ? y - 1 : 0;
        const int y1 = y < rows ? y : rows - 1;
        const Index ne = static_cast<Index>(slabs) * rows * cols;
        float out = 0.0f;
        for (int ez = z0; ez <= z1; ++ez) {
            for (int ey = y0; ey <= y1; ++ey) {
                const int lr_base = 4 * (z - ez) + 2 * (y - ey);
                const Index e_row = (static_cast<Index>(ez) * rows + ey) * cols;
                const Index corner_row = (static_cast<Index>(ez) * nr + ey) * nc;
                float acc = 0.0f;
                for (int dx = 0; dx < 2; ++dx) {
                    const int ex = xi - 1 + dx;
                    const int lr = lr_base + (1 - dx);
                    const float a = coef[e_row + ex];
                    const float b = coef[ne + e_row + ex];
                    const float d = coef[2 * ne + e_row + ex];
                    const Index corner = corner_row + ex;
                    for (int c = 0; c < 8; ++c) {
                        const Index cn = corner + static_cast<Index>(c >> 2) * plane +
                                         static_cast<Index>((c >> 1) & 1) * nc + (c & 1);
                        const float w = (a * unit[8 * lr + c]) + (b * unit[64 + 8 * lr + c]) + (d * unit[128 + 8 * lr + c]);
                        acc += w * (x[cn] * free_mask[cn]);
                    }
                }
                out += acc;
            }
        }
        const float f = free_mask[node];
        return (f * (out + (robin[node] * x[node]))) + ((1.0f - f) * x[node]);
    }
};

// ------------------------------------------------------------------ kernels

template <typename Op>
__global__ void apply_kernel(const Op op, const float* x, float* y, const Index n) {
    const Index node = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node < n) {
        y[node] = op.apply(x, node);
    }
}

// One lane sum per node line: lanes by index from the line start, combined in
// the fixed tree of pcbcore::lane_sum.
__global__ void line_dot_kernel(const float* a, const float* b, double* partial, const int lines, const int nc) {
    const int line = blockIdx.x * blockDim.x + threadIdx.x;
    if (line >= lines) {
        return;
    }
    const float* al = a + static_cast<Index>(line) * nc;
    const float* bl = b + static_cast<Index>(line) * nc;
    double acc[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int i = 0;
    for (; i + 8 <= nc; i += 8) {
        for (int k = 0; k < 8; ++k) {
            acc[k] += static_cast<double>(al[i + k]) * static_cast<double>(bl[i + k]);
        }
    }
    for (int k = 0; i < nc; ++i, ++k) {
        acc[k] += static_cast<double>(al[i]) * static_cast<double>(bl[i]);
    }
    partial[line] = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
}

// The partials in a fixed tree: thread t sums lines t, t + 256, ... in order,
// then the 256 sums meet pairwise.  Independent of anything but the count.
__global__ void reduce_kernel(const double* partial, const int lines, double* result) {
    __shared__ double sums[kThreads];
    double acc = 0.0;
    for (int line = threadIdx.x; line < lines; line += kThreads) {
        acc += partial[line];
    }
    sums[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (static_cast<int>(threadIdx.x) < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *result = sums[0];
    }
}

__global__ void start_kernel(const double* rhs, float* rhs_low, float* x, float* r, const Index n) {
    const Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        rhs_low[i] = static_cast<float>(rhs[i]);
        x[i] = 0.0f;
        r[i] = rhs_low[i];
    }
}

__global__ void jacobi_kernel(const float* r, const float* diag, float* z, const Index n) {
    const Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = r[i] / diag[i];
    }
}

// Z^T r over one patch per thread, rows then columns, as the CPU does.
__global__ void restrict_kernel(const float* r, const float* mask, float* coarse_r, const int layers, const int nr,
                                const int nc, const int block, const int coarse_rows, const int coarse_cols) {
    const Index per_layer = static_cast<Index>(coarse_rows) * coarse_cols;
    const Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= per_layer * layers) {
        return;
    }
    const int l = static_cast<int>(i / per_layer);
    const int yc = static_cast<int>((i - l * per_layer) / coarse_cols);
    const int xc = static_cast<int>(i - l * per_layer - static_cast<Index>(yc) * coarse_cols);
    double acc = 0.0;
    const int y_end = min(nr, (yc + 1) * block);
    const int x_end = min(nc, (xc + 1) * block);
    for (int yy = yc * block; yy < y_end; ++yy) {
        const Index row = (static_cast<Index>(l) * nr + yy) * nc;
        for (int xi = xc * block; xi < x_end; ++xi) {
            acc += static_cast<double>(r[row + xi] * mask[row + xi]);
        }
    }
    coarse_r[i] = static_cast<float>(acc);
}

// One coarse row per thread, a lane sum over the row as the CPU does.
__global__ void coarse_kernel(const float* cinv, const float* coarse_r, float* coarse_z, const Index ncoarse) {
    const Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= ncoarse) {
        return;
    }
    const float* ci = cinv + i * ncoarse;
    double acc[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    Index j = 0;
    for (; j + 8 <= ncoarse; j += 8) {
        for (int k = 0; k < 8; ++k) {
            acc[k] += static_cast<double>(ci[j + k]) * static_cast<double>(coarse_r[j + k]);
        }
    }
    for (int k = 0; j < ncoarse; ++j, ++k) {
        acc[k] += static_cast<double>(ci[j]) * static_cast<double>(coarse_r[j]);
    }
    coarse_z[i] = static_cast<float>(((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])));
}

__global__ void prolong_kernel(const float* coarse_z, const float* mask, float* z, const int layers, const int nr,
                               const int nc, const int block, const int coarse_rows, const int coarse_cols) {
    const Index n = static_cast<Index>(layers) * nr * nc;
    const Index node = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= n) {
        return;
    }
    const int line = static_cast<int>(node / nc);
    const int xi = static_cast<int>(node - static_cast<Index>(line) * nc);
    const int l = line / nr;
    const int yy = line - l * nr;
    const Index cbase = (static_cast<Index>(l) * coarse_rows + yy / block) * coarse_cols;
    z[node] += mask[node] * coarse_z[cbase + xi / block];
}

__global__ void update_kernel(const float alpha, const float* p, const float* q, float* x, float* r, const Index n) {
    const Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += alpha * p[i];
        r[i] -= alpha * q[i];
    }
}

__global__ void direction_kernel(const float beta, const float* z, float* p, const Index n) {
    const Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i] = z[i] + beta * p[i];
    }
}

__global__ void copy_kernel(const float* source, float* target, const Index n) {
    const Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        target[i] = source[i];
    }
}

// --------------------------------------------------------------- the solver

// The two-level PCG of pcg_*_core on one device; Op is DCOperator or
// HexOperator over device arrays owned by the derived class.
template <typename Op>
class DevicePCG : public InnerSolver {
public:
    DevicePCG(const int device, const Index nodes, const int lines, const int nc, const float* diagonal,
              const int block, const float* coarse_inverse, const int layers, const int nr)
        : device_(device),
          n_(nodes),
          lines_(lines),
          nc_(nc),
          nr_(nr),
          layers_(layers),
          block_(block),
          two_level_(coarse_inverse != nullptr),
          coarse_rows_((nr + block - 1) / block),
          coarse_cols_((nc + block - 1) / block),
          ncoarse_(two_level_ ? static_cast<Index>(layers) * coarse_rows_ * coarse_cols_ : 0) {
        check(cudaSetDevice(device_), "device selection");
        diag_.allocate(n_);
        diag_.upload(diagonal, n_);
        if (two_level_) {
            cinv_.allocate(ncoarse_ * ncoarse_);
            cinv_.upload(coarse_inverse, ncoarse_ * ncoarse_);
            coarse_r_.allocate(ncoarse_);
            coarse_z_.allocate(ncoarse_);
        }
        rhs_.allocate(n_);
        rhs_low_.allocate(n_);
        x_.allocate(n_);
        r_.allocate(n_);
        z_.allocate(n_);
        p_.allocate(n_);
        q_.allocate(n_);
        partial_.allocate(lines_);
        scalar_.allocate(1);
    }

    [[nodiscard]] long long size() const noexcept override { return n_; }

    InnerResult solve(const double* const rhs, float* const correction, const double tolerance,
                      const int max_iterations) override {
        check(cudaSetDevice(device_), "device selection");
        rhs_.upload(rhs, n_);
        const int nb = blocks_for(n_);
        start_kernel<<<nb, kThreads>>>(rhs_.get(), rhs_low_.get(), x_.get(), r_.get(), n_);
        InnerResult out;
        const double rhs_norm = std::sqrt(dot(rhs_low_.get(), rhs_low_.get()));
        if (rhs_norm == 0.0) {
            out.relative_residual = 0.0;
        } else {
            precondition();
            copy_kernel<<<nb, kThreads>>>(z_.get(), p_.get(), n_);
            double rz = dot(r_.get(), z_.get());
            while (out.iterations < max_iterations) {
                apply_kernel<<<nb, kThreads>>>(op(), p_.get(), q_.get(), n_);
                ++out.applications;
                const double curvature = dot(p_.get(), q_.get());
                if (!std::isfinite(curvature) || curvature <= 0.0) {
                    out.not_spd = true;
                    break;
                }
                const float alpha = static_cast<float>(rz / curvature);
                update_kernel<<<nb, kThreads>>>(alpha, p_.get(), q_.get(), x_.get(), r_.get(), n_);
                out.relative_residual = std::sqrt(dot(r_.get(), r_.get())) / rhs_norm;
                ++out.iterations;
                if (out.relative_residual <= tolerance) {
                    break;
                }
                precondition();
                const double next_rz = dot(r_.get(), z_.get());
                if (!std::isfinite(next_rz) || rz == 0.0) {
                    break;
                }
                const float beta = static_cast<float>(next_rz / rz);
                direction_kernel<<<nb, kThreads>>>(beta, z_.get(), p_.get(), n_);
                rz = next_rz;
            }
        }
        check(cudaGetLastError(), "kernel launch");
        x_.download(correction, n_);
        return out;
    }

protected:
    [[nodiscard]] virtual Op op() const noexcept = 0;
    [[nodiscard]] virtual const float* mask() const noexcept = 0;

private:
    double dot(const float* const a, const float* const b) {
        line_dot_kernel<<<blocks_for(lines_), kThreads>>>(a, b, partial_.get(), lines_, nc_);
        reduce_kernel<<<1, kThreads>>>(partial_.get(), lines_, scalar_.get());
        double value = 0.0;
        scalar_.download(&value, 1);
        return value;
    }

    void precondition() {
        const int nb = blocks_for(n_);
        jacobi_kernel<<<nb, kThreads>>>(r_.get(), diag_.get(), z_.get(), n_);
        if (!two_level_) {
            return;
        }
        restrict_kernel<<<blocks_for(ncoarse_), kThreads>>>(r_.get(), mask(), coarse_r_.get(), layers_, nr_, nc_,
                                                            block_, coarse_rows_, coarse_cols_);
        coarse_kernel<<<blocks_for(ncoarse_), kThreads>>>(cinv_.get(), coarse_r_.get(), coarse_z_.get(), ncoarse_);
        prolong_kernel<<<nb, kThreads>>>(coarse_z_.get(), mask(), z_.get(), layers_, nr_, nc_, block_, coarse_rows_,
                                         coarse_cols_);
    }

    int device_;
    Index n_;
    int lines_;
    int nc_;
    int nr_;
    int layers_;
    int block_;
    bool two_level_;
    int coarse_rows_;
    int coarse_cols_;
    Index ncoarse_;
    DeviceArray<float> diag_;
    DeviceArray<float> cinv_;
    DeviceArray<float> coarse_r_;
    DeviceArray<float> coarse_z_;
    DeviceArray<double> rhs_;
    DeviceArray<float> rhs_low_;
    DeviceArray<float> x_;
    DeviceArray<float> r_;
    DeviceArray<float> z_;
    DeviceArray<float> p_;
    DeviceArray<float> q_;
    DeviceArray<double> partial_;
    DeviceArray<double> scalar_;
};

class LayeredDCDevice final : public DevicePCG<DCOperator> {
public:
    LayeredDCDevice(const fem::layered_dc::OperatorView<float>& v, const float* diagonal, const int block,
                    const float* cinv, const int device)
        : DevicePCG<DCOperator>(device, v.node_count(), v.layers * (v.rows + 1), v.cols + 1, diagonal, block, cinv,
                                v.layers, v.rows + 1),
          layers_(v.layers),
          rows_(v.rows),
          cols_(v.cols) {
        const Index n = v.node_count();
        const Index elements = static_cast<Index>(v.layers) * v.rows * v.cols;
        const Index links = v.via_ptr[n];
        coef_.allocate(2 * elements);
        coef_.upload(v.coef, 2 * elements);
        unit_.allocate(32);
        unit_.upload(v.unit, 32);
        free_.allocate(n);
        free_.upload(v.free_nodes, n);
        mask_.allocate(n);
        mask_.upload(v.free_mask, n);
        std::vector<long long> ptr(v.via_ptr, v.via_ptr + n + 1);
        std::vector<long long> nbr(v.via_nbr, v.via_nbr + links);
        via_ptr_.allocate(n + 1);
        via_ptr_.upload(ptr.data(), n + 1);
        via_nbr_.allocate(std::max<Index>(1, links));
        via_nbr_.upload(nbr.data(), links);
        via_g_.allocate(std::max<Index>(1, links));
        via_g_.upload(v.via_g, links);
    }

protected:
    [[nodiscard]] DCOperator op() const noexcept override {
        return {coef_.get(), unit_.get(), free_.get(), mask_.get(), via_ptr_.get(), via_nbr_.get(), via_g_.get(),
                layers_, rows_, cols_};
    }
    [[nodiscard]] const float* mask() const noexcept override { return mask_.get(); }

private:
    int layers_;
    int rows_;
    int cols_;
    DeviceArray<float> coef_;
    DeviceArray<float> unit_;
    DeviceArray<std::uint8_t> free_;
    DeviceArray<float> mask_;
    DeviceArray<long long> via_ptr_;
    DeviceArray<long long> via_nbr_;
    DeviceArray<float> via_g_;
};

class ThermalHexDevice final : public DevicePCG<HexOperator> {
public:
    ThermalHexDevice(const fem::thermal_hex::OperatorView<float>& v, const float* diagonal, const int block,
                     const float* cinv, const int device)
        : DevicePCG<HexOperator>(device, v.node_count(), (v.slabs + 1) * (v.rows + 1), v.cols + 1, diagonal, block,
                                 cinv, v.slabs + 1, v.rows + 1),
          slabs_(v.slabs),
          rows_(v.rows),
          cols_(v.cols) {
        const Index n = v.node_count();
        const Index elements = static_cast<Index>(v.slabs) * v.rows * v.cols;
        coef_.allocate(3 * elements);
        coef_.upload(v.coef, 3 * elements);
        unit_.allocate(192);
        unit_.upload(v.unit, 192);
        robin_.allocate(n);
        robin_.upload(v.robin, n);
        free_.allocate(n);
        free_.upload(v.free_nodes, n);
        mask_.allocate(n);
        mask_.upload(v.free_mask, n);
    }

protected:
    [[nodiscard]] HexOperator op() const noexcept override {
        return {coef_.get(), unit_.get(), robin_.get(), free_.get(), mask_.get(), slabs_, rows_, cols_};
    }
    [[nodiscard]] const float* mask() const noexcept override { return mask_.get(); }

private:
    int slabs_;
    int rows_;
    int cols_;
    DeviceArray<float> coef_;
    DeviceArray<float> unit_;
    DeviceArray<float> robin_;
    DeviceArray<std::uint8_t> free_;
    DeviceArray<float> mask_;
};

}  // namespace

int device_count() noexcept {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        cudaGetLastError();  // clear the sticky "no device" error
        return 0;
    }
    return count;
}

std::unique_ptr<InnerSolver> make_layered_dc(const fem::layered_dc::OperatorView<float>& low,
                                             const float* const diagonal, const int block,
                                             const float* const coarse_inverse, const int device) {
    if (low.coef == nullptr || diagonal == nullptr || block < 1) {
        throw std::invalid_argument("the device solver needs the operator, its diagonal and a positive block");
    }
    return std::make_unique<LayeredDCDevice>(low, diagonal, block, coarse_inverse, device);
}

std::unique_ptr<InnerSolver> make_thermal_hex(const fem::thermal_hex::OperatorView<float>& low,
                                              const float* const diagonal, const int block,
                                              const float* const coarse_inverse, const int device) {
    if (low.coef == nullptr || diagonal == nullptr || block < 1) {
        throw std::invalid_argument("the device solver needs the operator, its diagonal and a positive block");
    }
    return std::make_unique<ThermalHexDevice>(low, diagonal, block, coarse_inverse, device);
}

fem::MpirResult solve_mpir(const std::function<void(const double*, double*)>& apply_high, InnerSolver& inner,
                           const double* const rhs, double* const x, const fem::MpirConfig& config) {
    const Index n = inner.size();
    fem::MpirResult result;
    const double rhs_norm = std::sqrt(lane_sum(n, [rhs](const std::ptrdiff_t i) { return rhs[i] * rhs[i]; }));
    const double scale = rhs_norm > 0.0 ? rhs_norm : 1.0;
    const double target = config.absolute_tolerance + config.relative_tolerance * scale;
    std::vector<double> ax(static_cast<std::size_t>(n), 0.0);
    std::vector<double> residual(static_cast<std::size_t>(n), 0.0);
    std::vector<float> correction(static_cast<std::size_t>(n), 0.0f);
    for (int outer = 0; outer <= config.max_outer_iterations; ++outer) {
        result.outer_iterations = outer;
        apply_high(x, ax.data());
        ++result.high_operator_applications;
        for (Index i = 0; i < n; ++i) {
            residual[i] = rhs[i] - ax[i];
        }
        const double norm =
            std::sqrt(lane_sum(n, [&residual](const std::ptrdiff_t i) { return residual[i] * residual[i]; }));
        result.relative_residual = norm / scale;
        if (norm <= target) {
            result.converged = true;
            break;
        }
        if (outer == config.max_outer_iterations) {
            break;
        }
        const InnerResult step = inner.solve(residual.data(), correction.data(), config.inner_relative_tolerance,
                                             config.max_inner_iterations);
        if (step.not_spd) {
            throw std::runtime_error("inner PCG requires a finite SPD operator");
        }
        result.inner_iterations += step.iterations;
        result.low_operator_applications += step.applications;
        result.history.push_back({outer + 1, result.relative_residual, step.iterations, step.relative_residual});
        for (Index i = 0; i < n; ++i) {
            x[i] += static_cast<double>(correction[i]);
        }
    }
    return result;
}

}  // namespace pcbcore::cuda
