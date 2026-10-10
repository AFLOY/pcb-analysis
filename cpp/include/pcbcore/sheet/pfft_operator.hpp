// Application of the precorrected-FFT (pFFT) partial-inductance operator of a
// graded sheet mesh.
//
// A family is one set of like-directed branches over ``planes`` stacked
// planes (the layers for x or y branches, the vertical levels for z).  Its
// application is
//
//   far[t] = w[t] * P (sum_s K(t, s) * (P^T (w[s] * I[s])))   (FFT convolution)
//   flux   = far + C I                                         (precorrection)
//
// with P the branch-to-grid projection (Lagrange stencils over the bar), w a
// per-plane, per-branch weight (bar length, or the level's span), K the grid
// kernel of the two planes' separation, and C the sparse near-field
// correction.  The projection, weights, kernels and correction are built by
// the caller; this class applies them.  Each plane's transform and each
// target plane's sum run on their own thread in a fixed order, so the result
// does not depend on the thread count.
#pragma once

#include <complex>
#include <cstdint>
#include <vector>

namespace pcbcore::sheet {

struct CsrMatrix {
    std::int64_t rows{0};
    std::int64_t cols{0};
    std::vector<std::int64_t> indptr;
    std::vector<std::int64_t> indices;
    std::vector<double> data;

    // y = A x (row by row, entries in stored order).
    void multiply(const double* x, double* y) const noexcept;
    // y = A^T x, accumulated row by row into y (which is zeroed first).
    void multiply_transposed(const double* x, double* y) const noexcept;
};

class PfftOperator {
public:
    PfftOperator(std::int64_t nodes_y, std::int64_t nodes_x);

    // A kernel table of the padded grid ``(2 nodes_y) x (2 nodes_x)``; returns its index.
    std::int64_t add_kernel(const double* table, int threads);

    // ``weights`` is planes x branches; ``kernel_of`` is planes x planes
    // indices into the added kernels; ``correction`` is (planes branches)^2.
    std::int64_t add_family(std::int64_t planes, CsrMatrix projection, std::vector<double> weights,
                            std::vector<std::int64_t> kernel_of, CsrMatrix correction);

    // ``currents`` and ``flux`` are planes x branches of that family.
    void apply(std::int64_t family, const double* currents, double* flux, int threads) const;

    [[nodiscard]] std::int64_t branches(std::int64_t family) const;
    [[nodiscard]] std::int64_t planes(std::int64_t family) const;
    [[nodiscard]] std::int64_t bytes() const noexcept;

private:
    using Spectrum = std::vector<std::complex<double>>;
    struct Family {
        std::int64_t planes;
        CsrMatrix projection;
        std::vector<double> weights;
        std::vector<std::int64_t> kernel_of;
        CsrMatrix correction;
    };

    std::int64_t nodes_y_;
    std::int64_t nodes_x_;
    std::size_t padded_rows_;
    std::size_t padded_cols_;
    std::size_t half_cols_;
    std::vector<Spectrum> kernels_;
    std::vector<Family> families_;
};

}  // namespace pcbcore::sheet
