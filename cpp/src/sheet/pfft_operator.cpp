#include "pcbcore/sheet/pfft_operator.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

#include "pcbcore/errors.hpp"
#include "pcbcore/fft.hpp"

namespace pcbcore::sheet {

namespace {

using std::size_t;

int team_size(const int threads, const std::int64_t tasks) {
    return static_cast<int>(std::max<std::int64_t>(1, std::min<std::int64_t>(threads, tasks)));
}

void check_csr(const CsrMatrix& m, const char* name) {
    if (m.indptr.size() != static_cast<size_t>(m.rows) + 1U || m.indices.size() != m.data.size() ||
        (m.rows > 0 && m.indptr.back() != static_cast<std::int64_t>(m.data.size()))) {
        throw InvalidInput(std::string(name) + ": inconsistent CSR arrays");
    }
    for (const std::int64_t column : m.indices) {
        if (column < 0 || column >= m.cols) {
            throw InvalidInput(std::string(name) + ": column index out of range");
        }
    }
}

}  // namespace

void CsrMatrix::multiply(const double* const x, double* const y) const noexcept {
    for (std::int64_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (std::int64_t k = indptr[static_cast<size_t>(r)]; k < indptr[static_cast<size_t>(r) + 1U]; ++k) {
            sum += data[static_cast<size_t>(k)] * x[indices[static_cast<size_t>(k)]];
        }
        y[r] = sum;
    }
}

void CsrMatrix::multiply_transposed(const double* const x, double* const y) const noexcept {
    std::fill(y, y + cols, 0.0);
    for (std::int64_t r = 0; r < rows; ++r) {
        const double xr = x[r];
        for (std::int64_t k = indptr[static_cast<size_t>(r)]; k < indptr[static_cast<size_t>(r) + 1U]; ++k) {
            y[indices[static_cast<size_t>(k)]] += data[static_cast<size_t>(k)] * xr;
        }
    }
}

PfftOperator::PfftOperator(const std::int64_t nodes_y, const std::int64_t nodes_x)
    : nodes_y_(nodes_y),
      nodes_x_(nodes_x),
      padded_rows_(static_cast<size_t>(2 * nodes_y)),
      padded_cols_(static_cast<size_t>(2 * nodes_x)),
      half_cols_(static_cast<size_t>(nodes_x) + 1U) {
    if (nodes_y < 1 || nodes_x < 1) {
        throw InvalidInput("the projection grid needs at least one node per axis");
    }
}

std::int64_t PfftOperator::add_kernel(const double* const table, const int /*threads*/) {
    Spectrum spectrum(padded_rows_ * half_cols_);
    fft::rfft2_padded(table, padded_rows_, padded_cols_, padded_cols_, padded_rows_, padded_cols_, spectrum.data());
    kernels_.push_back(std::move(spectrum));
    return static_cast<std::int64_t>(kernels_.size()) - 1;
}

std::int64_t PfftOperator::add_family(const std::int64_t planes, CsrMatrix projection, std::vector<double> weights,
                                      std::vector<std::int64_t> kernel_of, CsrMatrix correction) {
    if (planes < 1) {
        throw InvalidInput("a family needs at least one plane");
    }
    check_csr(projection, "projection");
    check_csr(correction, "correction");
    const std::int64_t n = projection.rows;
    if (projection.cols != nodes_y_ * nodes_x_) {
        throw InvalidInput("projection columns must be the grid nodes");
    }
    if (weights.size() != static_cast<size_t>(planes * n)) {
        throw InvalidInput("weights must be planes x branches");
    }
    if (kernel_of.size() != static_cast<size_t>(planes * planes)) {
        throw InvalidInput("kernel_of must be planes x planes");
    }
    for (const std::int64_t k : kernel_of) {
        if (k < 0 || k >= static_cast<std::int64_t>(kernels_.size())) {
            throw InvalidInput("kernel_of names a kernel that was not added");
        }
    }
    if (correction.rows != planes * n || correction.cols != planes * n) {
        throw InvalidInput("correction must be (planes branches) square");
    }
    families_.push_back(Family{planes, std::move(projection), std::move(weights), std::move(kernel_of),
                               std::move(correction)});
    return static_cast<std::int64_t>(families_.size()) - 1;
}

std::int64_t PfftOperator::branches(const std::int64_t family) const {
    return families_.at(static_cast<size_t>(family)).projection.rows;
}

std::int64_t PfftOperator::planes(const std::int64_t family) const {
    return families_.at(static_cast<size_t>(family)).planes;
}

std::int64_t PfftOperator::bytes() const noexcept {
    std::int64_t total = 0;
    for (const auto& k : kernels_) {
        total += static_cast<std::int64_t>(k.size() * sizeof(std::complex<double>));
    }
    for (const auto& f : families_) {
        total += static_cast<std::int64_t>((f.projection.data.size() + f.correction.data.size()) *
                                           (sizeof(double) + sizeof(std::int64_t)));
    }
    return total;
}

void PfftOperator::apply(const std::int64_t family_index, const double* const currents, double* const flux,
                         const int threads) const {
    const Family& family = families_.at(static_cast<size_t>(family_index));
    const std::int64_t planes = family.planes;
    const std::int64_t n = family.projection.rows;
    if (n == 0) {
        return;
    }
    const size_t nodes = static_cast<size_t>(nodes_y_ * nodes_x_);
    const size_t spectrum_size = padded_rows_ * half_cols_;
    std::vector<Spectrum> sources(static_cast<size_t>(planes), Spectrum(spectrum_size));
    const int team = team_size(threads, planes);
#pragma omp parallel num_threads(team) if (team > 1)
    {
        std::vector<double> weighted(static_cast<size_t>(n));
        std::vector<double> grid(nodes);
#pragma omp for schedule(static)
        for (std::int64_t p = 0; p < planes; ++p) {
            const double* const w = family.weights.data() + static_cast<size_t>(p * n);
            const double* const in = currents + static_cast<size_t>(p * n);
            for (std::int64_t b = 0; b < n; ++b) {
                weighted[static_cast<size_t>(b)] = in[b] * w[b];
            }
            family.projection.multiply_transposed(weighted.data(), grid.data());
            fft::rfft2_padded(grid.data(), static_cast<size_t>(nodes_y_), static_cast<size_t>(nodes_x_),
                              static_cast<size_t>(nodes_x_), padded_rows_, padded_cols_,
                              sources[static_cast<size_t>(p)].data());
        }
        Spectrum accumulated(spectrum_size);
        std::vector<double> potential_padded(padded_rows_ * padded_cols_);
        std::vector<double> potential(nodes);
        std::vector<double> interpolated(static_cast<size_t>(n));
#pragma omp for schedule(static)
        for (std::int64_t t = 0; t < planes; ++t) {
            std::fill(accumulated.begin(), accumulated.end(), std::complex<double>(0.0, 0.0));
            for (std::int64_t s = 0; s < planes; ++s) {
                const Spectrum& kernel =
                    kernels_[static_cast<size_t>(family.kernel_of[static_cast<size_t>(t * planes + s)])];
                const Spectrum& source = sources[static_cast<size_t>(s)];
                for (size_t k = 0; k < spectrum_size; ++k) {
                    accumulated[k] += kernel[k] * source[k];
                }
            }
            fft::irfft2(accumulated.data(), padded_rows_, padded_cols_, potential_padded.data());
            for (std::int64_t r = 0; r < nodes_y_; ++r) {
                std::copy(potential_padded.data() + static_cast<size_t>(r) * padded_cols_,
                          potential_padded.data() + static_cast<size_t>(r) * padded_cols_ +
                              static_cast<size_t>(nodes_x_),
                          potential.data() + static_cast<size_t>(r * nodes_x_));
            }
            family.projection.multiply(potential.data(), interpolated.data());
            const double* const w = family.weights.data() + static_cast<size_t>(t * n);
            double* const out = flux + static_cast<size_t>(t * n);
            for (std::int64_t b = 0; b < n; ++b) {
                out[b] = w[b] * interpolated[static_cast<size_t>(b)];
            }
        }
    }
    // The near-field correction, row by row over all planes.
    const std::int64_t total = planes * n;
    std::vector<double> near(static_cast<size_t>(total));
    const int rows_team = team_size(threads, total / 4096 + 1);
#pragma omp parallel for schedule(static) num_threads(rows_team) if (rows_team > 1)
    for (std::int64_t r = 0; r < total; ++r) {
        double sum = 0.0;
        for (std::int64_t k = family.correction.indptr[static_cast<size_t>(r)];
             k < family.correction.indptr[static_cast<size_t>(r) + 1U]; ++k) {
            sum += family.correction.data[static_cast<size_t>(k)] *
                   currents[family.correction.indices[static_cast<size_t>(k)]];
        }
        near[static_cast<size_t>(r)] = sum;
    }
    for (std::int64_t r = 0; r < total; ++r) {
        flux[r] += near[static_cast<size_t>(r)];
    }
}

}  // namespace pcbcore::sheet
