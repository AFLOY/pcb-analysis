#include "pcbcore/sheet/convolution_operator.hpp"

#include <algorithm>
#include <cstddef>

#include "pcbcore/errors.hpp"
#include "pcbcore/fft.hpp"

namespace pcbcore::sheet {

namespace {

using std::size_t;

std::int64_t pair_count(const std::int64_t count) { return count * (count + 1) / 2; }

int team_size(const int threads, const std::int64_t tasks) {
    return static_cast<int>(std::max<std::int64_t>(1, std::min<std::int64_t>(threads, tasks)));
}

}  // namespace

size_t ConvolutionOperator::pair_index(const std::int64_t first, const std::int64_t second,
                                       const std::int64_t count) const noexcept {
    // Row-major upper triangle: (0,0) .. (0,count-1), (1,1) .. ; reciprocity
    // makes (a, b) and (b, a) the same kernel.
    const std::int64_t a = std::min(first, second);
    const std::int64_t b = std::max(first, second);
    return static_cast<size_t>(a * count - a * (a - 1) / 2 + (b - a));
}

ConvolutionOperator::ConvolutionOperator(const std::int64_t layers, const std::int64_t rows,
                                         const std::int64_t cols, const std::int64_t levels,
                                         const double* const tables_x, const double* const tables_y,
                                         const double* const tables_z, const int threads)
    : layers_(layers),
      rows_(rows),
      cols_(cols),
      levels_(levels),
      padded_rows_(static_cast<size_t>(2 * rows)),
      padded_cols_(static_cast<size_t>(2 * cols)),
      half_cols_(static_cast<size_t>(cols) + 1U) {
    if (layers < 1 || rows < 1 || cols < 1 || levels < 0) {
        throw InvalidInput("the mesh needs at least one layer, row and column");
    }
    if (tables_x == nullptr || tables_y == nullptr || (levels > 0 && tables_z == nullptr)) {
        throw InvalidInput("missing kernel tables");
    }
    const size_t table_size = padded_rows_ * padded_cols_;
    const size_t spectrum_size = padded_rows_ * half_cols_;
    const std::int64_t in_plane = pair_count(layers);
    const std::int64_t vertical = pair_count(levels);
    kernels_x_.assign(static_cast<size_t>(in_plane), Spectrum(spectrum_size));
    kernels_y_.assign(static_cast<size_t>(in_plane), Spectrum(spectrum_size));
    kernels_z_.assign(static_cast<size_t>(vertical), Spectrum(spectrum_size));
    const std::int64_t tasks = 2 * in_plane + vertical;
    const int team = team_size(threads, tasks);
#pragma omp parallel for schedule(dynamic, 1) num_threads(team) if (team > 1)
    for (std::int64_t task = 0; task < tasks; ++task) {
        const double* table = nullptr;
        Spectrum* target = nullptr;
        if (task < in_plane) {
            table = tables_x + static_cast<size_t>(task) * table_size;
            target = &kernels_x_[static_cast<size_t>(task)];
        } else if (task < 2 * in_plane) {
            table = tables_y + static_cast<size_t>(task - in_plane) * table_size;
            target = &kernels_y_[static_cast<size_t>(task - in_plane)];
        } else {
            table = tables_z + static_cast<size_t>(task - 2 * in_plane) * table_size;
            target = &kernels_z_[static_cast<size_t>(task - 2 * in_plane)];
        }
        fft::rfft2_padded(table, padded_rows_, padded_cols_, padded_cols_, padded_rows_, padded_cols_,
                          target->data());
    }
}

std::int64_t ConvolutionOperator::spectrum_bytes() const noexcept {
    const auto per = static_cast<std::int64_t>(padded_rows_ * half_cols_ * sizeof(std::complex<double>));
    return per * static_cast<std::int64_t>(kernels_x_.size() + kernels_y_.size() + kernels_z_.size());
}

void ConvolutionOperator::convolve(const double* const currents, double* const flux, const std::int64_t count,
                                   const std::vector<Spectrum>& kernels, const int threads) const {
    const size_t plane = static_cast<size_t>(rows_ * cols_);
    const size_t spectrum_size = padded_rows_ * half_cols_;
    std::vector<Spectrum> sources(static_cast<size_t>(count), Spectrum(spectrum_size));
    const int team = team_size(threads, count);
#pragma omp parallel num_threads(team) if (team > 1)
    {
#pragma omp for schedule(static)
        for (std::int64_t layer = 0; layer < count; ++layer) {
            fft::rfft2_padded(currents + static_cast<size_t>(layer) * plane, static_cast<size_t>(rows_),
                              static_cast<size_t>(cols_), static_cast<size_t>(cols_), padded_rows_, padded_cols_,
                              sources[static_cast<size_t>(layer)].data());
        }
        Spectrum accumulated(spectrum_size);
        std::vector<double> product(padded_rows_ * padded_cols_);
#pragma omp for schedule(static)
        for (std::int64_t target = 0; target < count; ++target) {
            std::fill(accumulated.begin(), accumulated.end(), std::complex<double>(0.0, 0.0));
            for (std::int64_t source = 0; source < count; ++source) {
                const Spectrum& kernel = kernels[pair_index(target, source, count)];
                const Spectrum& spectrum = sources[static_cast<size_t>(source)];
                for (size_t k = 0; k < spectrum_size; ++k) {
                    accumulated[k] += kernel[k] * spectrum[k];
                }
            }
            fft::irfft2(accumulated.data(), padded_rows_, padded_cols_, product.data());
            double* const out = flux + static_cast<size_t>(target) * plane;
            for (std::int64_t r = 0; r < rows_; ++r) {
                std::copy(product.data() + static_cast<size_t>(r) * padded_cols_,
                          product.data() + static_cast<size_t>(r) * padded_cols_ + static_cast<size_t>(cols_),
                          out + static_cast<size_t>(r * cols_));
            }
        }
    }
}

void ConvolutionOperator::apply(const double* const currents_x, const double* const currents_y,
                                const double* const currents_z, double* const flux_x, double* const flux_y,
                                double* const flux_z, const int threads) const {
    convolve(currents_x, flux_x, layers_, kernels_x_, threads);
    convolve(currents_y, flux_y, layers_, kernels_y_, threads);
    if (currents_z != nullptr && flux_z != nullptr && levels_ > 0) {
        convolve(currents_z, flux_z, levels_, kernels_z_, threads);
    }
}

}  // namespace pcbcore::sheet
