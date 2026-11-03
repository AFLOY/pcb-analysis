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

// Spectrum entries a product task carries through every kernel and set: the
// stretch of all the sets' planes, their sums and the kernels' stretch stay
// in a core's cache (64 entries: 1 KiB a plane).
constexpr size_t kStretch = 64;

// How many sets go in one batch (measured on a Ryzen 7 9700X, 32 MiB of L3,
// docs/SHEET_OPERATOR_BATCHING_RESULTS.json): sets whose spectra fit in
// kCacheBytes go together up to that size, where the batch stays in the
// cache from the forward transforms to the inverse ones (more sets than
// that were up to 2.6 times slower a set); larger sets go together up to
// kWorkspaceBytes, where reading each kernel once for all of them is what
// counts (up to 1.2 times faster a set than one at a time).
constexpr size_t kCacheBytes = size_t{24} << 20;
constexpr size_t kWorkspaceBytes = size_t{1} << 30;

// Per-thread buffers that keep their memory from one application to the
// next (OpenMP's threads persist), so no step maps fresh pages.
thread_local std::vector<double> row_scratch;
thread_local std::vector<std::complex<double>> sum_scratch;

template <typename T>
T* scratch(std::vector<T>& buffer, const size_t size) {
    if (buffer.size() < size) {
        buffer.resize(size);
    }
    return buffer.data();
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

void ConvolutionOperator::apply(const double* const currents_x, const double* const currents_y,
                                const double* const currents_z, double* const flux_x, double* const flux_y,
                                double* const flux_z, const int threads) const {
    const double* const* z_in = currents_z != nullptr ? &currents_z : nullptr;
    double* const* z_out = flux_z != nullptr ? &flux_z : nullptr;
    apply_many(1, &currents_x, &currents_y, z_in, &flux_x, &flux_y, z_out, threads);
}

void ConvolutionOperator::apply_many(const std::int64_t count, const double* const* const currents_x,
                                     const double* const* const currents_y, const double* const* const currents_z,
                                     double* const* const flux_x, double* const* const flux_y,
                                     double* const* const flux_z, const int threads) const {
    if (count <= 0) {
        return;
    }
    const bool vertical = currents_z != nullptr && flux_z != nullptr && levels_ > 0;
    const auto planes = static_cast<size_t>(2 * layers_ + (vertical ? levels_ : 0));
    const size_t set_entries = planes * padded_rows_ * half_cols_;
    const size_t set_bytes = set_entries * sizeof(std::complex<double>);
    const size_t limit = set_bytes <= kCacheBytes ? kCacheBytes : kWorkspaceBytes;
    const auto batch = static_cast<std::int64_t>(
        std::max<size_t>(1, std::min<size_t>(static_cast<size_t>(count), limit / set_bytes)));
    const std::lock_guard<std::mutex> lock(workspace_mutex_);
    if (workspace_.size() < static_cast<size_t>(batch) * set_entries) {
        workspace_.resize(static_cast<size_t>(batch) * set_entries);
    }
    for (std::int64_t first = 0; first < count; first += batch) {
        const std::int64_t sets = std::min(batch, count - first);
        apply_batch(sets, currents_x + first, currents_y + first, vertical ? currents_z + first : nullptr,
                    flux_x + first, flux_y + first, vertical ? flux_z + first : nullptr, vertical, threads);
    }
}

void ConvolutionOperator::apply_batch(const std::int64_t count, const double* const* const currents_x,
                                      const double* const* const currents_y, const double* const* const currents_z,
                                      double* const* const flux_x, double* const* const flux_y,
                                      double* const* const flux_z, const bool vertical, const int threads) const {
    const size_t plane = static_cast<size_t>(rows_ * cols_);
    const size_t spectrum_size = padded_rows_ * half_cols_;
    const auto layers = static_cast<size_t>(layers_);
    const size_t planes = 2 * layers + (vertical ? static_cast<size_t>(levels_) : 0);
    const size_t transforms = static_cast<size_t>(count) * planes;
    const size_t row_entries = static_cast<size_t>(rows_) * padded_cols_;
    // Plane p of a set: the x layers, then the y layers, then the levels.
    const auto source = [&](const size_t set, const size_t p) -> const double* {
        if (p < layers) {
            return currents_x[set] + p * plane;
        }
        if (p < 2 * layers) {
            return currents_y[set] + (p - layers) * plane;
        }
        return currents_z[set] + (p - 2 * layers) * plane;
    };
    const auto target = [&](const size_t set, const size_t p) -> double* {
        if (p < layers) {
            return flux_x[set] + p * plane;
        }
        if (p < 2 * layers) {
            return flux_y[set] + (p - layers) * plane;
        }
        return flux_z[set] + (p - 2 * layers) * plane;
    };
    std::complex<double>* const spectra = workspace_.data();

    int team = team_size(threads, static_cast<std::int64_t>(transforms));
#pragma omp parallel for schedule(dynamic, 1) num_threads(team) if (team > 1)
    for (std::int64_t task = 0; task < static_cast<std::int64_t>(transforms); ++task) {
        const auto t = static_cast<size_t>(task);
        fft::rfft2_padded_rows(source(t / planes, t % planes), static_cast<size_t>(rows_),
                               static_cast<size_t>(cols_), static_cast<size_t>(cols_), padded_rows_, padded_cols_,
                               spectra + t * spectrum_size, scratch(row_scratch, row_entries));
    }

    // A family's planes, where they start in a set and the kernels they share.
    struct Family {
        size_t offset;
        size_t count;
        const std::vector<Spectrum>* kernels;
    };
    std::vector<Family> families{{0, layers, &kernels_x_}, {layers, layers, &kernels_y_}};
    if (vertical) {
        families.push_back({2 * layers, static_cast<size_t>(levels_), &kernels_z_});
    }
    const size_t stretches = (spectrum_size + kStretch - 1) / kStretch;
    team = team_size(threads, static_cast<std::int64_t>(stretches));
#pragma omp parallel for schedule(static) num_threads(team) if (team > 1)
    for (std::int64_t task = 0; task < static_cast<std::int64_t>(stretches); ++task) {
        const size_t begin = static_cast<size_t>(task) * kStretch;
        const size_t length = std::min(kStretch, spectrum_size - begin);
        // Every set's sums over this stretch, plane by plane, written back
        // over the spectra once all of them are read.
        std::complex<double>* const sums = scratch(sum_scratch, transforms * kStretch);
        std::fill(sums, sums + transforms * kStretch, std::complex<double>(0.0, 0.0));
        for (const Family& family : families) {
            for (size_t to = 0; to < family.count; ++to) {
                for (size_t from = 0; from < family.count; ++from) {
                    const std::complex<double>* const kernel =
                        (*family.kernels)[pair_index(static_cast<std::int64_t>(to), static_cast<std::int64_t>(from),
                                                     static_cast<std::int64_t>(family.count))]
                            .data() +
                        begin;
                    for (size_t set = 0; set < static_cast<size_t>(count); ++set) {
                        std::complex<double>* const sum = sums + (set * planes + family.offset + to) * kStretch;
                        const std::complex<double>* const spectrum =
                            spectra + (set * planes + family.offset + from) * spectrum_size + begin;
                        for (size_t k = 0; k < length; ++k) {
                            sum[k] += kernel[k] * spectrum[k];
                        }
                    }
                }
            }
        }
        for (size_t t = 0; t < transforms; ++t) {
            std::copy(sums + t * kStretch, sums + t * kStretch + length, spectra + t * spectrum_size + begin);
        }
    }

    team = team_size(threads, static_cast<std::int64_t>(transforms));
#pragma omp parallel for schedule(dynamic, 1) num_threads(team) if (team > 1)
    for (std::int64_t task = 0; task < static_cast<std::int64_t>(transforms); ++task) {
        const auto t = static_cast<size_t>(task);
        fft::irfft2_rows(spectra + t * spectrum_size, padded_rows_, padded_cols_, static_cast<size_t>(rows_),
                         static_cast<size_t>(cols_), target(t / planes, t % planes), static_cast<size_t>(cols_),
                         scratch(row_scratch, row_entries));
    }
}

}  // namespace pcbcore::sheet
