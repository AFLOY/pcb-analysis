#include "pcbcore/sheet/hoer_love.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>

#include "pcbcore/errors.hpp"

namespace pcbcore::sheet {

namespace {

using std::size_t;

constexpr double kPi = 3.141592653589793;
// The same expression as VACUUM_PERMEABILITY / (4 pi) on the Python side.
const double kMu0Over4PiExact = (4.0e-7 * kPi) / (4.0 * kPi);

constexpr double kSigns[4][2] = {{1.0, 1.0}, {1.0, -1.0}, {-1.0, 1.0}, {-1.0, -1.0}};

// log(numerator / denominator) where both are positive, else zero: every
// logarithm in the primitive carries a coefficient that vanishes where its
// argument degenerates.
inline double safe_log(const double numerator, const double denominator) noexcept {
    if (denominator > 0.0 && numerator > 0.0) {
        const double ratio = numerator / denominator;
        return ratio > 0.0 ? std::log(ratio) : 0.0;
    }
    return 0.0;
}

// arctan(numerator / denominator), zero where the denominator vanishes.
inline double safe_atan(const double numerator, const double denominator) noexcept {
    return denominator != 0.0 ? std::atan(numerator / denominator) : 0.0;
}

// The sixfold antiderivative of 1/r used by Hoer and Love.
inline double primitive(const double x, const double y, const double z) noexcept {
    const double x2 = x * x;
    const double y2 = y * y;
    const double z2 = z * z;
    const double rho = std::sqrt(x2 + y2 + z2);
    double term = (y2 * z2 / 4.0 - y2 * y2 / 24.0 - z2 * z2 / 24.0) * x * safe_log(x + rho, std::sqrt(y2 + z2));
    term += (x2 * z2 / 4.0 - x2 * x2 / 24.0 - z2 * z2 / 24.0) * y * safe_log(y + rho, std::sqrt(x2 + z2));
    term += (x2 * y2 / 4.0 - x2 * x2 / 24.0 - y2 * y2 / 24.0) * z * safe_log(z + rho, std::sqrt(x2 + y2));
    term += (x2 * x2 + y2 * y2 + z2 * z2 - 3.0 * x2 * y2 - 3.0 * y2 * z2 - 3.0 * z2 * x2) * rho / 60.0;
    term -= (x * y * z * z2 / 6.0) * safe_atan(x * y, z * rho);
    term -= (x * y * y2 * z / 6.0) * safe_atan(x * z, y * rho);
    term -= (x2 * x * y * z / 6.0) * safe_atan(y * z, x * rho);
    return term;
}

// The signed corner sum (before the cross-section scaling) and its largest term.
inline double corner_sum(const Bar& a, const Bar& b, const double du, const double dv, const double dw,
                         double* const largest) noexcept {
    double total = 0.0;
    double biggest = 0.0;
    for (const auto& sx : kSigns) {
        const double x = du + sx[0] * a.length / 2.0 + sx[1] * b.length / 2.0;
        for (const auto& sy : kSigns) {
            const double y = dv + sy[0] * a.width / 2.0 + sy[1] * b.width / 2.0;
            for (const auto& sz : kSigns) {
                const double z = dw + sz[0] * a.thickness / 2.0 + sz[1] * b.thickness / 2.0;
                const double value = primitive(x, y, z);
                total = total + (sx[0] * sx[1]) * (sy[0] * sy[1]) * (sz[0] * sz[1]) * value;
                biggest = std::max(biggest, std::abs(value));
            }
        }
    }
    *largest = biggest;
    return total;
}

// fftfreq(n, d=1/n): 0, 1, ..., then the negative offsets.
inline std::int64_t wrapped_offset(const std::int64_t index, const std::int64_t count) noexcept {
    const std::int64_t positive = (count - 1) / 2 + 1;
    return index < positive ? index : index - count;
}

int team_size(const int threads, const std::int64_t tasks) {
    return static_cast<int>(std::max<std::int64_t>(1, std::min<std::int64_t>(threads, tasks)));
}

[[noreturn]] void lost_precision(const double retained) {
    std::ostringstream message;
    message.precision(1);
    message << std::scientific << "partial inductance lost its precision to cancellation: only " << retained
            << " of the summed magnitude survives. The closed form is for cells near each other; use "
               "mutual_partial_inductance, which switches to the centre-to-centre limit past NEAR_RADIUS_CELLS";
    throw InvalidInput(message.str());
}

}  // namespace

double closed_form_mutual_inductance(const Bar& a, const Bar& b, const double du, const double dv, const double dw,
                                     double* const retained) noexcept {
    double largest = 0.0;
    const double total = corner_sum(a, b, du, dv, dw, &largest);
    if (retained != nullptr) {
        *retained = largest > 0.0 ? std::abs(total) / largest : 1.0;
    }
    const double scale = kMu0Over4PiExact / ((a.width * a.thickness) * (b.width * b.thickness));
    return scale * total;
}

double far_field_mutual_inductance(const Bar& a, const Bar& b, const double du, const double dv,
                                   const double dw) noexcept {
    const double radius = std::sqrt(du * du + dv * dv + dw * dw);
    return radius > 0.0 ? (kMu0Over4PiExact * a.length * b.length) / radius : 0.0;
}

double mutual_partial_inductance(const Bar& a, const Bar& b, const double du, const double dv, const double dw,
                                 const double near_radius, const double scale) noexcept {
    const double radius = std::sqrt(du * du + dv * dv + dw * dw);
    if (radius <= near_radius * scale) {
        return closed_form_mutual_inductance(a, b, du, dv, dw);
    }
    return far_field_mutual_inductance(a, b, du, dv, dw);
}

namespace {

// One wrapped table; ``offset(r, c, du, dv, dw)`` places the second bar.
template <typename Offset>
std::vector<double> tabulate(const std::int64_t rows, const std::int64_t cols, const Bar& a, const Bar& b,
                             const double near_radius, const double scale, const int threads, Offset offset) {
    if (rows < 1 || cols < 1) {
        throw InvalidInput("kernel shape must be positive");
    }
    std::vector<double> table(static_cast<size_t>(rows * cols), 0.0);
    std::vector<double> worst(static_cast<size_t>(rows), 1.0);
    const int team = team_size(threads, rows);
#pragma omp parallel for schedule(static) num_threads(team) if (team > 1)
    for (std::int64_t r = 0; r < rows; ++r) {
        const std::int64_t row_offset = wrapped_offset(r, rows);
        double row_worst = 1.0;
        for (std::int64_t c = 0; c < cols; ++c) {
            double du = 0.0;
            double dv = 0.0;
            double dw = 0.0;
            offset(row_offset, wrapped_offset(c, cols), du, dv, dw);
            const double radius = std::sqrt(du * du + dv * dv + dw * dw);
            double value = 0.0;
            if (radius <= near_radius * scale) {
                double retained = 1.0;
                value = closed_form_mutual_inductance(a, b, du, dv, dw, &retained);
                row_worst = std::min(row_worst, retained);
            } else {
                value = far_field_mutual_inductance(a, b, du, dv, dw);
            }
            table[static_cast<size_t>(r * cols + c)] = value;
        }
        worst[static_cast<size_t>(r)] = row_worst;
    }
    const double retained = *std::min_element(worst.begin(), worst.end());
    if (retained < 1e-11) {
        lost_precision(retained);
    }
    return table;
}

}  // namespace

std::vector<double> build_kernel(const std::int64_t rows, const std::int64_t cols, const Bar& a, const Bar& b,
                                 const double layer_separation, const bool along_x, const int near_radius_cells,
                                 const int threads) {
    if (a.length != b.length || a.width != b.width) {
        throw InvalidInput("the two layers must share their in-plane extents; the offsets of a convolution step by those");
    }
    const double scale = std::max({a.length, a.width, b.length, b.width});
    const double length = a.length;
    const double width = a.width;
    return tabulate(rows, cols, a, b, static_cast<double>(near_radius_cells), scale, threads,
                    [&](const std::int64_t r, const std::int64_t c, double& du, double& dv, double& dw) {
                        if (along_x) {
                            du = static_cast<double>(c) * length;
                            dv = static_cast<double>(r) * width;
                        } else {
                            du = static_cast<double>(r) * length;
                            dv = static_cast<double>(c) * width;
                        }
                        dw = layer_separation;
                    });
}

std::vector<double> build_vertical_kernel(const std::int64_t rows, const std::int64_t cols, const double pitch,
                                          const double span_a, const double span_b, const double center_separation,
                                          const int near_radius_cells, const int threads) {
    if (!(pitch > 0.0)) {
        throw InvalidInput("pitch must be positive");
    }
    const Bar a{span_a, pitch, pitch};
    const Bar b{span_b, pitch, pitch};
    return tabulate(rows, cols, a, b, static_cast<double>(near_radius_cells), pitch, threads,
                    [&](const std::int64_t r, const std::int64_t c, double& du, double& dv, double& dw) {
                        du = center_separation;
                        dv = static_cast<double>(c) * pitch;
                        dw = static_cast<double>(r) * pitch;
                    });
}

double closed_form_arrays(const std::int64_t count, const BarArrays& a, const BarArrays& b,
                          const OffsetArrays& offset, double* const out, const int threads) {
    if (count <= 0) {
        return 1.0;
    }
    constexpr std::int64_t chunk = 4096;
    const std::int64_t chunks = (count + chunk - 1) / chunk;
    std::vector<double> worst(static_cast<size_t>(chunks), 1.0);
    const int team = team_size(threads, chunks);
#pragma omp parallel for schedule(static) num_threads(team) if (team > 1)
    for (std::int64_t block = 0; block < chunks; ++block) {
        double block_worst = 1.0;
        const std::int64_t end = std::min(count, (block + 1) * chunk);
        for (std::int64_t k = block * chunk; k < end; ++k) {
            const Bar first{a.length[k * a.length_stride], a.width[k * a.width_stride],
                            a.thickness[k * a.thickness_stride]};
            const Bar second{b.length[k * b.length_stride], b.width[k * b.width_stride],
                             b.thickness[k * b.thickness_stride]};
            double retained = 1.0;
            out[k] = closed_form_mutual_inductance(first, second, offset.du[k * offset.du_stride],
                                                   offset.dv[k * offset.dv_stride], offset.dw[k * offset.dw_stride],
                                                   &retained);
            block_worst = std::min(block_worst, retained);
        }
        worst[static_cast<size_t>(block)] = block_worst;
    }
    return *std::min_element(worst.begin(), worst.end());
}

}  // namespace pcbcore::sheet
