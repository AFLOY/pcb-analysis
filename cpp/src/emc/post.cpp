#include "pcbcore/emc/post.hpp"

#include <cmath>

#include "pcbcore/errors.hpp"
#include "pcbcore/lane_sum.hpp"

namespace pcbcore::emc {

namespace {

constexpr double kPi = 3.14159265358979323846;

[[nodiscard]] std::vector<double> node_edges(const double* pitch, const int count) {
    std::vector<double> edges(static_cast<std::size_t>(count + 1), 0.0);
    for (int i = 0; i < count; ++i) {
        edges[i + 1] = edges[i] + pitch[i];
    }
    return edges;
}

}  // namespace

FarField far_field(const Complex* const pattern, const double* const theta, const double* const phi,
                   const double* const weight, const std::int64_t directions, const double k, const double distance,
                   const double eta) {
    FarField out;
    out.field.resize(static_cast<std::size_t>(3 * directions));
    out.e_theta.resize(static_cast<std::size_t>(directions));
    out.e_phi.resize(static_cast<std::size_t>(directions));
    const Complex scale = Complex{0.0, eta * k / (4.0 * kPi)} * std::exp(Complex{0.0, -k * distance}) / distance;
    std::vector<double> intensity(static_cast<std::size_t>(directions), 0.0);
    for (std::int64_t d = 0; d < directions; ++d) {
        const double ct = std::cos(theta[d]);
        const double st = std::sin(theta[d]);
        const double cp = std::cos(phi[d]);
        const double sp = std::sin(phi[d]);
        const double theta_hat[3] = {ct * cp, ct * sp, -st};
        const double phi_hat[3] = {-sp, cp, 0.0};
        Complex e_theta{};
        Complex e_phi{};
        double squared = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const Complex f = pattern[3 * d + axis];
            const Complex e = scale * f;
            out.field[3 * d + axis] = e;
            e_theta += e * theta_hat[axis];
            e_phi += e * phi_hat[axis];
            squared += std::norm(f);
        }
        out.e_theta[d] = e_theta;
        out.e_phi[d] = e_phi;
        intensity[d] = squared * weight[d];
    }
    out.radiated_power = eta * k * k / (32.0 * kPi * kPi) *
                         lane_sum(directions, [&](const std::ptrdiff_t d) { return intensity[d]; });
    return out;
}

std::array<Complex, 6> dipole_moments(const double* const position, const Complex* const moment,
                                      const std::int64_t count, const double* const origin_in) {
    double origin[3] = {0.0, 0.0, 0.0};
    if (origin_in != nullptr) {
        for (int a = 0; a < 3; ++a) {
            origin[a] = origin_in[a];
        }
    } else {
        std::vector<double> weight(static_cast<std::size_t>(count), 0.0);
        for (std::int64_t i = 0; i < count; ++i) {
            weight[i] = std::sqrt(std::norm(moment[3 * i]) + std::norm(moment[3 * i + 1]) + std::norm(moment[3 * i + 2]));
        }
        const double total = lane_sum(count, [&](const std::ptrdiff_t i) { return weight[i]; });
        for (int a = 0; a < 3; ++a) {
            origin[a] = total > 0.0
                            ? lane_sum(count, [&](const std::ptrdiff_t i) { return position[3 * i + a] * weight[i]; }) / total
                            : (count > 0 ? lane_sum(count, [&](const std::ptrdiff_t i) { return position[3 * i + a]; }) /
                                               static_cast<double>(count)
                                         : std::nan(""));
        }
    }
    std::array<Complex, 6> out{};
    for (int a = 0; a < 3; ++a) {
        const double re = lane_sum(count, [&](const std::ptrdiff_t i) { return moment[3 * i + a].real(); });
        const double im = lane_sum(count, [&](const std::ptrdiff_t i) { return moment[3 * i + a].imag(); });
        out[a] = Complex{re, im};
    }
    const auto cross = [&](const std::ptrdiff_t i, const int a) {
        const int b = (a + 1) % 3;
        const int c = (a + 2) % 3;
        const double rb = position[3 * i + b] - origin[b];
        const double rc = position[3 * i + c] - origin[c];
        return rb * moment[3 * i + c] - rc * moment[3 * i + b];
    };
    for (int a = 0; a < 3; ++a) {
        const double re = lane_sum(count, [&](const std::ptrdiff_t i) { return cross(i, a).real(); });
        const double im = lane_sum(count, [&](const std::ptrdiff_t i) { return cross(i, a).imag(); });
        out[3 + a] = 0.5 * Complex{re, im};
    }
    return out;
}

Dipoles dc_dipoles(const int layers, const int rows, const int cols, const std::uint8_t* const active,
                   const double* const thickness, const double* const pitch_x, const double* const pitch_y,
                   const double* const heights, const double* const density, const std::int64_t vias,
                   const std::int64_t* const via_lower, const std::int64_t* const via_upper,
                   const double* const via_current) {
    const std::vector<double> x = node_edges(pitch_x, cols);
    const std::vector<double> y = node_edges(pitch_y, rows);
    Dipoles out;
    for (int l = 0; l < layers; ++l) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                const std::int64_t e = (static_cast<std::int64_t>(l) * rows + r) * cols + c;
                if (active[e] == 0U) {
                    continue;
                }
                const double volume = thickness[l] * (pitch_y[r] * pitch_x[c]);
                out.position.insert(out.position.end(), {0.5 * (x[c] + x[c + 1]), 0.5 * (y[r] + y[r + 1]), heights[l]});
                out.moment.insert(out.moment.end(),
                                  {Complex{density[2 * e] * volume, 0.0}, Complex{density[2 * e + 1] * volume, 0.0}, Complex{}});
            }
        }
    }
    const std::int64_t plane = static_cast<std::int64_t>(rows + 1) * (cols + 1);
    for (std::int64_t k = 0; k < vias; ++k) {
        const std::int64_t lower_layer = via_lower[k] / plane;
        const std::int64_t upper_layer = via_upper[k] / plane;
        if (lower_layer < 0 || lower_layer >= layers || upper_layer < 0 || upper_layer >= layers) {
            throw InvalidInput("a via ends outside the layers");
        }
        const std::int64_t in_plane = via_lower[k] % plane;
        const std::int64_t row = in_plane / (cols + 1);
        const std::int64_t col = in_plane % (cols + 1);
        const double span = heights[upper_layer] - heights[lower_layer];
        out.position.insert(out.position.end(), {x[col], y[row], 0.5 * (heights[upper_layer] + heights[lower_layer])});
        out.moment.insert(out.moment.end(), {Complex{}, Complex{}, Complex{via_current[k] * span, 0.0}});
    }
    return out;
}

Dipoles terminal_closure(const int rows, const int cols, const double* const pitch_x, const double* const pitch_y,
                         const double* const heights, const std::int64_t terminals, const std::int64_t* const offsets,
                         const std::int64_t* const nodes, const double* const current) {
    const std::vector<double> x = node_edges(pitch_x, cols);
    const std::vector<double> y = node_edges(pitch_y, rows);
    std::vector<std::array<double, 3>> centroid(static_cast<std::size_t>(terminals));
    for (std::int64_t t = 0; t < terminals; ++t) {
        const std::int64_t begin = offsets[t];
        const std::int64_t count = offsets[t + 1] - begin;
        if (count < 1) {
            throw InvalidInput("a terminal needs at least one node");
        }
        const auto mean = [&](const auto& value) {
            return lane_sum(count, [&](const std::ptrdiff_t i) { return value(nodes + 3 * (begin + i)); }) /
                   static_cast<double>(count);
        };
        centroid[t] = {mean([&](const std::int64_t* n) { return x[n[2]]; }),
                       mean([&](const std::int64_t* n) { return y[n[1]]; }),
                       mean([&](const std::int64_t* n) { return heights[n[0]]; })};
    }
    Dipoles out;
    double source_current = 0.0;
    double star[3] = {0.0, 0.0, 0.0};
    for (std::int64_t t = 0; t < terminals; ++t) {
        if (current[t] > 0.0) {
            source_current += current[t];
            for (int a = 0; a < 3; ++a) {
                star[a] += centroid[t][a] * current[t];
            }
        }
    }
    if (!(source_current > 0.0)) {
        return out;
    }
    for (double& value : star) {
        value /= source_current;
    }
    // Current flows externally from sink pads to the star and on to source
    // pads; either way the element points from star to pad with weight I.
    for (std::int64_t t = 0; t < terminals; ++t) {
        if (current[t] == 0.0) {
            continue;
        }
        for (int a = 0; a < 3; ++a) {
            out.position.push_back(0.5 * (centroid[t][a] + star[a]));
            out.moment.emplace_back(current[t] * (centroid[t][a] - star[a]), 0.0);
        }
    }
    return out;
}

}  // namespace pcbcore::emc
