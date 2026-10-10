#include "pcbcore/fem/scalar_maxwell_system.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "pcbcore/errors.hpp"
#include "pcbcore/lane_sum.hpp"

namespace pcbcore::fem {

namespace {

using Complex = std::complex<double>;

constexpr double kStiffness1D[2][2] = {{1.0, -1.0}, {-1.0, 1.0}};
constexpr double kMass1D[2][2] = {{2.0 / 6.0, 1.0 / 6.0}, {1.0 / 6.0, 2.0 / 6.0}};
constexpr double kPi = 3.14159265358979323846;

template <typename To, typename From>
[[nodiscard]] std::vector<To> cast_all(const std::vector<From>& values) {
    std::vector<To> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = static_cast<To>(values[i]);
    }
    return out;
}

[[nodiscard]] std::vector<std::complex<float>> to_complex64(const std::vector<Complex>& values) {
    std::vector<std::complex<float>> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = {static_cast<float>(values[i].real()), static_cast<float>(values[i].imag())};
    }
    return out;
}

}  // namespace

ScalarMaxwellSystem::ScalarMaxwellSystem(const ScalarMaxwellMeshView& mesh, const double frequency_hz,
                                         const std::uint8_t* const dirichlet, const Complex* const dirichlet_values,
                                         const Complex* const source)
    : rows_(mesh.rows), cols_(mesh.cols), pitch_x_(mesh.pitch_x), pitch_y_(mesh.pitch_y) {
    if (rows_ < 1 || cols_ < 1) {
        throw InvalidInput("element_shape axes must be positive");
    }
    const std::int64_t elements = static_cast<std::int64_t>(rows_) * cols_;
    size_ = static_cast<std::int64_t>(rows_ + 1) * (cols_ + 1);
    permittivity_.assign(mesh.relative_permittivity, mesh.relative_permittivity + elements);
    permeability_.assign(mesh.relative_permeability, mesh.relative_permeability + elements);
    conductivity_.assign(mesh.conductivity, mesh.conductivity + elements);
    loss_tangent_.assign(mesh.loss_tangent, mesh.loss_tangent + elements);
    dirichlet_.assign(dirichlet, dirichlet + size_);
    dirichlet_values_.assign(dirichlet_values, dirichlet_values + size_);
    source_.assign(source, source + size_);

    // Stiffness (h_y/h_x) kron(M, S) + (h_x/h_y) kron(S, M), mass h_x h_y kron(M, M).
    stiffness_.assign(16, Complex{});
    mass_.assign(16, Complex{});
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const int yi = i >> 1, xi = i & 1, yj = j >> 1, xj = j & 1;
            stiffness_[4 * i + j] = (pitch_y_ / pitch_x_) * (kMass1D[yi][yj] * kStiffness1D[xi][xj]) +
                                    (pitch_x_ / pitch_y_) * (kStiffness1D[yi][yj] * kMass1D[xi][xj]);
            mass_[4 * i + j] = (pitch_x_ * pitch_y_) * (kMass1D[yi][yj] * kMass1D[xi][xj]);
        }
    }
    omega_ = 2.0 * kPi * frequency_hz;
    inverse_mu_.assign(static_cast<std::size_t>(elements), Complex{});
    reaction_.assign(static_cast<std::size_t>(elements), Complex{});
    for (std::int64_t e = 0; e < elements; ++e) {
        const double epsilon_scale = kEpsilon0 * permittivity_[e];
        const Complex epsilon{epsilon_scale, -(epsilon_scale * loss_tangent_[e])};
        inverse_mu_[e] = Complex{1.0 / (kMu0 * permeability_[e]), 0.0};
        const double square = omega_ * omega_;
        reaction_[e] = Complex{0.0 - square * epsilon.real(), omega_ * conductivity_[e] - square * epsilon.imag()};
    }

    free_.assign(static_cast<std::size_t>(size_), 0U);
    free_mask_.assign(static_cast<std::size_t>(size_), 0.0);
    for (std::int64_t n = 0; n < size_; ++n) {
        free_[n] = dirichlet_[n] != 0U ? 0U : 1U;
        free_mask_[n] = free_[n] != 0U ? 1.0 : 0.0;
    }
    all_free_.assign(static_cast<std::size_t>(size_), 1U);
    ones_.assign(static_cast<std::size_t>(size_), 1.0);

    diagonal_.assign(static_cast<std::size_t>(size_), Complex{});
    const int nc = cols_ + 1;
    for (int corner = 0; corner < 4; ++corner) {
        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < cols_; ++c) {
                const std::int64_t e = static_cast<std::int64_t>(r) * cols_ + c;
                const std::int64_t node = static_cast<std::int64_t>(r + (corner >> 1)) * nc + c + (corner & 1);
                diagonal_[node] += inverse_mu_[e] * stiffness_[5 * corner] + reaction_[e] * mass_[5 * corner];
            }
        }
    }
    double largest = 1.0;
    for (std::int64_t n = 0; n < size_; ++n) {
        if (free_[n] == 0U) {
            diagonal_[n] = Complex{1.0, 0.0};
        }
        largest = std::max(largest, std::abs(diagonal_[n]));
    }
    const double threshold = std::numeric_limits<double>::epsilon() * largest;
    for (std::int64_t n = 0; n < size_; ++n) {
        if (free_[n] != 0U && std::abs(diagonal_[n]) <= threshold) {
            throw InvalidInput("Jacobi diagonal is singular at this frequency");
        }
    }
    inverse_mu_low_ = to_complex64(inverse_mu_);
    reaction_low_ = to_complex64(reaction_);
    stiffness_low_ = to_complex64(stiffness_);
    mass_low_ = to_complex64(mass_);
    free_mask_low_ = cast_all<float>(free_mask_);
    diagonal_low_ = to_complex64(diagonal_);
}

scalar_maxwell::OperatorView<double> ScalarMaxwellSystem::high_view(const bool all_free) const noexcept {
    return {inverse_mu_.data(), reaction_.data(), stiffness_.data(), mass_.data(),
            all_free ? all_free_.data() : free_.data(), all_free ? ones_.data() : free_mask_.data(), rows_, cols_};
}

scalar_maxwell::OperatorView<float> ScalarMaxwellSystem::low_view() const noexcept {
    return {inverse_mu_low_.data(), reaction_low_.data(), stiffness_low_.data(), mass_low_.data(),
            free_.data(), free_mask_low_.data(), rows_, cols_};
}

void ScalarMaxwellSystem::apply_high(const Complex* const x, Complex* const y, const int threads) const {
    scalar_maxwell::apply_high(high_view(false), x, y, threads);
}

void ScalarMaxwellSystem::apply_low(const std::complex<float>* const x, std::complex<float>* const y,
                                    const int threads) const {
    scalar_maxwell::apply_low(low_view(), x, y, threads);
}

std::vector<Complex> ScalarMaxwellSystem::build_rhs(const int threads) const {
    std::vector<Complex> boundary(static_cast<std::size_t>(size_), Complex{});
    for (std::int64_t n = 0; n < size_; ++n) {
        if (dirichlet_[n] != 0U) {
            boundary[n] = dirichlet_values_[n];
        }
    }
    std::vector<Complex> action(static_cast<std::size_t>(size_), Complex{});
    scalar_maxwell::apply_high(high_view(true), boundary.data(), action.data(), threads);
    std::vector<Complex> rhs(static_cast<std::size_t>(size_), Complex{});
    for (std::int64_t n = 0; n < size_; ++n) {
        rhs[n] = dirichlet_[n] != 0U ? boundary[n] : source_[n] - action[n];
    }
    return rhs;
}

MpirResult ScalarMaxwellSystem::solve(const Complex* const rhs, Complex* const x, const MpirConfig& config,
                                      const scalar_maxwell::GmresOptions& gmres, const int threads) const {
    return scalar_maxwell::solve_mpir(low_view(), high_view(false), rhs, x, diagonal_low_.data(), config, gmres,
                                      threads);
}

ScalarMaxwellFields ScalarMaxwellSystem::element_fields(const Complex* const field) const {
    ScalarMaxwellFields out;
    const std::int64_t elements = static_cast<std::int64_t>(rows_) * cols_;
    out.centre.resize(static_cast<std::size_t>(elements));
    out.magnetic.resize(static_cast<std::size_t>(2 * elements));
    out.current.resize(static_cast<std::size_t>(elements));
    const int nc = cols_ + 1;
    for (int r = 0; r < rows_; ++r) {
        for (int c = 0; c < cols_; ++c) {
            const std::int64_t e = static_cast<std::int64_t>(r) * cols_ + c;
            const std::int64_t corner = static_cast<std::int64_t>(r) * nc + c;
            const Complex e00 = field[corner];
            const Complex e01 = field[corner + 1];
            const Complex e10 = field[corner + nc];
            const Complex e11 = field[corner + nc + 1];
            out.centre[e] = (((e00 + e01) + e10) + e11) / 4.0;
            const Complex dx = ((e01 + e11) - (e00 + e10)) / (2.0 * pitch_x_);
            const Complex dy = ((e10 + e11) - (e00 + e01)) / (2.0 * pitch_y_);
            const double scale = omega_ * (kMu0 * permeability_[e]);
            out.magnetic[2 * e] = Complex{-dy.imag(), dy.real()} / scale;    // j dy / (omega mu)
            out.magnetic[2 * e + 1] = Complex{dx.imag(), -dx.real()} / scale;  // -j dx / (omega mu)
            out.current[e] = conductivity_[e] * out.centre[e];
        }
    }
    return out;
}

std::pair<double, double> ScalarMaxwellSystem::losses(const Complex* const field) const {
    const std::int64_t elements = static_cast<std::int64_t>(rows_) * cols_;
    const int nc = cols_ + 1;
    std::vector<double> norm(static_cast<std::size_t>(elements), 0.0);
    for (int r = 0; r < rows_; ++r) {
        for (int c = 0; c < cols_; ++c) {
            const std::int64_t corner = static_cast<std::int64_t>(r) * nc + c;
            const Complex v[4] = {field[corner], field[corner + 1], field[corner + nc], field[corner + nc + 1]};
            double total = 0.0;
            for (int i = 0; i < 4; ++i) {
                Complex row{};
                for (int j = 0; j < 4; ++j) {
                    row += mass_[4 * i + j].real() * v[j];
                }
                total += (std::conj(v[i]) * row).real();
            }
            norm[static_cast<std::size_t>(r) * cols_ + c] = total;
        }
    }
    const double conduction =
        0.5 * lane_sum(elements, [&](const std::ptrdiff_t e) { return conductivity_[e] * norm[e]; });
    const double dielectric =
        0.5 * omega_ * kEpsilon0 *
        lane_sum(elements, [&](const std::ptrdiff_t e) { return permittivity_[e] * loss_tangent_[e] * norm[e]; });
    return {conduction, dielectric};
}

}  // namespace pcbcore::fem
