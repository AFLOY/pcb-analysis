// The prepared scalar (E_z) frequency-domain Maxwell system of one material
// mesh and frequency (frequency_domain.py): per-element 1/mu and reaction
// (j omega sigma - omega^2 eps), the Q1 stiffness and mass, the Dirichlet
// rows, the Jacobi diagonal, the lifted right-hand side, the MPIR solve and
// the element fields and losses.
#pragma once

#include <complex>
#include <cstdint>
#include <vector>

#include "pcbcore/fem/mpir.hpp"
#include "pcbcore/fem/scalar_maxwell.hpp"

namespace pcbcore::fem {

inline constexpr double kMu0 = 4.0e-7 * 3.14159265358979323846;
inline constexpr double kEpsilon0 = 8.8541878128e-12;

struct ScalarMaxwellMeshView {
    int rows{0};  // elements; nodes are (rows + 1, cols + 1)
    int cols{0};
    double pitch_x{0.0};
    double pitch_y{0.0};
    const double* relative_permittivity{nullptr};   // (rows, cols)
    const double* relative_permeability{nullptr};   // (rows, cols)
    const double* conductivity{nullptr};            // (rows, cols)
    const double* loss_tangent{nullptr};            // (rows, cols)
};

struct ScalarMaxwellFields {
    std::vector<std::complex<double>> centre;    // (rows, cols)
    std::vector<std::complex<double>> magnetic;  // (rows, cols, 2)
    std::vector<std::complex<double>> current;   // (rows, cols)
};

class ScalarMaxwellSystem {
public:
    // ``dirichlet`` (nodes) marks the fixed nodes, ``dirichlet_values`` and
    // ``source`` are nodal.  Throws InvalidInput when the Jacobi diagonal is
    // singular at this frequency.
    ScalarMaxwellSystem(const ScalarMaxwellMeshView& mesh, double frequency_hz, const std::uint8_t* dirichlet,
                        const std::complex<double>* dirichlet_values, const std::complex<double>* source);

    [[nodiscard]] std::int64_t size() const noexcept { return size_; }
    [[nodiscard]] const std::vector<std::complex<double>>& diagonal() const noexcept { return diagonal_; }

    void apply_high(const std::complex<double>* x, std::complex<double>* y, int threads) const;
    void apply_low(const std::complex<float>* x, std::complex<float>* y, int threads) const;
    [[nodiscard]] std::vector<std::complex<double>> build_rhs(int threads) const;
    [[nodiscard]] MpirResult solve(const std::complex<double>* rhs, std::complex<double>* x, const MpirConfig& config,
                                   const scalar_maxwell::GmresOptions& gmres, int threads) const;

    [[nodiscard]] ScalarMaxwellFields element_fields(const std::complex<double>* field) const;
    // Conduction and dielectric loss, W per metre of z depth.
    [[nodiscard]] std::pair<double, double> losses(const std::complex<double>* field) const;

private:
    [[nodiscard]] scalar_maxwell::OperatorView<double> high_view(bool all_free) const noexcept;
    [[nodiscard]] scalar_maxwell::OperatorView<float> low_view() const noexcept;

    int rows_{0};
    int cols_{0};
    std::int64_t size_{0};
    double pitch_x_{0.0};
    double pitch_y_{0.0};
    double omega_{0.0};
    std::vector<double> permittivity_;
    std::vector<double> permeability_;
    std::vector<double> conductivity_;
    std::vector<double> loss_tangent_;
    std::vector<std::uint8_t> dirichlet_;
    std::vector<std::complex<double>> dirichlet_values_;
    std::vector<std::complex<double>> source_;

    std::vector<std::complex<double>> inverse_mu_;
    std::vector<std::complex<double>> reaction_;
    std::vector<std::complex<double>> stiffness_;  // (4, 4)
    std::vector<std::complex<double>> mass_;       // (4, 4)
    std::vector<std::uint8_t> free_;
    std::vector<std::uint8_t> all_free_;
    std::vector<double> free_mask_;
    std::vector<double> ones_;
    std::vector<std::complex<double>> diagonal_;
    std::vector<std::complex<float>> inverse_mu_low_;
    std::vector<std::complex<float>> reaction_low_;
    std::vector<std::complex<float>> stiffness_low_;
    std::vector<std::complex<float>> mass_low_;
    std::vector<float> free_mask_low_;
    std::vector<std::complex<float>> diagonal_low_;
};

}  // namespace pcbcore::fem
