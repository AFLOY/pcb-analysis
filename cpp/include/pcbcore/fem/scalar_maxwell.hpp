// The frequency-domain scalar Maxwell Q1 operator (one complex potential per
// node of a 2-D element grid), applied matrix-free, and its MPIR solve: FP64
// complex outer residual around a complex64 restarted GMRES.  Implemented in
// src/electrical/matrix_free_mpir_fem/native/scalar_maxwell_q1.cpp.
#pragma once

#include <complex>
#include <cstdint>

#include "pcbcore/fem/mpir.hpp"

namespace pcbcore::fem::scalar_maxwell {

// Non-owning arrays of one precision: per-element 1/mu and reaction, the 4x4
// unit stiffness and mass, nodes (rows + 1) x (cols + 1).
template <typename T>
struct OperatorView {
    const std::complex<T>* inverse_mu{nullptr};
    const std::complex<T>* reaction{nullptr};
    const std::complex<T>* stiffness{nullptr};
    const std::complex<T>* mass{nullptr};
    const std::uint8_t* free_nodes{nullptr};
    const T* free_mask{nullptr};
    int element_rows{0};
    int element_columns{0};

    [[nodiscard]] std::int64_t node_count() const noexcept {
        return static_cast<std::int64_t>(element_rows + 1) * (element_columns + 1);
    }
};

struct GmresOptions {
    int restart{30};
    bool cgs2{false};        // classical Gram-Schmidt twice instead of modified
    bool float_dots{false};  // block-float dot accumulation instead of double
};

void apply_high(const OperatorView<double>& op, const std::complex<double>* x, std::complex<double>* y,
                int threads);

[[nodiscard]] MpirResult solve_mpir(const OperatorView<float>& low, const OperatorView<double>& high,
                                    const std::complex<double>* rhs, std::complex<double>* x,
                                    const std::complex<float>* diagonal, const MpirConfig& config,
                                    const GmresOptions& gmres, int threads);

}  // namespace pcbcore::fem::scalar_maxwell
