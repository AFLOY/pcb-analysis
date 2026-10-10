// The layered-PCB DC conduction operator (bilinear Q1 sheets joined by via
// conductances), applied matrix-free, and its MPIR solve: the C++ interface
// the solver core calls without going through Python.  Implemented in
// src/electrical/matrix_free_mpir_fem/native/layered_dc_q1.cpp.
#pragma once

#include <cstdint>

#include "pcbcore/fem/mpir.hpp"

namespace pcbcore::fem::layered_dc {

// Non-owning arrays of one precision.  ``coef`` is (2, layers, rows, cols),
// ``unit`` (2, 4, 4); nodes are (layers, rows + 1, cols + 1); the vias are a
// CSR over nodes (``via_ptr`` nodes + 1, ``via_nbr`` and ``via_g`` per link).
template <typename T>
struct OperatorView {
    const T* coef{nullptr};
    const T* unit{nullptr};
    const std::uint8_t* free_nodes{nullptr};
    const T* free_mask{nullptr};
    const std::int64_t* via_ptr{nullptr};
    const std::int64_t* via_nbr{nullptr};
    const T* via_g{nullptr};
    int layers{0};
    int rows{0};
    int cols{0};

    [[nodiscard]] std::int64_t node_count() const noexcept {
        return static_cast<std::int64_t>(layers) * (rows + 1) * (cols + 1);
    }
};

// y = A x on the FP64 operator (fixed rows identity, as the MPIR sees it).
void apply_high(const OperatorView<double>& op, const double* x, double* y, int threads);

// The coarse space of ``block`` x ``block`` node patches per layer.
[[nodiscard]] CoarseSpace assemble_coarse(const OperatorView<double>& op, int block, int threads);

// Solves A x = rhs; ``x`` holds the initial guess and receives the answer.
// ``coarse_inverse`` (float, size^2 row-major) may be null for Jacobi only.
[[nodiscard]] MpirResult solve_mpir(const OperatorView<float>& low, const OperatorView<double>& high,
                                    const double* rhs, double* x, const float* diagonal, int block,
                                    const float* coarse_inverse, const MpirConfig& config, int threads);

}  // namespace pcbcore::fem::layered_dc
