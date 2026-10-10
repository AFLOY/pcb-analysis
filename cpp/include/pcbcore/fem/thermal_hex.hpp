// The hexahedral Q1 conduction operator of a layered thermal mesh (with
// lumped Robin conductances), applied matrix-free, and its MPIR solve: the
// C++ interface the solver core calls without going through Python.
// Implemented in src/thermal/matrix_free_mpir_fem/native/hex_q1.cpp.
#pragma once

#include <cstdint>

#include "pcbcore/fem/mpir.hpp"

namespace pcbcore::fem::thermal_hex {

// Non-owning arrays of one precision.  ``coef`` is (3, slabs, rows, cols),
// ``unit`` (3, 8, 8); nodes are (slabs + 1, rows + 1, cols + 1) and ``robin``
// holds one lumped conductance per node.
template <typename T>
struct OperatorView {
    const T* coef{nullptr};
    const T* unit{nullptr};
    const T* robin{nullptr};
    const std::uint8_t* free_nodes{nullptr};
    const T* free_mask{nullptr};
    int slabs{0};
    int rows{0};
    int cols{0};

    [[nodiscard]] std::int64_t node_count() const noexcept {
        return static_cast<std::int64_t>(slabs + 1) * (rows + 1) * (cols + 1);
    }
};

void apply_high(const OperatorView<double>& op, const double* x, double* y, int threads);
void apply_low(const OperatorView<float>& op, const float* x, float* y, int threads);

[[nodiscard]] CoarseSpace assemble_coarse(const OperatorView<double>& op, int block, int threads);

[[nodiscard]] MpirResult solve_mpir(const OperatorView<float>& low, const OperatorView<double>& high,
                                    const double* rhs, double* x, const float* diagonal, int block,
                                    const float* coarse_inverse, const MpirConfig& config, int threads);

}  // namespace pcbcore::fem::thermal_hex
