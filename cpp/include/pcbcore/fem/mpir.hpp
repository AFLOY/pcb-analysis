// Shared types of the mixed-precision iterative refinement (MPIR) solves of
// the matrix-free Q1 operators: an FP64 outer residual loop around an FP32
// (or complex64) inner Krylov solve.
#pragma once

#include <cstdint>
#include <vector>

namespace pcbcore::fem {

struct MpirConfig {
    double relative_tolerance{1.0e-10};
    double absolute_tolerance{0.0};
    double inner_relative_tolerance{2.0e-3};
    int max_outer_iterations{8};
    int max_inner_iterations{200};
};

// One correction: the FP64 relative residual before it, and its inner solve.
struct MpirStep {
    int outer_iteration{0};
    double high_relative_residual{0.0};
    int inner_iterations{0};
    double inner_relative_residual{0.0};
};

struct MpirResult {
    bool converged{false};
    int outer_iterations{0};
    int inner_iterations{0};
    double relative_residual{1.0};
    int high_operator_applications{0};
    int low_operator_applications{0};
    std::vector<MpirStep> history;
};

// The two-level coarse space: Z^T A Z (row-major) and its symmetric inverse.
struct CoarseSpace {
    std::int64_t size{0};
    std::vector<double> matrix;
    std::vector<double> inverse;
};

}  // namespace pcbcore::fem
