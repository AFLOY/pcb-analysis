// The FP32 inner PCG of the layered DC and thermal hex MPIR solves on a CUDA
// device, and the FP64 outer refinement around it on the host.
//
// The device holds the low-precision operator, the Jacobi diagonal and the
// two-level coarse inverse of a prepared system; the host keeps the FP64
// action (the system's own) and drives the loop.  The algorithm is the CPU
// inner PCG step for step: the operator gathers each node from its elements
// in the CPU order, every inner product is a sum of per-line partials (each a
// lane sum over the line) combined in a fixed tree, and nothing is fused
// into an FMA, so a solve gives the same bits on every run and every launch
// configuration.  Against the CPU it agrees to rounding of the FP32 vector
// updates (the CPU build may fuse them) and therefore to the solver
// tolerance after the outer refinement.
//
// The header has no CUDA types, so host code compiled without nvcc uses it.
#pragma once

#include <functional>
#include <memory>

#include "pcbcore/fem/layered_dc.hpp"
#include "pcbcore/fem/mpir.hpp"
#include "pcbcore/fem/thermal_hex.hpp"

namespace pcbcore::cuda {

[[nodiscard]] int device_count() noexcept;

struct InnerResult {
    int iterations{0};
    int applications{0};
    double relative_residual{1.0};
    bool not_spd{false};
};

// One prepared operator resident on one device.
class InnerSolver {
public:
    virtual ~InnerSolver() = default;
    // Approximately solve A c = rhs in FP32 from c = 0; ``correction`` gets c.
    virtual InnerResult solve(const double* rhs, float* correction, double relative_tolerance,
                              int max_iterations) = 0;
    [[nodiscard]] virtual long long size() const noexcept = 0;
};

// ``diagonal`` and ``coarse_inverse`` (block > 0 with a two-level
// preconditioner, else null) are FP32 host arrays copied to the device.
[[nodiscard]] std::unique_ptr<InnerSolver> make_layered_dc(const fem::layered_dc::OperatorView<float>& low,
                                                           const float* diagonal, int block,
                                                           const float* coarse_inverse, int device);
[[nodiscard]] std::unique_ptr<InnerSolver> make_thermal_hex(const fem::thermal_hex::OperatorView<float>& low,
                                                            const float* diagonal, int block,
                                                            const float* coarse_inverse, int device);

// The MPIR outer loop of fem::*::solve_mpir with the inner solve on the
// device: ``apply_high`` is the system's FP64 action, ``x`` holds the initial
// guess and receives the solution.
[[nodiscard]] fem::MpirResult solve_mpir(const std::function<void(const double*, double*)>& apply_high,
                                         InnerSolver& inner, const double* rhs, double* x,
                                         const fem::MpirConfig& config);

}  // namespace pcbcore::cuda
