// Construction of the precorrected-FFT operator of a graded sheet mesh: the
// Lagrange projection of each bar onto the uniform grid, and the near-field
// precorrection (exact closed form minus what the grid credits each near
// pair with), family by family.
//
// Repeated pair configurations are evaluated once: pairs are grouped by their
// quantised geometry (1e-12 m) with a stable sort, the representative of each
// group being its first pair in list order, exactly as the NumPy build groups
// them.  Two choices the NumPy build makes with NumPy's own arithmetic are
// taken from the caller: whether the bars are sampled along x (np.mean of the
// extents) and the layer separations (Python's round(., 15)).
#pragma once

#include <cstdint>
#include <vector>

#include "pcbcore/sheet/pfft_operator.hpp"

namespace pcbcore::sheet {

struct ProjectionGridSpec {
    double origin_x{0.0};
    double origin_y{0.0};
    double pitch{0.0};
    std::int64_t nodes_x{0};
    std::int64_t nodes_y{0};
    int order{3};
    int near_radius_cells{3};
    int preconditioner_radius_cells{1};
};

struct PfftFamilyParts {
    CsrMatrix projection;           // branches x grid nodes
    CsrMatrix correction;           // (planes branches)^2: exact - grid for near pairs
    CsrMatrix near_exact;           // (planes branches)^2: exact, within the preconditioner radius
    std::vector<double> self_value; // planes x branches: each bar's own partial inductance
};

// ``mu0 / (4 pi r)`` between grid points at every wrapped offset of the padded
// grid, zero at r = 0.
[[nodiscard]] std::vector<double> pfft_kernel_table(const ProjectionGridSpec& grid, double separation);

// The in-plane family of one axis: ``n_rows x n_cols`` possible branches
// (row-major) with centres, lengths along the current and widths; ``layer_z``,
// ``layer_thickness`` per layer and ``separation`` (layers x layers, rounded
// as the caller keys them).
[[nodiscard]] PfftFamilyParts pfft_inplane_family(const ProjectionGridSpec& grid, bool axis_x, bool sample_along_x,
                                                  std::int64_t n_rows, std::int64_t n_cols, const double* centre_x,
                                                  const double* centre_y, const double* length, const double* width,
                                                  std::int64_t layers, const double* layer_z,
                                                  const double* layer_thickness, const double* separation,
                                                  int threads);

// The vertical family: one bar per cell (``rows x cols``) standing ``span``
// tall at each level, centred at the level's mid-plane ``z_mid``;
// ``separation`` is levels x levels.
[[nodiscard]] PfftFamilyParts pfft_vertical_family(const ProjectionGridSpec& grid, bool sample_along_x,
                                                   std::int64_t rows, std::int64_t cols, const double* centre_x,
                                                   const double* centre_y, const double* pitch_x,
                                                   const double* pitch_y, std::int64_t levels, const double* span,
                                                   const double* z_mid, const double* separation, int threads);

}  // namespace pcbcore::sheet
