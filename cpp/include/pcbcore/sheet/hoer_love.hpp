// Partial mutual inductance of parallel rectangular bars: the Hoer-Love closed
// form (a signed sum of a sixfold antiderivative of 1/r at the 64 corner
// differences of two boxes) and the centre-to-centre far limit, and the
// convolution tables a uniform sheet mesh is built from.
//
// Every term is added in the order the NumPy implementation adds it, and each
// table entry is computed on its own, so a table does not depend on the
// thread count.
#pragma once

#include <cstdint>
#include <vector>

namespace pcbcore::sheet {

struct Bar {
    double length;     // along the current
    double width;      // across it, in plane
    double thickness;  // vertical
};

// The closed form for bars ``a`` and ``b`` whose centres are ``(du, dv, dw)``
// apart (along, across, vertical).  ``retained`` receives |sum| / max |term|,
// the fraction of the signed sum cancellation left.
[[nodiscard]] double closed_form_mutual_inductance(const Bar& a, const Bar& b, double du, double dv, double dw,
                                                   double* retained = nullptr) noexcept;

// mu0/(4 pi) l_a l_b / r, zero at r = 0.
[[nodiscard]] double far_field_mutual_inductance(const Bar& a, const Bar& b, double du, double dv,
                                                 double dw) noexcept;

// The closed form within ``near_radius_cells * scale`` of the first bar, the
// far limit beyond (scale: the largest in-plane extent of either bar unless
// given).
[[nodiscard]] double mutual_partial_inductance(const Bar& a, const Bar& b, double du, double dv, double dw,
                                               double near_radius, double scale) noexcept;

// Coupling of a bar to every wrapped in-plane offset of a ``rows x cols``
// grid (negative offsets at the end of each axis), current along x (columns)
// or y (rows).  ``a`` and ``b`` share their in-plane extents.
[[nodiscard]] std::vector<double> build_kernel(std::int64_t rows, std::int64_t cols, const Bar& a, const Bar& b,
                                               double layer_separation, bool along_x, int near_radius_cells,
                                               int threads);

// Coupling of two levels of vertical branches (spans ``span_a``/``span_b``,
// mid-planes ``center_separation`` apart) over every wrapped offset.
[[nodiscard]] std::vector<double> build_vertical_kernel(std::int64_t rows, std::int64_t cols, double pitch,
                                                        double span_a, double span_b, double center_separation,
                                                        int near_radius_cells, int threads);

// The closed form over ``count`` pairs with per-pair bars and offsets (each
// input an array of ``count``; a stride of 0 broadcasts a scalar).  Returns
// the smallest retained fraction.
struct BarArrays {
    const double* length;
    const double* width;
    const double* thickness;
    std::int64_t length_stride;
    std::int64_t width_stride;
    std::int64_t thickness_stride;
};
struct OffsetArrays {
    const double* du;
    const double* dv;
    const double* dw;
    std::int64_t du_stride;
    std::int64_t dv_stride;
    std::int64_t dw_stride;
};
double closed_form_arrays(std::int64_t count, const BarArrays& a, const BarArrays& b, const OffsetArrays& offset,
                          double* out, int threads);

}  // namespace pcbcore::sheet
