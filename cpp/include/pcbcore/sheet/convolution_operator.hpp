// The partial-inductance operator of a uniform layered sheet mesh, applied by
// two-dimensional transforms.
//
// Like-directed branches couple through a kernel that depends only on their
// in-plane offset and on the layer pair, so each (axis, layer pair) is one
// circular convolution on a grid padded to twice the mesh; vertical branches
// couple among their levels the same way.  The kernel tables are computed by
// the caller (closed-form partial inductances); the operator keeps their
// spectra.  An application transforms every layer's currents, multiplies and
// sums the spectra over source layers in layer order, and transforms back:
// each output is computed by one thread in a fixed order, so the result does
// not depend on the thread count, nor on how many sets of currents are
// applied together.
//
// Several sets of currents (the excitations a lockstep Krylov solve carries,
// each as its real and imaginary part) are applied at once: every plane of
// every set is transformed on the team, and the products run over short
// stretches of the spectrum, each reading a stretch of every kernel once for
// all the sets.  The spectra live in a workspace the operator keeps between
// applications, so a solve does not map and fault in fresh pages on every
// step; one application at a time uses it, and concurrent callers wait.
#pragma once

#include <complex>
#include <cstdint>
#include <mutex>
#include <vector>

namespace pcbcore::sheet {

class ConvolutionOperator {
public:
    // ``tables_x``/``tables_y`` hold one padded ``(2 rows) x (2 cols)`` table
    // per layer pair ``first <= second`` in the order (0,0), (0,1), ...,
    // (0,L-1), (1,1), ...; ``tables_z`` the same over vertical levels.
    ConvolutionOperator(std::int64_t layers, std::int64_t rows, std::int64_t cols, std::int64_t levels,
                        const double* tables_x, const double* tables_y, const double* tables_z, int threads);

    // Currents and fluxes are ``(layers, rows, cols)`` (and ``(levels, rows,
    // cols)`` for z, which may be null when there are no levels).
    void apply(const double* currents_x, const double* currents_y, const double* currents_z, double* flux_x,
               double* flux_y, double* flux_z, int threads) const;

    // ``count`` sets of currents, set ``s`` at ``currents_x[s]`` and so on,
    // each given exactly what apply gives it alone.  ``currents_z`` and
    // ``flux_z`` are null for no vertical currents in any set.
    void apply_many(std::int64_t count, const double* const* currents_x, const double* const* currents_y,
                    const double* const* currents_z, double* const* flux_x, double* const* flux_y,
                    double* const* flux_z, int threads) const;

    [[nodiscard]] std::int64_t layers() const noexcept { return layers_; }
    [[nodiscard]] std::int64_t levels() const noexcept { return levels_; }
    [[nodiscard]] std::int64_t spectrum_bytes() const noexcept;

private:
    using Spectrum = std::vector<std::complex<double>>;

    [[nodiscard]] std::size_t pair_index(std::int64_t first, std::int64_t second, std::int64_t count) const noexcept;
    void apply_batch(std::int64_t count, const double* const* currents_x, const double* const* currents_y,
                     const double* const* currents_z, double* const* flux_x, double* const* flux_y,
                     double* const* flux_z, bool vertical, int threads) const;

    std::int64_t layers_;
    std::int64_t rows_;
    std::int64_t cols_;
    std::int64_t levels_;
    std::size_t padded_rows_;
    std::size_t padded_cols_;
    std::size_t half_cols_;
    std::vector<Spectrum> kernels_x_;
    std::vector<Spectrum> kernels_y_;
    std::vector<Spectrum> kernels_z_;
    // The spectra of the sets being applied, kept for the next application.
    mutable std::mutex workspace_mutex_;
    mutable Spectrum workspace_;
};

}  // namespace pcbcore::sheet
