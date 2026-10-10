// Post-processing of the tiled dipole superposition
// (emc/tiled_dipole_superposition/{far_field,moments,sources}.py): the far
// field of a pattern projected on theta/phi with its radiated power, the net
// dipole moments of a source set, and the current elements of a layered DC
// solve with the closure of its terminal currents.
#pragma once

#include <array>
#include <complex>
#include <cstdint>
#include <vector>

namespace pcbcore::emc {

using Complex = std::complex<double>;

struct FarField {
    std::vector<Complex> field;  // (directions, 3)
    std::vector<Complex> e_theta;
    std::vector<Complex> e_phi;
    double radiated_power{0.0};
};

// E = (j eta k / 4 pi) e^{-jkd} F / d for ``pattern`` F (directions, 3), its
// theta and phi components, and P = eta k^2 / (32 pi^2) sum |F|^2 w.
[[nodiscard]] FarField far_field(const Complex* pattern, const double* theta, const double* phi,
                                 const double* weight, std::int64_t directions, double wavenumber,
                                 double distance, double impedance);

// Net electric moment sum p and magnetic moment 1/2 sum (r - o) x p; the
// origin defaults (``origin`` null) to the |p|-weighted centroid.
[[nodiscard]] std::array<Complex, 6> dipole_moments(const double* position, const Complex* moment,
                                                    std::int64_t count, const double* origin);

struct Dipoles {
    std::vector<double> position;  // (count, 3)
    std::vector<Complex> moment;   // (count, 3)
};

// Element moments J t px py at the centres of active elements (layer order,
// row, column) and via moments I (z_upper - z_lower) at the via nodes.
[[nodiscard]] Dipoles dc_dipoles(int layers, int rows, int cols, const std::uint8_t* active, const double* thickness,
                                 const double* pitch_x, const double* pitch_y, const double* heights,
                                 const double* density, std::int64_t vias, const std::int64_t* via_lower,
                                 const std::int64_t* via_upper, const double* via_current);

// One element per terminal from the current-weighted centroid of the source
// pads to the terminal's centroid; ``terminal_nodes`` are (layer, row, col)
// triples grouped by ``offsets``.
[[nodiscard]] Dipoles terminal_closure(int rows, int cols, const double* pitch_x, const double* pitch_y,
                                       const double* heights, std::int64_t terminals, const std::int64_t* offsets,
                                       const std::int64_t* terminal_nodes, const double* current);

}  // namespace pcbcore::emc
