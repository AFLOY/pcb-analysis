// Near-field kernel moments of RT0 element pairs (the hybrid triangle/square sheet mesh).
//
// For each pair (outer element E, inner polygon F) and vertical offsets h_k with
// weights w_k: W = sum_k w_k int_E int_F [1; r - c_E][1; r' - c_F]^T / |r - r'|_h,
// the outer integral by the outer element's quadrature, the inner one in closed
// form (Wilton et al., IEEE Trans. AP 32(3), 1984).  Python's
// electrical.sheet_peec.hybrid_rt0.near_moments is the reference.
#pragma once

#include <cstdint>
#include <vector>

namespace pcbcore::sheet {

// outer_points (Ea, q, 2), outer_weights (Ea, q), outer_centroid (Ea, 2);
// inner_polygons (Eb, 4, 2) counter-clockwise, inner_sides (Eb) 3 or 4,
// inner_centroid (Eb, 2); pairs (P) of indices; heights/height_weights (H).
// Returns (P, 3, 3) row-major.
std::vector<double> rt0_near_moments(const double* outer_points, const double* outer_weights,
                                     const double* outer_centroid, std::int64_t outer_count, int quadrature,
                                     const double* inner_polygons, const std::int32_t* inner_sides,
                                     const double* inner_centroid, std::int64_t inner_count,
                                     const std::int64_t* pair_outer, const std::int64_t* pair_inner,
                                     std::int64_t pairs, const double* heights, const double* height_weights,
                                     int height_count, int threads);

// The two potentials of one polygon at one point: P0 = int 1/R, P1 = int (rho' - rho)/R.
void polygon_potentials(const double* polygon, int sides, double x, double y, double height, double& p0,
                        double& p1x, double& p1y);

}  // namespace pcbcore::sheet

namespace pcbcore::sheet {

// What a uniform grid gives element pairs: N_E^T G N_F, G = 1/sqrt(|x_i - x_j|^2 + gap^2)
// over each element's grid nodes (node_index (E, K) flat indices of a grid
// with ``columns`` columns per row, -1 padding; node_moments (E, K, 3)).
// Coincident nodes at zero gap contribute nothing.  Returns (P, 3, 3).
std::vector<double> rt0_grid_moments(const std::int64_t* node_index_a, const double* node_moments_a,
                                     std::int64_t count_a, const std::int64_t* node_index_b,
                                     const double* node_moments_b, std::int64_t count_b, int nodes_per_element,
                                     double origin_x, double origin_y, double pitch, std::int64_t columns,
                                     double gap, const std::int64_t* pair_a, const std::int64_t* pair_b,
                                     std::int64_t pairs, int threads);

}  // namespace pcbcore::sheet

namespace pcbcore::sheet {

// Element pairs whose centroids lie closer than reach = max(factor * max(diameter_a, diameter_b), floor),
// by a cell list (each pair once, ordered by a then b).  Returns the pairs as two equal-length vectors.
void rt0_near_pairs(const double* centroid_a, const double* diameter_a, std::int64_t count_a,
                    const double* centroid_b, const double* diameter_b, std::int64_t count_b, double factor,
                    double floor, int threads, std::vector<std::int64_t>& pair_a, std::vector<std::int64_t>& pair_b);

// Moments of element pairs by point quadrature on both elements, the kernel averaged over vertical
// offsets: W = sum_k w_k sum_i sum_j (w_i [1; x_i - c_a]) (w_j [1; x_j - c_b])^T / sqrt(|x_i - x_j|^2 + h_k^2).
// points (E, q, 2), weights (E, q), centroids (E, 2).  Returns (P, 3, 3) row-major.
std::vector<double> rt0_point_moments(const double* points_a, const double* weights_a, const double* centroid_a,
                                      std::int64_t count_a, int quadrature_a, const double* points_b,
                                      const double* weights_b, const double* centroid_b, std::int64_t count_b,
                                      int quadrature_b, const std::int64_t* pair_a, const std::int64_t* pair_b,
                                      std::int64_t pairs, const double* heights, const double* height_weights,
                                      int height_count, int threads);

}  // namespace pcbcore::sheet
