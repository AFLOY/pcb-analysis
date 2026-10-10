// The topology of a layered sheet mesh on a tensor grid: which cells are
// copper (nodes), which pairs of neighbouring copper cells join (in-plane
// branches), which vertical connections land on copper at both ends, and what
// every branch's resistance is.  Everything comes out in the order the NumPy
// mesh lists it: nodes by (layer, row, col), x branches by (layer, row, col),
// then y branches, then the vias in the order given.
#pragma once

#include <complex>
#include <cstdint>
#include <vector>

namespace pcbcore::sheet {

struct MeshTopology {
    std::int64_t layers{0};
    std::int64_t rows{0};
    std::int64_t cols{0};
    std::vector<std::int64_t> node_of;   // layers x rows x cols, -1 where there is no copper
    std::int64_t node_count{0};
    std::vector<std::int64_t> branch_x;  // (count, 3): layer, row, col of the left cell
    std::vector<std::int64_t> branch_y;  // (count, 3): layer, row, col of the lower-row cell
    std::vector<std::int64_t> via_kept;  // indices of the vias whose both ends are copper
    std::vector<std::int64_t> left;      // every branch: the node it leaves
    std::vector<std::int64_t> right;     // every branch: the node it enters
};

// ``vias`` is (count, 4): lower layer, upper layer, row, col.
[[nodiscard]] MeshTopology build_topology(std::int64_t layers, std::int64_t rows, std::int64_t cols,
                                          const std::uint8_t* occupancy, const std::int64_t* vias,
                                          std::int64_t via_count);

// Sheet resistance times length over width for in-plane branches, the given
// resistance for each kept via.  ``pitch_x`` has ``cols`` entries, ``pitch_y``
// ``rows``.
[[nodiscard]] std::vector<double> branch_resistance(const MeshTopology& mesh, const double* sheet_resistance,
                                                    const double* pitch_x, const double* pitch_y,
                                                    const double* via_resistance);

// Each terminal's current spread evenly over its cells that are copper.
// ``cell_start`` delimits each terminal's (row, col) pairs in ``cells``;
// ``usable`` receives how many cells of each terminal landed on copper.
[[nodiscard]] std::vector<std::complex<double>> source_vector(const MeshTopology& mesh, std::int64_t terminals,
                                                              const std::int64_t* layer,
                                                              const std::int64_t* cell_start,
                                                              const std::int64_t* cells,
                                                              const std::complex<double>* current,
                                                              std::vector<std::int64_t>& usable);

// Each node's two in-plane current-density phasors (A/mm^2): the mean of the
// branches on either side along each axis (a missing branch carries none),
// over the cell's cross-section.  ``thickness_m`` has one entry per layer.
void cell_density(const MeshTopology& mesh, const std::complex<double>* branch_current, const double* pitch_x,
                  const double* pitch_y, const double* thickness_m, std::complex<double>* along_x,
                  std::complex<double>* along_y);

}  // namespace pcbcore::sheet
