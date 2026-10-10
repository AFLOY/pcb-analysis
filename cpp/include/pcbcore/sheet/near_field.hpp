// The sparse near-field partial inductance of a uniform sheet mesh: the
// coupling of every branch to the like-directed branches within ``radius``
// cells, read off the convolution tables.  This is the ``L`` the solver's
// near-field preconditioner adds to the resistances.
#pragma once

#include <cstdint>
#include <vector>

namespace pcbcore::sheet {

struct NearFieldEntries {
    std::vector<std::int64_t> row;
    std::vector<std::int64_t> col;
    std::vector<double> value;
};

// ``branch_x``/``branch_y`` are (count, 3) arrays of (layer, row, col) in
// mesh branch order; ``vias`` is (count, 3) of (level, row, col) in via
// order.  Tables are padded ``(2 rows) x (2 cols)``, one per layer pair
// ``first <= second`` (upper triangle, row-major) and per level pair.  The
// entries come out in the order the NumPy implementation lists them, so the
// sparse matrix sums any duplicates the same way.
[[nodiscard]] NearFieldEntries uniform_near_field(std::int64_t layers, std::int64_t rows, std::int64_t cols,
                                                  std::int64_t levels, int radius, const double* tables_x,
                                                  const double* tables_y, const double* tables_z,
                                                  const std::int64_t* branch_x, std::int64_t count_x,
                                                  const std::int64_t* branch_y, std::int64_t count_y,
                                                  const std::int64_t* vias, std::int64_t count_vias);

}  // namespace pcbcore::sheet
