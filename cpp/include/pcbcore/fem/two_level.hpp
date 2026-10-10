// The patch width of the two-level (Jacobi plus patch-constant coarse)
// preconditioner shared by the layered DC and the thermal hex systems
// (electrical/matrix_free_mpir_fem/two_level.py).
#pragma once

#include <cstdint>

namespace pcbcore::fem {

// The coarse space never grows past this many patches unless the patch is
// already as wide as the mesh (two_level.DEFAULT_MAX_COARSE_SIZE).
inline constexpr std::int64_t kDefaultMaxCoarseSize = 2048;

// The smallest in-plane patch width, at least ``minimum``, whose coarse grid
// over a ``(layers, node_rows, node_cols)`` node grid holds at most
// ``max_coarse_size`` patches (two_level.choose_block_size).
[[nodiscard]] int choose_block_size(int layers, int node_rows, int node_cols,
                                    std::int64_t max_coarse_size = kDefaultMaxCoarseSize, int minimum = 4);

}  // namespace pcbcore::fem
