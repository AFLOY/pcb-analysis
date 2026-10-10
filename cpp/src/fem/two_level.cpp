#include "pcbcore/fem/two_level.hpp"

#include <algorithm>

namespace pcbcore::fem {

int choose_block_size(const int layers, const int node_rows, const int node_cols, const std::int64_t max_coarse_size,
                      const int minimum) {
    int block = std::max(1, minimum);
    while (true) {
        const std::int64_t coarse = static_cast<std::int64_t>(layers) * ((node_rows + block - 1) / block) *
                                    ((node_cols + block - 1) / block);
        if (coarse <= max_coarse_size || block >= std::max(node_rows, node_cols)) {
            return block;
        }
        ++block;
    }
}

}  // namespace pcbcore::fem
