#include "pcbcore/sheet/near_field.hpp"

#include <algorithm>
#include <cstddef>

#include "pcbcore/errors.hpp"

namespace pcbcore::sheet {

namespace {

using std::size_t;

size_t pair_index(const std::int64_t first, const std::int64_t second, const std::int64_t count) {
    const std::int64_t a = std::min(first, second);
    const std::int64_t b = std::max(first, second);
    return static_cast<size_t>(a * count - a * (a - 1) / 2 + (b - a));
}

std::int64_t wrap(const std::int64_t value, const std::int64_t period) {
    const std::int64_t r = value % period;
    return r < 0 ? r + period : r;
}

}  // namespace

NearFieldEntries uniform_near_field(const std::int64_t layers, const std::int64_t rows, const std::int64_t cols,
                                    const std::int64_t levels, const int radius, const double* const tables_x,
                                    const double* const tables_y, const double* const tables_z,
                                    const std::int64_t* const branch_x, const std::int64_t count_x,
                                    const std::int64_t* const branch_y, const std::int64_t count_y,
                                    const std::int64_t* const vias, const std::int64_t count_vias) {
    if (radius < 0) {
        throw InvalidInput("radius_cells must be non-negative");
    }
    const std::int64_t padded_rows = 2 * rows;
    const std::int64_t padded_cols = 2 * cols;
    const size_t table_size = static_cast<size_t>(padded_rows * padded_cols);
    const std::int64_t plane = rows * cols;
    NearFieldEntries out;
    const std::int64_t span = 2 * static_cast<std::int64_t>(radius) + 1;
    const size_t estimate = static_cast<size_t>((count_x + count_y) * layers + count_vias * levels) *
                            static_cast<size_t>(span * span);
    out.row.reserve(estimate);
    out.col.reserve(estimate);
    out.value.reserve(estimate);

    // One family of like-directed branches over ``count_of`` stacked planes.
    auto family = [&](const std::int64_t* const cells, const std::int64_t count, const std::int64_t planes,
                      const double* const tables, const std::int64_t offset, const bool first_occurrence_order) {
        // Position of the branch on each (plane, row, col); a repeated cell keeps
        // the last position, as a dict assignment would.
        std::vector<std::int64_t> lookup(static_cast<size_t>(planes * plane), -1);
        for (std::int64_t k = 0; k < count; ++k) {
            const std::int64_t p = cells[3 * k];
            const std::int64_t r = cells[3 * k + 1];
            const std::int64_t c = cells[3 * k + 2];
            if (p < 0 || p >= planes || r < 0 || r >= rows || c < 0 || c >= cols) {
                throw InvalidInput("branch cell outside the mesh");
            }
            lookup[static_cast<size_t>(p * plane + r * cols + c)] = k + offset;
        }
        std::vector<bool> visited(first_occurrence_order ? static_cast<size_t>(planes * plane) : 0U, false);
        for (std::int64_t k = 0; k < count; ++k) {
            const std::int64_t p = cells[3 * k];
            const std::int64_t r = cells[3 * k + 1];
            const std::int64_t c = cells[3 * k + 2];
            const size_t key = static_cast<size_t>(p * plane + r * cols + c);
            std::int64_t source = k + offset;
            if (first_occurrence_order) {
                // Vias are walked as the keys of a dict: each cell once, in the
                // order it first appears, with the position it was last given.
                if (visited[key]) {
                    continue;
                }
                visited[key] = true;
                source = lookup[key];
            }
            for (std::int64_t other = 0; other < planes; ++other) {
                const double* const table = tables + pair_index(p, other, planes) * table_size;
                for (std::int64_t dr = -radius; dr <= radius; ++dr) {
                    const std::int64_t tr = r + dr;
                    for (std::int64_t dc = -radius; dc <= radius; ++dc) {
                        const std::int64_t tc = c + dc;
                        if (tr < 0 || tr >= rows || tc < 0 || tc >= cols) {
                            continue;
                        }
                        const std::int64_t target = lookup[static_cast<size_t>(other * plane + tr * cols + tc)];
                        if (target < 0) {
                            continue;
                        }
                        out.row.push_back(source);
                        out.col.push_back(target);
                        out.value.push_back(
                            table[static_cast<size_t>(wrap(dr, padded_rows) * padded_cols + wrap(dc, padded_cols))]);
                    }
                }
            }
        }
    };
    family(branch_x, count_x, layers, tables_x, 0, false);
    family(branch_y, count_y, layers, tables_y, count_x, false);
    if (count_vias > 0 && levels > 0) {
        family(vias, count_vias, levels, tables_z, count_x + count_y, true);
    }
    return out;
}

}  // namespace pcbcore::sheet
