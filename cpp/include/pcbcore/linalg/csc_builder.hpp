// Compressed sparse column assembly from (row, col, value) triplets in a
// fixed order: entries are bucketed by column stably, sorted by row stably
// inside each column, and duplicates are summed in the order they were
// listed.  The same triplets therefore always give the same bits.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "pcbcore/linalg/sparse_lu.hpp"

namespace pcbcore::linalg {

template <typename Scalar>
struct Triplets {
    std::vector<std::int64_t> row;
    std::vector<std::int64_t> col;
    std::vector<Scalar> value;

    void add(const std::int64_t r, const std::int64_t c, const Scalar v) {
        row.push_back(r);
        col.push_back(c);
        value.push_back(v);
    }
    void reserve(const std::size_t count) {
        row.reserve(count);
        col.reserve(count);
        value.reserve(count);
    }
};

template <typename Scalar>
CscMatrixT<Scalar> csc_from_triplets(const std::int64_t rows, const std::int64_t cols, const Triplets<Scalar>& t) {
    using std::size_t;
    CscMatrixT<Scalar> matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    std::vector<std::int64_t> start(static_cast<size_t>(cols) + 1U, 0);
    for (const std::int64_t c : t.col) {
        ++start[static_cast<size_t>(c) + 1U];
    }
    std::partial_sum(start.begin(), start.end(), start.begin());
    std::vector<std::int64_t> order(t.value.size());
    {
        std::vector<std::int64_t> next(start.begin(), start.end() - 1);
        for (size_t e = 0; e < t.value.size(); ++e) {
            order[static_cast<size_t>(next[static_cast<size_t>(t.col[e])]++)] = static_cast<std::int64_t>(e);
        }
    }
    matrix.column_start.assign(static_cast<size_t>(cols) + 1U, 0);
    matrix.row_index.reserve(t.value.size());
    matrix.value.reserve(t.value.size());
    for (std::int64_t c = 0; c < cols; ++c) {
        const auto first = order.begin() + start[static_cast<size_t>(c)];
        const auto last = order.begin() + start[static_cast<size_t>(c) + 1U];
        std::stable_sort(first, last, [&t](const std::int64_t a, const std::int64_t b) {
            return t.row[static_cast<size_t>(a)] < t.row[static_cast<size_t>(b)];
        });
        const auto column_begin = static_cast<std::int64_t>(matrix.value.size());
        for (auto it = first; it != last; ++it) {
            const std::int64_t r = t.row[static_cast<size_t>(*it)];
            const Scalar v = t.value[static_cast<size_t>(*it)];
            if (static_cast<std::int64_t>(matrix.value.size()) > column_begin && matrix.row_index.back() == r) {
                matrix.value.back() += v;
            } else {
                matrix.row_index.push_back(r);
                matrix.value.push_back(v);
            }
        }
        matrix.column_start[static_cast<size_t>(c) + 1U] = static_cast<std::int64_t>(matrix.value.size());
    }
    return matrix;
}

}  // namespace pcbcore::linalg
