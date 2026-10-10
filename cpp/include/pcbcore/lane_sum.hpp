// Sums whose bits depend on nothing but the terms.
//
// A reduction left to the vectoriser (``#pragma omp simd reduction``, Eigen's
// ``dot`` or ``norm``) splits the terms by the address alignment of the first
// one, which changes from one allocation to the next; one split per thread
// changes with the team size.  Either way the same solve can round
// differently from run to run.  These sums give every term a lane by its
// index counted from the start, keep eight lanes in order, and combine them
// in a fixed tree; the compiler still vectorises the eight lanes, but cannot
// change which terms meet.
#pragma once

#include <cstddef>

namespace pcbcore {

template <typename Term>
[[nodiscard]] inline double lane_sum(const std::ptrdiff_t count, Term&& term) noexcept {
    constexpr int kLanes = 8;
    double acc[kLanes] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::ptrdiff_t i = 0;
    for (; i + kLanes <= count; i += kLanes) {
        for (int k = 0; k < kLanes; ++k) {
            acc[k] += term(i + k);
        }
    }
    for (int k = 0; i < count; ++i, ++k) {
        acc[k] += term(i);
    }
    return ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
}

// A sum over ``parts`` partial sums, in index order.
template <typename Part>
[[nodiscard]] inline double ordered_sum(const std::ptrdiff_t parts, Part&& part) noexcept {
    double acc = 0.0;
    for (std::ptrdiff_t p = 0; p < parts; ++p) {
        acc += part(p);
    }
    return acc;
}

}  // namespace pcbcore
