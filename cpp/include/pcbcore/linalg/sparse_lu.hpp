// Sparse LU factorisation of a square real or complex matrix by SuperLU.
//
// The same driver sequence as SciPy's splu: a fill-reducing column ordering
// (COLAMD unless asked otherwise), sp_preorder, d/zgstrf with threshold
// partial pivoting, then d/zgstrs for any number of right-hand sides.
// SuperLU runs on one thread with its bundled BLAS, so a factorisation is
// deterministic and never starts threads of its own.
#pragma once

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

namespace pcbcore::linalg {

// Compressed sparse column storage with sorted, duplicate-free row indices.
template <typename Scalar>
struct CscMatrixT {
    std::int64_t rows{0};
    std::int64_t cols{0};
    std::vector<std::int64_t> column_start;  // cols + 1 entries
    std::vector<std::int64_t> row_index;
    std::vector<Scalar> value;

    [[nodiscard]] std::int64_t nonzeros() const noexcept {
        return static_cast<std::int64_t>(value.size());
    }

    // y = A x, accumulated column by column (a fixed order).
    void multiply(const Scalar* x, Scalar* y) const noexcept;
};

using CscMatrix = CscMatrixT<double>;
using ComplexCscMatrix = CscMatrixT<std::complex<double>>;

enum class ColumnOrdering { natural, mmd_ata, mmd_at_plus_a, colamd };

template <typename Scalar>
class SparseLUT {
public:
    // Factors ``matrix``.  A zero pivot does not throw: ``singular()`` reports
    // it and ``solve`` then fills the right-hand sides with NaN, which is what
    // a caller that names the failing case wants to see.
    explicit SparseLUT(const CscMatrixT<Scalar>& matrix, ColumnOrdering ordering = ColumnOrdering::colamd,
                       double diagonal_pivot_threshold = 1.0);
    ~SparseLUT();
    SparseLUT(const SparseLUT&) = delete;
    SparseLUT& operator=(const SparseLUT&) = delete;

    [[nodiscard]] bool singular() const noexcept { return singular_; }
    [[nodiscard]] std::int64_t size() const noexcept { return size_; }
    // Nonzeros of L + U, for memory accounting.
    [[nodiscard]] std::int64_t factor_nonzeros() const noexcept { return factor_nonzeros_; }

    // Overwrites the column-major ``size() x columns`` block with the solutions.
    void solve(Scalar* right_hand_sides, std::int64_t columns) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::int64_t size_{0};
    std::int64_t factor_nonzeros_{0};
    bool singular_{false};
};

using SparseLU = SparseLUT<double>;
using ComplexSparseLU = SparseLUT<std::complex<double>>;

}  // namespace pcbcore::linalg
