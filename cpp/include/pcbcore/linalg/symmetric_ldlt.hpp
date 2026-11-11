// LDL^T factorisation of a complex symmetric (not Hermitian) sparse matrix by
// MUMPS, sequential build, for the sheet PEEC near preconditioner.
//
// The near saddle-point matrix made quasi-definite, [[Z, -A], [-A^T, -delta I]],
// is complex symmetric.  SuperLU factors it as a general matrix: an
// unsymmetric ordering, partial pivoting, both triangles, one thread.  MUMPS
// factors the upper triangle in a symmetric (METIS) ordering with threshold
// pivoting confined to its fronts, which on the strip pair at 0.2 mm took the
// factor from 159M to 39M entries and the factorisation from 60 s to 4 s.
//
// The pattern is analysed once, at construction; ``factor`` takes new values
// in the same pattern (another frequency of the same mesh) without a new
// ordering; ``solve`` overwrites any number of right-hand-side columns.
//
// MUMPS's threads are its OpenMP team, set per call (ICNTL(16)).  The BLAS it
// is linked against is built single-threaded, so the team is the only
// parallelism and no call starts threads beyond the count it is given.
#pragma once

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace pcbcore::linalg {

enum class SymmetricOrdering { automatic, amd, metis, scotch };

class SymmetricLdlt {
public:
    // ``row_start`` (size + 1 entries) and ``column`` give the upper triangle
    // (column >= row) in compressed-row form, zero-based, without duplicates.
    // ``controls``: (k, value) pairs set as MUMPS ICNTL(k) before the analysis
    // and kept for every later step.
    SymmetricLdlt(std::int64_t size, const std::int64_t* row_start, const std::int64_t* column,
                  SymmetricOrdering ordering, int threads,
                  const std::vector<std::pair<int, int>>& controls = {});
    ~SymmetricLdlt();
    SymmetricLdlt(const SymmetricLdlt&) = delete;
    SymmetricLdlt& operator=(const SymmetricLdlt&) = delete;

    // ``values`` in the order of ``column``.  ``pivot_threshold`` is MUMPS's
    // CNTL(1): 0 takes the diagonal pivots as they come, which the
    // quasi-definite form allows in exact arithmetic but which left residuals
    // of 1e-2 on the strip pair; the default 0.01 gave 4e-10.
    void factor(const std::complex<double>* values, double pivot_threshold, int threads);

    // Overwrites the column-major ``size() x columns`` block with the solutions.
    void solve(std::complex<double>* right_hand_sides, std::int64_t columns, int threads);

    [[nodiscard]] std::int64_t size() const noexcept { return size_; }
    [[nodiscard]] std::int64_t nonzeros() const noexcept { return nonzeros_; }
    [[nodiscard]] bool factored() const noexcept { return factored_; }
    // Entries of the factors (MUMPS INFOG(29)), for memory accounting.
    [[nodiscard]] std::int64_t factor_entries() const noexcept { return factor_entries_; }
    [[nodiscard]] double analysis_seconds() const noexcept { return analysis_seconds_; }
    [[nodiscard]] double factor_seconds() const noexcept { return factor_seconds_; }
    // MUMPS INFOG(k), one-based k, for diagnostics: 7 the ordering used,
    // 13 the delayed pivots, 20 the estimated factor entries, 29 the actual.
    [[nodiscard]] std::int64_t info(int k) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::int64_t size_{0};
    std::int64_t nonzeros_{0};
    std::int64_t factor_entries_{0};
    double analysis_seconds_{0.0};
    double factor_seconds_{0.0};
    bool factored_{false};
};

// The MUMPS version string this build links, e.g. "5.9.1".
std::string mumps_version();

}  // namespace pcbcore::linalg
