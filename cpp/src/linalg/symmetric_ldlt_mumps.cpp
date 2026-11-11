#include "pcbcore/linalg/symmetric_ldlt.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "pcbcore/errors.hpp"
#include "zmumps_c.h"

namespace pcbcore::linalg {

namespace {

// The MUMPS communicator value that the sequential build's MPI stub accepts.
constexpr MUMPS_INT kUseCommWorld = -987654;

// ICNTL(7): the sequential ordering.  Codes from the MUMPS user's guide.
MUMPS_INT ordering_code(const SymmetricOrdering ordering) {
    switch (ordering) {
        case SymmetricOrdering::automatic:
            return 7;
        case SymmetricOrdering::amd:
            return 0;
        case SymmetricOrdering::scotch:
            return 3;
        case SymmetricOrdering::metis:
            return 5;
    }
    return 7;
}

// A workspace estimate MUMPS found too small: INFOG(1) of -8, -9, -14, -15,
// -17 or -20 asks for a larger ICNTL(14) relaxation.
bool workspace_too_small(const MUMPS_INT code) {
    return code == -8 || code == -9 || code == -14 || code == -15 || code == -17 || code == -20;
}

double seconds_since(const std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

static_assert(sizeof(std::complex<double>) == sizeof(ZMUMPS_COMPLEX), "complex layouts differ");

}  // namespace

struct SymmetricLdlt::Impl {
    ZMUMPS_STRUC_C id{};
    std::vector<MUMPS_INT> row;     // one-based
    std::vector<MUMPS_INT> column;  // one-based
    std::vector<std::complex<double>> values;
    bool initialised{false};

    void call(const MUMPS_INT job, const char* step) {
        id.job = job;
        zmumps_c(&id);
        if (id.infog[0] < 0) {
            throw Singular(std::string("MUMPS ") + step + " failed: INFOG(1)=" + std::to_string(id.infog[0]) +
                           ", INFOG(2)=" + std::to_string(id.infog[1]));
        }
    }

    ~Impl() {
        if (initialised) {
            id.job = -2;
            zmumps_c(&id);
        }
    }
};

SymmetricLdlt::SymmetricLdlt(const std::int64_t size, const std::int64_t* const row_start,
                             const std::int64_t* const column, const SymmetricOrdering ordering, const int threads,
                             const std::vector<std::pair<int, int>>& controls)
    : impl_(std::make_unique<Impl>()), size_(size) {
    if (size < 1 || size > std::numeric_limits<MUMPS_INT>::max()) {
        throw InvalidInput("SymmetricLdlt needs a size between 1 and the MUMPS integer range");
    }
    if (row_start[0] != 0) {
        throw InvalidInput("row_start must begin at 0");
    }
    nonzeros_ = row_start[size];
    Impl& impl = *impl_;
    impl.row.resize(static_cast<std::size_t>(nonzeros_));
    impl.column.resize(static_cast<std::size_t>(nonzeros_));
    impl.values.assign(static_cast<std::size_t>(nonzeros_), std::complex<double>(0.0, 0.0));
    for (std::int64_t r = 0; r < size; ++r) {
        const std::int64_t first = row_start[r];
        const std::int64_t last = row_start[r + 1];
        if (last < first) {
            throw InvalidInput("row_start must not decrease");
        }
        for (std::int64_t k = first; k < last; ++k) {
            const std::int64_t c = column[k];
            if (c < r || c >= size) {
                throw InvalidInput("column indices must lie in the upper triangle of the matrix");
            }
            impl.row[static_cast<std::size_t>(k)] = static_cast<MUMPS_INT>(r + 1);
            impl.column[static_cast<std::size_t>(k)] = static_cast<MUMPS_INT>(c + 1);
        }
    }

    ZMUMPS_STRUC_C& id = impl.id;
    id.comm_fortran = kUseCommWorld;
    id.par = 1;  // the host works
    id.sym = 2;  // general symmetric: complex symmetric, LDL^T with pivoting
    id.job = -1;
    zmumps_c(&id);
    if (id.infog[0] < 0) {
        throw Singular("MUMPS initialisation failed: INFOG(1)=" + std::to_string(id.infog[0]));
    }
    impl.initialised = true;
    // No output streams: MUMPS writes nothing unless it is told where.
    id.icntl[0] = -1;
    id.icntl[1] = -1;
    id.icntl[2] = -1;
    id.icntl[3] = 0;
    id.icntl[6] = ordering_code(ordering);
    id.icntl[27] = 1;  // ICNTL(28): sequential analysis
    id.icntl[15] = std::max(1, threads);
    for (const auto& [k, value] : controls) {
        if (k < 1 || k > 60) {
            throw InvalidInput("ICNTL index must be 1..60");
        }
        id.icntl[k - 1] = static_cast<MUMPS_INT>(value);
    }

    id.n = static_cast<MUMPS_INT>(size);
    id.nnz = static_cast<MUMPS_INT8>(nonzeros_);
    id.irn = impl.row.data();
    id.jcn = impl.column.data();
    id.a = reinterpret_cast<ZMUMPS_COMPLEX*>(impl.values.data());

    const auto start = std::chrono::steady_clock::now();
    impl.call(1, "analysis");
    analysis_seconds_ = seconds_since(start);
}

SymmetricLdlt::~SymmetricLdlt() = default;

void SymmetricLdlt::factor(const std::complex<double>* const values, const double pivot_threshold,
                           const int threads) {
    Impl& impl = *impl_;
    std::copy(values, values + nonzeros_, impl.values.begin());
    ZMUMPS_STRUC_C& id = impl.id;
    id.icntl[15] = std::max(1, threads);
    id.cntl[0] = pivot_threshold;
    factored_ = false;
    const auto start = std::chrono::steady_clock::now();
    for (int attempt = 0;; ++attempt) {
        id.job = 2;
        zmumps_c(&id);
        if (id.infog[0] >= 0) {
            break;
        }
        if (attempt < 4 && workspace_too_small(id.infog[0])) {
            id.icntl[13] = std::max<MUMPS_INT>(id.icntl[13], 20) * 2;  // ICNTL(14): workspace relaxation, percent
            continue;
        }
        throw Singular("MUMPS factorisation failed: INFOG(1)=" + std::to_string(id.infog[0]) +
                       ", INFOG(2)=" + std::to_string(id.infog[1]));
    }
    factor_seconds_ = seconds_since(start);
    // INFOG(29): entries in the factors; negative means millions.
    const MUMPS_INT entries = id.infog[28];
    factor_entries_ = entries >= 0 ? static_cast<std::int64_t>(entries) : -static_cast<std::int64_t>(entries) * 1000000;
    factored_ = true;
}

void SymmetricLdlt::solve(std::complex<double>* const right_hand_sides, const std::int64_t columns,
                          const int threads) {
    if (!factored_) {
        throw InvalidInput("SymmetricLdlt::solve before factor");
    }
    if (columns < 1) {
        return;
    }
    if (columns > std::numeric_limits<MUMPS_INT>::max()) {
        throw InvalidInput("too many right-hand sides");
    }
    Impl& impl = *impl_;
    ZMUMPS_STRUC_C& id = impl.id;
    id.icntl[15] = std::max(1, threads);
    id.icntl[19] = 0;  // ICNTL(20): dense right-hand sides
    id.icntl[20] = 0;  // ICNTL(21): solution centralised, written over rhs
    id.rhs = reinterpret_cast<ZMUMPS_COMPLEX*>(right_hand_sides);
    id.nrhs = static_cast<MUMPS_INT>(columns);
    id.lrhs = static_cast<MUMPS_INT>(size_);
    impl.call(3, "solve");
    id.rhs = nullptr;
}

std::int64_t SymmetricLdlt::info(const int k) const {
    if (k < 1 || k > 80) {
        throw InvalidInput("INFOG index must be 1..80");
    }
    return static_cast<std::int64_t>(impl_->id.infog[k - 1]);
}

std::string mumps_version() {
    return MUMPS_VERSION;
}

}  // namespace pcbcore::linalg
