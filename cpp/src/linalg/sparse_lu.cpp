#include "pcbcore/linalg/sparse_lu.hpp"

#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

#include "pcbcore/errors.hpp"

#include "slu_ddefs.h"
#include "slu_zdefs.h"

namespace pcbcore::linalg {

namespace {

int ordering_spec(const ColumnOrdering ordering) {
    switch (ordering) {
        case ColumnOrdering::natural:
            return 0;
        case ColumnOrdering::mmd_ata:
            return 1;
        case ColumnOrdering::mmd_at_plus_a:
            return 2;
        case ColumnOrdering::colamd:
            return 3;
    }
    return 3;
}

colperm_t ordering_option(const ColumnOrdering ordering) {
    switch (ordering) {
        case ColumnOrdering::natural:
            return NATURAL;
        case ColumnOrdering::mmd_ata:
            return MMD_ATA;
        case ColumnOrdering::mmd_at_plus_a:
            return MMD_AT_PLUS_A;
        case ColumnOrdering::colamd:
            return COLAMD;
    }
    return COLAMD;
}

// The SuperLU entry points of one scalar type.
template <typename Scalar>
struct Driver;

template <>
struct Driver<double> {
    using Native = double;
    static void create_matrix(SuperMatrix* a, int n, int_t nnz, double* values, int_t* rows, int_t* starts) {
        dCreate_CompCol_Matrix(a, n, n, nnz, values, rows, starts, SLU_NC, SLU_D, SLU_GE);
    }
    static void create_dense(SuperMatrix* b, int n, int columns, double* values) {
        dCreate_Dense_Matrix(b, n, columns, values, n, SLU_DN, SLU_D, SLU_GE);
    }
    static void factor(superlu_options_t* options, SuperMatrix* permuted, int relax, int panel, int* etree,
                       int* perm_c, int* perm_r, SuperMatrix* lower, SuperMatrix* upper, GlobalLU_t* glu,
                       SuperLUStat_t* stat, int_t* info) {
        dgstrf(options, permuted, relax, panel, etree, nullptr, 0, perm_c, perm_r, lower, upper, glu, stat, info);
    }
    static void solve(SuperMatrix* lower, SuperMatrix* upper, int* perm_c, int* perm_r, SuperMatrix* b,
                      SuperLUStat_t* stat, int* info) {
        dgstrs(NOTRANS, lower, upper, perm_c, perm_r, b, stat, info);
    }
    static double quiet_nan() { return std::numeric_limits<double>::quiet_NaN(); }
};

template <>
struct Driver<std::complex<double>> {
    using Native = doublecomplex;
    static void create_matrix(SuperMatrix* a, int n, int_t nnz, std::complex<double>* values, int_t* rows,
                              int_t* starts) {
        zCreate_CompCol_Matrix(a, n, n, nnz, reinterpret_cast<doublecomplex*>(values), rows, starts, SLU_NC, SLU_Z,
                               SLU_GE);
    }
    static void create_dense(SuperMatrix* b, int n, int columns, std::complex<double>* values) {
        zCreate_Dense_Matrix(b, n, columns, reinterpret_cast<doublecomplex*>(values), n, SLU_DN, SLU_Z, SLU_GE);
    }
    static void factor(superlu_options_t* options, SuperMatrix* permuted, int relax, int panel, int* etree,
                       int* perm_c, int* perm_r, SuperMatrix* lower, SuperMatrix* upper, GlobalLU_t* glu,
                       SuperLUStat_t* stat, int_t* info) {
        zgstrf(options, permuted, relax, panel, etree, nullptr, 0, perm_c, perm_r, lower, upper, glu, stat, info);
    }
    static void solve(SuperMatrix* lower, SuperMatrix* upper, int* perm_c, int* perm_r, SuperMatrix* b,
                      SuperLUStat_t* stat, int* info) {
        zgstrs(NOTRANS, lower, upper, perm_c, perm_r, b, stat, info);
    }
    static std::complex<double> quiet_nan() {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan, nan};
    }
};

static_assert(sizeof(std::complex<double>) == sizeof(doublecomplex), "complex layouts differ");

}  // namespace

template <typename Scalar>
void CscMatrixT<Scalar>::multiply(const Scalar* const x, Scalar* const y) const noexcept {
    for (std::int64_t row = 0; row < rows; ++row) {
        y[row] = Scalar(0);
    }
    for (std::int64_t col = 0; col < cols; ++col) {
        const Scalar xj = x[col];
        for (std::int64_t k = column_start[static_cast<std::size_t>(col)];
             k < column_start[static_cast<std::size_t>(col) + 1U]; ++k) {
            y[row_index[static_cast<std::size_t>(k)]] += value[static_cast<std::size_t>(k)] * xj;
        }
    }
}

template <typename Scalar>
struct SparseLUT<Scalar>::Impl {
    SuperMatrix lower{};
    SuperMatrix upper{};
    std::vector<int> column_permutation;
    std::vector<int> row_permutation;
    bool factored{false};

    ~Impl() {
        if (factored) {
            Destroy_SuperNode_Matrix(&lower);
            Destroy_CompCol_Matrix(&upper);
        }
    }
};

template <typename Scalar>
SparseLUT<Scalar>::SparseLUT(const CscMatrixT<Scalar>& matrix, const ColumnOrdering ordering,
                             const double diagonal_pivot_threshold)
    : impl_(std::make_unique<Impl>()), size_(matrix.rows) {
    if (matrix.rows != matrix.cols) {
        throw InvalidInput("SparseLU needs a square matrix");
    }
    if (matrix.column_start.size() != static_cast<std::size_t>(matrix.cols) + 1U ||
        matrix.row_index.size() != matrix.value.size()) {
        throw InvalidInput("inconsistent compressed-column arrays");
    }
    constexpr auto limit = static_cast<std::int64_t>(std::numeric_limits<int>::max());
    if (matrix.rows > limit || matrix.nonzeros() > limit) {
        throw InvalidInput("SparseLU is built with 32-bit indices; the matrix is too large");
    }
    if (size_ == 0) {
        return;
    }
    const int n = static_cast<int>(size_);
    const auto nnz = static_cast<int_t>(matrix.nonzeros());

    // SuperLU keeps pointers into these during the factorisation only.
    std::vector<int_t> column_start(matrix.column_start.begin(), matrix.column_start.end());
    std::vector<int_t> row_index(matrix.row_index.begin(), matrix.row_index.end());
    std::vector<Scalar> value(matrix.value);
    SuperMatrix a{};
    Driver<Scalar>::create_matrix(&a, n, nnz, value.data(), row_index.data(), column_start.data());

    superlu_options_t options{};
    set_default_options(&options);
    options.ColPerm = ordering_option(ordering);
    options.DiagPivotThresh = diagonal_pivot_threshold;

    impl_->column_permutation.assign(static_cast<std::size_t>(n), 0);
    impl_->row_permutation.assign(static_cast<std::size_t>(n), 0);
    get_perm_c(ordering_spec(ordering), &a, impl_->column_permutation.data());

    std::vector<int> elimination_tree(static_cast<std::size_t>(n), 0);
    SuperMatrix permuted{};
    sp_preorder(&options, &a, impl_->column_permutation.data(), elimination_tree.data(), &permuted);

    const int panel_size = sp_ienv(1);
    const int relax = sp_ienv(2);
    SuperLUStat_t stat{};
    StatInit(&stat);
    GlobalLU_t glu{};
    int_t info = 0;
    Driver<Scalar>::factor(&options, &permuted, relax, panel_size, elimination_tree.data(),
                           impl_->column_permutation.data(), impl_->row_permutation.data(), &impl_->lower,
                           &impl_->upper, &glu, &stat, &info);
    StatFree(&stat);
    Destroy_CompCol_Permuted(&permuted);
    Destroy_SuperMatrix_Store(&a);

    if (info < 0) {
        throw std::logic_error("gstrf rejected argument " + std::to_string(-info));
    }
    if (info > static_cast<int_t>(n)) {
        throw std::bad_alloc();
    }
    impl_->factored = true;
    singular_ = info > 0;
    factor_nonzeros_ = static_cast<std::int64_t>(static_cast<SCformat*>(impl_->lower.Store)->nnz) +
                       static_cast<std::int64_t>(static_cast<NCformat*>(impl_->upper.Store)->nnz);
}

template <typename Scalar>
SparseLUT<Scalar>::~SparseLUT() = default;

template <typename Scalar>
void SparseLUT<Scalar>::solve(Scalar* const right_hand_sides, const std::int64_t columns) const {
    if (size_ == 0 || columns == 0) {
        return;
    }
    if (singular_) {
        const auto count = static_cast<std::size_t>(size_) * static_cast<std::size_t>(columns);
        for (std::size_t k = 0; k < count; ++k) {
            right_hand_sides[k] = Driver<Scalar>::quiet_nan();
        }
        return;
    }
    if (columns > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        throw InvalidInput("too many right-hand sides");
    }
    const int n = static_cast<int>(size_);
    SuperMatrix b{};
    Driver<Scalar>::create_dense(&b, n, static_cast<int>(columns), right_hand_sides);
    SuperLUStat_t stat{};
    StatInit(&stat);
    int info = 0;
    Driver<Scalar>::solve(&impl_->lower, &impl_->upper, impl_->column_permutation.data(),
                          impl_->row_permutation.data(), &b, &stat, &info);
    StatFree(&stat);
    Destroy_SuperMatrix_Store(&b);
    if (info != 0) {
        throw std::logic_error("gstrs rejected argument " + std::to_string(-info));
    }
}

template struct CscMatrixT<double>;
template struct CscMatrixT<std::complex<double>>;
template class SparseLUT<double>;
template class SparseLUT<std::complex<double>>;

}  // namespace pcbcore::linalg
