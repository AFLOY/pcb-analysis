// Restarted GMRES with left preconditioning, step for step as SciPy's
// scipy.sparse.linalg.gmres (1.12 and later): modified Gram-Schmidt, Givens
// rotations from LAPACK's xLARTG, the inner tolerance on the preconditioned
// residual adapted after each restart (gh-8400), and the outer test on
// ||b - A x|| <= max(atol, rtol ||b||).  ``maxiter`` counts restart cycles
// and the iteration count reported is the number of inner steps, as
// callback_type="pr_norm" counts them.
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

namespace pcbcore::linalg {

struct GmresResult {
    std::int64_t inner_iterations{0};
    int info{0};  // 0 converged, otherwise maxiter
};

namespace gmres_detail {

using Complex = std::complex<double>;

inline double norm2(const Complex* const x, const std::int64_t n) noexcept {
    double sum = 0.0;
    for (std::int64_t i = 0; i < n; ++i) {
        sum += std::norm(x[i]);
    }
    return std::sqrt(sum);
}

// vdot: conjugate of the first argument.
inline Complex vdot(const Complex* const a, const Complex* const b, const std::int64_t n) noexcept {
    Complex sum(0.0, 0.0);
    for (std::int64_t i = 0; i < n; ++i) {
        sum += std::conj(a[i]) * b[i];
    }
    return sum;
}

// [c s; -conj(s) c] [f; g] = [r; 0] with c real.
inline void lartg(const Complex f, const Complex g, double& c, Complex& s, Complex& r) noexcept {
    if (g == Complex(0.0, 0.0)) {
        c = 1.0;
        s = Complex(0.0, 0.0);
        r = f;
        return;
    }
    if (f == Complex(0.0, 0.0)) {
        const double g1 = std::abs(g);
        c = 0.0;
        s = std::conj(g) / g1;
        r = Complex(g1, 0.0);
        return;
    }
    const double f1 = std::abs(f);
    const double g1 = std::abs(g);
    const double d = std::sqrt(f1 * f1 + g1 * g1);
    const Complex phase = f / f1;
    c = f1 / d;
    s = phase * std::conj(g) / d;
    r = phase * d;
}

}  // namespace gmres_detail

// ``matvec(in, out)`` and ``psolve(in, out)`` apply A and M ~ A^-1 to length-n
// vectors.  ``x`` holds the initial guess (zeros for SciPy's default) and the
// answer.
inline GmresResult gmres(const std::function<void(const std::complex<double>*, std::complex<double>*)>& matvec,
                         const std::function<void(const std::complex<double>*, std::complex<double>*)>& psolve,
                         const std::vector<std::complex<double>>& b, std::vector<std::complex<double>>& x,
                         const double rtol, const double atol_in, std::int64_t restart, std::int64_t maxiter) {
    using gmres_detail::Complex;
    using gmres_detail::lartg;
    using gmres_detail::norm2;
    using gmres_detail::vdot;
    const auto n = static_cast<std::int64_t>(b.size());
    GmresResult result;
    const double bnrm2 = norm2(b.data(), n);
    const double atol = std::max(atol_in, rtol * bnrm2);
    if (bnrm2 == 0.0) {
        std::fill(x.begin(), x.end(), Complex(0.0, 0.0));
        return result;
    }
    const double eps = std::numeric_limits<double>::epsilon();
    if (maxiter <= 0) {
        maxiter = n * 10;
    }
    if (restart <= 0) {
        restart = 20;
    }
    restart = std::min(restart, n);

    std::vector<Complex> scratch(static_cast<std::size_t>(n));
    psolve(b.data(), scratch.data());
    const double mb_nrm2 = norm2(scratch.data(), n);
    double ptol_max_factor = 1.0;
    double ptol = mb_nrm2 * std::min(ptol_max_factor, atol / bnrm2);
    double presid = 0.0;

    const auto stride = static_cast<std::size_t>(n);
    std::vector<Complex> v(static_cast<std::size_t>(restart + 1) * stride);
    std::vector<Complex> h(static_cast<std::size_t>(restart * (restart + 1)), Complex(0.0, 0.0));
    std::vector<double> givens_c(static_cast<std::size_t>(restart), 0.0);
    std::vector<Complex> givens_s(static_cast<std::size_t>(restart), Complex(0.0, 0.0));
    std::vector<Complex> r(stride);
    std::vector<Complex> av(stride);
    std::vector<Complex> w(stride);
    auto H = [&](const std::int64_t col, const std::int64_t row) -> Complex& {
        return h[static_cast<std::size_t>(col * (restart + 1) + row)];
    };
    auto V = [&](const std::int64_t k) { return v.data() + static_cast<std::size_t>(k) * stride; };

    bool any_nonzero = std::any_of(x.begin(), x.end(), [](const Complex& value) { return value != Complex(0.0); });
    double rnorm = 0.0;
    for (std::int64_t iteration = 0; iteration < maxiter; ++iteration) {
        if (iteration == 0) {
            if (any_nonzero) {
                matvec(x.data(), av.data());
                for (std::int64_t i = 0; i < n; ++i) {
                    r[static_cast<std::size_t>(i)] = b[static_cast<std::size_t>(i)] - av[static_cast<std::size_t>(i)];
                }
            } else {
                r = b;
            }
            if (norm2(r.data(), n) < atol) {
                return result;
            }
        }
        psolve(r.data(), V(0));
        double tmp = norm2(V(0), n);
        {
            const double scale = 1.0 / tmp;
            for (std::int64_t i = 0; i < n; ++i) {
                V(0)[i] *= scale;
            }
        }
        std::vector<Complex> S(static_cast<std::size_t>(restart + 1), Complex(0.0, 0.0));
        S[0] = tmp;
        bool breakdown = false;
        std::int64_t col = 0;
        for (col = 0; col < restart; ++col) {
            matvec(V(col), av.data());
            psolve(av.data(), w.data());
            const double h0 = norm2(w.data(), n);
            for (std::int64_t k = 0; k <= col; ++k) {
                const Complex dot = vdot(V(k), w.data(), n);
                H(col, k) = dot;
                const Complex* const vk = V(k);
                for (std::int64_t i = 0; i < n; ++i) {
                    w[static_cast<std::size_t>(i)] -= dot * vk[i];
                }
            }
            const double h1 = norm2(w.data(), n);
            H(col, col + 1) = h1;
            std::copy(w.begin(), w.end(), V(col + 1));
            if (h1 <= eps * h0) {
                H(col, col + 1) = 0.0;
                breakdown = true;
            } else {
                const double scale = 1.0 / h1;
                for (std::int64_t i = 0; i < n; ++i) {
                    V(col + 1)[i] *= scale;
                }
            }
            for (std::int64_t k = 0; k < col; ++k) {
                const double c = givens_c[static_cast<std::size_t>(k)];
                const Complex s = givens_s[static_cast<std::size_t>(k)];
                const Complex n0 = H(col, k);
                const Complex n1 = H(col, k + 1);
                H(col, k) = c * n0 + s * n1;
                H(col, k + 1) = -std::conj(s) * n0 + c * n1;
            }
            double c = 1.0;
            Complex s(0.0, 0.0);
            Complex mag(0.0, 0.0);
            lartg(H(col, col), H(col, col + 1), c, s, mag);
            givens_c[static_cast<std::size_t>(col)] = c;
            givens_s[static_cast<std::size_t>(col)] = s;
            H(col, col) = mag;
            H(col, col + 1) = 0.0;
            const Complex next = -std::conj(s) * S[static_cast<std::size_t>(col)];
            S[static_cast<std::size_t>(col)] = c * S[static_cast<std::size_t>(col)];
            S[static_cast<std::size_t>(col + 1)] = next;
            presid = std::abs(next);
            ++result.inner_iterations;
            if (presid <= ptol || breakdown) {
                break;
            }
        }
        if (col == restart) {
            col = restart - 1;  // the loop ran out without a break
        }
        if (H(col, col) == Complex(0.0, 0.0)) {
            S[static_cast<std::size_t>(col)] = 0.0;
        }
        std::vector<Complex> y(S.begin(), S.begin() + col + 1);
        for (std::int64_t k = col; k > 0; --k) {
            if (y[static_cast<std::size_t>(k)] != Complex(0.0, 0.0)) {
                y[static_cast<std::size_t>(k)] /= H(k, k);
                const Complex t = y[static_cast<std::size_t>(k)];
                for (std::int64_t j = 0; j < k; ++j) {
                    y[static_cast<std::size_t>(j)] -= t * H(k, j);
                }
            }
        }
        if (y[0] != Complex(0.0, 0.0)) {
            y[0] /= H(0, 0);
        }
        for (std::int64_t k = 0; k <= col; ++k) {
            const Complex yk = y[static_cast<std::size_t>(k)];
            const Complex* const vk = V(k);
            for (std::int64_t i = 0; i < n; ++i) {
                x[static_cast<std::size_t>(i)] += yk * vk[i];
            }
        }
        matvec(x.data(), av.data());
        for (std::int64_t i = 0; i < n; ++i) {
            r[static_cast<std::size_t>(i)] = b[static_cast<std::size_t>(i)] - av[static_cast<std::size_t>(i)];
        }
        rnorm = norm2(r.data(), n);
        if (rnorm <= atol) {
            break;
        }
        if (breakdown) {
            break;
        }
        if (presid <= ptol) {
            ptol_max_factor = std::max(eps, 0.25 * ptol_max_factor);
        } else {
            ptol_max_factor = std::min(1.0, 1.5 * ptol_max_factor);
        }
        ptol = presid * std::min(ptol_max_factor, atol / rnorm);
    }
    result.info = rnorm <= atol ? 0 : static_cast<int>(maxiter);
    return result;
}

}  // namespace pcbcore::linalg
