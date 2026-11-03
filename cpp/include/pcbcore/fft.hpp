// Real two-dimensional transforms through pocketfft (the implementation NumPy
// and SciPy use), with no threads of its own: pocketfft is compiled with
// POCKETFFT_NO_MULTITHREADING and the callers run independent transforms on
// OpenMP teams sized from the thread budget.
#pragma once

#include <complex>
#include <cstddef>

namespace pcbcore::fft {

// Forward transform of a ``rows x cols`` real block (row-major, leading
// dimension ``ld``) zero-padded to ``padded_rows x padded_cols``; the
// half-spectrum is ``padded_rows x (padded_cols / 2 + 1)``, as numpy.fft.rfft2
// with ``s=(padded_rows, padded_cols)`` returns it.
void rfft2_padded(const double* input, std::size_t rows, std::size_t cols, std::size_t ld,
                  std::size_t padded_rows, std::size_t padded_cols, std::complex<double>* spectrum);

// Inverse of a ``padded_rows x (padded_cols / 2 + 1)`` half-spectrum, scaled
// by ``1 / (padded_rows * padded_cols)`` like numpy.fft.irfft2; ``output`` is
// the full ``padded_rows x padded_cols`` block and ``spectrum`` is clobbered.
void irfft2(std::complex<double>* spectrum, std::size_t padded_rows, std::size_t padded_cols, double* output);

// rfft2_padded into caller-owned memory: ``row_buffer`` holds ``rows x
// padded_cols`` doubles.  Only the ``rows`` rows that carry data are
// transformed along the columns' axis (the padding rows' transforms are
// zero); the rest is rfft2_padded's arithmetic, step for step.
void rfft2_padded_rows(const double* input, std::size_t rows, std::size_t cols, std::size_t ld,
                       std::size_t padded_rows, std::size_t padded_cols, std::complex<double>* spectrum,
                       double* row_buffer);

// irfft2 of which only the leading ``rows x cols`` block is wanted, written to
// ``output`` (leading dimension ``ld``): the inverse along the rows' axis runs
// in place in ``spectrum`` (clobbered) and the one along the columns' axis
// only for the ``rows`` rows kept, through ``row_buffer`` (``rows x
// padded_cols`` doubles).  Each kept entry is irfft2's, step for step.
void irfft2_rows(std::complex<double>* spectrum, std::size_t padded_rows, std::size_t padded_cols, std::size_t rows,
                 std::size_t cols, double* output, std::size_t ld, double* row_buffer);

}  // namespace pcbcore::fft
