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

}  // namespace pcbcore::fft
