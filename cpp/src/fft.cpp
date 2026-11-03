#include "pcbcore/fft.hpp"

#include <algorithm>
#include <vector>

#include "pocketfft_hdronly.h"

namespace pcbcore::fft {

namespace {

using pocketfft::shape_t;
using pocketfft::stride_t;

}  // namespace

void rfft2_padded(const double* const input, const std::size_t rows, const std::size_t cols, const std::size_t ld,
                  const std::size_t padded_rows, const std::size_t padded_cols, std::complex<double>* const spectrum) {
    std::vector<double> padded(padded_rows * padded_cols, 0.0);
    const std::size_t copy_rows = std::min(rows, padded_rows);
    const std::size_t copy_cols = std::min(cols, padded_cols);
    for (std::size_t r = 0; r < copy_rows; ++r) {
        std::copy(input + r * ld, input + r * ld + copy_cols, padded.data() + r * padded_cols);
    }
    const shape_t shape{padded_rows, padded_cols};
    const stride_t stride_in{static_cast<std::ptrdiff_t>(padded_cols * sizeof(double)),
                             static_cast<std::ptrdiff_t>(sizeof(double))};
    const std::size_t half = padded_cols / 2 + 1;
    const stride_t stride_out{static_cast<std::ptrdiff_t>(half * sizeof(std::complex<double>)),
                              static_cast<std::ptrdiff_t>(sizeof(std::complex<double>))};
    pocketfft::r2c(shape, stride_in, stride_out, shape_t{0, 1}, pocketfft::FORWARD, padded.data(), spectrum, 1.0,
                   1);
}

void irfft2(std::complex<double>* const spectrum, const std::size_t padded_rows, const std::size_t padded_cols,
            double* const output) {
    const shape_t shape{padded_rows, padded_cols};
    const std::size_t half = padded_cols / 2 + 1;
    const stride_t stride_in{static_cast<std::ptrdiff_t>(half * sizeof(std::complex<double>)),
                             static_cast<std::ptrdiff_t>(sizeof(std::complex<double>))};
    const stride_t stride_out{static_cast<std::ptrdiff_t>(padded_cols * sizeof(double)),
                              static_cast<std::ptrdiff_t>(sizeof(double))};
    const double scale = 1.0 / static_cast<double>(padded_rows * padded_cols);
    pocketfft::c2r(shape, stride_in, stride_out, shape_t{0, 1}, pocketfft::BACKWARD, spectrum, output, scale, 1);
}

void rfft2_padded_rows(const double* const input, const std::size_t rows, const std::size_t cols,
                       const std::size_t ld, const std::size_t padded_rows, const std::size_t padded_cols,
                       std::complex<double>* const spectrum, double* const row_buffer) {
    const std::size_t half = padded_cols / 2 + 1;
    const std::size_t copy_rows = std::min(rows, padded_rows);
    const std::size_t copy_cols = std::min(cols, padded_cols);
    for (std::size_t r = 0; r < copy_rows; ++r) {
        double* const row = row_buffer + r * padded_cols;
        std::copy(input + r * ld, input + r * ld + copy_cols, row);
        std::fill(row + copy_cols, row + padded_cols, 0.0);
    }
    const stride_t stride_real{static_cast<std::ptrdiff_t>(padded_cols * sizeof(double)),
                               static_cast<std::ptrdiff_t>(sizeof(double))};
    const stride_t stride_complex{static_cast<std::ptrdiff_t>(half * sizeof(std::complex<double>)),
                                  static_cast<std::ptrdiff_t>(sizeof(std::complex<double>))};
    // pocketfft's two-axis r2c is this: the last axis first, then the others
    // in place on the half-spectrum.
    if (copy_rows > 0) {
        pocketfft::r2c(shape_t{copy_rows, padded_cols}, stride_real, stride_complex, 1, pocketfft::FORWARD,
                       row_buffer, spectrum, 1.0, 1);
    }
    std::fill(spectrum + copy_rows * half, spectrum + padded_rows * half, std::complex<double>(0.0, 0.0));
    pocketfft::c2c(shape_t{padded_rows, half}, stride_complex, stride_complex, shape_t{0}, pocketfft::FORWARD,
                   spectrum, spectrum, 1.0, 1);
}

void irfft2_rows(std::complex<double>* const spectrum, const std::size_t padded_rows, const std::size_t padded_cols,
                 const std::size_t rows, const std::size_t cols, double* const output, const std::size_t ld,
                 double* const row_buffer) {
    const std::size_t half = padded_cols / 2 + 1;
    const std::size_t keep_rows = std::min(rows, padded_rows);
    const std::size_t keep_cols = std::min(cols, padded_cols);
    const stride_t stride_complex{static_cast<std::ptrdiff_t>(half * sizeof(std::complex<double>)),
                                  static_cast<std::ptrdiff_t>(sizeof(std::complex<double>))};
    const stride_t stride_real{static_cast<std::ptrdiff_t>(padded_cols * sizeof(double)),
                               static_cast<std::ptrdiff_t>(sizeof(double))};
    // pocketfft's two-axis c2r is this: every axis but the last (into a
    // temporary there, in place here), then the last one with the scale.
    pocketfft::c2c(shape_t{padded_rows, half}, stride_complex, stride_complex, shape_t{0}, pocketfft::BACKWARD,
                   spectrum, spectrum, 1.0, 1);
    if (keep_rows == 0) {
        return;
    }
    const double scale = 1.0 / static_cast<double>(padded_rows * padded_cols);
    pocketfft::c2r(shape_t{keep_rows, padded_cols}, stride_complex, stride_real, 1, pocketfft::BACKWARD, spectrum,
                   row_buffer, scale, 1);
    for (std::size_t r = 0; r < keep_rows; ++r) {
        std::copy(row_buffer + r * padded_cols, row_buffer + r * padded_cols + keep_cols, output + r * ld);
    }
}

}  // namespace pcbcore::fft
