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

}  // namespace pcbcore::fft
