// Flush subnormal results and inputs to zero for the lifetime of a scope.
//
// The FP32 inner solves meet subnormal corrections late in a solve; on x86 a
// subnormal operand makes SIMD arithmetic several times slower, and the FP64
// outer residual remains the acceptance criterion, so the inner kernels run
// with flush-to-zero and denormals-are-zero set and restore the caller's
// mode on exit.  aarch64 has one flush-to-zero bit (FPCR.FZ); other targets
// leave the mode alone.
#pragma once

#include <cstdint>

#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#include <xmmintrin.h>
#define PCBCORE_FLUSH_SSE 1
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
#define PCBCORE_FLUSH_AARCH64 1
#endif

namespace pcbcore {

class FlushSubnormals final {
public:
    FlushSubnormals() noexcept {
#if defined(PCBCORE_FLUSH_SSE)
        saved_ = _mm_getcsr();
        _mm_setcsr(static_cast<unsigned int>(saved_) | 0x8040U);  // FTZ | DAZ
#elif defined(PCBCORE_FLUSH_AARCH64)
        std::uint64_t fpcr = 0;
        __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
        saved_ = fpcr;
        fpcr |= (std::uint64_t{1} << 24);  // FZ
        __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr));
#endif
    }
    ~FlushSubnormals() noexcept {
#if defined(PCBCORE_FLUSH_SSE)
        _mm_setcsr(static_cast<unsigned int>(saved_));
#elif defined(PCBCORE_FLUSH_AARCH64)
        const std::uint64_t fpcr = saved_;
        __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr));
#endif
    }
    FlushSubnormals(const FlushSubnormals&) = delete;
    FlushSubnormals& operator=(const FlushSubnormals&) = delete;
    FlushSubnormals(FlushSubnormals&&) = delete;
    FlushSubnormals& operator=(FlushSubnormals&&) = delete;

private:
    std::uint64_t saved_{0};
};

}  // namespace pcbcore

// Products and sums the vectoriser must not fuse into an FMA.  GCC accepts
// the attribute per function; Clang and MSVC get -ffp-contract=off (Clang)
// or contract nothing by default (MSVC) for the whole kernel library.
#if defined(__GNUC__) && !defined(__clang__)
#define PCBCORE_NO_FP_CONTRACT __attribute__((optimize("-ffp-contract=off")))
#else
#define PCBCORE_NO_FP_CONTRACT
#endif
