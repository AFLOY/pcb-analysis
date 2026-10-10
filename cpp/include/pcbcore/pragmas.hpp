// Loop pragmas spelled for every compiler.  MSVC's default OpenMP runtime
// is version 2.0 and rejects ``omp simd``; there the loops are left to its
// own vectoriser.  GCC and Clang get the pragma unchanged.
#pragma once

#if defined(_MSC_VER) && !defined(__clang__)
#define PCBCORE_OMP_SIMD
#else
#define PCBCORE_OMP_SIMD _Pragma("omp simd")
#endif
