// v2 reader of the layered DC Q1 stencil: only the input vector is shifted per call.
//
// Per output tile of 32 x 32 nodes it reads the 34 x 34 node patch of x (aligned 128-byte
// pages of the row-major layout described in stencil_reader.cpp) and builds the nine shifted,
// unmasked input tiles with plain (non-volatile) word moves; the compute kernel masks them.
//
// Compile-time args: 0 T_r, 1 T_c, 2 SR, 3 NCH, 4+ accessor of x.
// Runtime args: 0 x address, 1 first tile, 2 tile count.
// Breakdown defines (results are then wrong): STENCIL_SKIP_READS, STENCIL_SKIP_MOVES.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB_X = 0;      // 9 tiles: (di + 1) * 3 + (dj + 1)
constexpr uint32_t CB_STAGE = 8;  // private staging area
constexpr uint32_t XW = 96;
constexpr uint32_t TILE_WORDS = 1024;

FORCE_INLINE void compiler_fence() { asm volatile("" ::: "memory"); }

FORCE_INLINE void build_tile(uint32_t* __restrict dst, const uint32_t* __restrict src, uint32_t row0, uint32_t col0) {
    for (uint32_t r = 0; r < 32; ++r) {
        const uint32_t* row = src + (r + row0) * XW + col0;
        uint32_t* face = dst + (r >> 4) * 512 + (r & 15) * 16;
#pragma GCC unroll 16
        for (uint32_t c = 0; c < 16; ++c) {
            face[c] = row[c];
        }
#pragma GCC unroll 16
        for (uint32_t c = 0; c < 16; ++c) {
            face[256 + c] = row[16 + c];
        }
    }
}

void kernel_main() {
    constexpr uint32_t T_r = get_compile_time_arg_val(0);
    constexpr uint32_t T_c = get_compile_time_arg_val(1);
    constexpr uint32_t SR = get_compile_time_arg_val(2);
    constexpr uint32_t NCH = get_compile_time_arg_val(3);
    constexpr auto x_args = TensorAccessorArgs<4>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);

    cb_reserve_back(CB_STAGE, 1);
    const uint32_t stage = get_write_ptr(CB_STAGE);
    const uint32_t* px = reinterpret_cast<const uint32_t*>(stage);

    for (uint32_t t = first; t < first + count; ++t) {
        const uint32_t layer = t / (T_r * T_c);
        const uint32_t ti = (t / T_c) % T_r;
        const uint32_t tj = t % T_c;
#ifndef STENCIL_SKIP_READS
        for (uint32_t pr = 0; pr < 34; ++pr) {
            const uint32_t page = (layer * SR + 32 * ti + pr) * NCH + tj;
            for (uint32_t k = 0; k < 3; ++k) {
                noc_async_read(x.get_noc_addr(page + k), stage + (pr * XW + k * 32) * 4, 128);
            }
        }
        noc_async_read_barrier();
#endif
        compiler_fence();
        cb_reserve_back(CB_X, 9);
#ifndef STENCIL_SKIP_MOVES
        uint32_t* xt = reinterpret_cast<uint32_t*>(get_write_ptr(CB_X));
        for (uint32_t di = 0; di < 3; ++di) {
            for (uint32_t dj = 0; dj < 3; ++dj) {
                build_tile(xt + (di * 3 + dj) * TILE_WORDS, px, di, 31 + dj);
            }
        }
#endif
        compiler_fence();
        cb_push_back(CB_X, 9);
    }
}
