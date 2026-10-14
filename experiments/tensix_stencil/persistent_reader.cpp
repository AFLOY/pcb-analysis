// Persistent reader: K sweeps of the cyclic-layout stencil (see stencil3_reader.cpp) in one
// program.  Sweep k reads p_k from buffer k mod 2; from the second sweep on it first waits
// until the coordinator has seen every core finish writing p_k (the "go" semaphore reaches k).
//
// Compile-time args: 0 T_r, 1 T_c, 2+ accessors of buffer 0 and buffer 1 (FP32 tiles in L1).
// Runtime args: 0 buffer 0, 1 buffer 1 addresses, 2 sweeps K, 3 tile count n, 4 .. 3 + n tile ids.
// Semaphore 0: go.  Breakdown define (results are then wrong): STENCIL_SKIP_MOVES skips the wrap shifts.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB_X = 0;
constexpr uint32_t CB_STAGE = 8;
constexpr uint32_t TILE_BYTES = 4096;

FORCE_INLINE void compiler_fence() { asm volatile("" ::: "memory"); }

// dst(r, c) = src(r + sr, c + sc) for sr, sc in {-1, 0, 1}, zero outside the tile.
FORCE_INLINE void shifted(uint32_t* __restrict dst, const uint32_t* __restrict src, int32_t sr, int32_t sc) {
    for (int32_t r = 0; r < 32; ++r) {
        uint32_t* lo = dst + (r >> 4) * 512 + (r & 15) * 16;
        uint32_t* hi = lo + 256;
        const int32_t rr = r + sr;
        if (rr < 0 || rr >= 32) {
#pragma GCC unroll 16
            for (int32_t c = 0; c < 16; ++c) {
                lo[c] = 0u;
                hi[c] = 0u;
            }
            continue;
        }
        const uint32_t* slo = src + (rr >> 4) * 512 + (rr & 15) * 16;
        const uint32_t* shi = slo + 256;
        if (sc == 0) {
#pragma GCC unroll 16
            for (int32_t c = 0; c < 16; ++c) {
                lo[c] = slo[c];
                hi[c] = shi[c];
            }
        } else if (sc > 0) {
#pragma GCC unroll 15
            for (int32_t c = 0; c < 15; ++c) {
                lo[c] = slo[c + 1];
                hi[c] = shi[c + 1];
            }
            lo[15] = shi[0];
            hi[15] = 0u;
        } else {
#pragma GCC unroll 15
            for (int32_t c = 1; c < 16; ++c) {
                lo[c] = slo[c - 1];
                hi[c] = shi[c - 1];
            }
            lo[0] = 0u;
            hi[0] = slo[15];
        }
    }
}

template <typename Accessor>
FORCE_INLINE void neighbours(const Accessor& p, uint32_t t, uint32_t stage, int32_t T_r, int32_t T_c) {
    const int32_t layer = t / (T_r * T_c);
    const int32_t pi = (t / T_c) % T_r;
    const int32_t qi = t % T_c;
    cb_reserve_back(CB_X, 9);
    const uint32_t xt = get_write_ptr(CB_X);
    int32_t shift_r[9], shift_c[9];
    uint32_t slot_of[9];
    uint32_t staged = 0;
    for (int32_t k = 0; k < 9; ++k) {
        int32_t p2 = pi + k / 3 - 1, q2 = qi + k % 3 - 1, sr = 0, sc = 0;
        if (p2 < 0) { p2 = T_r - 1; sr = -1; }
        if (p2 >= T_r) { p2 = 0; sr = 1; }
        if (q2 < 0) { q2 = T_c - 1; sc = -1; }
        if (q2 >= T_c) { q2 = 0; sc = 1; }
        shift_r[k] = sr;
        shift_c[k] = sc;
        const uint32_t source = (layer * T_r + p2) * T_c + q2;
        if (sr == 0 && sc == 0) {
            noc_async_read(p.get_noc_addr(source), xt + k * TILE_BYTES, TILE_BYTES);
        } else {
            slot_of[k] = staged;
            noc_async_read(p.get_noc_addr(source), stage + staged * TILE_BYTES, TILE_BYTES);
            ++staged;
        }
    }
    noc_async_read_barrier();
    compiler_fence();
#ifndef STENCIL_SKIP_MOVES
    for (int32_t k = 0; k < 9; ++k) {
        if (shift_r[k] != 0 || shift_c[k] != 0) {
            shifted(reinterpret_cast<uint32_t*>(xt + k * TILE_BYTES),
                    reinterpret_cast<const uint32_t*>(stage + slot_of[k] * TILE_BYTES), shift_r[k], shift_c[k]);
        }
    }
#endif
    compiler_fence();
    cb_push_back(CB_X, 9);
}

void kernel_main() {
    constexpr int32_t T_r = get_compile_time_arg_val(0);
    constexpr int32_t T_c = get_compile_time_arg_val(1);
    constexpr auto b0_args = TensorAccessorArgs<2>();
    constexpr auto b1_args = TensorAccessorArgs<b0_args.next_compile_time_args_offset()>();
    const auto b0 = TensorAccessor(b0_args, get_arg_val<uint32_t>(0));
    const auto b1 = TensorAccessor(b1_args, get_arg_val<uint32_t>(1));
    const uint32_t sweeps = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));

    cb_reserve_back(CB_STAGE, 1);
    const uint32_t stage = get_write_ptr(CB_STAGE);
    for (uint32_t k = 0; k < sweeps; ++k) {
        if (k > 0) {
            noc_semaphore_wait_min(go, k);
        }
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t t = get_arg_val<uint32_t>(4 + i);
            if (k % 2 == 0) {
                neighbours(b0, t, stage, T_r, T_c);
            } else {
                neighbours(b1, t, stage, T_r, T_c);
            }
        }
    }
}
