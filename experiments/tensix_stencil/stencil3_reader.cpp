// v3 reader of the layered DC Q1 stencil on the cyclic tile layout.
//
// Node (i, j) of a layer lives in tile (p, q) = (i mod T_r, j mod T_c) at position
// (r, c) = (i div T_r, j div T_c), so neighbour (i + di, j + dj) is position (r, c) of tile
// (p + di, q + dj) unless the tile index wraps; then it is tile (p + di) mod T_r at position
// r +/- 1 (and likewise for columns).  Interior neighbours are read as whole tiles straight
// into the compute CB; only wrapped ones are shifted by one position on this RISC-V, with
// zeros beyond the padded grid.
//
// Compile-time args: 0 T_r, 1 T_c, 2+ accessor of x (FP32 tiles, id (l T_r + p) T_c + q).
// Runtime args: 0 x address, 1 tile count n, 2 .. 1 + n tile ids (balanced by the host).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB_X = 0;      // 9 tiles: (di + 1) * 3 + (dj + 1)
constexpr uint32_t CB_STAGE = 8;  // up to 8 wrapped source tiles
constexpr uint32_t TILE_WORDS = 1024;
constexpr uint32_t TILE_BYTES = 4096;

FORCE_INLINE void compiler_fence() { asm volatile("" ::: "memory"); }

// dst(r, c) = src(r + sr, c + sc) for sr, sc in {-1, 0, 1}, zero outside the tile.  A tile is
// four row-major 16 x 16 faces; each destination row is assembled from contiguous runs.
FORCE_INLINE void shifted(uint32_t* __restrict dst, const uint32_t* __restrict src, int32_t sr, int32_t sc) {
    for (int32_t r = 0; r < 32; ++r) {
        uint32_t* lo = dst + (r >> 4) * 512 + (r & 15) * 16;  // columns 0..15
        uint32_t* hi = lo + 256;                              // columns 16..31
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
        } else if (sc > 0) {  // dst c <- src c + 1
#pragma GCC unroll 15
            for (int32_t c = 0; c < 15; ++c) {
                lo[c] = slo[c + 1];
                hi[c] = shi[c + 1];
            }
            lo[15] = shi[0];
            hi[15] = 0u;
        } else {  // dst c <- src c - 1
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

void kernel_main() {
    constexpr int32_t T_r = get_compile_time_arg_val(0);
    constexpr int32_t T_c = get_compile_time_arg_val(1);
    constexpr auto x_args = TensorAccessorArgs<2>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const uint32_t count = get_arg_val<uint32_t>(1);

    cb_reserve_back(CB_STAGE, 1);
    const uint32_t stage = get_write_ptr(CB_STAGE);

    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t t = get_arg_val<uint32_t>(2 + i);
        const int32_t layer = t / (T_r * T_c);
        const int32_t p = (t / T_c) % T_r;
        const int32_t q = t % T_c;
        cb_reserve_back(CB_X, 9);
        const uint32_t xt = get_write_ptr(CB_X);
        int32_t shift_r[9], shift_c[9];
        uint32_t staged = 0;
        uint32_t slot_of[9];
        for (int32_t k = 0; k < 9; ++k) {
            int32_t p2 = p + k / 3 - 1, q2 = q + k % 3 - 1, sr = 0, sc = 0;
            if (p2 < 0) { p2 = T_r - 1; sr = -1; }
            if (p2 >= T_r) { p2 = 0; sr = 1; }
            if (q2 < 0) { q2 = T_c - 1; sc = -1; }
            if (q2 >= T_c) { q2 = 0; sc = 1; }
            shift_r[k] = sr;
            shift_c[k] = sc;
            const uint32_t source = (layer * T_r + p2) * T_c + q2;
            if (sr == 0 && sc == 0) {
                noc_async_read(x.get_noc_addr(source), xt + k * TILE_BYTES, TILE_BYTES);
            } else {
                slot_of[k] = staged;
                noc_async_read(x.get_noc_addr(source), stage + staged * TILE_BYTES, TILE_BYTES);
                ++staged;
            }
        }
        noc_async_read_barrier();
        compiler_fence();
        for (int32_t k = 0; k < 9; ++k) {
            if (shift_r[k] != 0 || shift_c[k] != 0) {
                shifted(reinterpret_cast<uint32_t*>(xt + k * TILE_BYTES),
                        reinterpret_cast<const uint32_t*>(stage + slot_of[k] * TILE_BYTES), shift_r[k], shift_c[k]);
            }
        }
        compiler_fence();
        cb_push_back(CB_X, 9);
    }
}
