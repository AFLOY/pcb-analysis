// Reader of the layered DC Q1 stencil: builds, per output tile of 32 x 32 nodes, the nine
// shifted tiles of the masked input and the eight shifted element-coefficient tiles.
//
// Inputs are row-major FP32 arrays of 128-byte pages (32 values).  Layer l, storage row s,
// chunk k is page (l * SR + s) * NCH + k.  Node (i, j) is stored at row i + 1, column j + 32;
// element (i, j) of c0/c1 likewise, with zeros outside the grid.  The patch is read with
// aligned page reads into a staging area and shifted word by word on this RISC-V (integer
// moves only), so the values reach the compute kernel bit for bit.
//
// Compile-time args: 0 T_r, 1 T_c (tiles per layer), 2 SR, 3 NCH, 4+ accessors x, mask, c0, c1.
// Runtime args: 0 x, 1 mask, 2 c0, 3 c1 addresses, 4 first tile, 5 tile count.
// Breakdown defines (results are then wrong): STENCIL_SKIP_READS, STENCIL_SKIP_MOVES.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB_X = 0;      // 9 tiles: (di + 1) * 3 + (dj + 1)
constexpr uint32_t CB_C = 1;      // 8 tiles: element shift r, then c0 / c1
constexpr uint32_t CB_S = 2;      // 2 tiles: free mask (0.0 / 1.0), input at fixed nodes
constexpr uint32_t CB_STAGE = 8;  // private staging area
constexpr uint32_t XW = 96;       // staged values per patch row of x and mask
constexpr uint32_t CW = 64;       // staged values per patch row of c0 and c1
constexpr uint32_t TILE_WORDS = 1024;

using l1_words = volatile tt_l1_ptr uint32_t*;

// Tile (r, c) <- src[(r + row0) * width + col0 + c]; FP32 tiles are four row-major 16 x 16 faces.
FORCE_INLINE void build_tile(l1_words dst, l1_words src, uint32_t width, uint32_t row0, uint32_t col0) {
    for (uint32_t r = 0; r < 32; ++r) {
        l1_words row = src + (r + row0) * width + col0;
        l1_words face = dst + (r >> 4) * 512 + (r & 15) * 16;
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
    constexpr auto m_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto c0_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    constexpr auto c1_args = TensorAccessorArgs<c0_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const auto m = TensorAccessor(m_args, get_arg_val<uint32_t>(1));
    const auto c0 = TensorAccessor(c0_args, get_arg_val<uint32_t>(2));
    const auto c1 = TensorAccessor(c1_args, get_arg_val<uint32_t>(3));
    const uint32_t first = get_arg_val<uint32_t>(4);
    const uint32_t count = get_arg_val<uint32_t>(5);

    cb_reserve_back(CB_STAGE, 1);
    const uint32_t stage = get_write_ptr(CB_STAGE);
    const uint32_t sx = stage;
    const uint32_t sm = sx + 34 * XW * 4;
    const uint32_t sc0 = sm + 34 * XW * 4;
    const uint32_t sc1 = sc0 + 33 * CW * 4;
    l1_words px = reinterpret_cast<l1_words>(sx);
    l1_words pm = reinterpret_cast<l1_words>(sm);
    l1_words pc0 = reinterpret_cast<l1_words>(sc0);
    l1_words pc1 = reinterpret_cast<l1_words>(sc1);

    for (uint32_t t = first; t < first + count; ++t) {
        const uint32_t layer = t / (T_r * T_c);
        const uint32_t ti = (t / T_c) % T_r;
        const uint32_t tj = t % T_c;
        // Patch rows: node rows 32 ti - 1 .. 32 ti + 32 (x, mask), element rows 32 ti - 1 .. 32 ti + 31.
#ifndef STENCIL_SKIP_READS
        for (uint32_t pr = 0; pr < 34; ++pr) {
            const uint32_t page = (layer * SR + 32 * ti + pr) * NCH + tj;
            for (uint32_t k = 0; k < 3; ++k) {
                noc_async_read(x.get_noc_addr(page + k), sx + (pr * XW + k * 32) * 4, 128);
                noc_async_read(m.get_noc_addr(page + k), sm + (pr * XW + k * 32) * 4, 128);
            }
            if (pr < 33) {
                for (uint32_t k = 0; k < 2; ++k) {
                    noc_async_read(c0.get_noc_addr(page + k), sc0 + (pr * CW + k * 32) * 4, 128);
                    noc_async_read(c1.get_noc_addr(page + k), sc1 + (pr * CW + k * 32) * 4, 128);
                }
            }
        }
        noc_async_read_barrier();
#endif

        // Free mask and the input at fixed nodes, for the final where(free, A x, x).
        cb_reserve_back(CB_S, 2);
#ifndef STENCIL_SKIP_MOVES
        l1_words mask_tile = reinterpret_cast<l1_words>(get_write_ptr(CB_S));
        l1_words fixed_tile = mask_tile + TILE_WORDS;
        build_tile(mask_tile, pm, XW, 1, 32);
        build_tile(fixed_tile, px, XW, 1, 32);
        for (uint32_t w = 0; w < TILE_WORDS; ++w) {
            fixed_tile[w] = mask_tile[w] ? 0u : fixed_tile[w];
        }
#endif
        cb_push_back(CB_S, 2);

        // Masked input: working = where(free, x, 0).
        cb_reserve_back(CB_X, 9);
#ifndef STENCIL_SKIP_MOVES
        for (uint32_t w = 0; w < 34 * XW; ++w) {
            px[w] = pm[w] ? px[w] : 0u;
        }
        l1_words xt = reinterpret_cast<l1_words>(get_write_ptr(CB_X));
        for (uint32_t di = 0; di < 3; ++di) {
            for (uint32_t dj = 0; dj < 3; ++dj) {
                build_tile(xt + (di * 3 + dj) * TILE_WORDS, px, XW, di, 31 + dj);
            }
        }
#endif
        cb_push_back(CB_X, 9);

        // Element (i - a, j - b) for the node's local position r = 2 a + b.
        cb_reserve_back(CB_C, 8);
#ifndef STENCIL_SKIP_MOVES
        l1_words ct = reinterpret_cast<l1_words>(get_write_ptr(CB_C));
        for (uint32_t r = 0; r < 4; ++r) {
            const uint32_t a = r >> 1;
            const uint32_t b = r & 1;
            build_tile(ct + (2 * r) * TILE_WORDS, pc0, CW, 1 - a, 32 - b);
            build_tile(ct + (2 * r + 1) * TILE_WORDS, pc1, CW, 1 - a, 32 - b);
        }
#endif
        cb_push_back(CB_C, 8);
    }
}
