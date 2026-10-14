// Persistent compute: K Richardson sweeps p <- p + (-omega) A p of the layered DC Q1 stencil,
// A p in the operation order of stencil2_compute.cpp.  This core's operator tiles stay in c_3
// for all sweeps (18 per tile: 0..7 coefficients, 8..16 shifted free masks, 17 fixed-node mask);
// omega is a power of two, so the scaling is exact and the sweep matches NumPy's
// p + float32(-omega) * apply_low(p) bit for bit.
// Compile-time args: 0..15 U0, 16..31 U1, 32 -omega as FP32 bits.  Runtime args: 0 sweeps K, 1 tile count.
// Breakdown define (results are then wrong): STENCIL_SKIP_MATH packs DEST without the masking and the stencil.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

constexpr uint32_t CB_X = 0;
constexpr uint32_t CB_ST = 3;
constexpr uint32_t CB_XM = 4;
constexpr uint32_t CB_OUT = 16;
constexpr uint32_t STATIC_TILES = 18;

template <uint32_t R, uint32_t COL>
FORCE_INLINE void weighted(uint32_t base, uint32_t target) {
    constexpr uint32_t u0 = get_compile_time_arg_val(R * 4 + COL);
    constexpr uint32_t u1 = get_compile_time_arg_val(16 + R * 4 + COL);
    constexpr uint32_t a = R >> 1, b = R & 1, a2 = COL >> 1, b2 = COL & 1;
    constexpr uint32_t xi = (a2 + 1 - a) * 3 + (b2 + 1 - b);
    copy_init(CB_ST);
    copy_tile(CB_ST, base + 2 * R, target);
    copy_tile(CB_ST, base + 2 * R + 1, 3);
    binop_with_scalar_tile_init();
    mul_unary_tile(target, u0);
    mul_unary_tile(3, u1);
    add_binary_tile_init();
    add_binary_tile(target, 3, target);
    copy_init(CB_XM);
    copy_tile(CB_XM, xi, 3);
    mul_binary_tile_init();
    mul_binary_tile(target, 3, target);
}

template <uint32_t R>
FORCE_INLINE void contribution(uint32_t base, uint32_t target) {
    weighted<R, 0>(base, target);
    weighted<R, 1>(base, 2);
    add_binary_tile_init();
    add_binary_tile(target, 2, target);
    weighted<R, 2>(base, 2);
    add_binary_tile_init();
    add_binary_tile(target, 2, target);
    weighted<R, 3>(base, 2);
    add_binary_tile_init();
    add_binary_tile(target, 2, target);
}

void kernel_main() {
    const uint32_t sweeps = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr uint32_t minus_omega = get_compile_time_arg_val(32);
    compute_kernel_hw_startup(CB_X, CB_ST, CB_OUT);
    if (count > 0) {
        cb_wait_front(CB_ST, count * STATIC_TILES);
    }
    for (uint32_t k = 0; k < sweeps; ++k) {
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t base = i * STATIC_TILES;
            cb_wait_front(CB_X, 9);
            cb_reserve_back(CB_XM, 9);
#ifndef STENCIL_SKIP_MATH
            for (uint32_t m = 0; m < 9; ++m) {
                tile_regs_acquire();
                copy_init(CB_X);
                copy_tile(CB_X, m, 0);
                copy_init(CB_ST);
                copy_tile(CB_ST, base + 8 + m, 1);
                mul_binary_tile_init();
                mul_binary_tile(0, 1, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CB_XM);
                tile_regs_release();
            }
#endif
            cb_push_back(CB_XM, 9);
            cb_wait_front(CB_XM, 9);

            cb_reserve_back(CB_OUT, 1);
            tile_regs_acquire();
#ifndef STENCIL_SKIP_MATH
            contribution<0>(base, 0);
            contribution<1>(base, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            contribution<2>(base, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            contribution<3>(base, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            // q = A p * free + p * (1 - free)
            copy_init(CB_ST);
            copy_tile(CB_ST, base + 12, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            copy_init(CB_X);
            copy_tile(CB_X, 4, 1);
            copy_init(CB_ST);
            copy_tile(CB_ST, base + 17, 2);
            mul_binary_tile_init();
            mul_binary_tile(1, 2, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            // p + (-omega) q
            binop_with_scalar_tile_init();
            mul_unary_tile(0, minus_omega);
            copy_init(CB_X);
            copy_tile(CB_X, 4, 1);
            add_binary_tile_init();
            add_binary_tile(1, 0, 0);
#endif
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_OUT);
            tile_regs_release();
            cb_pop_front(CB_X, 9);
            cb_pop_front(CB_XM, 9);
            cb_push_back(CB_OUT, 1);
        }
    }
}
