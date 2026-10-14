// v2 compute of the layered DC Q1 stencil (SFPU, FP32 DEST), same operation order as v1:
//   working_k = x_k * free_k for the nine shifts (packed to c_4)
//   C_r = sum_col (c0 U0[r, col] + c1 U1[r, col]) working_col;  out = ((C_0 + C_1) + C_2) + C_3
//   out = out * free + x * (1 - free)
// c_3 holds per output tile: 0..7 coefficients (2 r + {c0, c1}), 8..16 shifted free masks,
// 17 the fixed-node mask.  Compile-time args: 0..15 U0, 16..31 U1 as FP32 bits.
// Runtime args: 0 tile count.
// Breakdown defines (results are then wrong): STENCIL_SKIP_MASK, STENCIL_SKIP_MATH.
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

template <uint32_t R, uint32_t COL>
FORCE_INLINE void weighted(uint32_t target) {
    constexpr uint32_t u0 = get_compile_time_arg_val(R * 4 + COL);
    constexpr uint32_t u1 = get_compile_time_arg_val(16 + R * 4 + COL);
    constexpr uint32_t a = R >> 1, b = R & 1, a2 = COL >> 1, b2 = COL & 1;
    constexpr uint32_t xi = (a2 + 1 - a) * 3 + (b2 + 1 - b);
    copy_init(CB_ST);
    copy_tile(CB_ST, 2 * R, target);
    copy_tile(CB_ST, 2 * R + 1, 3);
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
FORCE_INLINE void contribution(uint32_t target) {
    weighted<R, 0>(target);
    weighted<R, 1>(2);
    add_binary_tile_init();
    add_binary_tile(target, 2, target);
    weighted<R, 2>(2);
    add_binary_tile_init();
    add_binary_tile(target, 2, target);
    weighted<R, 3>(2);
    add_binary_tile_init();
    add_binary_tile(target, 2, target);
}

void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(CB_X, CB_ST, CB_OUT);
    for (uint32_t t = 0; t < count; ++t) {
        cb_wait_front(CB_X, 9);
        cb_wait_front(CB_ST, 18);
        cb_reserve_back(CB_XM, 9);
        for (uint32_t k = 0; k < 9; ++k) {
            tile_regs_acquire();
#ifndef STENCIL_SKIP_MASK
            copy_init(CB_X);
            copy_tile(CB_X, k, 0);
            copy_init(CB_ST);
            copy_tile(CB_ST, 8 + k, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
#endif
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_XM, k);
            tile_regs_release();
        }
        cb_push_back(CB_XM, 9);
        cb_wait_front(CB_XM, 9);

        cb_reserve_back(CB_OUT, 1);
        tile_regs_acquire();
#ifndef STENCIL_SKIP_MATH
        contribution<0>(0);
        contribution<1>(1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        contribution<2>(1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        contribution<3>(1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        copy_init(CB_ST);
        copy_tile(CB_ST, 12, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        copy_init(CB_X);
        copy_tile(CB_X, 4, 1);
        copy_init(CB_ST);
        copy_tile(CB_ST, 17, 2);
        mul_binary_tile_init();
        mul_binary_tile(1, 2, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        tile_regs_release();
        cb_pop_front(CB_X, 9);
        cb_pop_front(CB_XM, 9);
        cb_pop_front(CB_ST, 18);
        cb_push_back(CB_OUT, 1);
    }
}
