// Compute of the layered DC Q1 stencil on the SFPU with FP32 DEST, in the operation order of
// MatrixFreePCBOperator._apply_impl, so that the result matches the NumPy FP32 action:
//   for r: C_r = sum_col (c0 U0[r, col] + c1 U1[r, col]) x_col;  out = ((C_0 + C_1) + C_2) + C_3
//   out = where(free, out, x)  (as out * free + x_fixed)
// Compile-time args: 0..15 U0[r][col], 16..31 U1[r][col] as FP32 bits.  Runtime args: 0 tile count.
// Breakdown define (the result is then wrong): STENCIL_SKIP_MATH packs DEST without computing.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

constexpr uint32_t CB_X = 0;
constexpr uint32_t CB_C = 1;
constexpr uint32_t CB_S = 2;
constexpr uint32_t CB_OUT = 16;

template <uint32_t R, uint32_t COL>
FORCE_INLINE void weighted(uint32_t target) {
    constexpr uint32_t u0 = get_compile_time_arg_val(R * 4 + COL);
    constexpr uint32_t u1 = get_compile_time_arg_val(16 + R * 4 + COL);
    constexpr uint32_t a = R >> 1, b = R & 1, a2 = COL >> 1, b2 = COL & 1;
    constexpr uint32_t xi = (a2 + 1 - a) * 3 + (b2 + 1 - b);
    copy_init(CB_C);
    copy_tile(CB_C, 2 * R, target);
    copy_tile(CB_C, 2 * R + 1, 3);
    binop_with_scalar_tile_init();
    mul_unary_tile(target, u0);
    mul_unary_tile(3, u1);
    add_binary_tile_init();
    add_binary_tile(target, 3, target);
    copy_init(CB_X);
    copy_tile(CB_X, xi, 3);
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
    compute_kernel_hw_startup(CB_X, CB_C, CB_OUT);
    for (uint32_t t = 0; t < count; ++t) {
        cb_wait_front(CB_S, 2);
        cb_wait_front(CB_X, 9);
        cb_wait_front(CB_C, 8);
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
        copy_init(CB_S);
        copy_tile(CB_S, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        copy_tile(CB_S, 1, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        tile_regs_release();
        cb_pop_front(CB_S, 2);
        cb_pop_front(CB_X, 9);
        cb_pop_front(CB_C, 8);
        cb_push_back(CB_OUT, 1);
    }
}
