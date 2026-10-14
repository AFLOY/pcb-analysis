// Persistent compute with precomputed masked weights, batched over eight FP32 DEST tiles
// (dst_full_sync_en): K Richardson sweeps p <- p + (-omega) A p.
// c_3 holds per output tile 18 tiles: 0..15 the weight (c0 U0[r, col] + c1 U1[r, col]) of the
// node's element r times the free mask of that element's local node col (both rounded in FP32
// on the host, as the NumPy action does), 16 the free mask, 17 the fixed-node mask.  Then
//   C_r = sum_col W[r, col] p_col;  A p = ((C_0 + C_1) + C_2) + C_3;  q = A p * free + p * fixed
//   p <- p + (-omega) q
// which is the NumPy FP32 action's operation sequence value for value (W x m equals W (x m)).
// Compile-time args: 0 -omega as FP32 bits.  Runtime args: 0 sweeps K, 1 tile count.
// Operations of one kind are issued together (one init per batch); the operation order per node
// is that of persistent_weights_compute.cpp.
// Breakdown define (results are then wrong): STENCIL_SKIP_MATH packs DEST without the stencil.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

constexpr uint32_t CB_X = 0;
constexpr uint32_t CB_ST = 3;
constexpr uint32_t CB_OUT = 16;
constexpr uint32_t STATIC_TILES = 18;

// C_r = ((W0 p0 + W1 p1) + W2 p2) + W3 p3 into DEST tile `c` (2 .. 5 hold the weighted terms).
template <uint32_t R>
FORCE_INLINE void contribution(uint32_t base, uint32_t c) {
    constexpr uint32_t a = R >> 1, b = R & 1;
    constexpr uint32_t x0 = (0 + 1 - a) * 3 + (0 + 1 - b);
    constexpr uint32_t x1 = (0 + 1 - a) * 3 + (1 + 1 - b);
    constexpr uint32_t x2 = (1 + 1 - a) * 3 + (0 + 1 - b);
    constexpr uint32_t x3 = (1 + 1 - a) * 3 + (1 + 1 - b);
    copy_init(CB_ST);
    copy_tile(CB_ST, base + R * 4 + 0, c);
    copy_tile(CB_ST, base + R * 4 + 1, 3);
    copy_tile(CB_ST, base + R * 4 + 2, 4);
    copy_tile(CB_ST, base + R * 4 + 3, 5);
    copy_init(CB_X);
    copy_tile(CB_X, x0, 6);
    copy_tile(CB_X, x1, 7);
    mul_binary_tile_init();
    mul_binary_tile(c, 6, c);
    mul_binary_tile(3, 7, 3);
    copy_init(CB_X);
    copy_tile(CB_X, x2, 6);
    copy_tile(CB_X, x3, 7);
    mul_binary_tile_init();
    mul_binary_tile(4, 6, 4);
    mul_binary_tile(5, 7, 5);
    add_binary_tile_init();
    add_binary_tile(c, 3, c);
    add_binary_tile(c, 4, c);
    add_binary_tile(c, 5, c);
}

void kernel_main() {
    const uint32_t sweeps = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr uint32_t minus_omega = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(CB_X, CB_ST, CB_OUT);
    if (count > 0) {
        cb_wait_front(CB_ST, count * STATIC_TILES);
    }
    for (uint32_t k = 0; k < sweeps; ++k) {
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t base = i * STATIC_TILES;
            cb_wait_front(CB_X, 9);
            cb_reserve_back(CB_OUT, 1);
            tile_regs_acquire();
#ifndef STENCIL_SKIP_MATH
            contribution<0>(base, 0);
            contribution<1>(base, 2);
            add_binary_tile(0, 2, 0);
            contribution<2>(base, 2);
            add_binary_tile(0, 2, 0);
            contribution<3>(base, 2);
            add_binary_tile(0, 2, 0);
            // q = A p * free + p * fixed;  p + (-omega) q
            copy_init(CB_ST);
            copy_tile(CB_ST, base + 16, 1);
            copy_tile(CB_ST, base + 17, 2);
            copy_init(CB_X);
            copy_tile(CB_X, 4, 3);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            mul_binary_tile(3, 2, 2);
            add_binary_tile_init();
            add_binary_tile(0, 2, 0);
            binop_with_scalar_tile_init();
            mul_unary_tile(0, minus_omega);
            add_binary_tile_init();
            add_binary_tile(3, 0, 0);
#endif
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_OUT);
            tile_regs_release();
            cb_pop_front(CB_X, 9);
            cb_push_back(CB_OUT, 1);
        }
    }
}
