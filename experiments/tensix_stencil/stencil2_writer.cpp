// v2 writer of the layered DC Q1 stencil: streams the 18 precomputed operator tiles of each
// output tile (8 shifted element-coefficient tiles, 9 shifted free masks, the fixed-node mask)
// into c_3, one tile ahead of the compute kernel, and writes each result tile.
// Compile-time args: 0+ accessors of the operator tiles and of the output.
// Runtime args: 0 operator address, 1 output address, 2 first tile, 3 tile count; with
// STENCIL_TILE_LIST instead 2 tile count n, 3 .. 2 + n tile ids.
// Breakdown define (results are then wrong): STENCIL_SKIP_STATIC pushes c_3 without reading.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB_ST = 3;
constexpr uint32_t CB_OUT = 16;
constexpr uint32_t STATIC_TILES = 18;

template <typename Accessor>
FORCE_INLINE void push_operator(const Accessor& op, uint32_t t) {
    const uint32_t bytes = get_tile_size(CB_ST);
    cb_reserve_back(CB_ST, STATIC_TILES);
#ifndef STENCIL_SKIP_STATIC
    uint32_t dst = get_write_ptr(CB_ST);
    for (uint32_t k = 0; k < STATIC_TILES; ++k) {
        noc_async_read(op.get_noc_addr(t * STATIC_TILES + k), dst + k * bytes, bytes);
    }
    noc_async_read_barrier();
#endif
    cb_push_back(CB_ST, STATIC_TILES);
}

void kernel_main() {
    constexpr auto op_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<op_args.next_compile_time_args_offset()>();
    const auto op = TensorAccessor(op_args, get_arg_val<uint32_t>(0));
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(1));
#ifdef STENCIL_TILE_LIST
    const uint32_t count = get_arg_val<uint32_t>(2);
    auto tile = [](uint32_t i) { return get_arg_val<uint32_t>(3 + i); };
#else
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    auto tile = [first](uint32_t i) { return first + i; };
#endif
    const uint32_t bytes = get_tile_size(CB_OUT);
    if (count == 0) {
        return;
    }
    push_operator(op, tile(0));
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t t = tile(i);
        if (i + 1 < count) {
            push_operator(op, tile(i + 1));
        }
        cb_wait_front(CB_OUT, 1);
        noc_async_write(get_read_ptr(CB_OUT), out.get_noc_addr(t), bytes);
        noc_async_write_barrier();
        cb_pop_front(CB_OUT, 1);
    }
}
