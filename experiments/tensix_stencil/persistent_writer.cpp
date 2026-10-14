// Persistent writer: loads this core's operator tiles once (they stay in c_3 for all sweeps),
// then per sweep writes each result tile to buffer (k + 1) mod 2, waits for the writes, and
// arrives at the coordinator.  The coordinator (is_coordinator = 1) waits for every core's
// arrival and increments every core's "go" semaphore, which releases the readers and the
// writers waiting at the barrier.
//
// Compile-time args: 0+ accessors of the operator tiles, buffer 0 and buffer 1.
// Runtime args: 0 operator, 1 buffer 0, 2 buffer 1 addresses, 3 sweeps K, 4 coordinator x,
// 5 coordinator y, 6 is_coordinator, 7 cores m, 8 tile count n, 9 .. 8 + n tile ids, then on the
// coordinator m (x, y) pairs of every core.  Semaphores: 0 go, 1 arrivals (used on the coordinator).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB_ST = 3;
constexpr uint32_t CB_OUT = 16;
constexpr uint32_t STATIC_TILES = 18;

void kernel_main() {
    constexpr auto op_args = TensorAccessorArgs<0>();
    constexpr auto b0_args = TensorAccessorArgs<op_args.next_compile_time_args_offset()>();
    constexpr auto b1_args = TensorAccessorArgs<b0_args.next_compile_time_args_offset()>();
    const auto op = TensorAccessor(op_args, get_arg_val<uint32_t>(0));
    const auto b0 = TensorAccessor(b0_args, get_arg_val<uint32_t>(1));
    const auto b1 = TensorAccessor(b1_args, get_arg_val<uint32_t>(2));
    const uint32_t sweeps = get_arg_val<uint32_t>(3);
    const uint32_t coordinator_x = get_arg_val<uint32_t>(4);
    const uint32_t coordinator_y = get_arg_val<uint32_t>(5);
    const bool is_coordinator = get_arg_val<uint32_t>(6) != 0;
    const uint32_t cores = get_arg_val<uint32_t>(7);
    const uint32_t count = get_arg_val<uint32_t>(8);
    const uint32_t bytes = get_tile_size(CB_OUT);
    const uint32_t go = get_semaphore(0);
    const uint32_t arrivals = get_semaphore(1);
    const uint64_t arrive_at = get_noc_addr(coordinator_x, coordinator_y, arrivals);

    if (count > 0) {
        cb_reserve_back(CB_ST, count * STATIC_TILES);
        const uint32_t dst = get_write_ptr(CB_ST);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t t = get_arg_val<uint32_t>(9 + i);
            for (uint32_t k = 0; k < STATIC_TILES; ++k) {
                noc_async_read(op.get_noc_addr(t * STATIC_TILES + k), dst + (i * STATIC_TILES + k) * bytes, bytes);
            }
        }
        noc_async_read_barrier();
        cb_push_back(CB_ST, count * STATIC_TILES);
    }

    for (uint32_t k = 0; k < sweeps; ++k) {
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t t = get_arg_val<uint32_t>(9 + i);
            cb_wait_front(CB_OUT, 1);
            const uint64_t target = (k % 2 == 0) ? b1.get_noc_addr(t) : b0.get_noc_addr(t);
            noc_async_write(get_read_ptr(CB_OUT), target, bytes);
            noc_async_write_barrier();
            cb_pop_front(CB_OUT, 1);
        }
        noc_semaphore_inc(arrive_at, 1);
        noc_async_atomic_barrier();
        if (is_coordinator) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrivals), cores * (k + 1));
            for (uint32_t c = 0; c < cores; ++c) {
                const uint32_t x = get_arg_val<uint32_t>(9 + count + 2 * c);
                const uint32_t y = get_arg_val<uint32_t>(10 + count + 2 * c);
                noc_semaphore_inc(get_noc_addr(x, y, go), 1);
            }
            noc_async_atomic_barrier();
        }
        // Every core waits at the barrier, also one without tiles, whose arrivals would otherwise
        // run ahead and release later sweeps early.
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go), k + 1);
    }
}
