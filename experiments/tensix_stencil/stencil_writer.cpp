// Writer of the layered DC Q1 stencil: c_16 to tiles [first, first + count) of the FP32 output.
// Compile-time args: 0+ accessor of the output.  Runtime args: 0 address, 1 first tile, 2 count.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CB_OUT = 16;
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(0));
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t bytes = get_tile_size(CB_OUT);
    for (uint32_t t = first; t < first + count; ++t) {
        cb_wait_front(CB_OUT, 1);
        noc_async_write(get_read_ptr(CB_OUT), out.get_noc_addr(t), bytes);
        noc_async_write_barrier();
        cb_pop_front(CB_OUT, 1);
    }
}
