This folder holds optimization experiments and GPU tuning scripts.

## Memory operations probe

```sh
.venv/bin/python peec_fastopt/optimization_experiments/memory_ops_probe.py \
  --case-limit 4 --warmups 1 \
  --output benchmark-results/memory-ops.json
```

Compares voxel-cache on/off, `memory_reserve_fraction`, and
`release_pool_after_solve` on real `plane_opt` boards, and reports free VRAM
drift, CuPy pool size, and solve timing.
