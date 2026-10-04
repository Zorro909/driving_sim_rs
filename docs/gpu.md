# HIP implementation notes

The HIP library runs statistics, sensing, network inference and vehicle physics for independent cars. Rust owns selection, crossover, mutation, resets and checkpoints. Population mean and novelty run on the GPU with reductions in the original order; offspring weights stay uploaded for the next generation.

Kernels compact live work into eight original inference batches. Only due batches run sensing and inference. A separate physics list retains inactive cars until they settle exactly. Windows replay a HIP graph over the 24-tick statistics/inference period, with direct launches for remainders. Compatible windows reuse the graph; changing track, schedule, shape, population or output mapping invalidates it.

The split physics path separates collision phases and preserves ordered per-car contact generation. `ALTD_GPU_SPLIT=0` selects the fused alternative. `ALTD_GPU_GRAPH=0` bypasses graphs. These are differential validation options, not promises of better performance.

The network forward path uses double parameters and adds products in input order. This can dominate at large populations on GPUs with low double-precision throughput. Host buffers retain mutation noise and upload packing space. A 262,144-car population with 1,661 parameters requires about 3.24 GiB for that host double buffer alone; device memory, car state and checkpoint populations are additional.

`gpu::simulation` mirrors C layouts explicitly and exports/imports canonical car and agent state. The loader resolves the shared library at runtime, so ordinary CPU builds do not link ROCm. Rebuild the Rust executable and library together when their C API or layouts change.

Historical CPU/HIP checks cover primitive math, queries, sensors, inference, physics, turnover, batch wrapping, uneven populations, simulator reuse and resumed training. They compare flags, exact values, RNG state and population bytes. These finite checks do not establish exactness for every scene or device. Run the generated-input checks on your own GPU using [the GPU guide](../gpu/README.md). [Performance notes](performance.md) retain measured results and timing boundaries.
