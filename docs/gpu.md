# HIP implementation notes

The HIP library runs statistics, sensing, network inference and vehicle physics for independent cars. It also supports GPU turnover through the runner API. Population mean and novelty run on the GPU with reductions in the original order; that turnover path keeps offspring weights uploaded for the next generation. Rust owns resets and checkpoints.

## HIP sessions

Native `SessionOptions` accepts `backend: "hip"` as well as the default `"cpu"`. The same option is available to [local server](server.md) clients. WASM sessions reject HIP. The process opens the HIP library once and allows one HIP session at a time; a session that cannot use HIP reports CPU through `backend()` and a reason through `backend_note()`.

HIP sessions drive through the existing GPU window and generation paths, then read car and agent state back into the CPU runner. Readers, metrics and checkpoints therefore use the same runner state as CPU sessions. The simulator grows when a restored or started population exceeds its capacity, and track replacement uploads a new GPU world.

Before the first window after a start or restore, a session compares 12 CPU and HIP ticks exactly from a backed-up state. It restores that state before real driving. Track replacement also requires a new comparison. A mismatch or verification error switches the session to CPU; a later HIP window error returns an error without retrying.

Session turnover uses CPU `next_generation_traced`, which retains the parent records needed by format-2 checkpoints. The next HIP window uploads the new networks. Start, restore and network replacement also invalidate the uploaded network cache. This differs from the runner's GPU turnover path, which does not produce those parent records.

Native CPU and HIP sessions share checkpoint formats with WASM. Native and WASM results have known last-bit differences; switching engines preserves the checkpoint but may change later training.

## Kernels and validation

Kernels compact live work into eight original inference batches. Only due batches run sensing and inference. A separate physics list retains inactive cars until they settle exactly. Windows replay a HIP graph over the 24-tick statistics/inference period, with direct launches for remainders. Compatible windows reuse the graph; changing track, schedule, shape, population or output mapping invalidates it.

The split physics path separates collision phases and preserves ordered per-car contact generation. `ALTD_GPU_SPLIT=0` selects the fused alternative. `ALTD_GPU_GRAPH=0` bypasses graphs. These are differential validation options, not promises of better performance.

The network forward path uses double parameters and adds products in input order. This can dominate at large populations on GPUs with low double-precision throughput. Host buffers retain mutation noise and upload packing space. A 262,144-car population with 1,661 parameters requires about 3.24 GiB for that host double buffer alone; device memory, car state and checkpoint populations are additional.

`gpu::simulation` mirrors C layouts explicitly and exports/imports canonical car and agent state. The loader resolves the shared library at runtime, so ordinary CPU builds do not link ROCm. Rebuild the Rust executable and library together when their C API or layouts change.

Historical CPU/HIP checks cover primitive math, queries, sensors, inference, physics, turnover, batch wrapping, uneven populations, simulator reuse and resumed training. They compare flags, exact values, RNG state and population bytes. These finite checks do not establish exactness for every scene or device. Run the generated-input checks on your own GPU using [the GPU guide](../gpu/README.md). [Performance notes](performance.md) retain measured results and timing boundaries.
