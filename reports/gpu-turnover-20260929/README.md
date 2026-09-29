# GPU generation overhead, 2026-09-29

This change reduces time between driving generations. It batches exact CPU random draws, reuses allocation space, and moves population mean and novelty calculations to the GPU. Selection, crossover, mutation and vehicle reset still run on the CPU. The earlier physics, inference batching and graph improvements are present in both versions. The removed multi-tick experiment is absent from both.

Complete generations take 31–34% less time, corresponding to 1.44–1.53 times the training throughput on this workload.

| Cars | Before, seconds/generation | After, seconds/generation | Less time | Throughput |
|---:|---:|---:|---:|---:|
| 8,192 | 0.386 | 0.267 | 30.7% | 1.44× |
| 32,768 | 1.154 | 0.790 | 31.5% | 1.46× |
| 262,144 | 9.663 | 6.336 | 34.4% | 1.53× |

Turnover alone falls by about 56–58%. These phase medians are calculated separately from the complete-generation medians above.

| Cars | Turnover before, seconds | Turnover after, seconds |
|---:|---:|---:|
| 8,192 | 0.170 | 0.075 |
| 32,768 | 0.661 | 0.291 |
| 262,144 | 5.835 | 2.470 |

Raw timing results are in [baseline-training.json](baseline-training.json) and [optimized-training.json](optimized-training.json). All 36 baseline/optimized generation pairs, including warmups, have identical result digests and executed tick counts. At 256K, the three measured sample averages range from 9.384 to 9.797 seconds before and 6.308 to 6.542 seconds after.

The machine is an RX 7900 XTX with a Ryzen 9 7950X and 8 Rayon threads. Both versions use the same B06 checkpoint population, expanded by repeating its networks for larger populations, and a fresh Python-compatible RNG seeded with 1. The network shape is `[20,16,16,16,16,12,12,8,5]`, or 1,661 parameters per car. The checkpoint population SHA-256 is `1203d453895f86b2292f88c3a8f4c0d741f107d4bebdc1bb98f4fc723760f51a`.

Each sample runs three consecutive generations with wall and idle elimination enabled and a 5,400-tick maximum. At each population size, the baseline runs before the optimized version. Each version has one complete warmup and three measured samples. The reported time is the median of the three samples' average seconds per generation. Both profiling options are off, HIP graphs and split physics are on, and no other benchmark runs concurrently. The desktop session and an unrelated headless game process remain running; this is not an isolated benchmark machine. GPU utilization counters were not measured.

Timing includes driving, host/device state transfers, reproduction, novelty and installation, including turnover after the third generation to represent continued training. Loading the fixture, hashing results and checkpoint/log I/O are outside the timer. Initial runner construction is also outside the timer. These are early-elimination workloads, so the percentage gain will be smaller for generations that spend much longer driving.

The optimized path uploads the next population during turnover, where the baseline uploads it at the start of the next simulation. Compare complete-generation times to account for that change in phase boundaries.

The [baseline profile](baseline-training-32k.log) identified random-number preparation as the largest CPU stall. Uniform draws and their allocation took about 0.30 seconds per 32K-car generation, followed by about 0.07 seconds for Gaussian conversion. Mutation arithmetic itself took about 0.024 seconds. CPU population mean and agent construction with novelty cost about 0.12 seconds combined. This made a GPU mutation kernel alone a less useful first step.

The final implementation makes these changes:

- Python-compatible MT draws process state-sized blocks, preserving the sequence across every twist boundary. The compiler can vectorize tempering and conversion. Gaussian conversion keeps the existing math and cached second value.
- The GPU simulator retains one host parameter buffer. Reproduction uses it for mutation noise, then overwrites it with offspring parameters for upload. This avoids repeated population-sized allocation and initialization. Parent copies also run in parallel when crossover is disabled.
- GPU threads compute each parameter's mean in the original population order. GPU novelty parallelizes the game `pow(x, 2)` calls, then adds their results in the original parameter order. It downloads one novelty value per car.
- Uploaded offspring remain available for the next driving window. Car and agent downloads reuse the upload vectors.
- `ALTD_TRAIN_PROFILE=1` reports CPU stages and GPU transfer timings without changing simulation results. `gpu/benchmark.py --training` measures consecutive generations and checks result digests.

In the [final 32K profile](shared-buffer-training-32k.log), uniform preparation takes about 0.03 seconds and GPU mean plus novelty about 0.023 seconds. Gaussian conversion still takes about 0.07 seconds. The remaining agent construction and vehicle reset cost about 0.06 seconds, and packing/upload about 0.04 seconds. These profiles are diagnostic runs; final comparisons use the separate repeated measurements with profiling disabled.

The retained host buffer uses 8 bytes per population parameter, about 3.24 GiB for 262,144 cars with this network. It is shared between upload and mutation noise and released with the GPU simulator. The GPU adds one mean vector and one double per car for novelty. Full offspring parameters remain on the CPU for checkpointing and the current runner interface.

Accuracy checks include 45,415 CPU/GPU comparisons across `novelty`, `turnover`, `schedules`, `reuse`, `window` and `window3`, with zero mismatches. They cover both RNG backends, every supported crossover and selection algorithm, preserved parents, zero/nonzero mutation, adaptive mutation, weight decay, population resizing and consecutive generations. The [parity log](final-parity.log) records these results. All 35 Rust tests in the library, game-training fixtures and scratch CLI suite pass; see [the test log](rust-tests.log). New RNG tests compare batched draws against scalar draws at every MT state index and compare the entire serialized RNG state.

Another 12,812 comparisons pass with [direct launches](direct-parity.log), and 3,600 each pass on [A07](a07-turnover.log) and [Formula](formula-turnover.log), for 65,427 comparisons in total. The production CLI also passes a [checkpoint/resume comparison](cli-parity.log) with 65 cars and shape `[20,16,5]`: four generations followed by resume through generation six. CPU, GPU, CPU-to-GPU resume and GPU-to-CPU resume produce identical checkpoint metadata, RNG state, population bytes and log metrics after excluding timing fields.

For another performance pass, profile vehicle reset and agent reconstruction first. Moving those to the GPU could avoid more CPU work and state transfer, but must retain the passive reset sequence, vehicle reuse behavior and all imported checkpoint state. GPU Gaussian conversion is another candidate, now worth about 0.07 seconds at 32K, but it needs exact agreement with the CPU transcendental functions and Gaussian cache. Full GPU reproduction would also require selection, mutation and checkpoint ownership to change together. The current measurements support trying those as separate experiments rather than assuming they improve total throughput.

To reproduce the current measurements from `driving_sim_rs`:

```sh
bash gpu/build.sh
cargo build --release --example gpu_check
RAYON_NUM_THREADS=8 python3 gpu/benchmark.py --training \
  --library target/gpu/libaltd_gpu.so --output reports/gpu-training.json
```

The fixture is read from `ALTD_GPU_CKPT`, default `/tmp/altd-gpu-ckpt`. `--binary` and `--library` select the saved baseline executable and library for the earlier version. Raw reports record binary, library and population hashes, every timing, tick count and digest.
