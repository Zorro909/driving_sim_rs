# GPU simulator (HIP)

`gpu/sim` runs whole training generations on an AMD GPU, bit-exact with the CPU simulator. `altd-sim train-scratch --gpu` produces the same log, best-lap networks, checkpoints and `run.json` as a CPU run with the same arguments, so CPU and GPU runs can resume each other.

The library is loaded at runtime with `dlopen` (`src/gpu.rs`), so the crate builds and runs without ROCm. Only `--gpu` and the GPU examples need it.

## Build

```sh
gpu/build.sh    # writes target/gpu/libaltd_gpu.so; OUT=<dir> overrides
```

`build.sh` runs `gen_tables.py`, which regenerates `sim/double_tables.h` from `src/double_math_tables.rs`. It then compiles `sim/altd_gpu.hip` with `hipcc --offload-arch=gfx1100` (RX 7900 XTX). Two flags are required for exactness:

- `-ffp-contract=off`: hipcc otherwise fuses `a*b+c` into FMA, which rounds once.
- `-fhip-fp32-correctly-rounded-divide-sqrt`: float32 division and square root must round like the CPU.

For another GPU, change `--offload-arch`.

## Use

```sh
target/release/altd-sim train-scratch --gpu --out-dir ../training_runs/b06_gpu
```

`--threads` controls CPU work such as reproduction and packing transfers, including with `--gpu`. `--mode` only affects CPU driving. The GPU runs the tick-major order, which yields the same results as both CPU modes. The GPU runner rejects paused windows and scenes that use the native shared broadphase (some generated tracks).

| Variable | Effect |
|---|---|
| `ALTD_GPU_LIB` | Library path (default `target/gpu/libaltd_gpu.so` of this crate) |
| `ALTD_GPU_PROFILE=1` | GPU time per phase (stats, sensors, forward, step) after each window; `host` prints only host CPU and wall time |
| `ALTD_GPU_GRAPH=0` | Direct kernel launches instead of the per-period HIP graph |
| `ALTD_GPU_SPLIT=0` | Fused step kernel (`car_step`) instead of the split collision step |
| `ALTD_TRAIN_PROFILE=1` | Host timings for reproduction, generation installation and GPU transfers, emitted as JSON on stderr |

## Layout

| File | Content |
|---|---|
| `sim/math.h`, `double_tables.h`, `engine_exceptions.h` | Game math: float/double runtime routines, x87 `FSINCOS` emulation |
| `sim/world.h`, `track.h`, `rays.h` | Track arrays, BSP raycasts, nearest wall and path queries |
| `sim/sensors.h`, `stats.h` | Sensor readings, lap and score statistics |
| `sim/state.h`, `step.h` | Car state, driving, collision and contact solving |
| `sim/altd_gpu.hip` | Kernels, the window scheduler and the C API used by `src/gpu.rs` |
| `src/gpu_sim.rs` | Rust-side layouts and car/agent export and import |
| `src/training.rs` | `gpu_sim`, `advance_generation_gpu`, `advance_window_gpu`, `next_generation_gpu` |

Every tick runs statistics, sensors, network forward and the step as separate kernels. Live cars are compacted within their original eight inference batches. Sensor and forward grids cover only the batches due that tick; cars eliminated since the last compaction are filtered by their active flag. The separate physics list also retains inactive cars that have not settled exactly.

A window replays a HIP graph per 24-tick period, covering the statistics and inference cycles. Compatible windows reuse the executable graph. A changed schedule, population size, network shape or output mapping invalidates it; network weights can be replaced in place. The host polls a stop flag every 48 ticks, while the device applies the exact stopping tick before driving.

### Generation turnover

`next_generation_gpu` uploads the offspring once, computes the population mean and each network's novelty on the GPU, and keeps the weights there for the next driving window. Mean and novelty sums retain the CPU's addition order and game math, including `pow(x, 2)`.

Selection, crossover and mutation still run on the CPU. Python-compatible random draws process whole MT state blocks while preserving the exact stream, cached Gaussian and checkpoint state. Copying parents with crossover disabled runs in parallel. A reusable host buffer holds mutation noise, then packs the offspring for upload. It retains about 3.24 GiB at 262,144 cars with the benchmark's 1,661-parameter networks. Car and agent downloads reuse the corresponding upload allocations.

The CPU owns the installed population and resets vehicles as before, so checkpoint serialization and CPU/GPU resume remain compatible. Rebuild the HIP library along with the Rust executable when updating these APIs.

### Split collision step

The step splits into three kernels so that collision checks run in parallel:

1. `step_begin` (one thread per car): driving, contact validation, shape and broad-pair updates. It reserves one SAT item per broad pair (a wave prefix scan plus one `atomicAdd` per wave).
2. `sat_kernel` (one thread per car–pair item): the separating-axis test of that pair. The result goes to `pushes`.
3. `step_end` (one thread per car): contact generation from the pushes, merge, active set, solver, integration.

The split is exact because each SAT item reads and writes only its own pair's cached axis, and contact finding reads no contacts. `ALTD_GPU_SPLIT=0` composes the same functions in one kernel.

## Verification

```sh
cargo build --release --examples
target/release/examples/gpu_check                  # math primitives (millions of inputs each)
target/release/examples/gpu_check rays sensors infer stats step window
target/release/examples/gpu_check schedules reuse window3
target/release/examples/gpu_check novelty turnover
```

Each part compares every output bit against the CPU and prints `ALL OK`. `window` runs full generations and compares every agent. The defaults use B06. To check another track, set `ALTD_GPU_SCENE`, `ALTD_GPU_SPAWN`, `ALTD_GPU_NETWORK` and `ALTD_GPU_MODEL`, for example:

```sh
ALTD_GPU_SCENE=scenes_exact/autumn_04_formula_scene.json ALTD_GPU_SPAWN=traces/autumn_04_spawn.json \
ALTD_GPU_NETWORK=formula_network_template.json ALTD_GPU_MODEL=formula_trained_model_exact.json \
  target/release/examples/gpu_check step window
```

`gpu_engine_scan` compares the GPU engine sine/cosine with x87 `FSINCOS` for every float with |x| ≤ 16 and regenerates `sim/engine_exceptions.h` from the mismatches. Rebuild the library afterwards.

`schedules` checks partial windows, uneven populations, inference batch wrapping and statistics phases. `reuse` changes population and network shape while retaining the same GPU simulator, and checks graph reuse after replacing network weights.

`novelty` checks the GPU population calculations against CPU results across network shapes and population sizes. `turnover` compares consecutive CPU/GPU generations, offspring parameters and complete RNG states across selection and crossover modes, both RNG backends, mutation settings and population resizing.

## Repeatable benchmarks

```sh
RAYON_NUM_THREADS=8 python3 gpu/benchmark.py \
  --library target/gpu/libaltd_gpu.so --no-elimination \
  --output reports/gpu-full.json
```

The default sizes are 8,192, 32,768 and 262,144 cars, with a 5,400-tick time limit, one warmup and three timed samples. Every sample restores the same population. Timing includes network packing/upload, car and agent export/upload, simulation, download and state import. Fixture loading, resetting, hashing, reproduction and checkpoint I/O are outside the timer. JSON results include each sample, medians, state digests and library/checkpoint hashes. The game's statistics callback stops this time limit at callback 5,406, before its driving step.

The fixture uses `ALTD_GPU_CKPT`, defaulting to `/tmp/altd-gpu-ckpt`, as do the existing window checks. Populations larger than the saved checkpoint repeat its networks. Omit `--no-elimination` to measure early termination as well. `--cars`, `--ticks`, `--samples` and `--warmups` override the defaults.

Use normal graph mode for performance comparisons. `ALTD_GPU_PROFILE=host` preserves it, but `ALTD_GPU_PROFILE=1` disables graphs and adds per-phase events and periodic synchronization.

To include reproduction, installation and transfers between consecutive generations:

```sh
RAYON_NUM_THREADS=8 python3 gpu/benchmark.py --training \
  --library target/gpu/libaltd_gpu.so \
  --output reports/gpu-training.json
```

Training mode runs three consecutive generations per sample with elimination enabled. Each sample starts from the same checkpoint population and RNG seed. Timing includes turnover after every generation, including the last, to represent continued training. Fixture loading, hashing and checkpoint/log I/O remain outside the timer. Digests cover the installed offspring, RNG state, cars and agents. `--generations` changes the chain length; `--binary` selects a saved benchmark executable for comparisons. Leave both profiling variables unset for final timings.

The [generation-overhead report](../reports/gpu-turnover-20260929/README.md) compares complete training generations before and after these changes at all three population sizes.

## Performance (RX 7900 XTX, B06, 5,400 ticks, seeded from a 45 s network)

The [2026-09-29 benchmark report](../reports/gpu-performance-20260929/README.md)
compares the original and optimized implementations at 8K, 32K and 256K cars,
including the attempted optimizations, raw timings and parity checks. The older
measurements below use a different workload and are retained for context.

| Cars | CPU | GPU |
|---|---|---|
| 2,048 | 3.15 s/gen (8 threads) | 1.6 s/gen |
| 8,192 | – | 2.5–2.7 s/gen |

Late in a generation the step is latency-bound, at about 117 µs per tick. Around 10k eliminated cars never come to rest: contact bias and split impulses keep moving their position, rotation and contact depth. So they are stepped every tick, and they cannot be skipped without changing results.
