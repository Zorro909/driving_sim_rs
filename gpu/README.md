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

`--threads` and `--mode` have no effect with `--gpu`. The GPU runs the tick-major order, which yields the same results as both CPU modes. The GPU runner rejects paused windows and scenes that use the native shared broadphase (some generated tracks).

| Variable | Effect |
|---|---|
| `ALTD_GPU_LIB` | Library path (default `target/gpu/libaltd_gpu.so` of this crate) |
| `ALTD_GPU_PROFILE=1` | GPU time per phase (stats, sensors, forward, step) after each window; `host` prints only host CPU and wall time |
| `ALTD_GPU_GRAPH=0` | Direct kernel launches instead of the per-period HIP graph |
| `ALTD_GPU_SPLIT=0` | Fused step kernel (`car_step`) instead of the split collision step |

## Layout

| File | Content |
|---|---|
| `sim/math.h`, `double_tables.h`, `engine_exceptions.h` | Game math: float/double runtime routines, x87 `FSINCOS` emulation |
| `sim/world.h`, `track.h`, `rays.h` | Track arrays, BSP raycasts, nearest wall and path queries |
| `sim/sensors.h`, `stats.h` | Sensor readings, lap and score statistics |
| `sim/state.h`, `step.h` | Car state, driving, collision and contact solving |
| `sim/altd_gpu.hip` | Kernels, the window scheduler and the C API used by `src/gpu.rs` |
| `src/gpu_sim.rs` | Rust-side layouts and car/agent export and import |
| `src/training.rs` | `gpu_sim`, `advance_generation_gpu`, `advance_window_gpu` |

Every tick runs statistics, sensors, network forward and the step as separate kernels over the live cars. Each kernel is written with as little branching as possible, for example predicated loops over fixed maxima. A window runs as a HIP graph per 6-tick period. The host only reads back a stop flag.

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
```

Each part compares every output bit against the CPU and prints `ALL OK`. `window` runs full generations and compares every agent. The defaults use B06. To check another track, set `ALTD_GPU_SCENE`, `ALTD_GPU_SPAWN`, `ALTD_GPU_NETWORK` and `ALTD_GPU_MODEL`, for example:

```sh
ALTD_GPU_SCENE=scenes_exact/autumn_04_formula_scene.json ALTD_GPU_SPAWN=traces/autumn_04_spawn.json \
ALTD_GPU_NETWORK=formula_network_template.json ALTD_GPU_MODEL=formula_trained_model_exact.json \
  target/release/examples/gpu_check step window
```

`gpu_engine_scan` compares the GPU engine sine/cosine with x87 `FSINCOS` for every float with |x| ≤ 16 and regenerates `sim/engine_exceptions.h` from the mismatches. Rebuild the library afterwards.

## Performance (RX 7900 XTX, B06, 5,400 ticks, seeded from a 45 s network)

| Cars | CPU | GPU |
|---|---|---|
| 2,048 | 3.15 s/gen (8 threads) | 1.6 s/gen |
| 8,192 | – | 2.5–2.7 s/gen |

Late in a generation the step is latency-bound, at about 117 µs per tick. Around 10k eliminated cars never come to rest: contact bias and split impulses keep moving their position, rotation and contact depth. So they are stepped every tick, and they cannot be skipped without changing results.
