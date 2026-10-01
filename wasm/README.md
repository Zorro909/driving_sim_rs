# WebAssembly library (`altd_sim` for the browser)

The crate compiles to `wasm32-unknown-unknown` as a callable library: `src/wasm` exports the simulator to JavaScript through wasm-bindgen, `src/wasm/gpu.rs` exposes a raycaster, and `src/wasm/sim` runs full training windows on WebGPU. There is no application in the module; a page or worker creates simulations, advances them and reads their state. The CLI, the HIP GPU loader and the AVX2 kernel are left out of this target.

The tested CPU scenario matches the native build exactly: `wasm/test/run.mjs` compares a 32-car scenario across generation turnover with a native reference (`wasm/test/reference.json`) value for value.

## Build

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version "$(wasm/bindgen-version.sh)"   # must match Cargo.lock
wasm/build.sh                                                           # writes wasm/pkg
```

`wasm/build.sh` builds the ordinary package at `wasm/pkg` with the normal Rust toolchain and precompiled standard library. `TARGET=nodejs|bundler|deno` and `PROFILE=dev` remain supported for this package.

The threaded package uses its own Cargo target directory and the dated nightly in `wasm/thread-toolchain`. Install its components once:

```sh
rustup toolchain install nightly-2025-11-15 --component rust-src --target wasm32-unknown-unknown
THREADS=1 wasm/build.sh                         # writes wasm/pkg-threads
```

Both packages use `--locked`, the same profile, and `wasm-bindgen --target web`. The script rejects other bindgen targets for threads. `features.json` records the source fingerprint, thread capability, toolchain, optimizer, profile, binding version, and memory ceiling. Publish packages with matching source, profile and binding version. `OUT` changes the package destination; the script replaces it only when it is empty or holds an earlier package. `THREAD_TARGET_DIR` changes the threaded Cargo directory.

The threaded command rebuilds `std` and `panic_abort` with atomics and bulk memory. It imports shared memory, exports the TLS metadata required by wasm-bindgen, and keeps `web_sys_unstable_apis`. These settings apply only to that build. Memory stays memory32, with a 1 GiB growth ceiling and abort panics. Extra workers each need about 2 MiB of stack plus TLS; the page's preview module has separate memory.

Release optimization uses `wasm-opt -O3`, with `--enable-threads --enable-bulk-memory` for shared memory. The tested optimizer is Binaryen 130. The validated combination is nightly-2025-11-15, wasm-bindgen 0.2.129, wasm-bindgen-rayon 1.3.0 with `no-bundler`, and Binaryen 130. The adapter prints a warning about positional initializer arguments in its child workers; that published helper still initializes correctly with this binding version.

`TEST_THREADS=1` enables the test-only `testThreadIndices` diagnostic; do not distribute those packages.

`.cargo/config.toml` limits `-C target-cpu=native` to x86 hosts and passes `--cfg=web_sys_unstable_apis` to the wasm target, which web-sys still needs for its WebGPU bindings.

## Use

```js
import init, {
  Simulation,
  GpuRaycaster,
  requestGpuDevice,
} from "./wasm/pkg/altd_sim.js";
await init();

const [scene, network, model] = await Promise.all(
  [
    "scenes_exact/rally_b06_scene.json",
    "network.json",
    "rally_trained_model_exact.json",
  ].map(async (path) => (await fetch(path)).text()),
);

const sim = new Simulation(scene, network, model, {
  population: 256,
  seed: 1,
  batchCount: 1,
  eliminateOnWall: true,
  eliminateWhenIdle: true,
  spawn: { position: [3840, 3968], rotation: 1.5707963705062866 },
  settings: {
    selection_size: 20,
    preserve_parents_size: 4,
    mutation_rate: 0.2,
    weight_decay: 0,
  },
});
sim.startWithShape(Uint32Array.from([20, 16, 16, 16, 16, 12, 12, 8, 5])); // or sim.start() from the export's weights

for (let generation = 0; generation < 100; generation++) {
  sim.advanceGeneration(5400); // 90 s of game time under the game's time limit
  console.log(generation, sim.bestLap(), sim.metrics(0));
  sim.nextGeneration();
}
```

The files are those the CLI takes: a scene export, a network export with `inputs` and `outputs` (weights optional) and the sensor model.

### Options

An object or JSON string; every key is optional (`src/session.rs` `SessionOptions`).

| Key                                    | Default                                 | Meaning                                                                |
| -------------------------------------- | --------------------------------------- | ---------------------------------------------------------------------- |
| `population`                           | `settings.population` or 32             | Cars per generation                                                    |
| `seed`                                 | 0                                       | Training RNG seed (selection, mutation, Xavier networks)               |
| `batchCount`                           | 1                                       | `--batch-count`: inference batches per tick divisor                    |
| `statsPhase`                           | 0                                       | Statistics tick phase                                                  |
| `eliminateOnWall`, `eliminateWhenIdle` | false                                   | `--eliminate-on-wall`, `--idle-eliminate`                              |
| `spawn`                                | scene `reset_position`/`reset_rotation` | `{position: [x, y], rotation}`                                         |
| `mode`                                 | `"independent"`                         | `"independent"` or `"lockstep"`                                        |
| `settings`                             | CLI defaults                            | Evolution settings as `--settings` reads them                          |
| `gpuVerifyEvery`                       | 0                                       | WebGPU: cast every n-th sensor ray on the CPU too and count mismatches |

### `Simulation`

| Member                                                                                                              | Effect                                                                                                                                                                                                                                               |
| ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `start()`, `startWithShape(Uint32Array)`                                                                            | Install the first generation from the export's weights, or from a Xavier network of the shape                                                                                                                                                        |
| `advance(ticks, stopWhenInactive)`                                                                                  | Up to `ticks` physics ticks (60/s); returns the executed count                                                                                                                                                                                       |
| `advanceGeneration(timeLimitTicks)`                                                                                 | The rest of the generation under the game's time limit                                                                                                                                                                                               |
| `nextGeneration()`                                                                                                  | Reproduce and reset: `{preservedCount, rewards}`                                                                                                                                                                                                     |
| `setEvolutionSettings(json)`                                                                                        | Replace validated evolution settings for the next reproduction, preserving cars, statistics and RNG. Population changes take effect at turnover; omitted fields use the engine defaults.                                                             |
| `generation`, `tick`, `population`, `started`, `busy`                                                               | Getters                                                                                                                                                                                                                                              |
| `carStates()`                                                                                                       | `Float64Array`, `Simulation.carStateStride()` values per car in `Simulation.carStateFields()` order: position, rotation, velocity, angular velocity, active, score, laps, best lap (-1 without one), wall contacts, boost, wheel angle, update count |
| `metrics(i)`, `Simulation.metricNames()`                                                                            | The 14 training metrics (NaN when unset)                                                                                                                                                                                                             |
| `sensors(i)`, `sensorNames()`                                                                                       | The inputs a car reads now                                                                                                                                                                                                                           |
| `controls(i)`, `outputNames()`                                                                                      | `[acceleration, steering, brake, handbrake, boost]`                                                                                                                                                                                                  |
| `networkJson(i)`, `setNetworkJson(i, json)`                                                                         | A car's network as `{shape, weights, biases}`                                                                                                                                                                                                        |
| `checkpointJson()`, `restoreCheckpointJson(json)`                                                                   | Legacy JSON generation-boundary checkpoints                                                                                                                                                                                                          |
| `checkpointBytes()`, `restoreCheckpointBytes(Uint8Array)`, `checkpointGeneration`, `checkpointTick`                 | Binary generation-boundary checkpoints retaining exact float and RNG bits                                                                                                                                                                            |
| `generationSummary()`, `activeCount`                                                                                | Compact leader, score, lap, and activity statistics                                                                                                                                                                                                  |
| `bestLap()`                                                                                                         | `{index, time}` or null                                                                                                                                                                                                                              |
| `setPaused(bool)`                                                                                                   | GameManager.OnPause                                                                                                                                                                                                                                  |
| `trackWalls()`, `trackPath()`, `trackBounds()`, `spawn()`                                                           | Geometry for drawing                                                                                                                                                                                                                                 |
| `rayCount()`                                                                                                        | Ray sensors per car                                                                                                                                                                                                                                  |
| `advanceWithGpuRays(raycaster, ticks, stopWhenInactive)`, `advanceGenerationWithGpuRays(raycaster, timeLimitTicks)` | The same windows with the ray sensors cast on WebGPU; promises of the executed ticks                                                                                                                                                                 |
| `gpuRaysChecked()`, `gpuRayMismatches()`                                                                            | Counters of `gpuVerifyEvery`                                                                                                                                                                                                                         |

Errors from options, indices and shapes reject as JavaScript errors. Invalid scene, network or model files hit the library's own assertions, which abort the module; the console panic hook prints the message. A `Simulation` runs on its calling coordinator. The ordinary package runs Rayon inline. The threaded package dispatches its existing Rust loops to the initialized pool.

### Threaded initialization

Serve the document, modules and worker scripts with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` over HTTPS or localhost. CSP must allow `worker-src 'self' blob:` and `script-src 'self' 'wasm-unsafe-eval'`. Keep the complete `snippets` directory with each generated module.

Initialize inside a dedicated coordinator worker, before track conversion, world construction or any other Rayon operation:

```js
const lib = await import("./pkg-threads/altd_sim.js");
await lib.default({ module_or_path: "./pkg-threads/altd_sim_bg.wasm" });
await lib.initThreadPool(4);
console.log(lib.cpuThreadCount()); // 4, query only after initialization
// Now construct Simulation, start or restore it, and optionally verify WebGPU.
```

The global pool is fixed for the worker's lifetime. Pausing can retain it; stopping requires terminating the coordinator and its children. A failed or timed-out pool must be discarded before loading the ordinary package in a fresh coordinator. Do not run simulation calls on a partially initialized instance.

The webapp probes WASM threads and isolation, reserves one reported logical CPU for the coordinator/page, and caps the pool at eight. A missing or invalid hardware-concurrency value defaults to two logical CPUs, so the pool gets one worker. The `start` command's test override accepts 1 through 64 workers without changing saved runs. WebGPU uses the same initialized CPU pool for evolution, verification and CPU recovery.

### Full WebGPU training

```js
import { GpuSimulation } from "./wasm/pkg/altd_sim.js";
const device = await requestGpuDevice(true); // require a physical adapter
const gpu = await GpuSimulation.create(device, sim);
await gpu.verify(12); // compare against WASM, then restore the initial state
await gpu.advance(120, true);
await gpu.advanceGeneration(5400);
sim.nextGeneration(); // Rust evolution invalidates and replaces GPU networks
// Reuse gpu for the next generation. Await windows before mutating sim.
gpu.free();
device.destroy();
```

Sensing, inference, statistics, vehicle forces, and collision resolution run
in WGSL. Networks and static world data persist on the GPU. Advancement uses
one readback per bounded window, not per tick; longer requests split at 120
ticks. Each neuron has a workgroup lane, with ordered accumulation and integer
software binary64 arithmetic. Rust owns evolution and checkpoints.

`verify` accepts 1 to 60 ticks and keeps comparisons inside Rust. It checks
canonical car and agent state and restores the initial session on success or
failure. CPU mutations invalidate GPU uploads. Concurrent mutations reject
while `sim.busy` is true. A failed window retains the last committed WASM
state so callers can continue on CPU.

The backend supports independent cars with physics shapes and a BSP tree,
layer widths and input counts up to 64, and the native GPU mirror's contact
and wheel capacities. Paused physics and native shared broadphase reject.
Path sensors require exported Curve2D control points. `requestGpuDevice(true)`
rejects a software adapter; omit its argument for the older raycaster API.

CPU is faster for small populations on the tested machine. The 4,096-car
Formula benchmark ran about 2.7 times faster on an RX 7900 XTX. By default
the GPU is kept busy for at most 85 percent of wall time, in submissions of
about 4 ms, so the desktop stays responsive; `sim.setLoad(load)` changes the
share within (0, 1]. Cold
compilation can take a minute. See [measurements, exactness checks, and
reproduction commands](../../driving_webapp_public/TRAINING_PERFORMANCE.md).

### Raycaster API

```js
const device = await requestGpuDevice(); // or the host's own GPUDevice
const raycaster = await GpuRaycaster.create(device, sim); // compiles rays.wgsl, uploads the BSP tree
console.log(await raycaster.verify(sim, 4000, 4000, 7)); // {rays, rayMismatches, points, pointMismatches}
await sim.advanceGenerationWithGpuRays(raycaster, 5400);
const hits = await raycaster.raycast(Float32Array.from([x0, y0, x1, y1])); // [hitX, hitY, hit, 0] per ray
const near = await raycaster.closestWall(Float32Array.from([x, y, 0, 0])); // [x, y, 1, 0] per point
```

`src/wasm/rays.wgsl` is a port of the HIP `gpu/sim/rays.h`: the BSP raycast and closest-wall queries of `src/bsp.rs` as a compute shader, with the CPU's visiting order and operation order. A GPU window runs tick-major (`TrainingRunner::begin_ray_window` in `src/training.rs`): each tick the statistics run on the CPU, the ray sensors of the cars due for inference are cast in one dispatch, and inference and the physics step follow on the CPU. Natively, `training::tests::ray_window_matches_advance` shows this window reproduces `advance` in both modes.

The shader uses `src/wasm/float.wgsl` to reproduce the CPU's float32 rounding. Each addition, subtraction and multiplication passes through a bitcast and an XOR with a uniform zero, preventing fusion or reassociation across that boundary. Division operates on integer significands and rounds once to nearest, ties to even, including subnormal results. This avoids WGSL's approximate division (2.5 ULP): the hardware quotient is only an estimate, corrected with the exact integer remainder, and a bit-by-bit loop takes over when a bound on the estimate cannot prove the remainder exact. The simulator's `fp_sqrt` and binary64 `f64_div` work the same way; `PAGE=exact.html` compares them with the bit loops, and the branch-free binary64 addition, subtraction and multiplication with the SoftFloat code in `wasm/test/f64-reference.wgsl`. The RX 7900 XTX now matches the native reference in the tested simulation scenarios. `verify` and `gpuVerifyEvery` still compare GPU results with the CPU and count differences without replacing the GPU results. This raycaster API leaves physics, statistics, and inference on the CPU. The full training API above instead uses software binary64 in WGSL.

These tests cover finite scene coordinates on the tested hardware. WGSL may flush subnormal inputs or results of addition and multiplication, and NaN payload propagation is not standardized. The division regression checks subnormal and signed-zero results as raw bits, but treats NaN payloads as equivalent. Results on other devices still need verification.

A GPU window waits for one buffer read-back per tick, which costs about a millisecond of latency; with a few dozen cars that is slower than casting the rays on the CPU. It pays off for populations whose rays per tick outweigh the round trip, and it demonstrates the tick-major window an external raycaster drives.

## Test

Validate the raycaster and simulation shaders with Naga, the shader frontend used by Firefox, and translate them to SPIR-V, without a GPU:

```sh
cargo install naga-cli --version 30.0.1 --locked
node wasm/test/validate-shader.mjs
```

Set `NAGA=/path/to/naga` to use an executable outside `PATH`. This catches return-path validation failures that Chromium accepts. Both ray traversal functions exit their outer loops before returning their terminal results so that Naga sees an explicit value at the end of each function.

```sh
cargo run --release --example wasm_reference     # native reference for wasm/test/scenario.json
wasm/build.sh
node wasm/test/run.mjs                           # needs Playwright with Chromium (npm i -g playwright)
GPU=hardware node wasm/test/run.mjs              # real Vulkan adapter required
NO_GPU=1 node wasm/test/run.mjs                  # CPU comparison only
```

`run.mjs` serves the repository root, opens `wasm/test/index.html` in headless Chromium, and prints the page's report: the CPU scenario against the native reference, then the raycaster's `verify` and the same scenario driven with GPU rays against the CPU run, all compared bit for bit. The default `GPU=software` explicitly uses SwiftShader. `GPU=hardware` uses Vulkan and rejects fallback adapters. Reports include the adapter, browser version and, in hardware mode, Chromium's GPU device information. Missing WebGPU and query failures fail the test. `CHROMIUM=/path/to/chrome` overrides the browser; `NO_GPU=1` explicitly skips GPU checks.

The original hardware failures and the fix are recorded in [the initial validation report](../reports/wasm-webgpu-20260930/README.md) and [the exactness report](../reports/webgpu-exact-20260930/README.md). The corrected shader passes on the RX 7900 XTX with Mesa 26.2.2.

For broader checks, `SCENARIO` and `REFERENCE` select paths relative to the repository root. `VERIFY_RAYS`, `VERIFY_POINTS` and `VERIFY_SEED` control random query verification. `TICK_CHECK=N` additionally compares every tick for three generations of N ticks in each scheduler mode, including all exported states, controls, sensors, metrics and generation checkpoints. For example:

```sh
cargo run --release --example webgpu_division_reference
GPU=hardware DIVISION_FIXTURE=target/webgpu-division.bin \
  VERIFY_RAYS=65536 VERIFY_POINTS=65536 TICK_CHECK=400 node wasm/test/run.mjs

for track in a01 a07 b06; do
  GPU=hardware SCENARIO="wasm/test/scenarios/$track.json" \
    REFERENCE="wasm/test/scenarios/$track-reference.json" \
    VERIFY_RAYS=65536 VERIFY_POINTS=65536 TICK_CHECK=300 node wasm/test/run.mjs
done
```

The division fixture contains 1,049,732 native results, including signed zero, subnormal boundaries, infinities, and random operands across all exponent ranges. `DIVISION_FIXTURE` makes the browser test the production division helper against those raw bit patterns. The added Rally scenarios use 33 cars, different batch counts and elimination settings, and a larger network; their checked-in references come from `wasm_reference`.

### Threaded integration and measurements

```sh
PATH=/path/to/binaryen-130/bin:$PATH wasm/test/build-threads.sh
NO_GPU=1 PAGE=threads.html THREADS=2 node wasm/test/run.mjs
NO_GPU=1 PAGE=threads.html THREADS=4 BROWSER=firefox node wasm/test/run.mjs
GPU=hardware PAGE=threads.html THREADS=2 node wasm/test/run.mjs
NO_GPU=1 PAGE=threads.html BENCHMARK=1 THREADS=8 node wasm/test/run.mjs
```

The test builder writes optimized diagnostic packages under `target/wasm-thread-test`, leaving distributable packages alone, and writes a native reference under `target`. The worker checks shared memory, actual child workers and distinct Rayon worker indices. `PACKAGE_BASE=target/public-packages` can select a separately built package pair without diagnostic exports; indices are then omitted while full serial/threaded comparisons still run. It compares states, sensors, controls, metrics, summaries, rewards and checkpoint bytes through three turnovers in both scheduling modes, including serial/threaded checkpoint exchange.

The expanded native comparison exposes existing last-bit native/serial differences, while threaded and serial WASM match exactly. Reports contain both comparisons; `STRICT_NATIVE=1` treats any native difference as a failure. The older checked-in native state reference still passes. See [the implementation validation and benchmarks](../reports/wasm-rayon-20261001/README.md) for exact counts and limits. No floating-point or RNG algorithms changed in the threading integration.

## Portability notes

- `native_math::engine_sin_cos` reproduces the shipped Godot's x87 `FSINCOS` with inline assembly on x86. Other targets use `engine_sin_cos_portable`, the fdlibm kernels of the HIP port; it equals the instruction for every float with |x| ≤ 16 (`native_math::engine_tests` checks a sample in the suite and the whole domain with `cargo test --release --lib native_math -- --ignored`). Larger arguments, which the game's rotations do not reach, are unchecked.
- `f64` transcendental functions (`atan2` in the track curvature sensor, `ln`/`sin`/`cos` in the Python-compatible Gaussian draws) come from Rust's bundled libm on wasm instead of the host C library. The test scenario's two generations match the native run exactly; a rare last-bit difference between the two libraries remains possible.
- `std::time::Instant` and `std::env` are unavailable; `ALTD_TRAIN_PROFILE` timings are disabled on this target.
