# WebAssembly library (`altd_sim` for the browser)

The crate compiles to `wasm32-unknown-unknown` as a callable library: `src/wasm` exports the simulator to JavaScript through wasm-bindgen, and `src/wasm/gpu.rs` casts the ray sensors on WebGPU. There is no application in the module; a page or worker creates simulations, advances them and reads their state. The CLI, the HIP GPU loader and the AVX2 kernel are left out of this target.

The tested CPU scenario matches the native build exactly: `wasm/test/run.mjs` compares a 32-car scenario across generation turnover with a native reference (`wasm/test/reference.json`) value for value.

## Build

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version "$(wasm/bindgen-version.sh)"   # must match Cargo.lock
wasm/build.sh                                                           # writes wasm/pkg
```

`wasm/build.sh` runs `cargo rustc --release --lib --crate-type cdylib --target wasm32-unknown-unknown --no-default-features --features wasm`, then `wasm-bindgen --target web` (`TARGET=nodejs|bundler|deno` selects another flavour, `PROFILE=dev` skips optimizations) and `wasm-opt -O3` when it is installed. The output is `wasm/pkg/altd_sim.js`, `altd_sim_bg.wasm` and TypeScript declarations, about 0.8 MB. Native builds emit only `rlib`, avoiding Cargo output collisions between release test and CLI builds.

`.cargo/config.toml` limits `-C target-cpu=native` to x86 hosts and passes `--cfg=web_sys_unstable_apis` to the wasm target, which web-sys still needs for its WebGPU bindings.

## Use

```js
import init, { Simulation, GpuRaycaster, requestGpuDevice } from './wasm/pkg/altd_sim.js';
await init();

const [scene, network, model] = await Promise.all(
    ['scenes_exact/rally_b06_scene.json', 'network.json', 'rally_trained_model_exact.json']
        .map(async (path) => (await fetch(path)).text()));

const sim = new Simulation(scene, network, model, {
    population: 256, seed: 1, batchCount: 1,
    eliminateOnWall: true, eliminateWhenIdle: true,
    spawn: { position: [3840, 3968], rotation: 1.5707963705062866 },
    settings: { selection_size: 20, preserve_parents_size: 4, mutation_rate: 0.2, weight_decay: 0 },
});
sim.startWithShape(Uint32Array.from([20, 16, 16, 16, 16, 12, 12, 8, 5]));   // or sim.start() from the export's weights

for (let generation = 0; generation < 100; generation++) {
    sim.advanceGeneration(5400);          // 90 s of game time under the game's time limit
    console.log(generation, sim.bestLap(), sim.metrics(0));
    sim.nextGeneration();
}
```

The files are those the CLI takes: a scene export, a network export with `inputs` and `outputs` (weights optional) and the sensor model.

### Options

An object or JSON string; every key is optional (`src/session.rs` `SessionOptions`).

| Key | Default | Meaning |
|---|---|---|
| `population` | `settings.population` or 32 | Cars per generation |
| `seed` | 0 | Training RNG seed (selection, mutation, Xavier networks) |
| `batchCount` | 1 | `--batch-count`: inference batches per tick divisor |
| `statsPhase` | 0 | Statistics tick phase |
| `eliminateOnWall`, `eliminateWhenIdle` | false | `--eliminate-on-wall`, `--idle-eliminate` |
| `spawn` | scene `reset_position`/`reset_rotation` | `{position: [x, y], rotation}` |
| `mode` | `"independent"` | `"independent"` or `"lockstep"` |
| `settings` | CLI defaults | Evolution settings as `--settings` reads them |
| `gpuVerifyEvery` | 0 | WebGPU: cast every n-th sensor ray on the CPU too and count mismatches |

### `Simulation`

| Member | Effect |
|---|---|
| `start()`, `startWithShape(Uint32Array)` | Install the first generation from the export's weights, or from a Xavier network of the shape |
| `advance(ticks, stopWhenInactive)` | Up to `ticks` physics ticks (60/s); returns the executed count |
| `advanceGeneration(timeLimitTicks)` | The rest of the generation under the game's time limit |
| `nextGeneration()` | Reproduce and reset: `{preservedCount, rewards}` |
| `generation`, `tick`, `population`, `started`, `busy` | Getters |
| `carStates()` | `Float64Array`, `Simulation.carStateStride()` values per car in `Simulation.carStateFields()` order: position, rotation, velocity, angular velocity, active, score, laps, best lap (-1 without one), wall contacts, boost, wheel angle, update count |
| `metrics(i)`, `Simulation.metricNames()` | The 14 training metrics (NaN when unset) |
| `sensors(i)`, `sensorNames()` | The inputs a car reads now |
| `controls(i)`, `outputNames()` | `[acceleration, steering, brake, handbrake, boost]` |
| `networkJson(i)`, `setNetworkJson(i, json)` | A car's network as `{shape, weights, biases}` |
| `checkpointJson()`, `restoreCheckpointJson(json)` | Generation-boundary checkpoints: generation, RNG state, networks |
| `bestLap()` | `{index, time}` or null |
| `setPaused(bool)` | GameManager.OnPause |
| `trackWalls()`, `trackPath()`, `trackBounds()`, `spawn()` | Geometry for drawing |
| `rayCount()` | Ray sensors per car |
| `advanceWithGpuRays(raycaster, ticks, stopWhenInactive)`, `advanceGenerationWithGpuRays(raycaster, timeLimitTicks)` | The same windows with the ray sensors cast on WebGPU; promises of the executed ticks |
| `gpuRaysChecked()`, `gpuRayMismatches()` | Counters of `gpuVerifyEvery` |

Errors from options, indices and shapes reject as JavaScript errors. Invalid scene, network or model files hit the library's own assertions, which abort the module; the console panic hook prints the message. A `Simulation` runs on the calling thread (a worker keeps a page responsive); rayon's parallel loops run inline on this target.

### WebGPU

```js
const device = await requestGpuDevice();                  // or the host's own GPUDevice
const raycaster = await GpuRaycaster.create(device, sim); // compiles rays.wgsl, uploads the BSP tree
console.log(await raycaster.verify(sim, 4000, 4000, 7));  // {rays, rayMismatches, points, pointMismatches}
await sim.advanceGenerationWithGpuRays(raycaster, 5400);
const hits = await raycaster.raycast(Float32Array.from([x0, y0, x1, y1]));   // [hitX, hitY, hit, 0] per ray
const near = await raycaster.closestWall(Float32Array.from([x, y, 0, 0]));   // [x, y, 1, 0] per point
```

`src/wasm/rays.wgsl` is a port of the HIP `gpu/sim/rays.h`: the BSP raycast and closest-wall queries of `src/bsp.rs` as a compute shader, with the CPU's visiting order and operation order. A GPU window runs tick-major (`TrainingRunner::begin_ray_window` in `src/training.rs`): each tick the statistics run on the CPU, the ray sensors of the cars due for inference are cast in one dispatch, and inference and the physics step follow on the CPU. Natively, `training::tests::ray_window_matches_advance` shows this window reproduces `advance` in both modes.

The shader uses `src/wasm/float.wgsl` to reproduce the CPU's float32 rounding. Each addition, subtraction and multiplication passes through a bitcast and an XOR with a uniform zero, preventing fusion or reassociation across that boundary. Division operates on integer significands and rounds once to nearest, ties to even, including subnormal results. This avoids WGSL's approximate division. The RX 7900 XTX now matches the native reference in the tested simulation scenarios. `verify` and `gpuVerifyEvery` still compare GPU results with the CPU and count differences without replacing the GPU results. The rest of the simulator, including double-precision physics, statistics and inference, stays on the CPU because WGSL has no 64-bit floats.

These tests cover finite scene coordinates on the tested hardware. WGSL may flush subnormal inputs or results of addition and multiplication, and NaN payload propagation is not standardized. The division regression checks subnormal and signed-zero results as raw bits, but treats NaN payloads as equivalent. Results on other devices still need verification.

A GPU window waits for one buffer read-back per tick, which costs about a millisecond of latency; with a few dozen cars that is slower than casting the rays on the CPU. It pays off for populations whose rays per tick outweigh the round trip, and it demonstrates the tick-major window an external raycaster drives.

## Test

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

## Portability notes

- `native_math::engine_sin_cos` reproduces the shipped Godot's x87 `FSINCOS` with inline assembly on x86. Other targets use `engine_sin_cos_portable`, the fdlibm kernels of the HIP port; it equals the instruction for every float with |x| ≤ 16 (`native_math::engine_tests` checks a sample in the suite and the whole domain with `cargo test --release --lib native_math -- --ignored`). Larger arguments, which the game's rotations do not reach, are unchecked.
- `f64` transcendental functions (`atan2` in the track curvature sensor, `ln`/`sin`/`cos` in the Python-compatible Gaussian draws) come from Rust's bundled libm on wasm instead of the host C library. The test scenario's two generations match the native run exactly; a rare last-bit difference between the two libraries remains possible.
- `std::time::Instant` and `std::env` are unavailable; `ALTD_TRAIN_PROFILE` timings are disabled on this target.
