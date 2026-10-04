# WebAssembly and WebGPU

The WASM package exposes `Simulation`, generation checkpoints and optional WebGPU backends. The ordinary package runs the CPU simulator serially. The threaded package uses Rayon workers with shared memory. Expanded native/WASM comparisons contain known last-bit differences; the browser harness checks serial/threaded equality separately. See [fidelity notes](../docs/wasm.md).

## Build

Run from the repository root:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version "$(wasm/bindgen-version.sh)" --locked
wasm/build.sh
```

`wasm-bindgen` must match Cargo.lock, currently 0.2.129. Set `CARGO_NET_OFFLINE=true` when dependencies are cached. Ordinary builds use `--no-default-features --features wasm` and write `wasm/pkg/`. The script stages a complete package before replacing earlier output and records build settings in `features.json`.

```sh
rustup toolchain install "$(cat wasm/thread-toolchain)" --component rust-src --target wasm32-unknown-unknown
THREADS=1 wasm/build.sh
```

Threaded builds use scoped nightly-2025-11-15, rebuild `std` and `panic_abort` with atomics/bulk memory, import shared memory and write `wasm/pkg-threads/`. The memory32 growth ceiling is 2 GiB. `TARGET=web` is required for threads. For ordinary builds, `TARGET=nodejs`, `bundler` or `deno` selects other binding targets. `PROFILE=dev`, `OUT` and `THREAD_TARGET_DIR` customize output. If available, `wasm-opt` applies `-O3` and the required thread/bulk-memory flags; Binaryen 130 was tested. Test-only exports require `TEST_THREADS=1`.

`python3 -m pip install -r wasm/requirements-sim.txt` supplies pycparser 2.23 for regenerating the full simulation shader with `python3 wasm/gen-sim.py`. That tool reads HIP source, checks table freshness and writes `src/wasm/sim/generated.wgsl` plus its Rust world encoder. Use `--output-dir target/generated-sim` to generate both files elsewhere, or `--check` to verify the selected directory without writing files. Shader generation is a developer action, not a build prerequisite.

## Initialize a simulation

The checked-in scenes below come from fixed random-generator recipes. Campaign geometry is not needed.

```js
import init, { Simulation } from "./wasm/pkg/altd_sim.js";
await init();
const scenario = await (await fetch("wasm/test/scenario.json")).json();
const [scene, network, model] = await Promise.all(
  [scenario.scene, scenario.network, scenario.model].map(async path =>
    (await fetch(path)).text()),
);
const sim = new Simulation(scene, network, model, scenario.options);
sim.startWithShape(Uint32Array.from(scenario.shape));
sim.advanceGeneration(600);
console.log(sim.carStates(), sim.generationSummary());
sim.nextGeneration();
```

Options accept an object or JSON string. Keys include `population`, `seed`, `batchCount`, `statsPhase`, `mode`, `spawn`, `settings`, `eliminateOnWall`, `eliminateWhenIdle` and `gpuVerifyEvery`. Omitted spawn uses the scene's reset pose. `start()` uses exported weights; `startWithShape()` creates Xavier networks. `advance()` runs a bounded tick window, `advanceGeneration()` completes the remaining generation and `nextGeneration()` breeds/reset cars. `sensors(i)`, `controls(i)`, `metrics(i)`, `carStates()` and `generationSummary()` export observations. Binary `checkpointBytes()` and `restoreCheckpointBytes()` preserve generation-boundary bits and RNG state.

Run simulation in a coordinator worker to keep the page responsive. Shared-memory packages additionally require HTTPS or localhost with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`. CSP must allow `worker-src 'self' blob:` and `script-src 'self' 'wasm-unsafe-eval'`. Keep every generated `snippets/` helper alongside the module.

```js
// Inside a coordinator worker, before constructing worlds or using Rayon.
const lib = await import("./wasm/pkg-threads/altd_sim.js");
await lib.default({ module_or_path: "./wasm/pkg-threads/altd_sim_bg.wasm" });
await lib.initThreadPool(4);
console.log(lib.cpuThreadCount());
```

The pool lasts for the coordinator's lifetime. Terminate the coordinator and children to stop it. Discard failed pool initialization before retrying a serial package in a fresh worker.

## WebGPU

`GpuRaycaster` uploads the BSP tree, casts rays and nearest-wall queries, and leaves inference/physics on the CPU. `advanceGenerationWithGpuRays()` uses that raycaster for a complete generation. One readback per tick can make it slower for small populations.

`GpuSimulation` runs sensors, inference, statistics and physics in WGSL, retaining geometry and network buffers between windows. It uses software binary64 and ordered network accumulation. Rust still owns evolution and checkpoints.

```js
import { requestGpuDevice, GpuSimulation } from "./wasm/pkg/altd_sim.js";
const device = await requestGpuDevice(true);
const gpu = await GpuSimulation.create(device, sim);
await gpu.verify(12);
await gpu.advanceGeneration(600);
sim.nextGeneration();
gpu.free();
device.destroy();
```

Await GPU windows before changing the Session. A failed window retains the last committed WASM state for CPU continuation. Full GPU simulation rejects paused physics and native shared broadphase; it supports layer widths and input counts up to 64 and requires Curve2D control points for path sensors. Run device verification on each browser/adapter. [WASM fidelity](../docs/wasm.md) records device arithmetic limits; [performance](../docs/performance.md) distinguishes CPU worker scaling from WebGPU timings.

## Browser checks

Install Playwright with Chromium, locally or globally. The runner serves this repository with isolation headers and exits nonzero on comparison failures.

```sh
NO_GPU=1 node wasm/test/run.mjs
GPU=hardware node wasm/test/run.mjs
for track in rally_asphalt rally_mixed rally_ice; do
  NO_GPU=1 SCENARIO="wasm/test/scenarios/$track.json" \
    REFERENCE="wasm/test/scenarios/$track-reference.json" node wasm/test/run.mjs
done
```

`GPU=software` is the default and selects SwiftShader. `GPU=hardware` requires a physical Vulkan adapter and reports its identity. `NO_GPU=1` explicitly skips device checks. `CHROMIUM` selects an executable; `VERIFY_RAYS`, `VERIFY_POINTS`, `VERIFY_SEED` and `TICK_CHECK` expand ray/tick coverage. `PAGE=full-sim.html VERIFY_SIM=1 GPU=hardware` checks full GPU simulation. Shader validation can run without a GPU using Naga 30.0.1 and `node wasm/test/validate-shader.mjs`; `NAGA` selects its executable.

```sh
cargo run --release --offline --example wasm_reference -- wasm/test/scenario.json target/native-reference.json
cargo run --release --offline --example webgpu_division_reference -- target/webgpu-division.bin
GPU=hardware DIVISION_FIXTURE=target/webgpu-division.bin node wasm/test/run.mjs
CARGO_NET_OFFLINE=true wasm/test/build-threads.sh
NO_GPU=1 PAGE=threads.html THREADS=2 node wasm/test/run.mjs
```

The thread builder writes diagnostic packages under `target/wasm-thread-test/` and a native expanded reference under `target/`. It verifies shared memory and actual workers, serial/threaded observations through three turnovers in both modes, and checkpoint exchange. `STRICT_NATIVE=1` additionally requires exact expanded native parity and currently exposes the documented differences. `BENCHMARK=1` selects measurements instead of parity. `PACKAGE_BASE` selects a package pair; without diagnostic exports worker-index checks are omitted.

Regenerate all four public scenes, spawn poses, benchmark network and native state references with `cargo run --release --offline --example wasm_reference -- --generate-fixtures`. These are current-code regression fixtures, so review their diffs before accepting new observations. [Fixture recipes](test/README.md) describe their provenance.
