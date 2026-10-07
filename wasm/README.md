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

Options accept an object or JSON string. Keys include `population`, `seed`, `batchCount`, `statsPhase`, `mode`, `spawn`, `settings`, `mathProfile`, `eliminateOnWall`, `eliminateWhenIdle` and `gpuVerifyEvery`. Omitted spawn uses the scene's reset pose. `start()` uses exported weights; `startWithShape()` creates Xavier networks. `advance()` runs a bounded tick window, `advanceGeneration()` completes the remaining generation and `nextGeneration()` breeds/reset cars. `sensors(i)`, `controls(i)`, `metrics(i)`, `carStates()` and `generationSummary()` export observations. Binary `checkpointBytes()` and `restoreCheckpointBytes()` preserve generation-boundary bits and RNG state.

Run simulation in a coordinator worker to keep the page responsive. Shared-memory packages additionally require HTTPS or localhost with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`. CSP must allow `worker-src 'self' blob:` and `script-src 'self' 'wasm-unsafe-eval'`. Keep every generated `snippets/` helper alongside the module.

```js
// Inside a coordinator worker, before constructing worlds or using Rayon.
const lib = await import("./wasm/pkg-threads/altd_sim.js");
await lib.default({ module_or_path: "./wasm/pkg-threads/altd_sim_bg.wasm" });
await lib.initThreadPool(4);
console.log(lib.cpuThreadCount());
```

The pool lasts for the coordinator's lifetime. Terminate the coordinator and children to stop it. Discard failed pool initialization before retrying a serial package in a fresh worker.

## Math profiles

`mathProfiles()` returns `["proton", "win10-fma3", "win11-fma3"]`. Every serial/threaded package contains all three profiles, so a browser on Linux can run Windows arithmetic. Set `options.mathProfile` to choose explicitly:

```js
const sim = new Simulation(scene, network, model, {
  ...scenario.options, mathProfile: "win11-fma3",
});
```

With no explicit profile, the web package's `await init()` waits for detection and caches the default for later worlds/sessions in that worker or page. It uses `proton` outside Windows. On Windows, Chromium's high-entropy `platformVersion` hint distinguishes Windows 11 24H2 and newer (`win11-fma3`) from earlier Windows (`win10-fma3`). Windows browsers without that hint, such as Firefox/Safari, fall back to `win11-fma3`. `await detectMathProfile()` waits for the cached detection and returns the profile name; after `initSync()` or initialization through another binding target, await it before relying on the default because synchronous initialization cannot wait for a browser hint. An explicit profile overrides detection. Existing simulations retain the profile they were created with.

`trackScene(trackJson, name, templateJson, mathProfile?)` and `randomTrackScene(templateJson, settingsJson, seed, generation, mathProfile?)` accept the same optional profile name. Use the session's profile when preparing track geometry for it. Checkpoint restore requires the destination session to have the saved profile; earlier checkpoints without profile metadata use `proton`. [Kernel and fixture notes](../math/README.md) describe the source shared by the Windows backends.

## WebGPU

`GpuRaycaster` uploads the BSP tree, casts rays and nearest-wall queries, and leaves inference/physics on the CPU. `advanceGenerationWithGpuRays()` uses that raycaster for a complete generation. One readback per tick can make it slower for small populations.

`GpuSimulation` runs sensors, inference, statistics and physics in WGSL, retaining geometry and network buffers between windows. It uses software binary64 and ordered network accumulation. Rust still owns evolution and checkpoints.

All profiles are available in the same package. Pipeline creation specializes the generated shader for the session's profile and removes unreachable functions and tables for each entry point. Explicit fused multiply-add in Windows kernels uses software binary64 `f64_fma`; ordinary products and sums retain separate rounding. Profiled sine/cosine sensor offsets are prepared by the WASM CPU before upload.

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

Pending GPU completion and readback waits prompt Firefox's completion polling with empty queue submissions every 4 ms. The timer exists only while the original operation is pending, adds no simulation commands, and stops after completion, device loss or a submission exception. This avoids the fixed callback delays that can hold six-tick windows near 30 ticks/s. Rebuild both WASM packages and reload existing app workers to use the updated runtime.

## Browser checks

Install Playwright with Chromium, locally or globally. The runner serves this repository with isolation headers and exits nonzero on comparison failures.

```sh
NO_GPU=1 node wasm/test/run.mjs
NO_GPU=1 PAGE=math-profile-init.html node wasm/test/run.mjs
GPU=hardware node wasm/test/run.mjs
PAGE=f64.html node wasm/test/run.mjs
PAGE=math-profiles.html GPU=software node wasm/test/run.mjs
PAGE=math-profiles.html GPU=hardware node wasm/test/run.mjs
node --test wasm/test/runtime-waits.mjs
BROWSER=firefox FIREFOX=/usr/bin/firefox PAGE=bench.html \
  POPULATION=32 TICKS=120 STEP=6 KEEP_ALIVE=1 node wasm/test/run.mjs
for track in rally_asphalt rally_mixed rally_ice; do
  NO_GPU=1 SCENARIO="wasm/test/scenarios/$track.json" \
    REFERENCE="wasm/test/scenarios/$track-reference.json" node wasm/test/run.mjs
done
```

`GPU=software` is the Chromium default and selects SwiftShader. `GPU=hardware` uses the physical Vulkan adapter; Chromium reports its identity. `BROWSER=firefox` defaults to hardware and enables WebGPU preferences; `FIREFOX` selects a native executable through WebDriver BiDi, otherwise the runner uses Playwright Firefox. Firefox does not support the SwiftShader mode. `NO_GPU=1` explicitly skips device checks. `CHROMIUM` selects a Chromium executable; `VERIFY_RAYS`, `VERIFY_POINTS`, `VERIFY_SEED` and `TICK_CHECK` expand ray/tick coverage. `PAGE=full-sim.html VERIFY_SIM=1 GPU=hardware` checks full GPU simulation. Shader validation can run without a GPU using Naga 30.0.1 and `node wasm/test/validate-shader.mjs`; `NAGA` selects its executable.

The shader validator checks the complete modules and every simulation entry point under all three profiles. `node --test wasm/test/runtime-waits.mjs` exercises the public runtime against a device that delivers completions only when polled, checking progress, command order, snapshot bits, concurrency and cleanup without a browser or GPU. `PAGE=bench.html` separates wall time, mapping waits and per-kernel timestamp measurements; its `wallMs` gives throughput as `1000 * ticks / wallMs`. `PAGE=math-profile-init.html` tests default detection with delayed, rejected and unavailable browser hints; `PACKAGE_BASE` selects a threaded package for the same cases. `PAGE=f64.html` tests the software binary64 operations, including fused multiply-add, against exact references. `PAGE=math-profiles.html` compares device `atan2f`, `exp`, `pow` and `tanh` directly with the captured fixture bits for every profile. Full simulation on SwiftShader can stall during compilation of the network forward pipeline; the same limitation was reproduced on the PR#11 baseline with Chromium 151 and 153. Use a physical adapter for full simulation parity. This does not affect the CPU-only, raycaster or binary64 primitive checks.

```sh
cargo run --release --offline --example wasm_reference -- wasm/test/scenario.json target/native-reference.json
cargo run --release --offline --example webgpu_division_reference -- target/webgpu-division.bin
GPU=hardware DIVISION_FIXTURE=target/webgpu-division.bin node wasm/test/run.mjs
CARGO_NET_OFFLINE=true wasm/test/build-threads.sh
NO_GPU=1 PAGE=threads.html THREADS=2 node wasm/test/run.mjs
```

The thread builder writes diagnostic packages under `target/wasm-thread-test/` and a native expanded reference under `target/`. It verifies shared memory and actual workers, serial/threaded observations through three turnovers in both modes, and checkpoint exchange. `STRICT_NATIVE=1` additionally requires exact expanded native parity and currently exposes the documented differences. `BENCHMARK=1` selects measurements instead of parity. `PACKAGE_BASE` selects a package pair; without diagnostic exports worker-index checks are omitted.

Regenerate all four public scenes, spawn poses, benchmark network and native state references with `cargo run --release --offline --example wasm_reference -- --generate-fixtures`. These are current-code regression fixtures, so review their diffs before accepting new observations. [Fixture recipes](test/README.md) describe their provenance.
