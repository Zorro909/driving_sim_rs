# WASM and WebGPU validation, 2026-09-30

This records the original shader. The corrected shader now passes hardware
parity checks; see [the exact simulation validation](../webgpu-exact-20260930/README.md).

The patch is applied. The native and WASM CPU checks pass, but the WebGPU
raycaster fails exact parity on the RX 7900 XTX. This failure repeats in
Chromium 151.0.7922.34 and 153.0.8010.12 with identical mismatch counts.
SwiftShader passes the same tests.

## Hardware and references

The hardware runs use an AMD RDNA3 adapter with `isFallbackAdapter: false`.
Chromium reports `AMD Radeon RX 7900 XTX (RADV NAVI31)`, device `0x744c`,
Vulkan 1.4.354, Mesa 26.2.2. The host also has an integrated GPU and llvmpipe;
the reports identify the selected adapter and ANGLE renderer.

The baseline is commit `816c6dd03fc77b84c8ff15ac1719e86c2a354323`. Before
applying the patch, the release suite passed 108 tests and its CLI executable
was saved for the game-trace comparison. After applying the patch and fixing
the native build collision, both debug and release suites pass 113 tests.
One exhaustive portable trigonometry test remains ignored by default.

A fresh `wasm_reference` run produces exactly the supplied
`wasm/test/reference.json`, including its SHA-256. That reference remains
unchanged. The browser scenario uses Autumn 04 Formula, 32 cars, seed 5,
two inference batches, wall and idle elimination, six snapshots, and two
generation turnovers. The executed windows are `240,0,150,498,0,606`.
Each browser comparison checks 2,773 values, including array lengths.

The native accuracy checker also compares A01, A05, A06, A07 and B06 with
recorded game traces. All five one-step position checks have zero error.
A01, A07 and B06 closed-loop checks have zero error across the fields covered
by `--assert-closed-loop-exact`. Every summary value before and after the
patch is identical. A05 and A06 retain their baseline closed-loop drift and
small sensor differences; those captures were already outside the exact
closed-loop assertion. See the two game summaries for all values.

## Results

| Check | WASM CPU | SwiftShader | RX 7900 XTX |
| --- | ---: | ---: | ---: |
| Scenario vs native, differing values | 0 / 2,773 | 0 / 2,773 | 413 / 2,773 |
| Random ray queries, differing results | n/a | 0 / 4,000 | 117 / 4,000 |
| Closest-wall queries, differing results | n/a | 0 / 4,000 | 866 / 4,000 |
| Simulation sensor rays, differing results | n/a | 0 / 119,743 | 2,420 / 119,743 |
| Process exit status | 0 | 0 | 1 |

The first hardware simulation difference is car 0's `velocity_x` after
240 ticks, `-0.5522225499153137` versus `-0.5522210597991943`. The largest
absolute difference across the exported values is `2.202617198228836`.
These values have different physical units; this number is not a position
error bound. Tick counts remain identical. Mismatch counts require exact
float equality and do not distinguish small coordinate differences from
different geometry decisions.

The observed differences are consistent with the patch's documented WGSL
arithmetic limitations, including permitted fusion and division error.
The particular arithmetic operation responsible was not isolated. The tests
retain exact comparisons and fail on hardware. Use the WASM CPU path when
these reference results must remain exact.

As a hardware control, the existing HIP implementation passes `gpu_check
math rays` on the same RX 7900 XTX. It reports zero mismatches for eight math
primitives tested over millions of inputs and 1,048,576 raycasts plus
1,048,576 closest-wall queries on B06. This control tests the HIP code;
it does not validate the WebGPU shader.

## Changes made after applying the patch

The patch's global `crate-type = ["rlib", "cdylib"]` caused Cargo output
collisions under `cargo test --release`. The failures included undefined
symbols and apparently duplicated `serde_json::Value` types. Native builds
now emit `rlib`, and `wasm/build.sh` explicitly selects `cdylib` through
`cargo rustc --crate-type cdylib`. The original full release command then
passes, and the WASM build still succeeds.

The browser runner now accepts `GPU=hardware` and records adapter and
Chromium device information. It rejects software adapters in hardware mode,
fails when WebGPU is unavailable, and permits a bounded retry while
Chromium starts its Vulkan instance. `GPU=software` preserves the original
SwiftShader mode; `NO_GPU=1` explicitly runs only the CPU comparison.
The comparator checks collection lengths and reports maximum absolute error.

The CPU-only mode passes. A negative control restricting Vulkan to llvmpipe
causes hardware mode to reject the resulting SwiftShader fallback and exit 1.
Dependencies were installed for validation: wasm-bindgen-cli 0.2.129,
Playwright 1.63.0 under `/tmp/wasm-webgpu-tools`, and Chromium 153 in the
existing Playwright cache. Generated WASM assets remain ignored in `wasm/pkg`.

## Reproduce

Run from the repository root. Required tools are Rust with the wasm32 target,
matching wasm-bindgen-cli, and Playwright with Chromium.

```sh
cargo test --offline -- --test-threads=2
cargo test --release --offline -- --test-threads=2
cargo run --release --offline --example wasm_reference -- \
  wasm/test/scenario.json /tmp/wasm-webgpu-native-reference.json
cmp wasm/test/reference.json /tmp/wasm-webgpu-native-reference.json
wasm/build.sh

export NODE_PATH=/tmp/wasm-webgpu-tools/node_modules
export CHROMIUM="$HOME/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome"
GPU=hardware node wasm/test/run.mjs  # expected exit 1 on this hardware
GPU=software node wasm/test/run.mjs
NO_GPU=1 node wasm/test/run.mjs

python3 check_accuracy.py --label wasm-webgpu-check \
  --cases a01 a05 a06 a07 b06 --closed-loop --sensors \
  --assert-position 0 --assert-closed-loop-exact

gpu/build.sh
cargo build --release --offline --example gpu_check
target/release/examples/gpu_check math rays
```

## Saved evidence

- [Hardware, Chromium 151](hardware-chromium151.json)
- [Hardware, Chromium 153](hardware-chromium153.json)
- [SwiftShader, Chromium 153](swiftshader-chromium153.json)
- [CPU-only check](cpu-only.json) and [rejected software adapter](rejected-software.json)
- [Baseline game summary](baseline-game-summary.json) and [patched game summary](patched-game-summary.json)
- [Baseline test log](baseline-tests.log), [release test log](release-tests.log), [debug test log](debug-tests.log)
- [HIP hardware control](hip-math-rays.log) and [WASM build](wasm-build.log)
- [Provenance and artifact hashes](provenance.json)
