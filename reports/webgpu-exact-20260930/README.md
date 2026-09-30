# WebGPU exact simulation validation

The branch is rebased onto `main` at `3fa7fa4`. WebGPU now matches the
native simulation snapshots on Autumn 04 Formula, Rally A01, A07 and B06
using the RX 7900 XTX. All tested ray and closest-wall queries match the CPU.
The fix preserves rounding between operations and replaces approximate
float32 division with integer division and explicit rounding.

This report records the corrected shader. The [earlier report](../wasm-webgpu-20260930/README.md)
records the failures before this fix. The GPU continues to supply ray results;
CPU verification counts mismatches without replacing those results.

## Diagnosis and implementation

The first differing hit was one float32 bit away from the CPU's result.
WGSL permits reassociation and fusion, and permits division error instead
of requiring IEEE division rounding. See the [WGSL floating point rules](https://www.w3.org/TR/WGSL/#floating-point-accuracy).

`src/wasm/float.wgsl` wraps each addition, subtraction and multiplication
in a float-to-integer bitcast, an XOR with a uniform zero, and a bitcast
back to float. The uniform is zero at runtime but is unknown to the shader
compiler. This creates an observable rounding boundary that prevents
fusion or reassociation across operations.

Those boundaries alone reduced the original query differences but did not
eliminate them. Replacing division then removed the remaining differences.
The division helper normalizes 24-bit significands, generates the quotient
and remainder with unsigned integer arithmetic, and rounds once to nearest
with ties to even. Subnormal outputs use the unrounded quotient and a sticky
remainder to avoid double rounding. Its intermediate remainder fits in u32.

The same helper runs in a separate shader against native division results,
so its regression check does not implement a second copy of the algorithm.
The 1,049,732 input pairs include random finite operands across all exponent
ranges and explicit signed-zero, subnormal-boundary and infinity cases.
NaN results must be NaN, but their payload and sign are not compared.

| Autumn query test with seed 7 | Original shader | Rounding boundaries | Integer division added |
| --- | ---: | ---: | ---: |
| Different ray results out of 4,000 | 117 | 35 | 0 |
| Different closest points out of 4,000 | 866 | 70 | 0 |
| Different simulation sensor rays out of 119,743 | 2,420 | 601 | 0 |
| Different exported simulation values | 413 | 302 | 0 |

## Simulation and query results

The primary GPU is an AMD Radeon RX 7900 XTX, RDNA3, RADV NAVI31,
Vulkan 1.4.354, Mesa 26.2.2. Reports identify a hardware adapter with
`isFallbackAdapter: false` and include Chromium's ANGLE renderer.

| Track | Cars | Inference batches per tick | Random rays and points | Tick comparisons | Differences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Autumn 04 Formula | 32 | 2 | 65,536 each | 4,308,210 | 0 |
| Rally A01 | 33 | 1 | 65,536 each | 3,332,016 | 0 |
| Rally A07 | 33 | 3 | 65,536 each | 3,332,016 | 0 |
| Rally B06 | 33 | 8 | 65,536 each | 3,332,016 | 0 |

Each scenario also passes its six snapshots against a native
`wasm_reference` run. The original Autumn reference remains unchanged.
The Rally references are checked in beside their scenarios. They use a
larger network, an uneven population, different batch counts, and different
wall and idle elimination settings. Final GPU and WASM CPU generation
checkpoints match, including offspring parameters and RNG state.

The tick comparison runs both independent and lockstep CPU scheduling for
three generations each. Autumn uses 400 ticks per generation; Rally uses
300. Each tick compares every exported car state, control, sensor and
training metric. Turnover compares parent preservation, rewards and the
complete generation checkpoint. Across these runs, 14,304,258 comparisons
and 1,213,290 checked sensor rays have zero mismatches. The six-snapshot
scenario runs add another 886,405 checked sensor rays.

An additional Autumn run with seed 123456 matches all 1,048,576 random
rays and all 1,048,576 closest-wall queries. This also exercises larger
GPU buffers and dispatches.

The original scenario and all division cases also pass on Chromium
151.0.7922.34, Chromium 153.0.8010.12, and SwiftShader. A separate hardware
run selects the integrated AMD Ryzen 9 7950X GPU, RDNA2, RADV
RAPHAEL_MENDOCINO. It passes 65,536 rays, 65,536 points, all division
cases, and the original simulation scenario.

## Rebase and native checks

The rebase preserves `main`'s buffered random-track training and resolves
the README conflict by documenting both that work and the WASM library.
The native `training_tracks` module depends on HIP types and OS threads,
so it is gated out of the WASM target. The underlying random-track
generation library remains available.

Both native debug and release suites pass 118 tests. Their default runs
ignore the exhaustive engine trigonometry scan and the HIP track-switching
integration test. The latter passes separately after rebuilding the HIP
library. The exhaustive trigonometry scan was not run.

The A01, A05, A06, A07 and B06 game accuracy summary is identical to the
original baseline. All five one-step position assertions pass. The exact
closed-loop assertions pass for A01, A07 and B06; existing A05 and A06
closed-loop drift and small sensor differences remain unchanged.

## Reproduce

Run from the repository root with matching wasm-bindgen-cli and Playwright.
`CHROMIUM` can select either tested browser. In this environment, Playwright
is installed under `/tmp/wasm-webgpu-tools`.

```sh
export NODE_PATH=/tmp/wasm-webgpu-tools/node_modules
export CHROMIUM="$HOME/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome"
wasm/build.sh
cargo run --release --offline --example webgpu_division_reference

GPU=hardware DIVISION_FIXTURE=target/webgpu-division.bin \
  VERIFY_RAYS=65536 VERIFY_POINTS=65536 TICK_CHECK=400 node wasm/test/run.mjs

for track in a01 a07 b06; do
  cargo run --release --offline --example wasm_reference -- \
    "wasm/test/scenarios/$track.json" "target/$track-reference.json"
  cmp "wasm/test/scenarios/$track-reference.json" "target/$track-reference.json"
  GPU=hardware SCENARIO="wasm/test/scenarios/$track.json" \
    REFERENCE="wasm/test/scenarios/$track-reference.json" \
    VERIFY_RAYS=65536 VERIFY_POINTS=65536 TICK_CHECK=300 node wasm/test/run.mjs
done

GPU=hardware VERIFY_RAYS=1048576 VERIFY_POINTS=1048576 VERIFY_SEED=123456 \
  node wasm/test/run.mjs
GPU=software DIVISION_FIXTURE=target/webgpu-division.bin node wasm/test/run.mjs

cargo test --offline -- --test-threads=2
cargo test --release --offline -- --test-threads=2
gpu/build.sh
cargo test --release --offline --test training_tracks \
  gpu_track_switches_preserve_cpu_results_and_reuse_population_buffers \
  -- --ignored --test-threads=1
```

## Limits and evidence

These results establish exactness for the tested scenes, seeds, states and
hardware. They do not prove every possible simulation state or device.
Addition and multiplication still use hardware float32 operations; WGSL
permits flushing subnormal values. Arbitrary NaN payload propagation and
extreme nonfinite scene geometry are not covered. WASM and native f64
transcendental libraries may also differ on inputs outside these fixtures.
Integer division adds GPU instructions; these runs establish correctness,
not a throughput improvement.

Saved results include [Autumn](autumn-extended.json), [A01](a01-extended.json),
[A07](a07-extended.json), [B06](b06-extended.json),
[the million-query test](million-queries.json), [Chromium 151](chromium151.json),
[SwiftShader](swiftshader.json), and [the integrated GPU](integrated-gpu.json).
The [final build check](final-hardware.json) verifies the rebuilt WASM artifact.

The [original failure](before-rounding.json) and [rounding-only result](rounding-only.json)
preserve the diagnostic comparisons. Native evidence is in the
[release log](release-tests.log), [debug log](debug-tests.log),
[HIP integration log](hip-track-switching.log), and [game summary](game-summary.json).
[Provenance](provenance.json) records the rebase target and hashes of source,
references, the generated division fixture and the final WASM artifact.
