# Fidelity and regression coverage

The simulator targets the captured behavior of AI Learns To Drive. Physics mixes float32 state with float64 calculations, and callbacks, contact caches and inference batches have observable ordering. Matching formulas alone does not reproduce those results.

Numerical implementations retain operation order, intermediate precision, rounding boundaries, tie-breaking and RNG consumption. Scalar and AVX2 inference accumulate in the same order. HIP builds disable multiply-add contraction and require correctly rounded float32 division and square root. Spatial acceleration preserves traversal and first-equal-result decisions. A change to any of these rules needs exact comparisons before performance measurements.

## What the tests establish

Existing `tests/fixtures/` captures exercise native vehicle transitions, population lifecycle, contact/reset behavior, runtime math, sensors, statistics and training. They are finite oracle samples. Older traces sometimes omit body matrices or contact caches, so restoring their visible state cannot prove complete native state reconstruction.

Generated-track goldens record the simulator's current results for controlled contacts, restored states, scoring, population redraw and Session checkpoints. They detect regressions; they are not independent game captures. Fixtures and their origins are documented in [the fixture README](../tests/fixtures/README.md). Differential tests compare accelerated queries with brute force, and CLI tests cover settings, checkpoints, resume, generated batches and evaluation.

A frozen diagnostic comparison against an A07 capture matched 1,393 of 1,398 one-step frames exactly. The remaining five differed in angular velocity, rotation or velocity. Network inference matched all 1,398 frames with zero error. These results predate the public generated-track replacements and describe that capture, not every scene or closed-loop run.

## Platform limits

On x86, native `engine_sin_cos` uses x87 `FSINCOS`. The portable fdlibm-based implementation is validated against it over float32 arguments with absolute value at most 16; the regular suite samples the domain and an ignored exhaustive test scans it. Larger arguments are not covered by that claim.

Native and WASM use different host implementations for some float64 transcendental operations. The public generated Formula state scenario matched 2,773 native values, while expanded snapshots found 2,285 native differences per scheduling mode: 703 controls, 405 metrics and 1,177 checkpoint bytes. Serial and threaded WASM had the same native differences and matched each other in all 253,311 checked values per mode. A passing state-only comparison does not imply exact expanded native parity. Counts depend on the scenario and checkpoint layout.

The RX 7900 XTX WebGPU raycaster matched 4,000 random rays, 4,000 nearest-wall points and 119,743 in-simulation rays in that baseline. WGSL floating-point behavior, subnormal handling and NaN payloads still vary by device. Runtime verification counts discrepancies without replacing GPU results. See [WASM notes](wasm.md) for the comparison layers.

## Checking a change

Run native tests in debug and release, regenerate observations to temporary outputs, and compare exact float bits, RNG state and checkpoint bytes. Exclude only named timing and path metadata when comparing reports. Keep generated references separate from captured oracle data. Measure throughput only after those checks pass.
