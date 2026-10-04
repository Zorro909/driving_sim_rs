# WASM and WebGPU fidelity notes

The serial and threaded packages compile the same Session and TrainingRunner code. The threaded package rebuilds Rust's standard library with atomics and uses `wasm-bindgen-rayon`; it requires an initialized worker pool before constructing worlds or invoking Rayon. Both packages preserve the existing arithmetic and RNG algorithms.

Three comparisons answer different questions:

- Native versus WASM state/reference checks catch portability differences in the exported fields.
- Serial versus threaded WASM checks cover states, sensors, controls, metrics, rewards, summaries and binary checkpoints, including exchange between the two packages.
- WebGPU versus WASM checks validate query or full-window device results against the WASM CPU implementation.

Expanded native parity is incomplete. On the public generated Formula scenario, serial and threaded WASM matched each other in 253,311 values per mode, but both differed from native in 2,285 values per mode. Those differences comprised 703 controls, 405 metrics and 1,177 checkpoint bytes. The original captured-scene baseline had 2,532 native differences with the same exact serial/threaded equality. Counts depend on the scenario, build and checkpoint representation. `STRICT_NATIVE=1` makes the browser test fail on any expanded native difference.

## Device arithmetic

The WebGPU raycaster keeps simulation on the CPU and casts rays or nearest-wall queries on-device. Explicit float32 rounding boundaries prevent fusion and reassociation. Division corrects a hardware estimate using an exact integer remainder, with an integer-loop fallback; square root and software binary64 division use the same principle. Full `GpuSimulation` also implements binary64 arithmetic in integer WGSL operations and keeps network accumulation ordered.

Historical RX 7900 XTX checks with RADV 26.2.2 matched 65,536 rays and 65,536 nearest-wall points per scenario, and millions of in-simulation rays across Formula and three Rally scenes. A newer baseline repeated a smaller 4,000-query corpus and 119,743 in-simulation rays with zero differences. The public replacements contain four distinct generated tracks and native current-code references, not game-oracle captures.

WGSL permits device-dependent subnormal and NaN behavior. Raw division fixtures check signed zero, subnormal boundaries, infinities and broad exponent coverage; NaN payloads are treated as equivalent in that arithmetic diagnostic. Simulation comparisons retain exact values and count differences without substituting CPU results. Results on one GPU do not validate every browser adapter.

## Runtime and memory

Threaded WASM uses imported shared memory and a 2 GiB growth ceiling. This is a maximum, not an initial allocation. Workers need stack and TLS memory; an application's separately loaded preview module has its own memory. Shared-memory packages require cross-origin isolation and complete generated helper snippets.

A host must await GPU windows before mutating the Session. A failed full-GPU window keeps the last committed WASM state so it can continue on CPU. Paused physics and native shared broadphase are unsupported by full WebGPU. The raycaster has one readback per tick; full simulation batches bounded windows, which changes the population size at which device work can pay off.

[The WASM guide](../wasm/README.md) contains build commands, initialization and the browser harness. [Performance notes](performance.md) distinguish CPU worker scaling, startup and WebGPU measurements.
