# altd-sim

[![CI](https://github.com/Zorro909/driving_sim_rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Zorro909/driving_sim_rs/actions/workflows/ci.yml)

`altd-sim` is a Rust vehicle simulator and neural-network trainer for AI Learns To Drive. It runs headless on the CPU, through a runtime-loaded AMD HIP library, or in WebAssembly. It reproduces the game's floating-point arithmetic, contact ordering, sensors and evolution behavior against captured regression fixtures. Exactness claims apply to the tested inputs; [fidelity](docs/fidelity.md) records the limits.

The repository includes vehicle and sensor templates, tile resources, fixed generated tracks and self-contained tests. Campaign tracks are not distributed.

## Quick start

Rust 1.94 or newer is the tested native toolchain. Build from the repository root. `--offline` works when dependencies are already cached; omit it on the first build if needed.

```sh
cargo build --release --offline
python3 tools/benchmark.py --population 1024 --ticks 600 --repeats 5 \
  --report target/benchmark.json

target/release/altd-sim --threads 8 train-scratch \
  --scene assets/scenes/rally_template.json --track-mode random \
  --random-track-settings assets/random_track_settings.json \
  --population 256 --generations 3 --ticks 600 --seed 1729 \
  --eliminate-on-wall --idle-eliminate --save-final-candidate \
  --out-dir runs/example
```

The training command starts Xavier networks and generates a reproducible track each generation. Increase the population, generations and tick limit for longer runs. Supply the same scene and track settings when resuming, with `--resume` and a higher `--generations` target.

`.cargo/config.toml` uses `target-cpu=native` for x86 local builds, including the published CPU measurements. For a portable x86_64 binary, override it explicitly:

```sh
RUSTFLAGS='-C target-cpu=x86-64' cargo build --release --offline
```

AVX2 network and geometry paths use runtime detection and retain scalar fallbacks. Exact native trigonometry uses x87 on x86; other targets use the portable implementation described in the fidelity notes.

## Other backends

AMD GPU builds need ROCm and `hipcc`. The default architecture is `gfx1100`, for the RX 7900 XTX.

```sh
gpu/build.sh
# Add --gpu to the train-scratch command above.
```

For another supported AMD GPU, set `GPU_ARCH` when building. Rust builds do not require ROCm. See [GPU setup](gpu/README.md) for validation and profiling.

The default build also includes a local WebSocket server for native CPU or HIP sessions:

```sh
target/release/altd-sim --threads 8 serve
```

It listens on `127.0.0.1:47800/v1` and accepts the `https://drivinglab.jectrum.de` origin by default. `--allow-origin` replaces that list for other clients or local development. [Server documentation](docs/server.md) covers the protocol, security checks and checkpoint resume. Drive Lab's client integration is a separate change.

WebAssembly builds need the wasm32 target and the matching `wasm-bindgen` CLI:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version "$(wasm/bindgen-version.sh)" --locked
wasm/build.sh
```

The JavaScript library includes CPU simulation, a threaded CPU package and optional WebGPU backends. See [WASM setup](wasm/README.md) for initialization, browser requirements and tests. Expanded native/WASM comparisons have known last-bit differences; serial and threaded WASM are compared separately.

## CLI and output

`target/release/altd-sim --help` and `target/release/altd-sim COMMAND --help` show arguments. `bench` measures a fixed population; `train` evolves an exported network; `train-scratch` starts Xavier networks and supports checkpoints, generated-track batches and stop conditions; `evaluate` runs a frozen candidate against a hash-validated suite; `serve` exposes sessions over a loopback WebSocket. The `compare-*` commands compare user-supplied captures with the simulator. Scenes and recorded traces are explicit inputs. [CLI reference](docs/cli.md) covers defaults, resume behavior and report formats.

A `train-scratch` run writes:

| File | Contents |
| --- | --- |
| `run.json`, `log.jsonl`, `progress.json` | Effective options, completed generations and current progress |
| `checkpoint.json`, `checkpoint_gNNNNN.bin` | Generation-boundary population and RNG state |
| `best.json`, `best_laps/` | Networks that set a new best lap |
| `candidate.json` | Final fitness leader when `--save-final-candidate` is enabled |
| `tracks/` | Generated track geometry for replay |

## Code and tests

`src/math/` contains runtime-compatible arithmetic and vectors. `src/physics/` implements vehicles and contacts, while `src/track/` owns scene geometry, spatial queries and generated tracks. `src/nn/` provides scalar and AVX2 inference. `src/training/` owns populations, evolution, RNGs, evaluation and embedding sessions. `src/gpu/hip.rs` loads HIP; `src/gpu/simulation.rs` shares device layouts with `src/wasm/`, which supplies JavaScript and WebGPU bindings. The CLI entrypoint is `src/bin/altd-sim/main.rs`, beside its argument, comparison, benchmark and checkpoint modules. Rust imports follow these groups, such as `altd_sim::physics::car::Car` and `altd_sim::training::session::Session`. See [design notes](docs/design.md).

```sh
cargo test --release --offline
cargo test --offline
cargo build --offline --target wasm32-unknown-unknown --no-default-features --features wasm
```

Tests retain self-contained native captures and label generated golden data as current-code regressions. Browser integration checks live in `wasm/test/`; HIP equivalence checks use `gpu_check`. The exhaustive trigonometry scan and hardware-dependent tests are opt-in. See [performance](docs/performance.md), [GPU internals](docs/gpu.md) and [WASM fidelity](docs/wasm.md).

GitHub Actions checks formatting, native and WASM Clippy, native release and debug tests, rustdoc, Rust 1.93 compatibility, and WASM packages with CPU browser comparisons. Its native jobs override `target-cpu=native` with `x86-64` and include the flags in their cache keys, so cached builds can move between runner CPUs. Serial/threaded browser tests retain the documented expanded native differences. HIP and hardware WebGPU checks run separately because the hosted runners have no GPU.

The project uses the [MIT license](LICENSE). Godot-derived code retains the [Godot notice](GODOT_LICENSE), and numerical modules retain their source notices.
