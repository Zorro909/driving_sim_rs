# altd-sim

[![CI](https://github.com/Zorro909/driving_sim_rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Zorro909/driving_sim_rs/actions/workflows/ci.yml)

`altd-sim` is a Rust vehicle simulator and neural-network trainer for AI Learns To Drive. It runs headless on the CPU, through a runtime-loaded AMD HIP library, or in WebAssembly. It reproduces the game's floating-point arithmetic, contact ordering, sensors and evolution behavior against captured regression fixtures. Exactness claims apply to the tested inputs; [fidelity](docs/fidelity.md) records the limits.

The repository includes vehicle and sensor templates, tile resources, fixed generated tracks and self-contained tests. Campaign tracks are not distributed.

## Getting started

This guide sets up `altd-sim serve`, the local simulator for [Drive Lab](https://drivinglab.jectrum.de). Drive Lab keeps your tracks, runs and downloads in the browser; the server drives the cars on your CPU threads or an AMD GPU. That removes the browser's limits on cars and network size and its eight-thread cap.

### 1. Download

Download the archive for your system from [Releases](https://github.com/Zorro909/driving_sim_rs/releases). Tagged versions are stable; the `nightly` pre-release is rebuilt from every push to `main` and keeps the same file names. `SHA256SUMS` lists the archive checksums.

| System | Archive | Training on |
| --- | --- | --- |
| Linux x86_64 | `altd-sim-<version>-x86_64-unknown-linux-gnu.tar.gz` | CPU, or AMD GPU through HIP |
| Windows x86_64 | `altd-sim-<version>-x86_64-pc-windows-msvc.zip` | CPU |
| macOS, Apple silicon | `altd-sim-<version>-aarch64-apple-darwin.tar.gz` | CPU |

Unpack it and open a terminal in the unpacked folder:

```sh
tar -xzf altd-sim-nightly-x86_64-unknown-linux-gnu.tar.gz
cd altd-sim-nightly-x86_64-unknown-linux-gnu
./altd-sim --version
```

On Windows, extract the `.zip`, then open PowerShell in the folder and use `.\altd-sim.exe` wherever this guide says `./altd-sim`. macOS binaries are unsigned, so first remove the download quarantine with `xattr -d com.apple.quarantine altd-sim`. Apple silicon uses the portable trigonometry described in [fidelity](docs/fidelity.md).

### 2. Start the server

```sh
./altd-sim serve
```

```text
altd-sim serve: listening on ws://127.0.0.1:47800/v1
  allowed origins: https://drivinglab.jectrum.de
  CPU threads: 16
  HIP: available (AMD Radeon RX 7900 XTX)
```

Keep the terminal open while you train; <kbd>Ctrl</kbd>+<kbd>C</kbd> stops the server. It listens only on `127.0.0.1`, so the browser must run on the same computer.

### 3. Connect Drive Lab

1. In Drive Lab, open **New run**. Under **Compute**, enable **Use a local altd-sim**. The port field defaults to 47800.
2. If the browser asks to allow access to devices on your local network, allow it.
3. The status shows the connection, the CPU thread count and the GPU, if HIP is available. Choose **Native CPU** or **Native GPU (HIP)** and start the run.

Each browser connection gets its own session. Only one session at a time can use the GPU; another one asking for HIP trains on CPU and says why. If the server stops during training, the run stops and keeps its last saved generation. Start the server again and resume the run.

### Options

| Option | Use it to |
| --- | --- |
| `--threads N` | Limit the CPU threads, for example to keep the computer responsive. The default is every logical CPU the process may use. |
| `--math-profile PROFILE` | Reproduce `proton`, `win10-fma3` or `win11-fma3` arithmetic. The default follows the server's operating system; clients can select a profile for each session. |
| `--port N` | Listen on another port when 47800 is taken. Set the same port in Drive Lab. |
| `--allow-origin ORIGIN` | Allow another Drive Lab address, such as a development copy: `--allow-origin http://127.0.0.1:5173`. It replaces the default; repeat the option to allow several. |

```sh
./altd-sim serve --threads 8 --port 47801
```

All builds include all three math profiles. Windows 11 24H2 and newer default to `win11-fma3`; earlier Windows releases use `win10-fma3`; Linux and macOS use `proton`. To train on Linux for a game running on current Windows, for example, pass `--math-profile win11-fma3`. Checkpoints retain their profile. [Math profiles](math/README.md) describes the kernels and captured reference data.

### Training on an AMD GPU

The Linux archive contains `libaltd_gpu.so` next to `altd-sim`. It needs a ROCm 7.x runtime and one of the [covered GPU architectures](gpu/README.md). `./altd-sim gpu-info` checks that the library loads and prints `"layout": "ok"`; with a GPU present it also runs one kernel and prints the device, and fails if the library has no code for that GPU; the server's `HIP:` line shows the device or why HIP is unavailable.

A HIP session drives its first 12 ticks on both the GPU and the CPU and compares them. If they differ, or the network or track needs something HIP does not support, the session trains on CPU and Drive Lab shows the reason. Native CPU and HIP produce the same results. Browser training can differ from both in the last digits of a few values ([issue #3](https://github.com/Zorro909/driving_sim_rs/issues/3)), so a run moved between the browser and the server can develop differently from then on.

NVIDIA GPUs use the same backend: build the library with `gpu/build-cuda.sh` (Linux) or `gpu/build-cuda.ps1` (Windows) as described in the [GPU guide](gpu/README.md#nvidia-cuda). The release archives do not include it.

### Troubleshooting

- **`cannot listen on 127.0.0.1:47800: Address already in use`**: another server is already running, or another program uses the port. Stop it or choose another `--port`.
- **Drive Lab cannot connect**: check that the server is running and that both use the same port. A Drive Lab address other than `https://drivinglab.jectrum.de` must be allowed with `--allow-origin`, including its scheme and port. If you denied local network access, allow it again in the browser's site settings.
- **`HIP: unavailable`**: the line gives the reason. `./altd-sim gpu-info` reports library problems, and [GPU setup](gpu/README.md) covers ROCm and other GPU architectures.

[Server documentation](docs/server.md) describes the protocol and its security checks for other clients.

### Training without Drive Lab

The CLI can also train on its own. This example trains Rally networks on generated tracks and writes them to `runs/example`:

```sh
./altd-sim train-scratch --scene assets/scenes/rally_template.json --track-mode random \
  --random-track-settings assets/random_track_settings.json \
  --population 256 --generations 3 --ticks 600 --out-dir runs/example
```

Add `--gpu` to train on an AMD GPU. `./altd-sim --help` lists every command, and the [CLI reference](docs/cli.md) describes their inputs and outputs.

## Build from source

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

For other AMD GPUs, set `GPU_ARCH` to one or more architectures when building. Rust builds do not require ROCm. See [GPU setup](gpu/README.md) for validation and profiling.

The default build also includes a local WebSocket server for native CPU or HIP sessions:

```sh
target/release/altd-sim --threads 8 serve
```

[Getting started](#getting-started) explains its options and how to connect Drive Lab; [server documentation](docs/server.md) covers the protocol, security checks and checkpoint resume.

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

GitHub Actions checks formatting, native and WASM Clippy, native release and debug tests, Windows and macOS Clippy and release tests, the HIP library build and layout, rustdoc, Rust 1.93 compatibility, and WASM packages with CPU browser comparisons. The release workflow builds and smoke-tests the archives, then publishes tags and nightly builds. Its native jobs override `target-cpu=native` with `x86-64` and include the flags in their cache keys, so cached builds can move between runner CPUs. Serial/threaded browser tests retain the documented expanded native differences. HIP equivalence and hardware WebGPU checks run separately because the hosted runners have no GPU.

The project uses the [MIT license](LICENSE). Godot-derived code retains the [Godot notice](GODOT_LICENSE), and numerical modules retain their source notices.
