# GPU backends (AMD HIP, NVIDIA CUDA)

The GPU library runs whole driving generations on an AMD GPU (HIP) or an NVIDIA GPU (CUDA, see below). Captured CPU/HIP checks compare simulation values, population parameters, RNG state and checkpoint bytes exactly. Rust owns evolution and checkpoint I/O; both backends can resume the same generation-boundary checkpoints. See [fidelity](../docs/fidelity.md) for the scope of that evidence.

## Build and use

Linux release archives include `libaltd_gpu.so` beside `altd-sim`, built with ROCm 7.1 for the targets in `release-targets`. They need a ROCm 7.x runtime that provides `libamdhip64.so.7`, but no compiler. `altd-sim gpu-info` reports the loaded library. To build the library yourself, install ROCm with `hipcc`, then build from the repository root:

```sh
gpu/build.sh
cargo build --release --offline

target/release/altd-sim --threads 8 train-scratch --gpu \
  --scene assets/scenes/rally_template.json --track-mode random \
  --random-track-settings assets/random_track_settings.json \
  --population 256 --generations 3 --ticks 600 --seed 1729 \
  --eliminate-on-wall --idle-eliminate --out-dir runs/gpu-example
```

`OUT` overrides the library output directory, default `target/gpu`. `GPU_ARCH` lists one or more architectures separated by commas or spaces, default `gfx1100` for RX 7900 XTX; `rocminfo` reports a GPU's name. `GPU_ARCH="$(cat gpu/release-targets)"` builds the release library for every consumer target.

`--gpu` and `gpu-info` load `ALTD_GPU_LIB` when set. Otherwise they try `libaltd_gpu.so` beside the executable, then `target/gpu` of the checkout that built the binary. Load errors list each attempted path.

| Targets | GPUs |
| --- | --- |
| gfx803, gfx900, gfx902, gfx906, gfx909, gfx90c | Polaris and Vega: RX 470-590, Vega 56/64, Radeon VII, Vega APUs |
| gfx1010-gfx1013 | RDNA1: RX 5500-5700 XT |
| gfx1030-gfx1036 | RDNA2: RX 6400-6950 XT, Steam Deck, Radeon 610M/680M |
| gfx1100-gfx1103, gfx1150-gfx1153 | RDNA3 and RDNA3.5: RX 7600-7900 XTX, Radeon 780M/890M, Strix Halo |
| gfx1200, gfx1201 | RDNA4: RX 9060/9070 |

The ROCm 7.1 runtime officially supports fewer consumer GPUs than this list; the remaining targets are best effort. The captured CPU/HIP equivalence checks have run on gfx1100 only. The script verifies the checked-in double tables against their Rust bit patterns without rewriting tracked files. `python3 gpu/gen_tables.py --output PATH` writes a header explicitly; `--check` validates it.

The compiler flags `-ffp-contract=off` and `-fhip-fp32-correctly-rounded-divide-sqrt` preserve separate multiply/add rounding and correctly rounded float32 division/square root. Keep them when changing architecture. The native crate loads the library at runtime, so CPU-only builds do not require ROCm.

## NVIDIA (CUDA)

The same sources build for NVIDIA GPUs with `nvcc`; `sim/compat.h` maps the HIP runtime calls and warp helpers to CUDA. The backend, library ABI and checkpoints are unchanged: `--gpu`, `gpu-info` and `backend: "hip"` use whichever library is loaded. Install the CUDA toolkit (13.x tested) and a host compiler (Visual Studio Build Tools on Windows, gcc or clang on Linux), then build from the repository root:

```sh
gpu/build-cuda.sh                  # Linux, writes target/gpu/libaltd_gpu.so
gpu/build-cuda.ps1                 # Windows, writes target/gpu/altd_gpu.dll
```

`OUT` overrides the output directory. `CUDA_ARCH` defaults to `native`, which builds for the GPUs of the build machine. Without a GPU, nvcc 13.4 warns and builds `sm_75` code and PTX instead, which newer GPUs compile at load time. To build for other GPUs, or several, list `sm_` targets separated by commas or spaces; `nvidia-smi --query-gpu=compute_cap --format=csv` reports the number, so 8.6 is `sm_86`. A list also embeds PTX for its newest target, which the driver compiles at load time for a newer GPU. CUDA 13 builds for Turing (`sm_75`, RTX 20 series) and newer only; older GPUs need a CUDA 12 toolkit, which is untested. Both scripts check the double tables with `gpu/gen_tables.py --check` (`python3` on Linux, the `py` launcher or `python` on Windows). The binary loads `ALTD_GPU_LIB`, then the library beside the executable, then `target/gpu`, as on AMD. `altd-sim gpu-info` runs one kernel when a GPU is present, so a library built for another GPU fails there with HIP error 209.

nvcc rejects a Visual Studio newer than the toolkit supports. `gpu/build-cuda.ps1 -AllowUnsupportedCompiler` (or `CUDA_ALLOW_UNSUPPORTED_COMPILER=1`) builds anyway with a warning; NVIDIA has not validated that combination, so run the checks below before trusting the library.

`-fmad=false` is the CUDA counterpart of `-ffp-contract=off`; keep it, and keep nvcc's default correctly rounded float32 division and square root (no `-use_fast_math`, no `-ftz=true`). NVIDIA arithmetic also replaces every NaN result with its canonical NaN, so the places where the CPU returns a NaN operand or the x86 default NaN call `altd_nan_operand` and `altd_invalid` from `sim/compat.h`, which build those bits explicitly on both vendors.

The CPU/GPU checks above pass on an RTX 4070 Laptop (sm_89, Windows, CUDA 13.4), and on Linux on an RTX 2070 (sm_75, driver 580) and an RTX 3060 (sm_86, driver 595) with `build-cuda.sh` libraries from CUDA 13.4: `native`, and `sm_75,sm_86`. They also pass with a CUDA 13.0 `sm_75` library run through its PTX (`CUDA_FORCE_PTX_JIT=1`). On all three GPUs `train-scratch --gpu` checkpoints are byte-identical to the CPU's. The driver compiles PTX only from a toolkit no newer than itself, so PTX for newer GPUs needs a driver at least as new as the CUDA toolkit that built the library. Consumer NVIDIA GPUs run float64 at 1/64 of their float32 rate, which bounds the network forward pass. A Windows GPU that drives a display kills kernels that run for more than a couple of seconds; the windowed launches stay well below that in the checks above.

## Execution model

GPU driving uses tick-major execution for independent cars. Native shared broadphase and paused windows are unsupported. Generated CLI tracks disable shared broadphase. Track preparation runs on a CPU producer with a bounded queue; `--track-buffer-size` sets its depth, default eight. Population/network buffers stay allocated while each fresh track replaces geometry.

## Validation and measurements

```sh
cargo build --release --offline --example gpu_check
target/release/examples/gpu_check --help
target/release/examples/gpu_check schedules
```

The default check fixture is a fixed generated Rally track and seeded Xavier networks. `ALTD_GPU_SCENE`, `ALTD_GPU_SPAWN`, `ALTD_GPU_NETWORK`, `ALTD_GPU_MODEL` and `ALTD_GPU_CKPT` select explicit inputs or a saved population. No captured campaign scene is required. `gpu_check --help` lists primitive, query, sensing, inference, physics and generation checks. Exhaustive parts can take minutes.

Run benchmarks on an otherwise idle GPU:

```sh
RAYON_NUM_THREADS=8 python3 gpu/benchmark.py \
  --library target/gpu/libaltd_gpu.so --cars 8192 32768 \
  --samples 5 --warmups 1 --no-elimination --output target/gpu-fixed.json
RAYON_NUM_THREADS=8 python3 gpu/benchmark.py --training \
  --library target/gpu/libaltd_gpu.so --cars 8192 --samples 5 \
  --output target/gpu-training.json
```

The runner keeps every sample and validates result digests. `--checkpoint DIR` selects a saved population; omitting it uses the fixed Xavier recipe. `--scene`, `--spawn-trace`, `--network`, `--model` and `--binary` override inputs. Fixed-window timing excludes fixture construction, reset, hashing and reproduction; training timing includes consecutive generation turnover. Elimination can finish early, so inspect executed ticks before comparing results.

`gpu/profile.sh` collects phase timings, optional rocprofv3 traces/counters and diagnostic kernel builds. `--only 1,2`, `--sizes "8192 32768"`, `--checkpoint DIR` and `--output target/profile` select the work. Profiling changes execution behavior; final timing should run with profiling disabled. [Performance notes](../docs/performance.md) retain measured hardware results.

| Variable | Effect |
| --- | --- |
| `ALTD_GPU_LIB` | Runtime shared-library path |
| `ALTD_GPU_GRAPH=0` | Direct launches instead of HIP graph replay |
| `ALTD_GPU_SPLIT=0` | Fused instead of split physics |
| `ALTD_GPU_PROFILE=1` | Device phase timings; `host` reports host/wall time only |
| `ALTD_TRAIN_PROFILE=1` | CPU reproduction, installation and transfer timings |

[Implementation notes](../docs/gpu.md) explain batching, graph lifetime, exact reduction order and memory costs. `gpu_engine_scan` is an exhaustive native/GPU trigonometry developer tool; it requires an explicit thread count and output header, and is not part of the regular test suite.
