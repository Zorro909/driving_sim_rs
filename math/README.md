# Math profiles and reference data

The profile selects the C runtime arithmetic seen by the game. It is independent of the simulator's host, scheduling mode and CPU/GPU backend. All builds include every profile:

| Profile | Game runtime reproduced |
| --- | --- |
| `proton` | Wine/Proton's musl-based UCRT math |
| `win10-fma3` | The x64 Microsoft UCRT FMA3 path used by Windows 10 and Windows 11 through 23H2 |
| `win11-fma3` | The x64 Microsoft UCRT FMA3 path used by Windows 11 24H2, build 26100, and newer |

These names describe the arithmetic being reproduced. A Linux, macOS, ARM or WASM build can select either Windows profile without loading a Windows DLL or requiring the host to execute x86 FMA3 instructions. Native defaults detect the operating system/build; browsers use their platform hints. An explicit profile overrides detection. SSE2 UCRT variants are outside the supported profiles.

## Kernel sources

`kernels/ucrt.h` and `kernels/ucrt_tables.h` are the common source for the new Windows kernels:

- `tools/ucrt/gen-math.py` translates restricted C to `src/math/ucrt.rs` for native x86/ARM and WASM CPU execution.
- `gpu/sim/math.h` includes the same C functions and tables for HIP/CUDA.
- `wasm/gen-sim.py` translates the kernels to `src/wasm/sim/generated.wgsl`. Shader creation specializes each pipeline for the session's profile and removes unused functions/tables.

The shared functions cover `atan2f`, `exp`, `pow`, `tanh` and `log`. `sinf`/`cosf` start with the established musl implementation and apply sorted tables of every input where the Microsoft FMA3 result differs. The GPU's large-input trig restriction remains; the Rust implementation also includes the wide exception tables. WGSL prepares profiled sensor offsets on the WASM CPU, and evolution's `log` runs on the CPU, so those functions/tables are omitted from the WGSL module.

The Proton kernels retain their existing Rust and GPU ports. AVX2 network accumulation remains ordered with separate multiplication/addition; Proton uses its existing vectorized tanh, while Windows profiles apply their generated scalar tanh to each accumulated output. This change gives the Windows kernels a common source without rewriting the established musl and managed/Godot trig implementations.

Each explicit `fma` rounds once: Rust uses `f64::mul_add`, HIP/CUDA use double `fma`, and WGSL uses the integer-based software binary64 `f64_fma`. All other multiplications/additions preserve separate rounding. Keep `-ffp-contract=off` for HIP/C checks and `-fmad=false` for CUDA; neither flag disables an explicit `fma`. The kernel header documents its restricted types, casts, operation order and NaN bit construction.

Check generated code without modifying it:

```sh
python3 -m pip install -r wasm/requirements-sim.txt
python3 tools/ucrt/gen-math.py --check
python3 gpu/gen_tables.py --check
python3 wasm/gen-sim.py --check
```

Ordinary Rust/WASM builds consume committed generated sources and do not require Python or a Windows DLL. To regenerate Rust or WGSL after editing kernels, run the corresponding generator without `--check` and review its diff.

## Captured oracle fixtures

`fixtures/*.bin` contains results from genuine Microsoft UCRT DLLs executed under Wine, alongside Wine's builtin results. These are captured function oracles, separate from native current-code browser/session regression fixtures.

The capture compared FMA3-enabled Microsoft builds in two observed groups: G01, represented by `ucrtbase.dll` 10.0.22621.7510, and G02, represented by 10.0.26100.9444. The binaries, complete raw runs and DLL capture harness are external research inputs, not repository assets. `tools/ucrt/tables.py` pins the representative DLL hashes/version and extracts the committed constants. `tools/ucrt/scan_sincos.c` scans all 2³² float32 bit patterns, recording exceptions against Wine's builtin functions.

`tools/ucrt/fixtures.py` combines game-range and wide-range captures. It selects up to 2,000 differing rows plus 500 evenly spread rows, deduplicates them, and includes exhaustive sine/cosine scan samples. Current files contain 2,500 rows each, except `cosf.bin`, which has 2,499 after deduplication.

There is no file header. Each little-endian row contains input words followed by output words for `proton`, `win10-fma3`, `win11-fma3` in that order:

| Files | Word width | Input words | Words per row |
| --- | ---: | ---: | ---: |
| `sinf.bin`, `cosf.bin` | u32 | 1 | 4 |
| `atan2f.bin` | u32 | 2, ordered y then x | 5 |
| `exp.bin`, `log.bin`, `tanh.bin` | u64 | 1 | 4 |
| `pow.bin` | u64 | 2, ordered base then exponent | 5 |

With the external capture directory and scans available, regeneration is explicit:

```sh
python3 -I tools/ucrt/tables.py --win10 /path/to/g01/ucrtbase.dll \
  --win11 /path/to/g02/ucrtbase.dll --scan /path/to/scans
python3 -I tools/ucrt/fixtures.py /path/to/altd-wine --scan /path/to/scans
```

Fixture generation also needs NumPy. Do not replace oracle outputs with simulator-generated values to make a comparison pass. `tools/ucrt/check_kernels.c` compares the shared C/C++ kernels directly with the full external captures; its header contains compile/run commands.

## Verification scope

`cargo test math::profile` compares each primitive bit-for-bit with all three fixture columns. `gpu_check math` compares HIP/CUDA with the Rust implementations for every profile, and `ALTD_GPU_MATH` selects the simulation checks' profile. Browser CPU comparisons must generate a native reference using the same explicit `options.mathProfile`; full WebGPU comparisons use the WASM CPU in the page. Naga validates the full modules and all three profiles' specialized entry points without requiring a GPU.

`PAGE=math-profiles.html GPU=hardware node wasm/test/run.mjs` checks WebGPU `atan2f`, `exp`, `pow` and `tanh` directly against the captured oracle columns, including signed zeros and subnormal inputs, rather than relying on a driving scenario to reach those cases. `GPU=software` runs the same 30,000 comparisons on SwiftShader; CI runs these fixture checks and the software binary64 tests without requiring physical GPU hardware.

Primitive fixtures and successful tests on one adapter do not establish equality for every device, argument or complete game run. Existing host-dependent Python initialization and managed/Godot trig limits still apply. [Fidelity](../docs/fidelity.md), [the WASM guide](../wasm/README.md) and [GPU checks](../gpu/README.md) describe those boundaries.
