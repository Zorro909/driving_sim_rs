# Math profile verification — 2026-10-06

This implementation extends the NVIDIA support merged in [PR #11](https://github.com/Zorro909/driving_sim_rs/pull/11) with portable Proton and Windows UCRT math profiles. All three profiles remain available in a single native library and WASM package. WebGPU specializes the shared generated kernel source for a session before compiling its pipelines.

Checkpoint validation and population imports preserve the selected profile. Asynchronous browser initialization completes default-profile detection before resolving. WebGPU specializes profile dispatch and prunes unused tables; shader sign/selection operations preserve subnormal and negative-zero bits. No profile-specific copies of the kernel source were introduced.

| Check | Result |
| --- | --- |
| Linux x86_64 release suite | 190 passed, zero failures; six existing optional/exhaustive tests ignored |
| Debug suite and strict native clippy | Passed |
| ARM64 Linux musl under QEMU | All seven primitive fixture functions and the profile/RNG/checkpoint continuation matrix passed |
| Native/WASM CPU | All three profiles: 2,773 values each, zero mismatches |
| Browser automatic detection | Eight cases passed for both serial and threaded packages; Windows 10, Windows 11 before/after 24H2, missing/denied/invalid hints, and synchronous followed by asynchronous initialization |
| Serial/threaded worker simulation | Two actual Rayon workers; 253,341 values per scheduling mode, zero serial/threaded mismatches |
| HIP, AMD gfx1100 | Rebuilt library; all-profile math checks passed, with 4,194,304 base inputs per primitive/profile and trig exception keys; Windows simulation/inference/evolution checks passed |
| CUDA, RTX 3060 Ti | 79,304,139 primitive comparisons across profiles and 5,781,863 simulation comparisons per profile, zero mismatches |
| Full WebGPU, AMD RDNA 3 | Formula and mixed Rally scenarios for every profile: 410,858 total comparisons, zero mismatches |
| Captured WGSL primitive fixtures | 30,000 raw-bit comparisons across all profiles on both AMD and SwiftShader, zero mismatches; includes atan2f, exp, pow, tanh |
| WGSL binary64 | 200,000 cases for each of 22 operations, including fused multiply-add and cancellation; zero mismatches on SwiftShader |
| Generated code and shader validation | Rust/table/shader generators current; complete shaders and all 21 specialized pipelines pass Naga 30.0.1 and SPIR-V translation |
| Build/lint compatibility | MSRV 1.93, Windows GNU and ARM macOS/Linux cross-checks; strict serial/diagnostic/threaded WASM clippy, documentation, formatting, workflow parsing passed |

Final serial and threaded distribution packages are in `wasm/pkg/` and `wasm/pkg-threads/`, with matching source fingerprints. Diagnostic packages and browser reports are under `target/wasm-thread-test/`, `target/math-profile-debug/`, and `target/math-profile-test/`.

CUDA used a single all-profile Linux library with CUDA 13.0.88 and driver 580.173.02. The temporary GPU instance was destroyed after verification. Local logs, the library, source snapshot and reproduction script are preserved in `target/cuda-verification/`; build artifacts and raw execution logs are excluded from the repository.

Known boundaries remain explicit:

- Full simulator pipeline compilation stalls/fails on the tested SwiftShader browser backend, including the unmodified PR #11 forward shader. Full simulation was verified on the AMD hardware adapter; all profile primitive fixtures and binary64 operations passed on SwiftShader.
- Expanded native/WASM observations retain the documented host-math differences in Python initialization: 2,285 differences per mode in this fixture. Serial and threaded WASM match each other exactly. See [fidelity notes](../docs/fidelity.md).
- GPU trig inputs outside the supported range are reported as fallback flags and are excluded from exact device comparisons. CUDA recorded 4,590,875 such flags alongside its primitive checks.
- ARM runtime verification used QEMU. Windows/macOS native runtime execution and Windows CUDA DLL compilation were not exercised locally. Windows browsers without version hints use the documented `win11-fma3` fallback.

See [math sources and fixture provenance](README.md), [WASM commands](../wasm/README.md), and [GPU commands](../gpu/README.md) to reproduce the checks.
