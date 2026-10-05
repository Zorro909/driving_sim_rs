# Native server implementation review

Date: 2026-10-05

Two independent reviewers used GPT-6.1-sol at ultra effort. The native comparison starts at `origin/main`, commit `0852de991c6e6e2d9868f74346bb0ac9d3260774`. The companion webapp comparison starts at `aebb7ab` and ends at `e99e65541cb502302a32ac2d1bdc4d53b9aee0a9`, branch `t3code/native-simulator-client`. The webapp repository has no configured remote.

The spec is [the approved native server design](../superpowers/specs/2026-10-05-native-server-design.md). Standards sources include `docs/design.md` and the code-review skill's code-smell baseline. The reviewers checked the implementation, then reviewed the fixes below.

## Standards

The reviewer found no hard coding-standard violations or actionable code smells. The correctness review found that an ordinary HIP window error could panic and abort the release server. The new Session path returns errors from world creation, simulator creation, world replacement, network upload, state transfers and window execution. Unsupported scenes and networks fall back during initialization or verification.

The follow-up review found leaked resources after partial GPU constructor failures. Constructors now retain ownership before copying and release partial allocations before returning null. World cleanup excludes borrowed host pointers; simulator cleanup leaves its borrowed World intact.

The final review reports no remaining correctness findings, hard coding-standard violations or actionable code smells.

## Spec

The reviewer found that a hidden layer wider than HIP's 16 inference lanes could abort the server rather than fall back. The HIP path now validates its network limits before upload; verification restores the initial runner state before CPU fallback. Real-socket regressions cover a 32-wide hidden layer and a scene without physics shapes.

The device-name export was incorrectly mandatory. Older compatible libraries now probe device availability through the legacy math function and report the loaded library path. A real-socket HIP training test passed with a library that lacks the name export.

The worker now ignores queued pause and resume commands after a run fails or finishes. A deterministic browser test injects an ordinary native RPC error, checks the saved failed state, and resumes from its checkpoint. Removing the guards makes that test fail.

The final review confirms the constructor cleanup and fault-injection coverage, with no remaining high-confidence spec findings.

## Initial validation

- Rust debug and release suites: 173 passed and five existing tests ignored in each.
- Real-socket server suite: ten passed, including CPU parity, binary/JSON checkpoints, invalid requests, Origin/Host checks, HIP claim release and unsupported HIP fallback.
- Actual RX 7900 XTX Session checks passed for CPU/HIP parity, restart, restore, track replacement, population growth and fallback.
- Constructor cleanup passed all 38 World and 21 simulator injected failures, plus successful and null destruction. The same harness detects a leak in the previous library. ROCm CI runs it without a GPU.
- Native and WASM Clippy, Rust 1.93 all-target compilation, formatting and rustdoc passed.
- Serial and threaded WASM builds passed. CPU reference comparison checked 2,773 values with zero mismatches.
- Webapp build, TypeScript, 31 unit tests, immutable asset publication and formatting passed.
- Full browser suite: 45 passed, five conditional tests skipped. All ten native integration cases passed.
- Hardware WebGPU training, partial-window CPU fallback and contact/reproduction parity checks passed.

Remaining findings: Standards 0; Spec 0. Native/WASM last-bit differences remain tracked in [issue #3](https://github.com/Zorro909/driving_sim_rs/issues/3).

## Size limits and platform follow-up

The user requested removal of all size constraints for local simulator connections. The server now has no configured WebSocket frame or message ceiling. The companion frontend change is commit `432e7da` on `t3code/native-simulator-client`. Native settings bypass browser network, population, stage, file and checkpoint limits while retaining valid values and shared checkpoint field widths. Unsupported HIP shapes continue on native CPU.

Large checkpoints, run metadata and imported tracks use 32 MiB IndexedDB chunks, with replacement and deletion in one transaction. Database version 4 closes older connections before writing the new format. Review found and fixed the unchunked track path, oversized plan edits after browser fallback, and incompatible readers in old tabs. Startup failures with no training history can restart when the local simulator returns.

Both reviewers rereviewed these fixes with no remaining findings. The frontend build, TypeScript, formatting, 35 unit tests and immutable asset checks passed. The full browser suite passed 52 tests with five expected skips, including all 13 native cases and four storage cases. Storage regressions round-trip checkpoints, metadata and tracks above 127 MiB and check rollback, replacement, deletion and schema migration.

Platform CI exposed x87 precision-dependent large-angle reduction on Windows, a host-specific Xavier checksum on macOS, and ARM default NaN signs. The fixes preserve captured bits rather than relax comparisons. The macOS Xavier oracle compares every parameter and RNG bit with same-host CPython. Session tests retain exact Linux GNU goldens and compare both execution modes with the legacy Runner on every host, including exact checkpoint replay.

The final native release suite passed 176 tests with five existing tests ignored. All 35 vehicle tests passed under AArch64 QEMU after reproducing the macOS distance-sensor failure. Debug/release sensor checks and Darwin Clippy passed. Fresh serial/threaded WASM builds passed, with 2,773 CPU reference values and zero mismatches. The rebuilt frontend passed asset checks and all 13 native integration cases again.
