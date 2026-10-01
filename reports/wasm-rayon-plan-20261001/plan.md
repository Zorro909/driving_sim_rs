# Plan for multithreaded WASM CPU training

Prepared on 2026-10-01 from the working trees of `driving_sim_rs` and
`../driving_webapp_public`. This is an implementation plan. No threaded build
or browser validation has been performed. Upstream facts and source links are
also recorded in [research.md](research.md).

The current webapp uses one background worker for training. Its Rust Rayon
operations run on that worker's single thread. `wasm/build.sh` builds ordinary
`wasm32-unknown-unknown` with the precompiled standard library; it supplies no
atomics/shared-memory flags or thread-pool adapter. The app never calls
`initThreadPool`. The checked-in WASM README describes this explicitly.
Rayon-core's fallback is visible in
[the installed dependency source](/var/home/zorro/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/rayon-core-1.13.0/src/registry.rs:209).

The intended ownership is:

```text
Browser page
  |-- ordinary WASM instance for track conversion and previews
  |-- training coordinator Worker
        |-- threaded WASM instance owning Session and WebGPU handles
        |-- N Rayon Workers, each with its own WASM instance
              all training instances share one WASM linear memory
```

Keep the page's module separate. Rayon work may block while waiting for other
threads; the UI should continue receiving input. The existing training worker
remains the only caller that mutates the session. Rayon closures borrow pure
Rust simulation data while those calls execute. No car populations need to be
serialized to JavaScript or copied between worker messages.

1. Add an optional WASM threading feature and initialization export.

   In `Cargo.toml`, add `wasm-threads = ["wasm", "dep:wasm-bindgen-rayon"]`.
   Put the optional dependency under the WASM target dependencies:

   ```toml
   [target.'cfg(target_arch = "wasm32")'.dependencies]
   wasm-bindgen-rayon = { version = "=1.3.0", optional = true, features = ["no-bundler"] }
   ```

   Use the published release and commit its resolved dependencies in
   `Cargo.lock`. The app uses `@vite-ignore` dynamic imports from `public`, so
   Vite never processes this glue. That makes the adapter's `no-bundler`
   feature appropriate, even though the app itself uses Vite. Keep
   `wasm-bindgen --target web` for the threaded build.
   [Published adapter documentation](https://docs.rs/wasm-bindgen-rayon/1.3.0/wasm_bindgen_rayon/).

   In `src/wasm/mod.rs`, add:

   ```rust
   #[cfg(feature = "wasm-threads")]
   pub use wasm_bindgen_rayon::init_thread_pool;
   ```

   Add a diagnostic export `cpuThreadCount()` returning
   `rayon::current_num_threads()`. Call it only after pool initialization in a
   threaded instance, since querying Rayon can initialize its global pool.
   Report 1 for the ordinary build. Leave the current panic hook in place;
   the WASM start hook must not execute Rayon work.

2. Extend the build script to produce two independent packages.

   Preserve ordinary output at `wasm/pkg` and add threaded output at
   `wasm/pkg-threads`, selected by `THREADS=1`. Give the threaded Cargo build
   its own target directory, for example `target/wasm-threads`, and derive
   the input path for wasm-bindgen from that directory. Both outputs must use
   the same source, Cargo lockfile, profile, and `GAME_DATA` setting. Extend
   `features.json` with `threads` and record the threaded toolchain and memory
   limit for build diagnostics. Ignore the new generated package in
   `.gitignore`.

   Pin a recent dated nightly in the WASM build configuration, with `rust-src`
   installed. Do not turn on nightly or atomics for all native builds or for
   the ordinary WASM package. `nightly-2025-11-15` is the initial candidate
   named by upstream main; validate it with our lockfile before committing the
   pin. The release's older tested nightly is not an established working
   choice for this project's present dependencies.

   The threaded build command should have this shape, preserving the
   `web_sys_unstable_apis` flag that an explicit `RUSTFLAGS` overrides:

   ```sh
   RUSTFLAGS='--cfg=web_sys_unstable_apis -C target-feature=+atomics,+bulk-memory -C link-arg=--shared-memory -C link-arg=--max-memory=1073741824 -C link-arg=--import-memory -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base' \
     cargo +nightly-2025-11-15 rustc \
       -Z build-std=panic_abort,std \
       --target-dir target/wasm-threads \
       --profile release --lib --crate-type cdylib \
       --target wasm32-unknown-unknown \
       --no-default-features --features wasm,wasm-threads

   wasm-bindgen --target web --out-dir wasm/pkg-threads --out-name altd_sim \
     target/wasm-threads/wasm32-unknown-unknown/release/altd_sim.wasm
   ```

   The script must add `game-data` when requested and handle dev profiles.
   Reject threaded `TARGET=nodejs|bundler|deno` choices with an actionable
   error. Continue matching the wasm-bindgen CLI to `wasm/bindgen-version.sh`.

   These explicit linker flags follow
   [upstream's current build recipe](https://github.com/RReverser/wasm-bindgen-rayon#building-rust-code).
   With our wasm-bindgen 0.2.129, the threading transform requires imported
   memory and TLS metadata. Its
   [source checks those requirements](https://github.com/wasm-bindgen/wasm-bindgen/blob/0.2.129/crates/cli-support/src/transforms/threads/mod.rs#L69).
   The published adapter's shorter README is insufficient for this binding
   version. The release helper's positional module/memory initialization
   remains supported by
   [0.2.129's generator](https://github.com/wasm-bindgen/wasm-bindgen/blob/0.2.129/crates/cli-support/src/js/mod.rs#L1572),
   although it prints a deprecation warning. A successful worker smoke test
   is still required before treating this combination as validated.

   The 1 GiB maximum is an initial ceiling, not a 1 GiB allocation on startup.
   Check large supported populations before fixing it. The current
   wasm-bindgen transform allocates a default 2 MiB stack per extra worker,
   plus TLS and other allocations. Measure memory and account for the page's
   separate WASM instance. Keep memory32 and the existing abort panic policy.

   Run optimization with the required features enabled:

   ```sh
   wasm-opt -O3 --enable-threads --enable-bulk-memory \
     -o wasm/pkg-threads/altd_sim_bg.wasm wasm/pkg-threads/altd_sim_bg.wasm
   ```

   [Binaryen's feature options](https://github.com/WebAssembly/binaryen/blob/main/src/tools/tool-options.h)
   enable validation of these operations. Validate and run the optimized
   artifact, and never use a pass that strips atomics or shared-memory
   semantics. Pin the tested optimizer version in the build documentation.

3. Publish complete, immutable package directories.

   Update `scripts/prepare-assets.mjs` to read and validate both packages,
   copy each package recursively including every `snippets` file, and reject
   mismatched `gameData` choices. Clear staging destinations first so stale
   snippets cannot survive a changed dependency or game-data build.

   Hash both packages and vehicle templates into a build ID, then publish:

   ```text
   public/assets/wasm/<build-id>/single/altd_sim.js
   public/assets/wasm/<build-id>/single/altd_sim_bg.wasm
   public/assets/wasm/<build-id>/single/snippets/...
   public/assets/wasm/<build-id>/threads/altd_sim.js
   public/assets/wasm/<build-id>/threads/altd_sim_bg.wasm
   public/assets/wasm/<build-id>/threads/snippets/...
   ```

   Record both relative package URLs and capabilities in `build.json` and
   expose them through Vite's existing build constants. Use a common TypeScript
   simulator interface or generated declaration location outside the hashed
   URL path. The ordinary package lacks `initThreadPool`; narrow that method
   only when loading the threaded package.

   Update `src/simulator.ts` to load the `single` package for track imports
   and previews. Remove the current snippet-only query rewriting for these
   packages. The adapter's helper fetches its own `import.meta.url` and
   imports the root module in each child. A build ID in the directory versions
   the whole dependency tree automatically. Top-level `?v=` alone does not
   version relative imports. Derive URLs from the existing assets base so
   deployment at `/public-demo/` still works. Retain previous build directories
   across deployments while old tabs can still request them.

4. Enable browser isolation and allow the adapter's workers.

   Send these response headers:

   ```http
   Cross-Origin-Opener-Policy: same-origin
   Cross-Origin-Embedder-Policy: require-corp
   ```

   Apply them through `public/_headers` and the host configuration, Vite's
   `server.headers` and `preview.headers`,
   `scripts/serve-test.mjs`, and `driving_sim_rs/wasm/test/run.mjs`.
   Include COEP on served worker scripts; serving these policies throughout
   the app is the straightforward configuration. Require HTTPS in deployment
   or a trustworthy localhost origin. Check `crossOriginIsolated` in both
   the page and training worker. Keep all package files and locally bundled
   fonts same-origin. If cross-origin assets are added later, their responses
   need suitable CORS or CORP permission.
   [Browser isolation requirements](https://developer.mozilla.org/en-US/docs/Web/API/Window/crossOriginIsolated).

   Change both CSP definitions, in `vite.config.ts` and `public/_headers`,
   from `worker-src 'self'` to `worker-src 'self' blob:`. Preserve
   `script-src 'self' 'wasm-unsafe-eval'`. The published adapter
   [creates module workers from fetched blob URLs](https://docs.rs/crate/wasm-bindgen-rayon/1.3.0/source/src/workerHelpers.no-bundler.js).
   Test the actual header CSP on workers as well as the HTML meta CSP. If
   blob workers are unacceptable, a reviewed helper change to use a same-origin
   worker URL is a separate task, rather than dropping `no-bundler`.

   Isolation headers cannot be supplied by an HTML meta tag. A deployment
   without the required headers uses the single-thread package. Update the
   README's hosting instructions and the production-readiness statement that
   currently says isolation is unnecessary. Document header-capable hosting
   for multicore training; a host without these headers retains serial CPU
   and current WebGPU behavior.

5. Initialize the pool inside the training worker before using the simulator.

   Add a worker-side loader used by `src/simulation.worker.ts`.
   Its selection condition is `self.crossOriginIsolated`, available
   `SharedArrayBuffer`, and an actual WASM threads feature probe. Add
   `wasm-feature-detect` to the webapp and use its `threads()` probe.
   Treat missing support as an ordinary single-thread selection.
   [Feature probe implementation](https://github.com/GoogleChromeLabs/wasm-feature-detect/blob/main/src/detectors/threads/index.js).

   For a capable worker the order is:

   ```ts
   const lib = await import(/* @vite-ignore */ threadedPackageUrl);
   await lib.default({ module_or_path: threadedWasmUrl });
   await lib.initThreadPool(threadCount);
   // Only now: sceneFor(), new Simulation(), start/restore, WebGPU verify.
   ```

   Await once per worker/module lifetime. Even constructing a world can
   invoke Rayon, through `NearestGrid::build`, so placing pool initialization
   just before `sim.advance()` is too late. Stage changes reuse the initialized
   pool. The adapter installs a fixed global pool; changing its size requires
   a new worker. Its
   [pool implementation](https://docs.rs/crate/wasm-bindgen-rayon/1.3.0/source/src/lib.rs)
   calls `build_global()`.

   Start with an automatic pool size of
   `max(1, min(8, floor(hardwareConcurrency) - 1))`, treating an absent or invalid
   hardware-concurrency value as 2. This is a proposed conservative default
   to benchmark, not a proven optimum. `threadCount` counts Rayon worker
   threads; the coordinator and browser page are additional threads. Keep
   a test override for 1, 2, 4, 8, and larger counts. Avoid changing the saved
   run schema unless a user-facing thread setting is later justified.

   Load the threaded module for eligible CPU and WebGPU runs. WebGPU's CPU
   evolution and verification can use the same pool, and existing WebGPU
   failure recovery can continue on multicore CPU in the same session.

6. Make startup failure and cleanup deterministic.

   The adapter's helper waits for child readiness but does not attach a
   rejecting worker-error handler or a startup timeout. Therefore a rejected
   import and a child that never becomes ready both need handling.

   Add boot phases and a `ready` event to `src/protocol.ts`. Put a bounded
   worker-boot watchdog in `src/main.ts`, with a separate pool phase deadline,
   for example 15 seconds after module initialization. Do not apply this
   short pool timeout to downloads, shader compilation, or session restoration.

   On threaded module/pool startup failure or timeout, terminate the original
   coordinator and retry the same start command once in a fresh coordinator
   forced to use the ordinary package. Keep the training Web Lock through
   the retry, suppress messages from the previous worker, and do not mark the
   run failed or write a training checkpoint before fallback is decided.
   Do not call simulation methods on a partially initialized threaded instance.
   A second boot failure reports the actual error.

   Do not retry ordinary application errors such as an invalid track or a
   missing checkpoint under the banner of a threading failure. After training
   starts, a pool crash is a failed worker and a checkpoint-resume operation;
   live state recovery from damaged shared memory is outside this change.

   Keep pause as a pause, with its pool available for resume. Freeing a
   `Simulation` does not destroy the global Rayon pool. Stop/switch/delete must
   terminate the coordinator, and tests must show its child workers terminate
   too. Repeated start/stop cycles must not accumulate workers or shared memory.
   Preserve the existing bounded simulation windows and event-loop yields.

7. Reuse the Rust work division and preserve exact arithmetic.

   The useful loops are already present:

   - `src/training.rs`: independent cars run whole tick windows in
     `par_iter_mut()`, avoiding a barrier for each tick. Population creation,
     statistics, checkpoint restoration, and some network operations also
     use Rayon.
   - `src/evolution.rs`: mutation and ordered population assembly use Rayon.
   - `src/pyrandom.rs`: uniform RNG draws stay sequential while scoped tasks
     transform completed Gaussian blocks.
   - `src/world.rs`: nearest-wall grid construction runs in parallel.
   - `src/simulation.rs`: some native-style physics phases use Rayon; coupled
     physics still has synchronization and sequential work.

   Leave these loops and scheduling semantics intact for the first integration.
   Verify that captured Rust types satisfy `Send`/`Sync` under atomics, that
   `SegmentGrid`'s thread-local scratch is independent per worker, and that
   no closure captures JS or WebGPU objects. `Simulation`'s `Rc<RefCell<Session>>`
   remains on the coordinator. It does not need an `Arc<Mutex<Session>>`
   conversion to parallelize the contents of a borrowed session.

   Keep RNG consumption and floating-point accumulation order. In particular,
   do not replace ordered folds with parallel floating-point reductions or
   seed a separate RNG per worker. Threading does not enable the native AVX2
   path on WASM. SIMD is a separate performance project.

8. Report the actual CPU mode and update packaging documentation.

   Extend the existing `engine` event with `cpuThreads` and an optional CPU
   threading fallback reason. Show `CPU, 1 thread` or `CPU, N threads` in the
   running detail. Fix the current UI logic that treats every `engine.note`
   as a WebGPU failure; missing isolation is a different cause. Ordinary lack
   of thread support should not produce a failure toast.

   Add webapp scripts that build both packages, including matching
   game-data variants. Update `wasm/README.md`, the webapp README,
   `PRODUCTION_READINESS.md`, and the performance notes after measurement.
   Extend `scripts/third-party-notices.mjs` to consider the union of crates
   shipped by both WASM builds. It currently requires an MIT license file,
   while wasm-bindgen-rayon is Apache-2.0. Include that license and any required
   notices rather than failing the generator or omitting the crate.
   [Adapter license](https://docs.rs/crate/wasm-bindgen-rayon/1.3.0/source/LICENSE).

9. Validate correctness, deployment behavior, and performance before release.

   Add browser integration checks to the existing tests rather than tests
   that only restate loader branches:

   - On an isolated origin, initialize a pool of at least 2 and confirm its
     reported count, actual child workers, shared memory, and work executed on
     multiple Rayon worker indices using a test-only diagnostic.
     Verify the final optimized artifact, not just the pre-optimization build.
   - Extend the native-reference comparison to run from a training worker
     with 1, 2, and 4 pool threads. Compare exported states, sensors, controls,
     metrics, rewards, generation summaries, and checkpoint bytes through
     multiple generation turnovers and both scheduler modes. Keep the
     independent-car path as the main performance case.
   - Check pause/resume, stage changes, saved and random tracks, restart from
     checkpoints, and resume between serial and threaded builds in both
     directions. Check that pool count never changes during stage transitions.
   - Exercise missing isolation, a failed feature probe, blocked blob workers,
     missing helper files, and child initialization that never signals ready.
     Confirm a bounded fresh-worker serial retry and the retained Web Lock.
   - Run the built deployment with its actual response CSP and headers at `/`
     and `/public-demo/`. Exercise a warmed cache across two builds, ensuring
     nested worker imports stay in the same immutable build directory.
   - Run repeated worker creation/destruction and confirm no surviving child
     workers or growing retained shared-memory allocations. Check UI and pause
     response times while large CPU windows run.
   - Run existing WebGPU verification and failure recovery with the threaded
     package. Confirm GPU failures continue on the CPU with the initialized
     pool and matching state.
   - Test Chromium and Firefox, and Safari/WebKit on the supported devices.
     A browser without working WASM threads must still pass serial fallback.

   Browser training tests need the same authorized local game-data fixtures
   they use today. Keep both public packages without game data by default.

   Benchmark identical scenes and seeds at populations 32, 256, 1,024, and
   4,096 with serial WASM and pools of 1, 2, 4, and 8. Include cold startup,
   warm simulation throughput, evolution time, complete generation time,
   peak memory, and pause latency. Recheck the WebGPU crossover against the
   new CPU baseline. Do not promise linear scaling; small populations may
   lose time to worker dispatch and synchronization. Use these measurements
   to choose the automatic pool cap and any small-workload sequential cutoff.

Implement in this order: build/export and isolated worker smoke test; dual
asset publication and isolation/CSP; worker loader and startup fallback;
runtime reporting and licensing; parity/lifecycle/deployment tests and
benchmarks. The first smoke test must settle the exact nightly, wasm-bindgen,
adapter, and optimizer combination before app integration proceeds.
