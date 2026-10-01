# wasm-bindgen-rayon research

Checked 2026-10-01 against upstream documentation, published source, and the project's installed wasm-bindgen 0.2.129 sources. This is planning evidence. No threaded build or browser experiment was performed.

## Published release and current upstream differ

The latest crates.io release is **1.3.0**, published 2024-12-21. Its package records source revision `6a09d723ed5eed586d1db808f9cc56625b36ff77`. It declares Apache-2.0 and depends on Rayon with `web_spin_lock`. The crate's `nightly` feature is empty and does not configure the compiler. [Crates.io metadata](https://crates.io/api/v1/crates/wasm-bindgen-rayon), [published Cargo.toml](https://github.com/RReverser/wasm-bindgen-rayon/blob/6a09d723ed5eed586d1db808f9cc56625b36ff77/Cargo.toml).

The release README names `nightly-2024-08-02`. Current main, revision `4eea1fa55a965ad516ef9f9e9449704c7eac91c5`, names `nightly-2025-11-15` and adds explicit linker flags. Those are distinct recipes. A current project should pin a dated nightly that builds its existing lockfile and then validate the released adapter with that compiler. Neither old README pin proves compatibility with this project's newer dependencies. [Released README](https://github.com/RReverser/wasm-bindgen-rayon/blob/6a09d723ed5eed586d1db808f9cc56625b36ff77/README.md), [current README](https://github.com/RReverser/wasm-bindgen-rayon/blob/4eea1fa55a965ad516ef9f9e9449704c7eac91c5/README.md).

## Required Rust build settings

Use nightly Cargo and rustc, install `rust-src`, and rebuild the standard library with `-Z build-std=panic_abort,std`. Compiling only this crate with atomics does not make the prebuilt standard library thread-safe. Scope this to the threaded artifact. Cargo documents that every build invocation participating in a `build-std` build needs the flag. [Cargo build-std documentation](https://doc.rust-lang.org/cargo/reference/unstable.html#build-std).

The current upstream configuration is:

```text
-C target-feature=+atomics,+bulk-memory
-C link-arg=--shared-memory
-C link-arg=--max-memory=1073741824
-C link-arg=--import-memory
-C link-arg=--export=__wasm_init_tls
-C link-arg=--export=__tls_size
-C link-arg=--export=__tls_align
-C link-arg=--export=__tls_base
```

Use `wasm-bindgen --target web`. The adapter needs the compiled module object and JavaScript snippets, so its bundler support does not mean `wasm-bindgen --target bundler` is supported. Existing direct Cargo plus wasm-bindgen commands can satisfy these requirements; wasm-pack is optional. The 1 GiB limit above is upstream's example ceiling. Choose a memory budget for this app. [Current adapter README](https://github.com/RReverser/wasm-bindgen-rayon/blob/4eea1fa55a965ad516ef9f9e9449704c7eac91c5/README.md).

The additional linker settings are relevant to the project's actual bindgen version. Its threads transform rejects a shared memory that is not imported, rejects active data segments, and requires the TLS symbols above. Do not copy only the older release's two target features. [wasm-bindgen 0.2.129 threads transform](https://github.com/wasm-bindgen/wasm-bindgen/blob/0.2.129/crates/cli-support/src/transforms/threads/mod.rs#L55).

Repository implication: the current wasm target flags include `--cfg=web_sys_unstable_apis`. Environment `RUSTFLAGS` overrides configuration flags, so a threaded build command must retain this existing setting. Keep separate output and Cargo target directories for threaded and serial artifacts. This follows the existing [.cargo/config.toml](../../.cargo/config.toml) and [build script](../../wasm/build.sh).

## Loading and workers

Enable the adapter's `no-bundler` feature when serving its generated package as ordinary browser modules. The webapp's Vite build does not bundle its `public/assets/wasm` imports, so this feature is appropriate. Export `wasm_bindgen_rayon::init_thread_pool` and await JavaScript `initThreadPool(n)` immediately after module initialization, before CPU code can touch Rayon. Keep both a serial and a threaded build for runtime selection. [Adapter documentation](https://docs.rs/wasm-bindgen-rayon/latest/wasm_bindgen_rayon/).

Released 1.3.0's no-bundler helper fetches its own source, makes a blob URL, and creates module workers. Each child imports the generated main JavaScript module, initializes it with the same compiled module and memory, announces readiness, and enters Rayon's worker loop. It roots worker objects in `_workers` to avoid a Firefox collection bug. Current main uses different worker initialization code, so assess the release source when shipping 1.3.0. [Released helper](https://github.com/RReverser/wasm-bindgen-rayon/blob/6a09d723ed5eed586d1db808f9cc56625b36ff77/src/workerHelpers.no-bundler.js).

The released helper passes deprecated positional arguments to `pkg.default(module, memory)`. wasm-bindgen 0.2.129's generated threaded web initializer still accepts those arguments, retains the supplied memory, and emits a warning. Source inspection supports compatibility; only a real threaded browser test establishes it. A fork is unnecessary solely to adopt object arguments. [Initializer generator](https://github.com/wasm-bindgen/wasm-bindgen/blob/0.2.129/crates/cli-support/src/js/mod.rs#L1552).

Initialize the pool in the existing simulation worker, which becomes the owner of the nested Rayon workers. The Worker constructor is exposed inside dedicated workers. Synchronous parallel Rust exports still block their calling JavaScript worker until they return, so retain bounded simulation windows to process pause and update messages. [Worker constructor exposure](https://html.spec.whatwg.org/multipage/workers.html#the-worker-interface), [first-party browser threading guide](https://web.dev/articles/webassembly-threads).

## Hosting requirements

Serve over HTTPS or a trustworthy local development origin. Send `Cross-Origin-Opener-Policy: same-origin` on the document and `Cross-Origin-Embedder-Policy: require-corp` on the document and worker responses. Verify `crossOriginIsolated` in the page and owner worker. Audit external resource loads for compatible CORS or CORP responses. The headers must come from the hosting server; setting HTML metadata is insufficient. [Chrome SharedArrayBuffer guide](https://developer.chrome.com/blog/enabling-shared-array-buffer/), [cross-origin isolation deployment guide](https://web.dev/articles/cross-origin-isolation-guide).

The existing CSP allows only `worker-src 'self'`. The released blob helper requires `worker-src 'self' blob:` unless replaced with a same-origin worker helper. Ship the entire generated `snippets` directory, preserve its relative module layout, and serve worker JavaScript with a JavaScript MIME type and Wasm with `application/wasm`. Use versioned package directories to keep child imports and the owner's module consistent across deploys. These packaging choices follow the [helper's imports and fetches](https://github.com/RReverser/wasm-bindgen-rayon/blob/6a09d723ed5eed586d1db808f9cc56625b36ff77/src/workerHelpers.no-bundler.js) and the [current webapp CSP](../../../driving_webapp_public/public/_headers).

## Detection, recovery, and pool lifetime

Test isolation and shared-memory availability before importing the threaded artifact. `wasm-feature-detect`'s threads detector also tests sending a SharedArrayBuffer through MessageChannel and validates a Wasm threads fixture. This is stronger than checking `hardwareConcurrency`. Use a separate serial artifact when unsupported. [Detector source](https://github.com/GoogleChromeLabs/wasm-feature-detect/blob/main/src/detectors/threads/index.js), [fixture](https://github.com/GoogleChromeLabs/wasm-feature-detect/blob/main/src/detectors/threads/module.wat).

The released helper's readiness promise listens only for a message. It has no timeout or worker error listener. Recommendation: have the page enforce an initialization deadline, handle owner-worker failure, terminate the failed owner, and create a fresh serial owner worker. A promise timeout inside the owner does not cancel initialization. Check that nested workers stop after owner termination in supported browsers. [Released helper](https://github.com/RReverser/wasm-bindgen-rayon/blob/6a09d723ed5eed586d1db808f9cc56625b36ff77/src/workerHelpers.no-bundler.js).

The adapter calls `ThreadPoolBuilder::build_global`. Rayon allows global initialization once; changing its size later requires a new worker/runtime. Reuse the pool across training sessions, and pass a positive bounded integer. Avoid blindly allocating one worker per reported logical CPU. [Adapter Rust implementation](https://docs.rs/wasm-bindgen-rayon/latest/src/wasm_bindgen_rayon/lib.rs.html), [Rayon pool lifecycle](https://docs.rs/rayon/latest/rayon/struct.ThreadPoolBuilder.html#method.build_global).

Each secondary Wasm instance allocates a default 2 MiB stack plus TLS within shared linear memory. JavaScript workers also have browser overhead. The memory maximum is a growth limit, not a recommendation to allocate that much immediately. Benchmark pool sizes and large populations against the budget. The adapter offers no public resize or teardown API. [Versioned thread allocation source](https://github.com/wasm-bindgen/wasm-bindgen/blob/0.2.129/crates/cli-support/src/transforms/threads/mod.rs#L12).

Keep CPU parallel closures free of JavaScript and WebGPU handles. `JsValue` implements Send and Sync only without the atomics target feature; threaded builds therefore expose borrowing and capture mistakes that the current serial build can hide. Preserve Rust's checks rather than adding unsafe Send/Sync implementations. [JsValue trait availability](https://docs.rs/wasm-bindgen/latest/wasm_bindgen/struct.JsValue.html#impl-Send-for-JsValue).

The app's notices generator must support the adapter's Apache-2.0 license. Its current MIT-only handling will need updating. This is a concrete packaging requirement discovered from the [published manifest](https://github.com/RReverser/wasm-bindgen-rayon/blob/6a09d723ed5eed586d1db808f9cc56625b36ff77/Cargo.toml) and the [repository notices script](../../../driving_webapp_public/scripts/third-party-notices.mjs).

## Implementation gates

First prove one released adapter, dated nightly, current bindgen version, and full flags initialize inside the existing worker under production CSP and isolation headers. Then verify exact state and checkpoint parity for pool sizes 1, 2, and several workers. Exercise missing headers, failed child-worker assets, denied blob workers, initialization timeout, restart, and serial fallback. Measure startup, memory, and steady-state throughput independently. Those are proposed acceptance criteria, not results from this research.
