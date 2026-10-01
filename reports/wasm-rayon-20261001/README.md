# WASM Rayon implementation validation

Measured 2026-10-01, with final app checks on 2026-10-02, on Linux with an AMD Ryzen 9 7950X, Chromium
153.0.8010.12, Firefox 155.0, and an RX 7900 XTX with RADV 26.2.2.
The simulator base is `87bd8ba`; the webapp base is `1923c2c`.
Webapp changes live in the separate worktree
`/var/home/zorro/.t3/worktrees/driving_webapp_public/t3code-rayon-plan`,
branch `t3code/implement-wasm-rayon-plan`.

## Validated build combination

- nightly-2025-11-15 with rust-src and rebuilt std/panic_abort
- wasm-bindgen CLI and crate 0.2.129
- published wasm-bindgen-rayon 1.3.0 with no-bundler
- Binaryen 130, `-O3 --enable-threads --enable-bulk-memory`
- imported shared memory, exported TLS metadata, memory32, abort panics
- initial threaded memory growth ceiling of 1 GiB

The first smoke test completed before webapp integration. It initialized two
child workers under response CSP, COOP and COEP, reported two CPU threads,
passed the same shared memory to both children, and observed Rayon indices
0 and 1. The adapter's positional initializer warning is expected for this
released helper. Packages are generated from a clean staging directory.

## Correctness and release limits

The optimized diagnostic packages compare every state, sensor, control,
metric, reward, summary and checkpoint byte through three turnovers in both
independent and lockstep modes. Each mode checks 1,744,559 values. Chromium
pools 1, 2 and 4 and Firefox pool 2 have zero serial/threaded mismatches.
The checks restore threaded checkpoints in serial WASM and restore serial
checkpoints in threaded WASM before comparing subsequent execution.

The expanded native reference is deliberately kept separate. It has 19,542
differences per mode in both serial and threaded WASM: 784 control values,
427 metric values and 18,331 checkpoint bytes. States, sensors, summaries
and rewards have no differences. The first control differs by two binary64
ULPs. Native/serial transcendental differences were already a documented
portability limit. This integration preserves the existing algorithms and
RNG consumption; it does not relax comparisons between the two WASM builds.
Reports include both comparisons, and `STRICT_NATIVE=1` fails on native
differences. Strict expanded native parity remains an unmet release gate.
The original checked-in native state reference still passes.

Hardware WebGPU verification passes with an initialized two-worker pool,
including three generation windows compared against threaded CPU. The app's
hardware GPU tests cover startup, exact contact parity, and CPU continuation
after an injected GPU-window failure.

Browser tests exercise actual isolation/CSP at `/` and `/public-demo/`,
missing isolation, failed feature probes, missing helpers, denied blob
workers, and children that never announce readiness. Pool timeout failures
retry in about 16 seconds in a fresh serial coordinator while retaining the
Web Lock. Invalid application checkpoints do not retry. A second boot
failure reports the serial error and releases the lock. Pause/resume retains
the pool, and random-track stage changes rebuild sessions without rebuilding
it. Three start/pause/delete cycles return Chromium worker targets to the
baseline each time. A 1,024-car app run checks page input and pause completion
within 1.5 seconds. The existing 5,000-car checkpoint test passes below the
memory ceiling.

A panic or out-of-memory abort in a pool worker ends only that worker, and the
coordinator then waits forever for its Rayon job. Once the pool is up, the page
pings the coordinator every 5 seconds. Any message counts as a reply. If no
reply arrives for 60 seconds, the page stops the run with an error, without a
serial retry. A late timer, which happens when the page was suspended or
throttled, sends a new ping instead of failing. Browser tests drive the page
clock: a coordinator that never replies fails after the deadline, and a paused
coordinator keeps replying for over two minutes without an error.

The coordinator waits for the page to acknowledge `ready` before starting
training or writing its first checkpoint. The browser test holds that
acknowledgement and confirms no checkpoint or training progress appears,
then releases it and checks that training starts and pauses normally.

A separate Linux process-memory check runs four 4,096-car start/pause/delete
cycles with four pool workers and the page's ordinary preview module loaded.
It samples the RSS of processes reported by Chromium's SystemInfo API every
100 ms. The sampled combined peak is 875.4 MiB; shared pages can be counted in
more than one process. After deletion and page GC, renderer RSS is 176.9,
177.2, 178.5 and 180.3 MiB, and renderer virtual size returns to within 1 MiB
of the first cleaned-up run. Every cycle returns worker targets to zero.
The small retained growth does not resemble retaining another session's
linear memory or address-space reservation each cycle.

These process measurements include browser, page and checkpoint overhead.
They are separate from the benchmark's peak linear-memory numbers. Chromium's
detailed allocation dumps time out while Rayon workers are idle and do not
give a direct count of retained WASM backing stores. Allocator-level retention,
Safari/WebKit and supported mobile devices still need validation. No production
deployment was performed.

## Completed checks

| Check                                                                         | Result                                                                                     |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Native release library tests                                                  | 32 passed, 2 ignored                                                                       |
| Original native-reference browser test                                        | 2,773 comparisons, zero differences                                                        |
| Full Chromium app suite                                                       | 35 passed, 2 hardware-only skips                                                           |
| Full Firefox app suite                                                        | 33 passed, 4 GPU/CDP skips                                                                 |
| Warmed-cache revisions and ready acknowledgement                              | Included in both full browser suites                                                       |
| Hardware GPU app suite                                                        | 8 passed, including availability checks and injected failure recovery                      |
| Asset publication integration                                                 | Passed with real packages, source mismatch rejection, retained old IDs and missing helpers |
| TypeScript, Vite build, domain tests, Prettier, diff whitespace, shell syntax | Passed                                                                                     |
| Threaded dev build                                                            | Passed, optimizer omitted as intended                                                      |
| Threaded TARGET=nodejs rejection                                              | Passed with actionable TARGET=web error                                                    |

The warmed-cache test runs successive imports of two generated-glue revisions
with warm module/WASM caches, then returns to the older revision. Every nested
request stays under the selected immutable directory at `/public-demo/`.
Filesystem publication/retention is checked separately using the actual
asset-preparation script.

The deployment
archive retains its already approved immutable history independently of
release pruning. Legacy WASM test pages now use an external entry module,
so the stricter response script CSP also works for the existing native/GPU
harnesses.

## CPU and GPU measurements

The Formula scene and seed are `wasm/test/scenario.json`, with a `[20, 16, 5]`
network, two inference batches and elimination disabled for the benchmark.
Each population runs four 180-tick generations; the first is warm-up and
the tables report the median of the next three. CPU advancement uses adaptive
windows around 50 ms, with MessageChannel yields. Evolution and complete
generation time are measured separately. Startup measures module and pool
initialization in a fresh browser/coordinator, excluding scene/session setup.
These are a single sweep with local HTTP assets, not a scaling guarantee.

Warm simulation milliseconds per generation:

| Cars | Serial | Pool 1 | Pool 2 | Pool 4 | Pool 8 | WebGPU with pool 8 |
| ---- | -----: | -----: | -----: | -----: | -----: | -----------------: |
| 32   |   12.1 |   12.6 |    7.7 |    4.4 |    3.6 |              363.8 |
| 256  |   95.9 |  104.4 |   54.2 |   28.2 |   14.4 |              358.3 |
| 1024 |  382.0 |  436.3 |  214.3 |  112.0 |   57.0 |              182.6 |
| 4096 | 1577.7 | 1805.5 |  929.8 |  454.6 |  228.9 |              622.4 |

Complete generation milliseconds, including evolution and worker yields:

| Cars | Serial | Pool 1 | Pool 2 | Pool 4 | Pool 8 | WebGPU with pool 8 |
| ---- | -----: | -----: | -----: | -----: | -----: | -----------------: |
| 32   |   12.7 |   13.6 |    8.5 |    5.1 |    4.9 |              365.2 |
| 256  |  100.4 |  109.1 |   57.1 |   30.8 |   18.3 |              361.5 |
| 1024 |  399.5 |  455.7 |  226.3 |  120.2 |   66.1 |              189.0 |
| 4096 | 1657.3 | 1882.3 |  980.3 |  514.7 |  263.8 |              647.7 |

The largest case is about 6.9 times faster for advancement and 6.3 times
faster per complete generation with eight workers than with serial WASM.
A one-worker shared-memory package is slower than ordinary WASM in this sweep.
Eight workers improve warm throughput even at 32 cars, but startup is more
expensive. Keep the proposed cap of eight and leave scheduling/cutoffs alone
for this first integration. Revisit short-run and low-core-count defaults
with broader measurements.

The multicore CPU beats this WebGPU path at all four measured populations.
The previous single-thread crossover no longer applies. A crossover above
4,096 cars has not been established.

| Runtime | Module/pool startup ms | Largest peak linear memory MiB | Largest evolution median ms | Max coordinator message latency ms |
| ------- | ---------------------: | -----------------------------: | --------------------------: | ---------------------------------: |
| Serial  |                   14.1 |                           97.6 |                        75.3 |                              188.7 |
| Pool 1  |                   31.2 |                          112.8 |                        76.8 |                              205.3 |
| Pool 2  |                   33.9 |                          101.6 |                        47.5 |                              118.0 |
| Pool 4  |                   38.6 |                          105.7 |                        29.7 |                               70.8 |
| Pool 8  |                   94.5 |                          113.7 |                        34.5 |                              100.0 |

Memory is the peak committed WASM linear-memory size observed by the
coordinator. It excludes JavaScript worker overhead, GPU allocations and the
page's separate preview module. The maximum is a ceiling, not startup
allocation. Message latency includes session setup and evolution; the
actual app pause check is separate. Maximum CPU windows for the largest
population were approximately 61 to 89 ms in this sweep.

## Reproduce

```sh
rustup toolchain install nightly-2025-11-15 --component rust-src --target wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version "$(wasm/bindgen-version.sh)" --locked
PATH=/path/to/binaryen-130/bin:$PATH wasm/test/build-threads.sh
for count in 1 2 4; do
  NO_GPU=1 PAGE=threads.html THREADS=$count node wasm/test/run.mjs
done
NO_GPU=1 PAGE=threads.html THREADS=2 BROWSER=firefox node wasm/test/run.mjs
STRICT_NATIVE=1 NO_GPU=1 PAGE=threads.html THREADS=2 node wasm/test/run.mjs
GPU=hardware PAGE=threads.html THREADS=2 node wasm/test/run.mjs
for count in 0 1 2 4 8; do
  NO_GPU=1 PAGE=threads.html BENCHMARK=1 THREADS=$count node wasm/test/run.mjs
done
GPU=hardware PAGE=threads.html BENCHMARK=1 THREADS=8 node wasm/test/run.mjs
```

`THREADS=0` in the browser harness means ordinary WASM. In `wasm/build.sh`,
`THREADS=0` selects ordinary output and `THREADS=1` selects threaded output.
The test builder keeps diagnostic exports under `target/wasm-thread-test`.
Distributable packages omit those exports.

For this pair of worktrees:

```sh
cd /var/home/zorro/.t3/worktrees/driving_webapp_public/t3code-rayon-plan
export SIM_ROOT=/var/home/zorro/.t3/worktrees/driving_sim_rs/t3code-2326a554
PATH=/path/to/binaryen-130/bin:$PATH npm run wasm
npm run build
npm run notices
npm run test:assets
npm test
npx playwright test --workers=2
BROWSER=firefox npx playwright test --workers=2
GPU=hardware npx playwright test -g GPU --workers=1
```

From the simulator worktree, run the Linux memory check against the already
built webapp:

```sh
APP_ROOT=/path/to/driving_webapp_public node reports/wasm-rayon-20261001/memory-check.mjs
```

The checked-in `memory-chromium.json` records process RSS, virtual sizes and
large anonymous mappings. Reserved address space with zero resident bytes is
not committed WASM memory.

JSON files in this directory contain benchmark timings and comparisons,
without redistributing game fixtures or native checkpoint contents.
