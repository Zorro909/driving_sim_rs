# Performance measurements

All timings below describe the stated hardware and inputs. Generated-track benchmarks in this tree provide a repeatable public workload; historical campaign/checkpoint measurements are retained for context and cannot be reproduced from public assets alone.

## Native CPU baseline

Measured on an AMD Ryzen 9 7950X, Linux x86_64, rustc 1.94 and LLVM 21.1.8, with local `target-cpu=native`. Each workload had an untimed warmup and seven measured samples. Worker processes used eight distinct physical cores. Desktop activity and a concurrent GPU training process remained active.

The fixed bench workloads used 1,024 cars, 3,600 timed ticks, two warmup ticks, a fixed seed/network/model and independent execution. The generated scene used seed 1729, generation zero, slot zero. The random batch used 512 cars, three generations, three tracks per generation, 3,600 ticks per track and eight workers. Random timing includes simulation, turnover and track installation waits. Throughput uses the requested car-tick count; elimination can reduce actual executed work.

| Workload | Workers | Median seconds | Requested car-ticks/s |
| --- | ---: | ---: | ---: |
| Captured campaign scene | 1 | 7.474565 | 493,193 |
| Captured campaign scene | 8 | 0.987717 | 3,732,242 |
| Generated scene | 1 | 7.508362 | 490,973 |
| Generated scene | 8 | 1.004230 | 3,670,872 |
| Random training batch | 8 | 4.673350 | 3,549,660 |

An identical-source repeat changed the medians by -0.15%, -0.95%, +0.06%, -2.33% and -0.37%, respectively. This measures noise; it is not a speedup claim. Public benchmark recipes use different generated geometry and Xavier weights, so their numbers need their own baseline.

```sh
cargo build --release --offline
python3 tools/benchmark.py --population 1024 --ticks 3600 --repeats 7 \
  --threads 1 8 --report target/cpu-benchmark.json
```

The runner works from any directory, keeps every JSON sample and reports medians. `--scene`, `--spawn-trace`, `--network`, `--model`, `--modes` and `--binary` select another workload. It measures the CLI simulation interval; process launch, loading and the initial population are outside that interval.

## Historical HIP measurements

AMD Radeon RX 7900 XTX, gfx1100, HIP 7.1.52802, ROCm clang 20 and Ryzen 9 7950X with eight Rayon workers. Inputs were a captured Rally campaign scene and a frozen 8,192-network checkpoint, repeated for larger populations. Shape was `20,16,16,16,16,12,12,8,5`, with 1,661 double parameters per car and two inference batches. The desktop remained active.

Full generation medians after one warmup and three measured runs were 2.094 seconds at 8,192 cars, 5.715 at 32,768 and 40.315 at 262,144. Wall and idle elimination were disabled; the 5,400-tick limit ended at statistics callback 5,406. Timers included packing, transfers, simulation and state import, and excluded reset, hashing, reproduction and checkpoint I/O. A later kernel investigation recorded production wall times of about 1.86, 4.93 and 36.2 seconds with unchanged digests, but did not record the same repeated-median protocol. Treat those later values as diagnostic observations.

Separate turnover optimization measurements used three consecutive generations with elimination enabled, one warmup and three timed samples. Complete-generation medians changed from 0.386 to 0.267 seconds at 8,192 cars, 1.154 to 0.790 at 32,768 and 9.663 to 6.336 at 262,144. They include turnover after the third generation and exclude loading, hashing and checkpoint I/O. All 36 before/after generation pairs matched result digests and executed ticks. This workload ended early and must not be compared as a full 5,400-tick run.

Use [GPU tooling](../gpu/README.md) for fresh measurements on generated inputs. A GPU performance gate requires an otherwise idle GPU; these historical results do not establish throughput under concurrent training load.

## Historical browser measurements

Ryzen 9 7950X, Chromium 153.0.8010.12 on Linux, RX 7900 XTX with RADV 26.2.2. A Formula scene used shape `20,16,5`, two inference batches and no elimination. Four 180-tick generations ran per population; the first was warmup and the median of the next three is shown. Complete time includes evolution and coordinator yields. This was one sweep.

| Cars | Serial complete ms | Eight-worker complete ms | WebGPU with eight workers ms |
| --- | ---: | ---: | ---: |
| 32 | 12.7 | 4.9 | 365.2 |
| 256 | 100.4 | 18.3 | 361.5 |
| 1,024 | 399.5 | 66.1 | 189.0 |
| 4,096 | 1,657.3 | 263.8 | 647.7 |

At 4,096 cars, eight-worker CPU advancement took 228.9 ms versus 1,577.7 ms serial. Module/pool startup was 14.1 ms serial and 94.5 ms with eight workers. Peak observed linear memory in the largest case was 97.6 MiB serial and 113.7 MiB with eight workers, excluding JavaScript workers, GPU allocations and any separate preview module. These inputs are historical; rerun the [public browser harness](../wasm/README.md) to establish results for the generated scenarios and another browser/device.

## Firefox completion waits — 2026-10-06

Firefox 157 on the local Linux/RX 7900 XTX system reproduced the 30 ticks/s limit. In the public Formula benchmark, six-tick calls took about 200 ms at both 32 and 1,024 cars. At 32 cars, the 120 measured ticks spent 3,975 ms in advancement while GPU timestamps covered only 192 ms. Setting GPU load to 1 retained the same limit, ruling out the configured idle budget as its cause.

The runtime now prompts completion delivery with empty queue submissions every 4 ms while awaiting queued work or mapped readback. It retains the original promises and command order, cancelling the timer after completion, device loss or a submission exception. The earlier workaround was absent from this source tree. The Windows/Proton kernel sources are unchanged by this fix.

| Workload | Cars | Before ticks/s | After ticks/s |
| --- | ---: | ---: | ---: |
| Formula, fixed six-tick windows | 32 | 30.19 | 688.07 |
| Formula, fixed six-tick windows | 1,024 | 30.27 | 259.63 |
| Normally served app, Rally, adaptive windows | 512 | 29.98 | 249.53 |

Formula used the public default scenario, shape `20,16,5`, Proton, disabled elimination and 120 warmup ticks, with 120 measured ticks at 32 cars and 60 at 1,024 cars. The benchmark records per-pass GPU timestamps. The app used its Test Loop fixture, default shape `20,16,16,12,8,5`, one inference batch, eight CPU workers, Proton, load 0.85 and disabled elimination. App values are medians of the reported rates during a four-second observation; whole-observation rates were 29.88 and 258.92 ticks/s. The final served run used the rebuilt assets without package or runtime overrides. Compilation and startup are excluded. These are diagnostic before/after observations, not repeated benchmark medians.

A separate app comparison completed three generations and 558 ticks with each runtime, using identical rebuilt WASM packages and changing only the JavaScript completion waits. Checkpoint bytes, every population network, terminal car states, summaries and stable generation logs matched exactly. The completed outcome SHA-256 was `7086f0ce5b7f555d7bc2d7a4ef5c2db321d71ebb3c21dd93eaa9324ffa02eedc`. Time through the third turnover changed from 18,688 ms to 1,405 ms; the adaptive pump used 90 calls before and 37 after.

Two additional runs through normally served assets completed the same 558 ticks in 2,339 and 2,385 ms, with identical outcomes and stable logs. A prior diagnostic run with package routing reported 1,143 ticks/s; the normal served measurements above provide the conservative result. One initial instrumented run stalled while waiting for its generation outcome. Its detailed phase was not captured, and the stall did not recur in the three subsequent completed comparisons; its cause remains unresolved.

Three alternating Chromium 153 comparisons kept the WASM, shader, Win11 profile, 1,024 active cars and 120-tick bulk windows identical, changing only the completion-wait runtime. Median wall time was 355.4 ms before and 355.0 ms after. Background GPU activity was present before browser launches, so these observations do not measure isolated peak throughput.

Both serial and threaded production packages were rebuilt. Full Firefox simulation comparisons passed for all three profiles: 184,558 checks, zero mismatches. The Node completion-wait regression suite passes 16 tests against the fixed runtime; the frozen original runtime fails nine of them. See [the browser commands](../wasm/README.md) to repeat the Formula workload or run the regression suite without a GPU.
