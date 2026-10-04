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

AMD Radeon RX 7900 XTX, gfx1100, HIP 7.1.52802, ROCm clang 20 and Ryzen 9 7950X with eight Rayon workers. Inputs were a B06 Rally scene and a frozen 8,192-network checkpoint, repeated for larger populations. Shape was `20,16,16,16,16,12,12,8,5`, with 1,661 double parameters per car and two inference batches. The desktop remained active.

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
