# What limits the game's training speed

A benchmark of the running game (AI Learns To Drive, Godot 4.3 Mono, .NET 8.0.2) through the AltdMcp mod. Each factor is varied on its own while the rest stays fixed. It complements the code review in [game-optimizations-20261002](../game-optimizations-20261002/README.md), whose finding numbers are used below.

Markers: ✓ measured or read in the code, ? inferred from the measurements, ✗ not identified.

## Short answer

1. **The tick pipeline is latency-bound, not CPU-bound.** ✓ With 400 cars, the game keeps only about 7.5 of the 32 logical CPUs busy, and no thread is saturated: the main thread is busy about 70% of the time, and each .NET worker about 20%. Asking for more speed changes nothing: 4× gives 2.9×, 8× gives 3.0×, and 16× gives 3.0×. Each tick runs main-thread work, an engine phase on a second thread, and two .NET parallel loops one after another, and every car waits at each step.
2. **Every car tick costs a lot of CPU.** ✓ The trained rally network (20 inputs, 8 hidden layers) uses about **100 µs of CPU per car tick**. Even a network with 2 inputs and no hidden layers, on cars that stand still, uses about 22 µs. For comparison, the Rust port runs a whole car tick in about 3.3 µs on one thread. ? The game's figures include threads spinning while they wait, so they are upper bounds.
3. **The expensive parts are the path sensors and MathNet inference.** ✓ Each path sensor (`correct_direction`, `track_curvature`) adds about 30 µs of CPU per car tick, and the 8 hidden layers add about 28 µs. That confirms findings 12/21 (path projection) and 17 (inference). Vision rays are cheap at about 1 µs each, so finding 22 (BSP) matters less than estimated. The other sensors cost nothing measurable.
4. **Throughput peaks at about 400 cars and then falls.** ✓ It is about 72,000 car ticks/s at 400 cars, 60,000 at 800 and 51,000 at 1,600, because the CPU per car tick rises from about 100 to 140 µs. ? The cause was not identified; cache misses or GC on a larger heap are candidates.
5. **Generation turnover is minor.** ✓ It takes about 9 ms with 100 cars, 40 ms with 400 and 460 ms with 1,600, plus about 30 ms to generate a random track with 400 cars. With 5 s generations, that is at most about 7% of the wall time.
6. **Rendering does not appear to be the limit.** ? The threads outside the main thread and the .NET pool use about 0.6–1.3 cores with 200 or more cars. With 1 car, the main thread is busy only 28% of the time at the 16× cap. ✓ The game is not headless: `--headless2` has no effect, and the Vulkan swapchain threads stay active.

## Setup

- **Machine:** AMD Ryzen 9 7950X (16 cores, 32 threads), Fedora 44, kernel 7.2. The game runs as the Windows build under Proton Hotfix with `WINEFSYNC=1`, from Flatpak Steam, in a window.
- **Network:** `zz-benchmark-20261002`, a copy of `20260930-230014-generalized-model-rally-long-edaf30-latest`:
  - shape [20,16,16,16,16,12,12,12,8,5], rally car, 13 rays;
  - sensors: boost_capacity, correct_direction, grip, velocity_front, velocity_side, track_curvature, wheel_angle.
  - Variants were made with `update_network`. Where a variant had to keep the same driving, new inputs or layers got zero weights.
- **Track:** A07 Three Terrains.
- **Settings:**
  - mutation 0, so every car runs the same network; one test used mutation 0.1;
  - no elimination, no time limit, so there is no turnover during measurement;
  - requested speed 16× (960 physics ticks/s, 64 steps per frame).
- **Measurement:**
  - Each case starts a fresh session and waits 5 s.
  - It then measures for 15 s: the change in simulated time over the change in wall time, from the mod's `ThroughputMeter`.
  - It also reads the CPU time and context switches of every game thread from `/proc`.
- **Unit:** a car tick is one car for one physics tick (1/60 s of game time), so car ticks/s = speed multiplier × 60 × cars.
- **Noise:** about ±5% between runs. The ray sweep keeps the driving identical and still varies from 66k to 73k car ticks/s.

## Results

All runs: `results.jsonl`. Scripts: `sweep.py` (groups: population, population_small_net, rays, path_sensors, layers, sensors, turnover, random_tracks, multiplier, spread), `sample_states.py`, `joint_states.py`.

### Population (trained network, cars driving)

| Cars | × real time | Car ticks/s | Wall ms per tick | Main thread (cores) | .NET pool (cores) | Other (cores) | CPU µs per car tick |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.8 (capped) | 946 | 1.06 | 0.28 | 0.19 | 2.58 | n/a |
| 25 | 15.7 (capped) | 23,624 | 1.06 | 0.59 | 2.26 | 2.69 | n/a |
| 50 | 14.0 | 41,933 | 1.19 | 0.70 | 3.83 | 3.03 | n/a |
| 100 | 8.9 | 53,278 | 1.88 | 0.69 | 4.12 | 2.31 | 134 |
| 200 | 5.0 | 59,746 | 3.35 | 0.68 | 4.51 | 1.29 | 109 |
| 400 | 3.0 | 71,875 | 5.56 | 0.69 | 5.72 | 1.02 | 103 |
| 800 | 1.25 | 60,189 | 13.3 | 0.58 | 5.93 | 0.74 | 120 |
| 1,600 | 0.53 | 51,196 | 31.3 | 0.49 | 6.04 | 0.65 | 140 |

- **"Other"** covers the remaining threads: rendering and Vulkan presentation, Godot's engine threads, and audio. With few cars the game renders many frames per second, which is where the 2.6 cores go.
- **Mutated cars:** with mutation 0.1, the cars spread out along the track (scores −48 to 156). The result is about the same: 70,675 car ticks/s with 400 cars and 56,128 with 1,600. So stacking identical cars on one spot does not distort the population results.
- **Requested speed:** with 400 cars, requesting 4×, 8× or 16× gives 2.88×, 3.01× and 3.02×.

### What each input costs (400 cars, cars standing still)

The base is the network reduced to 1 ray, the speed sensor and no hidden layers; its cars do not move. Each row adds one thing with zero weights, so the cars behave exactly as in the base.

| Variant | × real time | Car ticks/s | .NET pool CPU µs per car tick | Added µs |
|---|---:|---:|---:|---:|
| base | 8.70 | 208,680 | 16.5 | |
| + grip, boost_capacity, velocity_front, velocity_side or wheel_angle (each alone) | 8.9–9.1 | 213–218k | 16.4–17.9 | ≈ 0 |
| + 12 vision rays | 7.40 | 177,620 | 28.0 | 11.5 (≈ 1 per ray) |
| + correct_direction | 5.40 | 129,613 | 46.9 | 30.4 |
| + track_curvature | 5.37 | 128,888 | 47.2 | 30.7 |
| + 8 hidden layers (16,16,16,16,12,12,12,8) | 4.90 | 117,685 | 44.3 | 27.8 |

? Both path sensors call `Curve2D.GetClosestOffset` over every baked point of the track, at the same point (see [04](../game-optimizations-20261002/04-spatial-queries.md#path-projection)). That fits their equal cost. The cost per sensor will grow with track length: A07 has 12 tiles, while A08 has 36 and B09 has 48.

### Sweeps with the trained network (400 cars, cars driving or crawling)

| Sweep | Range of car ticks/s | Note |
|---|---|---|
| Rays 1 → 13 (same driving) | 66k–73k | no trend: within noise |
| Path sensors removed / added with zero weights | 78.1k / 72.5k | pool CPU 3.4 → 5.7 cores |
| Hidden layers 0, 1, 2, 4, 8, 10 | 75k, 88k, 83k, 76k, 72k, 67k | the driving changes with the layers |

With the full network, a single change moves the wall time less than in the table above. The other parallel work already sets how long each phase takes, so the extra work mostly fills idle cores. This is consistent with point 1. Removing several costs together should still add up.

### Where the time goes within a tick (400 cars, trained network)

The run state of the main thread, the busiest engine thread and the .NET workers was sampled together about 1,500 times per second (`joint_pop400.json`). The sampling is not atomic, so the shares are rough.

| Share of wall time | What is running |
|---:|---|
| ≈ 36% | .NET parallel loops, with the main thread taking part. On average about 12 cores are busy during these phases. |
| ≈ 32% | the main thread alone: game code on the main thread and the engine's per-tick work |
| ≈ 11% | only the engine thread `2327051`, while the main thread waits on it |
| ≈ 9% | workers finishing while the main thread waits |
| ≈ 7% | nearly nothing: threads waking up or handing over work |

- **Thread `2327051`:**
  - It belongs to a block of 33 threads created at startup. All of them wake about 4 times per tick, while `2327051` does most of the work.
  - ? This looks like Godot's `WorkerThreadPool` running the 2D physics step in group tasks.
  - ✗ It is not identified. Wine does not show Godot's thread names.
- **Main thread (`states_pop400.json`, `states_pop1600.json`):**
  - With 400 cars it runs 69% of the time and waits on futexes for the rest.
  - With 1,600 cars it runs only 49% of the time, and `2327051` runs 42%.
- **.NET workers:**
  - They spend about 75% of their time blocked in a Wine server pipe read (`anon_pipe_read`).
  - ? This is how a thread-pool wait on an I/O completion port looks under Proton; fsync does not speed it up.
  - The pool has between 18 and 41 threads, more under heavier load.

### Turnover (5 s generations)

| Cars | Track | Overhead per generation | × real time with turnover | without |
|---:|---|---:|---:|---:|
| 100 | A07 | 9 ms | 9.37 | 8.88 |
| 400 | A07 | 39 ms | 2.79 | 3.00 |
| 1,600 | A07 | 460 ms | 0.53 | 0.53 |
| 100 | random, 20 tiles | 9 ms | 8.36 | 8.88 |
| 400 | random, 20 tiles | 68 ms | 2.48 | 3.00 |

- **What "overhead" counts:** the meter counts as overhead the frames in which the game is not "running and unpaused". ? Turnover work done inside a physics tick would be counted as running, so these numbers are a lower bound.
- **Random tracks:** these runs are slower beyond the turnover itself. ? The 20-tile tracks are longer than A07, so the path sensors cost more per call.

## What this means for the optimization report

| Finding | Measured | Suggested priority |
|---|---|---|
| 12, 21: path projection (memoize, then narrow) | ≈ 30 µs per sensor per car tick, the largest single cost; grows with track length | **first** |
| 17: inference without allocations | ≈ 28 µs per car tick for 8 small layers | **first** |
| 14, 19, 20: per-tick pipeline (fewer barriers, ranges, stats in parallel, fewer engine calls) | the tick is a serial chain; 70% of the CPUs are idle; throughput falls beyond 400 cars | **second.** This is what lets more cores help. |
| 24: headless simulation | removes the per-tick engine phase and its waits | still the largest structural gain |
| 22: flat BSP with boxes | ≈ 1 µs per ray | lower than estimated |
| 13, 18, 23: turnover | ≤ 7% with 5 s generations | low unless generations are very short |
| 15: cosmetic work | ? rendering does not appear to limit throughput at ≥ 200 cars | low for throughput |

The base cost of about 17 µs per car tick, for a 2-input network on a car that stands still, covers the per-car force phase, the network wrappers and the engine calls. That is the target of findings 1–4, 10 and 20.

## Caveats

- **Proton.** The game runs under Proton, not native Windows. Waking threads and waiting for them goes through Wine. ? On Windows, the gaps between phases are probably shorter, but the per-car CPU costs should be similar.
- **One machine, one track.** Only A07 was used for most runs. The path-sensor cost depends on track length.
- **CPU counts include waiting.** CPU time includes spin-waiting, so CPU µs per car tick is an upper bound on the real work.
- **No profiler.** This benchmark only uses the mod's API and `/proc`. Splitting the main thread's time into engine physics, game statistics and rendering needs the phase timers (step 2) or an EventPipe trace (step 3) proposed earlier.

## State after the benchmark

- **Settings:** the training settings were restored from `settings_before.json` and checked against it field by field; no field differs. The game is back in the main menu.
- **Benchmark network:** `zz-benchmark-20261002` is still a user network, because the mod has no delete tool. It holds the last variant tested: 1 ray, the speed sensor, and 8 hidden layers with zero weights. Delete it in the game's menu when it is no longer needed.
