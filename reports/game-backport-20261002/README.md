# Backporting two optimizations into the game: first iteration

Two findings from [game-optimizations-20261002](../game-optimizations-20261002/README.md) were ported into the running game (AI Learns To Drive, Godot 4.3 Mono, .NET 8). They were measured A/B with the method of [game-benchmark-20261002](../game-benchmark-20261002/README.md).

- **Finding 17:** the network forward pass without MathNet temporaries.
- **Finding 12:** the direction and curvature sensors share one path projection.

Both are bit-exact. Throughput doesn't depend on which one is on.

Markers: ✓ measured or read in the code, ? inferred, ✗ not identified.

## Short answer

1. **Training is 37% faster with 400 cars and 52% faster with 1,600.** ✓ This uses the trained rally network with mutated cars spread along the track, which is how real training looks:
   - 400 cars: 83,008 → 113,306 car ticks/s;
   - 1,600 cars: 64,846 → 98,454.
   - The .NET pool CPU per car tick falls from 78 to 51 µs at 400 cars.
2. **The forward pass alone gives +17%, and the path memo alone +10%.** ✓ Same case: 400 mutated cars.
   - Together they save about 3.2 µs of wall time per car tick, close to the sum of the separate savings (1.8 + 1.1 µs).
   - So the effects add up. The combined percentage is larger than the sum of the two percentages only because throughput is the reciprocal of time.
3. **The forward pass removes about 80% of the cost of 8 hidden layers.** ✓ For the 8-layer variant, the pool CPU per car tick falls from 47.6 to 22.9 µs. The layers cost 30.6 µs more than the network without them, so about 6 µs of that cost is left.
4. **The path memo removes about 40% of the path-sensor cost.** ✓ For the tiny network plus both path sensors with mutated cars, the pool CPU falls from 62.7 to 44.7 µs per car tick (+21% throughput). That is as expected:
   - the memo saves one of the two closest-offset queries and one of the three transform samples per car tick;
   - the hit rates are exactly 50% and 33%.
   - Making each remaining query cheaper is finding 21, which is not done yet.
5. **Results are identical.** ✓ Verify mode runs the original code next to the new code and compares every result bit by bit:
   - 4.29M forward passes and 8.64M path-cache hits were checked in the game, with mutated networks and with generation turnover. There were 0 mismatches.
   - Offline on .NET 8, 3.49M outputs of 20,000 random networks were compared, including ±0, ±1e±300 and subnormals. There were 0 mismatches.
6. **Mutation 0 overstates the path memo.** ✓ With mutation 0, all cars are identical and stand on the same spot, so the cache also hits across cars: 97–99% instead of 50%/33%.
   - The mutation-0 rows below (+60% at 400 cars, +65% for the path sensors) are therefore not representative.
   - The earlier benchmark found that stacking identical cars does not distort the baseline throughput, but it does distort this cache.

## What was changed

The game DLL gets small hooks. The optimized code lives in a separate mod DLL, and the original code stays in place as the fallback.

- **`tools/AltdOpt/Patcher`** (Mono.Cecil) turns `patched/AILearnsToDrive.dll` (SpeedMod with the MCP mod, sha256 `f5ee5bb5…`) into `patched/opt/AILearnsToDrive.dll`:
  - **New class `AILearnsToDrive.AltdOptHooks`:**
    - an `Instance` and the flags `ForwardOn` and `PathOn`;
    - static wrappers that call the engine when the hook is off.
  - **Forward pass:** the body of `NeuralNetwork.Forward` moves to the new method `ForwardOriginal`. `Forward` becomes `ForwardOn ? Instance.Forward(this, input) : ForwardOriginal(input)`.
  - **Path sensors:** `CorrectDirectionSensor` and `TrackCurvatureSensor` make 5 `Curve2D.GetClosestOffset` and `SampleBakedWithRotation` calls. These now go through the wrappers.
  - **Loader:** `Main._Ready` calls `AltdOptHooks.Load()`. It loads `AILearnsToDrive.Opt.dll` into the game's own AssemblyLoadContext if the file exists.
- **`tools/AltdOpt/Runtime`** builds `AILearnsToDrive.Opt.dll`, the hook implementations:
  - **Forward:** the same arithmetic as MathNet's managed provider, in the same order:
    - for each output, `num = 0.0; num += x[r] * W[r, k]`;
    - then `tanh((0.0 + num) + b[k])`;
    - two reused per-thread buffers, so the only allocation is the returned array.
    - Anything unusual falls back to `ForwardOriginal`: another matrix type, a shape mismatch, null input, or a provider other than the managed one. That keeps the original's errors.
  - **Path memo:** per thread, the last closest-offset query and the last two transform samples are kept.
    - Keys are the curve reference and the float bits, so a hit returns exactly what the engine would.
    - A new `Curve2D` is created per track (✓ `TrackManager`, `Track.GetPathCurve`).
  - **Config:** the mod reads `altd-opt.json` (`forward`, `path_memo`, `verify`) next to the DLL every 100 ms.
  - **Status:** it writes its counters to `altd-opt-status.json` once per second.
- **`tools/AltdOpt/Check`** is the offline exactness test: hooked forward pass against `ForwardOriginal` on random networks.
- `tools/PatchCheck` JIT-compiled the hooked types and the MCP mod against the hooked DLL on .NET 8: 1,731 methods, 0 failures.

## Results

Same setup as the earlier benchmark:
- the `zz-benchmark-20261002` network;
- A07 Three Terrains;
- 16× requested;
- no elimination or time limit;
- 5 s warm-up (8 s for mutated cars), then 15 s measured.

The configurations of each case were run interleaved. CPU per car tick includes spin-waiting.

All runs are in `results.jsonl`: one `measure()` row per run, plus a companion row with the mod's counters. The scripts are `bench.py` (groups verify, trained, mutated, path, layers, tiny, populations, trained_mutated) and `summarize.py`.

### Realistic case: mutated cars (mutation 0.1)

| Case | Config | Runs | Car ticks/s (each) | Mean | vs off | Pool CPU µs per car tick | Path cache hits (offset / sample) |
|---|---|---:|---|---:|---:|---:|---|
| Trained, 400 cars | off | 4 | 83.5k, 83.2k, 84.2k, 81.2k | 83,008 | | 77.7 | |
| | forward | 2 | 95.1k, 99.5k | 97,340 | +17.3% | 70.1 | |
| | path_memo | 2 | 94.3k, 87.6k | 90,945 | +9.6% | 60.6 | 50% / 33% |
| | both | 4 | 115.4k, 114.1k, 111.7k, 112.0k | 113,306 | **+36.5%** | 51.0 | 50% / 33% |
| Trained, 1,600 cars | off | 2 | 65.0k, 64.7k | 64,846 | | 113.2 | |
| | both | 2 | 101.8k, 95.1k | 98,454 | **+51.8%** | 80.6 | 50% / 33% |
| Tiny network + both path sensors | off | 2 | 113.1k, 116.7k | 114,886 | | 62.7 | |
| | path_memo | 2 | 140.8k, 137.4k | 139,106 | +21.1% | 44.7 | 50% / 33% |

### Identical cars (mutation 0), as in the earlier benchmark

| Case | Config | Runs | Car ticks/s (each) | Mean | vs off | Pool CPU µs per car tick | Path cache hits |
|---|---|---:|---|---:|---:|---:|---|
| Trained, 100 cars | off | 2 | 63.2k, 67.0k | 65,073 | | 70.6 | |
| | both | 2 | 96.1k, 96.0k | 96,048 | +47.6% (at the 16× cap) | 26.3 | 94% / 92% ⚠ |
| Trained, 400 cars | off | 3 | 87.2k, 82.7k, 80.3k | 83,387 | | 78.5 | |
| | forward | 3 | 96.4k, 95.6k, 92.0k | 94,656 | +13.5% | 70.1 | |
| | path_memo | 3 | 92.8k, 98.2k, 95.5k | 95,482 | +14.5% ⚠ | 48.2 | 97% / 96% ⚠ |
| | both | 3 | 131.5k, 133.2k, 134.5k | 133,096 | +59.6% ⚠ | 37.6 | 98% / 97% ⚠ |
| Trained, 1,600 cars | off | 2 | 57.9k, 57.0k | 57,450 | | 111.4 | |
| | both | 2 | 83.9k, 80.0k | 81,968 | +42.7% ⚠ | 68.0 | 99% / 99% ⚠ |
| Tiny + 8 zero-weight layers | off | 3 | 149.3k, 148.7k, 142.0k | 146,669 | | 47.6 | |
| | forward | 3 | 235.1k, 243.7k, 210.1k | 229,626 | **+56.6%** | 22.9 | |
| Tiny + both path sensors (zero weights, cars stand still) | off | 3 | 132.3k, 129.3k, 130.1k | 130,571 | | 65.2 | |
| | path_memo | 3 | 216.2k, 214.6k, 217.1k | 215,986 | +65.4% ⚠ | 29.5 | 99% / 98% ⚠ |
| Tiny network (speed only, 0 hidden layers) | off | 2 | 254.3k, 257.2k | 255,762 | | 17.0 | |
| | both | 2 | 281.8k, 270.0k | 275,906 | +7.9% | 13.9 | |

- **⚠ Cache hits across cars:** these rows include cache hits between different cars, which cannot happen in real training. The forward-pass rows are unaffected, because it has no cache.
- **Tiny network:** even a single layer gains 3 µs per car tick. MathNet allocates four matrices per layer and call (product, bias row, sum, tanh). ? This also suggests the hooks themselves cost nothing measurable.
- **Combined gain at mutation 0:** with identical cars, the combined gain at 400 cars is clearly more than additive: 4.5 µs wall per car tick saved, against 1.4 + 1.5 µs separately. ✗ The cause was not identified. The mutated runs are additive.

### Compared with the earlier benchmark

- **Baseline:** today's baseline is faster than on the earlier run. With 400 mutated cars, off gives 83.0k car ticks/s against 70.7k earlier, and with 1,600 cars 64.8k against 56.1k. ✗ The cause was not identified, so only the interleaved same-session comparisons above count.
- **Path sensors:** the earlier benchmark costed the path sensors at about 30 µs each and the 8 hidden layers at about 28 µs. Today:
  - the forward port saves 25 µs of the layer cost;
  - the path memo saves 18 µs of the ~46 µs the two sensors cost on driving cars.

## Caveats

- **Proton, one machine, one track.** These are the same caveats as the earlier benchmark. The path-sensor saving depends on track length, because ? the remaining closest-offset queries scale with the number of baked points.
- **MathNet provider.** The forward pass matches only MathNet's managed provider. The mod checks the provider at startup and keeps the original otherwise. The game ships no native provider (✓ no MKL or OpenBLAS DLL; `UseDefault` picks the managed one).
- **Path memo and thread order.** The memo depends on both sensors of one car running back to back on one thread. ✓ That gives the 50%/33% hit rates measured; if the order changed, hits would fall but results would stay exact.
- **Exactness coverage.** Verify mode was run on A07 and on mutated rally networks. Other tracks and car types were not checked in the game. Because the cache keys are exact, a hit can only return what the engine returned for the same curve and input.

## State after the benchmark

- **Installed:**
  - `AILearnsToDrive.dll` in the game folder is the hooked build (sha256 `2a460199…`);
  - `AILearnsToDrive.Opt.dll` is next to it;
  - `altd-opt.json` turns both optimizations on, with verify off.
  - The previous DLL is `AILearnsToDrive.dll.before-opt-20261002` (sha256 `f5ee5bb5…`, the same as `patched/AILearnsToDrive.dll`).
- **To turn the optimizations off:** set `{"forward": false, "path_memo": false}` in `altd-opt.json`. It applies within 0.1 s, even in the middle of a run.
- **To remove the hooks:** with the game closed, copy the backup back over `AILearnsToDrive.dll`.
- **Settings:** the training settings were restored from `settings_before.json` and checked field by field; no field differs. No training session is running.
- **Benchmark network:** `zz-benchmark-20261002` still holds the last variant tested: 1 ray, speed and both path sensors, no hidden layers.

## For the game's developer

[`for-developer/`](for-developer/README.md), also packed as `altd-speedups-20261002.zip`, has the two changes as plain source without the hooks:
- the new `NeuralNetwork.Forward`, `PathQueryCache.cs` and the 5 changed sensor lines, also as a diff against the decompiled game;
- a standalone .NET 8 test that compares the new forward pass bit for bit with the MathNet version and times both;
- a short summary of these measurements.

## Next iterations

The candidates, ordered by measured remaining cost:

1. **Finding 21: a faster closest-offset query.** About 28 µs of path-sensor cost remains per car tick. A C# projection over the baked points with a spatial grid could cut most of it. It is only exact if it reproduces Godot's `GetClosestOffset` float for float, so it needs an engine-source check first.
2. **Findings 1–4, 10 and 20: the fixed cost per car.** The base is about 14 µs of pool CPU per car tick, covering the force phase, the network wrappers and the engine calls. Today it is the largest cost left for small networks.
3. **Findings 14 and 19: turnover and the phase structure.** The tick is still latency-bound: about 8 of 32 cores are busy.
