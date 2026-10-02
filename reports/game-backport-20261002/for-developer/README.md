# Two exact speed-ups for training

These are two small code changes. Together they make training about **37% faster with 400 cars** and **52% faster with 1,600 cars** on the trained rally network. Every network output and sensor value stays bit-for-bit identical, so training results do not change.

| Change | Files | Training speed, 400 cars |
|---|---|---:|
| Network forward pass without MathNet temporaries | `NeuralNetwork.cs` | +17% |
| Direction and curvature sensors share one path query | new `PathQueryCache.cs`, both sensors | +10% |
| Both | | **+37%** |

## The changes

- **`changes.diff`:** all changes, as a diff against the decompiled current Steam release. Your source may be formatted differently, so read it rather than apply it.
- **`changes/NeuralNetwork.Forward.cs`:** the new `Forward` method and its two scratch fields. They replace the current `Forward`. The file is a `partial record` only so the test can compile it.
- **`changes/PathQueryCache.cs`:** a new class. The two sensors call it instead of `Curve2D.GetClosestOffset` and `SampleBakedWithRotation`; those are 5 lines in the diff.

### 1. Forward pass (`NeuralNetwork.Forward`)

- **What changes:** the current version allocates four matrices per layer, about 30 KB per call for an 8-layer network. With 400 cars that is roughly 2.5 GB/s of garbage across the worker threads. The new version loops over MathNet's storage arrays directly with two reused per-thread buffers. It allocates only the returned array.
- **Why the results stay exact:** it computes the same sums in the same order as MathNet's managed provider, which the game uses:
  - for each output, `num = 0.0; num += x[r] * W[r, k]`;
  - then `0.0 + num`;
  - then `+ bias`;
  - then `Math.Tanh`.
- **Constraint:** exactness holds against MathNet's managed provider, which is the only one the game ships. If a native provider (MKL/OpenBLAS) were ever enabled, the old code would change its rounding and the two would no longer match.
- **Behavior change:** a wrong input length now throws an `ArgumentException` with a clear message, where MathNet threw its own `ArgumentException`.

### 2. Shared path query (`PathQueryCache`)

- **What repeats today:** `CorrectDirectionSensor` and `TrackCurvatureSensor` run back to back for the same car on the same thread. Both compute `GetClosestOffset` for the same point, and both sample the transform at that offset.
- **How the cache works:** `PathQueryCache` remembers the last offset query and the last two samples on each thread, so the second sensor reuses the first one's answers.
  - That saves 1 of 2 offset queries and 1 of 3 samples per car tick, which are the measured hit rates (50% and 33%).
  - Keys are the curve instance and the exact float bits, so a hit returns exactly what Godot returned for the same question.
- **Assumption:** a curve's points are not edited in place while it is in use. Today a new `Curve2D` is created per track. If that ever changes, call `PathQueryCache.Clear()`.
- **Scope:** it only helps networks that use both path sensors. The default sensor set uses neither.
- **Alternative:** computing the closest offset once per car per tick and passing it to both sensors would give the same saving.

## How to test

**Forward pass, without the game:**
```
cd test
dotnet run -c Release
```
- It needs only the .NET 8 SDK and MathNet.Numerics 6.0.0-beta1 from NuGet.
- It compiles `changes/NeuralNetwork.Forward.cs` unchanged and compares it with the current MathNet version over 20,000 random networks, including ±0, ±1e±300 and subnormal values.
- Output on our machine:
```
MathNet provider: Managed
Exactness: 3,493,805 outputs compared, 0 differ.
Speed, shape [20,16,16,16,16,12,12,12,8,5], one thread:
  MathNet: 4.05 µs per call, 29,624 bytes allocated
  new:     2.19 µs per call, 64 bytes allocated (1.8x faster)
```

**Path query cache, in the game:** build with `PATH_QUERY_CACHE_VERIFY` defined. Every cache hit is then recomputed with Godot, and any difference is reported with `GD.PushError`. Train for a while with a mutated population.

## How it was measured

- **Setup:** the same changes were added to the released game. Training throughput was measured through a mod with each change switched on and off, interleaved, in the same session:
  - Windows build under Proton, Ryzen 9 7950X;
  - track A07 Three Terrains;
  - trained rally network: 20 inputs, 8 hidden layers, both path sensors;
  - mutation 0.1, 16× requested;
  - 15 s per run.
- **Exactness in the game:** a verify mode compared every new result bit for bit with the old code: 4.29M forward passes and 8.64M cache hits, with generation turnover. 0 differed.

| 400 cars | Car ticks/s (runs) | vs before | Worker-thread CPU per car tick |
|---|---|---:|---:|
| before | 83.5k, 83.2k, 84.2k, 81.2k | | 78 µs |
| forward pass | 95.1k, 99.5k | +17% | 70 µs |
| path query cache | 94.3k, 87.6k | +10% | 61 µs |
| both | 115.4k, 114.1k, 111.7k, 112.0k | **+37%** | 51 µs |

| Other cases | Before | After | Change |
|---|---:|---:|---:|
| 1,600 cars, both changes | 64.8k | 98.5k | +52% |
| 8 hidden layers on a 2-input network, forward pass | 146.7k | 229.6k | +57% |
| Small network with both path sensors, path cache | 114.9k | 139.1k | +21% |

A car tick is one car for one physics tick. Run-to-run noise is about ±5%.

## Further opportunities

These were seen in the measurements but are not part of this bundle:
- **The closest-offset query:** the path sensors still cost about 28 µs per car tick. Godot's `GetClosestOffset` checks every baked segment of the track, so its cost grows with track length. A grid over the baked segments that skips segments exactly would cut most of it and keep the results identical. Our Rust reimplementation of the simulation does this.
- **The fixed cost per car:** a 2-input network on a car that stands still still costs about 14 µs per car tick. That covers the force phase, the network wrappers and the engine calls.
- **Idle cores:** the tick runs several parallel phases one after another, and only about 8 of 32 cores are busy.
