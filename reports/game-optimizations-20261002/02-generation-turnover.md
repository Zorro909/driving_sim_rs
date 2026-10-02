# 02: Generation turnover

Findings 13, 18 and 23 in the [overview](README.md). These costs fall between generations rather than on every tick. During fast training with short generations, every pause between generations is time the simulation is not running.

## What happens at turnover

`TrainGameManager.SetNextGeneration` (✓ decompiled), for each evolution:

1. `ComputeRewardsAndGetHighest` computes the rewards of every agent, then sorts them.
2. Generation statistics are recorded and autosave may run. Autosave loads the user's evolution from disk (`Repositories.EvolutionUser.Load`).
3. `EvolutionManager.Reproduce`:
   - `FilterAlgorithm.Apply`
   - `RewardAlgorithmUtils.Apply` (rewards computed a **second** time)
   - selection, preserved parents (LINQ `OrderByDescending`)
   - crossover, mutation, weight decay
4. On random tracks: `TrackFactory.GenerateRandomTrack`, then `TrackManager.Init`, then `VehicleManager.Init` and clearing the skidmarks, all on the main thread.
5. `AgentsManager.InstantiateNextGeneration` computes each network's novelty and assigns networks to vehicles.

Settings are read with `Repositories.Settings.Get()` more than ten times along the way.

## Rewards

### Game

`RewardAlgorithmUtils.Apply` (✓):

- For each reward model it builds two arrays (normalized and absolute). That part is reasonable.
- For **each agent** it then builds a `Dictionary<RewardModel, double>` of that agent's weighted parts, and computes two weighted sums with `normalizedRatios.Sum(...)`. Each sum looks up `rewards[ratio.Key]` for every term.

Callers (✓):

| Caller | When | Extra work |
|---|---|---|
| `GameManager.UpdateRewards` → `ComputeRewardsAndGetHighest` | every 60 physics ticks, for each evolution | sorts all results |
| `TrainGameManager.SetNextGeneration` → `ComputeRewardsAndGetHighest` | turnover | sorts all results |
| `EvolutionManager.Reproduce` → `Apply` | turnover, right after the call above | |
| `AgentsManager.GetRewardResultByAgent` | saving the selected agent | computes every agent, returns one |

### Suggestion

- Keep one `RewardTable` per evaluation, with flat arrays: `double[] relative`, `double[] absolute`, and a `double[models * agents]` of the weighted parts.
- Sum the terms in the same order as `normalizedRatios` is enumerated today. That keeps the result exact.
- Build the per-agent dictionaries only when the UI asks for one agent's breakdown.
- At turnover, compute the table once in `SetNextGeneration` and pass it to `Reproduce`. `Reproduce` filters the agents first, so either compute the table for the filtered set, or check that `FilterAlgorithm` keeps everyone in the common case and reuse the table then. ? The rank normalization depends on which agents are included, so a table for all agents does not equal a table for the filtered agents in general.

✓ The Rust port does this with arrays (`evolution.rs`, `rank_normalize` and `reward_values`). Its rank normalization gives the same values as the game's `NormalizeToUniform01`.

Impact: low to medium. The every-60-ticks call matters most with large populations, since it runs during the simulation.

## Reproduction

### Game

For each child (✓):

1. **Crossover** (`CrossoverAlgorithmUtils`):
   - `None`: `GetDeepCopy` of the parent.
   - `SinglePoint` and `Uniform`: `GetVector()` on both parents, a new `double[]`, then `NeuralNetwork.FromVector`. `GetVector` and `FromVector` copy elements one at a time through MathNet's indexer.
2. **Mutation** (`MutationAlgorithmUtils.ApplyXavier`), per layer:
   ```csharp
   network.Weights[i].Clone() + DenseMatrix.CreateRandom(rows, cols, new Normal(0.0, stddev))
   ```
   That is three matrices per layer: the clone, the noise matrix and the sum. The clone is not needed, because `+` already returns a new matrix. The biases work the same way.
3. **Weight decay** (`WeightDecayAlgorithm.Apply`): when `decayRate <= 0` it still calls `GetDeepCopy()`. Otherwise it creates a scaled copy of every matrix.

So each child is copied at least three times, and every copy allocates new MathNet objects. Everything runs on the main thread.

Novelty (`AgentsManager.InstantiateNextGeneration`, ✓):

- `CalculateAverageVector` calls `networks[0].GetVector()` only to get the length, then `GetVector()` for each network.
- `MathUtils.CalculateVectorDistance` computes `Mathf.Sqrt(v1.Select((t, i) => Mathf.Pow(t - v2[i], 2.0)).Sum())`, which flattens each network once more and goes through LINQ.

### Rust port

- `breed` (`evolution.rs`) first adds up how many random numbers each child needs. It draws all of them in one go, in the order the children would have drawn them one after another (`standard_normals_into`). Then it mutates the children in parallel, each reading its own slice of the noise.
- With crossover `None`, `mutated_copy` writes the child in one pass: copy, mutation and decay together. Each parameter becomes `p + (0.0 + z * deviation)`, multiplied by `1 - decay` if decay is positive. That is the same arithmetic as the game's `Clone() + CreateRandom(...)` followed by decay. The `0.0 +` reproduces MathNet's `Normal` sample, `mean + stddev * z`.
- Novelty: `install_with_novelty` (`training.rs`) computes the mean vector column by column in parallel, then each network's distance in parallel.

### Suggested C# implementation

Use the flat parameter arrays from [01](01-network-inference.md):

```csharp
// 1. Draw the noise serially, in the same order as today.
// 2. Mutate the children in parallel.
double[][] noise = new double[children][];
// DrawLikeCreateRandom fills each layer like CreateRandom does (see below) and
// writes the values into the flat GetVector layout.
for (int c = 0; c < children; c++) noise[c] = DrawLikeCreateRandom(shape, stddevs[c], rng);

Parallel.For(0, children, c =>
{
    double[] src = parents[c], dst = childParams[c];   // dst reused from a pool
    double keep = 1.0 - decay;
    for (int k = 0; k < src.Length; k++)
    {
        double v = src[k] + noise[c][k];               // noise = MathNet samples: 0.0 + stddev * z
        dst[k] = decay > 0.0 ? v * keep : v;
    }
});
```

Points to keep in mind:

- **Order of random numbers.** ✓ `DenseMatrix.CreateRandom` fills the matrix in MathNet's storage order, which is column-major. The game also draws all weight matrices first, then all bias vectors. The Rust port's exact replay of the game's random streams (`Network::mutate_xavier_game`) relies on this: it reads weight noise as `noise[col * rows + row]` and writes it into the flat layout at `row * cols + col`. ? The game seeds `Normal` from MathNet's shared default source, so runs cannot be replayed today anyway. If replays are not needed, the draw order can change freely. If they are, keep the order above.
- **Exact arithmetic.** MathNet's sample is `mean + stddev * z` with mean 0. Write the noise as `0.0 + stddev * z` to get the same rounding.
- **Draw noise in the same batches.** ✓ When MathNet fills a whole array, it first fetches about `4 * count / π` uniform numbers (rounded up to an even count) and then runs the polar method over them. The Rust port copies exactly this (`GameRandom::normal_array`). A loop of single `Sample()` calls consumes the stream differently. To keep the stream, keep one array fill per layer with the same length, but fill a plain `double[]` (for example `Normal.Samples(rng, buffer, 0.0, stddev)`) instead of creating a matrix.
- **Crossover.** With flat arrays, `SinglePoint` becomes two `Array.Copy` calls, and `Uniform` becomes a loop that draws one random number per element in the same order as today.
- **Weight decay.** Skip the copy when decay is 0.
- **Pooling.** Reuse last generation's parameter arrays for the new children instead of allocating new ones. A network is no longer needed once all of its children are written.
- **Novelty.** Flatten each network once (or use the flat array directly). Compute the mean in parallel over parameter columns, adding networks in index order. Compute each network's distance in parallel, with a plain loop that adds `Math.Pow(d, 2.0)` in index order. ? `d * d` is very likely equal to `Math.Pow(d, 2.0)` for every input on .NET 8, but confirm it on the target platforms first. ? Godot's `Mathf.Pow(double, double)` forwards to `Math.Pow`; the Godot source was not checked here.

Impact: medium. This scales with population times parameter count. ✓ In the Rust benchmark, a whole turnover for 128 cars with a 1,661-parameter network takes 3.7 ms (`reports/cpu-port-bench.json`, `turnover_seconds`). ? The game's current turnover time was not measured.

## Background track preparation

### Game

With random tracks, `SetNextGeneration` (✓) runs synchronously:

1. `TrackFactory.GenerateRandomTrack(config)` retries up to 200 attempts at a closed path, then assigns surfaces and blocks.
2. `TrackManager.Init(track)` (✓):
   - clears `SurfaceCache`,
   - `TileMap.DrawTrack(track)`,
   - `SetPathByTrack`: builds `Curve2D`, bakes it and copies the baked points,
   - updates the background sprite,
   - `new BspTreeRaycaster(TileMap.GetCollisionBspTreeSegments(track))`. The BSP build runs `FindBestPartition`, which tries every segment as a splitter and classifies every other segment against it: O(m²) per node.
3. `VehicleManager.Init` and `SkidmarksContainer.ClearChildren()`.

The simulation is stopped for all of this.

### Rust port

`training_tracks.rs` runs a bounded background producer (`--track-buffer-size`). While generation N runs, it generates track N+1 and builds its BSP, ray tree, segment grids and tile table. The trainer only takes the finished track at turnover.

### Suggestion

Split `TrackManager.Init` into a pure part and a scene part:

```csharp
sealed record PreparedTrack(Track Track, Vector2[] BakedPoints, BspTreeRaycaster Raycaster, TrackSurface[] SurfaceGrid);

// Worker thread, started at the beginning of each generation:
Task<PreparedTrack> next = Task.Run(() =>
{
    Track t = TrackFactory.GenerateRandomTrack(config, rng);
    var segments = BuildBspSegments(t, collisionPolygonsByTile);   // from cached TileSet data
    return new PreparedTrack(t, BakePath(t), new BspTreeRaycaster(segments), BuildSurfaceGrid(t));
});

// Main thread at turnover:
PreparedTrack p = next.Result;      // normally already finished
TrackManager.Install(p);            // TileMap.DrawTrack, sprite, Path2D.Curve, swap fields
```

Things that need care:

- `GetCollisionBspTreeSegments` reads `TileData` from the `TileMap` (✓), which is scene state. The collision polygons depend only on the tile type, so read them once on the main thread (for example in `_Ready`), and build the segments for a new track from that cache on the worker.
- ? Building a `Curve2D` (a `Resource`) on a worker thread is probably allowed, but Godot does not guarantee thread safety for every resource. The safe option is to port the bake to C# (the Rust port does this exactly in `curve.rs`), or to bake on the main thread. Baking is cheap compared with the BSP build.
- `TileMap.DrawTrack`, the sprite, `VehicleManager.Init` and the skidmarks stay on the main thread.
- The random number generator for tracks must be separate from the one used for evolution if the order of draws matters.

Impact: medium. It removes the pause at turnover, which matters most with short generations. It does not speed up the simulation itself.

## Smaller turnover items

- Autosave: `Repositories.EvolutionUser.Load` reads from disk at every turnover when autosave is enabled. Keep the loaded object in memory, or move the save to a background task. (✓ the call is present; ? its cost was not measured.)
- Read the settings once at the start of `SetNextGeneration` and pass the values along (finding 10).
- `UpdateTimeLimit` (every 6 ticks) evaluates a LINQ `All` over every agent. An active-agent counter, updated when an agent is eliminated, makes it O(1).
