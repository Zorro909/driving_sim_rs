# CPU driving and training: game vs. Rust port

Date: 2026-10-02

This report compares the game's CPU driving and training code with the Rust port in this repository (`altd_sim`). It lists optimizations the game's developer could apply, ordered from least to most complex. Each finding gets a short overview here. Findings that need more explanation link to a detailed document.

## Sources and method

- **Game:** `AILearnsToDrive.dll` (Godot 4.3 Mono, .NET 8, C#), decompiled with `ilspycmd` into `/tmp/ald-decomp`. File paths below use the decompiled namespace folders, for example `AILearnsToDrive.Lib.Vehicles/Vehicle.cs`. Names in the original source should match; line numbers may not.
- **Rust:** `src/` in this crate, plus the measurements in `../docs/benchmarks/README.md` and `reports/cpu-port-*.json`.
- **No game profiling was done.** The impact ratings come from counting work per car per tick and from what the Rust port measured. They are estimates. Profile before committing to the larger items.

Confidence markers:

| Marker | Meaning |
|---|---|
| ✓ | Verified by reading the decompiled game code and the Rust code |
| ? | Inferred from engine or library behaviour that was not checked in this codebase |

**Exactness.** The Rust port aims to reproduce the game bit for bit. ✓ Its fidelity experiment (`../experiments/driving_sim_rs_fidelity/README.md`) reports zero error for every captured position, velocity, rotation and network output on three recorded game episodes (A01, A07, B06). Most of the port's optimizations are tested to give identical results to its straightforward version. Every finding below says whether it changes the game's results. "Exact" means the same floating-point operations run in the same order, so trained networks, replays and saved runs behave the same.

## Detailed documents

| # | Document | Covers findings |
|---|---|---|
| 1 | [01-network-inference.md](01-network-inference.md) | 17 |
| 2 | [02-generation-turnover.md](02-generation-turnover.md) | 13, 18, 23 |
| 3 | [03-per-tick-pipeline.md](03-per-tick-pipeline.md) | 14, 19, 20 |
| 4 | [04-spatial-queries.md](04-spatial-queries.md) | 11, 12, 16, 21, 22 |
| 5 | [05-headless-simulation.md](05-headless-simulation.md) | 24, 25 |

## Main implementation differences

| Area | Game | Rust port | |
|---|---|---|---|
| Physics | `RigidBody2D` with a custom integrator, stepped by GodotPhysics2D. Forces are applied through `PhysicsServer2D`. | Ports Godot's float32 SAT and sequential-impulse solver (`collision.rs`) and its BVH broadphase (`broadphase.rs`). Each car owns its state in plain fields. | ✓ |
| Tick order | Tick-major. Every physics tick runs a parallel inference phase, a parallel force phase, the engine step, then sequential statistics. | **Independent mode** is car-major: one thread runs one car through a whole window of ticks (`training.rs`, `advance_window`). Lockstep and shared-broadphase modes keep the tick-major order where it matters. | ✓ |
| Inference | MathNet `Matrix<double>` per layer, with new matrices on every call. | One flat `Vec<f64>` per network, reusable scratch buffers, and an AVX2 kernel that gives identical results (`network.rs`, `network_simd.rs`). | ✓ |
| Inference batching | `BatchCount` updates `clamp(round(8 / BatchCount), 1, 8)` of 8 batches per tick. | Same rule (`Schedule::infers`). | ✓ |
| Ray sensors | Pointer-based `BspTreeRaycaster` with no bounding volumes. | The same BSP, flattened into an array, with subtree bounding boxes that skip walls exactly (`bsp.rs`, `RayTree`). | ✓ |
| Path sensors | `Curve2D.GetClosestOffset`, a linear scan of every baked point, run separately by each path sensor. | A grid that skips segments exactly (`segment_grid.rs`, `curve.rs`). The result is cached per pose and shared by both path sensors (`car.rs`, `sensor_path_offset`). | ✓ |
| Score projection | Windowed scan of ±450 px around the previous offset. | The same window, scanned 8 segments at a time with AVX2 (`path_segments.rs`). | ✓ |
| Surface lookup | `ConcurrentDictionary<Vector2I, TrackSurface>`, queried per wheel per tick. | Dense tile array (`world.rs`, `TileTable`). | ✓ |
| Statistics | Sequential on the main thread every 6 ticks (`AgentsManager.UpdateGenerationStats`). | Per agent, inside each car's parallel window (`TrainingAgent::update_stats`). | ✓ |
| Reproduction | Crossover, mutation and weight decay each make a full copy of the network. Everything runs on one thread. | Mutation and decay fused into one copy. Noise is drawn serially, then children are mutated in parallel (`evolution.rs`, `breed`). | ✓ |
| Random tracks | Generated, and the BSP rebuilt, synchronously inside `SetNextGeneration`. | Generated ahead of time by a bounded background producer (`training_tracks.rs`). | ✓ |
| Extra training features | none | Multiple tracks per generation (`--tracks-per-generation`), mutation-rate schedule (`--mutation-start` / `--mutation-end`), plateau stop, checkpoints with lineage (`--resume`), GPU backend (`--gpu`). | ✓ |
| Random numbers | .NET `Random` and MathNet's sampler. | A Python-compatible generator by default. The game's .NET streams can be selected for exact replays (`TrainingRandom::Game`). | ✓ |
| Stability check | `Vehicle.GetIsStable` compares the *signed* angular velocity: `AngularVelocity < 0.001f`. | Copies this on purpose (`TrainingRunner::settle_reset`). | ✓ |

## Findings, ranked by complexity

Impact is the estimated effect on training throughput: **H**igh, **M**edium or **L**ow.

### Trivial (local edits, minutes each)

1. **Look up network outputs by index, not by dictionary.** ✓ Impact L
   `NeuralNetworkOutput.TryGetValueNorm` does a `Dictionary` lookup on every call. `Vehicle.PhysicsProcess` calls it 5 times per car per physics tick, and `AgentStats` calls it 4 more times per stats tick. Resolve the five output types to array indices (or -1) once, when the network shape is set. Exact.

2. **Cache `VehicleSurfaceProperties`.** ✓ Impact L
   `TrackSurfaceConstants.GetVehicleSurfaceProperties` builds a 12-field struct from Godot-exported properties every physics tick for every car. Build it once in `_Ready`, and rebuild it only when one of those properties changes. Exact.

3. **Skip the handbrake `Ease` when the handbrake is released.** ✓ Impact L
   `VehicleConstants.GetLateralGrip` calls `Mathf.Ease(handbrake, 0.3)` (a `pow`) for every wheel on every tick. `Ease(0, 0.3)` is exactly 0, which is the most common input, so the `pow` can be skipped in that case. Exact.

4. **Read the car's transform once per sensor pass.** ✓ Impact L–M
   `RayCastSensor.GetNormalizedValue` reads `vehicle.GlobalPosition` and `vehicle.Transform` for every ray. Each read is a call into the engine. Read both once per car per inference and pass them to every ray. Exact, because the transform does not change during the input phase.

5. **Use a ring buffer for the moving score difference.** ✓ Impact L
   `AgentStats.UpdateMovingScoreDiff` uses a `Queue<double>` and LINQ `Sum()` on every stats update for every agent. A fixed `double[10]` summed from oldest to newest gives the same result without an enumerator allocation. Exact. The Rust port sums in the same order.

6. **Cache the steering wheel.** ✓ Impact L
   `Vehicle.GetWheelAngleDegrees` runs a LINQ `First` on every call. Store the steering wheel's index once. Exact.

7. **Compute network novelty without LINQ and repeated flattening.** ✓ Impact L
   `AgentsManager.InstantiateNextGeneration` calls `CalculateAverageVector`. That function flattens a whole network just to read its length, then flattens every network again. Next, `MathUtils.CalculateVectorDistance` flattens each network a second time and sums `Mathf.Pow(d, 2.0)` through LINQ. Flatten each network once, use a plain loop, and run it in parallel per network as the Rust port does (`install_with_novelty`). Exact if the summation order and `Math.Pow` are kept. ? Replacing `Math.Pow(d, 2)` with `d * d` is very likely exact too, but check it on the target C runtime first.

8. **Remove the O(N²) `FreeUnusedVehicles`.** ✓ Impact L
   It calls `List.Remove` in a loop. Use `RemoveAll` with a set, or swap-remove. Exact, as long as the order of the remaining vehicles is kept: `RemoveAll` keeps order, swap-remove does not.

9. **Replace `node == null` with `node is null` in the BSP.** ✓ Impact L
   `BspNode` is a `record`, so `==` calls the record equality operator on every node visited during raycasts and nearest-wall queries. Exact.

10. **Cut per-tick allocations and repeated settings reads.** ✓ Impact L
    `GameManager` allocates a new `GameTickArgs` on every physics tick. Tick callbacks and `SetNextGeneration` call `Repositories.Settings.Get()` many times (more than 10 times in `SetNextGeneration` alone). Reuse one args object and read the settings once per callback. Exact.

**Probable bug (not an optimization):** `Vehicle.GetIsStable` tests `AngularVelocity < 0.001f` with no absolute value, so a car spinning clockwise counts as stable. ✓ Fixing it changes when generations start. The Rust port copies the current behaviour and would need the same change.

### Low (one subsystem, about an hour each)

11. **Store track surfaces in a dense tile array.** ✓ Impact L–M → [04](04-spatial-queries.md#surface-lookup)
    `TrackManager.GetSurfaceAt` does a `ConcurrentDictionary` lookup for every wheel on every tick. That is 4 lookups per car per tick, plus `PointToCoords`. Tracks are small, bounded tile grids, so an array indexed by `(x - x0, y - y0)` built in `TrackManager.Init` is enough, as Rust's `TileTable` shows. Exact.

12. **Reuse track queries within a tick.** ✓ Impact M → [04](04-spatial-queries.md#per-tick-memoization)
    The direction and curvature sensors both call `GetClosestTrackPointInsideTile` and `Curve2D.GetClosestOffset` on the same point. On stats ticks, the wall-distance sensor and `AgentStats.UpdateDistanceFromWall` both run `Raycaster.FindClosestPoint` from the same position. Compute each once per car per tick. Exact. Rust caches the path offset per pose.

13. **Compute rewards once, without dictionaries per agent.** ✓ Impact L–M → [02](02-generation-turnover.md#rewards)
    `RewardAlgorithmUtils.Apply` builds a `Dictionary` for every agent and sums through LINQ. Callers recompute it again and again:
    - `ComputeRewardsAndGetHighest` and then `EvolutionManager.Reproduce` at every turnover.
    - `UpdateRewards` every 60 ticks, followed by a sort.
    - `GetRewardResultByAgent`, which computes every agent's reward to return one.

    Use arrays, compute once per turnover, and pass the result along. Exact if the summation order is kept.

14. **Partition the parallel loops by range.** ✓ Impact L–M → [03](03-per-tick-pipeline.md#partitioning)
    `BatchPool.ApplyOnBatchParallel` (`Parallel.For`) and `VehicleManager.ApplyPhysicsProcess` (`Parallel.ForEach` over a `List`) call a delegate for every car, and each call does very little work. Use `Partitioner.Create(0, n, chunk)` or another fixed range per worker. Exact.

15. **Skip cosmetic work during fast or unwatched training.** ✓ Impact L–M
    Several things do work that does not affect the simulation:
    - `VehicleManager.UpdateSound` runs a LINQ group-by every 6 ticks.
    - `UpdateVehicleZIndices` sorts every agent through LINQ every 18 ticks.
    - Camera updates, sensor redraws, skidmarks and per-frame wheel and icon updates (`Vehicle._Process`) run all the time.

    Turn these off or throttle them at high speed or when the window is hidden. Exact for the simulation.

16. **Make the spawn-overlap check cheaper.** ✓ Impact L–M → [04](04-spatial-queries.md#car-to-car-queries)
    While vehicle collisions are enabled, `VehicleManager.UpdateClearedOverlaps` compares every car that has not cleared its overlap against every other car on every tick. It reads `Node2D.Position` (an engine call) inside the inner loop, so it costs up to O(N²) engine calls per tick until the cars separate. Cars spawn on the same point when the starting grid is off, so this case is common. Read positions into an array once per tick and use a uniform grid. `UpdateVehicleInteractions` also rebuilds `ObstacleSnapshot` through LINQ every tick; reuse that array. Exact.

### Medium (several files, careful exactness work)

17. **Run inference without allocations, optionally with SIMD.** ✓ Impact **H** → [01](01-network-inference.md)
    `NeuralNetwork.Forward` creates at least four temporary MathNet matrices per layer. With the usual 8-layer network that is more than 30 matrices per car per inference, at up to 60 inferences per car per second. Keep each network as one flat `double[]` in the existing `GetVector` layout, and run a plain loop over reusable buffers in the same summation order. The Rust port shows this is bit-exact against the game. `Vector256<double>` can then process 4 outputs at once and stays exact.

18. **Make reproduction and novelty cheaper.** ✓ Impact M (at turnover) → [02](02-generation-turnover.md#reproduction)
    For each child, crossover, then mutation (`Clone()` plus a random matrix), then weight decay (a full copy, even when decay is 0) each produce a new network. `GetVector` and `FromVector` copy elements one by one through MathNet's indexer. All of this runs on one thread. With flat arrays, one pass can mutate, decay and write the child. Rust draws all the noise in order first, then mutates children in parallel, which keeps the random-number order.

19. **Parallelize agent statistics and run each car's tick work in one task.** ✓ Impact M → [03](03-per-tick-pipeline.md#parallel-statistics)
    `UpdateGenerationStats` runs `AgentStats.UpdateStats` sequentially for every agent. That includes a path projection, a BSP nearest-wall query and `Curve2D.SampleBaked`. Most of this only touches one agent's data. Run the pure part in parallel, then apply the side effects (icons, achievements, idle elimination, lap observers) on the main thread in agent order. Exact if the side effects keep their order.

20. **Make fewer engine calls in the vehicle physics.** ✓ Impact M → [03](03-per-tick-pipeline.md#engine-calls)
    For every car on every tick, `ImpulseAccumulator.Flush` makes two `PhysicsServer2D` calls, and `ClampVelocity` may call `BodySetState`. `_IntegrateForces` calls `GetContactColliderObject(i) is Vehicle` for every contact, which wraps an object only to test its type. Apply impulses through the `PhysicsDirectBodyState2D` already passed to `_IntegrateForces`, or merge them into one call, and compare collider IDs instead of objects. ? Moving the impulses into the integrator changes when they are applied, so check exactness against a replay.

21. **Narrow the path-sensor projection.** ✓ Impact M–H → [04](04-spatial-queries.md#path-projection)
    `CorrectDirectionSensor` and `TrackCurvatureSensor` call `Curve2D.GetClosestOffset`, which scans every baked point of the track, for every car on every inference. Score tracking already uses a ±450 px window (`TrackManager.GetClosestOffsetNear`). Either reuse a similar window, or add a grid of baked segments that skips segments exactly, as Rust's `SegmentGrid` does. Only the grid version is guaranteed exact. This only matters for networks that use these sensors; the default sensor set does not.

### High (new data structures or concurrency)

22. **Flatten the BSP raycaster into an array with exact bounding boxes.** ✓ Impact **H** → [04](04-spatial-queries.md#bsp-raycaster)
    `BspTreeRaycaster` is a pointer-linked tree with no bounding volumes, so a ray explores every subtree on its side of each partition line. Rust stores the same tree in preorder in one array, with float32 wall data and per-node and per-subtree boxes. A box test skips exactly the walls whose intersection test would fail anyway. The hit is identical. Ray sensors are usually most of a network's inputs, so this is likely one of the largest per-tick costs.

23. **Prepare the next random track in the background.** ✓ Impact M (removes turnover pauses) → [02](02-generation-turnover.md#background-track-preparation)
    On random tracks, `SetNextGeneration` calls `TrackFactory.GenerateRandomTrack` and `TrackManager.Init` synchronously. `Init` redraws the TileMap and builds a new `BspTreeRaycaster`, which costs O(n²) or more in wall segments. Generate the track, its BSP, its baked path and its surface table on a worker thread during the current generation. At turnover, only swap them in and update the scene on the main thread.

### Very high (architectural)

24. **Train in a headless simulation outside the scene tree.** ✓ Impact **H** → [05](05-headless-simulation.md)
    During training, cars only collide with static walls (or optionally with each other), and the scene already uses a custom integrator. A plain C# simulation, with no nodes and no `PhysicsServer2D`, can step each car independently. One worker can then run one car for a whole window of ticks, as Rust's independent mode does. That removes per-tick barriers, engine calls and cache misses. The Rust port shows that Godot's solver can be reproduced exactly. Visual cars would replay the state for display.

25. **Optional GPU backend.** ✓ Impact very high at large populations; very high cost → [05](05-headless-simulation.md#gpu-backend)
    The Rust port has an exact HIP simulator (`gpu_sim.rs`, `gpu/`). It only pays off at populations far beyond a typical game session, and it depends on the headless simulation (24). It is listed for completeness.

## Suggested order

1. Findings 1–10 and 15: small, safe and exact.
2. Finding 17 (inference): likely the largest single gain, and exact.
3. Findings 22 and 21 (sensor queries), then 11–12.
4. Findings 19, 20 and 14 (the per-tick pipeline), then 13, 18 and 23 (turnover).
5. Consider finding 24 only if the gains above are not enough.

## Reference: measured Rust throughput

These figures come from this repository's benchmarks. They show what a game-exact CPU implementation can reach; they are not a prediction for the game.

| Case | Car ticks per second |
|---|---|
| 1 thread, 1,000 cars (`../docs/benchmarks/README.md`) | about 306,000 |
| 32 threads, 1,000 cars, 30 s window (`../docs/benchmarks/README.md`) | about 5.77 M, 96× real time |
| 2 threads, 128 cars, 1,661-parameter network (`reports/cpu-port-bench.json`) | about 1.05 M |
