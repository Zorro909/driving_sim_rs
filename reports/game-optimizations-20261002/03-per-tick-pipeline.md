# 03: The per-tick pipeline

Findings 14, 19 and 20 in the [overview](README.md). Complexity: low to medium.

## What one physics tick does in the game

Everything below is registered with `_physicsTickObserver` in `GameManager` (✓ decompiled). The numbers in brackets are the observer priority and the tick interval.

| Step | Work | Threads |
|---|---|---|
| Stats [0, every 6] | `AgentsManager.UpdateGenerationStats` → `Agent.UpdateStats` for every agent | main thread, one agent after another |
| Time limit [10, every 6] | `UpdateTimeLimit`: LINQ `All` over agents | main |
| Rewards [100, every 60] | `UpdateRewards`: rewards for every agent, then a sort | main |
| Sound [100, every 6] | `VehicleManager.UpdateSound` | main |
| Vehicle interactions [999] | `UpdateVehicleInteractions`, when car collisions are on | main |
| Camera, sensor drawing [1000] | `CamManager.Update`, `RedrawSensors` | main |
| Inference [1001] | `UpdateGeneration` → `BatchPool.ApplyOnBatchParallel`: `UpdateInput` (sensors), `UpdateActive`, `UpdateOutput` (forward pass) for the cars in this tick's batches | `Parallel.For`, one delegate per car |
| Forces [1001] | `VehicleManager.ApplyPhysicsProcess` → `Vehicle.PhysicsProcess` for every vehicle | `Parallel.ForEach` over a `List<Vehicle>` |
| Engine step | GodotPhysics2D integrates the bodies and calls `_IntegrateForces` on each car | engine |

So each tick has at least two fork-join barriers (inference, forces), plus the engine step. Every sixth tick, the main thread also runs the stats for every agent while the workers wait.

### Measured

From the [game benchmark](../game-benchmark-20261002/README.md#where-the-time-goes-within-a-tick-400-cars-trained-network): 400 cars, a trained 20-input network, 32 logical CPUs, under Proton.

- **Nothing is saturated.** ✓ The game reaches 3.0× real time while using about 7.5 cores in total. The main thread is busy about 70% of the time, and each of the 30–40 .NET workers about 20%.
- **More speed does not help.** ✓ Requesting 4×, 8× or 16× gives the same result.
- **Rough shares of a tick,** from sampling the threads' run states (✓):

  | Share | What runs |
  |---:|---|
  | ≈ 36% | the .NET parallel loops; about 12 cores busy on average |
  | ≈ 32% | the main thread alone |
  | ≈ 11% | one engine thread alone, while the main thread waits. ? It is probably Godot's `WorkerThreadPool` running the physics step. |
  | ≈ 16% | workers finishing, or threads waking up |

- **More cars do not help either.** ✓ Car ticks per second peak at about 400 cars and fall to about 51,000 at 1,600, because the CPU per car tick rises from about 100 to 140 µs.

So shortening the serial chain matters as much as making each car cheaper. That means fewer barriers, the stats off the main thread, and fewer engine calls.

## Partitioning

### Game

`BatchPool.ApplyOnBatchParallel` uses `Parallel.For(0, n, options, i => action(Values[i]))` (✓). `ApplyPhysicsProcess` uses `Parallel.ForEach(_vehicles, options, v => v.PhysicsProcess(delta))` over a `List` (✓). `Repositories.Settings.GetParallelismOptions()` returns a cached `ParallelOptions` and only updates it when the setting changes (✓), so it costs nothing.

? `Parallel.For` splits the range dynamically and calls the delegate once per index. `Parallel.ForEach` over a `List<T>` uses a partitioner that hands out small chunks under a lock. The work per car is small (a few microseconds), so the scheduling overhead is a noticeable share.

### Suggestion

Hand each worker a contiguous range:

```csharp
var ranges = Partitioner.Create(0, vehicles.Length, Math.Max(16, vehicles.Length / (4 * workers)));
Parallel.ForEach(ranges, options, r =>
{
    for (int i = r.Item1; i < r.Item2; i++) vehicles[i].PhysicsProcess(delta);
});
```

- Use an array instead of a `List` for `_vehicles` during a generation.
- Exact: each car's work is independent, and the order between cars does not matter within these two phases.

✓ The Rust port gives each worker whole cars (`agents.par_iter_mut().with_max_len(1)` in `advance_window`). That works because each task is a whole window of ticks (see below), not a single tick.

## Parallel statistics

### Game

`Agent.UpdateStats` (✓) runs for every agent on the main thread every 6 ticks:

1. `AgentStats.UpdateStats(vehicle)`:
   - collision count,
   - path projection `TrackManager.GetClosestOffsetNear` (a ±450 px window),
   - score, moving score difference (`Queue<double>` + LINQ `Sum`),
   - wall distance: `Raycaster.FindClosestPoint` (a BSP search),
   - distance to the center line: `Curve2D.SampleBaked`,
   - speed, steering, distance, drift (`atan2`), momentum, throttle, braking, idle time.
2. `Vehicle.Icons.SetState(...)`: changes the scene.
3. If active: idle elimination (`Vehicle.OnIdleness`) and the speed and drift achievements (`Repositories.Achievement`).

Step 1 is most of the cost, and it only reads track data and writes that agent's stats. ? `Curve2D.SampleBaked` is a call into the engine; Godot allows calls on a `Resource` from several threads as long as nothing modifies it, but this was not checked against the Godot source.

### Suggestion

Split it into two passes:

```csharp
// Pass 1: parallel, no side effects.
Parallel.ForEach(Partitioner.Create(0, agents.Length), r =>
{
    for (int i = r.Item1; i < r.Item2; i++) agents[i].Stats.UpdateStats(agents[i].Vehicle);
});

// Pass 2: main thread, in agent order, as today.
foreach (var a in agents) a.ApplyStatsSideEffects();   // icons, idle elimination, achievements
```

- Exact, as long as pass 2 keeps the current order. Idle elimination and achievements only depend on that agent's own stats.
- For `Curve2D.SampleBaked`, use the baked point array that `TrackManager` already keeps (`_bakedPoints`, `_bakedOffsets`), with a C# port of the sampling, if calling the engine from several threads causes problems. ✓ The Rust port reproduces it in `curve.rs`.
- The per-agent work also gets cheaper with the ring buffer (finding 5) and with reusing the nearest-wall query (finding 12).

✓ In the Rust port, `TrainingAgent::update_stats` runs inside each car's parallel window.

## Fusing a car's tick work

In the game, one car's tick is split across two parallel loops: sensors and inference in one, forces in the other. Each loop has its own barrier, and each pass loads the car's data into the cache again.

Inference only runs for the cars in the current batch, but the force phase runs for every car. A car's force phase only needs its own outputs from the latest inference. So, for each car, the order "inference (if this is its batch), then forces" can run in one task:

```csharp
Parallel.ForEach(ranges, options, r =>
{
    for (int i = r.Item1; i < r.Item2; i++)
    {
        var agent = agents[i];
        if (agent.IsActive && schedule.Infers(i, tick)) { agent.UpdateInput(); agent.UpdateActive(); agent.UpdateOutput(); }
        agent.Vehicle.PhysicsProcess(delta);
    }
});
```

Conditions for this to be exact:

- ✓ The sensors only read the car's own transform and static track data, unless `CanSensorsSeeCars` is on. With car sensing on, sensors read `ObstacleSnapshot`, which is rebuilt in `UpdateVehicleInteractions` before the inference step and does not change during it. So this also holds.
- The vehicles list (`_vehicles`) and the agents in the batch pool must be mapped to each other. Vehicles that do not belong to an active agent (for example, cars left from a previous generation) still need `PhysicsProcess`.
- `UpdateActive` calls `Vehicle.SetIsActive(false)` on elimination from a worker thread. That already happens today inside `ApplyOnBatchParallel`, so this is not a new risk.

This removes one barrier per tick. The bigger step is to run a whole window of ticks per car, which only works outside the engine. See [05](05-headless-simulation.md).

## Engine calls

### Game

Per active car per tick (✓):

- `Vehicle.PhysicsProcess` calls `Output.TryGetValueNorm` 5 times (a dictionary lookup each time, finding 1).
- `VehiclePhysics.ProcessWheels` takes the wheels as `IEnumerable<Wheel>`. For each wheel it calls `TrackManager.GetSurfaceAt` (finding 11) and `TrackSurfaceConstants.GetSurfaceProperties`.
- `this.GetVehicleSurfaceProperties()` builds a new 12-field struct every tick (finding 2).
- `ImpulseAccumulator.Flush` calls `PhysicsServer2D.BodyApplyCentralImpulse` and `PhysicsServer2D.BodyApplyTorqueImpulse`.
- `VehiclePhysics.ClampVelocity` calls `PhysicsServer2D.BodySetState` when the car exceeds `MaxVelocity`.
- In `_IntegrateForces`, `state.GetContactColliderObject(i) is Vehicle` creates or looks up a managed wrapper for the collider just to test its type.

`LinearVelocity` and `Transform` are read from cached state (`_state.LinearVelocity`), not from the engine, so those reads are already cheap (✓).

### Suggestion

- **Contacts.** `VehicleManager` already keeps a `HashSet<Rid> _vehicleRids` (✓). Test `state.GetContactColliderRid(i)` against it, or keep a similar set of instance IDs for `GetContactColliderId(i)`. That avoids creating or looking up the managed wrapper.
- **Impulses and the velocity clamp.** The impulses are summed in `ImpulseAccumulator`, so they could be applied in `_IntegrateForces` through `state.ApplyCentralImpulse` and `state.ApplyTorqueImpulse`, and the clamp could set `state.LinearVelocity` there. That saves two or three server calls per car per tick. To stay exact, keep today's behaviour, which the Rust port reproduces (`car.rs`, ✓):
  - the impulse is added to the velocity first, as `velocity + impulse * (1 / mass)`;
  - the clamp checks the cached velocity from *before* this tick's impulses;
  - when the clamp fires, `BodySetState` overwrites the velocity, so that tick's central impulse is dropped (the torque impulse is kept).

  ? Whether Godot applies `state.ApplyCentralImpulse` at the same point of the step as `BodyApplyCentralImpulse` was not checked. Compare against a recorded replay before shipping.
- **Wheels.** Take `Wheel[]` instead of `IEnumerable<Wheel>`. ? A `foreach` over the interface allocates an enumerator if the underlying type is a `List`.

Impact: medium. The two to three server calls per car per tick are likely the largest remaining fixed cost of the force phase. ? With Godot's physics running on a separate thread, the `PhysicsServer2D` calls may also be queued and synchronized; that was not checked.

## Where the Rust port goes further

| | Game | Rust (independent mode) |
|---|---|---|
| Unit of parallel work | one car, one phase of one tick | one car, a whole window of ticks |
| Barriers | at least 2 per tick | 1 per window |
| Statistics | main thread, sequential | inside the car's task |
| Physics state | engine bodies | plain fields on the car |

`TrainingRunner::advance_window` (`training.rs`, ✓) runs each car through the window: stats on stats ticks, the stop checks, then `drive_agent` (sensors, network, physics step). A car that stops early (time limit or inactive) waits; a second parallel pass then brings those cars up to the tick at which the last car stopped, so the result equals the tick-major loop. This needs the physics outside the engine, which is the subject of [05](05-headless-simulation.md).
