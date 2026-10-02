# 05: A headless training simulation

Findings 24 and 25 in the [overview](README.md). Complexity: very high. This is an architectural change. Consider it only if the smaller findings are not enough.

## Why the scene tree limits training speed

During training, each car is a `RigidBody2D` node with a custom integrator. Every physics tick (✓ decompiled):

1. The game runs its parallel inference and force phases ([03](03-per-tick-pipeline.md)), and each phase ends at a barrier.
2. Forces reach the engine through `PhysicsServer2D` calls, two or three per car.
3. GodotPhysics2D steps every body: broadphase, contact generation against the walls, the solver, then `_IntegrateForces` on each car.
4. The main thread runs statistics, rewards, sound, camera and other observers.

Every car waits for the slowest car at each barrier, 60 times per simulated second. Every car's data is loaded into the cache several times per tick, once per phase. The engine step is a separate phase that ? the game cannot parallelize itself (GodotPhysics2D's threading was not checked here).

But during training, a car only interacts with static walls, unless car collisions or car sensing are on. Its future depends only on its own state, its network and the track. Nothing forces one car to wait for another before the generation ends.

## What the Rust port does

The Rust port is a full reimplementation of the parts of Godot the cars use, in float32, in Godot's operation order:

| Part | Rust module | ✓ |
|---|---|---|
| Car physics from `Vehicle`, `Wheel`, `VehiclePhysics` | `car.rs` | ✓ |
| Godot's SAT contact generation and sequential-impulse solver against static convex tile shapes | `collision.rs` | ✓ |
| Godot's BVH broadphase, when the contact order must match a shared scene | `broadphase.rs` | ✓ |
| `Curve2D` bake, `GetClosestOffset`, `SampleBaked` | `curve.rs` | ✓ |
| `BspTreeRaycaster`, plus the faster `RayTree` | `bsp.rs` | ✓ |
| .NET `Math` functions where they differ from Rust's (`tanh`, `pow`, `atan2`, sine tables) | `double_math.rs`, `managed_trig.rs` | ✓ |
| Training loop, statistics, rewards, reproduction | `training.rs`, `evolution.rs` | ✓ |

The fidelity experiment (`../experiments/driving_sim_rs_fidelity/`) replays traces recorded from the game. ✓ On three episodes (A01, A07, B06) every captured position, velocity, rotation and network output matches exactly. Two older captures lack contact data, so their contacts could not be checked.

With the physics in plain fields, `TrainingRunner::advance_window` (`training.rs`) runs in **car-major** order: one worker takes one car and runs it through a whole window of ticks (statistics, stop checks, sensors, network, physics), then takes the next car. Cars that stop early are brought up to the common stop tick afterwards, so the result equals running all cars tick by tick. The trainer only synchronizes once per window.

Measured results (from this repository, not the game):

| Case | Throughput |
|---|---|
| 1 thread, 1,000 cars | about 306,000 car ticks per second |
| 32 threads, 1,000 cars, 30 s window | about 5.77 M car ticks per second, 96× real time |
| 2 threads, 128 cars, 1,661-parameter network | about 1.05 M car ticks per second |

## How the game could do this

### Scope

Keep the scene for everything the player sees. Add a separate C# simulation for training at high speed:

```
TrainingSim (plain C#, no Godot nodes)
 ├─ TrackData       walls (flat BSP), baked path, surface grid      [04]
 ├─ CarState[]      position, angle, velocities, wheel state         plain structs
 ├─ FlatNetwork[]   parameters + scratch                             [01]
 └─ Step(car, tick) sensors → forward → wheels → integrate → contacts
```

### Steps

1. **Port the car physics.** `VehiclePhysics`, `Wheel` and `Vehicle._IntegrateForces` are already game code, so this mostly means replacing `PhysicsServer2D` calls with fields.
2. **Port the contacts.** Training cars only touch static, convex tile collision polygons. The game needs the rectangle-against-convex-polygon case of GodotPhysics2D's SAT and its sequential-impulse solver. This is the hardest part if results must stay exact. `collision.rs` shows which operations and which order are needed, including Godot's contact caching and solver iterations.
3. **Use the data structures from [04](04-spatial-queries.md)** for sensors and surfaces.
4. **Run car-major windows.** `Parallel.For` over cars, each task looping over the window's ticks:

   ```csharp
   Parallel.ForEach(Partitioner.Create(0, cars.Length, 1), r =>
   {
       for (int c = r.Item1; c < r.Item2; c++)
           for (long t = window.From; t <= window.To; t++)
           {
               if (t % 6 == 0) { stats[c].Update(...); if (ShouldStop(c, t)) { stopTick[c] = t; break; } }
               if (schedule.Infers(c, t)) Infer(c);
               Step(c);
           }
   });
   // Then: cars that stopped early catch up to the latest stop tick, as in training.rs.
   ```

   The window length sets how often the main thread gets control back, for example to update the UI or to check for a pause.

5. **Display.** Copy the positions of the visible cars (or the leader) into their nodes once per frame. Cosmetic work (skidmarks, sound, icons) reads from those copies.
6. **Fall back to the scene** when cars must interact. With car collisions or car sensing on, every car depends on every other car at each tick, so the windows must be one tick long (tick-major). The headless simulation still avoids the engine calls in that mode.

### Exactness

The Rust port shows that an exact port is possible, but it took a dedicated effort, including traces from the game and the .NET math reimplementations. In C#, the .NET math functions come for free, but Godot's float32 SAT and solver still have to be ported operation by operation, and checked against recorded runs.

If exact replays between the headless simulation and the scene are not required, a simpler contact model would work for training. But networks trained in one would then drive slightly differently in the other.

## GPU backend

The Rust port also has an exact GPU simulator (`gpu_sim.rs`, `gpu/sim/`, HIP on AMD). On an RX 7900 XTX (`reports/gpu-performance-20260929/README.md`):

| Cars | Seconds per 5,400-tick generation |
|---:|---:|
| 8,192 | 2.09 |
| 32,768 | 5.72 |
| 262,144 | 40.3 |

That is about 21 M car ticks per second at 8,192 cars.

For the game, this is not a practical next step:

- It requires the headless simulation (24) first.
- It only pays off with thousands of cars. ? Typical game populations were not checked, but they are likely far smaller.
- Exactness on the GPU needs control over floating-point contraction and math functions in the shader compiler. A Godot compute shader or a cross-vendor API gives less control than HIP.

It is listed for completeness.
