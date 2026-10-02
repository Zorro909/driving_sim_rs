# 04: Spatial queries

Findings 11, 12, 16, 21 and 22 in the [overview](README.md). These are the track lookups done for every car on every tick: ray sensors, the nearest wall, the projection onto the track path, and surface lookups.

## Queries per car

| Query | Game implementation | Called from | How often |
|---|---|---|---|
| Ray vs. walls | `BspTreeRaycaster.Raycast` | `RayCastSensor` | each ray, each inference |
| Ray vs. cars | loop over `VehicleManager.ObstacleSnapshot` | `RayCastSensor`, if `CanSensorsSeeCars` | each ray, each inference |
| Nearest wall point | `BspTreeRaycaster.FindClosestPoint` | `DistanceFromWallSensor`, `AgentStats` | each inference; each stats tick |
| Path projection (full) | `Curve2D.GetClosestOffset` (engine, all baked points) | `CorrectDirectionSensor`, `TrackCurvatureSensor` | once per sensor, each inference |
| Path projection (windowed) | `TrackManager.GetClosestOffsetNear`, ±450 px | `AgentStats` | each stats tick |
| Surface under a wheel | `TrackManager.GetSurfaceAt` → `ConcurrentDictionary` | `VehiclePhysics.ProcessWheels` | 4 per car per tick |

All ✓ from the decompiled code.

## BSP raycaster

### Game

`BspTreeRaycaster` (✓):

- Built from the wall segments of the track's tile collision polygons. `FindBestPartition` tries every segment as the splitter and scores it as `|front - back| + 10 * splits`, which is O(m²) per node.
- Each `BspNode` is a `record` object with `Partition`, `Segments` (coplanar walls), `Front` and `Back`.
- `Raycast(node, ray)`:
  1. Classify the ray's start and end against the partition line (epsilon 0.0001).
  2. Recurse into the near child first. Return its hit if there is one.
  3. Otherwise, intersect the partition and the coplanar segments and keep the closest hit.
  4. If there is none, and the two endpoints are on different sides, recurse into the far child.

The traversal has no bounding volumes. A ray whose endpoints are both in front of a partition still descends the whole front subtree, even when every wall in it is far from the ray. Every node visit is a pointer dereference to a separate heap object, and `node == null` calls the record's equality operator (finding 9).

### Rust port

`RayTree` (`bsp.rs`, ✓) is built from the same tree, so it visits nodes in the same order and returns the same hit:

- **One array of nodes, in preorder.** Each node stores the index range of its walls and the indexes of its children (`u32`, with `u32::MAX` for none).
- **One array of walls** as float32 `(start, end - start)`: the node's partition first, then its coplanar segments.
- **Two boxes per node:** `own` covers the node's walls, `subtree` covers its walls and all of its descendants. Each box is a float32 center and half-size, rounded outwards so it always covers the walls.
- **The pruning test**, `Bounds::beside(ray)`:

  ```rust
  let cross = b.x * (cy - start.y) - b.y * (cx - start.x);      // b = end - start
  cross.abs() > b.x.abs() * (hy + margin) + b.y.abs() * (hx + margin)
  ```

  This is true when the whole box lies on one side of the ray's infinite line, at least `margin` away from it. A wall in such a box cannot be hit: both of its endpoints are on the same side of the ray's line, so the intersection test fails anyway.
  - At the start of each node: if `subtree` is beside the ray, return "no hit" for the whole subtree.
  - Before testing the node's own walls: if `own` is beside the ray, skip them.
- **Exactness.** The comment on `RayTree::raycast` gives the float32 error bound. With `margin = 1e-3 + 2e-5 * (C + S)`, where `C` bounds the wall coordinates and `S` the ray start, the box test only skips walls whose intersection test would certainly fail. This includes the rounding in the game's `TryGetIntersection`. Rays shorter than about 1e-10 never skip.

✓ `tests/exact_speedups.rs` compares `RayTree` against the plain BSP port on more than 2 million rays (more than 500,000 hits). It covers the game's exported tracks, generated tracks, poses from a recorded game trace, and tracks shifted far from the origin, where float32 spacing is coarse.

### Suggested C# implementation

```csharp
struct RayNode { public int First, Last, Front, Back; public Box Own, Subtree; }
struct Box { public float Cx, Cy, Hx, Hy; }

sealed class FlatBsp
{
    RayNode[] _nodes;
    Vector2[] _wallStart, _wallDelta;   // or one float[] with 4 values per wall

    public Vector2? Raycast(int i, in Ray r)
    {
        ref readonly RayNode n = ref _nodes[i];
        if (Beside(n.Subtree, r)) return null;
        // classify start and end against wall n.First (the partition), same epsilon as today
        // near child, own walls (skip if Beside(n.Own, r)), far child: same order as BspTreeRaycaster
    }
}
```

- Build it from the existing `BspNode` tree after construction, in preorder.
- Keep `TryGetIntersection` and `ClassifyPoint` exactly as they are, in float32.
- Compute the margin per ray as Rust does. Use the box bounds derivation from `bsp.rs` (rounded outwards) and do not compute the box in double precision and then cast.
- Write a test that casts millions of random rays (including rays along walls and rays that end exactly on walls) through both raycasters and requires identical results.

`FindClosestPoint` can also use the flat array. Its existing pruning (the distance to the partition's infinite line) stays the same. A box distance test could be added for the far subtree, but it has to be checked for exactness separately, since it changes which walls are visited. Visiting fewer walls is only safe if a skipped wall can never be strictly closer.

Impact: high. ✓ The game's default sensor set (`NeuralNetworkUtils.GetDefaultSensors`) is 5 vision rays of 300 to 800 px, plus grip and speed. Every car casts all of them on every inference. ? How much the box test saves depends on the track. On open layouts with long walls, a ray typically touches only a few subtrees.

## Car-to-car queries

When `CanSensorsSeeCars` is on, `RayCastSensor` loops over every entry of `ObstacleSnapshot` for every ray (✓). That is O(cars × rays) per car per inference, so O(N²) in total. `VehicleObstacle.TryGetRayIntersection` first tests a bounding circle against the ray (`GetClosestPointOnSegment`), then tests the 4 edges.

`VehicleManager.UpdateVehicleInteractions` (every tick, when collisions are on):

- `UpdateClearedOverlaps` compares each car that has not cleared its spawn overlap against all other cars. It reads `vehicle.Position` (an engine call) in the inner loop.
- It rebuilds `ObstacleSnapshot` with LINQ `Where`, `Select` and `ToArray`.

### Suggestion

1. Read all car positions into an array once per tick.
2. Build a uniform grid of the obstacles' bounding circles (cell size about the ray length or the car size), and reuse its arrays from tick to tick.
3. For each ray, walk the cells along the ray (a DDA), or visit the cells that overlap the ray's bounding box, and test only the obstacles in them.
4. For overlaps, test only the obstacles in the same and neighbouring cells.

The ray result is the minimum squared distance over all obstacles, so the order in which they are tested does not matter: the result is exact. Mark obstacles as visited per ray, because a car can sit in several cells.

? The Rust port does not simulate car-to-car collisions or car sensing (no such code in `src/`), so there is no reference implementation for this one.

## Path projection

### Game

`CorrectDirectionSensor` and `TrackCurvatureSensor` (✓) each compute:

```csharp
Vector2 p = Track.GetClosestTrackPointInsideTile(pos).Lerp(pos, 0.5f);
float offset = curve2D.GetClosestOffset(p);
```

`GetClosestOffset` is Godot's own implementation: it checks every baked segment. Tiles are 768 px wide and the path is baked every 5 px (✓ as exported to the Rust port, `random_track.rs`), so a track of 20 tiles has on the order of 3,000 baked points. Each call is that many point-segment distance tests. Both sensors repeat it at the same point for the same car on the same tick. `GetClosestTrackPointInsideTile` uses `GetTileByPos` and LINQ `First()` / `Last()` on the tile's connections.

`AgentStats` already uses a windowed version, `TrackManager.GetClosestOffsetNear`: a binary search (`FindBakedIndex`) for the previous offset, then a linear scan of ±450 px around it.

### Rust port

- `SegmentGrid` (`segment_grid.rs`, ✓) is a grid of 64 px cells. Each cell lists the baked segments that pass through it, in ascending index order (CSR format). A query searches rings of cells outward until no unvisited segment can be closer than the best found so far. It then reruns the original loop body on that subset in ascending order, so ties and NaN handling stay the same.
- `Curve::closest_offset` (`curve.rs`, ✓) always visits segment 0 first, as Godot does, then the grid's candidates.
- `SensorScratch::path_cache` (`car.rs`, ✓) stores the sensor point and its offset for the current pose. The direction and curvature sensors share it.

### Suggestion

1. **Compute it once per car per inference** (finding 12, low effort). The two sensors use the same point. Store `(position, offset)` on the vehicle, and reuse the offset when the position is the same.
2. **Narrow the scan** (medium effort). Either:
   - **Window, like `GetClosestOffsetNear`.** Simple, but not exact. Near crossings, or after a large jump, the window can miss the global nearest segment. The sensors use a point *between* the car and the track center, so the window should be centred on the car's last sensor offset, not on its score offset.
   - **Grid, like `SegmentGrid`.** Exact. It needs a C# port of Godot's `GetClosestOffset` loop (the segment projection and the `(nearest, distance)` update), which is also a chance to drop the engine call. The Rust code shows the arithmetic in float32 order.

Impact: medium to high, depending on track length and how many direction or curvature sensors the network uses. If neither sensor is enabled, the cost is zero.

## Per-tick memoization

Besides the path projection above:

- **Nearest wall.** `DistanceFromWallSensor` calls `Raycaster.FindClosestPoint(GetActualPosition())`. On stats ticks, `AgentStats.UpdateDistanceFromWall` calls it again from the vehicle's position. ? If both positions are the same value at that point of the tick, cache the result per car per tick. Check this first: the sensor runs in the inference phase, the stats run before it in the same tick, and both must see the same position.
- **Car transform.** `RayCastSensor.GetNormalizedValue` reads `vehicle.GlobalPosition` and `vehicle.Transform` for every ray (✓). Read them once per car per inference (finding 4).

## Surface lookup

`TrackManager.GetSurfaceAt` (✓):

```csharp
Vector2I key = TrackUtils.PointToCoords(position);
if (SurfaceCache.TryGetValue(key, out var value)) return value;     // ConcurrentDictionary
value = Track.Tiles.GetValueOrDefault(key)?.Block.Surface ?? TrackSurface.Asphalt;
SurfaceCache[key] = value;
```

This runs 4 times per car per tick. `ConcurrentDictionary.TryGetValue` is lock-free, but it still hashes the key and follows a bucket chain.

Suggestion: in `TrackManager.Init`, build a `TrackSurface[]` covering the track's tile bounds plus a border filled with `Asphalt`:

```csharp
int ix = coords.X - _x0, iy = coords.Y - _y0;
return (uint)ix < (uint)_w && (uint)iy < (uint)_h ? _surfaces[iy * _w + ix] : TrackSurface.Asphalt;
```

Exact: it returns the same value as today. ✓ The Rust port does this (`world.rs`, `TileTable`), and indexes tiles by `floor(x / 768)`. `PointToCoords` must stay unchanged so positions on tile borders map to the same tile.
