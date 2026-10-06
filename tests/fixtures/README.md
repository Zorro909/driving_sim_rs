# Test fixtures

Campaign tracks are not distributed.

The `native_*` and `windows_*` fixtures capture vehicle, sensor, collision,
curve, random-number and scheduler behavior from the game runtime. Scenes in
these fixtures contain synthetic collision cases or generated tracks. The
runtime captures use Godot 4.3; packed-game captures identify build 77dcf97d8
and Windows .NET 8.0.2 where recorded. Capture harnesses include OracleNode,
TileMapOracle, GeneratedTrackOracle, BroadPhaseOracle and PopulationOracle.
The capture harnesses and their raw logs are not distributed. Fixture provenance
describes the setup, without linking to unavailable files.

The `recorded_rally_*` files retain short vehicle-state captures for diagnostics,
free-motion transitions and velocity-sensor tests. Their original collider
exports have been removed. Recorded positions, contact points and scalar sensor
outputs remain; these files contain no track tiles, wall polygons or paths.
The tests use the public vehicle template and generated geometry. These limited
states do not establish parity for a complete campaign track.

`native_generated_*` captures check the game on generated geometry. The
random-track generator inputs have fixed seeds, and captured outputs retain
their game-oracle status. `windows_tracks*` and `native_generated_curves.json`
check generated track layouts, RNG state and curve baking. The tile resources
used by the generator remain in `src/data/track_resources.json`.

The generated TileMap captures are named `native_tilemap_generated_*`.
These captures and the generated terrain/sensor replacements use seed 1729,
generation zero and `assets/random_track_settings.json` with the public Rally
template. Their `recipe` object stores `seed`, `generation` and `slot`.
`tests/support/oracle.rs` rebuilds the scene with `training_scene_at`, recursively
applies `scene_overrides`, and restores `physics_shape_order` where native wall
insertion order matters. Redraw events carry their own recipes. Vehicle changes,
terrain-only scenes and reset poses are explicit overrides; captured output
values remain in the fixture.

| Generated game capture | Track slot |
| --- | ---: |
| `native_generated_dirt180.json` | 2 |
| `native_generated_ice180.json` | 0 |
| `native_generated_curve_mixed.json`, `native_generated_ordered_sensors32.json`, `native_generated_path_sensors32.json` | 3 |
| `native_generated_stats_drive.jsonl`, `native_generated_stats_laps.jsonl` | 30 |
| `native_tilemap_generated_slot05*`, `native_tilemap_generated_single600.json` | 5 |
| `native_tilemap_generated_slot06*`, `native_tilemap_generated_truck600.json` | 6 |
| `native_tilemap_generated_snowmobile600.json` | 3 |
| `native_tilemap_generated_coalesced600.json` | 0, then 6 |
| `native_tilemap_generated_redraw600.json`, `native_tilemap_generated_redraw_resize600.json` | 2, then 5, 11, 2 |

The redraw events occur at ticks 150, 300 and 450. Captures use the packed game
DLL and vehicles in Godot 4.3 build 77dcf97d8 with Windows .NET 8.0.2 under
Proton Hotfix. The parked TileMap fixtures retain native wall-query ordering
frames. Their unit test rebuilds the generated polygons from the recipe.

The two generated statistics files start with a `FIDELITY_CAPTURE` JSON header
that records provenance and the recipe. Their `FIDELITY_STATS` rows contain game
AgentStats and lap-observer outputs. The drive capture has 234 samples; the lap
capture has 126 samples, two laps, drift, collisions and inactive states.
These statistics captures are game oracles with synthetic vehicle-state inputs.

`native_vision_lengths.json` is a table of sensor-length function outputs, with
no track geometry. The runtime math tables and scheduler fixtures also contain
no campaign track data.

`generated/` contains current-code regression data captured before the repository
restructure. It is not a game-engine oracle. The crate's random-track generator
uses seed 1729, generation zero, and track slots zero and one with
`assets/random_track_settings.json` and the geometry-free Rally template.
Tests build the scenes from that recipe, avoiding duplicate scene geometry.

The generated data checks three wall encounters, trace restoration with and
without body basis, sensors, contact impulse caches, complete score and lap
statistics, native population resizing and redraw, and two Session generations
in both execution modes. Four binary files check exact generation-boundary
checkpoint bytes. Floating-point state is stored as hexadecimal f64 bits, with
f32 state widened to f64 first. These eight golden outputs total about 2.9 MB.

The Session snapshots and checkpoint bytes were captured on Linux x86_64 with
GNU libm. Their exact golden comparisons run on that platform. On every
platform, the Session test also compares both execution modes with the legacy
`TrainingRunner` on the same host, including every parameter and RNG bit, and
checks that restoring a checkpoint reproduces the next generation's state and
checkpoint bytes exactly. Gaussian initialization uses host float64 math, so
equal RNG draws need not yield equal parameters across native hosts.

`generated_saved_track.json` is a compact saved-track input made from the same
seed and track slot one. The saved-track import test regenerates its encoding and
checks scene conversion and curve baking. It is current-code generated input.

Fixture JSON is parsed with the crate's round-trip float parser. Keep numeric
values and array order intact when editing provenance text or adding coverage.

`ucrt_vectors.json` holds inputs and results recorded from the live Windows
`ucrtbase.dll` (sinf, cosf, atan2f, exp, tanh, pow, as hex bit patterns), including
special values and NaN payloads. `tests/ucrt_math.rs` replays them against
`src/math/ucrt_math.rs`; `cargo run --release --example ucrt_probe -- --export
tests/fixtures/ucrt_vectors.json` rewrites them on Windows. They are not captures
of the game: the `windows_*` captures above come from the game under Proton and
match the default `proton` math, not the Windows UCRT.
