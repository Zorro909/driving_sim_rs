# Test fixtures

The `native_*` and `windows_*` fixtures are self-contained captures of vehicle,
sensor, collision, curve, random-number, and scheduler behavior from the game
runtime. Some captures include their own small scene or tilemap. The `a07_*` and
`b06_*` files contain short recorded states; free-motion and velocity-sensor
tests use them with the public vehicle template and generated geometry.
Campaign contact tests that needed separate scene exports were replaced by
generated regressions. Existing capture contents remain available as diagnostic
inputs.

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

`generated_saved_track.json` is a compact saved-track input made from the same
seed and track slot one. The saved-track import test regenerates its encoding and
checks scene conversion and curve baking. It is current-code generated input.

Fixture JSON is parsed with the crate's round-trip float parser. Keep numeric
values and array order intact when editing provenance text or adding coverage.
