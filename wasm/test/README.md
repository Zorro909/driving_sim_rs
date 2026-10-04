# Generated browser fixtures

All four scenarios use fixed tracks produced by `training_scene_at` with seed 1729, generation zero and distinct slots. Each scenario's `generated` object records its geometry-free template and exact settings. The default Formula track has 12 tiles and asphalt/dirt/ice; Rally variants have 16 asphalt tiles, 20 mixed-surface tiles and 24 ice tiles.

Scenes, spawn poses and `*-reference.json` state snapshots were generated from the simulator before numerical refactors. They are current-code regressions, not game-oracle captures. Population sizes, inference batches, scheduler modes, network shapes and elimination combinations retain the browser coverage across Formula and Rally.

Regenerate from the repository root:

```sh
cargo run --release --offline --example wasm_reference -- --generate-fixtures
```

`benchmark.network.json` is Xavier initialization of shape `20,16,16,16,16,12,12,8,5` with PyRandom seed 1 and Rally input/output names. `tools/benchmark.py` uses it with the mixed Rally scene and its spawn file. Inputs are packaged only in the repository test tree; Cargo packaging excludes this directory.

The native reference example accepts a scenario and output path to capture new observations into `target/` for comparison before updating tracked references. Expanded threaded observations go to `target/wasm-thread-reference.json`. Division fixtures are generated separately and are not checked in.
