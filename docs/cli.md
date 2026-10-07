# CLI reference

Run commands from the repository root or an unpacked release archive; both contain `assets/`. `altd-sim --version` identifies the build, including nightly packages. `--threads N` sets the Rayon pool; the default respects CPU affinity. `--mode independent|lockstep` selects native scheduling. Both are global flags. Each command's `--help` is the complete argument reference.

`--math-profile proton|win10-fma3|win11-fma3` is also global and selects the game's C runtime arithmetic independently of the CPU or GPU backend. Every build includes all three profiles. Omitted profiles use `proton` on Linux/macOS, `win10-fma3` on Windows before build 26100, and `win11-fma3` on Windows build 26100 and newer. For example, `altd-sim --math-profile win11-fma3 train-scratch ...` trains for current Windows on any supported host. [Math profiles](../math/README.md) describes the version boundaries and implementation.

## Commands and inputs

| Command | Required inputs and purpose |
| --- | --- |
| `bench` | `--scene`, `--network` with weights, `--spawn-trace`; fixed-population timing and optional state dumps |
| `train` | `--scene`, weighted `--network`, `--model`, `--output`; evolution of an existing network |
| `train-scratch` | `--scene`; Xavier initialization, checkpoints, generated batches and candidate export |
| `evaluate` | `--network`, `--model`, `--suite`, `--report`; evaluate a frozen candidate without evolution |
| `serve` | No input files; expose CPU or HIP sessions over a loopback WebSocket |
| `compare-trace` | Positional scene, trace and report; replay controls from a recorded trace |
| `compare-one-step` | Positional scene, trace and report; restore each recorded frame before stepping |
| `compare-closed-loop` | Positional scene, trace, network, model and report; network-driven trajectory comparison |
| `compare-network` | Positional network, trace and report; compare inference with recorded controls |
| `compare-sensors` | Positional scene, trace, model, trajectory and report; compare sensor observations |
| `compare-score` | Positional scene, trace and report; compare lap/score statistics |
| `gpu-info` | Optional `--library`; load the GPU library used by `--gpu`, check its struct layouts and exports and, when a GPU is present, run one kernel on it |

Scene and capture paths have no campaign defaults. `bench` and `train-scratch` default their sensor model to `assets/models/rally.json`. Scratch training defaults its sensor/control names to `assets/networks/rally.json`; these templates have no trained weights. The defaults are compiled into the binary, and `run.json` records them as `builtin:<path>`. Formula runs should explicitly use the Formula network/model assets and scene template.

A fixed track requires `--spawn-trace` for `train` and `train-scratch`. The trace's first frame supplies the initial pose. Random mode needs only a geometry-free scene template; generated geometry supplies its reset pose. `--spawn-trace` is ignored in random mode. `bench` always requires its explicit spawn trace and supports `--spawn-index`.

```sh
target/release/altd-sim --threads 8 bench \
  --scene wasm/test/scenarios/rally_mixed.scene.json \
  --spawn-trace wasm/test/scenarios/rally_mixed.spawn.json \
  --network wasm/test/scenarios/benchmark.network.json \
  --population 1024 --ticks 600

cargo run --release --offline --example extract_saved_track -- \
  tests/fixtures/generated_saved_track.json example \
  assets/scenes/rally_template.json target/imported-scene.json
```

The importer takes a saved-track export, a track name, a scene template and an output scene. Its bundled input is generated regression data. `tools/benchmark.py` runs the public generated-track bench repeatedly and keeps every report.

## Local server

```sh
target/release/altd-sim --threads 8 serve --port 47800
target/release/altd-sim serve --allow-origin http://127.0.0.1:5173
```

`serve` binds `127.0.0.1` only and defaults to port 47800. Repeat `--allow-origin` to replace the default `https://drivinglab.jectrum.de` allowlist. The shared Rayon pool uses the global `--threads` flag. The server prints its address, allowed origins, thread count and HIP availability. It is included by the default `server` feature. See [the server protocol](server.md) for handshake checks, session operations, binary frames and resume behavior. Drive Lab client integration is a separate change.

## Scratch training

Defaults are 8,192 cars, 50,000 target generations, a 5,400-tick time limit, seed 1, two inference batches and network shape `20,16,16,16,16,12,12,8,5`. The output directory defaults to `runs/example`. Pass short explicit limits for a smoke run. `--init-network` uses supplied weights; `--init-population` imports a compatible checkpoint into a new run directory.

Evolution starts with tournament selection of 20 parents, no crossover, four preserved parents, adaptive mutation and no weight decay. Mutation decays geometrically from 0.4 to 0.0125; `--schedule linear` changes that schedule. `--settings FILE` overrides evolution defaults except requested population and scheduled mutation. It accepts a plain object or a nested `settings` object. Explicit elimination switches and `--reward` take precedence over the file.

Elimination defaults off. `--eliminate-on-wall` deactivates cars after native wall contact. `--idle-eliminate` deactivates cars after more than 80 consecutive statistics updates with insufficient forward progress. The corresponding `--no-*` switches disable them. Passive physics and pending contacts continue for inactive cars.

`--reward distance` selects on `total_score`; `--reward best-lap-time` selects on reciprocal best completed lap time. Without an explicit reward, settings-file rewards remain, with distance as the fallback. Cars without a completed lap have no best-lap signal.

## Generated track batches

```sh
target/release/altd-sim --threads 8 train-scratch \
  --scene assets/scenes/rally_template.json --track-mode random \
  --random-track-settings assets/random_track_settings.json \
  --tracks-per-generation 3 --track-buffer-size 8 \
  --population 256 --generations 3 --ticks 600 --seed 1729 \
  --save-final-candidate --out-dir runs/example
```

Each unchanged population evaluates the same track batch, with fresh spawn and a separate tick limit on each track. Selection uses the mean of normalized per-track reward ranks. Reproduction and the mutation schedule advance once after the complete batch. Fixed mode allows one track only. Queue depth changes preparation capacity, not the RNG stream.

Without a settings file, track lengths range from 12 to 40 even tiles, all block types are enabled, and one, two or three distinct surfaces are sampled with equal probability into consecutive sections. `assets/random_track_settings.json` is an explicit variant that also samples double-block choices and random/section surface distribution.

| Track field | Values |
| --- | --- |
| `length` | Fixed even count, or inclusive `{"min": 12, "max": 40}` range; supported grid capacity 4 to 76 |
| `allow_double` | Boolean or list of boolean choices |
| `surfaces` | Fixed distinct IDs, list of sets, or `{"pool": [0,1,2], "count": [1,2,3]}` |
| `distribution` | `0` random per tile, `1` consecutive sections, or choices |
| `start` | `null` for a fresh position, or fixed grid `[x,y]` |

Surface IDs are asphalt 0, dirt 1 and ice 2. Failed generation retries are bounded; invalid or impossible settings fail rather than silently changing a fixed requested length. Track seeds depend on run seed, generation and batch slot independently of reproduction RNG, worker count and backend.

## Outputs, resume and stopping

`run.json` stores effective options from the latest invocation. `log.jsonl` has one completed-generation row, with per-track statistics for random batches. `progress.json` publishes current batch progress atomically. Generated track exports live under `tracks/`. `best.json` and `best_laps/` retain the fastest individual lap, even when selection uses another reward.

Checkpoints are saved every `--checkpoint-every` generations, default 100, and after normal termination. Zero disables them. `checkpoint.json` identifies a binary population file and stores shape, population, generation and RNG state. `--resume` restores the latest complete generation boundary and removes replayed log/geometry entries. It cannot preserve every mid-tick native contact cache.

```sh
target/release/altd-sim train-scratch --resume \
  --scene assets/scenes/rally_template.json --track-mode random \
  --random-track-settings assets/random_track_settings.json \
  --population 256 --generations 6 --ticks 600 --out-dir runs/example
```

`--generations` is the total target, including saved generations. Supply desired options again: omitted options use CLI defaults, except track seed/count retained in checkpoint metadata. An explicit `--tracks-per-generation` replaces the saved count. Preserve scene and track settings to reproduce the original sequence. Saved networks/RNG take precedence over a new `--seed` or `--init-network` during resume.

`train-scratch --resume` also retains the checkpoint's math profile; an explicit different `--math-profile` is rejected. Checkpoints predating profiles use `proton`. New `run.json` and checkpoint metadata record `math_profile` so moving a run to another host does not change its arithmetic.

Shape must match every checkpoint layer; matching parameter counts alone is insufficient. Population reductions keep the first networks, while expansions repeat them in order. Mutation uses the newly supplied schedule at the restored generation. Earlier checkpoints without shape/population recover those fields from adjacent `run.json` before updating current options.

`--save-final-candidate` writes the complete-batch fitness leader before breeding, with lowest-index tie breaking. It preserves exact weights, ordered names, generation and reward metadata without consuming RNG. A completed checkpoint confirms its filename/hash/generation. A candidate without that confirmation is not a completed-stage result.

`--stop-score-above`, `--stop-lap-below`, `--stop-lapped-percent` and `--stop-plateau` stop after a qualifying complete batch. Lap targets require a car to lap every track. Plateau counters restart per invocation, using `--plateau-metric lap|score`. A `stop_request` file in the run directory stops after the current batch; a stale file is removed at startup. SIGTERM does not write a final checkpoint. Negative thresholds use `=`, for example `--stop-score-above=-100`.

## Frozen evaluation

`evaluate` reads a version-1 suite containing `vehicle`, `options` and distinct `tracks`. Every track specifies relative scene/spawn filenames, their SHA-256 hashes and a finite tick limit. Options specify CPU backend, inference batch count and elimination. The evaluator validates finite network parameters, shape, ordered sensor/control names, vehicle compatibility and input hashes.

A suite may set `options.math_profile` to a supported profile. An explicit CLI profile must agree with it. Without either, evaluation detects the host's profile; the report records the effective `math_profile`.

Each track gets fresh state, contacts, controls, sensors and statistics. The evaluator never breeds or modifies training checkpoints/logs. Its atomic report records candidate/suite/simulator hashes, per-track score, progress, lap completion, optional best lap, collisions and executed ticks. An interrupted run does not publish a successful report. Existing tests build fully contained suites from fixed generated tracks.
