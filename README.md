# Rust driving simulation (`altd-sim`)

`driving_sim_rs` is the CPU simulator and trainer for AI Learns To Drive. It incorporates the accuracy and performance work from `experiments/driving_sim_rs_fidelity`: native float physics and collision ordering, runtime math, exact sensor parameters, interpolated lap times, accelerated spatial queries, and AVX2 network inference. It also includes the bit-exact HIP GPU simulator (`--gpu`, see [gpu/README.md](gpu/README.md)).

The original Rust implementation matched the Python simulator. This implementation targets the game, so Python comparison reports and older training results are no longer expected to match. The old throughput measurements in [docs/benchmarks](../docs/benchmarks/README.md) describe that earlier implementation.

## Build and run

```sh
cd driving_sim_rs
cargo build --release --offline

target/release/altd-sim bench --population 1000 --ticks 1800

target/release/altd-sim train-scratch \
  --out-dir ../training_runs/b06_fidelity \
  --eliminate-on-wall --idle-eliminate
```

On an AMD GPU (built by `gpu/build.sh`), `train-scratch --gpu` simulates the generations with identical results. See [gpu/README.md](gpu/README.md).

`--threads N` sets the Rayon worker count; the default uses the CPU affinity mask. `--mode independent|lockstep` selects the scheduler. Exact native trigonometry requires x86 or x86_64. AVX2 acceleration is detected at runtime and has a scalar fallback.

`bench` and `train-scratch` default to the higher-precision scenes in `scenes_exact/` and the sensor model in `rally_trained_model_exact.json`. Explicit `--scene` and `--model` paths still work, including older exports with their precision limitations.

Other commands are `train`, `compare-trace`, `compare-one-step`, `compare-closed-loop`, `compare-network`, `compare-sensors`, and `compare-score`. Run a command with `--help` for its arguments.

## From-scratch training

`train-scratch` defaults to B06 Hard, 8,192 cars, 50,000 generations, and a 5,400-tick time limit. It starts with a random Xavier network of shape `20,16,16,16,16,12,12,8,5`. The network template supplies input/output names; its weights are ignored unless supplied through `--init-network`.

Evolution defaults are tournament selection of 20 parents, no crossover, four preserved parents, adaptive mutation, no weight decay, and reward `total_score` multiplied by 100. Mutation decays geometrically from `--mutation-start 0.4` to `--mutation-end 0.0125`; `--schedule linear` selects a linear decay. `--generations` is the total target generation count, including when resuming. The resumed generation's mutation rate is calculated from the newly supplied schedule. Generations after the schedule's end use `--mutation-end`.

`--reward distance` selects for distance along the track, the existing `total_score` metric. `--reward best-lap-time` selects for faster completed laps using `best_lap_performance`, the reciprocal of the car's best lap time. Explicit `--reward` replaces any rewards in `--settings`; when omitted, settings-file rewards are preserved, with distance as the default.

You can switch rewards when resuming:

```sh
target/release/altd-sim train-scratch \
  --out-dir ../training_runs/b06_fidelity --resume --reward best-lap-time
```

Cars without a completed lap have no best-lap reward signal. Distance training can establish lap completion before switching to best lap time. The printed score remains distance for comparison; selection uses the chosen reward, recorded in `run.json` under `settings.rewards`.

Elimination is off by default:

- `--eliminate-on-wall` deactivates a car when its native callback reports wall contact.
- `--idle-eliminate` deactivates a car after more than 80 consecutive statistics updates with insufficient forward progress. Statistics run at 10 Hz; progress uses a rolling ten-sample mean below 0.1 score units.
- `--no-eliminate-on-wall` and `--no-idle-eliminate` explicitly disable either option.

`--settings file.json` overrides the evolution defaults, except population and scheduled mutation rate. It also accepts `eliminate` and `idle_eliminate` as booleans or numeric switches. Settings may be a plain object or nested under `settings`. Explicit CLI switches take precedence.

Deactivated cars remain in the population. Their network inference and driving stop, and their driving scores and lap statistics stop updating. Passive physics and pending contact accounting continue, preserving native contact ordering without removing bodies or shifting inference batches. A generation finishes on the statistics callback at which every car is inactive, or when its time limit expires. The game checks its time limit before driving, so a generation's callback count can exceed `--ticks` slightly.

Each generation prints lap and score summaries. Lap crossings are interpolated within the statistics interval and rounded to milliseconds, as in the game. Selection uses the configured reward, independently of which car sets the fastest lap.

| File | Contents |
|---|---|
| `run.json` | Effective options from the latest start or resume, including elimination settings |
| `log.jsonl` | Per-generation laps, scores, active-car count, simulated ticks, and timings |
| `best_laps/gNNNNN_T.TTs.json`, `best.json` | Networks that set a new best lap, with input/output names and training metadata |
| `checkpoint.json`, `checkpoint_gNNNNN.bin` | Population parameters, shape, population size, RNG state, generation, and best-lap record |

Checkpoints are written at generation boundaries every `--checkpoint-every` generations, default 100, and after the last generation when training stops on its own. Zero disables checkpointing. Resuming replays work after the latest checkpoint, truncating the corresponding log entries. Resuming a final checkpoint with a higher `--generations` continues exactly as an uninterrupted run would.

### Stop conditions

Training normally ends at `--generations`. These options end it earlier, after the first generation that meets any of them:

| Option | Stops when |
|---|---|
| `--stop-score-above=S` | The generation's best distance score is at least `S` |
| `--stop-lap-below=T` | The generation's fastest lap is at most `T` seconds |
| `--stop-lapped-percent=P` | At least `P`% of cars completed a lap in the generation |
| `--stop-plateau=N` | `--plateau-metric` (`lap`, the default, or `score`) has not improved for `N` generations |

The plateau count starts at zero in each invocation, including a resume. For the lap metric, generations without a completed lap count as no improvement. Creating a file named `stop_request` in `--out-dir` ends training after the current generation; a request left over from before the trainer started is deleted and ignored.

The last generation's checkpoint records why training ended in `stop_reason`, for example `{"condition": "lap_below", "generation": 812, "value": 38.412, "threshold": 38.5}`. The condition is one of `stop_request`, `lap_below`, `score_above`, `lapped_percent`, `plateau` (which adds `metric`), or `generations`, checked in that order. SIGTERM exits without a final checkpoint. Pass negative thresholds with `=`, as in `--stop-score-above=-100`.

## Resume with different settings

```sh
target/release/altd-sim train-scratch \
  --out-dir ../training_runs/b06_fidelity --resume \
  --generations 60000 --ticks 7200 --population 4096 \
  --mutation-start 0.2 --mutation-end 0.01 \
  --eliminate-on-wall --idle-eliminate
```

A resume accepts changes to training parameters, including elimination, population, track, time limit, mutation schedule, and evolution settings. Supply the desired options again; omitted options use CLI defaults rather than inheriting `run.json`. The checkpoint supplies its generation, networks, and RNG state. `--seed` and `--init-network` do not replace saved networks or RNG state during resume.

The requested `--shape` must match the checkpoint's full layer layout, even if a different layout would have the same number of parameters. The binary length must match the saved shape and population. New checkpoints store both fields in `checkpoint.json`. Older checkpoints read them from the adjacent `run.json` once and preserve them in checkpoint metadata before updating the run options.

Reducing population keeps the first saved networks. Increasing population repeats saved networks in order; later reproduction uses the requested population and evolution settings. `--init-population path/to/checkpoint.json` uses the same validation and resizing in a new run directory, starting a fresh best-lap record. An ordinary resume retains the saved best-lap record, even if track settings change.

Existing checkpoints can be continued with the new simulator when their shape matches. Physics changes mean the continuation differs from the old simulator. Native contact caches are also not serialized, so bit-identical physical continuation across process restarts is not guaranteed.

## CPU implementation

- `car`, `collision`, `simulation`, and `broadphase` implement vehicle callbacks, passive physics, contact solving, and native population lifecycle.
- `world`, `curve`, `bsp`, `segment_grid`, and `path_segments` load tracks and accelerate ray, nearest-wall, and path queries while preserving arithmetic and tie order.
- `network` and `network_simd` implement scalar and AVX2 inference with the same accumulation order.
- `godot_math`, `native_math`, `managed_trig`, and `double_math` reproduce the captured game's math routines.
- `training` and `evolution` implement scoring, scheduling, reproduction, and vehicle reuse. `game_random` supports captured game RNG streams through `--game-rng-state`; ordinary seeded training retains the Python-compatible RNG.
- `random_track` exposes generated-track and redraw support through the library. Random-track CLI orchestration remains separate.

Independent mode runs each car through a window to keep its state and network in cache, catching inactive cars up to the final callback. Lockstep advances all cars one tick at a time. Scenes requesting native shared broadphase use the tick-major path in either mode to retain native ordering.

## Verification

```sh
cargo test --offline -- --test-threads=2
cargo test --release --offline -- --test-threads=2
python check_accuracy.py --label cpu-port \
  --cases a01 a05 a06 a07 b06 --closed-loop --sensors \
  --assert-position 0 --assert-closed-loop-exact
```

The imported regression fixtures cover recorded native vehicle/population transitions, sensors, statistics, runtime math, random tracks, and accelerated-query equivalence. CLI tests cover elimination, settings overrides, resumed parameter changes, and checkpoint shape/size validation. The accuracy checker normalizes the game's float32 JSON state before comparison and writes reports under `reports/`.

The experiment's [ROUND3.md](../experiments/driving_sim_rs_fidelity/ROUND3.md) and [ROUND4.md](../experiments/driving_sim_rs_fidelity/ROUND4.md) record the original capture evidence and remaining fidelity limits. Native body matrices are absent from older traces; passing these finite fixtures is not proof of complete state reconstruction.

Godot-derived code retains its notice in [GODOT_LICENSE](GODOT_LICENSE); math source notices remain in their modules.
