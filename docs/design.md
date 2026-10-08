# Design notes

The simulator keeps arithmetic and execution ordering visible. A small algebraic rewrite can change rounding, contact choices or evolution, so numeric code and scheduling need stronger regression evidence than ordinary I/O changes.

## Modules and ownership

| Directory | Modules | Responsibility |
| --- | --- | --- |
| `src/math/` | `godot_math`, `native_math`, `managed_trig`, `double_math`, `pymath`, `vec2` | Captured rounding, runtime arithmetic and vectors |
| `src/physics/` | `car`, `collision`, `simulation`, `broadphase`, `trace_state` | Forces, passive bodies, contacts, population physics and recorded-state reconstruction |
| `src/track/` | `world`, `curve`, `bsp`, `segment_grid`, `path_segments`, `random_track`, `training_tracks` | Scene loading, geometry queries, tile generation and bounded preparation queue |
| `src/nn/` | `network`, `network_simd` | Flat parameters and scalar/AVX2 inference |
| `src/training/` | `agent`, `lineage`, `sensors`, `stats`, `runner`, `evolution`, `batch_evaluation`, `game_random`, `pyrandom` | Populations, statistics, ordered execution, reproduction and RNG backends |
| `src/training/` | `session`, `evaluation` | Embedding, generation checkpoints and evaluation reports |
| `src/gpu/` | `hip`, `simulation` | Runtime-loaded HIP C API, shared layouts and GPU windows |
| `src/wasm/` | `gpu`, `sim` | JavaScript bindings, WebGPU rays and full simulation |

Library imports follow the directory groups. For example, `physics::car::Car` and `track::world::World` describe their roles directly. The training scheduler types remain available from `training`, and embedding sessions live in `training::session`. HIP loading uses `gpu::hip`; portable GPU layouts live in `gpu::simulation`.

The CLI entrypoint is `src/bin/altd-sim/main.rs`. Its sibling modules use Cargo's standard binary layout: `args` defines the command contract, `comparisons` produces recorded-trace reports, `bench` and `train` run their workloads, and `scratch` owns run configuration, stopping rules, checkpoint I/O and execution. Scratch parameter checkpoints and Session lineage checkpoints remain separate formats.

The GPU validation example uses private modules in `examples/gpu_check/` for math, rays, simulation, evolution and benchmarks. Shared fixture setup and mismatch reporting stay outside the checked numerical implementations.

CPU runners own populations and checkpoint serialization. HIP retains uploaded networks and geometry between windows; CPU evolution installs offspring and preserves the CPU/GPU resume format. WebGPU full simulation mirrors state in browser buffers and commits complete windows back to the Session. A failed window leaves the last committed state available for CPU continuation.

## Execution modes

Independent mode advances each car through a window, keeping its state and network near the CPU. Lockstep advances the population one tick at a time. Scenes requesting native shared broadphase require tick-major execution to preserve population contact ordering. Generated CLI training disables that shared broadphase so CPU and GPU use independent car physics.

Statistics run before driving. Inference uses eight original index-based batches, so compacting inactive work must preserve each car's batch identity. Eliminated cars stop driving and scoring, but passive physics and pending contact accounting continue. The generation limit is checked on the statistics callback, so actual callback counts can slightly exceed a requested tick cap.

Raycaster windows keep physics and inference on the CPU and replace only ray queries. HIP and full WebGPU windows run those phases on-device. These paths have distinct scheduling and memory costs; shared arithmetic names do not make their loop structure interchangeable.

## Reproducible tracks and checkpoints

Track seeds depend on the run seed, generation and batch slot. Their RNG stream is separate from network reproduction. Bounded queue depth and worker timing therefore do not affect which tracks a generation receives. Each batch shares a track across its population and breeds once after all tracks.

Scratch CLI checkpoints save the next generation's population and RNG state at generation boundaries. Session checkpoints can encode parent lineage instead of every child network while retaining exact reconstruction. Those formats have different consumers and should be treated as separate contracts. Neither promises to serialize every mid-tick native contact cache.

## Numerical constraints

Keep float32/float64 transitions, reduction order, runtime transcendental behavior and equal-distance query decisions explicit. Disable FMA contraction in HIP builds. Preserve cached Gaussian draws and both captured game RNG streams. Generate shared numeric tables from their bit patterns and check the generated header before building; GPU builds must not rewrite source files.

Performance changes need workload medians as well as exact state, checkpoint and RNG comparisons. Profiling paths alter launch behavior, so final timing should use production settings. [Fidelity](fidelity.md) and [performance](performance.md) describe the evidence and its boundaries.

## Formatting and lint checks

`cargo fmt --all -- --check` uses the repository's 120-column width. Native checks use `cargo clippy --offline --all-targets -- -D warnings`. Browser bindings use `cargo clippy --offline --target wasm32-unknown-unknown --no-default-features --features wasm -- -D warnings`. Scoped lint allowances explain preserved runtime constants, nonfinite comparisons, indexed numerical loops and packed GPU layouts. They do not relax warnings for unrelated code.

## Optimizers

`TrainingRunner` evaluates a whole population, scores it with `reward_values` (ranks within the generation), and hands the cars and scores to an `Optimizer` (`src/training/optimizer.rs`), which returns the next population. `settings.algorithm` chooses one; the runner replaces its optimizer when the name changes, and the new one continues from the cars it is next given.

- `"ga"` is the default: selection, crossover, mutation and decay as in `evolution.rs`. Its random draws, and so every GA checkpoint and test, are unchanged.
- `"ars"` (`src/training/ars.rs`) is Augmented Random Search V2-t with an elite pool. It keeps `heads` search points `theta` (one by default); the population is the elites, one `theta` per head, and antithetic probes `theta ± nu·d` whose pairs are split evenly among the heads (the remainder goes to head 0). The scores steer one step per head, `alpha/(b·nu)·Σ(r+ − r−)·d` over the best `top_frac` of its pairs, skipped when its scores are all alike. The heads do not interact. Elites are the best distinct cars of the last population, old elites included; they are re-run unchanged every generation, so ranks from different generations are never compared. Settings live under `settings.ars` (`nu`, `alpha`, `top_frac`, `elite_count`, `max_weight`, `heads` and the `alpha_*` fields below), and `population` must be at least `elite_count + 3·heads`.

Each head adapts its own `alpha`, starting from `alpha`. Every `alpha_adapt_every` generations the fast and slow moving averages (decays `alpha_adapt_fast`, `alpha_adapt_slow`) of its search point's raw reward are compared: a fast average more than `alpha_adapt_threshold · |slow|` above the slow one multiplies `alpha` by `alpha_adapt_up`, one below it by `alpha_adapt_down`, always within `[alpha_min, alpha_max]` (equal bounds freeze `alpha`). The raw reward is the weighted mean of the reward metrics (`raw_reward_values`); a car missing a metric adds nothing to the averages. `nu` is not adapted.

Selection rewards stay rank based for every algorithm. ARS therefore behaves like rank-based OpenAI-ES with top-b pair selection rather than reference ARS on raw magnitudes; only the step size trend uses raw values.

Session checkpoints record how a population was made, so each optimizer has its own format: `ALTDCKP2` for GA parent lineage and `ALTDCKP4` for ARS, which stores every head's `theta` and step size state, the elite pool, the sampling settings and the generator before the directions were drawn (`ALTDCKP3`, one head without step size state, still loads). Restoring one resamples the population and the directions exactly. A full population checkpoint (`ALTDCKP1`, JSON checkpoints, scratch CLI checkpoints) carries no search state, so ARS continues from the best restored car. A changed population size does the same.
