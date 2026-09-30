# GPU simulator performance investigation, 2026-09-29 (second pass)

Scope: the driving simulation (`src/training.rs`, `src/evolution.rs`, `src/pyrandom.rs`)
and the HIP kernels in `gpu/sim`, with the constraint that every result stays
bit-identical. This pass was done on a machine without a GPU or ROCm, so it
combines code inspection with the measurements already in `reports/`, host-side
measurements that need no GPU, and kernel edits that are exact by construction
and type-checked with clang's HIP front end. **The kernel edits have not been run
on hardware yet; run the `gpu_check` parts listed under Verification before use.**

## Measured on the RX 7900 XTX (`gpu/profile.sh`, ROCm 7.1)

The per-phase GPU timings of complete generations, with the kernels of this
pass. The result digests of all eight runs equal the September baseline's, so
these kernels reproduce the previous results bit for bit on hardware.

Full run (5,406 ticks, no elimination), GPU seconds per generation:

| Cars | stats | sensors | forward | step | GPU total | wall | forward share |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 0.13 | 0.55 | 0.90 | 0.55 | 2.11 | 2.26 | 43% |
| 32,768 | 0.18 | 1.14 | 3.08 | 1.01 | 5.40 | 5.72 | 57% |
| 262,144 | 0.87 | 7.08 | 25.09 | 5.80 | 38.84 | 40.60 | 65% |

Early-stop run (elimination on, 462–486 ticks):

| Cars | stats | sensors | forward | step | wall |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 0.009 | 0.030 | 0.061 | 0.035 | 0.20 |
| 32,768 | 0.012 | 0.045 | 0.179 | 0.067 | 0.54 |
| 262,144 | 0.048 | 0.234 | 1.376 | 0.597 | 3.97 |

What this changes about the earlier reading:

- **The forward pass is the bottleneck, not the step.** It costs a constant
  35 ns per car-inference from 32K cars up (40 ns at 8K), 4.6 ms per tick at
  262,144 cars. That is only 380 GB/s of parameter traffic, so it is not
  bandwidth bound: it is double-precision issue bound. RDNA3 runs double
  precision at 1/32 of the single-precision rate, so a wave's forward pass of
  about 700 double instructions (232 for the matrix products, the rest
  `tanh`) would already take 2 ms per tick; the measured 4.6 ms corresponds to
  about 2,100, which matches the three `tanh` ranges being executed one after
  another whenever the lanes of a wave fall in different ranges.
- **The step is cheap at scale** (4 ns per car-tick, 15% at 262,144) and a
  latency floor at small populations (101 µs per tick at 8K, 26%). Its
  section timings (`-DALTD_STEP_TIMING`, thread time per car step) put the
  wheel loop at 39%, the broad-pair update at 28% (91% of steps run it, since
  the leaf box grows by only 0.1 px), SAT and contact generation at 18%, and
  the solver at only 4% in full runs (42% in the early-stop run, where 56% of
  the eliminated cars never settle and keep pressing against walls).
- **Sensors** are 17–26% and grow along a generation as cars spread out
  (0.51 s to 0.92 s per 600 ticks at 262,144).
- **Split versus fused step** is a wash in production mode: within 4% either
  way at 8K and 32K, full and early-stop, with identical digests. Not worth a
  dual-mode dispatch.
- The GPU kernels account for 96% of wall time at 262,144 cars; host work
  and transfers are the remaining 1.8 s.

### The forward-pass fix: branch-free `game_tanh`

`gpu/sim/math.h` now evaluates `tanh` without branches: one `expm1` per lane
on a selected argument (`2x` or `-2x`), then selects between the range
results, exactly as the CPU's AVX2 `tanh4` does. Every lane computes the same
expression the scalar code computes on its range; unselected values are
dropped. Compiled as host C++ on x86 with contraction off, the new function
equals the original bit for bit over 35 million inputs (gcc and clang) covering every
threshold, every `expm1` case boundary, denormals, infinities and NaN
payloads. The first hardware run of it showed a NaN payload difference (the
GPU folds `fabs` into a select as a float modifier, which rewrites NaN high
words under IEEE mode), so the NaN, infinity and |x| > 20 range keeps its
original branch; `gpu_check math` then passed.

Measured on the workstation with all digests unchanged: the forward pass went
from 25.09 to 24.20 s at 262,144 cars, 3.08 to 3.01 s at 32K and 0.90 to
0.83 s at 8K, so 3–7%, far less than the divergence reading predicted. The
compiler had evidently merged the branches already, and the cost is the
double-precision instruction count itself. Two further exact reductions
followed: the forward kernel packs each layer's outputs densely across the
block's cars, so the 12-, 8- and 5-wide layers occupy 6, 4 and 3 waves of the
block instead of 8 with idle lanes (about 20% fewer issue slots over the
network), and `tanh` uses one division per range instead of two and integer
comparisons for its integer `k`. `gpu/profile.sh` level 5 builds diagnostic
libraries with the activation, the weight traffic or the products removed to
attribute the remaining time; `-DALTD_TANH_BRANCHY` rebuilds the scalar tanh
for an A/B, and `gpu_check math infer window` verifies every variant that is
meant to be exact.

Third hardware run, with the packing and the tanh reductions (all digests
unchanged, `gpu_check math novelty turnover step` all OK): forward 20.88 s at
262,144 cars (from 24.20), 2.62 s at 32K (from 3.01), 0.73 s at 8K (from
0.83); production wall clock 36.2 s, 4.93 s and 1.86 s, against the original
40.6, 5.72 and 2.09 s. The attribution at 32K (forward 2.62 s): the
activation costs 0.82 s, the products with their weight loads 1.03 s, the
weight memory traffic alone 0.40 s, and 0.77 s remains as per-layer overhead;
the scalar tanh is 9% slower than the branch-free one now. Per wave that is
about 91 cycles per weight row where the two double operations need 32 to 40:
the row loop stalls on its streaming loads (each row is a fresh cache line),
and each layer starts with a cold load and a cold bias. The forward kernel
therefore now loads weight rows eight at a time one chunk ahead, and issues a
layer's first chunk and bias during the previous layer's activation, with the
products still added in input order.

Fourth run: the pipelining changed nothing (forward 2.64 s at 32K, 21.2 s at
262,144, digests unchanged). The likely reason is that `__syncthreads()`
compiles to a barrier plus a workgroup fence that, in the default WGP mode,
waits for every outstanding global load, so loads issued before a layer
barrier are drained there. With 16 barriers per inference, that also makes
the barriers the prime suspect for the 30% of the forward pass that neither
the products nor the activation account for. The kernel now alternates
between two LDS activation buffers, one barrier per layer, and the level-5
attribution gained three variants: `diag4` (layer loop only), `diag5` (the
same without layer barriers, a timing-only race) and `cumode` (a `-mcumode`
build, whose workgroup fences need not drain global loads).

## Where the time went before this pass (from the earlier reports)

From `gpu-performance-20260929` (B06, 5,400 ticks, no elimination):

| Cars | s/generation | µs per tick | ns per car-tick |
|---:|---:|---:|---:|
| 8,192 | 2.094 | 387 | 47 |
| 32,768 | 5.715 | 1,057 | 32 |
| 262,144 | 40.315 | 7,457 | 28 |

A linear fit gives about 150 µs of per-tick cost that does not shrink with the
population plus about 28 ns per car-tick. The fixed part is not launch overhead
alone: the physics step runs about six dependent kernels per tick, each of whose
critical path is one thread's serial chain (driving, pair update, SAT, contact
generation, 16 solver iterations, integration, transcendental angle math, plus
dozens of dependent global loads), and that chain does not get shorter with
fewer cars. The report's own note ("117 µs per tick late in a generation") is
the same effect.

The per-car part at large populations has an irreducible component: the forward
pass reads each car's own 1,661 double parameters (13.3 KB) every second tick,
1.74 GB per tick at 262,144 cars, which is 1.8–2.5 ms of the 7.5 ms per tick at
the RX 7900 XTX's bandwidth, about 10 s of the 40 s generation. Nothing exact can
shrink it (weights are per car, double precision, and incompressible), so the
remaining levers at scale are the step kernels (memory-latency bound with a
3,200-byte array-of-structures car layout) and the sensors.

For training generations with elimination (`gpu-turnover-20260929`), the window
is 3.5–4.5 s and turnover 2.4 s at 262,144 cars. The turnover profile at 32K
(`shared-buffer-training-32k.log`) scales to roughly: Gaussian draws 0.8 s,
network packing and upload 0.3 s, parent copies 0.1–0.25 s, mutation 0.2 s,
novelty 0.2 s, installation 0.5 s, plus 0.3–0.4 s of car and agent export at the
start of the next window.

## Changes in this pass

All of them keep every floating-point operation and its order. Verified here by
the unit tests named below, by `cargo test` (81 tests; three end-to-end tests
need `../docs/traces` fixtures that are not in this checkout), and by running the
baseline and modified `altd-sim` binaries side by side on the self-contained
Autumn 04 fixture: logs, RNG state, population bytes, checkpoints and best-lap
files were identical for a tournament/no-crossover run and a
roulette/single-point/weight-decay run, each followed by a `--resume`.

### Host side (measurable without a GPU)

1. **Retained export buffers** (`export_state_into`, `GpuSim::take_state_buffers`).
   `advance_window_gpu` built fresh `Vec<GpuCar>`/`Vec<GpuAgent>` every window
   through a `Result` collect, which allocates and page-faults 3,200 bytes per car
   and concatenates per-thread pieces. The buffers are now retained and filled in
   place. On this 4-core machine at 32K cars: 0.07–0.09 s per window before,
   0.006–0.010 s after warm-up. Their profile showed 0.045 s at 32K on the
   workstation; expect about 0.3 s per generation saved at 262,144 cars.
   Test: `export_state_into_matches_per_agent_export`.
2. **Offspring are moved into the agents** (`install_with_novelty` takes
   `Vec<Network>`; `next_generation_gpu` returns a `Turnover` summary). The GPU
   turnover cloned every offspring network into its agent, a full population copy
   (3.5 GB at 262,144 cars). `gpu_check`'s turnover check now compares the
   installed agents' networks. Test: `owned_install_matches_cloning_install`.
3. **Fused copy and mutation** (`evolution::mutated_copy`). With crossover
   `none`, reproduction cloned each parent, then mutated the clone in place, then
   applied weight decay: one copy pass and two read-modify-write passes over the
   population. Each child is now written once as `p + (0.0 + z * deviation)`,
   times `(1.0 - decay)` when positive, the same expression tree as before.
   Other crossover algorithms keep the previous path. Test:
   `fused_copy_mutation_matches_sequential_reproduction` (all three crossover
   algorithms, three selection algorithms, zero/positive/adaptive mutation, weight
   decay, a cached Gaussian across the call, generator state and next draw).
4. **Pipelined Gaussian draws** (`PyRandom::standard_normals_into`). The
   Mersenne Twister is sequential, but its Box–Muller transform now runs on the
   pool for each 65,536-value block while the next block is drawn, instead of
   after all draws. On the workstation the transform was 0.07 s and the draws
   0.03 s per 32K generation, so the overlap saves up to the draw time. Tests:
   the existing batched-draw tests plus counts around and beyond the block size.

Together these remove three full-population copies and one full-population
allocation per generation and overlap the RNG with its transform. Scaling the
workstation's 32K profile, that is roughly 0.10 s of the 0.79 s generation at
32K and 0.8–1.0 s of the 6.3 s generation at 262,144 cars. These are estimates
from their profile, not new measurements on that machine.

### Kernels (type-checked, not yet run)

5. `forward_car`: the shared activation rows are `MAX_WIDTH + 1` doubles apart.
   The four cars of a wave read the same column at the same time, and rows 256
   bytes apart map to the same LDS banks.
6. `step_begin`: `godot_ease(handbrake, 0.3)` is evaluated once per distinct
   input per step. Its input is the handbrake control or 0.0 for wheels without
   handbrake power (every B06 wheel), so four `pow` calls per car-tick become one.
7. `step_end`: the car frame (two normalizations) is computed only when a broad
   pair actually pushes; the basis it reads does not change inside the loop.
8. `sat_pair` and `polygon_supports`: the unrolled point loops leave at the
   shape's point count instead of predicating all 16 iterations. B06 shapes have
   3 points (116 of 225) or 4 (77), so about three quarters of the projection
   work per axis was predicated away.
9. `population_mean_kernel`: only `stride` (1,661) threads exist, one per
   parameter, each with one load in flight per sequential add, so the 3.5 GB
   population at 262,144 cars streamed far below memory bandwidth. Each block
   now stages 64 networks' slice of its 64 parameters in LDS with all 256
   threads loading (coalesced rows, many loads in flight), then the 64 summing
   threads add the staged rows in ascending network order, exactly as before.

## Findings not implemented, in order of expected value

- **Forward pass, after the branch-free tanh.** Measure with the hardware
  counters (`gpu/profile.sh` level 3 needs `rocprofv3`; it was not installed
  for the run above): `SQ_INSTS_VALU_*_F64` against `SQ_BUSY_CYCLES` tells how
  close the kernel is to the double-precision issue limit. Remaining exact
  levers are lane utilization (the 12-, 8- and 5-wide layers leave 4, 8 and
  11 of 16 lanes idle during `tanh`, about 21% of the layer work) and the
  `tanh` division count (three per lane now; the `q`/`q2` pair could become
  one division with a selected numerator and denominator, as the same
  operation, if it matches `tanh4`'s tests). Beyond that the remaining cost is
  the arithmetic itself at 1/32 rate, which is the price of double precision on
  this GPU.
- **Sensors** at 17–26%: the BSP stacks live in scratch (208 bytes per lane);
  a parent link per node would move them to two 64-bit register masks, and
  storing the traversal side per level saves the wall reload and three
  `classify` calls at each pop. Computing the path offset in `w_step_end_kernel`
  removes one dependent launch per tick.
- **Step at small populations** (latency floor of about 100 µs per tick): the
  wheel loop is 39% of a step's thread time, dominated by the four `exp`
  evaluations and the surface lookups; the broad-pair update is 28% and runs
  on 91% of steps because the leaf box grows by only 0.1 px (the game's rule,
  so the update frequency cannot change, but its cell scan could compare the
  float32 leaf against float32 shape boxes where those are float32 exact).
  A hot/cold split of the 3,200-byte `Car` struct remains the structural
  lever for the memory side.
- **Split versus fused step**: measured, no difference worth acting on.
- **Compute the path offset in `w_step_end_kernel`** for cars that read sensors
  next tick instead of a separate `w_path_offset_kernel`: one fewer dependent
  launch per tick with the same pure function of the same position. The first
  tick of a window needs it from `w_start_kernel` or one direct launch.
- **Register stacks for the BSP walks.** `raycast` and `closest_wall_point` keep
  a 48-entry index stack in scratch; with a parent link per node the stack
  becomes two 64-bit masks in registers, and storing `front_first` per level
  saves the wall reload and three `classify` calls at each pop.
- **Vehicle reset and passive steps on the GPU** (their own suggestion):
  removes the per-generation export and the CPU passive-step loop, but the step
  kernel must first model `pending_transform_reset`.
- **Novelty `pow(x, 2)`** cannot become `x * x` (musl `pow` is not correctly
  rounded), as the code already notes; the mean kernel above is the exact lever.
- **Direct packing of offspring into the upload buffer** would remove one more
  population copy (`pack`, 0.2 s at 262,144 cars) but needs a second host buffer
  because the noise buffer is reused for packing.
- **`sincos` for the Gaussian transform** was measured here: glibc's separate
  `sin` and `cos` are already fused by the compiler, so there is nothing to gain.

## Verification

```sh
cargo test --release            # 81 pass here; 3 need ../docs/traces fixtures
gpu/build.sh && cargo build --release --examples
target/release/examples/gpu_check math novelty turnover      # kernels 6–9 and the turnover path
target/release/examples/gpu_check step window schedules reuse window3
ALTD_GPU_SPLIT=0 target/release/examples/gpu_check step window3
RAYON_NUM_THREADS=8 python3 gpu/benchmark.py --training --library target/gpu/libaltd_gpu.so --output reports/gpu-training.json
```

`examples/host_bench.rs` times the host stages on the Autumn 04 fixture without
a GPU. The differential CLI comparison used `train-scratch` on the same fixture
with `--eliminate-on-wall --idle-eliminate`, 384 cars for 4 + 1 generations and
96 cars for 3 + 1 generations with roulette selection, single-point crossover and
weight decay, comparing everything except timing fields.
