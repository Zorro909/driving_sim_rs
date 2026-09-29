# GPU performance investigation, 2026-09-29

The retained implementation adds physics launch bounds, partitions live work by
original inference batch, and reuses compatible HIP graphs. Physics arithmetic,
contact order, inference timing, float precision, and exact settling rules are
unchanged. The GPU library was rebuilt with the original accuracy flags.

## Method

- AMD Radeon RX 7900 XTX, gfx1100, HIP 7.1.52802, clang 20.0.0.rocm.
- Ryzen 9 7950X, `RAYON_NUM_THREADS=8` for every timed run.
- B06 Hard, rally vehicle, checkpoint generation 7,000 from `/tmp/altd-gpu-ckpt`.
- The fixture has 8,192 networks. Larger populations repeat them. Network shape
  is `20,16,16,16,16,12,12,8,5`, with 1,661 double parameters per car.
- `batch_count=2`, which updates four of the eight original inference batches
  per tick. Batch identity remains tied to the original car index.
- The full-run workload disables wall and idle elimination. A 5,400-tick time
  limit ends at statistics callback 5,406, before that callback's driving step.
- The early-stop workload enables both elimination modes and uses the same
  time limit. This fixture eliminates every car after 462 callbacks at 8K and
  486 at 32K/256K, so it must not be confused with the full-run workload.
- Final comparisons use one warmup and three timed samples. The intermediate
  ablations use one warmup and two timed samples. GPU workloads ran serially.
- Each sample restores the same car/agent state and uploads the same networks.
  Timing includes network packing/upload, state export/upload, the window,
  download and state import. It excludes fixture construction, resetting,
  hashing, reproduction and checkpoint I/O.
- Normal graph execution and split physics remain enabled. Phase profiling is
  disabled because `ALTD_GPU_PROFILE=1` changes the execution path.

The baseline library was compiled from an untouched source copy with the same
compiler and build flags. JSON files record individual timings and hashes of
the library, checkpoint and final simulation state. The workstation desktop
remained running; small differences between ablations should be treated as noise.

## Full-run results

Median seconds per generation. These use three measured samples after warmup.

| Cars | Baseline | Final | Speedup | Time reduction |
|---:|---:|---:|---:|---:|
| 8,192 | 2.363 | 2.094 | 1.128x | 11.4% |
| 32,768 | 6.290 | 5.715 | 1.101x | 9.1% |
| 262,144 | 43.838 | 40.315 | 1.087x | 8.0% |

Every warmup and measured sample matches the baseline final-state digest at
its population size. Digests include normalized car state, agent statistics,
controls and scheduler state. CPU comparisons are reported separately below.

See `baseline-full.json` and `final-full.json` for individual measurements.

## Early-stop results

Both elimination modes are enabled. This checkpoint stops early; these are
not 5,400-tick simulations. Medians use three timed samples after warmup.

| Cars | Callbacks | Baseline seconds | Final seconds | Speedup |
|---:|---:|---:|---:|---:|
| 8,192 | 462 | 0.215 | 0.210 | 1.023x |
| 32,768 | 486 | 0.635 | 0.595 | 1.068x |
| 262,144 | 486 | 4.384 | 4.315 | 1.016x |

Final states match the baseline for every sample. Improvements are smaller
than in the full runs; the 256K difference is within the observed run-to-run
variation and should not be treated as a clear speedup.

## Ablations

These are cumulative until the retained combination; medians use two samples.
Differences below about one percent are not a reason to prefer a variant.

| Variant | 8K seconds | 32K seconds |
|---|---:|---:|
| Original | 2.363 | 6.290 |
| Launch bounds | 2.362 | 6.084 |
| + Original-batch work lists | 2.090 | 5.804 |
| + Subgroup inference | 2.131 | 5.801 |
| + Graph reuse | 2.131 | 5.810 |
| + Cached car frame | 2.163 | 5.841 |
| + Parallel contact generation | 2.134 | 5.855 |
| Retained combination, 64 threads | 2.083 | 5.731 |
| Retained combination, 128 threads | 2.120 | 5.698 |
| Retained combination, 256 threads | 2.096 | 5.710 |

The original row uses the three-sample baseline. Graph caching removes setup
work, but its isolated runtime effect is below the resolution of these runs.
Batch-specific work lists provide the clearest improvement.

## Retained changes

1. Physics kernels declare their actual 64-thread launch bound. Compiled
   `w_step_end_kernel` LDS allocation falls from 32,768 to 2,048 bytes per block;
   its 80 VGPRs and 112 bytes of private storage are unchanged. The fused
   alternative receives the same bound. See `kernel-resources.json`.
2. Live lists have eight separately compacted segments. Statistics rebuild the
   segments; sensor and forward grids visit only scheduled segments. Partial
   batches remain padded safely and retain original car indices. The independent
   physics list still steps inactive cars until they settle bit-for-bit.
3. The simulator retains a compatible executable graph between windows. Changes
   to schedule, population size, network shape or output mapping invalidate it.
   Replacing weights in the existing buffer does not. This removes repeated
   graph construction without changing captured computation.

## Experiments removed

- A 16-lane shuffle implementation of inference removed block barriers and LDS
  usage, but did not improve the measured runtime. The original shared-memory
  forward implementation remains.
- Computing the car frame once per tick increased intermediate-buffer traffic
  and did not improve runtime in this workload.
- Parallel contact generation preserved exact results but required larger pair
  buffers and did not produce a net speedup. The existing parallel SAT and
  ordered per-car contact generation remain.
- Physics blocks of 128 and 256 threads did not offer a consistent advantage
  over 64 across both tested population sizes.

No full car-state layout rewrite or multi-tick persistent kernel was introduced.
Those remain separate, larger experiments.

## Validation

The retained build passed 130,412 agent comparisons against the CPU with zero
flagged errors or mismatches, covering:

- B06 partial windows, uneven populations, statistics phases and batch wrapping.
- Reused simulators with changed population size, network shape, weights and
  window length, including a remainder after graph replay.
- Full generations and reproduction with wall elimination, idle elimination,
  and both enabled.
- Direct launches with `ALTD_GPU_GRAPH=0`.
- Fused physics with `ALTD_GPU_SPLIT=0`.
- A07 Three Terrains with its contact-reference spawn.

The `final-*.log` files contain the CPU comparisons. Every baseline/final
benchmark sample at all three sizes also produced identical state digests, for
both full-length and early-stop workloads. This is regression evidence for the
covered inputs, not an exhaustive proof over every possible scene.

The discarded combined geometry/contact/shuffle experiment additionally passed
75,776 inference comparisons and 819,200 individual physics comparisons before
being rejected on performance. Those checks are in `contacts-parity.log`.

```sh
RAYON_NUM_THREADS=8 target/release/examples/gpu_check schedules reuse window window3
RAYON_NUM_THREADS=8 ALTD_GPU_GRAPH=0 target/release/examples/gpu_check schedules reuse window3
RAYON_NUM_THREADS=8 ALTD_GPU_SPLIT=0 target/release/examples/gpu_check schedules reuse window3
RAYON_NUM_THREADS=8 ALTD_GPU_SCENE=scenes_exact/rally_a07_contact_scene.json \
  ALTD_GPU_SPAWN=../docs/traces/rally_a07_contact_reference.json \
  target/release/examples/gpu_check schedules reuse window window3
```

## Reproduction

From `driving_sim_rs`, with the same frozen checkpoint available:

```sh
gpu/build.sh
cargo build --release --example gpu_check
RAYON_NUM_THREADS=8 python3 gpu/benchmark.py \
  --library target/gpu/libaltd_gpu.so --no-elimination \
  --output reports/gpu-full.json
RAYON_NUM_THREADS=8 python3 gpu/benchmark.py \
  --library target/gpu/libaltd_gpu.so \
  --output reports/gpu-early-stop.json
```

`ALTD_GPU_CKPT` selects another checkpoint. Its checksum must match the recorded
one to reproduce these inputs. The benchmark runner's default populations are
8,192, 32,768 and 262,144.
