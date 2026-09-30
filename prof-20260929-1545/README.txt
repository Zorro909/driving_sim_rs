altd GPU profiling run 2026-09-29T15:45:55+02:00
repo: /home/zorro/git/AILearnsToDrive/driving_sim_rs  git: 39bb745 18 modified files
library: /home/zorro/git/AILearnsToDrive/driving_sim_rs/target/gpu/libaltd_gpu.so
checkpoint: /tmp/altd-gpu-ckpt
sizes: 8192 32768 262144  rayon threads: 8

cb3395899f94cf7aadbb907e2771d7173987b80dd97730f78b4e77de5fdc009b  /home/zorro/git/AILearnsToDrive/driving_sim_rs/target/gpu/libaltd_gpu.so
level 1: ALTD_GPU_PROFILE=1 prints GPU seconds per phase and per 600 ticks; graphs are disabled in this mode
phases_8192_elim0: rc=0 seconds=3 :: env ALTD_GPU_BENCH_CARS=8192 ALTD_GPU_BENCH_ELIMINATE=0 ALTD_GPU_BENCH_WARMUPS=0 ALTD_GPU_BENCH_SAMPLES=1 ALTD_GPU_PROFILE=1 target/release/examples/gpu_check benchfixed
phases_8192_elim1: rc=0 seconds=1 :: env ALTD_GPU_BENCH_CARS=8192 ALTD_GPU_BENCH_ELIMINATE=1 ALTD_GPU_BENCH_WARMUPS=0 ALTD_GPU_BENCH_SAMPLES=1 ALTD_GPU_PROFILE=1 target/release/examples/gpu_check benchfixed
phases_32768_elim0: rc=0 seconds=7 :: env ALTD_GPU_BENCH_CARS=32768 ALTD_GPU_BENCH_ELIMINATE=0 ALTD_GPU_BENCH_WARMUPS=0 ALTD_GPU_BENCH_SAMPLES=1 ALTD_GPU_PROFILE=1 target/release/examples/gpu_check benchfixed
phases_32768_elim1: rc=0 seconds=2 :: env ALTD_GPU_BENCH_CARS=32768 ALTD_GPU_BENCH_ELIMINATE=1 ALTD_GPU_BENCH_WARMUPS=0 ALTD_GPU_BENCH_SAMPLES=1 ALTD_GPU_PROFILE=1 target/release/examples/gpu_check benchfixed
phases_262144_elim0: rc=0 seconds=51 :: env ALTD_GPU_BENCH_CARS=262144 ALTD_GPU_BENCH_ELIMINATE=0 ALTD_GPU_BENCH_WARMUPS=0 ALTD_GPU_BENCH_SAMPLES=1 ALTD_GPU_PROFILE=1 target/release/examples/gpu_check benchfixed
phases_262144_elim1: rc=0 seconds=16 :: env ALTD_GPU_BENCH_CARS=262144 ALTD_GPU_BENCH_ELIMINATE=1 ALTD_GPU_BENCH_WARMUPS=0 ALTD_GPU_BENCH_SAMPLES=1 ALTD_GPU_PROFILE=1 target/release/examples/gpu_check benchfixed
phases_8192_elim1_fused: rc=0 seconds=0 :: env ALTD_GPU_BENCH_CARS=8192 ALTD_GPU_BENCH_ELIMINATE=1 ALTD_GPU_BENCH_WARMUPS=0 ALTD_GPU_BENCH_SAMPLES=1 ALTD_GPU_PROFILE=1 ALTD_GPU_SPLIT=0 target/release/examples/gpu_check benchfixed
phases_32768_elim1_fused: rc=0 seconds=3 :: env ALTD_GPU_BENCH_CARS=32768 ALTD_GPU_BENCH_ELIMINATE=1 ALTD_GPU_BENCH_WARMUPS=0 ALTD_GPU_BENCH_SAMPLES=1 ALTD_GPU_PROFILE=1 ALTD_GPU_SPLIT=0 target/release/examples/gpu_check benchfixed
level 1b: production-mode wall clock, split vs fused step (graph on, no profiling)
wall_split1_8192_elim0: rc=0 seconds=7 :: env ALTD_GPU_SPLIT=1 ALTD_GPU_BENCH_CARS=8192 ALTD_GPU_BENCH_ELIMINATE=0 ALTD_GPU_BENCH_WARMUPS=1 ALTD_GPU_BENCH_SAMPLES=2 target/release/examples/gpu_check benchfixed
wall_split1_8192_elim1: rc=0 seconds=1 :: env ALTD_GPU_SPLIT=1 ALTD_GPU_BENCH_CARS=8192 ALTD_GPU_BENCH_ELIMINATE=1 ALTD_GPU_BENCH_WARMUPS=1 ALTD_GPU_BENCH_SAMPLES=2 target/release/examples/gpu_check benchfixed
wall_split1_32768_elim0: rc=0 seconds=20 :: env ALTD_GPU_SPLIT=1 ALTD_GPU_BENCH_CARS=32768 ALTD_GPU_BENCH_ELIMINATE=0 ALTD_GPU_BENCH_WARMUPS=1 ALTD_GPU_BENCH_SAMPLES=2 target/release/examples/gpu_check benchfixed
wall_split1_32768_elim1: rc=0 seconds=4 :: env ALTD_GPU_SPLIT=1 ALTD_GPU_BENCH_CARS=32768 ALTD_GPU_BENCH_ELIMINATE=1 ALTD_GPU_BENCH_WARMUPS=1 ALTD_GPU_BENCH_SAMPLES=2 target/release/examples/gpu_check benchfixed
wall_split0_8192_elim0: rc=0 seconds=8 :: env ALTD_GPU_SPLIT=0 ALTD_GPU_BENCH_CARS=8192 ALTD_GPU_BENCH_ELIMINATE=0 ALTD_GPU_BENCH_WARMUPS=1 ALTD_GPU_BENCH_SAMPLES=2 target/release/examples/gpu_check benchfixed
wall_split0_8192_elim1: rc=0 seconds=1 :: env ALTD_GPU_SPLIT=0 ALTD_GPU_BENCH_CARS=8192 ALTD_GPU_BENCH_ELIMINATE=1 ALTD_GPU_BENCH_WARMUPS=1 ALTD_GPU_BENCH_SAMPLES=2 target/release/examples/gpu_check benchfixed
wall_split0_32768_elim0: rc=0 seconds=20 :: env ALTD_GPU_SPLIT=0 ALTD_GPU_BENCH_CARS=32768 ALTD_GPU_BENCH_ELIMINATE=0 ALTD_GPU_BENCH_WARMUPS=1 ALTD_GPU_BENCH_SAMPLES=2 target/release/examples/gpu_check benchfixed
wall_split0_32768_elim1: rc=0 seconds=5 :: env ALTD_GPU_SPLIT=0 ALTD_GPU_BENCH_CARS=32768 ALTD_GPU_BENCH_ELIMINATE=1 ALTD_GPU_BENCH_WARMUPS=1 ALTD_GPU_BENCH_SAMPLES=2 target/release/examples/gpu_check benchfixed
level 4: a second library built with -DALTD_STEP_TIMING (atomics slow the step; only the ratios matter)
build_timing_library: rc=0 seconds=4 :: hipcc -O3 -ffp-contract=off --offload-arch=gfx1100 -fPIC -shared -std=c++20 -fhip-fp32-correctly-rounded-divide-sqrt -DALTD_STEP_TIMING -o /home/zorro/git/AILearnsToDrive/driving_sim_rs/target/gpu/libaltd_gpu_timing.so gpu/sim/altd_gpu.hip -lamdhip64
sections_8192_elim0: rc=0 seconds=3 :: env ALTD_GPU_BENCH_CARS=8192 ALTD_GPU_BENCH_ELIMINATE=0 ALTD_GPU_BENCH_WARMUPS=0 ALTD_GPU_BENCH_SAMPLES=1 ALTD_GPU_LIB=/home/zorro/git/AILearnsToDrive/driving_sim_rs/target/gpu/libaltd_gpu_timing.so ALTD_GPU_PROFILE=1 target/release/examples/gpu_check benchfixed
sections_8192_elim1: rc=0 seconds=1 :: env ALTD_GPU_BENCH_CARS=8192 ALTD_GPU_BENCH_ELIMINATE=1 ALTD_GPU_BENCH_WARMUPS=0 ALTD_GPU_BENCH_SAMPLES=1 ALTD_GPU_LIB=/home/zorro/git/AILearnsToDrive/driving_sim_rs/target/gpu/libaltd_gpu_timing.so ALTD_GPU_PROFILE=1 target/release/examples/gpu_check benchfixed
sections_262144_elim0: rc=0 seconds=52 :: env ALTD_GPU_BENCH_CARS=262144 ALTD_GPU_BENCH_ELIMINATE=0 ALTD_GPU_BENCH_WARMUPS=0 ALTD_GPU_BENCH_SAMPLES=1 ALTD_GPU_LIB=/home/zorro/git/AILearnsToDrive/driving_sim_rs/target/gpu/libaltd_gpu_timing.so ALTD_GPU_PROFILE=1 target/release/examples/gpu_check benchfixed
sections_262144_elim1: rc=0 seconds=15 :: env ALTD_GPU_BENCH_CARS=262144 ALTD_GPU_BENCH_ELIMINATE=1 ALTD_GPU_BENCH_WARMUPS=0 ALTD_GPU_BENCH_SAMPLES=1 ALTD_GPU_LIB=/home/zorro/git/AILearnsToDrive/driving_sim_rs/target/gpu/libaltd_gpu_timing.so ALTD_GPU_PROFILE=1 target/release/examples/gpu_check benchfixed
  kernel resource usage (VGPRs, SGPRs, scratch, LDS, occupancy) of the production flags
kernel_resources: rc=0 seconds=4 :: hipcc -O3 -ffp-contract=off --offload-arch=gfx1100 -fPIC -shared -std=c++20 -fhip-fp32-correctly-rounded-divide-sqrt -Rpass-analysis=kernel-resource-usage -o /home/zorro/git/AILearnsToDrive/driving_sim_rs/prof-20260929-1545/.tmp_resources.so gpu/sim/altd_gpu.hip -lamdhip64
