#!/usr/bin/env bash
# Interactive profiling wizard for the GPU simulator. Collects, into one
# directory, everything needed to see where a generation's time goes:
#
#   1  built-in per-phase GPU time (ALTD_GPU_PROFILE=1) and split/fused wall clock
#   2  per-kernel durations with rocprofv3 (kernel trace + stats)
#   3  hardware counters of the physics/sensor/forward kernels with rocprofv3
#   4  step-section cycle counts (a -DALTD_STEP_TIMING library) and kernel resources
#   5  forward-pass attribution: diagnostic libraries with parts of the network
#      kernel removed (-DALTD_FORWARD_DIAG), and the scalar tanh for an A/B
#
# Every step is optional and failures are recorded, not fatal. Run from anywhere:
#   gpu/profile.sh              # asks before each level
#   gpu/profile.sh --yes        # accept every default without asking
#   gpu/profile.sh --only 1,2   # run only these levels, without asking
#   gpu/profile.sh --sizes "8192 32768"   # populations (default: 8192 32768 262144)
#
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"
export ALTD_GPU_CKPT=~/git/AILearnsToDrive/training_runs/b06_fidelity


YES=0; ONLY=""; SIZES="8192 32768 262144"
while [ $# -gt 0 ]; do
  case "$1" in
    --yes|-y) YES=1 ;;
    --only) ONLY="$2"; shift ;;
    --sizes) SIZES="$2"; shift ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

OUT="$REPO/prof-$(date +%Y%m%d-%H%M)"
mkdir -p "$OUT"
LOG="$OUT/README.txt"
CHECK="target/release/examples/gpu_check"
LIB="${ALTD_GPU_LIB:-$REPO/target/gpu/libaltd_gpu.so}"
export ALTD_GPU_CKPT="${ALTD_GPU_CKPT:-/tmp/altd-gpu-ckpt}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-8}"
# Keep the integrated GPU (the Ryzen's gfx1036) out of every run.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
note() { printf '%s\n' "$*" | tee -a "$LOG"; }
ask() { # ask "question" default(y|n) -> returns 0 for yes
  local q=$1 d=$2 a
  if [ "$YES" = 1 ]; then a=$d; else
    if [ "$d" = y ]; then read -r -p "$q [Y/n] " a </dev/tty; a=${a:-y}; else read -r -p "$q [y/N] " a </dev/tty; a=${a:-n}; fi
  fi
  [[ $a =~ ^[Yy] ]]
}
want_level() { # level number: honoured --only, else asks
  local n=$1 title=$2
  if [ -n "$ONLY" ]; then
    [[ ",$ONLY," == *",$n,"* ]] || return 1
    echo; bold "Level $n: $title"; return 0
  fi
  echo; bold "Level $n: $title"; ask "Run level $n?" y
}
# run_logged <name> <command...>: runs, times, records the exit code; stdout+stderr to $OUT/<name>.txt
run_logged() {
  local name=$1; shift
  local start=$SECONDS
  printf '  %-44s' "$name"
  "$@" > "$OUT/$name.txt" 2>&1
  local rc=$?
  local secs=$((SECONDS - start))
  if [ $rc -eq 0 ]; then echo "ok    (${secs}s)"; else echo "FAILED rc=$rc (${secs}s), see $name.txt"; fi
  echo "$name: rc=$rc seconds=$secs :: $*" >> "$LOG"
  return $rc
}
bench_env() { # bench_env cars eliminate [extra VAR=val ...]
  local cars=$1 elim=$2; shift 2
  echo env ALTD_GPU_BENCH_CARS="$cars" ALTD_GPU_BENCH_ELIMINATE="$elim" ALTD_GPU_BENCH_WARMUPS=0 ALTD_GPU_BENCH_SAMPLES=1 "$@"
}

bold "GPU profiling wizard, output in $OUT"
{
  echo "altd GPU profiling run $(date -Iseconds)"
  echo "repo: $REPO  git: $(git rev-parse --short HEAD 2>/dev/null) $(git status --short 2>/dev/null | wc -l) modified files"
  echo "library: $LIB"; echo "checkpoint: $ALTD_GPU_CKPT"; echo "sizes: $SIZES  rayon threads: $RAYON_NUM_THREADS"
  echo
} > "$LOG"

# ---- preflight -------------------------------------------------------------
echo; bold "Preflight"
missing=0
for tool in cargo hipcc rocminfo python3; do
  if command -v "$tool" >/dev/null; then echo "  $tool: $(command -v "$tool")"; else echo "  $tool: MISSING"; missing=1; fi
done
if command -v rocprofv3 >/dev/null; then HAVE_ROCPROF=1; echo "  rocprofv3: $(command -v rocprofv3)"; else HAVE_ROCPROF=0; echo "  rocprofv3: missing (levels 2 and 3 will be skipped)"; fi
[ $missing = 1 ] && { echo "install the missing tools first"; exit 1; }
if [ ! -f "$ALTD_GPU_CKPT/checkpoint.json" ]; then
  echo "  checkpoint: $ALTD_GPU_CKPT/checkpoint.json not found (set ALTD_GPU_CKPT to the B06 checkpoint directory)"; exit 1
fi
echo "  checkpoint: $ALTD_GPU_CKPT ($(python3 -c "import json;m=json.load(open('$ALTD_GPU_CKPT/checkpoint.json'));print('generation',m.get('generation'),'population',m.get('population'))"))"
{ rocminfo | grep -E "^\s*(Name|Marketing Name|Compute Unit|Max Clock Freq|Wavefront Size):" ; } > "$OUT/device.txt" 2>&1
{ hipcc --version; echo; rocminfo | grep -m1 -i "ROCk\|version"; [ $HAVE_ROCPROF = 1 ] && rocprofv3 --version; } > "$OUT/toolchain.txt" 2>&1
grep -m1 "Marketing Name" "$OUT/device.txt" | sed 's/^\s*/  /'
if [ ! -f "$LIB" ] || [ "$(find gpu/sim -newer "$LIB" | wc -l)" -gt 0 ]; then
  if ask "GPU library missing or older than gpu/sim sources. Build it now with gpu/build.sh?" y; then run_logged build_library gpu/build.sh || exit 1; fi
fi
sha256sum "$LIB" >> "$LOG"
if [ ! -x "$CHECK" ] || [ "$(find src examples -newer "$CHECK" -name '*.rs' | wc -l)" -gt 0 ]; then
  run_logged build_examples cargo build --release --examples || exit 1
fi
if ask "Run the exactness checks first (gpu_check math novelty turnover step, a few minutes)?" y; then
  run_logged exactness_check "$CHECK" math novelty turnover step
  grep -E "ALL OK|FAILURES|mismatch" "$OUT/exactness_check.txt" | tail -3 | sed 's/^/    /'
fi
echo "  Close games and other GPU load now; the step kernels are latency sensitive."
ask "Continue?" y || exit 0

# ---- level 1: built-in phases ---------------------------------------------
if want_level 1 "built-in per-phase GPU time and split/fused wall clock"; then
  note "level 1: ALTD_GPU_PROFILE=1 prints GPU seconds per phase and per 600 ticks; graphs are disabled in this mode"
  for cars in $SIZES; do for elim in 0 1; do
    run_logged "phases_${cars}_elim${elim}" $(bench_env "$cars" "$elim" ALTD_GPU_PROFILE=1) "$CHECK" benchfixed
    grep -h "altd_gpu window:" "$OUT/phases_${cars}_elim${elim}.txt" | head -2 | sed 's/^/      /'
  done; done
  for cars in $SIZES; do [ "$cars" -le 32768 ] || continue
    run_logged "phases_${cars}_elim1_fused" $(bench_env "$cars" 1 ALTD_GPU_PROFILE=1 ALTD_GPU_SPLIT=0) "$CHECK" benchfixed
  done
  note "level 1b: production-mode wall clock, split vs fused step (graph on, no profiling)"
  : > "$OUT/split_vs_fused.jsonl"
  for split in 1 0; do for cars in $SIZES; do [ "$cars" -le 32768 ] || continue; for elim in 0 1; do
    run_logged "wall_split${split}_${cars}_elim${elim}" env ALTD_GPU_SPLIT=$split ALTD_GPU_BENCH_CARS=$cars ALTD_GPU_BENCH_ELIMINATE=$elim ALTD_GPU_BENCH_WARMUPS=1 ALTD_GPU_BENCH_SAMPLES=2 "$CHECK" benchfixed
    grep '^{' "$OUT/wall_split${split}_${cars}_elim${elim}.txt" | sed "s/^{/{\"split\":$split,/" >> "$OUT/split_vs_fused.jsonl"
  done; done; done
fi

# ---- level 2: rocprofv3 kernel trace ---------------------------------------
if [ $HAVE_ROCPROF = 1 ] && want_level 2 "per-kernel durations with rocprofv3 (kernel trace + stats)"; then
  note "level 2: rocprofv3 --kernel-trace --stats per run; graph mode unless kernels do not appear individually"
  for cars in $SIZES; do for elim in 0 1; do
    dir="$OUT/trace_${cars}_elim${elim}"
    run_logged "trace_${cars}_elim${elim}" $(bench_env "$cars" "$elim") rocprofv3 --kernel-trace --stats --output-format csv -d "$dir" -o run -- "$CHECK" benchfixed
    stats=$(find "$dir" -name '*kernel_stats.csv' 2>/dev/null | head -1)
    if [ -z "$stats" ] || ! grep -q "w_step\|step_kernel" "$stats"; then
      note "  graph launches were not traced as kernels; repeating with ALTD_GPU_GRAPH=0"
      run_logged "trace_${cars}_elim${elim}_direct" $(bench_env "$cars" "$elim" ALTD_GPU_GRAPH=0) rocprofv3 --kernel-trace --stats --output-format csv -d "${dir}_direct" -o run -- "$CHECK" benchfixed
      stats=$(find "${dir}_direct" -name '*kernel_stats.csv' 2>/dev/null | head -1)
    fi
    [ -n "$stats" ] && { echo "      top kernels by total time:"; sort -t, -k3 -g -r "$stats" 2>/dev/null | head -6 | cut -d, -f1-4 | sed 's/^/        /'; }
  done; done
fi

# ---- level 3: hardware counters -------------------------------------------
if [ $HAVE_ROCPROF = 1 ] && want_level 3 "hardware counters of the step, sensor and forward kernels"; then
  note "level 3: rocprofv3 --pmc passes over 300 ticks with direct launches"
  rocprofv3 --list-avail > "$OUT/counters_available.txt" 2>&1
  # Candidate passes; names absent from this GPU's list are dropped, empty passes skipped.
  passes=(
    "SQ_WAVES SQ_BUSY_CYCLES SQ_WAVE_CYCLES GRBM_GUI_ACTIVE"
    "SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS SQ_INSTS_SMEM"
    "SQ_INSTS_VMEM_RD SQ_INSTS_VMEM_WR SQ_INST_LEVEL_VMEM SQ_WAIT_INST_ANY"
    "SQ_ACTIVE_INST_VALU SQ_ACTIVE_INST_VMEM SQ_ACTIVE_INST_LDS SQ_INST_LEVEL_LDS"
    "GL2C_HIT GL2C_MISS TCP_TOTAL_CACHE_ACCESSES TCP_TCC_READ_REQ"
    "SQ_LEVEL_WAVES SQ_INSTS_BRANCH SQ_INSTS_FLAT SQ_INST_CYCLES_VMEM_RD"
    "SQ_INSTS_VALU_TRANS_F64 SQ_INSTS_VALU_FMA_F64 SQ_INSTS_VALU_ADD_F64 SQ_INSTS_VALU_MUL_F64"
  )
  : > "$OUT/pmc.txt"; dropped=""
  for pass in "${passes[@]}"; do
    keep=""
    for c in $pass; do if grep -qw "$c" "$OUT/counters_available.txt"; then keep="$keep $c"; else dropped="$dropped $c"; fi; done
    [ -n "$keep" ] && echo "pmc:$keep" >> "$OUT/pmc.txt"
  done
  note "  counter passes:"; sed 's/^/    /' "$OUT/pmc.txt" | tee -a "$LOG"
  [ -n "$dropped" ] && note "  not available on this GPU, dropped:$dropped"
  for cars in $SIZES; do [ "$cars" = 32768 ] && continue
    run_logged "counters_${cars}" $(bench_env "$cars" 0 ALTD_GPU_GRAPH=0 ALTD_GPU_BENCH_TICKS=300) rocprofv3 -i "$OUT/pmc.txt" --kernel-include-regex "w_step|sat_kernel|w_sensor|w_forward|w_stats|step_kernel" --output-format csv -d "$OUT/counters_${cars}" -o run -- "$CHECK" benchfixed
  done
fi

# ---- level 4: step sections and kernel resources ---------------------------
if want_level 4 "step-section cycle counts (timing build) and kernel resource usage"; then
  note "level 4: a second library built with -DALTD_STEP_TIMING (atomics slow the step; only the ratios matter)"
  FLAGS=(-O3 -ffp-contract=off --offload-arch=gfx1100 -fPIC -shared -std=c++20 -fhip-fp32-correctly-rounded-divide-sqrt)
  TLIB="$REPO/target/gpu/libaltd_gpu_timing.so"
  python3 gpu/gen_tables.py 2>/dev/null
  if run_logged build_timing_library hipcc "${FLAGS[@]}" -DALTD_STEP_TIMING -o "$TLIB" gpu/sim/altd_gpu.hip -lamdhip64; then
    : > "$OUT/step_sections.txt"
    for cars in $SIZES; do [ "$cars" = 32768 ] && continue; for elim in 0 1; do
      run_logged "sections_${cars}_elim${elim}" $(bench_env "$cars" "$elim" ALTD_GPU_LIB="$TLIB" ALTD_GPU_PROFILE=1) "$CHECK" benchfixed
      { echo "== $cars cars, eliminate=$elim"; grep -E "step sections|altd_gpu window:" "$OUT/sections_${cars}_elim${elim}.txt"; } >> "$OUT/step_sections.txt"
    done; done
  fi
  note "  kernel resource usage (VGPRs, SGPRs, scratch, LDS, occupancy) of the production flags"
  run_logged kernel_resources hipcc "${FLAGS[@]}" -Rpass-analysis=kernel-resource-usage -o "$OUT/.tmp_resources.so" gpu/sim/altd_gpu.hip -lamdhip64
  rm -f "$OUT/.tmp_resources.so"
fi

# ---- level 5: forward-pass attribution -------------------------------------
if want_level 5 "forward-pass attribution (diagnostic builds; their results are wrong by design)"; then
  note "level 5: forward seconds of a 32K full run per variant. diag1 = no activation, diag2 = no weight traffic, diag3 = activation only, diag4 = layer loop only, diag5 = diag4 without layer barriers, branchy = scalar tanh, cumode = -mcumode build"
  FLAGS=(-O3 -ffp-contract=off --offload-arch=gfx1100 -fPIC -shared -std=c++20 -fhip-fp32-correctly-rounded-divide-sqrt)
  python3 gpu/gen_tables.py 2>/dev/null
  : > "$OUT/forward_attribution.txt"
  for variant in "current:" "cumode:-mcumode" "branchy:-DALTD_TANH_BRANCHY" "diag1:-DALTD_FORWARD_DIAG=1" "diag2:-DALTD_FORWARD_DIAG=2" "diag3:-DALTD_FORWARD_DIAG=3" "diag4:-DALTD_FORWARD_DIAG=4" "diag5:-DALTD_FORWARD_DIAG=5"; do
    name=${variant%%:*}; define=${variant#*:}
    DLIB="$REPO/target/gpu/libaltd_gpu_$name.so"
    if [ "$name" = current ]; then DLIB="$LIB"; else
      run_logged "build_$name" hipcc "${FLAGS[@]}" $define -o "$DLIB" gpu/sim/altd_gpu.hip -lamdhip64 || continue
    fi
    run_logged "forward_$name" $(bench_env 32768 0 ALTD_GPU_LIB="$DLIB" ALTD_GPU_PROFILE=1) "$CHECK" benchfixed
    line=$(grep -h "altd_gpu window:" "$OUT/forward_$name.txt" | head -1)
    echo "$name: $line" >> "$OUT/forward_attribution.txt"; echo "      $name: ${line#*cars: }"
  done
fi

# ---- pack -------------------------------------------------------------------
echo; bold "Done"
TAR="$REPO/$(basename "$OUT").tar.gz"
tar czf "$TAR" -C "$REPO" "$(basename "$OUT")"
echo "  results: $OUT"
echo "  archive: $TAR ($(du -h "$TAR" | cut -f1)), send this file"
grep -c "rc=0" "$LOG" | sed 's/^/  steps succeeded: /'
grep "rc=[1-9]" "$LOG" | sed 's/^/  failed: /'
exit 0
