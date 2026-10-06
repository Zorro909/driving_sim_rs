#!/usr/bin/env bash
# Build libaltd_gpu.so (the GPU simulator parts) for NVIDIA GPUs into $OUT.
# CUDA_ARCH lists one or more targets separated by commas or spaces, default sm_89 (RTX 40xx).
# -fmad=false: nvcc otherwise fuses a*b+c into FMA, which rounds once. Keep the default
# -prec-div=true -prec-sqrt=true -ftz=false and never use -use_fast_math.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$HERE/../target/gpu}"
mkdir -p "$OUT"
uv run "$HERE/gen_tables.py" --check
CUDA_ARCH="${CUDA_ARCH:-sm_89}"
ARCH_FLAGS=()
for arch in ${CUDA_ARCH//,/ }; do
    ARCH_FLAGS+=("-gencode=arch=compute_${arch#sm_},code=$arch")
done
if [ "${#ARCH_FLAGS[@]}" -eq 0 ]; then
    echo "CUDA_ARCH lists no targets" >&2
    exit 1
fi
nice -n 19 nvcc -x cu -O3 -std=c++20 -fmad=false "${ARCH_FLAGS[@]}" -shared -Xcompiler -fPIC \
    -o "$OUT/libaltd_gpu.so" "$HERE/sim/altd_gpu.hip"
echo "built $OUT/libaltd_gpu.so for ${CUDA_ARCH}" >&2
