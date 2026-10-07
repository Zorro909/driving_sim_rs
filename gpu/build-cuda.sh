#!/usr/bin/env bash
# Build libaltd_gpu.so (the GPU simulator parts) for NVIDIA GPUs into $OUT.
# CUDA_ARCH is native (the default: the GPUs of this machine; without one nvcc warns and uses
# sm_75) or one or more sm_ targets separated by commas or spaces. A list also embeds PTX for its
# newest target, which the driver compiles at load time for newer GPUs.
# -fmad=false: nvcc otherwise fuses a*b+c into FMA, which rounds once. Keep the default
# -prec-div=true -prec-sqrt=true -ftz=false and never use -use_fast_math.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$HERE/../target/gpu}"
mkdir -p "$OUT"
python3 "$HERE/gen_tables.py" --check
CUDA_ARCH="${CUDA_ARCH:-native}"
ARCH_FLAGS=()
if [ "$CUDA_ARCH" = native ]; then
    ARCH_FLAGS=("-arch=native")
else
    newest=0
    for arch in ${CUDA_ARCH//,/ }; do
        if ! [[ "$arch" =~ ^sm_[0-9]+$ ]]; then
            echo "CUDA_ARCH target $arch is not sm_NN (or the whole value native)" >&2
            exit 1
        fi
        ARCH_FLAGS+=("-gencode=arch=compute_${arch#sm_},code=$arch")
        if [ "${arch#sm_}" -gt "$newest" ]; then newest="${arch#sm_}"; fi
    done
    if [ "${#ARCH_FLAGS[@]}" -eq 0 ]; then
        echo "CUDA_ARCH lists no targets" >&2
        exit 1
    fi
    ARCH_FLAGS+=("-gencode=arch=compute_$newest,code=compute_$newest")
fi
nice -n 19 nvcc -x cu -O3 -std=c++20 -fmad=false "${ARCH_FLAGS[@]}" -shared -Xcompiler -fPIC \
    -o "$OUT/libaltd_gpu.so" "$HERE/sim/altd_gpu.hip"
echo "built $OUT/libaltd_gpu.so for ${CUDA_ARCH}" >&2
