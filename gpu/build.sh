#!/usr/bin/env bash
# Build libaltd_gpu.so (the GPU simulator parts) into $OUT.
# GPU_ARCH lists one or more offload targets separated by commas or spaces.
# -ffp-contract=off: hipcc otherwise fuses a*b+c into FMA, which rounds once.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$HERE/../target/gpu}"
mkdir -p "$OUT"
python3 "$HERE/gen_tables.py" --check
GPU_ARCH="${GPU_ARCH:-gfx1100}"
ARCH_FLAGS=()
for arch in ${GPU_ARCH//,/ }; do
    ARCH_FLAGS+=("--offload-arch=$arch")
done
if [ "${#ARCH_FLAGS[@]}" -eq 0 ]; then
    echo "GPU_ARCH lists no targets" >&2
    exit 1
fi
nice -n 19 hipcc -O3 -ffp-contract=off "${ARCH_FLAGS[@]}" -fPIC -shared -std=c++20 -fhip-fp32-correctly-rounded-divide-sqrt \
    -o "$OUT/libaltd_gpu.so" "$HERE/sim/altd_gpu.hip" -lamdhip64
echo "built $OUT/libaltd_gpu.so for ${ARCH_FLAGS[*]#--offload-arch=}" >&2
