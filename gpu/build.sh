#!/usr/bin/env bash
# Build libaltd_gpu.so (the GPU simulator parts) into $OUT.
# -ffp-contract=off: hipcc otherwise fuses a*b+c into FMA, which rounds once.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$HERE/../target/gpu}"
mkdir -p "$OUT"
python3 "$HERE/gen_tables.py"
nice -n 19 hipcc -O3 -ffp-contract=off --offload-arch=gfx1100 -fPIC -shared -std=c++20 -fhip-fp32-correctly-rounded-divide-sqrt \
    -o "$OUT/libaltd_gpu.so" "$HERE/sim/altd_gpu.hip" -lamdhip64
echo "built $OUT/libaltd_gpu.so" >&2
